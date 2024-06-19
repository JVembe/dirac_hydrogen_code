/*
  This program reads base matrices saved in CSR format.
  To produce those matrices, run the following:

  # this saves the base g matrices
  dumpCouplings -g ../input_example.json

  # this saves the Hamiltonian matrices
  dumpCouplings ../input_example.json

  # this combines the invididual Hamiltonian matrices
  # into the full couplings matrix structure, and computes
  # domain decomposition / node renumbering
  python3 main.py

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <unistd.h>

#include "csr.h"
#include "superlu.h"
#include "bicgstab.h"
#include "solver.h"
#include "utils.h"
#include "../src/tictoc.h"
//#include "../src/potential.h"

#ifdef USE_MPI
#include <mpi.h>
#endif

#if defined USE_CUDA | defined USE_HIP
#include "gpu_sparse.h"
#include "gpu_bicgstab.h"
#include "gpu_solver.h"
#endif

#define max(a,b) ((a)>(b)?(a):(b))

int rank = 0, nranks = 1;

void vec_read(const char *fname, int start, int end, cdouble_t **out)
{
    size_t nread;
    FILE *fd = fopen(fname, "r");
    if(!fd) ERROR("cant open %s\n", fname);

    int n;
    nread = fread(&n, sizeof(int), 1, fd);
    if(nread!=1) ERROR("wrong file format in %s\n", fname);

    nread = fseek(fd, sizeof(cdouble_t)*start, SEEK_CUR);
    if(nread!=0) ERROR("wrong file format in %s\n", fname);
    nread = ftell(fd);
    printf("current pos %li\n", nread);

    // read only local vector part
    n = end-start;
    *out = (cdouble_t*)malloc(sizeof(cdouble_t)*n);
    nread = fread(*out, sizeof(cdouble_t), n, fd);
    if(nread!=n) ERROR("wrong file format in %s\n", fname);
    printf("%s local size %d\n", fname, n);
    fclose(fd);
}


typedef struct {
    sparse_csr_t *H, *S;
} HS_matrices;

void HS_spmv_fun(const void *mat, cdouble_t *x, cdouble_t *out)
{
    HS_matrices *hsptr = (HS_matrices*)mat;

    // Setup the communication for H: this is cheap - just sets up recv pointers for x
    csr_init_communication(hsptr->H, x, rank, nranks);
    csr_comm(hsptr->H, rank, nranks);
    csr_spmv(0, csr_nrows(hsptr->H), hsptr->H, x, out);

    // S has only local vector entries - x has to be shifted compared to H
    csr_spmv(0, csr_nrows(hsptr->S), hsptr->S, x + csr_local_rowoffset(hsptr->H), out);
}


void LU_precond_fun(const void *precond, const cdouble_t *rhs, cdouble_t *x)
{
    slu_lu_solve((slu_LU_t*)precond, (doublecomplex*)rhs, (doublecomplex*)x);
}

#if defined USE_CUDA | defined USE_HIP

typedef struct {
    sparse_csr_t *H, *S;
    gpu_sparse_csr_t *gpuH, *gpuS;
} gpu_HS_matrices;

void gpu_HS_spmv_fun(const void *mat, gpu_dense_vec_t *x, gpu_dense_vec_t *out, csr_data_t alpha, csr_data_t beta)
{
    gpu_HS_matrices *hsptr = (gpu_HS_matrices*)mat;

    // Setup the communication for H: this is cheap - just sets up recv pointers for x
    csr_init_communication(hsptr->H, (csr_data_t*)x->x, rank, nranks);
    csr_comm(hsptr->H, rank, nranks);
    gpu_spmv(hsptr->gpuH, x, out, alpha, beta);

    // S has only local vector entries - x has to be shifted compared to H
    gpu_spmv_local(hsptr->gpuS, x, out, alpha, CMPLX(1,0));
}

void gpu_LU_precond_fun(const void *_precond, const gpu_dense_vec_t *rhs, gpu_dense_vec_t *x)
{
    gpu_lu_t *precond = (gpu_lu_t*)_precond;
    gpu_lu_solve(precond->L, precond->U, rhs, x, precond->temp);
}

#endif

int main(int argc, char *argv[])
{
    char fname[256];
    sparse_csr_t Hall = {}, Hfull = {}, Hfull_blk = {}, Sdiag = {}, S = {}, S_blk = {};
    sparse_csr_t Hst = {}, Hst_blk = {};
    sparse_csr_t *g, *gt, *h0, *s0;
    sparse_csr_t *H;
    cdouble_t *psi0;

    int opt;
    int lmax;
    double intensity, omega, cycles, time;
    double dt, h;
    int cnt;

    slu_matrix_t sluA = {};
    slu_LU_t sluLU = {};

    if(argc<5){
        fprintf(stderr, "Usage: %s -l time -l lmax -i intensity -o omega -c cycles\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    while ((opt = getopt(argc, argv, "t:l:i:o:c:")) != -1) {
        switch (opt) {
        case 't': time = atof(optarg); break;
        case 'l': lmax = atoi(optarg); break;
        case 'i': intensity = atof(optarg); break;
        case 'o': omega = atof(optarg); break;
        case 'c': cycles = atof(optarg); break;
        default:
            fprintf(stderr, "Usage: %s [-lioc]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    dt = 0.125;
    h  = 1;

    printf("Parameters:\n");
    printf(" - time      %f\n", time);
    printf(" - lmax      %d\n", lmax);
    printf(" - intensity %lf\n", intensity);
    printf(" - omega     %lf\n", omega);
    printf(" - cycles    %lf\n", cycles);
    printf(" - dt        %lf\n", dt);
    printf(" - h         %lf\n", h);

    // time-dependent part
    beyondDipolePulse_t bdpp;
    beyondDipolePulse_init(&bdpp, intensity, omega, cycles);

    // read the full couplings matrix structure
    // each non-zero value denotes a submatrix of dimensions same as the G matrices
    snprintf(fname, 255, "H.csr");
    csr_read(fname, &Hall);

    // create a diagonal matrix for the S-part (and the stationary H part)
    csr_diag(&Sdiag, csr_nrows(&Hall));

    // copy partitioning from Hall to Sdiag
    Sdiag.row_cpu_dist = Hall.row_cpu_dist;
    Sdiag.row_beg = Hall.row_beg;
    Sdiag.row_end = Hall.row_end;
    if(Hall.perm){
        Sdiag.perm = (csr_index_t*)malloc(sizeof(csr_index_t)*Sdiag.nrows);
        memcpy(Sdiag.perm, Hall.perm + Hall.local_offset, sizeof(csr_index_t)*Hall.nrows);
    }

#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    if(nranks>1 && Hall.npart != nranks)
        ERROR("There are %d MPI ranks, but the matrix is partition for %d ranks.\n", nranks, Hall.npart);
#endif

    // get local partition of the global H matrix
    if(nranks>1){
        csr_get_partition(&Hfull_blk, &Hall, rank, Hall.npart);

        // partitioning of the S-part is trivial, but still needs to be done
        csr_get_partition(&S_blk, &Sdiag, rank, Hall.npart);

        /* DEBUG */
        /* snprintf(fname, 255, "H_part%d.csr", rank); */
        /* csr_write(fname, &Hfull_blk); */
    } else {
        Hfull_blk = Hall;
        S_blk = Sdiag;
    }
    /* MPI_Barrier(MPI_COMM_WORLD); */
    /* MPI_Finalize(); */
    /* exit(0); */
    // read the individual Hamiltonian matrices H0 and H1
    // 2 matrices for each value of 0:lmax-1
    // Note: these are global, with original node indices - no partitioning
    H = malloc(sizeof(sparse_csr_t)*lmax*2);
    cnt = 0;
    for(int l=0; l<lmax; l++){
        snprintf(fname, 255, "H0l%d.csr", l);
        csr_read(fname, H+cnt);
        cnt++;
        snprintf(fname, 255, "H1l%d.csr", l);
        csr_read(fname, H+cnt);
        cnt++;
    }

    // load the base g matrices
    // 4 matrices: g0, g1, g2, g3
    // alpha 0:6-1
    // l     0:lmax-1
    g = malloc(sizeof(sparse_csr_t)*6*lmax*4);
    cnt = 0;
    for(int a=0; a<6; a++){
        for(int l=0; l<lmax; l++){
            for(int gi=0; gi<4; gi++){
                snprintf(fname, 255, "g%da%dl%d.csr", gi, a, l);
                csr_read(fname, g+cnt);
                cnt++;
            }
        }
    }
    
    // compute conjugate transpose of all g matrices
    gt = malloc(sizeof(sparse_csr_t)*6*lmax*4);
    cnt = 0;
    for(int a=0; a<6; a++){
        for(int l=0; l<lmax; l++){
            for(int gi=0; gi<4; gi++){
                csr_copy(gt+cnt, g+cnt);
                csr_conj_transpose(gt+cnt, g+cnt);
                cnt++;
            }
        }
    }
   
    h0 = malloc(sizeof(sparse_csr_t)*4);
    cnt = 0;
    for(int hi = 0; hi < 4; hi++) {
        snprintf(fname, 255, "h0%d.csr",hi);
        csr_read(fname,h0+cnt);
        cnt++;
    }

    s0 = malloc(sizeof(sparse_csr_t)*4);
    cnt = 0;
    for(int s = 0; s < 3; s++) {
        snprintf(fname, 255, "s%d.csr",s);
        csr_read(fname,s0+cnt);
        cnt++;
    }

    // create the full rank-local matrix blocked structure
    // same Ap and Ai as Hfull_blk
    // Ax modified to store sub-matrices (g.nnz)
    int blkdim = csr_nrows(&g[0]);
    csr_copy(&Hst_blk, &S_blk);

    // setup block parameters for the blocked matrices
    csr_block_params(&Hfull_blk, blkdim, csr_nnz(&g[0]));
    csr_block_params(&S_blk, blkdim, csr_nnz(&s0[0]));
    csr_block_params(&Hst_blk, blkdim, csr_nnz(&s0[0]));

    // allocate storage for Hfull_blk.Ax - testing
    // Hfull_blk.Ax = malloc(sizeof(csr_data_t)*Hfull_blk.nnz*Hfull_blk.blk_nnz);

    // each submatrix will have the same non-zero structure as the base g matrices
    sparse_csr_t submatrix;
    csr_copy(&submatrix, g);

    // create the non-blocked rank-local Hfull matrix structure
    //  - each non-zero is converted to blkdim x blkdim submatrix
    //  - each row has row_nnz(Hfull_blk)*row_nnz(G) non-zero entries
    //  - create Hfull.Ap and Hfull.Ai accordingly
    // Dimensions of Hfull and Hfull_blk are the same, and they have
    // the same number of non-zeros. However, their Ap and Ai differ:
    // Hfull_blk inherits Ap and Ai directly from Hfull_blk - one non-zero per entire submatrix.
    // Hfull stores all non-zeros independently in a native, non-blocked csr storage
    csr_unblock_matrix(&Hfull, &Hfull_blk, &submatrix);
    csr_unblock_matrix(&S, &S_blk, &submatrix);
    csr_unblock_matrix(&Hst, &Hst_blk, &submatrix);
    
    // Hfull_blk has the comm info copied from Hfull_blk,
    // but Hfull doesnt - it has to be modified by the block size.
    // At this point there is no comm info in Hfull - so copy it.
    csr_unblock_comm_info(&Hfull, &Hfull_blk, rank, nranks);

#if defined USE_CUDA
    gpu_init_model_matrices(6*lmax*4, g, gt, h0);
    gpu_compute_row_col(lmax, H, &Hfull_blk, &Hfull);

    csr_data_t ft[6];
    beoyndDipolePulse_axialPart(&bdpp, dt, ft);
    PRINTF0("f(t)\n");
    for(int i=0; i<6; i++) PRINTF0("(%lf,%lf)\n", creal(ft[i]), cimag(ft[i]));

    gpu_sparse_csr_t gpu_Hfull = {};
    gpu_compute_timedep_matrices(h, dt, ft, lmax, &Hfull_blk, &Hfull, &gpu_Hfull);
    printf("CUDA compute timedep matrix "); tic();
    gpu_compute_timedep_matrices(h, dt, ft, lmax, &Hfull_blk, &Hfull, &gpu_Hfull);
    toc();

    /* csr_ijk_write("gpu.ijk", &Hfull); */
    /* MPI_Finalize(); */
    /* exit(0); */
#endif
    
    // block-diagonal matrices - no communication
    // csr_unblock_comm_info(&S, &S_blk, rank, nranks);
    // csr_unblock_comm_info(&Hst, &Hst_blk, rank, nranks);

    // read initial state
    snprintf(fname, 255, "psi0.vec");
    vec_read(fname, Hfull.row_beg, Hfull.row_end, &psi0);

    for(int r=0; r<nranks; r++){
        if(rank == r){
            printf("%d: All matrices read correctly. System info:\n", rank);
            printf(" - H dim:         %d x %d\n", csr_nrows(&H[0]), csr_ncols(&H[0]));
            printf(" - Hall dim:      %d x %d\n", csr_nrows(&Hall), csr_ncols(&Hall));
            printf(" - Hall nnz:      %d\n", csr_nnz(&Hall));
            printf(" - Hfull_blk dim: %d x %d\n", csr_nrowblocks(&Hfull_blk), csr_ncolblocks(&Hfull_blk));
            printf(" - Hfull dim:     %d x %d\n", csr_nrows(&Hfull_blk), csr_ncols(&Hfull_blk));
            printf(" - Hfull nnz:     %d\n", csr_nnz(&Hfull_blk));
            fflush(stdout);
        }
#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }

    // allocate x and y vectors for SpMV: y = spA*x
    csr_data_t *x, *yblk, *yfull, *ydev, *rhs;

    // x and rhs have to have the non-loca entries for communication - hence ncols
    x     = (csr_data_t *)calloc(csr_ncols(&Hfull_blk), sizeof(csr_data_t));
    rhs   = (csr_data_t *)calloc(csr_ncols(&Hfull_blk), sizeof(csr_data_t));
    yblk  = (csr_data_t *)calloc(csr_nrows(&Hfull_blk), sizeof(csr_data_t));
    yfull = (csr_data_t *)calloc(csr_nrows(&Hfull_blk), sizeof(csr_data_t));
    ydev  = (csr_data_t *)calloc(csr_nrows(&Hfull_blk), sizeof(csr_data_t));

    { // Compute
        csr_index_t row, col, colp;

        // TODO get the time to compute time-dependent f(a,t)
        csr_data_t ft[6];
        complex ihdt = I*h*dt/2;
        time = time + dt;
        beoyndDipolePulse_axialPart(&bdpp, dt, ft);
        PRINTF0("f(t)\n");
        for(int i=0; i<6; i++) PRINTF0("(%lf,%lf)\n", creal(ft[i]), cimag(ft[i]));
        
        // time-dependent part of the Hamiltonian
        compute_timedep_matrices(h, dt, &submatrix, ft, lmax, &Hfull_blk, &Hfull, h0, H, g, gt);
        
        // The Hfull_blk matrix contains all computed submatrices.
        // The submatrices are stored as a sub-block in the csr storage
        // meaning that the relevant Ax parts can be used directly
        // as Ax arrays in a template submatrix csr structure, e.g.
        // csr_block_link(&submatrix, &Hfull_blk, row, col);

        if(Hfull_blk.Ax){

            // initialize input vector. non-local parts are set to nan to verify communication:
            // all non-local entries are received from peers, hence set to non-nan during communication
            for(int i=0; i<csr_ncols(&Hfull); i++) x[i] = CMPLX(NAN,NAN);
            for(int i=0; i<csr_nrows(&Hfull); i++)
                x[csr_local_rowoffset(&Hfull) + i] = CMPLX(1, 0); // CMPLX(Hfull_blk.row_beg*blkdim + i, Hfull_blk.row_beg*blkdim + i);

            tic(); PRINTF0("blocked spmv ");
            csr_init_communication(&Hfull_blk, x, rank, nranks);
            csr_comm(&Hfull_blk, rank, nranks);
            for(row = 0; row < Hfull_blk.nrows; row++){

                // for non-zero blocks in each row
                for(colp = Hfull_blk.Ap[row]; colp < Hfull_blk.Ap[row+1]; colp++){

                    // NOTE: rows and cols are remapped wrt. the original numbering in H
                    col = Hfull_blk.Ai[colp];

                    csr_block_link(&submatrix, &Hfull_blk, row, col);

                    csr_data_t *xin, *yout;
                    xin  = x + col*blkdim;
                    yout = yblk + row*blkdim;

                    // perform spmv
                    csr_spmv(0, csr_nrows(&submatrix), &submatrix, xin, yout);

                    // remember that at this stage xin and yout are renumbered wrt. the original node numbering
                }
            }
            toc();
        }

        // initialize input vector. non-local parts are set to nan to verify communication:
        // all non-local entries are received from peers, hence set to non-nan during communication
        for(int i=0; i<csr_ncols(&Hfull); i++) x[i] = CMPLX(NAN,NAN);
        for(int i=0; i<csr_nrows(&Hfull); i++)
            x[csr_local_rowoffset(&Hfull) + i] = CMPLX(1, 1); // CMPLX(Hfull_blk.row_beg*blkdim + i, Hfull_blk.row_beg*blkdim + i);

        // perform spmv for the non-blocked Hfull matrix (native csr storage)
        tic(); PRINTF0("non-blocked spmv ");
        csr_init_communication(&Hfull, x, rank, nranks);
        csr_comm(&Hfull, rank, nranks);
        csr_spmv(0, csr_nrows(&Hfull), &Hfull, x, yfull);
        toc();

        // perform spmv for the non-blocked Hfull matrix (native csr storage)
        tic(); PRINTF0("matrix-free spmv (warmup)");
        csr_init_communication(&Hfull, x, rank, nranks);
        csr_comm(&Hfull, rank, nranks);
        matfree_spmv_H(0, csr_nrows(&Hfull), &Hfull_blk, x, yblk,
                       h, dt, &submatrix, ft, lmax, h0, H, g, gt);
        toc();

        for(int i=0; i<csr_nrows(&Hfull_blk); i++)
            yblk[i] = CMPLX(0,0);
        tic(); PRINTF0("matrix-free spmv ");
        csr_init_communication(&Hfull, x, rank, nranks);
        csr_comm(&Hfull, rank, nranks);
        matfree_spmv_H(0, csr_nrows(&Hfull), &Hfull_blk, x, yblk,
                       h, dt, &submatrix, ft, lmax, h0, H, g, gt);
        toc();
        
        // validate - compare yblk and yfull results
        compare_vectors(yfull, yblk, csr_nrows(&Hfull));            

#if defined USE_CUDA | defined USE_HIP
        {
            // initialize input vector. non-local parts are set to nan to verify communication:
            // all non-local entries are received from peers, hence set to non-nan during communication
            for(int i=0; i<csr_ncols(&Hfull); i++) x[i] = CMPLX(NAN,NAN);
            for(int i=0; i<csr_nrows(&Hfull); i++)
                x[csr_local_rowoffset(&Hfull) + i] = CMPLX(1, 0); // CMPLX(Hfull_blk.row_beg*blkdim + i, Hfull_blk.row_beg*blkdim + i);

            gpu_sparse_init();
            gpu_blas_init();
            gpu_spmv_test(Hfull, x, ydev);

            // compare with CPU results
            compare_vectors(ydev, yfull, csr_nrows(&Hfull));

            // gpu_spmb_block_test(Hfull_blk, x, yfull, g);
        }
#endif

        // compute the stationary part once
        compute_stationary_matrices(h, dt, &submatrix, &S_blk, &S, &Hst_blk, &Hst, h0, s0);

#if defined USE_CUDA | defined USE_HIP
        // test local only spmv
        {
            // initialize input vector. non-local parts are set to nan to verify communication:
            // all non-local entries are received from peers, hence set to non-nan during communication
            for(int i=0; i<csr_ncols(&Hfull); i++) x[i] = CMPLX(NAN,NAN);
            for(int i=0; i<csr_nrows(&Hfull); i++)
                x[csr_local_rowoffset(&Hfull) + i] = CMPLX(1, 0); // CMPLX(Hfull_blk.row_beg*blkdim + i, Hfull_blk.row_beg*blkdim + i);

            // perform spmv for the non-blocked S matrix (native csr storage)
            for(int i=0; i<csr_nrows(&Hfull); i++) yfull[i] = 0;
            tic(); PRINTF0("S spmv ");
            csr_spmv(0, csr_nrows(&S), &S, x + csr_local_rowoffset(&Hfull), yfull);
            toc();

            gpu_dense_vec_t xgpu = {0}, ygpu = {0};
            gpu_put_vec(&xgpu, x, csr_ncols(&Hfull));
            gpu_vec_local_part(&xgpu, csr_nrows(&Hfull), csr_local_rowoffset(&Hfull));
            gpu_put_vec(&ygpu, NULL, csr_nrows(&S));

            gpu_sparse_csr_t gpuS = {};
            gpu_put_csr(&gpuS, &S);

            tic(); PRINTF0("GPU S spmv ");
            gpu_spmv_local(&gpuS, &xgpu, &ygpu, CMPLX(1,0), CMPLX(0,0));
            toc();

            gpu_get_vec(ydev, &ygpu);

            // compare with CPU results
            compare_vectors(ydev, yfull, csr_nrows(&Hfull));

            // cleanup
            gpu_free_vec(&xgpu);
            gpu_free_vec(&ygpu);
            gpu_free_csr(&gpuS);
        }
#else
        {
            for(int i=0; i<csr_ncols(&Hfull); i++) x[i] = CMPLX(NAN,NAN);
            for(int i=0; i<csr_nrows(&Hfull); i++)
                x[csr_local_rowoffset(&Hfull) + i] = CMPLX(i, 2*i); // CMPLX(Hfull_blk.row_beg*blkdim + i, Hfull_blk.row_beg*blkdim + i);

            // perform spmv for the non-blocked S matrix (native csr storage)
            for(int i=0; i<csr_nrows(&Hfull); i++) yfull[i] = 0;
            tic(); PRINTF0("S spmv ");
            csr_spmv(0, csr_nrows(&S), &S, x + csr_local_rowoffset(&Hfull), yfull);
            toc();            

            csr_data_t *ymatfree = (csr_data_t *)calloc(csr_nrows(&Hfull_blk), sizeof(csr_data_t));
            tic(); PRINTF0("S matfree spmv ");
            matfree_spmv_S(0, csr_nrows(&S), &S_blk, x + csr_local_rowoffset(&Hfull), ymatfree,
                           h, dt, s0, &submatrix);
            toc();            
            
            // compare with CPU results
            compare_vectors(ymatfree, yfull, csr_nrows(&Hfull));

            free(ymatfree);
        }
#endif

        // compute the preconditioner once - based on the stationary part
        slu_LU_t sluLU = compute_preconditioner(&S, &Hst);

        // CPU solve using SuperLU
        slu_lu_solve(&sluLU, (doublecomplex*)x + csr_local_rowoffset(&Hfull), (doublecomplex*)yblk);
        
#if defined USE_CUDA | defined USE_HIP
        // compare SuperLU solve with cusparse / hipsparse solve
        sparse_csr_t hostL = {}, hostU = {};
        {
            csr_index_t *LAi, *LAj;
            csr_index_t *UAi, *UAj;
            csr_data_t *LAx, *UAx;
            csr_index_t Lnnz, Unnz;

            // convert from SuperLU super-node format to csr format
            slu_LU2coo(sluLU.L, sluLU.U,
                       &LAi, &LAj, (doublecomplex**)&LAx, &Lnnz,
                       &UAi, &UAj, (doublecomplex**)&UAx, &Unnz);

            csr_coo2csr(&hostU, UAi, UAj, UAx, csr_nrows(&Hfull), Unnz);
            csr_coo2csr(&hostL, LAi, LAj, LAx, csr_nrows(&Hfull), Lnnz);

            free(LAi); free(LAj); free(LAx);
            free(UAi); free(UAj); free(UAx);

            gpu_sparse_csr_t gpuL = {}, gpuU = {};
            gpu_put_csr(&gpuL, &hostL);
            gpu_put_csr(&gpuU, &hostU);

            gpu_dense_vec_t xgpu = {0}, ygpu = {0}, tempgpu = {0};
            gpu_put_vec(&xgpu, x + csr_local_rowoffset(&Hfull), csr_nrows(&Hfull));
            gpu_put_vec(&ygpu, NULL, csr_nrows(&Hfull));
            gpu_put_vec(&tempgpu, NULL, csr_nrows(&Hfull));

            // GPU solve using cusparse / hipsparse
            tic(); PRINTF0("GPU LU analyze ");
            gpu_lu_analyze(&gpuL, &gpuU, &xgpu, &ygpu, &tempgpu);
            toc();

            tic(); PRINTF0("GPU LU solve ");
            gpu_lu_solve(&gpuL, &gpuU, &xgpu, &ygpu, &tempgpu);
            toc();

            gpu_get_vec(ydev, &ygpu);

            // compare with CPU results
            compare_vectors(ydev, yblk, csr_nrows(&Hfull));

            // cleanup
            gpu_free_vec(&xgpu);
            gpu_free_vec(&ygpu);
            gpu_free_vec(&tempgpu);
            gpu_free_csr(&gpuL);
            gpu_free_csr(&gpuU);
        }
#endif

        // test the Dirac solver

        // initial state
        for(int i=0; i<csr_ncols(&Hfull); i++) x[i] = CMPLX(NAN,NAN);
        for(int i=0; i<csr_nrows(&Hfull); i++) x[csr_local_rowoffset(&Hfull) + i] = psi0[i];

        // rhs = (S-H)*psi(n-1)
        for(int i=0; i<csr_nrows(&Hfull); i++) rhs[i] = 0;
        csr_init_communication(&Hfull, x, rank, nranks);
        csr_comm(&Hfull, rank, nranks);
        csr_spmv(0, csr_nrows(&Hfull), &Hfull, x, rhs);
        for(int i=0; i<csr_nrows(&Hfull); i++) rhs[i] = -1*rhs[i];
        csr_spmv(0, csr_nrows(&S), &S, x + csr_local_rowoffset(&Hfull), rhs);

        solver_workspace_t wsp = {0};
        int iters = 500;
        double tol_error = 1e-16;
        HS_matrices mat;
        mat.H = &Hfull;
        mat.S = &S;

        MPI_Barrier(MPI_COMM_WORLD);
        PRINTF0("CPU BICGSTAB\n"); tic();
        bicgstab(HS_spmv_fun, &mat, rhs, x, csr_nrows(&Hfull), csr_ncols(&Hfull), csr_local_rowoffset(&Hfull),
                 LU_precond_fun, &sluLU, &wsp, &iters, &tol_error);
        MPI_Barrier(MPI_COMM_WORLD);
        toc();

#if defined USE_CUDA | defined USE_HIP
        {
            gpu_HS_matrices gpumat = {};
            gpu_sparse_csr_t gpuS = {}, gpuHfull = {};

            gpu_put_csr(&gpuS, &S);

            MPI_Barrier(MPI_COMM_WORLD);
            PRINTF0("host to device H"); tic();
            gpu_put_csr(&gpuHfull, &Hfull);
            MPI_Barrier(MPI_COMM_WORLD);
            toc();
            
            gpumat.H = &Hfull;
            gpumat.gpuH = &gpuHfull;
            gpumat.S = &S;
            gpumat.gpuS = &gpuS;

            gpu_sparse_csr_t gpuL = {}, gpuU = {};
            gpu_dense_vec_t tempgpu = {};
            gpu_lu_t gpuLU = {};
            gpu_put_csr(&gpuL, &hostL);
            gpu_put_csr(&gpuU, &hostU);
            gpuLU.L = &gpuL;
            gpuLU.U = &gpuU;
            gpu_put_vec(&tempgpu, NULL, csr_nrows(&Hfull));
            gpuLU.temp = &tempgpu;

            // store the CPU result for comparison
            for(int i=0; i<csr_nrows(&Hfull); i++) yblk[i] = x[csr_local_rowoffset(&Hfull) + i];

            // initial state
            for(int i=0; i<csr_ncols(&Hfull); i++) x[i] = CMPLX(NAN,NAN);
            for(int i=0; i<csr_nrows(&Hfull); i++) x[csr_local_rowoffset(&Hfull) + i] = psi0[i];

            gpu_dense_vec_t xgpu = {0}, rhsgpu = {0};
            gpu_put_vec(&xgpu, x, csr_ncols(&Hfull));
            gpu_vec_local_part(&xgpu, csr_nrows(&Hfull), csr_local_rowoffset(&Hfull));
            gpu_put_vec(&rhsgpu, NULL, csr_nrows(&Hfull));

            gpu_lu_analyze(&gpuL, &gpuU, &rhsgpu, &xgpu, &tempgpu);

            // rhs = (S-H)*psi(n-1)
            csr_init_communication(&Hfull, (csr_data_t*)xgpu.x, rank, nranks);
            csr_comm(&Hfull, rank, nranks);
            gpu_spmv(&gpuHfull, &xgpu, &rhsgpu, CMPLX(-1,0), CMPLX(0,0));
            gpu_spmv_local(&gpuS, &xgpu, &rhsgpu, CMPLX(1,0), CMPLX(1,0));

            gpu_get_vec(ydev, &rhsgpu);

            compare_vectors(ydev, rhs, csr_nrows(&Hfull));

            gpu_solver_workspace_t gpuwsp = {0};
            int iters = 500;
            double tol_error = 1e-16;

            MPI_Barrier(MPI_COMM_WORLD);
            PRINTF0("GPU BICGSTAB\n"); tic();
            gpu_bicgstab(gpu_HS_spmv_fun, &gpumat, &rhsgpu, &xgpu, csr_nrows(&Hfull), csr_ncols(&Hfull), csr_local_rowoffset(&Hfull),
                         gpu_LU_precond_fun, &gpuLU, &gpuwsp, &iters, &tol_error);
            MPI_Barrier(MPI_COMM_WORLD);
            toc();

            gpu_get_vec(ydev, &xgpu);

            // compare with CPU results
            compare_vectors(ydev, yblk, csr_nrows(&Hfull));
        }
#endif
    }

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
#endif
}

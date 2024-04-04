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
#include "../src/tictoc.h"
//#include "../src/potential.h"

#ifdef USE_MPI
#include <mpi.h>
#endif

#if defined USE_CUDA | defined USE_HIP
#include "gpu_sparse.h"
#endif

int rank = 0, nranks = 1;
int *ikarr = NULL;


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

    // read only local vector part
    n = end-start;
    *out = (cdouble_t*)malloc(sizeof(cdouble_t)*n);
    nread = fread(*out, sizeof(cdouble_t), n, fd);
    if(nread!=n) ERROR("wrong file format in %s\n", fname);
    fclose(fd);
}

void vec_read_perm(const char *fname, csr_index_t **out)
{
    size_t nread;
    FILE *fd = fopen(fname, "r");
    if(!fd) ERROR("cant open %s\n", fname);

    int n;
    nread = fread(&n, sizeof(int), 1, fd);
    if(nread!=1) ERROR("wrong file format in %s\n", fname);

    nread = fseek(fd, sizeof(cdouble_t)*n, SEEK_CUR);
    if(nread!=0) ERROR("wrong file format in %s\n", fname);

    // read only local vector part
    *out = (csr_index_t*)malloc(sizeof(csr_index_t)*n);
    nread = fread(*out, sizeof(csr_index_t), n, fd);
    if(nread!=n) ERROR("wrong file format in %s\n", fname);
    fclose(fd);
}

// simple C-fied beyondDipolePulse implementation
typedef struct
{
    double E0;
    double omega;
    double T;
} beyondDipolePulse_t;


void beyondDipolePulse_init(beyondDipolePulse_t *this, double E0, double omega, double N)
{
    this->E0 = E0;
    this->omega = omega;
    this->T = (double)N*2*M_PI / omega;
}


void beoyndDipolePulse_axialPart(beyondDipolePulse_t *this, double t, cdouble_t *out)
{
    cdouble_t phi = CMPLX(-M_PI/2, 0);
    double E0 = this->E0;
    double omega = this->omega;
    double T = this->T;
    out[0] = ( -E0/(4*omega) * cos( phi + t * (2 * M_PI / T + omega)));
    out[1] = ( -E0/(4*omega) * sin( phi + t * (2 * M_PI / T + omega)));
    out[2] = ( -E0/(4*omega) * cos(-phi + t * (2 * M_PI / T - omega)));
    out[3] = (  E0/(4*omega) * sin(-phi + t * (2 * M_PI / T - omega)));
    out[4] = (  E0/(2*omega) * cos( phi + t * omega));
    out[5] = (  E0/(2*omega) * sin( phi + t * omega));
}


void compare_vectors(csr_data_t *v1, csr_data_t *v2, csr_index_t dim)
{
    // validate - compare yblk and yfull results
    for(int i=0; i<dim; i++) {
        if(isnan(cimag(v1[i]+v2[i])) || isnan(creal(v1[i]+v2[i]))) {
            printf("nan in vector!\n");
            continue;
        }
        if(fabs(cimag(v1[i]-v2[i]))>1e-8) fprintf(stderr, "v1 %e v2 %e diff %e i\n",
                                                  cimag(v1[i]), cimag(v2[i]), cimag(v1[i]) - cimag(v2[i]));
        if(fabs(creal(v1[i]-v2[i]))>1e-8) fprintf(stderr, "v1 %e v3 %e diff %e\n",
                                                  creal(v1[i]), creal(v2[i]), creal(v1[i]) - creal(v2[i]));
    }
}


typedef struct {
    sparse_csr_t *H, *S;
} HS_matrices;

int cnt = 0;

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


int main(int argc, char *argv[])
{
    char fname[256];
    sparse_csr_t Hall, Hfull, Hfull_blk, Sdiag, S, S_blk;
    sparse_csr_t Hst, Hst_blk;
    sparse_csr_t *g, *gt, *h0, *s0;
    sparse_csr_t *H;
    cdouble_t *psi0;

    int opt;
    int lmax;
    double intensity, omega, cycles, maxtime = 0, time = 0;
    double dt, h;
    int cnt;

    if(argc<5){
        fprintf(stderr, "Usage: %s -l time -l lmax -i intensity -o omega -c cycles\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    while ((opt = getopt(argc, argv, "t:l:i:o:c:")) != -1) {
        switch (opt) {
        case 't': maxtime = atof(optarg); break;
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

    if(0 == rank){
        printf("Parameters:\n");
        printf(" - time      %f\n", maxtime);
        printf(" - lmax      %d\n", lmax);
        printf(" - intensity %lf\n", intensity);
        printf(" - omega     %lf\n", omega);
        printf(" - cycles    %lf\n", cycles);
        printf(" - dt        %lf\n", dt);
        printf(" - h         %lf\n", h);
    }

    // get local partition of the global H matrix
    if(nranks>1){
        csr_get_partition(&Hfull_blk, &Hall, rank, Hall.npart);

        // partitioning of the S-part is trivial, but still needs to be done
        csr_get_partition(&S_blk, &Sdiag, rank, Hall.npart);
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

    // setup block parameters for the blocked matrices.
    // NOTE: Ax (data) memory is not allocated.
    csr_block_params(&Hfull_blk, blkdim, csr_nnz(&g[0]));
    csr_block_params(&S_blk, blkdim, csr_nnz(&s0[0]));
    csr_block_params(&Hst_blk, blkdim, csr_nnz(&s0[0]));

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

    // xall is the solution vector collected on rank 0 for un-permuting and printing
    csr_data_t *xall = NULL, *xorig = NULL;
    csr_index_t *xperm = NULL;
    if(0 == rank) {
        vec_read_perm("psi0.vec", &xperm);
        xall  = (csr_data_t *)calloc(csr_nrows(&Hall)*blkdim, sizeof(csr_data_t));
        xorig = (csr_data_t *)calloc(csr_nrows(&Hall)*blkdim, sizeof(csr_data_t));
    }

    // allocate x and y vectors for SpMV: y = spA*x
    // xall is the solution vector collected on rank 0 for un-permuting and printing
    csr_data_t *x, *rhs;

    // rhs = (S-H)*psi0
    // x(=psi0) has to have the non-loca entries for communication - hence ncols
    x     = (csr_data_t *)calloc(csr_ncols(&Hfull_blk), sizeof(csr_data_t));
    rhs   = (csr_data_t *)calloc(csr_nrows(&Hfull_blk), sizeof(csr_data_t));

    for(int i=0; i<csr_ncols(&Hfull); i++) x[i] = CMPLX(NAN,NAN);
    for(int i=0; i<csr_nrows(&Hfull); i++) x[csr_local_rowoffset(&Hfull) + i] = psi0[i];

    // compute the stationary part once
    compute_stationary_matrices(h, dt, &submatrix, &S_blk, &S, &Hst_blk, &Hst, h0, s0);

    // compute the preconditioner once - based on the stationary part
    slu_LU_t sluLU = compute_preconditioner(&S, &Hst);

    if(0){
        csr_index_t *LAi, *LAj;
        csr_index_t *UAi, *UAj;
        csr_data_t *LAx, *UAx;
        csr_index_t Lnnz, Unnz;
        sparse_csr_t spL, spU;
        
        slu_LU2coo(sluLU.L, sluLU.U,
                   &LAi, &LAj, (doublecomplex**)&LAx, &Lnnz,
                   &UAi, &UAj, (doublecomplex**)&UAx, &Unnz);
        
        csr_coo2csr(&spU, UAi, UAj, UAx, csr_nrows(&Hfull), Unnz);
        csr_coo2csr(&spL, LAi, LAj, LAx, csr_nrows(&Hfull), Lnnz);

        free(LAi); free(LAj); free(LAx);
        free(UAi); free(UAj); free(UAx);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);

    // time iterations
    int iter = 0;
    solver_workspace_t wsp = {0};
    time = 0;
    while(time <= maxtime){
        csr_data_t ft[6] = {0};
        complex ihdt = I*h*dt/2;
        time = time + dt;
        beoyndDipolePulse_axialPart(&bdpp, time, ft);

        if(rank==0) {
            printf("------- simulation time %e\n", time);
            printf("f(t)\n");
            for(int i=0; i<6; i++) printf("(%lf,%lf)\n", creal(ft[i]), cimag(ft[i]));
        }

        // time-dependent part of the Hamiltonian
        compute_timedep_matrices(h, dt, &submatrix, ft, lmax, &Hfull_blk, &Hfull, h0, H, g, gt);

        csr_ijk_write("Hds.ijk", &Hfull);
        exit(0);

        if(rank==0){
            tic(); printf("solve iteration %d\n", iter);
        }

        // rhs = (S-H)*psi(n-1)
        for(int i=0; i<csr_nrows(&Hfull); i++) rhs[i] = 0;
        csr_init_communication(&Hfull, x, rank, nranks);
        csr_comm(&Hfull, rank, nranks);
        csr_spmv(0, csr_nrows(&Hfull), &Hfull, x, rhs);
        for(int i=0; i<csr_nrows(&Hfull); i++) rhs[i] = -1*rhs[i];
        csr_spmv(0, csr_nrows(&S), &S, x + csr_local_rowoffset(&Hfull), rhs);

        int iters = 500;
        double tol_error = 1e-16;
        HS_matrices mat;
        mat.H = &Hfull;
        mat.S = &S;
        bicgstab(HS_spmv_fun, &mat, rhs, x, csr_nrows(&Hfull), csr_ncols(&Hfull), csr_local_rowoffset(&Hfull),
                 LU_precond_fun, &sluLU, &wsp, &iters, &tol_error);

        if(rank==0) toc();

        // collect the results on rank 0 for un-permuting and printing
        if(0 == rank){

            // submit recv requests on solution vector parts
            MPI_Request comm_requests[nranks];
            comm_requests[0] = MPI_REQUEST_NULL;
            for(int r=1; r<nranks; r++){
                csr_index_t row_beg = Hall.row_cpu_dist[r]*blkdim;
                csr_index_t row_end = Hall.row_cpu_dist[r+1]*blkdim;
                CHECK_MPI(MPI_Irecv(xall+row_beg, 2*(row_end-row_beg), MPI_DOUBLE, r, 0, MPI_COMM_WORLD, comm_requests+r));
            }

            // copy local vector part
            for(int i=0; i<csr_nrows(&Hfull); i++) xall[i] = x[i];

            // progress communication
            CHECK_MPI(MPI_Waitall(nranks, comm_requests, MPI_STATUSES_IGNORE));

            // un-permute the solution vector
            for(int i=0; i<csr_nrows(&Hall)*blkdim; i++){
                xorig[xperm[i]] = xall[i];
            }            

            // write to file
            FILE *fd;
            char fname[256];
            snprintf(fname, 255, "x%d.out", iter);
            fd = fopen(fname, "w+");
            for(int i=0; i<csr_nrows(&Hall)*blkdim; i++){
                fprintf(fd, "%e %e\n", creal(xorig[i]), cimag(xorig[i]));
            }
            /* fwrite(xorig, sizeof(csr_data_t), csr_nrows(&Hall)*blkdim, fd); */
            fclose(fd);

        } else {
            CHECK_MPI(MPI_Send(x + csr_local_rowoffset(&Hfull), 2*csr_nrows(&Hfull), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD));
        }

        iter++;
    }

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
#endif
}

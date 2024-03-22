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
#include "../src/tictoc.h"
//#include "../src/potential.h"

#define SoL 137.035999084

#ifdef USE_MPI
#include <mpi.h>
#endif

#if defined USE_CUDA | defined USE_HIP
#include "gpu_sparse.h"
#endif

#define max(a,b) ((a)>(b)?(a):(b))

int rank = 0, nranks = 1;

void compute_timedep_matrices(double h, double dt, sparse_csr_t *submatrix, csr_data_t *ft, int lmax,
                              sparse_csr_t *Hfull_blk, sparse_csr_t *Hfull,
                              const sparse_csr_t *h0, const sparse_csr_t *H, const sparse_csr_t *g, const sparse_csr_t *gt);

void compute_stationary_matrices(double h, double dt, sparse_csr_t *submatrix,
                                 sparse_csr_t *S_blk, sparse_csr_t *S,
                                 sparse_csr_t *Hst_blk, sparse_csr_t *Hst,
                                 const sparse_csr_t *h0, const sparse_csr_t *s0);

slu_LU_t compute_preconditioner(const sparse_csr_t *S, const sparse_csr_t *Hst);

// from spnrbasis.cpp
int ik(int i) {
    int ii = i/4;
    int abskappa = (int)(0.5*(sqrt(8.0*ii+1.0) - 1.0)) + 1;
    int sgnmod = max(4,abskappa*4);
    double sgnfloor = 2*abskappa * (abskappa - 1);
    int sgnkappa = ((i-sgnfloor)/sgnmod >= 0.5) - ((i-sgnfloor)/sgnmod < 0.5);
    return abskappa * sgnkappa;
}

double imu(int i) {
    int abskappa = abs(ik(i));
    int kmod = max(2,2*abskappa);
    double mu = i%kmod - abskappa + 0.5;
    return mu;
}

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

void rankprint(char *fname, cdouble_t *v, int n);

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
    slu_LU_t *slup = (slu_LU_t*)precond;
    slu_lu_solve(*slup, (doublecomplex*)rhs, (doublecomplex*)x);
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

    printf("Parameters:\n");
    printf(" - time      %f\n", maxtime);
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

    // allocate x and y vectors for SpMV: y = spA*x
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
    MPI_Barrier(MPI_COMM_WORLD);

    // time iterations
    int iter = 0;
    solver_workspace_t wsp = {0};
    time = 0;
    while(time <= maxtime){
        // TODO get the time to compute time-dependent f(a,t)
        csr_data_t ft[6] = {0};
        complex ihdt = I*h*dt/2;
        time = time + dt;       
        beoyndDipolePulse_axialPart(&bdpp, time, ft);

        if(rank==0) {
            printf("------- simulatoin time %e\n", time);
            printf("f(t)\n");
            for(int i=0; i<6; i++) printf("(%lf,%lf)\n", creal(ft[i]), cimag(ft[i]));
        }
        
        // time-dependent part of the Hamiltonian
        if(rank==0){
            tic(); printf("compute Ht ");
        }
        compute_timedep_matrices(h, dt, &submatrix, ft, lmax, &Hfull_blk, &Hfull, h0, H, g, gt);
        if(rank==0) toc();

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

        {
            char fname[256];
            snprintf(fname, 255, "x%d.out", iter);
            rankprint(fname, x + csr_local_rowoffset(&Hfull), csr_nrows(&Hfull));
        }
        
        iter++;
    }
    
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
#endif
}

void compute_timedep_matrices(double h, double dt, sparse_csr_t *submatrix, csr_data_t *ft, int lmax,
                              sparse_csr_t *Hfull_blk, sparse_csr_t *Hfull,
                              const sparse_csr_t *h0, const sparse_csr_t *H, const sparse_csr_t *g, const sparse_csr_t *gt)
{
    csr_index_t row, col, colp;
    complex ihdt = I*h*dt/2;

    // for all rows
    for(row = 0; row < csr_nrowblocks(Hfull_blk); row++){

        // for non-zeros in each row
        for(colp = Hfull_blk->Ap[row]; colp < Hfull_blk->Ap[row+1]; colp++){

            // NOTE: rows and cols in Hfull_blk are remapped wrt. the original numbering in H
            col = Hfull_blk->Ai[colp];

            // apply node renumbering - if available
            csr_index_t orig_row = row;
            csr_index_t orig_col = col;
            if(Hfull_blk->perm) {
                orig_row = Hfull_blk->perm[csr_local_rowoffset(Hfull_blk) + row];
                orig_col = Hfull_blk->perm[col];
            }

            // calculate kappa and mu parameters from row/col indices
            // see spnrbasis::bdpalphsigmaXmat
            int ki = (int)ik(orig_row); // k'
            int kj = (int)ik(orig_col); // k

            const sparse_csr_t *pg0, *pg1, *pg2, *pg3;
            const sparse_csr_t *pgt0, *pgt1, *pgt2, *pgt3;
            csr_data_t H0[lmax], H1[lmax];

            // prefetch the Hamiltonian values H0(l) and H1(l)
            for(int l=0; l<lmax; l++){
                H0[l] = csr_get_value(H + 2*l + 0, orig_row, orig_col);
                H1[l] = csr_get_value(H + 2*l + 1, orig_row, orig_col);
            }

            csr_zero(submatrix);

            // stationary part of H
            // TODO: check if this is faster than extraction from Hst
            if(orig_row==orig_col) {
                for(csr_index_t i=0; i<csr_nnz(submatrix); i++){
                    submatrix->Ax[i] =
                        ihdt*(h0[0].Ax[i]         +
                              h0[1].Ax[i]*ki      +
                              h0[2].Ax[i]*ki*ki   +
                              h0[3].Ax[i]*ki*ki*ki);
                }
            }

            // the H matrix is still updated with Hst, so do not clear the submatrix
            for(int l=0; l<lmax; l++){
                if(H0[l] != CMPLX(0,0)){

                    for(int a=0; a<6; a++){
                        if((a%2!=l%2)) { //Skip redundant matrices
                            pg0 = g + a*4*lmax + l*4;
                            pg1 = pg0 + 1;
                            pg2 = pg0 + 2;
                            pg3 = pg0 + 3;

                            // g matrices all have the same nnz pattern,
                            // so we can operate directly on the internal storage Ax
                            for(csr_index_t i=0; i<csr_nnz(submatrix); i++){
                                submatrix->Ax[i] +=
                                    SoL*ihdt*ft[a]*H0[l]*(pg0->Ax[i]        +
                                                          pg1->Ax[i]*ki     +
                                                          pg2->Ax[i]*kj     +
                                                          pg3->Ax[i]*ki*kj) ;
                            }
                        }
                    }
                }
            }

            for(int l=0; l<lmax; l++){
                if(H1[l] != CMPLX(0,0)){

                    for(int a=0; a<6; a++){
                        if((a%2!=l%2)) { //Skip redundant matrices
                            pgt0 = gt + a*4*lmax + l*4;
                            pgt1 = pgt0 + 1;
                            pgt2 = pgt0 + 2;
                            pgt3 = pgt0 + 3;

                            // g matrices all have the same nnz pattern,
                            // so we can operate directly on the internal storage Ax
                            for(csr_index_t i=0; i<csr_nnz(submatrix); i++){
                                submatrix->Ax[i] +=
                                    SoL*ihdt*ft[a]*H1[l]*(pgt0->Ax[i]       +
                                                          pgt1->Ax[i]*kj    +
                                                          pgt2->Ax[i]*ki    +
                                                          pgt3->Ax[i]*ki*kj);

                            }
                        }
                    }
                }
            }

            // store the submatrix in the global Hfull_blk
            csr_block_insert(Hfull_blk, row, col, submatrix->Ax);
        }
    }

    csr_blocked_to_full(Hfull, Hfull_blk, submatrix);
}

void compute_stationary_matrices(double h, double dt, sparse_csr_t *submatrix,
                                 sparse_csr_t *S_blk, sparse_csr_t *S,
                                 sparse_csr_t *Hst_blk, sparse_csr_t *Hst,
                                 const sparse_csr_t *h0, const sparse_csr_t *s0)
{
    csr_index_t row, col, colp;
    int blkdim = S_blk->blk_dim;
    complex ihdt = I*h*dt/2;

    tic(); printf("matrix assembly S ");
    // S is block-diagonal, so row == col
    for(row = 0; row < csr_nrowblocks(S_blk); row++){
        csr_index_t orig_row = row;
        if(S_blk->perm) {
            orig_row = S_blk->perm[row];
        }
        int ki = (int)ik(orig_row); // k'
        
        csr_zero(submatrix);
        for(csr_index_t i=0; i < csr_nnz(submatrix); i++){
            submatrix->Ax[i] =
                s0[0].Ax[i]       +
                s0[1].Ax[i]*ki    +
                s0[2].Ax[i]*ki*ki ;
        }

        // store the submatrix in the global S_blk
        csr_block_insert(S_blk,row,row,submatrix->Ax);

        // stationary part of H
        csr_zero(submatrix);
        for(csr_index_t i=0; i<csr_nnz(submatrix); i++){
            submatrix->Ax[i] =
                ihdt*(h0[0].Ax[i]         +
                      h0[1].Ax[i]*ki      +
                      h0[2].Ax[i]*ki*ki   +
                      h0[3].Ax[i]*ki*ki*ki);
        }

        // store the submatrix in the global Hst_blk - need it for ILU
        csr_block_insert(Hst_blk, row, row, submatrix->Ax);
    }

    tic(); printf("convert S to non-blocked matrix ");
    csr_blocked_to_full(S, S_blk, submatrix);
    toc();

    tic(); printf("convert Hst to non-blocked matrix ");
    csr_blocked_to_full(Hst, Hst_blk, submatrix);
    toc();
}

slu_LU_t compute_preconditioner(const sparse_csr_t *S, const sparse_csr_t *Hst)
{
    csr_index_t row, col, colp;
    slu_matrix_t sluA;
    slu_LU_t sluLU;

    csr_data_t *P = (csr_data_t*)malloc(sizeof(csr_data_t)*Hst->nnz);

    tic(); printf("construct base preconditioner matrix ");

    // preconditioner is based on ILU of S+Hst, so compute Hst = S+Hst
    // this is trivial since S and Hst have the same, block-diagonal structure
    for(row = 0; row < Hst->nrows; row++){
        for(colp = Hst->Ap[row]; colp < Hst->Ap[row+1]; colp++){
            P[colp] = S->Ax[colp] + Hst->Ax[colp];
        }
    }
    toc();

    // compute ILU
    sluA = slu_create_matrix(csr_nrows(Hst), csr_ncols(Hst), csr_nnz(Hst), P, Hst->Ai, Hst->Ap);
    sluLU = slu_compute_ilu(sluA);
    free(P);

    return sluLU;
}

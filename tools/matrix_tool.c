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
#include "../src/tictoc.h"
//#include "../src/potential.h"

#ifdef USE_MPI
#include <mpi.h>
#endif

#if defined USE_CUDA | defined USE_HIP
#include "gpu_sparse.h"
#endif

#define max(a,b) ((a)>(b)?(a):(b))


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
        if(fabs(cimag(v1[i]-v2[i]))>1e-8) fprintf(stderr, "%e i\n", cimag(v1[i]) - cimag(v2[i]));
        if(fabs(creal(v1[i]-v2[i]))>1e-8) fprintf(stderr, "%e\n", creal(v1[i]) - creal(v2[i]));
    }
}

int main(int argc, char *argv[])
{
    char fname[256];
    sparse_csr_t Hall, Hfull, Hfull_blk, Hpart, Sdiag, S, S_blk, Spart;
    sparse_csr_t *g, *gt, *h0, *s0;
    sparse_csr_t *H;
    int opt;
    int lmax;
    double intensity, omega, cycles;
    int cnt;

    if(argc<5){
        fprintf(stderr, "Usage: %s -l lmax -i intensity -o omega -c cycles\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    while ((opt = getopt(argc, argv, "l:i:o:c:")) != -1) {
        switch (opt) {
        case 'l': lmax = atoi(optarg); break;
        case 'i': intensity = atof(optarg); break;
        case 'o': omega = atof(optarg); break;
        case 'c': cycles = atof(optarg); break;
        default:
            fprintf(stderr, "Usage: %s [-lioc]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    printf("Parameters:\n");
    printf(" - lmax      %d\n", lmax);
    printf(" - intensity %lf\n", intensity);
    printf(" - omega     %lf\n", omega);
    printf(" - cycles    %lf\n", cycles);

    // time-dependent part
    beyondDipolePulse_t bdpp;
    beyondDipolePulse_init(&bdpp, intensity, omega, cycles);

    // read the full couplings matrix structure
    // each non-zero value denotes a submatrix of dimensions same as the G matrices
    snprintf(fname, 255, "H.csr");
    csr_read(fname, &Hall);

    // for convenience, create a diagonal matrix for the S-part
    csr_diag(&Sdiag, csr_nrows(&Hall));

    // copy partitioning from Hall
    Sdiag.row_cpu_dist = Hall.row_cpu_dist;
    Sdiag.row_beg = Hall.row_beg;
    Sdiag.row_end = Hall.row_end;
    if(Hall.perm){
        Sdiag.perm = (csr_index_t*)malloc(sizeof(csr_index_t)*Sdiag.nrows);
        memcpy(Sdiag.perm, Hall.perm + Hall.local_offset, sizeof(csr_index_t)*Hall.nrows);
    }
    
    int rank = 0, nranks = 1;
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    if(nranks>1 && Hall.npart != nranks)
        ERROR("There are %d MPI ranks, but the matrix is partition for %d ranks.\n", nranks, Hall.npart);
#endif

    // get local partition of the global H matrix
    if(nranks>1){
        csr_get_partition(&Hpart, &Hall, rank, Hall.npart);

        // partitioning of the S-part is trivial, but still needs to be done
        csr_get_partition(&Spart, &Sdiag, rank, Hall.npart);

        /* DEBUG */
        /* snprintf(fname, 255, "H_part%d.csr", rank); */
        /* csr_write(fname, &Hpart); */
    } else {
        Hpart = Hall;
        Spart = Sdiag;
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
    // same Ap and Ai as Hpart
    // Ax modified to store sub-matrices (g.nnz)
    int blkdim = csr_nrows(&g[0]);
    csr_copy(&Hfull_blk, &Hpart);
    csr_copy(&S_blk, &Spart);

    csr_block_params(&Hfull_blk, blkdim, csr_nnz(&g[0]));
    csr_block_params(&S_blk, blkdim, csr_nnz(&s0[0]));

    // create the non-blocked rank-local Hfull matrix structure
    //  - each non-zero is converted to blkdim x blkdim submatrix
    //  - each row has row_nnz(Hpart)*row_nnz(G) non-zero entries
    //  - create Hfull.Ap and Hfull.Ai accordingly
    // Dimensions of Hfull and Hfull_blk are the same, and they have
    // the same number of non-zeros. However, their Ap and Ai differ:
    // Hfull_blk inherits Ap and Ai directly from Hpart - one non-zero per entire submatrix.
    // Hfull stores all non-zeros independently in a native, non-blocked csr storage
    csr_unblock_matrix(&Hfull, &Hfull_blk, g);
    csr_unblock_matrix(&S, &S_blk, s0);

    // Hfull_blk has the comm info copied from Hpart,
    // but Hfull doesnt - it has to be modified by the block size.
    // At this point there is no comm info in Hfull - so copy it.
    csr_unblock_comm_info(&Hfull, &Hfull_blk, rank, nranks);
    csr_unblock_comm_info(&S, &S_blk, rank, nranks);

    for(int r=0; r<nranks; r++){
        if(rank == r){
            printf("%d: All matrices read correctly. System info:\n", rank);
            printf(" - H dim:         %d x %d\n", csr_nrows(&H[0]), csr_ncols(&H[0]));
            printf(" - Hall dim:      %d x %d\n", csr_nrows(&Hall), csr_ncols(&Hall));
            printf(" - Hall nnz:      %d\n", csr_nnz(&Hall));
            printf(" - Hpart dim:     %d x %d\n", csr_nrows(&Hpart), csr_ncols(&Hpart));
            printf(" - Hpart nnz:     %d\n", csr_nnz(&Hpart));
            printf(" - Hfull dim:     %d x %d\n", csr_nrows(&Hfull_blk), csr_ncols(&Hfull_blk));
            printf(" - Hfull nnz:     %d\n", csr_nnz(&Hfull_blk));
        }
#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }

    // allocate x and y vectors for SpMV: y = spA*x
    csr_data_t *x, *yblk, *yfull;
    x     = (csr_data_t *)calloc(csr_ncols(&Hfull_blk), sizeof(csr_data_t));
    yblk  = (csr_data_t *)calloc(csr_nrows(&Hfull_blk), sizeof(csr_data_t));
    yfull = (csr_data_t *)calloc(csr_nrows(&Hfull_blk), sizeof(csr_data_t));

    { // Compute
        csr_index_t row, col, colp;

        // each submatrix will have the same non-zero structure as the base g matrices
        sparse_csr_t submatrix;
        csr_copy(&submatrix, g);

        tic(); printf("matrix assembly H ");
        // for all rows
        for(row = 0; row < csr_nrows(&Hpart); row++){

            // for non-zeros in each row
            for(colp = Hpart.Ap[row]; colp < Hpart.Ap[row+1]; colp++){

                // NOTE: rows and cols in Hpart are remapped wrt. the original numbering in H
                col = Hpart.Ai[colp];

                // apply node renumbering - if available
                csr_index_t orig_row = row;
                csr_index_t orig_col = col;
                if(Hpart.perm) {
                    orig_row = Hpart.perm[csr_local_offset(&Hpart) + row];
                    orig_col = Hpart.perm[col];
                }

                // calculate kappa and mu parameters from row/col indices
                // see spnrbasis::bdpalphsigmaXmat
                int ki = (int)ik(orig_row); // k'
                int kj = (int)ik(orig_col); // k

                // double mui = imu(orig_row); // mu'
                // double muj = imu(orig_col); // mu

                sparse_csr_t *pg0, *pg1, *pg2, *pg3;
                sparse_csr_t *pgt0, *pgt1, *pgt2, *pgt3;
                csr_data_t H0[lmax], H1[lmax];
                csr_data_t ft[6];

                // TODO get the time to compute time-dependent f(a,t)
                double t = 1;
                beoyndDipolePulse_axialPart(&bdpp, t, ft);

                // prefetch the Hamiltonian values H0(l) and H1(l)
                for(int l=0; l<lmax; l++){
                    H0[l] = csr_get_value(H + 2*l + 0, orig_row, orig_col);
                    H1[l] = csr_get_value(H + 2*l + 1, orig_row, orig_col);
                }

                csr_zero(&submatrix);

                if(orig_row==orig_col) {
                    for(csr_index_t i=0; i<csr_nnz(&submatrix); i++){
                        submatrix.Ax[i] =
                            h0[0].Ax[i]         +
                            h0[1].Ax[i]*ki      +
                            h0[2].Ax[i]*ki*ki   +
                            h0[3].Ax[i]*ki*ki*ki;
                    }
                }

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
                                for(csr_index_t i=0; i<csr_nnz(&submatrix); i++){
                                    submatrix.Ax[i] +=
                                        ft[a]*H0[l]*(pg0->Ax[i]        +
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
                                for(csr_index_t i=0; i<csr_nnz(&submatrix); i++){
                                    submatrix.Ax[i] +=
                                        ft[a]*H1[l]*(pgt0->Ax[i]       +
                                                     pgt1->Ax[i]*kj    +
                                                     pgt2->Ax[i]*ki    +
                                                     pgt3->Ax[i]*ki*kj);

                                }
                            }
                        }
                    }
                }

                // store the submatrix in the global Hfull_blk
                csr_block_insert(&Hfull_blk, row, col, submatrix.Ax);
            }
        }
        toc();

        tic(); printf("matrix assembly S ");
        // S is block-diagonal, so row == col
        for(row = 0; row < csr_nrows(&Spart); row++){
            csr_index_t orig_row = row;
            int ki = (int)ik(orig_row); // k'
            if(Spart.perm) {
                orig_row = Spart.perm[row];
            }
            csr_zero(&submatrix);
            for(csr_index_t i=0; i < csr_nnz(&submatrix); i++){
                submatrix.Ax[i] =
                    s0[0].Ax[i]       +
                    s0[1].Ax[i]*ki    +
                    s0[2].Ax[i]*ki*ki ;
            }
                    
            // store the submatrix in the global S_blk
            csr_block_insert(&S_blk,row,row,submatrix.Ax);
        }

        toc();
        
        // convert blocked Hfull_blk to non-blocked Hfull
        // could be done immediately above, but we do it here for timing purposes
        tic(); printf("convert HI to non-blocked matrix ");
        for(row = 0; row < Hfull_blk.nrows; row++){

            // for non-zeros in each row
            for(colp = Hfull_blk.Ap[row]; colp < Hfull_blk.Ap[row+1]; colp++){

                // NOTE: rows and cols are remapped wrt. the original numbering in H
                col = Hfull_blk.Ai[colp];

                csr_block_link(&submatrix, &Hfull_blk, row, col);

                // insert into non-blocked Hfull matrix
                csr_index_t row_blk, col_blk, colp_blk;
                csr_index_t row_dst, col_dst;
                csr_index_t valp = 0;
                for(row_blk=0; row_blk<blkdim; row_blk++){

                    row_dst = row*blkdim + row_blk;
                    for(colp_blk=submatrix.Ap[row_blk]; colp_blk<submatrix.Ap[row_blk+1]; colp_blk++){
                        col_blk = submatrix.Ai[colp_blk];
                        col_dst = col*blkdim + col_blk;

                        csr_set_value(&Hfull, row_dst, col_dst, submatrix.Ax[valp]);
                        valp++;
                    }
                }
            }
        }
        toc();

        tic(); printf("convert S to non-blocked matrix ");
        for(row = 0; row < S_blk.nrows; row++){

            // for non-zeros in each row
            for(colp = S_blk.Ap[row]; colp < S_blk.Ap[row+1]; colp++){

                // NOTE: rows and cols are remapped wrt. the original numbering in H
                col = S_blk.Ai[colp];

                csr_block_link(&submatrix, &S_blk, row, col);

                // insert into non-blocked Hfull matrix
                csr_index_t row_blk, col_blk, colp_blk;
                csr_index_t row_dst, col_dst;
                csr_index_t valp = 0;
                for(row_blk=0; row_blk<blkdim; row_blk++){

                    row_dst = row*blkdim + row_blk;
                    for(colp_blk=submatrix.Ap[row_blk]; colp_blk<submatrix.Ap[row_blk+1]; colp_blk++){
                        col_blk = submatrix.Ai[colp_blk];
                        col_dst = col*blkdim + col_blk;

                        csr_set_value(&S, row_dst, col_dst, submatrix.Ax[valp]);
                        valp++;
                    }
                }
            }
        }
        toc();
        
        // The Hfull_blk matrix contains all computed submatrices.
        // The submatrices are stored as a sub-block in the csr storage
        // meaning that the relevant Ax parts can be used directly
        // as Ax arrays in a template submatrix csr structure, e.g.
        // csr_block_link(&submatrix, &Hfull_blk, row, col);

        // set up communication buffers
        csr_init_communication(&Hfull_blk, x, rank, nranks);
        csr_init_communication(&Hfull, x, rank, nranks);
               
        // initialize input vector. non-local parts are set to nan to verify communication:
        // all non-local entries are received from peers, hence set to non-nan during communication
        for(int i=0; i<csr_ncols(&Hfull_blk); i++) x[i] = CMPLX(NAN,NAN);
        for(int i=0; i<csr_nrows(&Hfull_blk); i++)
            x[csr_local_offset(&Hfull_blk) + i] = CMPLX(1, 1); // CMPLX(Hfull_blk.row_beg*blkdim + i, Hfull_blk.row_beg*blkdim + i);

        tic(); printf("    blocked spmv ");
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

        // initialize input vector. non-local parts are set to nan to verify communication:
        // all non-local entries are received from peers, hence set to non-nan during communication
        for(int i=0; i<csr_ncols(&Hfull); i++) x[i] = CMPLX(NAN,NAN);
        for(int i=0; i<csr_nrows(&Hfull); i++)
            x[csr_local_offset(&Hfull) + i] = CMPLX(1, 1); // CMPLX(Hfull.row_beg + i, Hfull.row_beg + i);

        // perform spmv for the non-blocked Hfull matrix (native csr storage)
        tic(); printf("non-blocked spmv ");
        csr_comm(&Hfull, rank, nranks);
        csr_spmv(0, csr_nrows(&Hfull), &Hfull, x, yfull);
        toc();
        
        // validate - compare yblk and yfull results
        compare_vectors(yfull, yblk, csr_nrows(&Hfull));

#if defined USE_CUDA | defined USE_HIP
        gpu_spmv_test(Hfull, x, yfull);
        // gpu_spmb_block_test(Hfull_blk, x, yfull, g);
#endif


        // DEBUG: write out the result vectors for comparison with single-rank result
        /*
          for(int r=0; r<nranks; r++){
          if(rank == r){
          for(row = 0; row < csr_nrows(&Hfull_blk); row++) fprintf(stderr, "%e %e\n", creal(yblk[row]), cimag(yblk[row]));
          }
          MPI_Barrier(MPI_COMM_WORLD);
          }
          MPI_Barrier(MPI_COMM_WORLD);
        */
    }

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
#endif
}

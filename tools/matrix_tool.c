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

#define max(a,b) ((a)>(b)?(a):(b))

#include "csr.h"
#include "../src/tictoc.h"

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

int main(int argc, char *argv[])
{
    if(argc<2){
        printf("Usage: matrix_tool <lmax>\n");
        return 0;
    }

    char fname[256];
    sparse_csr_t Hall, Hfull, Hfull_blk;
    sparse_csr_t *g, *gt;
    sparse_csr_t *H;
    int lmax = atoi(argv[1]);
    int cnt;

    // read the full couplings matrix structure
    // each non-zero value denotes a submatrix of dimensions same as the G matrices
    snprintf(fname, 255, "H.csr");
    csr_read(fname, &Hall);

    // read the individual Hamiltonian matrices H0 and H1
    // 2 matrices for each value of 0:lmax-1
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

    // create the full matrix blocked structure
    // same Ap and Ai as Hall
    // Ax modified to store sub-matrices (g.nnz)
    int blkdim = csr_dim(&g[0]);
    csr_copy(&Hfull_blk, &Hall);
    csr_block_update(&Hfull_blk, blkdim, csr_nnz(&g[0]));

    // create the non-blocked Hfull matrix structure
    //  - each non-zero is converted to blkdim x blkdim submatrix
    //  - each row has row_nnz(Hall)*row_nnz(G) non-zero entries
    //  - create Hfull.Ap and Hfull.Ai accordingly
    // Dimensions of Hfull and Hfull_blk are the same, and they have
    // the same number of non-zeros. However, their Ap and Ai differ:
    // Hfull_blk inherits Ap and Ai directly from Hall - one non-zero
    // per entire submatrix.
    // Hfull stores all non-zeros independently in a native, non-blocked csr storage
    csr_allocate(&Hfull, csr_dim(&Hfull_blk), csr_nnz(&Hfull_blk));
    {
        // iterators over Hall
        csr_index_t row, col, colp;

        // iterators over G (block submatrices)
        csr_index_t row_blk, colp_blk;

        // iterators over non-blocked Hfull
        csr_index_t col_dst, rowp_dst;
        rowp_dst = 1;

        // for all rows
        for(row = 0; row < csr_dim(&Hall); row++){

            // each row and each column are expanded into submatrices of size blkdim
            for(row_blk=0; row_blk<blkdim; row_blk++){

                // row in the expanded matrix
                // row_dst = row*blkdim + row_blk;

                // for non-zeros in each Hall row - fill the expanded row
                for(colp = Hall.Ap[row]; colp < Hall.Ap[row+1]; colp++){
                    col = Hall.Ai[colp];

                    for(colp_blk=g[0].Ap[row_blk]; colp_blk<g[0].Ap[row_blk+1]; colp_blk++){

                        // column in the expanded matrix
                        col_dst = col*blkdim + g[0].Ai[colp_blk];

                        // update Hfull.Ai and Hfull.Ap
                        Hfull.Ai[Hfull.Ap[rowp_dst]] = col_dst;
                        Hfull.Ap[rowp_dst]++;
                    }
                }

                // next Hfull row - start where the last one ends
                Hfull.Ap[rowp_dst+1] = Hfull.Ap[rowp_dst];
                rowp_dst++;
            }
        }
    }

    printf("All matrices read correctly. System info:\n");
    printf(" - H dim: %d\n", csr_dim(&H[0]));
    printf(" - Hall dim: %d\n", csr_dim(&Hall));
    printf(" - Hall nnz: %d\n", csr_nnz(&Hall));
    printf(" - Hfull_blk dim: %d\n", csr_dim(&Hfull_blk));
    printf(" - Hfull_blk nnz: %d\n", csr_nnz(&Hfull_blk));

    // allocate x and y vectors for SpMV
    csr_data_t *x, *yblk, *yfull;
    x = (csr_data_t *)calloc(csr_dim(&Hfull_blk), sizeof(csr_data_t));
    yblk = (csr_data_t *)calloc(csr_dim(&Hfull_blk), sizeof(csr_data_t));
    yfull = (csr_data_t *)calloc(csr_dim(&Hfull_blk), sizeof(csr_data_t));

    {
        csr_index_t row, col, colp;

        // each submatrix will have the same non-zero structure as the base g matrices
        sparse_csr_t submatrix;
        csr_copy(&submatrix, g);

        tic();
        // for all rows
        for(row = 0; row < csr_dim(&Hall); row++){

            // for non-zeros in each row
            for(colp = Hall.Ap[row]; colp < Hall.Ap[row+1]; colp++){

                // NOTE: rows and cols in Hall are remapped wrt. the original numbering in H
                col = Hall.Ai[colp];

                // apply node renumbering - if available
                csr_index_t orig_row = row;
                csr_index_t orig_col = col;
                if(Hall.map) {
                    orig_row = Hall.map[row];
                    orig_col = Hall.map[col];
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

                // TODO get the time-dependent f(a,t)
                for(int a=0; a<6; a++){
                    ft[a] = CMPLX(1,0);
                }

                // prefetch the Hamiltonian values H0(l) and H1(l)
                for(int l=0; l<lmax; l++){
                    H0[l] = csr_get_value(H + 2*l + 0, orig_row, orig_col);
                    H1[l] = csr_get_value(H + 2*l + 1, orig_row, orig_col);
                }

                csr_zero(&submatrix);
                for(int l=0; l<lmax; l++){
                    if(H0[l] != CMPLX(0,0)){

                        for(int a=0; a<6; a++){

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

                for(int l=0; l<lmax; l++){
                    if(H1[l] != CMPLX(0,0)){

                        for(int a=0; a<6; a++){

                            pgt0 = gt + a*4*lmax + l*4;
                            pgt1 = pgt0 + 1;
                            pgt2 = pgt0 + 2;
                            pgt3 = pgt0 + 3;

                            // g matrices all have the same nnz pattern,
                            // so we can operate directly on the internal storage Ax
                            for(csr_index_t i=0; i<csr_nnz(&submatrix); i++){
                                submatrix.Ax[i] -=
                                    ft[a]*H1[l]*(pgt0->Ax[i]       +
                                                 pgt1->Ax[i]*ki    +
                                                 pgt2->Ax[i]*kj    +
                                                 pgt3->Ax[i]*ki*kj);
                            }
                        }
                    }
                }

                // store the submatrix in the global Hfull_blk
                csr_block_insert(&Hfull_blk, row, col, submatrix.Ax);
            }
        }
        toc();

        // convert blocked Hfull_blk to non-blocked Hfull
        // could be done immediately above, but we do it here for timing purposes
        tic();
        // for all rows
        for(row = 0; row < csr_dim(&Hall); row++){

            // for non-zeros in each row
            for(colp = Hall.Ap[row]; colp < Hall.Ap[row+1]; colp++){

                // NOTE: rows and cols in Hall are remapped wrt. the original numbering in H
                col = Hall.Ai[colp];

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
        
        // The Hfull_blk matrix contains all computed submatrices.
        // The submatrices are stored as a sub-block in the csr storage
        // meaning that the relevant Ax parts can be used directly
        // as Ax arrays in a template submatrix csr structure, e.g.
        // csr_block_link(&submatrix, &Hfull_blk, row, col);

        // initialize input vector
        for(int i=0; i<csr_dim(&Hfull_blk); i++) x[i] = CMPLX(1,0);

        // DEBUG set all matrix nnz values to 1
        // for(csr_index_t i=0; i<csr_nnz(&Hfull_blk); i++) Hfull_blk.Ax[i] = CMPLX(1,0);

        tic();
        // for all block rows
        for(row = 0; row < Hfull_blk.dim; row++){

            // for non-zero blocks in each row
            for(colp = Hfull_blk.Ap[row]; colp < Hfull_blk.Ap[row+1]; colp++){

                // NOTE: rows and cols in Hall are remapped wrt. the original numbering in H
                col = Hfull_blk.Ai[colp];

                csr_data_t *xin, *yout;
                csr_block_link(&submatrix, &Hfull_blk, row, col);
                xin  = x + col*blkdim;
                yout = yblk + row*blkdim;

                // perform spmv
                spmv_crs_f(0, csr_dim(&submatrix), &submatrix, xin, yout);

                // remember that at this stage xin and yout are renumbered wrt. the original node numbering
            }
        }
        toc();

        // perform spmv for the non-blocked Hfull matrix (native CSR storage)
        tic();
        spmv_crs_f(0, csr_dim(&Hfull), &Hfull, x, yfull);
        toc();

        // validate - compare yblk and yfull results
        for(int i=0; i<csr_dim(&Hfull); i++) {
            if(fabs(cimag(yfull[i]-yblk[i]))>1e-10) printf("%e *i\n", cimag(yfull[i]) - cimag(yblk[i]));
            if(fabs(creal(yfull[i]-yblk[i]))>1e-10) printf("%e\n", creal(yfull[i]) - cimag(yblk[i]));
        }
    }
}

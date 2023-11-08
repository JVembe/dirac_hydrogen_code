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
    FILE *fd;
    sparse_csr_t Hall;
    sparse_csr_t *g;
    sparse_csr_t *H;
    int lmax = atoi(argv[1]);
    int cnt;

    // read the full couplings matrix structure
    snprintf(fname, 255, "H.csr");
    csr_read(fname, &Hall);

    // read the individual Hamiltonian matrices
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

    printf("all matrices read correctly\n");
    
    {
        csr_index_t row, col, colp;

        // each submatrix will have the same non-zero structure as the base g matrices
        sparse_csr_t submatrix;
        csr_copy(&submatrix, g);

        // for all rows
        for(row = 0; row < Hall.dim; row++){
            // for non-zeros in each row
            for(colp = Hall.Ap[row]; colp < Hall.Ap[row+1]; colp++){
                col = Hall.Ai[colp];

                // apply node renumbering - if available
                csr_index_t mrow = row;
                csr_index_t mcol = col;
                if(Hall.map) {
                    mrow = Hall.map[row];
                    mcol = Hall.map[col];
                }

                // calculate kappa and mu parameters from row/col indices
                // see spnrbasis::bdpalphsigmaXmat
                int ki = (int)ik(row); // k'
                int kj = (int)ik(col); // k

                double mui = imu(row); // mu'
                double muj = imu(col); // mu

                csr_zero(&submatrix);
                for(int a=0; a<6; a++){
                    for(int l=0; l<lmax; l++){
                    }
                }
            }
        }
    }

    // compose the full sparse matrix

    // compose the matrix row by row

    // compose the individual sub-matrices separately for each non-zero
}

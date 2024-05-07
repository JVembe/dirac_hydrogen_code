#include "solver.h"
#include <omp.h>
extern int rank;
static int *ikarr = NULL;

void compute_timedep_matrices_slow(double h, double dt, sparse_csr_t *_submatrix, csr_data_t *ft, int lmax,
                                   sparse_csr_t *Hfull_blk, sparse_csr_t *Hfull,
                                   const sparse_csr_t *h0, const sparse_csr_t *H, const sparse_csr_t *g, const sparse_csr_t *gt)
{
    if(NULL == ikarr){
        // precompute ik indices
        ikarr = malloc(sizeof(csr_index_t)*csr_ncolblocks(Hfull_blk));
        for(csr_index_t col = 0; col < csr_ncolblocks(Hfull_blk); col++){
            csr_index_t orig_col = col;
            if(Hfull_blk->perm) {
                orig_col = Hfull_blk->perm[col];
            }
            ikarr[col] = (int)ik(orig_col);
        }
    }

    char *envar;
    int nthr = 1;
    envar = getenv("MAT_NUM_THREADS");
    if(NULL == envar) envar = getenv("OMP_NUM_THREADS");
    if(NULL != envar) nthr = atoi(envar);
    if(0 == nthr) nthr = 1;
    tic(); PRINTF0("compute Ht on %d threads ", nthr);

#pragma omp parallel num_threads(nthr)
    {
        csr_index_t row, col, colp;
        complex ihdt = I*h*dt/2;

        // copy submatrix - should be thread-local
        sparse_csr_t lsubmatrix, *submatrix;
        submatrix = &lsubmatrix;
        csr_copy(submatrix, _submatrix);

        int nrows = csr_nrowblocks(Hfull_blk);

#pragma omp for
        // for all rows
        for(row = 0; row < nrows; row++){

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
                /* int ki = (int)ik(orig_row); // k' */
                /* int kj = (int)ik(orig_col); // k */

                // precomputed is faster
                int ki = ikarr[csr_local_rowoffset(Hfull_blk) + row];
                int kj = ikarr[col];

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
                if(orig_row==orig_col) {
                    for(csr_index_t i=0; i<csr_nnz(submatrix); i++){
                        submatrix->Ax[i] =
                            ihdt*(h0[0].Ax[i]         +
                                  h0[1].Ax[i]*ki      +
                                  h0[2].Ax[i]*ki*ki   +
                                  h0[3].Ax[i]*ki*ki*ki);
                    }
                }

                // instead we can extract it from Hst_blk, but speed is the same / worse
                /* if(orig_row==orig_col) { */
                /*     csr_block_link(&tmpsp, Hst_blk, row, row); */
                /*     memcpy(submatrix->Ax, tmpsp.Ax, csr_nnz(submatrix)*sizeof(csr_data_t)); */
                /* } else { */
                /*     csr_zero(submatrix); */
                /* } */

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

                /* // store the submatrix in the global Hfull_blk */
                /* if(Hfull_blk->Ax) csr_block_insert(Hfull_blk, row, col, submatrix->Ax); */

                // store immediately in non-blocked Hfull matrix
                csr_full_insert(Hfull, row, col, submatrix);
            }
        }
        csr_free(submatrix);
    }

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    if(rank==0){
        toc();
    }

    // un-block, if above we used the blocked storage
    // csr_blocked_to_full(Hfull, Hfull_blk, submatrix);
}

static csr_data_t **pgsum = NULL;
static csr_data_t **pgtsum = NULL;

void compute_timedep_matrices(double h, double dt, sparse_csr_t *_submatrix, csr_data_t *ft, int lmax,
                              sparse_csr_t *Hfull_blk, sparse_csr_t *Hfull,
                              const sparse_csr_t *h0, const sparse_csr_t *H, const sparse_csr_t *g, const sparse_csr_t *gt)
{
    if(NULL == ikarr){
        // precompute ik indices
        ikarr = malloc(sizeof(csr_index_t)*csr_ncolblocks(Hfull_blk));
        for(csr_index_t col = 0; col < csr_ncolblocks(Hfull_blk); col++){
            csr_index_t orig_col = col;
            if(Hfull_blk->perm) {
                orig_col = Hfull_blk->perm[col];
            }
            ikarr[col] = (int)ik(orig_col);
        }
    }

    char *envar;
    int nthr = 1;
    envar = getenv("MAT_NUM_THREADS");
    if(NULL == envar) envar = getenv("OMP_NUM_THREADS");
    if(NULL != envar) nthr = atoi(envar);
    if(0 == nthr) nthr = 1;
    tic(); PRINTF0("compute Ht on %d threads ", nthr);

#pragma omp parallel num_threads(nthr)
    {
        csr_index_t row, col, colp;
        complex ihdt = I*h*dt/2;

        // copy submatrix - should be thread-local
        sparse_csr_t lsubmatrix, *submatrix;
        submatrix = &lsubmatrix;
        csr_copy(submatrix, _submatrix);

        int nrows = csr_nrowblocks(Hfull_blk);

        const csr_data_t *pg0, *pg1, *pg2, *pg3;
        const csr_data_t *pgt0, *pgt1, *pgt2, *pgt3;

        // precopute G-sums and Gt-sums
#pragma omp master
        {
            if(NULL == pgsum){
                pgsum = calloc(lmax*4, sizeof(csr_data_t*));
                pgtsum = calloc(lmax*4, sizeof(csr_data_t*));
                for(int i=0; i<lmax*4; i++){
                    pgsum[i]  = calloc(submatrix->nnz, sizeof(csr_data_t));
                    pgtsum[i] = calloc(submatrix->nnz, sizeof(csr_data_t));
                }
            }
            for(int l=0; l<lmax; l++){
                memset(pgsum[l*4 + 0], 0, sizeof(csr_data_t)*submatrix->nnz);
                memset(pgsum[l*4 + 1], 0, sizeof(csr_data_t)*submatrix->nnz);
                memset(pgsum[l*4 + 2], 0, sizeof(csr_data_t)*submatrix->nnz);
                memset(pgsum[l*4 + 3], 0, sizeof(csr_data_t)*submatrix->nnz);

                memset(pgtsum[l*4 + 0], 0, sizeof(csr_data_t)*submatrix->nnz);
                memset(pgtsum[l*4 + 1], 0, sizeof(csr_data_t)*submatrix->nnz);
                memset(pgtsum[l*4 + 2], 0, sizeof(csr_data_t)*submatrix->nnz);
                memset(pgtsum[l*4 + 3], 0, sizeof(csr_data_t)*submatrix->nnz);
                for(int a=0; a<6; a++){
                    if((a%2!=l%2)) { //Skip redundant matrices

                        // G-sums
                        pg0 = g[a*4*lmax + l*4 + 0].Ax;
                        pg1 = g[a*4*lmax + l*4 + 1].Ax;
                        pg2 = g[a*4*lmax + l*4 + 2].Ax;
                        pg3 = g[a*4*lmax + l*4 + 3].Ax;

                        // Gt-sums
                        pgt0 = gt[a*4*lmax + l*4 + 0].Ax;
                        pgt1 = gt[a*4*lmax + l*4 + 1].Ax;
                        pgt2 = gt[a*4*lmax + l*4 + 2].Ax;
                        pgt3 = gt[a*4*lmax + l*4 + 3].Ax;

                        // g matrices all have the same nnz pattern,
                        // so we can operate directly on the internal storage Ax
                        for(csr_index_t i=0; i<csr_nnz(submatrix); i++){
                            pgsum[l*4 + 0][i]  += SoL*ihdt*ft[a]*pg0[i];  //
                            pgsum[l*4 + 1][i]  += SoL*ihdt*ft[a]*pg1[i];  // ki
                            pgsum[l*4 + 2][i]  += SoL*ihdt*ft[a]*pg2[i];  // kj
                            pgsum[l*4 + 3][i]  += SoL*ihdt*ft[a]*pg3[i];  // ki*kj

                            pgtsum[l*4 + 0][i] += SoL*ihdt*ft[a]*pgt0[i]; //
                            pgtsum[l*4 + 1][i] += SoL*ihdt*ft[a]*pgt1[i]; // kj
                            pgtsum[l*4 + 2][i] += SoL*ihdt*ft[a]*pgt2[i]; // ki
                            pgtsum[l*4 + 3][i] += SoL*ihdt*ft[a]*pgt3[i]; // ki*kj
                        }
                    }
                }
            }
        }

#pragma omp barrier
#pragma omp for
        // for all rows
        for(row = 0; row < nrows; row++){

            csr_index_t loc = Hfull_blk->Ap[row]*Hfull_blk->blk_nnz;

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
                /* int ki = (int)ik(orig_row); // k' */
                /* int kj = (int)ik(orig_col); // k */

                // precomputed is faster
                int ki = ikarr[csr_local_rowoffset(Hfull_blk) + row];
                int kj = ikarr[col];

                csr_data_t H0[lmax], H1[lmax];

                // prefetch the Hamiltonian values H0(l) and H1(l)
                for(int l=0; l<lmax; l++){
                    H0[l] = csr_get_value(H + 2*l + 0, orig_row, orig_col);
                    H1[l] = csr_get_value(H + 2*l + 1, orig_row, orig_col);
                }

                csr_zero(submatrix);

                // stationary part of H
                if(orig_row==orig_col) {
                    for(csr_index_t i=0; i<csr_nnz(submatrix); i++){
                        submatrix->Ax[i] =
                            ihdt*(h0[0].Ax[i]         +
                                  h0[1].Ax[i]*ki      +
                                  h0[2].Ax[i]*ki*ki   +
                                  h0[3].Ax[i]*ki*ki*ki);
                    }
                }

                // instead we can extract it from Hst_blk, but speed is the same / worse
                /* if(orig_row==orig_col) { */
                /*     csr_block_link(&tmpsp, Hst_blk, row, row); */
                /*     memcpy(submatrix->Ax, tmpsp.Ax, csr_nnz(submatrix)*sizeof(csr_data_t)); */
                /* } else { */
                /*     csr_zero(submatrix); */
                /* } */

                // the H matrix is still updated with Hst, so do not clear the submatrix
                for(int l=0; l<lmax; l++){

                    if(H0[l] != CMPLX(0,0) && H1[l] != CMPLX(0,0)) {
                        for(csr_index_t i=0; i<csr_nnz(submatrix); i++){
                            submatrix->Ax[i] +=
                                H0[l]*(pgsum[l*4 + 0][i]         +
                                       pgsum[l*4 + 1][i]*ki      +
                                       pgsum[l*4 + 2][i]*kj      +
                                       pgsum[l*4 + 3][i]*ki*kj)  +
                                H1[l]*(pgtsum[l*4 + 0][i]        +
                                       pgtsum[l*4 + 1][i]*kj     +
                                       pgtsum[l*4 + 2][i]*ki     +
                                       pgtsum[l*4 + 3][i]*ki*kj) ;
                        }
                    } else {
                        if(H0[l] != CMPLX(0,0)){
                            for(csr_index_t i=0; i<csr_nnz(submatrix); i++){
                                submatrix->Ax[i] +=
                                    H0[l]*(pgsum[l*4 + 0][i]        +
                                           pgsum[l*4 + 1][i]*ki     +
                                           pgsum[l*4 + 2][i]*kj     +
                                           pgsum[l*4 + 3][i]*ki*kj) ;
                            }
                        }

                        if(H1[l] != CMPLX(0,0)){
                            for(csr_index_t i=0; i<csr_nnz(submatrix); i++){
                                submatrix->Ax[i] +=
                                    H1[l]*(pgtsum[l*4 + 0][i]        +
                                           pgtsum[l*4 + 1][i]*kj     +
                                           pgtsum[l*4 + 2][i]*ki     +
                                           pgtsum[l*4 + 3][i]*ki*kj) ;
                            }
                        }
                    }
                }

                // store the submatrix in the global Hfull_blk
                if(Hfull_blk->Ax) csr_block_insert(Hfull_blk, row, col, submatrix->Ax);

                // store immediately in non-blocked Hfull matrix
                // csr_full_insert(Hfull, row, col, submatrix);
                for(csr_index_t i=0; i<csr_nnz(submatrix); i++){
                    csr_index_t dst = Hfull->Ai_sub_map[loc++];
                    Hfull->Ax[dst] = submatrix->Ax[i];
                }
            }
        }
        csr_free(submatrix);
    }

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    if(rank==0){
        toc();
    }
}

void compute_stationary_matrices(double h, double dt, sparse_csr_t *submatrix,
                                 sparse_csr_t *S_blk, sparse_csr_t *S,
                                 sparse_csr_t *Hst_blk, sparse_csr_t *Hst,
                                 const sparse_csr_t *h0, const sparse_csr_t *s0)
{
    csr_index_t row, col, colp;
    int blkdim = S_blk->blk_dim;
    complex ihdt = I*h*dt/2;

    tic(); PRINTF0("matrix assembly S ");
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
        // csr_block_insert(S_blk,row,row,submatrix->Ax);
        csr_full_insert(S, row, row, submatrix);

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
        // csr_block_insert(Hst_blk, row, row, submatrix->Ax);
        csr_full_insert(Hst, row, row, submatrix);
    }
    toc();

    // un-block, if above we used the blocked storage
    // csr_blocked_to_full(S, S_blk, submatrix);
    // csr_blocked_to_full(Hst, Hst_blk, submatrix);
}

slu_LU_t compute_preconditioner(const sparse_csr_t *S, const sparse_csr_t *Hst)
{
    csr_index_t row, col, colp;
    slu_matrix_t sluA;
    slu_LU_t sluLU;

    csr_data_t *P = (csr_data_t*)malloc(sizeof(csr_data_t)*Hst->nnz);

    tic(); PRINTF0("construct base preconditioner matrix ");

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

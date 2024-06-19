#ifndef _SOLVER_H
#define _SOLVER_H

#include <stdlib.h>
#include "types.h"
#include "solver_common.h"
#include "csr.h"
#include "superlu.h"
#include "../src/tictoc.h"

void beyondDipolePulse_init(beyondDipolePulse_t *this, double E0, double omega, double N);
void beoyndDipolePulse_axialPart(beyondDipolePulse_t *this, double t, cdouble_t *out);

void compute_timedep_matrices(double h, double dt, sparse_csr_t *submatrix, csr_data_t *ft, int lmax,
                              sparse_csr_t *Hfull_blk, sparse_csr_t *Hfull,
                              const sparse_csr_t *h0, const sparse_csr_t *H, const sparse_csr_t *g, const sparse_csr_t *gt);

void compute_stationary_matrices(double h, double dt, sparse_csr_t *submatrix,
                                 sparse_csr_t *S_blk, sparse_csr_t *S,
                                 sparse_csr_t *Hst_blk, sparse_csr_t *Hst,
                                 const sparse_csr_t *h0, const sparse_csr_t *s0);

slu_LU_t compute_preconditioner(const sparse_csr_t *S, const sparse_csr_t *Hst);

void matfree_spmv_S(csr_index_t row_l, csr_index_t row_u, const sparse_csr_t *S_blk, const csr_data_t *x, csr_data_t *result,
                    double h, double dt, const sparse_csr_t *s0, sparse_csr_t *submatrix);

void matfree_spmv_H(csr_index_t row_l, csr_index_t row_u, const sparse_csr_t *Hfull_blk, const csr_data_t *x, csr_data_t *result,
                    double h, double dt, sparse_csr_t *submatrix, csr_data_t *ft, int lmax,
                    const sparse_csr_t *h0, const sparse_csr_t *H, const sparse_csr_t *g, const sparse_csr_t *gt);

#endif /* _SOLVER_H */

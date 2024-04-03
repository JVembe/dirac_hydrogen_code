#ifndef _SOLVER_H
#define _SOLVER_H

#include <stdlib.h>
#include "types.h"
#include "csr.h"
#include "superlu.h"
#include "../src/tictoc.h"

#define SoL 137.035999084

int ik(int i);

void compute_timedep_matrices(double h, double dt, sparse_csr_t *submatrix, csr_data_t *ft, int lmax,
                              sparse_csr_t *Hfull_blk, sparse_csr_t *Hfull,
                              const sparse_csr_t *h0, const sparse_csr_t *H, const sparse_csr_t *g, const sparse_csr_t *gt);

void compute_stationary_matrices(double h, double dt, sparse_csr_t *submatrix,
                                 sparse_csr_t *S_blk, sparse_csr_t *S,
                                 sparse_csr_t *Hst_blk, sparse_csr_t *Hst,
                                 const sparse_csr_t *h0, const sparse_csr_t *s0);

slu_LU_t compute_preconditioner(const sparse_csr_t *S, const sparse_csr_t *Hst);


#endif /* _SOLVER_H */

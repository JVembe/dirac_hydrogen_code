#ifndef _INCLUDE_SUPERLU_H
#define _INCLUDE_SUPERLU_H

#include <slu_zdefs.h>

typedef struct {
    void *ptr;
} slu_matrix_t;

typedef struct {
    void *A;
    void *L;
    void *U;
    void *perm_c, *perm_r;
    void *R, *C;
    void *etree;
    void *mem_usage;
    void *stat;
    char equed;
} slu_LU_t;

slu_matrix_t slu_create_matrix(int nrows, int ncols, int nnz, void *Ax, int *Ai, int *Ap);
slu_LU_t slu_compute_ilu(slu_matrix_t A);
void slu_LU2coo(const SuperMatrix *L, const SuperMatrix *U,
                int_t **LAi_out, int_t **LAj_out, doublecomplex **LAx_out, int_t *Lnnz,
                int_t **UAi_out, int_t **UAj_out, doublecomplex **UAx_out, int_t *Unnz);
void slu_lu_solve(slu_LU_t *lu, doublecomplex *rhs, doublecomplex *x);

#endif /* _INCLUDE_SUPERLU_H */

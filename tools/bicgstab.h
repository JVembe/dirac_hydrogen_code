#ifndef _BICGSTAB_H
#define _BICGSTAB_H

#include "types.h"

typedef struct {
    void *r, *r0;
    void *v, *p;
    void *y, *z;
    void *s, *t;
} solver_workspace_t;

typedef void (*spmv_fun)(const void *mat, cdouble_t *x, cdouble_t *out);
typedef void (*precond_fun)(const void *precond, const cdouble_t *rhs, cdouble_t *x);

void bicgstab(spmv_fun spmv, const void *mat, const cdouble_t *rhs, cdouble_t *x, int nrow, int ncol, int local_col_beg,
              precond_fun psolve, const void *precond, solver_workspace_t *wsp, int *iters, double *tol_error);

#endif

#ifndef _BICGSTAB_H
#define _BICGSTAB_H

#include "csr.h"

#define ALLOC_VECTOR(n) (cdouble_t*)calloc(n, sizeof(cdouble_t));

typedef void (*spmv_fun)(const void *mat, const cdouble_t *x, cdouble_t *out);
typedef void (*precond_fun)(const void *precond, const cdouble_t *rhs, cdouble_t *x);

int bicgstab(spmv_fun spmv, const void *mat, const cdouble_t *rhs, cdouble_t *x, int n,             
             precond_fun psolve, const void *precond, int *iters, double *tol_error);

#endif

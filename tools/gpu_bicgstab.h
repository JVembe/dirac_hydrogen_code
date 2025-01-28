/*
	GaDE - Gpu-accelerated solver for the time dependent Dirac Equation
	
    Copyright (C) 2025  Johanne Elise Vembe <johannevembe@gmail.com>
    Copyright (C) 2025  Marcin Krotkiewski <marcink@uio.no>
	Copyright (C) 2025  Hicham Agueny <hicham.agueny@uib.no>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#ifndef _GPU_BICGSTAB_H
#define _GPU_BICGSTAB_H

#include "gpu_sparse.h"

extern gpublasHandle_t handle;

static inline gpu_complex_t gpuZdotc(int n, const gpu_complex_t *x, int incx, const gpu_complex_t *y, int incy)
{
    gpu_complex_t result = {0};
    CHECK_GPU_BLAS(_gpuZdotc(handle, n, x, incx, y, incy, &result));
    return result;
}

#define gpuZaxpy(...) CHECK_GPU_BLAS(_gpuZaxpy(handle, __VA_ARGS__));
#define gpuZcopy(...) CHECK_GPU_BLAS(_gpuZcopy(handle, __VA_ARGS__));
#define gpuZscal(...) CHECK_GPU_BLAS(_gpuZscal(handle, __VA_ARGS__));
#define gpuZaxpby(n, alpha, x, incx, beta, y, incy)	\
  {							\
    gpuZscal(n, &beta, y, incy);			\
    gpuZaxpy(n, &alpha, x, incx, y, incy);		\
  }


typedef struct {
    gpu_dense_vec_t r, r0;
    gpu_dense_vec_t v, p;
    gpu_dense_vec_t y, z;
    gpu_dense_vec_t s, t;
} gpu_solver_workspace_t;

typedef void (*gpu_spmv_fun)(const void *mat, gpu_dense_vec_t *x, gpu_dense_vec_t *out, csr_data_t alpha, csr_data_t beta);
typedef void (*gpu_precond_fun)(const void *precond, const gpu_dense_vec_t *rhs, gpu_dense_vec_t *x);


void gpu_blas_init();

void gpu_bicgstab(gpu_spmv_fun spmv, const void *mat, const gpu_dense_vec_t *rhs, gpu_dense_vec_t *x, 
                  int nrow, int ncol, int local_col_beg,
                  gpu_precond_fun psolve, const void *precond, gpu_solver_workspace_t *wsp, int *iters, double *tol_error);


#endif

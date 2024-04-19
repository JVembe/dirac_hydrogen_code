#ifndef _GPU_BICGSTAB_H
#define _GPU_BICGSTAB_H

#include "gpu_sparse.h"

#ifdef USE_CUDA
#include <cublas_v2.h>

#define CHECK_GPU_BLAS(func) {						\
        cublasStatus_t status = (func);					\
        if (status != CUBLAS_STATUS_SUCCESS) {                          \
            printf("cublas API failed at line %d error: %s\n", __LINE__, \
                   cublasGetStatusString(status));                      \
            exit(1);                                                    \
        }                                                               \
    }

typedef cuDoubleComplex gpu_complex_t;

#define gpublasHandle_t  cublasHandle_t
#define gpublasCreate    cublasCreate

#define gpuMakeComplex make_cuDoubleComplex
#define gpuCsub cuCsub
#define gpuCadd cuCadd
#define gpuCabs cuCabs
#define gpuCdiv cuCdiv
#define gpuCmul cuCmul
#define gpuCreal cuCreal
#define gpuCimag cuCimag

#define _gpuZdotc  cublasZdotc
#define _gpuZaxpy  cublasZaxpy
#define _gpuZscal  cublasZscal
#define _gpuZcopy  cublasZcopy

#endif

#ifdef USE_HIP
#define gpublasHandle_t  hipblasHandle_t
#define gpublasCreate    hipblasCreate

#define gpuMakeComplex make_hipDoubleComplex
#define gpuCsub hipCsub
#define gpuCadd hipCadd
#define gpuCabs hipCabs
#define gpuCdiv hipCdiv
#define gpuCmul hipCmul
#define gpuCreal hipCreal
#define gpuCimag hipCimag

#define _gpuZdotc  hipblasZdotc
#define _gpuZaxpy  hipblasZaxpy
#define _gpuZscal  hipblasZscal
#define _gpuZcopy  hipblasZcopy

#endif

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

#ifndef GPU_SPARSE_H
#define GPU_SPARSE_H

#include "csr.h"

#if defined USE_CUDA
#include <cuda_runtime_api.h>
#include <cusparse.h>

typedef cuDoubleComplex gpu_complex_t;

// Error types and constants
#define gpuError_t                                cudaError_t
#define gpuSuccess                                cudaSuccess

// Device management
#define gpuGetDeviceCount                         cudaGetDeviceCount
#define gpuSetDevice                              cudaSetDevice
#define gpuDeviceSynchronize                      cudaDeviceSynchronize

// Memory management
#define gpuMalloc                                 cudaMalloc
#define gpuFree                                   cudaFree
#define gpuMemcpy                                 cudaMemcpy
#define gpuMemset                                 cudaMemset
#define gpuMemcpyHostToDevice                     cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost                     cudaMemcpyDeviceToHost

// Error handling
#define gpuGetLastError                           cudaGetLastError
#define gpuGetErrorString                         cudaGetErrorString

// Stream management
#define gpuStream_t                               cudaStream_t
#define gpuStreamCreate                           cudaStreamCreate
#define gpuStreamDestroy                          cudaStreamDestroy
#define gpuStreamSynchronize                      cudaStreamSynchronize

// Sparse library handles and operations
#define gpusparseHandle_t                         cusparseHandle_t
#define gpusparseCreateHandle                     cusparseCreateHandle
#define gpusparseDestroy                          cusparseDestroy
#define gpusparseCreate                           cusparseCreate

// Sparse matrix-vector (SpMV) specific
#define gpusparseSpMatDescr_t                     cusparseSpMatDescr_t
#define gpusparseDnVecDescr_t                     cusparseDnVecDescr_t
#define gpusparseCreateCsr                        cusparseCreateCsr
#define gpusparseCreateDnVec                      cusparseCreateDnVec
#define gpusparseDestroyDnVec                     cusparseDestroyDnVec
#define gpusparseDestroySpMat                     cusparseDestroySpMat
#define gpusparseSpMV_bufferSize                  cusparseSpMV_bufferSize
#define gpusparseSpMV                             cusparseSpMV

// Sparse library constants
#define GPUSPARSE_INDEX_32I                       CUSPARSE_INDEX_32I
#define GPUSPARSE_INDEX_BASE_ZERO                 CUSPARSE_INDEX_BASE_ZERO
#define GPU_C_64F                                 CUDA_C_64F
#define GPUSPARSE_OPERATION_NON_TRANSPOSE         CUSPARSE_OPERATION_NON_TRANSPOSE
#define GPUSPARSE_SPMV_CSR_ALG1                   CUSPARSE_SPMV_CSR_ALG1

#define CHECK_GPU(func) {						\
        cudaError_t status = (func);					\
        if (status != cudaSuccess) {					\
            printf("CUDA API failed at line %d error: %s\n", __LINE__,	\
                   cudaGetErrorString(status));				\
            exit(1);                                                    \
        }                                                               \
    }

#define CHECK_GPUSPARSE(func) {						\
        cusparseStatus_t status = (func);                               \
        if (status != CUSPARSE_STATUS_SUCCESS) {                        \
            printf("CUSPARSE API failed at line %d error %d.\n", __LINE__, \
                   status);                                             \
            exit(1);                                                    \
        }                                                               \
    }

#elif defined(USE_HIP)

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipsparse.h>
#include <hip/hip_complex.h>

// Error types and constants
#define gpuError_t                                hipError_t
#define gpuSuccess                                hipSuccess

// Device management
#define gpuGetDeviceCount                         hipGetDeviceCount
#define gpuSetDevice                              hipSetDevice
#define gpuDeviceSynchronize                      hipDeviceSynchronize

// Memory management
#define gpuMalloc                                 hipMalloc
#define gpuFree                                   hipFree
#define gpuMemcpy                                 hipMemcpy
#define gpuMemset                                 hipMemset
#define gpuMemcpyHostToDevice                     hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost                     hipMemcpyDeviceToHost

// Error handling
#define gpuGetLastError                           hipGetLastError
#define gpuGetErrorString                         hipGetErrorString

// Stream management
#define gpuStream_t                               hipStream_t
#define gpuStreamCreate                           hipStreamCreate
#define gpuStreamDestroy                          hipStreamDestroy
#define gpuStreamSynchronize                      hipStreamSynchronize

// Sparse library handles and operations
#define gpusparseHandle_t                         hipsparseHandle_t
#define gpusparseCreateHandle                     hipsparseCreateHandle
#define gpusparseDestroy                          hipsparseDestroy
#define gpusparseCreate                           hipsparseCreate

// Sparse matrix-vector (SpMV) specific
#define gpusparseSpMatDescr_t                     hipsparseSpMatDescr_t
#define gpusparseDnVecDescr_t                     hipsparseDnVecDescr_t
#define gpusparseCreateCsr                        hipsparseCreateCsr
#define gpusparseCreateDnVec                      hipsparseCreateDnVec
#define gpusparseSpMV_bufferSize                  hipsparseSpMV_bufferSize
#define gpusparseSpMV                             hipsparseSpMV

// Sparse library constants
#define GPUSPARSE_INDEX_32I                       HIPSPARSE_INDEX_32I
#define GPUSPARSE_INDEX_BASE_ZERO                 HIPSPARSE_INDEX_BASE_ZERO
#define GPU_C_64F                                 HIP_C_64F
#define GPUSPARSE_OPERATION_NON_TRANSPOSE         HIPSPARSE_OPERATION_NON_TRANSPOSE
#define GPUSPARSE_SPMV_CSR_ALG1                   HIPSPARSE_SPMV_CSR_ALG1

#define CHECK_GPU(func) {						\
        hipError_t status = (func);					\
        if (status != hipSuccess) {					\
            printf("HIP API failed at line %d error: %s\n", __LINE__,	\
                   hipGetErrorString(status));				\
            exit(1);                                                    \
        }                                                               \
    }

#define CHECK_GPUSPARSE(func) {						\
        hipsparseStatus_t status = (func);                              \
        if (status != HIPSPARSE_STATUS_SUCCESS) {                       \
            printf("HIPSPARSE API failed at line %d error %d.\n", __LINE__, \
                   status);                                             \
            exit(1);                                                    \
        }                                                               \
    }

#endif

#if defined USE_CUDA | defined USE_HIP

typedef struct {
    gpusparseSpMatDescr_t desc;
    csr_index_t nrows, ncols, nnz;    
    csr_index_t *Ap;
    csr_index_t *Ai;
    csr_data_t  *Ax;
    cusparseSpSVDescr_t spsvDescr;
    void *cuBuffer;
} gpu_sparse_csr_t;

typedef struct {
    gpusparseDnVecDescr_t desc;
    gpusparseDnVecDescr_t desc_local;
    csr_index_t dim, local_dim;
    csr_index_t local_offset;
    gpu_complex_t *x;
} gpu_dense_vec_t;

typedef struct {
    gpu_sparse_csr_t *L, *U;
    gpu_dense_vec_t *temp;
} gpu_lu_t;

void gpu_sparse_init();
void gpu_sparse_fini();

void gpu_put_csr(gpu_sparse_csr_t *Agpu, const sparse_csr_t *Ahost);
void gpu_free_csr(gpu_sparse_csr_t *Agpu);

void gpu_put_vec(gpu_dense_vec_t *xgpu, const csr_data_t *xhost, csr_index_t dim);
void gpu_vec_local_part(gpu_dense_vec_t *xgpu, csr_index_t dim, csr_index_t local_offset);
void gpu_get_vec(csr_data_t *xhost, const gpu_dense_vec_t *xgpu);
void gpu_free_vec(gpu_dense_vec_t *xgpu);

void gpu_spmv(gpu_sparse_csr_t *Hfull, gpu_dense_vec_t *x, gpu_dense_vec_t *y, csr_data_t alpha, csr_data_t beta);
void gpu_spmv_local(gpu_sparse_csr_t *Hfull, gpu_dense_vec_t *x, gpu_dense_vec_t *y, csr_data_t alpha, csr_data_t beta);
void gpu_spmv_block(sparse_csr_t H_blk, csr_data_t *x, csr_data_t *y, sparse_csr_t *g);
void gpu_spmb_block_test(sparse_csr_t H_blk, csr_data_t *x, csr_data_t *yfull, sparse_csr_t *g);
void gpu_spmv_test(sparse_csr_t Hfull, csr_data_t *x, csr_data_t *yfull);

void gpu_lu_analyze(gpu_sparse_csr_t *L, gpu_sparse_csr_t *U, gpu_dense_vec_t *x, gpu_dense_vec_t *y, gpu_dense_vec_t *temp);
void gpu_lu_solve(const gpu_sparse_csr_t *L, const gpu_sparse_csr_t *U, const gpu_dense_vec_t *x, gpu_dense_vec_t *y, gpu_dense_vec_t *temp);

#endif

#endif /* GPU_SPARSE_H */

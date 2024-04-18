#ifndef _CUDA_TRANSLATE_H
#define _CUDA_TRANSLATE_H

#include <cuda_runtime_api.h>
#include <cusparse.h>

typedef cuDoubleComplex gpu_complex_t;

// Error types and constants
#define gpuError_t                                cudaError_t
#define gpuSuccess                                cudaSuccess

// Device management
#define gpuGetDeviceCount                         cudaGetDeviceCount
#define gpuGetDeviceProperties                    cudaGetDeviceProperties
#define gpuGetDevice                              cudaGetDevice
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

#endif /* _CUDA_TRANSLATE_H */

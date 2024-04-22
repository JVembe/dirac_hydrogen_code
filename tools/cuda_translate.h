#ifndef _CUDA_TRANSLATE_H
#define _CUDA_TRANSLATE_H

#if defined USE_CUDA | defined __CUDACC__

#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <cublas_v2.h>

typedef cuDoubleComplex gpu_complex_t;

// Error types and constants
#define gpuError_t                                cudaError_t
#define gpuSuccess                                cudaSuccess
#define gpuDeviceProp                             struct cudaDeviceProp

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
#define gpuMemcpyToSymbol                         cudaMemcpyToSymbol
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
#define gpusparseSpSVDescr_t                      cusparseSpSVDescr_t
#define gpusparseSpSV_createDescr                 cusparseSpSV_createDescr
#define gpusparseCreateCsr                        cusparseCreateCsr
#define gpusparseCreateDnVec                      cusparseCreateDnVec
#define gpusparseDestroyDnVec                     cusparseDestroyDnVec
#define gpusparseDestroySpMat                     cusparseDestroySpMat
#define gpusparseSpMV_bufferSize                  cusparseSpMV_bufferSize
#define gpusparseSpMV                             cusparseSpMV
#define gpusparseSpMatSetAttribute                cusparseSpMatSetAttribute
#define gpusparseSpSV_bufferSize                  cusparseSpSV_bufferSize
#define gpusparseSpSV_analysis                    cusparseSpSV_analysis
#define gpusparseSpSV_solve                       cusparseSpSV_solve

// Sparse library constants
#define GPUSPARSE_INDEX_32I                       CUSPARSE_INDEX_32I
#define GPUSPARSE_INDEX_BASE_ZERO                 CUSPARSE_INDEX_BASE_ZERO
#define GPU_C_64F                                 CUDA_C_64F
#define GPUSPARSE_OPERATION_NON_TRANSPOSE         CUSPARSE_OPERATION_NON_TRANSPOSE
#define GPUSPARSE_SPMV_CSR_ALG1                   CUSPARSE_SPMV_CSR_ALG1
#define GPUPARSE_FILL_MODE_LOWER                  CUSPARSE_FILL_MODE_LOWER
#define GPUSPARSE_SPMAT_FILL_MODE                 CUSPARSE_SPMAT_FILL_MODE
#define GPUSPARSE_SPMAT_DIAG_TYPE                 CUSPARSE_SPMAT_DIAG_TYPE
#define GPUSPARSE_DIAG_TYPE_NON_UNIT              CUSPARSE_DIAG_TYPE_NON_UNIT
#define GPUSPARSE_FILL_MODE_LOWER                 CUSPARSE_FILL_MODE_LOWER
#define GPUSPARSE_FILL_MODE_UPPER                 CUSPARSE_FILL_MODE_UPPER
#define GPUSPARSE_SPSV_ALG_DEFAULT                CUSPARSE_SPSV_ALG_DEFAULT

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

#endif /* _CUDA_TRANSLATE_H */

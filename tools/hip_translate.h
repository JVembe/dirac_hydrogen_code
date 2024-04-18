#ifndef _HIP_TRANSLATE_H
#define _HIP_TRANSLATE_H

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

#endif /* _HIP_TRANSLATE_H *

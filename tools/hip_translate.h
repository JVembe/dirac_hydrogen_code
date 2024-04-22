#ifndef _HIP_TRANSLATE_H
#define _HIP_TRANSLATE_H

#if defined USE_HIP || defined __HIPCC__

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipsparse/hipsparse.h>
#include <hip/hip_complex.h>
#include <hipblas/hipblas.h>

// Error types and constants
#define gpuError_t                                hipError_t
#define gpuSuccess                                hipSuccess
#define gpuDeviceProp                             hipDeviceProp_t

// Device management
#define gpuGetDeviceCount                         hipGetDeviceCount
#define gpuGetDeviceProperties                    hipGetDeviceProperties
#define gpuGetDevice                              hipGetDevice
#define gpuSetDevice                              hipSetDevice
#define gpuDeviceSynchronize                      hipDeviceSynchronize

// Memory management
#define gpuMalloc                                 hipMalloc
#define gpuFree                                   hipFree
#define gpuMemcpy                                 hipMemcpy
#define gpuMemcpyToSymbol                         hipMemcpyToSymbol
#define gpuMemset                                 hipMemset
#define gpuMemcpyHostToDevice                     hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost                     hipMemcpyDeviceToHost
#define gpuDeviceSynchronize                      hipDeviceSynchronize

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
#define gpusparseSpSVDescr_t                      hipsparseSpSVDescr_t
#define gpusparseSpSV_createDescr                 hipsparseSpSV_createDescr
#define gpusparseCreateCsr                        hipsparseCreateCsr
#define gpusparseCreateDnVec                      hipsparseCreateDnVec
#define gpusparseDestroyDnVec                     hipsparseDestroyDnVec
#define gpusparseDestroySpMat                     hipsparseDestroySpMat
#define gpusparseSpMV_bufferSize                  hipsparseSpMV_bufferSize
#define gpusparseSpMV                             hipsparseSpMV
#define gpusparseSpMatSetAttribute                hipsparseSpMatSetAttribute
#define gpusparseSpSV_bufferSize                  hipsparseSpSV_bufferSize
#define gpusparseSpSV_analysis                    hipsparseSpSV_analysis
#define gpusparseSpSV_solve                       hipsparseSpSV_solve

// Sparse library constants
#define GPUSPARSE_INDEX_32I                       HIPSPARSE_INDEX_32I
#define GPUSPARSE_INDEX_BASE_ZERO                 HIPSPARSE_INDEX_BASE_ZERO
#define GPU_C_64F                                 HIP_C_64F
#define GPUSPARSE_OPERATION_NON_TRANSPOSE         HIPSPARSE_OPERATION_NON_TRANSPOSE
#define GPUSPARSE_SPMV_CSR_ALG1                   HIPSPARSE_SPMV_CSR_ALG1
#define GPUPARSE_FILL_MODE_LOWER                  HIPSPARSE_FILL_MODE_LOWER
#define GPUSPARSE_SPMAT_FILL_MODE                 HIPSPARSE_SPMAT_FILL_MODE
#define GPUSPARSE_SPMAT_DIAG_TYPE                 HIPSPARSE_SPMAT_DIAG_TYPE
#define GPUSPARSE_DIAG_TYPE_NON_UNIT              HIPSPARSE_DIAG_TYPE_NON_UNIT
#define GPUSPARSE_FILL_MODE_LOWER                 HIPSPARSE_FILL_MODE_LOWER
#define GPUSPARSE_FILL_MODE_UPPER                 HIPSPARSE_FILL_MODE_UPPER
#define GPUSPARSE_SPSV_ALG_DEFAULT                HIPSPARSE_SPSV_ALG_DEFAULT

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

#define CHECK_GPU_BLAS(func) {						\
        hipblasStatus_t status = (func);				\
        if (status != HIPBLAS_STATUS_SUCCESS) {				\
            printf("hipblas API failed at line %d error: %s\n", __LINE__, \
                   hipblasStatusToString(status));			\
            exit(1);                                                    \
        }                                                               \
    }

typedef hipDoubleComplex gpu_complex_t;
#define make_gpu_complex_t                        make_hipDoubleComplex
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

#endif /* _HIP_TRANSLATE_H */

#ifndef _TYPES_H
#define _TYPES_H

#include <stdint.h>
#include <strings.h>

#if defined __CUDACC__ | defined __HIPCC__
typedef gpu_complex_t cdouble_t;
typedef gpu_complex_t csr_data_t;
#else
#include <complex.h>
typedef double complex cdouble_t;
typedef double complex csr_data_t;
#endif

typedef unsigned int csr_index_t;
typedef csr_index_t dimType;
typedef csr_index_t indexType;

#ifdef USE_MPI
#include <mpi.h>
#define MPI_CSR_INDEX_T MPI_INT
// #define MPI_CSR_DATA_T  MPI_DOUBLE
#endif

// Define the CMPLX macro manually if it's not already defined
#ifndef CMPLX
#define CMPLX(x, y) ((double complex)((double)(x) + _Complex_I * (double)(y)))
#endif

#endif

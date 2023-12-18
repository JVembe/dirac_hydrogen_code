#ifndef _TYPES_H
#define _TYPES_H

#include <complex.h>
#include <stdint.h>
#include <strings.h>

typedef int csr_index_t;
typedef double complex csr_data_t;

typedef csr_index_t dimType;
typedef csr_index_t indexType;

#ifdef USE_MPI
#include <mpi.h>
#define MPI_CSR_INDEX_T MPI_INT
// #define MPI_CSR_DATA_T  MPI_DOUBLE
#endif

#endif

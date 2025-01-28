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

typedef int csr_index_t;
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

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

#ifndef _UTILS_H
#define _UTILS_H

#include <stdio.h>
#include <stdlib.h>

#include "types.h"

#define PRINTF0(...) if(rank==0) printf(__VA_ARGS__)

#ifdef DEBUG
#define PDEBUG(...) printf(__VA_ARGS__)
#else
#define PDEBUG(...)
#endif

#define HERE fprintf(stderr, "HERE %s:%d\n", __FUNCTION__, __LINE__);
#define MIN(a, b) ((a)<(b)?(a):(b))
#define MAX(a, b) ((a)>(b)?(a):(b))

#define ERROR(...)                              \
    {                                           \
        fprintf(stderr,  __VA_ARGS__);          \
        exit(1);                                \
    }

#ifdef USE_MPI
#define CHECK_MPI(func) {                                       \
        int status = (func);                                    \
        if (status != 0) {                                      \
            char string[256];                                   \
            int resultlen;                                      \
            MPI_Error_string(status, string, &resultlen);       \
            printf("MPI failed at line %s:%d error: %s\n",      \
                   __FUNCTION__, __LINE__, string);             \
            exit(1);                                            \
        }                                                       \
    }
#endif

void compare_vectors(csr_data_t *v1, csr_data_t *v2, csr_index_t dim);

#endif

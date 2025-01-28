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

#ifndef _INCLUDE_SUPERLU_H
#define _INCLUDE_SUPERLU_H

#include <slu_zdefs.h>

typedef struct {
    void *ptr;
} slu_matrix_t;

typedef struct {
    void *A;
    void *L;
    void *U;
    void *perm_c, *perm_r;
    void *R, *C;
    void *etree;
    void *mem_usage;
    void *stat;
    char equed;
} slu_LU_t;

slu_matrix_t slu_create_matrix(int nrows, int ncols, int nnz, void *Ax, int *Ai, int *Ap);
slu_LU_t slu_compute_ilu(slu_matrix_t A);
void slu_LU2coo(const SuperMatrix *L, const SuperMatrix *U,
                int_t **LAi_out, int_t **LAj_out, doublecomplex **LAx_out, int_t *Lnnz,
                int_t **UAi_out, int_t **UAj_out, doublecomplex **UAx_out, int_t *Unnz);
void slu_lu_solve(slu_LU_t *lu, doublecomplex *rhs, doublecomplex *x);

#endif /* _INCLUDE_SUPERLU_H */

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

#ifndef _BICGSTAB_H
#define _BICGSTAB_H

#include "types.h"

typedef struct {
    void *r, *r0;
    void *v, *p;
    void *y, *z;
    void *s, *t;
} solver_workspace_t;

typedef void (*spmv_fun)(const void *mat, cdouble_t *x, cdouble_t *out);
typedef void (*precond_fun)(const void *precond, const cdouble_t *rhs, cdouble_t *x);

void bicgstab(spmv_fun spmv, const void *mat, const cdouble_t *rhs, cdouble_t *x, int nrow, int ncol, int local_col_beg,
              precond_fun psolve, const void *precond, solver_workspace_t *wsp, int *iters, double *tol_error);

#endif

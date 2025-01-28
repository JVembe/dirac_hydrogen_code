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

#ifndef SORTED_LIST_H
#define SORTED_LIST_H

#include "types.h"

#define ALLOC_BLOCKSIZE 32

void sorted_list_create(dimType **list, dimType *size);
dimType sorted_list_locate(dimType *list, dimType nelems, dimType value);
void sorted_list_add(dimType **list, dimType *nelems, dimType *lsize, dimType value);
void sorted_list_add_static(dimType *list, dimType *nelems, dimType value);

#endif

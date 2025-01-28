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

#include "sorted_list.h"

#include <stdlib.h>

void sorted_list_create(dimType **list, dimType *size)
{
  if(*size==0) *size = ALLOC_BLOCKSIZE;
  (*list) = (dimType*)malloc(sizeof(dimType)*(*size));
}

dimType sorted_list_locate(dimType *list, dimType nelems, dimType value)
{
    dimType l, u, m;

    l = 0;
    u = nelems;

    /* locate the range by bisection */
    /* the search is slower for short lists */
    while(u-l>32){
        m = (l+u)/2;
        if(list[m]>value){
            u=m;
        } else {
            l=m;
        }
    }

    /* locate the value by linear search */
    while(l<u){
        if(list[l]>=value) break;
        l++;
    }

    return l;
}

void sorted_list_add(dimType **list, dimType *nelems, dimType *lsize, dimType value)
{
    dimType l, u;

    l = sorted_list_locate(*list, *nelems, value);
    if(l<*nelems && (*list)[l]==value) return;

    /* check if we have enough of memory in the list */
    if(*nelems==*lsize){
        /* (*lsize) += ALLOC_BLOCKSIZE; */
        (*lsize) *= 2;
        (*list) = (dimType*)realloc((*list), sizeof(dimType)*(*lsize));
    }

    /* insert into array */
    u = *nelems;
    while(l!=u){
        (*list)[u] = (*list)[u-1];
        u--;
    }

    (*list)[l] = value;
    (*nelems)++;
}

void sorted_list_add_static(dimType *list, dimType *nelems, dimType value)
{
    dimType l, u;

    l = sorted_list_locate(list, *nelems, value);
    if(l<*nelems && list[l]==value) return;

    /* insert into array */
    u = *nelems;
    while(l!=u){
        list[u] = list[u-1];
        u--;
    }

    list[l] = value;
    (*nelems)++;
}

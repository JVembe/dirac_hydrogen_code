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

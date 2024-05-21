#ifndef SORTED_LIST_H
#define SORTED_LIST_H

#include "types.h"

#define ALLOC_BLOCKSIZE 32

void sorted_list_create(dimType **list, dimType *size);
dimType sorted_list_locate(dimType *list, dimType nelems, dimType value);
void sorted_list_add(dimType **list, dimType *nelems, dimType *lsize, dimType value);
void sorted_list_add_static(dimType *list, dimType *nelems, dimType value);

#endif

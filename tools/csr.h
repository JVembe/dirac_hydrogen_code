#ifndef _CSR_H
#define _CSR_H

#include <stdint.h>
#include <complex.h>
#include <strings.h>

#define ERROR(...)                                                    \
    {                                                                 \
        fprintf(stderr,  __VA_ARGS__);                                \
        exit(1);                                                      \
    }

typedef int csr_index_t;
typedef double complex csr_data_t;

typedef struct {
    csr_index_t dim, nnz;
    csr_index_t *map;
    csr_index_t *Ap;
    csr_index_t *Ai;
    csr_data_t  *Ax;
} sparse_csr_t;


void csr_copy(sparse_csr_t *out, const sparse_csr_t *in)
{
    out->dim = in->dim;
    out->nnz = in->nnz;

    out->Ap = (csr_index_t*)malloc(sizeof(csr_index_t)*(out->dim+1));
    memcpy(out->Ap, in->Ap, sizeof(csr_index_t)*(out->dim+1));
    out->Ai = (csr_index_t*)malloc(sizeof(csr_index_t)*out->nnz);
    memcpy(out->Ai, in->Ai, sizeof(csr_index_t)*out->nnz);
    out->Ax = (csr_data_t*)malloc(sizeof(csr_data_t)*out->nnz);
    memcpy(out->Ax, in->Ax, sizeof(csr_data_t)*out->nnz);

    if(in->map){
        out->map = (csr_index_t*)malloc(sizeof(csr_index_t)*out->dim);
        memcpy(out->map, in->map, sizeof(csr_index_t)*out->dim);
    }
}

void csr_zero(sparse_csr_t *sp)
{
    bzero(sp->Ax, sizeof(csr_data_t)*sp->nnz);
}

void csr_read(const char *fname, sparse_csr_t *sp)
{
    size_t nread;

    FILE *fd = fopen(fname, "r");
    if(!fd) ERROR("cant open %s\n", fname);

    // storage format: dim, nnz, Ap, Ai, Ax
    fread(&sp->dim, sizeof(int), 1, fd);
    fread(&sp->nnz, sizeof(int), 1, fd);

    sp->Ap = (csr_index_t*)malloc(sizeof(csr_index_t)*(sp->dim+1));
    nread = fread(sp->Ap, sizeof(csr_index_t), (sp->dim+1), fd);
    if(nread!=(sp->dim+1)) ERROR("wrong file format in %s\n", fname);
    if(sp->Ap[sp->dim] != sp->nnz) ERROR("wrong file format (nnz) in %s: dim %d nnz %d Ap %d\n",
                                           fname, sp->dim, sp->nnz, sp->Ap[sp->dim]);

    sp->Ai = (csr_index_t*)malloc(sizeof(csr_index_t)*sp->nnz);
    nread = fread(sp->Ai, sizeof(csr_index_t), sp->nnz, fd);
    if(nread!=sp->nnz) ERROR("wrong file format in %s\n", fname);

    sp->Ax = (csr_data_t*)malloc(sizeof(csr_data_t)*sp->nnz);
    nread = fread(sp->Ax, sizeof(csr_data_t), sp->nnz, fd);
    if(nread!=sp->nnz) ERROR("wrong file format in %s\n", fname);

    // check if we have node number maps
    sp->map = (csr_index_t*)malloc(sizeof(csr_index_t)*sp->dim);
    nread = fread(sp->map, sizeof(csr_index_t), sp->dim, fd);
    if(nread!=sp->dim){
        free(sp->map);
        sp->map = NULL;
    }

    fclose(fd);
}

#endif /* _CSR_H */

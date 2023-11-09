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
    csr_index_t blk_dim, blk_nnz;
    csr_index_t *map;
    csr_index_t *Ap;
    csr_index_t *Ai;
    csr_data_t  *Ax;
    int is_link;
} sparse_csr_t;

void csr_print(const sparse_csr_t *sp)
{
    for(csr_index_t row = 0; row < sp->dim; row++){
        for(csr_index_t cp = sp->Ap[row]; cp < sp->Ap[row+1]; cp++){
            printf("%d %d %lf + %lfi\n", row, sp->Ai[cp], creal(sp->Ax[cp]), cimag(sp->Ax[cp]));
        }
    }
}

void csr_free(sparse_csr_t *sp)
{
    free(sp->map);
    sp->map = NULL;
    free(sp->Ap);
    sp->Ap  = NULL;
    free(sp->Ai);
    sp->Ai  = NULL;
    if(!sp->is_link) free(sp->Ax);
    sp->Ax = NULL;
    sp->dim = 0;
    sp->nnz = 0;
    sp->blk_nnz = 1;
    sp->blk_dim = 1;
    sp->is_link = 0;
}

void csr_copy(sparse_csr_t *out, const sparse_csr_t *in)
{
    out->dim = in->dim;
    out->nnz = in->nnz;
    out->blk_nnz = in->blk_nnz;
    out->blk_dim = in->blk_dim;
    out->is_link = 0;
    out->map = NULL;

    out->Ap = (csr_index_t*)malloc(sizeof(csr_index_t)*(out->dim+1));
    memcpy(out->Ap, in->Ap, sizeof(csr_index_t)*(out->dim+1));
    out->Ai = (csr_index_t*)malloc(sizeof(csr_index_t)*out->nnz);
    memcpy(out->Ai, in->Ai, sizeof(csr_index_t)*out->nnz);
    out->Ax = (csr_data_t*)malloc(sizeof(csr_data_t)*out->nnz*out->blk_nnz);
    memcpy(out->Ax, in->Ax, sizeof(csr_data_t)*out->nnz*out->blk_nnz);

    if(in->map){
        out->map = (csr_index_t*)malloc(sizeof(csr_index_t)*out->dim);
        memcpy(out->map, in->map, sizeof(csr_index_t)*out->dim);
    }
}

void csr_block_update(sparse_csr_t *sp, csr_index_t blk_dim, csr_index_t blk_nnz)
{
    sp->blk_dim = blk_dim;
    sp->blk_nnz = blk_nnz;

    // update Ax storage
    free(sp->Ax);
    sp->Ax = (csr_data_t*)malloc(sizeof(csr_data_t)*sp->nnz*sp->blk_nnz);
    bzero(sp->Ax, sizeof(csr_data_t)*sp->nnz*sp->blk_nnz);

    // do NOT update Ap pointers - need them for Ai
}

void csr_block_insert(sparse_csr_t *sp, csr_index_t row, csr_index_t col, csr_data_t *blk_ptr)
{
    csr_index_t cp;
    for(cp = sp->Ap[row]; cp < sp->Ap[row+1]; cp++){
        if(sp->Ai[cp]>=col) break;
    }
    if(sp->Ai[cp]!=col) ERROR("cant insert block: (%d,%d) not present in CSR.", row, col);
    memcpy(sp->Ax + cp*sp->blk_nnz, blk_ptr, sizeof(csr_data_t)*sp->blk_nnz);
}

void csr_zero(sparse_csr_t *sp)
{
    bzero(sp->Ax, sizeof(csr_data_t)*sp->nnz*sp->blk_nnz);
}

void csr_read(const char *fname, sparse_csr_t *sp)
{
    size_t nread;

    FILE *fd = fopen(fname, "r");
    if(!fd) ERROR("cant open %s\n", fname);

    // non-blocked by default
    sp->blk_dim = 1;
    sp->blk_nnz = 1;
    
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

csr_data_t csr_get_value(const sparse_csr_t *sp, csr_index_t row, csr_index_t col)
{
    csr_index_t cp;

    // NOTE: this only works for non-blocked matrices
    
    // locate column in the given row
    // could be improved with bisection for large nubmer of entries per row
    for(cp = sp->Ap[row]; cp < sp->Ap[row+1]; cp++){
        if(sp->Ai[cp]>=col) break;
    }
    if(sp->Ai[cp]!=col) return CMPLX(0,0);
    return sp->Ax[cp];
}

void csr_set_value(const sparse_csr_t *sp, csr_index_t row, csr_index_t col, csr_data_t val)
{
    csr_index_t cp;

    // NOTE: this only works for non-blocked matrices
    
    // locate column in the given row
    // could be improved with bisection for large nubmer of entries per row
    for(cp = sp->Ap[row]; cp < sp->Ap[row+1]; cp++){
        if(sp->Ai[cp]>=col) break;
    }
    if(sp->Ai[cp]!=col) ERROR("cant set matrix value: (%d,%d) not present in CSR.", row, col);
    sp->Ax[cp] = val;
}

void csr_conj_transpose(sparse_csr_t *out, const sparse_csr_t *in)
{
    // NOTE: this only works for non-blocked matrices

    for(csr_index_t row = 0; row < in->dim; row++){
        for(csr_index_t cp = in->Ap[row]; cp < in->Ap[row+1]; cp++){
            csr_set_value(out, in->Ai[cp], row, conj(in->Ax[cp]));
        }
    }
}

#endif /* _CSR_H */

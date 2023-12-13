#include "csr.h"

#ifdef USE_PREFETCHING
#include <xmmintrin.h>
#endif

void csr_print(const sparse_csr_t *sp)
{
    for(csr_index_t row = 0; row < sp->dim; row++){
        for(csr_index_t cp = sp->Ap[row]; cp < sp->Ap[row+1]; cp++){
            printf("%d %d %lf + %lfi\n", row, sp->Ai[cp], creal(sp->Ax[cp]), cimag(sp->Ax[cp]));
        }
    }
}

void csr_allocate(sparse_csr_t *out, csr_index_t dim, csr_index_t nnz)
{
    out->dim = dim;
    out->nnz = nnz;
    out->blk_nnz = 1;
    out->blk_dim = 1;
    out->is_link = 0;
    out->map = NULL;
    out->Ap = (csr_index_t*)calloc((out->dim+1), sizeof(csr_index_t));
    out->Ai = (csr_index_t*)calloc(out->nnz, sizeof(csr_index_t));
    out->Ax = (csr_data_t*)calloc(out->nnz*out->blk_nnz, sizeof(csr_data_t));
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

void csr_block_link(sparse_csr_t *sp_blk, sparse_csr_t *sp, csr_index_t row, csr_index_t col)
{
    csr_index_t cp;
    for(cp = sp->Ap[row]; cp < sp->Ap[row+1]; cp++){
        if(sp->Ai[cp]>=col) break;
    }
    if(sp->Ai[cp]!=col) ERROR("cant insert block: (%d,%d) not present in CSR.", row, col);
    sp_blk->is_link = 1;
    sp_blk->Ax = sp->Ax + cp*sp->blk_nnz;
    sp_blk->dim = sp->blk_dim;
    sp_blk->nnz = sp->blk_nnz;
}

csr_index_t csr_dim(sparse_csr_t *sp_blk)
{
    return sp_blk->dim*sp_blk->blk_dim;
}

csr_index_t csr_nnz(sparse_csr_t *sp_blk)
{
    return sp_blk->nnz*sp_blk->blk_nnz;
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
    nread = fread(&sp->dim, sizeof(csr_index_t), 1, fd);
    nread = fread(&sp->nnz, sizeof(csr_index_t), 1, fd);

    // assume 1 partition
    sp->npart = 1;
    sp->row_beg = 0;
    sp->row_end = sp->dim;
    sp->local_dim = sp->dim;

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
    } else {

        // we need the partitioning
        nread = fread(&sp->npart, sizeof(csr_index_t), 1, fd);
        if(nread!=1) ERROR("wrong file format in %s: partitioning info not found\n", fname);

        sp->row_cpu_dist = (csr_index_t*)malloc(sizeof(csr_index_t)*(sp->npart+1));
        nread = fread(sp->row_cpu_dist, sizeof(csr_index_t), sp->npart+1, fd);
        if(nread!=(sp->npart+1)) ERROR("wrong file format in %s: partitioning info inconsistent\n", fname);

        // matrix requires repartitioning
        sp->row_beg = -1;
        sp->row_end = -1;
    }

    fclose(fd);
}

void csr_write(const char *fname, const sparse_csr_t *sp)
{
    size_t nwrite;

    FILE *fd = fopen(fname, "w+");
    if(!fd) ERROR("cant open %s\n", fname);

    // storage format: dim, nnz, Ap, Ai, Ax
    nwrite = fwrite(&sp->local_dim, sizeof(csr_index_t), 1, fd);
    nwrite = fwrite(&sp->nnz, sizeof(csr_index_t), 1, fd);

    nwrite = fwrite(sp->Ap, sizeof(csr_index_t), (sp->local_dim+1), fd);
    if(nwrite!=(sp->local_dim+1)) ERROR("cant write file %s\n", fname);

    nwrite = fwrite(sp->Ai, sizeof(csr_index_t), sp->nnz, fd);
    if(nwrite!=sp->nnz) ERROR("cant write file %s\n", fname);

    nwrite = fwrite(sp->Ax, sizeof(csr_data_t), sp->nnz, fd);
    if(nwrite!=sp->nnz) ERROR("cant write file %s\n", fname);

    fclose(fd);
}

/* walk through Ai and save communication (non-local) columns */
void csr_analyze_communication(sparse_csr_t *sp, int rank, int nranks)
{

    /* allocate communication lists */
    for(int irank=0; irank<nranks; irank++){
        if(irank == rank) continue;
        sorted_list_create(sp->comm_pattern + rank*nranks+irank,
                           sp->comm_pattern_size + rank*nranks+irank);
    }

    csr_index_t row_l = sp->row_cpu_dist[rank];
    csr_index_t row_end = sp->row_cpu_dist[rank+1];
    csr_index_t *row_cpu_dist   = sp->row_cpu_dist;
    csr_index_t **comm_pattern  = sp->comm_pattern + rank*nranks;
    csr_index_t *comm_pattern_size  = sp->comm_pattern_size + rank*nranks;
    csr_index_t *n_comm_entries = sp->n_comm_entries + rank*nranks;

    csr_index_t  col;
    csr_index_t  mincol = row_l, maxcol = row_end-1;
    csr_index_t *Ap_local = sp->Ap;
    csr_index_t *Ai_local = sp->Ai;

    /* go through all non-zero entries, find non-local column access */
    for(csr_index_t j=0; j<Ap_local[row_end-row_l]; j++){

        col    = Ai_local[j];

        /* all local columns are assumed to be non-empty */
        if(col>=row_l && col<row_end) continue;

        mincol = MIN(col, mincol);
        maxcol = MAX(col, maxcol);

        /* add to target processor */
        for(int irank=0; irank < nranks; irank++){
            if(col >= row_cpu_dist[irank] && col < row_cpu_dist[irank+1]){

                /* add communication entry */
                sorted_list_add(comm_pattern+irank,
                                n_comm_entries+irank,
                                comm_pattern_size+irank,
                                col);
                break;
            }
        }
    }
}

void csr_remove_empty_columns(sparse_csr_t *sp, int rank, int nranks)
{
    csr_index_t n_lower = 0, n_upper = 0;
    csr_index_t col, newcol;
    csr_index_t iter = 0;
    csr_index_t *Ap = sp->Ap;
    csr_index_t *Ai = sp->Ai;
    csr_index_t row_beg = sp->row_beg;
    csr_index_t row_end = sp->row_end;
    int irank, irank_end;

    /* count communication entries in the lower triangular part */
    for(irank=0; irank<rank; irank++)
        n_lower += sp->n_comm_entries[rank*nranks+irank];

    /* count communication entries in the upper triangular part */
    for(irank=rank+1; irank<nranks; irank++)
        n_upper += sp->n_comm_entries[rank*nranks+irank];

    /* Remap the column indices using communication maps. */
    /* Ai entries that access non-local vector parts are changed */
    /* so that they references vector entries corresponding to the position */
    /* of the original column id in the communication map */
    for(csr_index_t i=0; i<row_end-row_beg; i++){
        for(csr_index_t j=Ap[i]; j<Ap[i+1]; j++){

            col = Ai[j];

            /* all local columns are assumed to be non-empty: we do not remove them */
            if(col>=row_beg && col<row_end) {
                col = n_lower + col - row_beg;
            } else {

                if(col<row_beg) {
                    newcol = 0;
                    irank    = 0;
                    irank_end= rank;
                } else {
                    newcol = n_lower+row_end-row_beg;
                    irank    = rank+1;
                    irank_end= nranks;
                }

                /* which irank does the index belong to? */
                for(; irank<irank_end; irank++){
                    if(col>=sp->row_cpu_dist[irank+1])
                        /* next irank, so increase by all comm entries for this irank */
                        newcol += sp->n_comm_entries[rank*nranks+irank];
                    else {
                        /* this is the irank - locate the cold id in the communication list */
                        newcol += sorted_list_locate(sp->comm_pattern[rank*nranks+irank],
                                                     sp->n_comm_entries[rank*nranks+irank], col);
                        break;
                    }
                }
                col = newcol;

                /* update the entry in the communication pattern */
            }
            Ai[j] = col;
        }
    }

    /*
      Remap communication pattern accordingly
      comm_pattern and comm_pattern_ext still use
      global indices. Change them so that they reflect
      the changes we made to the Ai matrix
    */
    /* { */

        /* remap communication entries that access local irankead vector part */
        /* for(irank=0; irank<nranks; irank++){ */
        /*     for(csr_index_t i=0; i<sp->n_comm_entries[irank*nranks+rank]; i++){ */
        /*         sp->comm_pattern_ext[irank*nranks+rank][i] += n_lower - row_beg; */
        /*     } */
        /* } */

        /* remap communication entries that access external vector entries */
        /* newcol = 0; */
        /* for(irank=0; irank<nranks; irank++){ */
        /*     for(csr_index_t i=0; i<sp->n_comm_entries[rank*nranks+irank]; i++){ */
        /*         csr_index_t temp = sp->comm_pattern[rank*nranks+irank][i]; */
        /*         sp->comm_pattern[rank*nranks+irank][i] = newcol + i; */
        /*         if(temp>=row_end){ */
        /*             sp->comm_pattern[rank*nranks+irank][i] += row_end-row_beg; */
        /*         } */
        /*     } */
        /*     newcol += sp->n_comm_entries[rank*nranks+irank]; */
        /* } */
    /* } */

    /* sp->local_offset = n_lower; */
    /* sp->maxcol = n_lower+row_end-row_beg+n_upper-1; */
    /* sp->mincol = 0; */
}

void csr_get_partition(sparse_csr_t *out, const sparse_csr_t *sp, int rank, int nranks)
{
    out->dim = sp->dim;
    out->row_cpu_dist = sp->row_cpu_dist;
    out->row_beg = sp->row_cpu_dist[rank];
    out->row_end = sp->row_cpu_dist[rank+1];
    out->local_dim = out->row_end - out->row_beg;
    out->nnz = sp->Ap[out->row_end] - sp->Ap[out->row_beg];

    out->Ap = (csr_index_t*)malloc(sizeof(csr_index_t)*(sp->local_dim+1));
    out->Ai = (csr_index_t*)malloc(sizeof(csr_index_t)*sp->nnz);
    out->Ax = (csr_data_t*)malloc(sizeof(csr_data_t)*sp->nnz);

    // initialize communication data structures
    out->comm_pattern  = (csr_index_t**)calloc(nranks*nranks, sizeof(csr_index_t*));
    out->comm_pattern_size  = (csr_index_t*)calloc(nranks*nranks, sizeof(csr_index_t));
    out->n_comm_entries = (csr_index_t*)calloc(nranks*nranks, sizeof(csr_index_t));
    out->recv_ptr = (csr_data_t**)calloc(nranks, sizeof(csr_data_t*));

    // local row pointers
    for(csr_index_t row = out->row_beg, local_row = 0; row <= out->row_end; row++, local_row++){
        out->Ap[local_row] = sp->Ap[row] - sp->Ap[out->row_beg];
    }

    // local rows
    for(csr_index_t nz = sp->Ap[out->row_beg], local_nz = 0; local_nz < out->nnz; nz++, local_nz++){
        out->Ai[local_nz] = sp->Ai[nz];
        out->Ax[local_nz] = sp->Ax[nz];
    }

    csr_analyze_communication(out, rank, nranks);
    csr_remove_empty_columns(out, rank, nranks);
}

void csr_init_communication(sparse_csr_t *sp, csr_data_t *recv_vector, int rank, int nranks)
{
    csr_data_t *recv_location = recv_vector;

    /* setup pointers to recv buffers */
    for(int irank=0; irank<nranks; irank++){
        if(irank == rank) {
            /* update by local vector entries */
            recv_location += sp->row_end-sp->row_beg;
            continue;
        }

        /* here is our recv address */
        sp->recv_ptr[irank] = recv_location;

        /* proceed to next irank */
        recv_location += sp->n_comm_entries[rank*nranks+irank];
    }
}

csr_data_t csr_get_value(const sparse_csr_t *sp, csr_index_t row, csr_index_t col)
{
    csr_index_t cp;

    // NOTE: this only works for non-blocked matrices

    cp = sp->Ap[row] + sorted_list_locate(sp->Ai+sp->Ap[row], sp->Ap[row+1]-sp->Ap[row], col);
    if(sp->Ai[cp]!=col) return CMPLX(0,0);
    return sp->Ax[cp];
}

void csr_set_value(const sparse_csr_t *sp, csr_index_t row, csr_index_t col, csr_data_t val)
{
    csr_index_t cp;

    // NOTE: this only works for non-blocked matrices

    cp = sp->Ap[row] + sorted_list_locate(sp->Ai+sp->Ap[row], sp->Ap[row+1]-sp->Ap[row], col);
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


void spmv_crs_f(csr_index_t row_beg, csr_index_t row_end, sparse_csr_t *sp, const csr_data_t *x, csr_data_t *result)
{
    csr_data_t  *Ax = sp->Ax;
    csr_index_t *Ap = sp->Ap;
    csr_index_t *Ai = sp->Ai;

    csr_index_t i, j;

    register csr_data_t stemp;

    csr_index_t *tempAi;
    csr_data_t   *tempAx;
    tempAi = (csr_index_t*)(Ai+Ap[row_beg]);
    tempAx = (csr_data_t *)(Ax+Ap[row_beg]);

    for(i=row_beg; i<row_end; i++){

        stemp = 0;

        for(j=Ap[i]; j<Ap[i+1]; j++){
#ifdef USE_PREFETCHING
            _mm_prefetch((char*)&tempAx[128], _MM_HINT_NTA);
            _mm_prefetch((char*)&tempAi[128], _MM_HINT_NTA);
#endif
            stemp             += x[*tempAi++]*(*tempAx++);
        }

        result[i] += stemp;
    }
}

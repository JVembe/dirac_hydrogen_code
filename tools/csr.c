#include "csr.h"
#include <unistd.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef USE_PREFETCHING
#include <xmmintrin.h>
#endif

#include "types.h"
void csr_print(const char *fname, const sparse_csr_t *sp)
{
    printf("write file %s\n", fname);
    FILE *fd = fopen(fname, "w+");
    if(!fd) ERROR("cant open %s\n", fname);

    for(csr_index_t row = 0; row < sp->nrows; row++){
        for(csr_index_t cp = sp->Ap[row]; cp < sp->Ap[row+1]; cp++){
            fprintf(fd, "%d %d %le %le\n", row, sp->Ai[cp], creal(sp->Ax[cp]), cimag(sp->Ax[cp]));
        }
    }

    fclose(fd);
}


void csr_allocate(sparse_csr_t *out, csr_index_t nrows, csr_index_t ncols, csr_index_t nnz)
{
    out->nnz = nnz;
    out->nrows = nrows;
    out->ncols = ncols;
    out->blk_nnz = 1;
    out->blk_dim = 1;
    out->is_link = 0;
    out->perm = NULL;
    out->Ap = (csr_index_t*)calloc((out->nrows+1), sizeof(csr_index_t));
    out->Ai = (csr_index_t*)calloc(out->nnz, sizeof(csr_index_t));
    out->Ax = (csr_data_t*)calloc(out->nnz*out->blk_nnz, sizeof(csr_data_t));

    out->row_cpu_dist       = NULL;
    out->comm_pattern       = NULL;
    out->comm_pattern_size  = NULL;
    out->n_comm_entries     = NULL;
    out->recv_ptr           = NULL;
    out->send_ptr           = NULL;
    out->send_vec           = NULL;
    out->send_type          = NULL;
    out->npart              = 1;
    out->row_beg            = 0;
    out->row_end            = 0;
    out->local_offset       = 0;
}

void csr_free(sparse_csr_t *sp)
{
    free(sp->perm);    sp->perm = NULL;
    free(sp->Ap);      sp->Ap  = NULL;
    free(sp->Ai);      sp->Ai  = NULL;
    if(!sp->is_link) free(sp->Ax);  sp->Ax = NULL;
    sp->nnz = 0;
    sp->nrows = 0;
    sp->ncols = 0;
    sp->blk_nnz = 1;
    sp->blk_dim = 1;
    sp->is_link = 0;

    /* TODO: free */
    sp->row_cpu_dist       = NULL;
    sp->comm_pattern       = NULL;
    sp->comm_pattern_size  = NULL;
    sp->n_comm_entries     = NULL;
    sp->recv_ptr           = NULL;
    sp->send_ptr           = NULL;
    sp->send_vec           = NULL;
    sp->send_type          = NULL;
    sp->npart              = 1;
    sp->row_beg            = 0;
    sp->row_end            = 0;
    sp->local_offset       = 0;
}

void csr_copy(sparse_csr_t *out, const sparse_csr_t *in)
{
    // simplified version - assume same pattern, simply copy the non-zeros
    if(out->nnz == in->nnz){
        memcpy(out->Ax, in->Ax, sizeof(csr_data_t)*out->nnz*out->blk_nnz);
        return;
    }
    
    out->nnz = in->nnz;
    out->nrows = in->nrows;
    out->ncols = in->ncols;
    out->blk_nnz = in->blk_nnz;
    out->blk_dim = in->blk_dim;
    out->is_link = 0;

    out->Ap = (csr_index_t*)malloc(sizeof(csr_index_t)*(out->nrows+1));
    memcpy(out->Ap, in->Ap, sizeof(csr_index_t)*(out->nrows+1));
    out->Ai = (csr_index_t*)malloc(sizeof(csr_index_t)*out->nnz);
    memcpy(out->Ai, in->Ai, sizeof(csr_index_t)*out->nnz);
    out->Ax = (csr_data_t*)malloc(sizeof(csr_data_t)*out->nnz*out->blk_nnz);
    memcpy(out->Ax, in->Ax, sizeof(csr_data_t)*out->nnz*out->blk_nnz);

    out->perm = NULL;
    if(in->perm){
        out->perm = (csr_index_t*)malloc(sizeof(csr_index_t)*out->ncols);
        memcpy(out->perm, in->perm, sizeof(csr_index_t)*out->ncols);
    }

    /* TODO: reallocate and copy */
    out->row_cpu_dist       = in->row_cpu_dist;
    out->comm_pattern       = in->comm_pattern;
    out->comm_pattern_size  = in->comm_pattern_size;
    out->n_comm_entries     = in->n_comm_entries;
    out->recv_ptr           = in->recv_ptr;
    out->send_ptr           = in->send_ptr;
    out->send_vec           = in->send_vec;
    out->send_type          = in->send_type;
    out->row_beg            = in->row_beg;
    out->row_end            = in->row_end;
    out->local_offset       = in->local_offset;
}

void csr_diag(sparse_csr_t *out, csr_index_t dim)
{
    csr_allocate(out, dim, dim, dim);

    // fill the diagonal with non-zero entries
    for(csr_index_t i=0; i<dim; i++) {
        out->Ai[i] = i;
        out->Ap[i] = i;
    }
    out->Ap[dim] = dim;
}

void csr_block_params(sparse_csr_t *sp, csr_index_t blk_dim, csr_index_t blk_nnz)
{
    sp->blk_dim = blk_dim;
    sp->blk_nnz = blk_nnz;

    // update Ax storage
    if(!sp->is_link) free(sp->Ax);  sp->Ax = NULL;
    sp->is_link = 0;

    // do not allocate new storage - can be done when needed
    // sp->Ax = (csr_data_t*)malloc(sizeof(csr_data_t)*sp->nnz*sp->blk_nnz);
    // bzero(sp->Ax, sizeof(csr_data_t)*sp->nnz*sp->blk_nnz);

    // do NOT update Ap pointers - need them for Ai
}

void csr_block_insert(sparse_csr_t *sp, csr_index_t row, csr_index_t col, csr_data_t *blk_ptr)
{
    csr_index_t cp;
    cp = sp->Ap[row] + sorted_list_locate(sp->Ai+sp->Ap[row], sp->Ap[row+1]-sp->Ap[row], col);
    if(cp==sp->Ap[row+1] || sp->Ai[cp]!=col) ERROR("cant insert block: (%d,%d) not present in CSR.", row, col);
    memcpy(sp->Ax + cp*sp->blk_nnz, blk_ptr, sizeof(csr_data_t)*sp->blk_nnz);
}

void csr_full_insert(sparse_csr_t *Afull, csr_index_t row, csr_index_t col, sparse_csr_t *submatrix)
{
    int blkdim = submatrix->nrows;
    csr_index_t row_blk, col_blk, colp_blk;
    csr_index_t row_dst, col_dst;

    // insert into non-blocked Hfull matrix
    for(row_blk=0; row_blk<blkdim; row_blk++){

        // submatrix row and col
        colp_blk=submatrix->Ap[row_blk];
        col_blk = submatrix->Ai[colp_blk];

        // Afull row and col
        col_dst = col*blkdim + col_blk;
        row_dst = row*blkdim + row_blk;

        // locate row start in Afull
        csr_index_t cp = Afull->Ap[row_dst] +
            sorted_list_locate(Afull->Ai + Afull->Ap[row_dst], Afull->Ap[row_dst+1] - Afull->Ap[row_dst], col_dst);
        if(cp==Afull->Ap[row_dst+1] || Afull->Ai[cp]!=col_dst)
            ERROR("cant set matrix value: (%d,%d) not present in CSR.", row_dst, col_dst);

        csr_index_t nrowent = submatrix->Ap[row_blk+1] - colp_blk;
        memcpy(Afull->Ax + cp, submatrix->Ax + colp_blk, nrowent*sizeof(csr_data_t));
    }
}

void csr_block_link(sparse_csr_t *sp_blk, sparse_csr_t *sp, csr_index_t row, csr_index_t col)
{
    csr_index_t cp;
    cp = sp->Ap[row] + sorted_list_locate(sp->Ai+sp->Ap[row], sp->Ap[row+1]-sp->Ap[row], col);
    if(cp==sp->Ap[row+1] || sp->Ai[cp]!=col) ERROR("cant insert block: (%d,%d) not present in CSR.", row, col);
    sp_blk->is_link = 1;
    sp_blk->Ax = sp->Ax + cp*sp->blk_nnz;
    sp_blk->nrows = sp->blk_dim;
    sp_blk->ncols = sp->blk_dim;
    sp_blk->nnz = sp->blk_nnz;
}

csr_index_t csr_ncols(const sparse_csr_t *sp_blk)
{
    return sp_blk->ncols*sp_blk->blk_dim;
}

csr_index_t csr_nrows(const sparse_csr_t *sp_blk)
{
    return sp_blk->nrows*sp_blk->blk_dim;
}

csr_index_t csr_ncolblocks(const sparse_csr_t *sp_blk)
{
    return sp_blk->ncols;
}

csr_index_t csr_nrowblocks(const sparse_csr_t *sp_blk)
{
    return sp_blk->nrows;
}

csr_index_t csr_local_rowoffset(const sparse_csr_t *sp_blk)
{
    return sp_blk->local_offset;
}

csr_index_t csr_nnz(const sparse_csr_t *sp_blk)
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
    nread = fread(&sp->nrows, sizeof(csr_index_t), 1, fd);
    // TODO: change this to account for generalmatrices
    sp->ncols = sp->nrows;
    nread = fread(&sp->nnz, sizeof(csr_index_t), 1, fd);

    // assume 1 partition
    sp->npart = 1;
    sp->row_beg = 0;
    sp->row_end = sp->nrows;
    sp->local_offset = 0;

    sp->Ap = (csr_index_t*)malloc(sizeof(csr_index_t)*(sp->nrows+1));
    nread = fread(sp->Ap, sizeof(csr_index_t), (sp->nrows+1), fd);
    if(nread!=(sp->nrows+1)) ERROR("wrong file format in %s\n", fname);
    if(sp->Ap[sp->nrows] != sp->nnz) ERROR("wrong file format (nnz) in %s: nrows %d nnz %d Ap %d\n",
                                           fname, sp->nrows, sp->nnz, sp->Ap[sp->nrows]);

    sp->Ai = (csr_index_t*)malloc(sizeof(csr_index_t)*sp->nnz);
    nread = fread(sp->Ai, sizeof(csr_index_t), sp->nnz, fd);
    if(nread!=sp->nnz) ERROR("wrong file format in %s\n", fname);

    sp->Ax = (csr_data_t*)malloc(sizeof(csr_data_t)*sp->nnz);
    nread = fread(sp->Ax, sizeof(csr_data_t), sp->nnz, fd);
    if(nread!=sp->nnz) ERROR("wrong file format in %s\n", fname);

    // check if we have node number perms
    sp->perm = (csr_index_t*)malloc(sizeof(csr_index_t)*sp->ncols);
    nread = fread(sp->perm, sizeof(csr_index_t), sp->ncols, fd);
    if(nread!=sp->ncols){
        free(sp->perm);
        sp->perm = NULL;
    } else {

        // we need the partitioning
        nread = fread(&sp->npart, sizeof(csr_index_t), 1, fd);
        if(nread!=1) ERROR("wrong file format in %s: partitioning info not found\n", fname);

        sp->row_cpu_dist = (csr_index_t*)malloc(sizeof(csr_index_t)*(sp->npart+1));
        nread = fread(sp->row_cpu_dist, sizeof(csr_index_t), sp->npart+1, fd);
        if(nread!=(sp->npart+1)) ERROR("wrong file format in %s: partitioning info inconsistent\n", fname);
    }

    fclose(fd);
}

void csr_write(const char *fname, const sparse_csr_t *sp)
{
    size_t nwrite;

    FILE *fd = fopen(fname, "w+");
    if(!fd) ERROR("cant open %s\n", fname);

    // storage format: dim, nnz, Ap, Ai, Ax
    // TODO: change this to account for generalmatrices
    nwrite = fwrite(&sp->nrows, sizeof(csr_index_t), 1, fd);
    nwrite = fwrite(&sp->nnz, sizeof(csr_index_t), 1, fd);

    nwrite = fwrite(sp->Ap, sizeof(csr_index_t), csr_nrows(sp)+1, fd);
    if(nwrite!=csr_nrows(sp)+1) ERROR("cant write file %s\n", fname);

    nwrite = fwrite(sp->Ai, sizeof(csr_index_t), sp->nnz, fd);
    if(nwrite!=sp->nnz) ERROR("cant write file %s\n", fname);

    nwrite = fwrite(sp->Ax, sizeof(csr_data_t), sp->nnz, fd);
    if(nwrite!=sp->nnz) ERROR("cant write file %s\n", fname);

    fclose(fd);
}

void csr_ijk_write(const char *fname, const sparse_csr_t *sp)
{
    csr_index_t *Aj;
    Aj = (csr_index_t *)malloc(sizeof(csr_index_t)*sp->nnz);

    for(csr_index_t i=0; i<sp->nrows; i++){
        for(csr_index_t j=sp->Ap[i]; j<sp->Ap[i+1]; j++){
            Aj[j] = i;
        }
    }

    size_t nwrite;

    FILE *fd = fopen(fname, "w+");
    if(!fd) ERROR("cant open %s\n", fname);

    // storage format: dim, nnz, Ai, Aj, Ax
    nwrite = fwrite(&sp->nrows, sizeof(csr_index_t), 1, fd);
    nwrite = fwrite(&sp->nnz, sizeof(csr_index_t), 1, fd);

    nwrite = fwrite(sp->Ai, sizeof(csr_index_t), sp->nnz, fd);
    if(nwrite!=sp->nnz) ERROR("cant write file %s\n", fname);

    nwrite = fwrite(Aj, sizeof(csr_index_t), sp->nnz, fd);
    if(nwrite!=sp->nnz) ERROR("cant write file %s\n", fname);

    nwrite = fwrite(sp->Ax, sizeof(csr_data_t), sp->nnz, fd);
    if(nwrite!=sp->nnz) ERROR("cant write file %s\n", fname);

    fclose(fd);

    free(Aj);
}

/* walk through Ai and save communication (non-local) columns */
void csr_analyze_comm(sparse_csr_t *sp, int rank, int nranks)
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

void csr_exchange_comm_info(sparse_csr_t *sp, int rank, int nranks)
{
#ifdef USE_MPI

    if(nranks==1) return;

    /* first exchange the number of communication entries between all ranks */
    CHECK_MPI(MPI_Allgather(MPI_IN_PLACE /* sp->n_comm_entries+rank*nranks */, nranks, MPI_CSR_INDEX_T,
                            sp->n_comm_entries, nranks, MPI_CSR_INDEX_T, MPI_COMM_WORLD));

    /* now exchange the actual column indices */
    /* first come recv requests, then send requests */
    MPI_Request comm_requests[2*nranks];

    /* pre-post recv requests for all ranks that will send to us */
    for(int irank=0; irank<nranks; irank++){
        csr_index_t nent = sp->n_comm_entries[irank*nranks + rank];
        comm_requests[irank] = MPI_REQUEST_NULL;
        if(0 != nent){

            /* allocate index buffer */
            sp->comm_pattern[irank*nranks + rank] = (csr_index_t*)calloc(nent, sizeof(csr_index_t));
            sp->n_comm_entries[irank*nranks + rank] = nent;

            /* submit a recv */
            CHECK_MPI(MPI_Irecv(sp->comm_pattern[irank*nranks + rank], nent, MPI_CSR_INDEX_T, irank, 0,
                                MPI_COMM_WORLD, comm_requests+irank));
        }
    }


    /* send column indices to the ranks that should receive them */
    for(int irank=0; irank<nranks; irank++){
        csr_index_t nent = sp->n_comm_entries[rank*nranks + irank];
        comm_requests[nranks + irank] = MPI_REQUEST_NULL;
        if(0 != nent){

            /* submit a send */
            CHECK_MPI(MPI_Isend(sp->comm_pattern[rank*nranks + irank], nent, MPI_CSR_INDEX_T, irank, 0,
                                MPI_COMM_WORLD, comm_requests+nranks+irank));
        }
    }

    /* progress the communication */
    CHECK_MPI(MPI_Waitall(2*nranks, comm_requests, MPI_STATUSES_IGNORE));

#endif
}

void csr_remove_empty_columns(sparse_csr_t *sp, int rank, int nranks, csr_index_t *perm)
{
    csr_index_t n_lower = 0, n_upper = 0;
    csr_index_t col, new_col, orig_col;
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

    /* re-map the permutation vector */
    sp->ncols = row_end-row_beg+n_lower+n_upper;
    sp->local_offset = n_lower;
    sp->perm = (csr_index_t*)malloc(sizeof(csr_index_t)*sp->ncols);

    /* Remap the column indices using communication perms. */
    /* Ai entries that access non-local vector parts are changed */
    /* so that they references vector entries corresponding to the position */
    /* of the original column id in the communication perm */
    for(csr_index_t i=0; i<row_end-row_beg; i++){
        for(csr_index_t j=Ap[i]; j<Ap[i+1]; j++){

            col = Ai[j];
            orig_col = perm[col];

            /* all local columns are assumed to be non-empty: we do not remove them */
            if(col>=row_beg && col<row_end) {
                col = n_lower + col - row_beg;
            } else {

                if(col<row_beg) {
                    new_col = 0;
                    irank    = 0;
                    irank_end= rank;
                } else {
                    new_col = n_lower+row_end-row_beg;
                    irank    = rank+1;
                    irank_end= nranks;
                }

                /* which irank does the index belong to? */
                for(; irank<irank_end; irank++){
                    if(col>=sp->row_cpu_dist[irank+1])
                        /* next irank, so increase by all comm entries for this irank */
                        new_col += sp->n_comm_entries[rank*nranks+irank];
                    else {
                        /* this is the irank - locate the cold id in the communication list */
                        new_col += sorted_list_locate(sp->comm_pattern[rank*nranks+irank],
                                                      sp->n_comm_entries[rank*nranks+irank], col);
                        break;
                    }
                }
                col = new_col;
            }

            /* remap the row indices */
            Ai[j] = col;

            /* remap the permutation */
            sp->perm[col] = orig_col;
        }
    }

    /*
      Remap communication pattern accordingly. comm_pattern still uses global indices.
      Change it so that it reflects the changes we made to the Ai matrix.
    */

    /* remap communication entries that access local irankead vector part */
    for(irank=0; irank<nranks; irank++){
        for(csr_index_t i=0; i<sp->n_comm_entries[irank*nranks+rank]; i++){
            sp->comm_pattern[irank*nranks+rank][i] += n_lower - row_beg;
        }
    }

    /*
    // DEBUG: write out the result vectors for comparison with single-rank result
    for(int r=0; r<nranks; r++){
    if(rank == r){
    // for(int i = 0; i < nranks*nranks; i++) fprintf(stdout, "%d ", sp->n_comm_entries[i]); fprintf(stdout, "\n");
    for(int irank = 0; irank < nranks; irank++) {
    csr_index_t nent = sp->n_comm_entries[irank*nranks + rank];
    printf("%d: send to %d: %d\n", rank, irank, nent);
    if(nent){
    for(int i=0; i<nent; i++) printf("%d ", sp->comm_pattern[irank*nranks + rank][i]); printf("\n");
    }
    }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank==0) printf("-------------------------\n");

    for(int r=0; r<nranks; r++){
    if(rank == r){
    // for(int i = 0; i < nranks*nranks; i++) fprintf(stdout, "%d ", sp->n_comm_entries[i]); fprintf(stdout, "\n");
    for(int irank = 0; irank < nranks; irank++) {
    csr_index_t nent = sp->n_comm_entries[rank*nranks + irank];
    printf("%d: recv from %d: %d\n", rank, irank, nent);
    if(nent){
    for(int i=0; i<nent; i++) printf("%d ", sp->comm_pattern[rank*nranks + irank][i]); printf("\n");
    }
    }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    */
}

void csr_get_partition(sparse_csr_t *out, const sparse_csr_t *sp, int rank, int nranks)
{
    csr_index_t row_beg = sp->row_cpu_dist[rank];
    csr_index_t row_end = sp->row_cpu_dist[rank+1];
    csr_allocate(out, row_end-row_beg, csr_ncols(sp), sp->Ap[row_end]-sp->Ap[row_beg]);

    /* keep the row distribution info in each partition - needed for communication */
    /* TODO: copy / reallocate row_cpu_dist */
    out->row_cpu_dist = sp->row_cpu_dist;
    out->row_beg = row_beg;
    out->row_end = row_end;

    /* initialize communication data structures */
    out->comm_pattern       = (csr_index_t**)calloc(nranks*nranks, sizeof(csr_index_t*));
    out->comm_pattern_size  = (csr_index_t*)calloc(nranks*nranks, sizeof(csr_index_t));
    out->n_comm_entries     = (csr_index_t*)calloc(nranks*nranks, sizeof(csr_index_t));
    out->recv_ptr           = (csr_data_t**)calloc(nranks, sizeof(csr_data_t*));
    out->send_ptr           = (csr_data_t**)calloc(nranks, sizeof(csr_data_t*));

    /* localize row pointers */
    for(csr_index_t row = row_beg, local_row = 0; row <= row_end; row++, local_row++){
        out->Ap[local_row] = sp->Ap[row] - sp->Ap[row_beg];
    }

    /* copy local matrix values and column indices */
    for(csr_index_t nz = sp->Ap[row_beg], local_nz = 0; local_nz < out->nnz; nz++, local_nz++){
        out->Ai[local_nz] = sp->Ai[nz];
        out->Ax[local_nz] = sp->Ax[nz];
    }

    /* prepare communication: */
    /*  - analyze local matrix part, identify non-local column access */
    /*  - exchange communication info with neighbors */
    csr_analyze_comm(out, rank, nranks);
    csr_exchange_comm_info(out, rank, nranks);

    /* Remap the matrices for local indexing. */
    /* This essentially means that all local Ai indices */
    /* access columns from 0 to number of local columns, and not the entire vector length. */
    /* This is a must for scalability, otherwise each rank would have to allocate O(n) vectors */
    /* instead of O(n/nranks) vectors */
    csr_remove_empty_columns(out, rank, nranks, sp->perm);
}

void csr_unblock_matrix(sparse_csr_t *out, const sparse_csr_t *in, const sparse_csr_t *sp_blk)
{
    // iterators over Hall
    csr_index_t row, col, colp;

    // iterators over G (block submatrices)
    csr_index_t row_blk, colp_blk;

    // iterators over non-blocked Hfull
    csr_index_t col_dst, rowp_dst;
    rowp_dst = 1;

    // allocate memory for the unblocked matrix
    csr_allocate(out, csr_nrows(in), csr_ncols(in), csr_nnz(in));

    // this is true only for 1 rank. fixed later in csr_unblock_comm_info
    out->row_beg = 0;
    out->row_end = out->nrows;

    // for all rows
    for(row = 0; row < in->nrows; row++){

        // each row and each column are expanded into submatrices of size in->blk_dim
        for(row_blk=0; row_blk<in->blk_dim; row_blk++){

            // row in the expanded matrix
            // row_dst = row*in->blk_dim + row_blk;

            // for non-zeros in each Hpart row - fill the expanded row
            for(colp = in->Ap[row]; colp < in->Ap[row+1]; colp++){
                col = in->Ai[colp];

                for(colp_blk=sp_blk->Ap[row_blk]; colp_blk<sp_blk->Ap[row_blk+1]; colp_blk++){

                    // column in the expanded matrix
                    col_dst = col*in->blk_dim + sp_blk->Ai[colp_blk];

                    // update out->Ai and out->Ap
                    out->Ai[out->Ap[rowp_dst]] = col_dst;
                    out->Ap[rowp_dst]++;
                }
            }

            // next Hfull row - start where the last one ends
            out->Ap[rowp_dst+1] = out->Ap[rowp_dst];
            rowp_dst++;
        }
    }
}

void csr_blocked_to_full(sparse_csr_t *Afull, sparse_csr_t *Ablk, sparse_csr_t *submatrix)
{
    csr_index_t row, col, colp;
    int blkdim = Ablk->blk_dim;
    for(row = 0; row < Ablk->nrows; row++){

        // for non-zeros in each row
        for(colp = Ablk->Ap[row]; colp < Ablk->Ap[row+1]; colp++){

            // NOTE: rows and cols are remapped wrt. the original numbering in H
            col = Ablk->Ai[colp];

            csr_block_link(submatrix, Ablk, row, col);

            // insert into non-blocked Hfull matrix
            csr_index_t row_blk, col_blk, colp_blk;
            csr_index_t row_dst, col_dst;
            csr_index_t valp = 0;
            for(row_blk=0; row_blk<blkdim; row_blk++){

                row_dst = row*blkdim + row_blk;
                for(colp_blk=submatrix->Ap[row_blk]; colp_blk<submatrix->Ap[row_blk+1]; colp_blk++){
                    col_blk = submatrix->Ai[colp_blk];
                    col_dst = col*blkdim + col_blk;

                    csr_set_value(Afull, row_dst, col_dst, submatrix->Ax[valp]);
                    valp++;
                }
            }
        }
    }
}

/* spell out blocked communication data structures */
/* to explicit, non-blockes indices used by a non-blocked matrix */
void csr_unblock_comm_info(sparse_csr_t *out, const sparse_csr_t *in, int rank, int nranks)
{
    if(nranks==1) return;

    /* initialize communication data structures */
    out->row_cpu_dist = (csr_index_t*)calloc(nranks+1, sizeof(csr_index_t));
    for(csr_index_t i=0; i<nranks+1; i++)
        out->row_cpu_dist[i] = in->row_cpu_dist[i]*in->blk_dim;
    out->row_beg = out->row_cpu_dist[rank];
    out->row_end = out->row_cpu_dist[rank+1];

    out->n_comm_entries     = (csr_index_t*)calloc(nranks*nranks, sizeof(csr_index_t));
    for(int i=0; i<nranks*nranks; i++)
        out->n_comm_entries[i] = in->n_comm_entries[i]*in->blk_dim;

    out->comm_pattern       = (csr_index_t**)calloc(nranks*nranks, sizeof(csr_index_t*));
    out->comm_pattern_size  = (csr_index_t*)calloc(nranks*nranks, sizeof(csr_index_t));
    for(int irank=0; irank<nranks; irank++){
        csr_index_t comm_iter;
        csr_index_t nent;
        int comm_idx;

        comm_idx = irank*nranks + rank;
        nent = out->n_comm_entries[comm_idx];
        if(nent){
            out->comm_pattern_size[comm_idx] = nent;
            out->comm_pattern[comm_idx] = (csr_index_t*)calloc(nent, sizeof(csr_index_t));
            comm_iter = 0;
            for(csr_index_t j=0; j<in->n_comm_entries[comm_idx]; j++){
                for(csr_index_t jj=0; jj<in->blk_dim; jj++){
                    out->comm_pattern[comm_idx][comm_iter++] = in->comm_pattern[comm_idx][j]*in->blk_dim + jj;
                }
            }
        }

        /* this is the received part - no need to unblock it, those are contiguous vector parts */
        /* comm_idx = rank*nranks + irank; */
        /* nent = out->n_comm_entries[rank*nranks + irank]; */
        /* if(nent){ */
        /*     out->comm_pattern_size[comm_idx] = nent; */
        /*     out->comm_pattern[comm_idx] = (csr_index_t*)calloc(nent, sizeof(csr_index_t)); */
        /*     comm_iter = 0; */
        /*     for(csr_index_t j=0; j<in->n_comm_entries[comm_idx]; j++){ */
        /*         for(csr_index_t jj=0; jj<in->blk_dim; jj++){ */
        /*             out->comm_pattern[comm_idx][comm_iter++] = in->comm_pattern[comm_idx][j]*in->blk_dim + jj; */
        /*         } */
        /*     } */
        /* } */
    }

    /* create indexed datatypes for send operations */
    out->send_type = malloc(nranks*sizeof(MPI_Datatype));
    for(int irank=0; irank<nranks; irank++){
        out->send_type[irank] = MPI_DATATYPE_NULL;
        csr_index_t count = in->n_comm_entries[irank*nranks+rank];
        if(count){
            int *blocklengths  = malloc(sizeof(int)*count);
            for(int i=0; i<count; i++)
                blocklengths[i] = in->blk_dim*2;

            int *displacements = malloc(sizeof(int)*count);
            for(int i=0; i<count; i++)
                displacements[i] = in->comm_pattern[irank*nranks+rank][i]*in->blk_dim*2;

            CHECK_MPI(MPI_Type_indexed(count, blocklengths, displacements, MPI_DOUBLE, out->send_type + irank));
            CHECK_MPI(MPI_Type_commit(out->send_type + irank));

            free(blocklengths);
            free(displacements);
        }
    }


    /* details filled in init_communication */
    out->recv_ptr           = (csr_data_t**)calloc(nranks, sizeof(csr_data_t*));
    out->send_ptr           = (csr_data_t**)calloc(nranks, sizeof(csr_data_t*));

    out->npart              = nranks;
    out->local_offset       = in->local_offset*in->blk_dim;
}

void csr_init_communication(sparse_csr_t *sp, csr_data_t *px, int rank, int nranks)
{
    csr_data_t *recv_location = px;

    if(nranks==1) return;

    /* store the vector pointer for sending data */
    sp->send_vec = px;

    /* setup pointers to recv buffers */
    for(int irank=0; irank<nranks; irank++){
        if(irank == rank) {
            /* update by local vector entries */
            recv_location += (sp->row_end-sp->row_beg)*sp->blk_dim;
            continue;
        }

        /* setup the recv address */
        if(sp->n_comm_entries[rank*nranks+irank]){
            sp->recv_ptr[irank] = recv_location;
        }
        recv_location += sp->n_comm_entries[rank*nranks+irank]*sp->blk_dim;

        /* send buffers needed for explicit packing */
        if(NULL==sp->send_type){

            /* setup the send buffer */
            if(sp->n_comm_entries[irank*nranks+rank] && !sp->send_ptr[irank]){
                csr_index_t nent = sp->n_comm_entries[irank*nranks+rank]*sp->blk_dim;
                sp->send_ptr[irank] = (csr_data_t*)calloc(nent, sizeof(csr_data_t));
            }
        }
    }
}

void csr_comm(const sparse_csr_t *sp, int rank, int nranks)
{
#ifdef USE_MPI

    if(nranks==1) return;
    MPI_Request comm_requests[2*nranks];

    /* pre-post recv requests */
    for(int irank=0; irank<nranks; irank++){
        comm_requests[irank] = MPI_REQUEST_NULL;
        if(sp->recv_ptr[irank]){
            csr_index_t nent = sp->n_comm_entries[rank*nranks+irank]*sp->blk_dim;
            CHECK_MPI(MPI_Irecv(sp->recv_ptr[irank], nent*2, MPI_DOUBLE, irank, 0,
                                MPI_COMM_WORLD, comm_requests+irank));
        }
    }

    for(int irank=0; irank<nranks; irank++){
        comm_requests[nranks+irank] = MPI_REQUEST_NULL;
        if(sp->send_type){
            /* use MPI_Type_indexed for sending - supports GPU */
            if(MPI_DATATYPE_NULL != sp->send_type[irank]){
                CHECK_MPI(MPI_Isend(sp->send_vec, 1, sp->send_type[irank], irank, 0,
                                    MPI_COMM_WORLD, comm_requests+nranks+irank));
            }
        } else {
            /* gather sent data into send buffers and send */
            if(sp->send_ptr[irank]){
                csr_index_t nent = sp->n_comm_entries[irank*nranks+rank]*sp->blk_dim;
                csr_index_t col_local;
                csr_index_t col_iter = 0;
                for(csr_index_t i=0; i<sp->n_comm_entries[irank*nranks+rank]; i++){
                    for(int j=0; j<sp->blk_dim; j++){
                        col_local = sp->comm_pattern[irank*nranks+rank][i]*sp->blk_dim + j;
                        sp->send_ptr[irank][col_iter++] = sp->send_vec[col_local];
                    }
                }
                CHECK_MPI(MPI_Isend(sp->send_ptr[irank], nent*2, MPI_DOUBLE, irank, 0,
                                    MPI_COMM_WORLD, comm_requests+nranks+irank));
            }
        }
    }

    /* progress the communication */
    CHECK_MPI(MPI_Waitall(2*nranks, comm_requests, MPI_STATUSES_IGNORE));

#endif
}

csr_data_t csr_get_value(const sparse_csr_t *sp, csr_index_t row, csr_index_t col)
{
    // NOTE: this only works for non-blocked matrices

    csr_index_t  cp = sp->Ap[row] + sorted_list_locate(sp->Ai+sp->Ap[row], sp->Ap[row+1]-sp->Ap[row], col);
    if(cp==sp->Ap[row+1] || sp->Ai[cp]!=col) return CMPLX(0,0);
    return sp->Ax[cp];
}

void csr_set_value(const sparse_csr_t *sp, csr_index_t row, csr_index_t col, csr_data_t val)
{
    // NOTE: this only works for non-blocked matrices

    csr_index_t cp = sp->Ap[row] + sorted_list_locate(sp->Ai+sp->Ap[row], sp->Ap[row+1]-sp->Ap[row], col);
    if(cp==sp->Ap[row+1] || sp->Ai[cp]!=col) ERROR("cant set matrix value: (%d,%d) not present in CSR.", row, col);
    sp->Ax[cp] = val;
}

void csr_conj_transpose(sparse_csr_t *out, const sparse_csr_t *in)
{
    // NOTE: this only works for non-blocked matrices

    for(csr_index_t row = 0; row < in->nrows; row++){
        for(csr_index_t cp = in->Ap[row]; cp < in->Ap[row+1]; cp++){
            csr_set_value(out, in->Ai[cp], row, conj(in->Ax[cp]));
        }
    }
}

void csr_spmv(csr_index_t row_beg, csr_index_t row_end, const sparse_csr_t *sp, const csr_data_t *x, csr_data_t *result)
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

        stemp = CMPLX(0, 0);

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

void csr_coo2csr(sparse_csr_t *sp, const csr_index_t *rowidx, const csr_index_t *colidx, const csr_data_t *val,
                 csr_index_t matrix_dim, csr_index_t nnz)
{

    csr_index_t **lists_dynamic = 0;
    csr_index_t *n_list_elems_dynamic = 0;
    csr_index_t *list_size_dynamic = 0;
    csr_index_t *Ap_out = NULL;
    csr_index_t *Ai_out = NULL;
    csr_data_t  *Ax_out = NULL;

    /* if we run out of space, allocate dynamic lists */
    lists_dynamic = calloc(sizeof(csr_index_t*)*matrix_dim, 1);
    n_list_elems_dynamic = calloc(sizeof(csr_index_t)*matrix_dim, 1);
    list_size_dynamic = calloc(sizeof(csr_index_t)*matrix_dim, 1);

    /* find unique row / column pairs */
    for(csr_index_t i=0; i<nnz; i++){
        csr_index_t row, col, pos;
        row = rowidx[i];
        col = colidx[i];

        /* check if a given row has a dynamic list */
        if(!list_size_dynamic[row]){
            sorted_list_create(lists_dynamic + row, list_size_dynamic + row);
        }

        /* add column to row, ignores existing entries */
        sorted_list_add(lists_dynamic + row, n_list_elems_dynamic + row, list_size_dynamic + row, col);
    }

    /* allocate CSR data structures: Ap */
    Ap_out = calloc(sizeof(csr_index_t)*(matrix_dim+1), 1);

    /* Count a cumulative sum of the number of non-zeros in every matrix row. */
    for(csr_index_t i=0; i<matrix_dim; i++) {
        Ap_out[i+1] = Ap_out[i] + n_list_elems_dynamic[i];
    }

    /* allocate CSR data structures: Ai and Ax */
    csr_index_t csr_nnz = Ap_out[matrix_dim];
    Ai_out = malloc(sizeof(csr_index_t)*csr_nnz);
    Ax_out = calloc(sizeof(csr_data_t)*csr_nnz, 1);

    /* copy rows into Ai */
    for(csr_index_t i=0; i<matrix_dim; i++){
        csr_index_t rowp = Ap_out[i];
        csr_index_t nent = Ap_out[i+1] - Ap_out[i];
        if(n_list_elems_dynamic[i] != nent) ERROR("wrong number of row entries in coo2csr");
        memcpy(Ai_out+rowp, lists_dynamic[i], sizeof(csr_index_t)*nent);
    }

    /* accumulate values into Ax */
    for(csr_index_t i=0; i<nnz; i++){
        csr_index_t row, col;
        row = rowidx[i];
        col = colidx[i];

        csr_index_t nent = Ap_out[row+1] - Ap_out[row];
        csr_index_t cp = Ap_out[row] + sorted_list_locate(Ai_out + Ap_out[row], nent, col);
        if(cp==Ap_out[row+1] || Ai_out[cp]!=col) ERROR("cant set matrix value: (%d,%d) not present in CSR.", row, col);
        Ax_out[cp] += val[i];
    }

    /* cleanup */
    for(csr_index_t i=0; i<matrix_dim; i++){
        free(lists_dynamic[i]);
    }
    free(lists_dynamic);
    free(n_list_elems_dynamic);
    free(list_size_dynamic);

    /* output */
    sp->nnz = csr_nnz;
    sp->nrows = matrix_dim;
    sp->ncols = matrix_dim;
    sp->blk_nnz = 1;
    sp->blk_dim = 1;
    sp->is_link = 0;
    sp->perm = NULL;
    sp->Ap = Ap_out;
    sp->Ai = Ai_out;
    sp->Ax = Ax_out;

    sp->row_cpu_dist       = NULL;
    sp->comm_pattern       = NULL;
    sp->comm_pattern_size  = NULL;
    sp->n_comm_entries     = NULL;
    sp->recv_ptr           = NULL;
    sp->send_ptr           = NULL;
    sp->send_vec           = NULL;
    sp->npart              = 1;
    sp->row_beg            = 0;
    sp->row_end            = 0;
    sp->local_offset       = 0;
}

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

#ifndef _CSR_H
#define _CSR_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "types.h"
#include "utils.h"
#include "sorted_list.h"

typedef struct {
    csr_index_t nrows, ncols, nnz;    
    csr_index_t blk_dim, blk_nnz;
    csr_index_t npart, row_beg, row_end, local_offset;
    csr_index_t *perm, *row_cpu_dist;
    csr_index_t **comm_pattern, *comm_pattern_size, *n_comm_entries;
    csr_data_t  **recv_ptr, **send_ptr, *send_vec;
#ifdef USE_MPI
    MPI_Datatype *send_type;
#else
    // need same size of the structure for GPUs
    void *voidptr;
#endif
    csr_index_t *Ap;
    csr_index_t *Ai;
    csr_data_t  *Ax;
    csr_index_t *Ai_sub_map;
    int is_link;
} sparse_csr_t;

void csr_print(const char *fname, const sparse_csr_t *sp);
void csr_allocate(sparse_csr_t *out, csr_index_t nrows, csr_index_t ncols, csr_index_t nnz);
void csr_free(sparse_csr_t *sp);
void csr_copy(sparse_csr_t *out, const sparse_csr_t *in);
void csr_zero(sparse_csr_t *sp);
void csr_diag(sparse_csr_t *sp, csr_index_t dim);

csr_index_t csr_nnz(const sparse_csr_t *sp_blk);
csr_index_t csr_nrows(const sparse_csr_t *sp_blk);
csr_index_t csr_ncols(const sparse_csr_t *sp_blk);
csr_index_t csr_ncolblocks(const sparse_csr_t *sp_blk);
csr_index_t csr_nrowblocks(const sparse_csr_t *sp_blk);
csr_index_t csr_local_rowoffset(const sparse_csr_t *sp_blk);

void csr_read(const char *fname, sparse_csr_t *sp);
void csr_write(const char *fname, const sparse_csr_t *sp);
void csr_ijk_write(const char *fname, const sparse_csr_t *sp);

void csr_block_params(sparse_csr_t *sp, csr_index_t blk_dim, csr_index_t blk_nnz);
void csr_block_insert(sparse_csr_t *sp, csr_index_t row, csr_index_t col, csr_data_t *blk_ptr);
void csr_full_insert(sparse_csr_t *Afull, csr_index_t row, csr_index_t col, sparse_csr_t *submatrix);
void csr_block_link(sparse_csr_t *sp_blk, sparse_csr_t *sp, csr_index_t row, csr_index_t col);

void csr_get_partition(sparse_csr_t *out, const sparse_csr_t *sp, int rank, int nranks);
void csr_unblock_matrix(sparse_csr_t *out, const sparse_csr_t *in, const sparse_csr_t *sp_blk);
void csr_blocked_to_full(sparse_csr_t *Afull, sparse_csr_t *Ablk, sparse_csr_t *submatrix);
void csr_unblock_comm_info(sparse_csr_t *out, const sparse_csr_t *in, int rank, int nranks);
void csr_init_communication(sparse_csr_t *sp, csr_data_t *px, int rank, int nranks);
void csr_comm(const sparse_csr_t *sp, int rank, int nranks);
void csr_print_comm_info(sparse_csr_t *sp, int rank, int nranks, int blkdim);

csr_data_t csr_get_value(const sparse_csr_t *sp, csr_index_t row, csr_index_t col);
void csr_set_value(const sparse_csr_t *sp, csr_index_t row, csr_index_t col, csr_data_t val);
void csr_conj_transpose(sparse_csr_t *out, const sparse_csr_t *in);

void csr_spmv(csr_index_t row_l, csr_index_t row_u, const sparse_csr_t *sp, const csr_data_t *x, csr_data_t *result);
void csr_bspmv(csr_index_t row_l, csr_index_t row_u, const sparse_csr_t *sp, const csr_data_t *x, csr_data_t *result, csr_index_t blk_size, csr_index_t nblk);

void csr_coo2csr(sparse_csr_t *out, const csr_index_t *rowidx, const csr_index_t *colidx, const csr_data_t *val, csr_index_t matrix_dim, csr_index_t nnz);

#endif /* _CSR_H */

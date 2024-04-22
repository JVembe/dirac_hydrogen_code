#ifndef GPU_SPARSE_H
#define GPU_SPARSE_H

#include "csr.h"

#if defined(USE_CUDA)
#include "cuda_translate.h"
#elif defined(USE_HIP)
#include "hip_translate.h"
#endif

#if defined USE_CUDA | defined USE_HIP | defined __CUDACC__ || defined __HIPCC__

typedef struct {
    csr_index_t nrows, ncols, nnz;    
    csr_index_t *Ap;
    csr_index_t *Ai;
    csr_data_t  *Ax;
    csr_index_t *Ai_sub_map;
    gpusparseSpMatDescr_t desc;
    gpusparseSpSVDescr_t spsvDescr;
    void *cuBuffer;
} gpu_sparse_csr_t;

typedef struct {
    gpusparseDnVecDescr_t desc;
    gpusparseDnVecDescr_t desc_local;
    csr_index_t dim, local_dim;
    csr_index_t local_offset;
    gpu_complex_t *x;
} gpu_dense_vec_t;

typedef struct {
    gpu_sparse_csr_t *L, *U;
    gpu_dense_vec_t *temp;
} gpu_lu_t;

void gpu_sparse_init();
void gpu_sparse_fini();

void gpu_put_csr(gpu_sparse_csr_t *Agpu, const sparse_csr_t *Ahost);
void gpu_free_csr(gpu_sparse_csr_t *Agpu);

void gpu_put_vec(gpu_dense_vec_t *xgpu, const csr_data_t *xhost, csr_index_t dim);
void gpu_vec_local_part(gpu_dense_vec_t *xgpu, csr_index_t dim, csr_index_t local_offset);
void gpu_get_vec(csr_data_t *xhost, const gpu_dense_vec_t *xgpu);
void gpu_free_vec(gpu_dense_vec_t *xgpu);

void gpu_spmv(gpu_sparse_csr_t *Hfull, gpu_dense_vec_t *x, gpu_dense_vec_t *y, csr_data_t alpha, csr_data_t beta);
void gpu_spmv_local(gpu_sparse_csr_t *Hfull, gpu_dense_vec_t *x, gpu_dense_vec_t *y, csr_data_t alpha, csr_data_t beta);
void gpu_spmv_block(sparse_csr_t H_blk, csr_data_t *x, csr_data_t *y, sparse_csr_t *g);
void gpu_spmb_block_test(sparse_csr_t H_blk, csr_data_t *x, csr_data_t *yfull, sparse_csr_t *g);
void gpu_spmv_test(sparse_csr_t Hfull, csr_data_t *x, csr_data_t *yfull);

void gpu_lu_analyze(gpu_sparse_csr_t *L, gpu_sparse_csr_t *U, gpu_dense_vec_t *x, gpu_dense_vec_t *y, gpu_dense_vec_t *temp);
void gpu_lu_solve(const gpu_sparse_csr_t *L, const gpu_sparse_csr_t *U, const gpu_dense_vec_t *x, gpu_dense_vec_t *y, gpu_dense_vec_t *temp);

#endif

#endif /* GPU_SPARSE_H */

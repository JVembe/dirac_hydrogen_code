#ifndef _CUDA_SOLVER_H
#define _CUDA_SOLVER_H

#ifdef __cplusplus
extern "C" {
#endif
       
    void cuda_init_model_matrices(int nmatrices, const sparse_csr_t *cpu_g,
                                  const sparse_csr_t *cpu_gt, const sparse_csr_t *cpu_h0);
    void cuda_compute_row_col(int lmax, const sparse_csr_t *H,
                              sparse_csr_t *Hfull_blk, sparse_csr_t *Hfull);
    void cuda_compute_timedep_matrices(double h, double dt, csr_data_t *ft, int lmax,
                                       const sparse_csr_t *Hfull_blk, sparse_csr_t *Hfull,
                                       gpu_sparse_csr_t *gpu_Hfull);
    
#ifdef __cplusplus
}
#endif

#endif /* _CUDA_SOLVER_H */

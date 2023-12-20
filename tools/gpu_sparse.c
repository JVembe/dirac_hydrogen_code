#include "gpu_sparse.h"

void tic();
void toc();
void compare_vectors(csr_data_t *v1, csr_data_t *v2, csr_index_t dim);

#if defined USE_CUDA | defined USE_HIP

// Function for performing sparse matrix-vector multiplication on GPU.
void gpu_spmv(sparse_csr_t Hfull, csr_data_t *x, csr_data_t *y) {
    gpusparseHandle_t gpuhandle;

    printf("create device sprase handle..."); fflush(stdout);
    CHECK_GPUSPARSE(gpusparseCreate(&gpuhandle));
    toc();

    // Allocate device memory for CSR matrix and vectors
    csr_index_t *dAp, *dAi;
    csr_data_t  *dAx, *dpx, *dpy;

    printf("allocate device memory..."); fflush(stdout);
    tic();
    CHECK_GPU(gpuMalloc((void**) &dAp, (1+csr_nrows(&Hfull))*sizeof(csr_index_t)));
    CHECK_GPU(gpuMalloc((void**) &dAi, csr_nnz(&Hfull)*sizeof(csr_index_t)));
    CHECK_GPU(gpuMalloc((void**) &dAx, csr_nnz(&Hfull)*sizeof(csr_data_t)));
    CHECK_GPU(gpuMalloc((void**) &dpx, csr_ncols(&Hfull)*sizeof(csr_data_t)));
    CHECK_GPU(gpuMalloc((void**) &dpy, csr_nrows(&Hfull)*sizeof(csr_data_t)));
    CHECK_GPU(gpuMemcpy(dpy, y, csr_nrows(&Hfull)*sizeof(csr_data_t), gpuMemcpyHostToDevice));
    toc();

    // Copy the CSR matrix and vectors from host to device
    printf("Copy data from host to device..."); fflush(stdout);
    tic();
    CHECK_GPU(gpuMemcpy(dAp, Hfull.Ap, (1+csr_nrows(&Hfull))*sizeof(csr_index_t), gpuMemcpyHostToDevice));
    CHECK_GPU(gpuMemcpy(dAi, Hfull.Ai, csr_nnz(&Hfull)*sizeof(csr_index_t), gpuMemcpyHostToDevice));
    CHECK_GPU(gpuMemcpy(dAx, Hfull.Ax, csr_nnz(&Hfull)*sizeof(csr_data_t),  gpuMemcpyHostToDevice));
    CHECK_GPU(gpuMemcpy(dpx, x, csr_ncols(&Hfull)*sizeof(csr_data_t),  gpuMemcpyHostToDevice));
    toc();

    // Create CSR matrix and vectors
    printf("Initialize device sparse structures..."); fflush(stdout);
    tic();
    gpusparseSpMatDescr_t dHfull;
    CHECK_GPUSPARSE(gpusparseCreateCsr(&dHfull, csr_nrows(&Hfull), csr_ncols(&Hfull), csr_nnz(&Hfull),
                                       dAp, dAi, dAx,
                                       GPUSPARSE_INDEX_32I, GPUSPARSE_INDEX_32I, GPUSPARSE_INDEX_BASE_ZERO, GPU_C_64F));

    gpusparseDnVecDescr_t dx, dy;
    CHECK_GPUSPARSE(gpusparseCreateDnVec(&dx, csr_ncols(&Hfull), dpx, GPU_C_64F));
    CHECK_GPUSPARSE(gpusparseCreateDnVec(&dy, csr_nrows(&Hfull), dpy, GPU_C_64F));

    csr_data_t alpha = CMPLX(1,0), beta = CMPLX(0,0);
    size_t bufferSize;

    // Analyze matrix and calculate buffer size for the SpMV operation
    CHECK_GPUSPARSE(gpusparseSpMV_bufferSize(gpuhandle,
                                             GPUSPARSE_OPERATION_NON_TRANSPOSE, (const void*)&alpha, dHfull, dx,
                                             (const void*)&beta, dy,
                                             GPU_C_64F, GPUSPARSE_SPMV_CSR_ALG1, (size_t*)&bufferSize));

    // Allocate buffer for the SpMV operation
    csr_data_t *dbuffer;
    CHECK_GPU(gpuMalloc((void**) &dbuffer, bufferSize*sizeof(csr_data_t)));
    toc();

    // Perform the SpMV operation
    printf("device spmv..."); fflush(stdout);
    tic();
    CHECK_GPUSPARSE(gpusparseSpMV(gpuhandle,
                                  GPUSPARSE_OPERATION_NON_TRANSPOSE, (const void*)&alpha, dHfull, dx,
                                  (const void*)&beta, dy,
                                  GPU_C_64F, GPUSPARSE_SPMV_CSR_ALG1, dbuffer));
    toc();

    // Copy the result back to the host
    printf("copy result from device to host..."); fflush(stdout);
    tic();
    CHECK_GPU(gpuMemcpy(y, dpy, csr_nrows(&Hfull)*sizeof(csr_data_t), gpuMemcpyDeviceToHost));
    toc();

    // Clean up device resources
    CHECK_GPU(gpuFree(dAp));
    CHECK_GPU(gpuFree(dAi));
    CHECK_GPU(gpuFree(dAx));
    CHECK_GPU(gpuFree(dpx));
    CHECK_GPU(gpuFree(dpy));
    CHECK_GPU(gpuFree(dbuffer));
    CHECK_GPUSPARSE(gpusparseDestroy(gpuhandle));

    // Not sure if needed but keeping for debugging purposes.
    //   gpuDeviceSynchronize();
}

void gpu_spmv_block(sparse_csr_t H_blk, csr_data_t *x, csr_data_t *y, sparse_csr_t *g) {

    sparse_csr_t submatrix;
    csr_copy(&submatrix, g);

    int blkdim = csr_nrows(&g[0]);

    csr_index_t row, col, colp;

    for(row = 0; row < csr_nrows(&H_blk); row++){
        // for non-zero blocks in each row
        for(colp = H_blk.Ap[row]; colp < H_blk.Ap[row+1]; colp++){

            col = H_blk.Ai[colp];
            csr_block_link(&submatrix, &H_blk, row, col);
            csr_data_t *xin;
            xin  = x + col*blkdim;
            csr_data_t *yout_gpu;
            yout_gpu = y + row*blkdim;

            gpu_spmv(submatrix, xin, yout_gpu);

            // remember that at this stage xin and yout are renumbered wrt. the original node numbering
        }
    }

}

void gpu_spmb_block_test(sparse_csr_t H_blk, csr_data_t *x, csr_data_t *yfull, sparse_csr_t *g) {

    // allocate y vector for GPU SpMV
    csr_data_t *y_gpu;
    y_gpu = (csr_data_t *)calloc(csr_nrows(&H_blk), sizeof(csr_data_t));

    // printf(" - H dim: %d\n", csr_dim(&H[0]));
    // printf(" - Hall dim: %d\n", csr_dim(&Hall));
    // printf(" - Hall nnz: %d\n", csr_nnz(&Hall));
    // printf(" - Hfull_blk dim: %d\n", csr_dim(&Hfull_blk));
    // printf(" - Hfull_blk nnz: %d\n", csr_nnz(&Hfull_blk));

    gpu_spmv_block(H_blk, x, y_gpu, g);

    compare_vectors(yfull, y_gpu, csr_nrows(&H_blk));
}

// Function for testing the result of the sparse matrix-vector multiplication.
void gpu_spmv_test(sparse_csr_t Hfull, csr_data_t *x, csr_data_t *yfull) {

    // Print device info
    int deviceCount;
    CHECK_GPU(gpuGetDeviceCount(&deviceCount));
    printf("Device count: %d\n", deviceCount);

    // Allocate vector for GPU SpMV result
    csr_data_t *gpu_result;
    gpu_result = (csr_data_t *)calloc(csr_nrows(&Hfull), sizeof(csr_data_t));

    // Perform the matrix-vector multiplication on the GPU
    gpu_spmv(Hfull, x, gpu_result);

    // Validate - compare yfull and gpu results
    compare_vectors(yfull, gpu_result, csr_nrows(&Hfull));

    // Clean up host resources
    free(gpu_result);
}

#endif

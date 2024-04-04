#include "gpu_sparse.h"
#include "../src/tictoc.h"

void tic();
void toc();
void compare_vectors(csr_data_t *v1, csr_data_t *v2, csr_index_t dim);

#if defined USE_CUDA | defined USE_HIP

static gpusparseHandle_t sparseHandle;

extern int rank, nranks;

void gpu_sparse_init()
{ 
    // Print device info
    int deviceCount;
    CHECK_GPU(gpuGetDeviceCount(&deviceCount));
    printf("Device count: %d\n", deviceCount);
    CHECK_GPUSPARSE(gpusparseCreate(&sparseHandle));   
}

void gpu_sparse_fini()
{ 
    CHECK_GPUSPARSE(gpusparseDestroy(sparseHandle));
}

// copy sparse matrix from host to device
void gpu_put_csr(gpu_sparse_csr_t *Agpu, const sparse_csr_t *Ahost)
{
    // Allocate device memory for CSR matrix and vectors
    csr_index_t *dAp, *dAi;
    csr_data_t  *dAx;

    CHECK_GPU(gpuMalloc((void**) &dAp, (1+csr_nrows(Ahost))*sizeof(csr_index_t)));
    CHECK_GPU(gpuMalloc((void**) &dAi, csr_nnz(Ahost)*sizeof(csr_index_t)));
    CHECK_GPU(gpuMalloc((void**) &dAx, csr_nnz(Ahost)*sizeof(csr_data_t)));

    // Copy the CSR matrix and vectors from host to device
    CHECK_GPU(gpuMemcpy(dAp, Ahost->Ap, (1+csr_nrows(Ahost))*sizeof(csr_index_t), gpuMemcpyHostToDevice));
    CHECK_GPU(gpuMemcpy(dAi, Ahost->Ai, csr_nnz(Ahost)*sizeof(csr_index_t), gpuMemcpyHostToDevice));
    CHECK_GPU(gpuMemcpy(dAx, Ahost->Ax, csr_nnz(Ahost)*sizeof(csr_data_t),  gpuMemcpyHostToDevice));

    // Create CSR matrix and vectors
    gpusparseSpMatDescr_t Adevice;
    CHECK_GPUSPARSE(gpusparseCreateCsr(&Adevice, csr_nrows(Ahost), csr_ncols(Ahost), csr_nnz(Ahost),
                                       dAp, dAi, dAx,
                                       GPUSPARSE_INDEX_32I, GPUSPARSE_INDEX_32I, GPUSPARSE_INDEX_BASE_ZERO, GPU_C_64F));
    Agpu->desc = Adevice;
    Agpu->Ap = dAp;
    Agpu->Ai = dAi;
    Agpu->Ax = dAx;
    Agpu->nnz = csr_nnz(Ahost);
    Agpu->nrows = csr_nrows(Ahost);
    Agpu->ncols = csr_ncols(Ahost);
    Agpu->cuBuffer = NULL;
}

void gpu_put_vec(gpu_dense_vec_t *xgpu, const csr_data_t *xhost, csr_index_t dim)
{
    CHECK_GPU(gpuMalloc((void**) &xgpu->x, dim*sizeof(csr_data_t)));
    if(xhost) {
        CHECK_GPU(gpuMemcpy(xgpu->x, xhost, dim*sizeof(csr_data_t), gpuMemcpyHostToDevice));
    } else {
        CHECK_GPU(gpuMemset(xgpu->x, 0, dim*sizeof(csr_data_t)));
    }
    CHECK_GPUSPARSE(gpusparseCreateDnVec(&xgpu->desc, dim, xgpu->x, GPU_C_64F));
    xgpu->dim = dim;
    xgpu->desc_local = NULL;
}

void gpu_vec_local_part(gpu_dense_vec_t *xgpu, csr_index_t dim, csr_index_t local_offset)
{
    CHECK_GPUSPARSE(gpusparseCreateDnVec(&xgpu->desc_local, dim, xgpu->x + local_offset, GPU_C_64F));
    xgpu->local_offset = local_offset;
}

void gpu_get_vec(csr_data_t *xhost, const gpu_dense_vec_t *xgpu)
{
    CHECK_GPU(gpuMemcpy(xhost, xgpu->x, xgpu->dim*sizeof(csr_data_t), gpuMemcpyDeviceToHost));
}

void gpu_lu_analyze(gpu_sparse_csr_t *L, gpu_sparse_csr_t *U, gpu_dense_vec_t *x, gpu_dense_vec_t *y) {
    csr_data_t alpha = 1;
    size_t bufferSize;
    int value;

    cusparseSpSV_createDescr(&L->spsvDescr);
    value = CUSPARSE_FILL_MODE_LOWER;
    CHECK_GPU(cusparseSpMatSetAttribute(L->desc, CUSPARSE_SPMAT_FILL_MODE, &value, sizeof(value)));
    value = CUSPARSE_DIAG_TYPE_NON_UNIT;
    CHECK_GPU(cusparseSpMatSetAttribute(L->desc, CUSPARSE_SPMAT_DIAG_TYPE, &value, sizeof(value)));
    CHECK_GPU(cusparseSpSV_bufferSize(sparseHandle,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      &alpha,
                                      L->desc,
                                      x->desc,
                                      y->desc,
                                      CUDA_C_64F,
                                      CUSPARSE_SPSV_ALG_DEFAULT,
                                      L->spsvDescr,
                                      &bufferSize));
    CHECK_GPU(gpuMalloc((void**) &L->cuBuffer, bufferSize));
    CHECK_GPU(cusparseSpSV_analysis(sparseHandle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha,
                                    L->desc,
                                    x->desc,
                                    y->desc,
                                    CUDA_C_64F,
                                    CUSPARSE_SPSV_ALG_DEFAULT,
                                    L->spsvDescr,
                                    L->cuBuffer));

    cusparseSpSV_createDescr(&U->spsvDescr);
    value = CUSPARSE_FILL_MODE_UPPER;
    CHECK_GPU(cusparseSpMatSetAttribute(U->desc, CUSPARSE_SPMAT_FILL_MODE, &value, sizeof(value)));
    value = CUSPARSE_DIAG_TYPE_NON_UNIT;
    CHECK_GPU(cusparseSpMatSetAttribute(U->desc, CUSPARSE_SPMAT_DIAG_TYPE, &value, sizeof(value)));
    CHECK_GPU(cusparseSpSV_bufferSize(sparseHandle,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      &alpha,
                                      U->desc,
                                      x->desc,
                                      y->desc,
                                      CUDA_C_64F,
                                      CUSPARSE_SPSV_ALG_DEFAULT,
                                      U->spsvDescr,
                                      &bufferSize));
    CHECK_GPU(gpuMalloc((void**) &U->cuBuffer, bufferSize));
    CHECK_GPU(cusparseSpSV_analysis(sparseHandle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha,
                                    U->desc,
                                    x->desc,
                                    y->desc,
                                    CUDA_C_64F,
                                    CUSPARSE_SPSV_ALG_DEFAULT,
                                    U->spsvDescr,
                                    U->cuBuffer));
}

void gpu_lu_solve(gpu_sparse_csr_t *L, gpu_sparse_csr_t *U, gpu_dense_vec_t *x, gpu_dense_vec_t *y, gpu_dense_vec_t *temp) {
    csr_data_t alpha = 1;
    CHECK_GPU(cusparseSpSV_solve(sparseHandle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha,
                                 L->desc,
                                 x->desc,
                                 temp->desc,
                                 CUDA_C_64F,
                                 CUSPARSE_SPSV_ALG_DEFAULT,
                                 L->spsvDescr));
    CHECK_GPU(cusparseSpSV_solve(sparseHandle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha,
                                 U->desc,
                                 temp->desc,
                                 y->desc,
                                 CUDA_C_64F,
                                 CUSPARSE_SPSV_ALG_DEFAULT,
                                 U->spsvDescr));
}


// Function for performing sparse matrix-vector multiplication on GPU.
void gpu_spmv(gpu_sparse_csr_t *Hfull, gpu_dense_vec_t *x, gpu_dense_vec_t *y) {
    csr_data_t alpha = CMPLX(1,0), beta = CMPLX(1,0);
    if(NULL == Hfull->cuBuffer){
        // Analyze matrix and calculate buffer size for the SpMV operation
        size_t bufferSize;
        CHECK_GPUSPARSE(gpusparseSpMV_bufferSize(sparseHandle,
                                                 GPUSPARSE_OPERATION_NON_TRANSPOSE, (const void*)&alpha, Hfull->desc, x->desc,
                                                 (const void*)&beta, y->desc,
                                                 GPU_C_64F, GPUSPARSE_SPMV_CSR_ALG1, (size_t*)&bufferSize));

        // Allocate buffer for the SpMV operation
        CHECK_GPU(gpuMalloc((void**) &Hfull->cuBuffer, bufferSize*sizeof(csr_data_t)));
    }
    
    CHECK_GPUSPARSE(gpusparseSpMV(sparseHandle,
                                  GPUSPARSE_OPERATION_NON_TRANSPOSE, (const void*)&alpha, Hfull->desc, x->desc,
                                  (const void*)&beta, y->desc,
                                  GPU_C_64F, GPUSPARSE_SPMV_CSR_ALG1, Hfull->cuBuffer));
}


// Function for performing sparse matrix-vector multiplication on GPU: local vector part
void gpu_spmv_local(gpu_sparse_csr_t *Hfull, gpu_dense_vec_t *x, gpu_dense_vec_t *y) {
    csr_data_t alpha = CMPLX(1,0), beta = CMPLX(1,0);
    if(NULL == Hfull->cuBuffer){
        // Analyze matrix and calculate buffer size for the SpMV operation
        size_t bufferSize;
        CHECK_GPUSPARSE(gpusparseSpMV_bufferSize(sparseHandle,
                                                 GPUSPARSE_OPERATION_NON_TRANSPOSE, (const void*)&alpha, Hfull->desc, x->desc_local,
                                                 (const void*)&beta, y->desc,
                                                 GPU_C_64F, GPUSPARSE_SPMV_CSR_ALG1, (size_t*)&bufferSize));

        // Allocate buffer for the SpMV operation
        CHECK_GPU(gpuMalloc((void**) &Hfull->cuBuffer, bufferSize*sizeof(csr_data_t)));
    }

    CHECK_GPUSPARSE(gpusparseSpMV(sparseHandle,
                                  GPUSPARSE_OPERATION_NON_TRANSPOSE, (const void*)&alpha, Hfull->desc, x->desc_local,
                                  (const void*)&beta, y->desc,
                                  GPU_C_64F, GPUSPARSE_SPMV_CSR_ALG1, Hfull->cuBuffer));
}


// Function for performing sparse matrix-vector multiplication on GPU.
void gpu_spmv_old(sparse_csr_t Hfull, csr_data_t *x, csr_data_t *y) {

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

    csr_data_t alpha = CMPLX(1,0), beta = CMPLX(1,0);
    size_t bufferSize;

    // Analyze matrix and calculate buffer size for the SpMV operation
    CHECK_GPUSPARSE(gpusparseSpMV_bufferSize(sparseHandle,
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
    CHECK_GPUSPARSE(gpusparseSpMV(sparseHandle,
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

            // gpu_spmv(submatrix, xin, yout_gpu);
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

    gpu_sparse_csr_t gpuHfull;
    gpu_put_csr(&gpuHfull, &Hfull);

    gpu_dense_vec_t xgpu, ygpu;
    gpu_put_vec(&xgpu, x, csr_ncols(&Hfull));
    gpu_put_vec(&ygpu, NULL, csr_nrows(&Hfull));

    /* // Analyze matrix and calculate buffer size for the SpMV operation */
    /* csr_data_t alpha = CMPLX(1,0), beta = CMPLX(1,0); */
    /* size_t bufferSize; */
    /* CHECK_GPUSPARSE(gpusparseSpMV_bufferSize(sparseHandle, */
    /*                                          GPUSPARSE_OPERATION_NON_TRANSPOSE, (const void*)&alpha, gpuHfull.desc, xgpu.desc, */
    /*                                          (const void*)&beta, ygpu.desc, */
    /*                                          GPU_C_64F, GPUSPARSE_SPMV_CSR_ALG1, (size_t*)&bufferSize)); */

    /* // Allocate buffer for the SpMV operation */
    /* CHECK_GPU(gpuMalloc((void**) &gpuHfull.cuBuffer, bufferSize*sizeof(csr_data_t))); */

    // Allocate vector for GPU SpMV result
    csr_data_t *gpu_result;
    gpu_result = (csr_data_t *)calloc(csr_nrows(&Hfull), sizeof(csr_data_t));

    // Perform the matrix-vector multiplication on the GPU
    csr_init_communication(&Hfull, xgpu.x, rank, nranks);

    PRINTF0("Device comm: "); tic();
    csr_comm(&Hfull, rank, nranks);
    toc();

    PRINTF0("Device spmv: "); tic();
    gpu_spmv(&gpuHfull, &xgpu, &ygpu);
    toc();

    gpu_get_vec(yfull, &ygpu);

    // Clean up host resources
    free(gpu_result);
}

#endif

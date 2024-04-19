#include "gpu_sparse.h"
#include "../src/tictoc.h"

void compare_vectors(csr_data_t *v1, csr_data_t *v2, csr_index_t dim);

#if defined USE_CUDA | defined USE_HIP

static gpusparseHandle_t sparseHandle;

extern int rank, nranks;

void gpu_sparse_init()
{
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    // Print device info
    int device;
    gpuDeviceProp prop;
    CHECK_GPU(gpuGetDevice(&device));
    CHECK_GPU(gpuGetDeviceProperties(&prop, device));
    printf("rank %d: device %s pciBusID %d\n", rank, prop.name, prop.pciBusID);
    CHECK_GPUSPARSE(gpusparseCreate(&sparseHandle));
    fflush(stdout);
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
}

void gpu_sparse_fini()
{
    CHECK_GPUSPARSE(gpusparseDestroy(sparseHandle));
}

size_t total_bytes = 0;

// copy sparse matrix from host to device
void gpu_put_csr(gpu_sparse_csr_t *Agpu, const sparse_csr_t *Ahost)
{
    csr_data_t  *dAx;
    size_t bytes = 0;

    if(NULL != Agpu->Ax){
        dAx = Agpu->Ax;
    } else {
        // Allocate device memory for CSR matrix and vectors
        csr_index_t *dAp, *dAi;
        CHECK_GPU(gpuMalloc((void**) &dAp, (1+csr_nrows(Ahost))*sizeof(csr_index_t)));
        CHECK_GPU(gpuMalloc((void**) &dAi, csr_nnz(Ahost)*sizeof(csr_index_t)));
        CHECK_GPU(gpuMalloc((void**) &dAx, csr_nnz(Ahost)*sizeof(csr_data_t)));
        bytes += (1+csr_nrows(Ahost))*sizeof(csr_index_t);
        bytes += csr_nnz(Ahost)*sizeof(csr_index_t);
        bytes += csr_nnz(Ahost)*sizeof(csr_data_t);

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

        CHECK_GPU(gpuMemcpy(dAp, Ahost->Ap, (1+csr_nrows(Ahost))*sizeof(csr_index_t), gpuMemcpyHostToDevice));
        CHECK_GPU(gpuMemcpy(dAi, Ahost->Ai, csr_nnz(Ahost)*sizeof(csr_index_t), gpuMemcpyHostToDevice));

        Agpu->Ai_sub_map = NULL;
        if(Ahost->Ai_sub_map) {
            csr_index_t *dAi_sub_map;
            CHECK_GPU(gpuMalloc((void**) &dAi_sub_map, csr_nnz(Ahost)*sizeof(csr_index_t)));
            CHECK_GPU(gpuMemcpy(dAi_sub_map, Ahost->Ai_sub_map, csr_nnz(Ahost)*sizeof(csr_index_t), gpuMemcpyHostToDevice));
            Agpu->Ai_sub_map = dAi_sub_map;
            bytes += csr_nnz(Ahost)*sizeof(csr_index_t);
        }

        total_bytes += bytes;
        PDEBUG("gpu_put_csr: allocated %li bytes on GPU (total %li)\n", bytes, total_bytes);
    }

    // Copy the CSR matrix values from host to device
    CHECK_GPU(gpuMemcpy(dAx, Ahost->Ax, csr_nnz(Ahost)*sizeof(csr_data_t),  gpuMemcpyHostToDevice));
}

void gpu_free_csr(gpu_sparse_csr_t *Agpu)
{
    CHECK_GPU(gpusparseDestroySpMat(Agpu->desc));
    CHECK_GPU(gpuFree(Agpu->Ap));
    CHECK_GPU(gpuFree(Agpu->Ai));
    CHECK_GPU(gpuFree(Agpu->Ax));
    CHECK_GPU(gpuFree(Agpu->cuBuffer));
}

void gpu_put_vec(gpu_dense_vec_t *xgpu, const csr_data_t *xhost, csr_index_t dim)
{
    size_t bytes = dim*sizeof(csr_data_t);
    CHECK_GPU(gpuMalloc((void**) &xgpu->x, dim*sizeof(csr_data_t)));
    total_bytes += bytes;
    PDEBUG("gpu_put_vec: allocated %li bytes on GPU (total %li)\n", bytes, total_bytes);

    if(xhost) {
        CHECK_GPU(gpuMemcpy(xgpu->x, xhost, dim*sizeof(csr_data_t), gpuMemcpyHostToDevice));
    } else {
        CHECK_GPU(gpuMemset(xgpu->x, 0, dim*sizeof(csr_data_t)));
    }
    CHECK_GPUSPARSE(gpusparseCreateDnVec(&xgpu->desc, dim, xgpu->x, GPU_C_64F));
    xgpu->dim = dim;
    xgpu->desc_local = NULL;
    xgpu->local_offset = 0;
}

void gpu_vec_local_part(gpu_dense_vec_t *xgpu, csr_index_t dim, csr_index_t local_offset)
{
    CHECK_GPUSPARSE(gpusparseCreateDnVec(&xgpu->desc_local, dim, xgpu->x + local_offset, GPU_C_64F));
    xgpu->local_offset = local_offset;
    xgpu->local_dim = dim;
}

void gpu_get_vec(csr_data_t *xhost, const gpu_dense_vec_t *xgpu)
{
    if(xgpu->local_dim){
        CHECK_GPU(gpuMemcpy(xhost, xgpu->x + xgpu->local_offset, xgpu->local_dim*sizeof(csr_data_t), gpuMemcpyDeviceToHost));
    } else {
        CHECK_GPU(gpuMemcpy(xhost, xgpu->x, xgpu->dim*sizeof(csr_data_t), gpuMemcpyDeviceToHost));
    }
}

void gpu_free_vec(gpu_dense_vec_t *xgpu)
{
    CHECK_GPU(gpuFree(xgpu->x));
    CHECK_GPU(gpusparseDestroyDnVec(xgpu->desc));
    if(xgpu->desc_local)
        CHECK_GPU(gpusparseDestroyDnVec(xgpu->desc_local));
}

void gpu_lu_analyze(gpu_sparse_csr_t *L, gpu_sparse_csr_t *U, gpu_dense_vec_t *x, gpu_dense_vec_t *y, gpu_dense_vec_t *temp)
{
    csr_data_t alpha = 1;
    size_t bufferSize;
    int value;
    size_t bytes;

    gpusparseSpSV_createDescr(&L->spsvDescr);
    value = GPUSPARSE_FILL_MODE_LOWER;
    CHECK_GPU(gpusparseSpMatSetAttribute(L->desc, GPUSPARSE_SPMAT_FILL_MODE, &value, sizeof(value)));
    value = GPUSPARSE_DIAG_TYPE_NON_UNIT;
    CHECK_GPU(gpusparseSpMatSetAttribute(L->desc, GPUSPARSE_SPMAT_DIAG_TYPE, &value, sizeof(value)));
    CHECK_GPU(gpusparseSpSV_bufferSize(sparseHandle,
                                      GPUSPARSE_OPERATION_NON_TRANSPOSE,
                                      &alpha,
                                      L->desc,
                                      x->desc,
                                      temp->desc,
                                      GPU_C_64F,
                                      GPUSPARSE_SPSV_ALG_DEFAULT,
                                      L->spsvDescr,
                                      &bufferSize));
    CHECK_GPU(gpuMalloc((void**) &L->cuBuffer, bufferSize));
    bytes += bufferSize;

    CHECK_GPU(gpusparseSpSV_analysis(sparseHandle,
                                    GPUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha,
                                    L->desc,
                                    x->desc,
                                    temp->desc,
                                    GPU_C_64F,
                                    GPUSPARSE_SPSV_ALG_DEFAULT,
                                    L->spsvDescr,
                                    L->cuBuffer));

    gpusparseSpSV_createDescr(&U->spsvDescr);
    value = GPUSPARSE_FILL_MODE_UPPER;
    CHECK_GPU(gpusparseSpMatSetAttribute(U->desc, GPUSPARSE_SPMAT_FILL_MODE, &value, sizeof(value)));
    value = GPUSPARSE_DIAG_TYPE_NON_UNIT;
    CHECK_GPU(gpusparseSpMatSetAttribute(U->desc, GPUSPARSE_SPMAT_DIAG_TYPE, &value, sizeof(value)));

    gpusparseDnVecDescr_t outdesc = y->desc;
    if(y->desc_local) outdesc = y->desc_local;

    CHECK_GPU(gpusparseSpSV_bufferSize(sparseHandle,
                                      GPUSPARSE_OPERATION_NON_TRANSPOSE,
                                      &alpha,
                                      U->desc,
                                      x->desc,
                                      outdesc,
                                      GPU_C_64F,
                                      GPUSPARSE_SPSV_ALG_DEFAULT,
                                      U->spsvDescr,
                                      &bufferSize));
    CHECK_GPU(gpuMalloc((void**) &U->cuBuffer, bufferSize));
    bytes += bufferSize;

    CHECK_GPU(gpusparseSpSV_analysis(sparseHandle,
                                    GPUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha,
                                    U->desc,
                                    x->desc,
                                    outdesc,
                                    GPU_C_64F,
                                    GPUSPARSE_SPSV_ALG_DEFAULT,
                                    U->spsvDescr,
                                    U->cuBuffer));

    total_bytes += bytes;
    PDEBUG("gpu_lu_analyze: allocated %li bytes on GPU (total %li)\n", bytes, total_bytes);
}

void gpu_lu_solve(const gpu_sparse_csr_t *L, const gpu_sparse_csr_t *U, const gpu_dense_vec_t *x,
                  gpu_dense_vec_t *y, gpu_dense_vec_t *temp)
{
    csr_data_t alpha = 1;
    (gpusparseSpSV_solve(sparseHandle,
                                 GPUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha,
                                 L->desc,
                                 x->desc,
                                 temp->desc,
                                 GPU_C_64F,
                                 GPUSPARSE_SPSV_ALG_DEFAULT,
                                 L->spsvDescr
#ifdef USE_HIP
			         ,L->cuBuffer
#endif
			 ));

    gpusparseDnVecDescr_t outdesc = y->desc;
    if(y->desc_local) outdesc = y->desc_local;

    (gpusparseSpSV_solve(sparseHandle,
				 GPUSPARSE_OPERATION_NON_TRANSPOSE,
				 &alpha,
				 U->desc,
				 temp->desc,
				 outdesc,
				 GPU_C_64F,
				 GPUSPARSE_SPSV_ALG_DEFAULT,
				 U->spsvDescr
#ifdef USE_HIP
			         ,U->cuBuffer
#endif
			 ));
}


// Function for performing sparse matrix-vector multiplication on GPU.
void gpu_spmv(gpu_sparse_csr_t *Hfull, gpu_dense_vec_t *x, gpu_dense_vec_t *y, csr_data_t alpha, csr_data_t beta)
{
    if(NULL == Hfull->cuBuffer){
        // Analyze matrix and calculate buffer size for the SpMV operation
        size_t bufferSize;
        CHECK_GPUSPARSE(gpusparseSpMV_bufferSize(sparseHandle,
                                                 GPUSPARSE_OPERATION_NON_TRANSPOSE, (const void*)&alpha, Hfull->desc, x->desc,
                                                 (const void*)&beta, y->desc,
                                                 GPU_C_64F, GPUSPARSE_SPMV_CSR_ALG1, (size_t*)&bufferSize));

        // Allocate buffer for the SpMV operation
        CHECK_GPU(gpuMalloc((void**) &Hfull->cuBuffer, bufferSize));
        size_t bytes = bufferSize;
        total_bytes += bytes;
        PDEBUG("gpu_spmv: allocated %li bytes on GPU (total %li)\n", bytes, total_bytes);
    }

    CHECK_GPUSPARSE(gpusparseSpMV(sparseHandle,
                                  GPUSPARSE_OPERATION_NON_TRANSPOSE, (const void*)&alpha, Hfull->desc, x->desc,
                                  (const void*)&beta, y->desc,
                                  GPU_C_64F, GPUSPARSE_SPMV_CSR_ALG1, Hfull->cuBuffer));
}


// Function for performing sparse matrix-vector multiplication on GPU: local vector part
void gpu_spmv_local(gpu_sparse_csr_t *Hfull, gpu_dense_vec_t *x, gpu_dense_vec_t *y, csr_data_t alpha, csr_data_t beta)
{
    if(NULL == Hfull->cuBuffer){
        // Analyze matrix and calculate buffer size for the SpMV operation
        size_t bufferSize;
        CHECK_GPUSPARSE(gpusparseSpMV_bufferSize(sparseHandle,
                                                 GPUSPARSE_OPERATION_NON_TRANSPOSE, (const void*)&alpha, Hfull->desc, x->desc_local,
                                                 (const void*)&beta, y->desc,
                                                 GPU_C_64F, GPUSPARSE_SPMV_CSR_ALG1, (size_t*)&bufferSize));

        // Allocate buffer for the SpMV operation
        CHECK_GPU(gpuMalloc((void**) &Hfull->cuBuffer, bufferSize));
        size_t bytes = bufferSize;
        total_bytes += bytes;
        PDEBUG("gpu_spmv_local: allocated %li bytes on GPU (total %li)\n", bytes, total_bytes);
    }

    CHECK_GPUSPARSE(gpusparseSpMV(sparseHandle,
                                  GPUSPARSE_OPERATION_NON_TRANSPOSE, (const void*)&alpha, Hfull->desc, x->desc_local,
                                  (const void*)&beta, y->desc,
                                  GPU_C_64F, GPUSPARSE_SPMV_CSR_ALG1, Hfull->cuBuffer));
}

void gpu_spmv_block(sparse_csr_t H_blk, csr_data_t *x, csr_data_t *y, sparse_csr_t *g)
{

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

void gpu_spmb_block_test(sparse_csr_t H_blk, csr_data_t *x, csr_data_t *yfull, sparse_csr_t *g)
{

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
void gpu_spmv_test(sparse_csr_t Hfull, csr_data_t *x, csr_data_t *yfull)
{

    gpu_sparse_csr_t gpuHfull = {};
    gpu_put_csr(&gpuHfull, &Hfull);

    gpu_dense_vec_t xgpu = {}, ygpu = {};
    gpu_put_vec(&xgpu, x, csr_ncols(&Hfull));
    gpu_put_vec(&ygpu, NULL, csr_nrows(&Hfull));

    // Perform the matrix-vector multiplication on the GPU

    PRINTF0("Device comm: "); tic();
    csr_init_communication(&Hfull, (csr_data_t*)xgpu.x, rank, nranks);
    csr_comm(&Hfull, rank, nranks);
    toc();

    PRINTF0("Device spmv: "); tic();
    gpu_spmv(&gpuHfull, &xgpu, &ygpu, CMPLX(1,0), CMPLX(1,0));
    toc();

    gpu_get_vec(yfull, &ygpu);

    // Clean up host resources
    gpu_free_csr(&gpuHfull);
    gpu_free_vec(&xgpu);
    gpu_free_vec(&ygpu);
}

#endif

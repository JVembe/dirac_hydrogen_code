/*
  This program reads base matrices saved in CSR format.
  To produce those matrices, run the following:

  # this saves the base g matrices
  dumpCouplings -g ../input_example.json

  # this saves the Hamiltonian matrices
  dumpCouplings ../input_example.json

  # this combines the invididual Hamiltonian matrices
  # into the full couplings matrix structure, and computes
  # domain decomposition / node renumbering
  python3 main.py

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

#define max(a,b) ((a)>(b)?(a):(b))


// Define the CMPLX macro manually if it's not already defined
#ifndef CMPLX
#define CMPLX(x, y) ((double complex)((double)(x) + _Complex_I * (double)(y)))
#endif

#include "csr.h"
#include "../src/tictoc.h"


#if defined USE_CUDA
#include <cuda_runtime_api.h>
#include <cusparse.h>

// Error types and constants
#define gpuError_t                                cudaError_t
#define gpuSuccess                                cudaSuccess

// Device management
#define gpuGetDeviceCount                         cudaGetDeviceCount
#define gpuSetDevice                              cudaSetDevice
#define gpuDeviceSynchronize                      cudaDeviceSynchronize

// Memory management
#define gpuMalloc                                 cudaMalloc
#define gpuFree                                   cudaFree
#define gpuMemcpy                                 cudaMemcpy
#define gpuMemcpyHostToDevice                     cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost                     cudaMemcpyDeviceToHost

// Error handling
#define gpuGetLastError                           cudaGetLastError
#define gpuGetErrorString                         cudaGetErrorString

// Stream management
#define gpuStream_t                               cudaStream_t
#define gpuStreamCreate                           cudaStreamCreate
#define gpuStreamDestroy                          cudaStreamDestroy
#define gpuStreamSynchronize                      cudaStreamSynchronize

// Sparse library handles and operations
#define gpusparseHandle_t                         cusparseHandle_t
#define gpusparseCreateHandle                     cusparseCreateHandle
#define gpusparseDestroy                          cusparseDestroy
#define gpusparseCreate                           cusparseCreate

// Sparse matrix-vector (SpMV) specific
#define gpusparseSpMatDescr_t                     cusparseSpMatDescr_t
#define gpusparseDnVecDescr_t                     cusparseDnVecDescr_t
#define gpusparseCreateCsr                        cusparseCreateCsr
#define gpusparseCreateDnVec                      cusparseCreateDnVec
#define gpusparseSpMV_bufferSize                  cusparseSpMV_bufferSize
#define gpusparseSpMV                             cusparseSpMV

// Sparse library constants
#define GPUSPARSE_INDEX_32I                       CUSPARSE_INDEX_32I
#define GPUSPARSE_INDEX_BASE_ZERO                 CUSPARSE_INDEX_BASE_ZERO
#define GPU_C_64F                                 CUDA_C_64F
#define GPUSPARSE_OPERATION_NON_TRANSPOSE         CUSPARSE_OPERATION_NON_TRANSPOSE
#define GPUSPARSE_SPMV_CSR_ALG1                   CUSPARSE_SPMV_CSR_ALG1

#define CHECK_GPU(func) {						\
    cudaError_t status = (func);					\
    if (status != cudaSuccess) {					\
      printf("CUDA API failed at line %d error: %s\n", __LINE__,	\
	     cudaGetErrorString(status));				\
      exit(1);								\
    }									\
  }

#define CHECK_GPUSPARSE(func) {						\
    cusparseStatus_t status = (func);					\
    if (status != CUSPARSE_STATUS_SUCCESS) {				\
      printf("CUSPARSE API failed at line %d error %d.\n", __LINE__,	\
	     status);							\
      exit(1);								\
    }									\
  }

#elif defined(USE_HIP)

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipsparse.h>
#include <hip/hip_complex.h>

// Error types and constants
#define gpuError_t                                hipError_t
#define gpuSuccess                                hipSuccess

// Device management
#define gpuGetDeviceCount                         hipGetDeviceCount
#define gpuSetDevice                              hipSetDevice
#define gpuDeviceSynchronize                      hipDeviceSynchronize

// Memory management
#define gpuMalloc                                 hipMalloc
#define gpuFree                                   hipFree
#define gpuMemcpy                                 hipMemcpy
#define gpuMemset                                 hipMemset
#define gpuMemcpyHostToDevice                     hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost                     hipMemcpyDeviceToHost

// Error handling
#define gpuGetLastError                           hipGetLastError
#define gpuGetErrorString                         hipGetErrorString

// Stream management
#define gpuStream_t                               hipStream_t
#define gpuStreamCreate                           hipStreamCreate
#define gpuStreamDestroy                          hipStreamDestroy
#define gpuStreamSynchronize                      hipStreamSynchronize

// Sparse library handles and operations
#define gpusparseHandle_t                         hipsparseHandle_t
#define gpusparseCreateHandle                     hipsparseCreateHandle
#define gpusparseDestroy                          hipsparseDestroy
#define gpusparseCreate                           hipsparseCreate

// Sparse matrix-vector (SpMV) specific
#define gpusparseSpMatDescr_t                     hipsparseSpMatDescr_t
#define gpusparseDnVecDescr_t                     hipsparseDnVecDescr_t
#define gpusparseCreateCsr                        hipsparseCreateCsr
#define gpusparseCreateDnVec                      hipsparseCreateDnVec
#define gpusparseSpMV_bufferSize                  hipsparseSpMV_bufferSize
#define gpusparseSpMV                             hipsparseSpMV

// Sparse library constants
#define GPUSPARSE_INDEX_32I                       HIPSPARSE_INDEX_32I
#define GPUSPARSE_INDEX_BASE_ZERO                 HIPSPARSE_INDEX_BASE_ZERO
#define GPU_C_64F                                 HIP_C_64F
#define GPUSPARSE_OPERATION_NON_TRANSPOSE         HIPSPARSE_OPERATION_NON_TRANSPOSE
#define GPUSPARSE_SPMV_CSR_ALG1                   HIPSPARSE_SPMV_CSR_ALG1

#define CHECK_GPU(func) {						\
    hipError_t status = (func);					\
    if (status != hipSuccess) {					\
      printf("HIP API failed at line %d error: %s\n", __LINE__,	\
	     hipGetErrorString(status));				\
      exit(1);								\
    }									\
  }

#define CHECK_GPUSPARSE(func) {						\
    hipsparseStatus_t status = (func);					\
    if (status != HIPSPARSE_STATUS_SUCCESS) {				\
      printf("HIPSPARSE API failed at line %d error %d.\n", __LINE__,	\
	     status);							\
      exit(1);								\
        }									\
  }

#endif


// from spnrbasis.cpp
int ik(int i) {
    int ii = i/4;
    int abskappa = (int)(0.5*(sqrt(8.0*ii+1.0) - 1.0)) + 1;
    int sgnmod = max(4,abskappa*4);
    double sgnfloor = 2*abskappa * (abskappa - 1);
    int sgnkappa = ((i-sgnfloor)/sgnmod >= 0.5) - ((i-sgnfloor)/sgnmod < 0.5);
    return abskappa * sgnkappa;
}

double imu(int i) {
    int abskappa = abs(ik(i));
    int kmod = max(2,2*abskappa);
    double mu = i%kmod - abskappa + 0.5;
    return mu;
}

void compare_vectors(csr_data_t *v1, csr_data_t *v2, csr_index_t dim)
{
    // validate - compare yblk and yfull results
    for(int i=0; i<dim; i++) {
        if(isnan(cimag(v1[i]+v2[i])) || isnan(creal(v1[i]+v2[i]))) {
            printf("nan in vector!\n");
            continue;
        }
        if(fabs(cimag(v1[i]-v2[i]))>1e-10) printf("%e *i\n", cimag(v1[i]) - cimag(v2[i]));
        if(fabs(creal(v1[i]-v2[i]))>1e-10) printf("%e\n", creal(v1[i]) - creal(v2[i]));
    }
}

#if defined USE_CUDA | defined USE_HIP

// Function for performing sparse matrix-vector multiplication on GPU.
void gpu_spmv(sparse_csr_t Hfull, csr_data_t *x, csr_data_t *y) {
  gpusparseHandle_t gpuhandle;
  CHECK_GPUSPARSE(gpusparseCreate(&gpuhandle));

  // Allocate device memory for CSR matrix and vectors
  csr_index_t *dAp, *dAi;
  csr_data_t  *dAx, *dpx, *dpy;

  CHECK_GPU(gpuMalloc((void**) &dAp, (1+csr_nrows(&Hfull))*sizeof(csr_index_t)));
  CHECK_GPU(gpuMalloc((void**) &dAi, csr_nnz(&Hfull)*sizeof(csr_index_t)));
  CHECK_GPU(gpuMalloc((void**) &dAx, csr_nnz(&Hfull)*sizeof(csr_data_t)));
  CHECK_GPU(gpuMalloc((void**) &dpx, csr_ncols(&Hfull)*sizeof(csr_data_t)));
  CHECK_GPU(gpuMalloc((void**) &dpy, csr_nrows(&Hfull)*sizeof(csr_data_t)));
  CHECK_GPU(gpuMemcpy(dpy, y, csr_dim(&Hfull)*sizeof(csr_data_t), gpuMemcpyHostToDevice));


  // Copy the CSR matrix and vectors from host to device
  CHECK_GPU(gpuMemcpy(dAp, Hfull.Ap, (1+csr_nrows(&Hfull))*sizeof(csr_index_t), gpuMemcpyHostToDevice));
  CHECK_GPU(gpuMemcpy(dAi, Hfull.Ai, csr_nnz(&Hfull)*sizeof(csr_index_t), gpuMemcpyHostToDevice));
  CHECK_GPU(gpuMemcpy(dAx, Hfull.Ax, csr_nnz(&Hfull)*sizeof(csr_data_t),  gpuMemcpyHostToDevice));
  CHECK_GPU(gpuMemcpy(dpx, x, csr_ncols(&Hfull)*sizeof(csr_data_t),  gpuMemcpyHostToDevice));

  // Create CSR matrix and vectors
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

  // Perform the SpMV operation
  CHECK_GPUSPARSE(gpusparseSpMV(gpuhandle,
                                GPUSPARSE_OPERATION_NON_TRANSPOSE, (const void*)&alpha, dHfull, dx,
                                (const void*)&beta, dy,
                                GPU_C_64F, GPUSPARSE_SPMV_CSR_ALG1, dbuffer));

  // Copy the result back to the host
  CHECK_GPU(gpuMemcpy(y, dpy, csr_nrows(&Hfull)*sizeof(csr_data_t), gpuMemcpyDeviceToHost));

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

    for(row = 0; row < H_blk.dim; row++){
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

int main(int argc, char *argv[])
{
    if(argc<2){
        printf("Usage: matrix_tool <lmax>\n");
        return 0;
    }

    char fname[256];
    sparse_csr_t Hall, Hfull, Hfull_blk, Hpart;
    sparse_csr_t *g, *gt;
    sparse_csr_t *H;
    int lmax = atoi(argv[1]);
    int cnt;

    // read the full couplings matrix structure
    // each non-zero value denotes a submatrix of dimensions same as the G matrices
    snprintf(fname, 255, "H.csr");
    csr_read(fname, &Hall);

    int rank = 0, nranks = 1;
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    if(nranks>1 && Hall.npart != nranks)
        ERROR("There are %d MPI ranks, but the matrix is partition for %d ranks.\n", nranks, Hall.npart);
#endif

    // get local partition of the global H matrix
    if(nranks>1){
        csr_get_partition(&Hpart, &Hall, rank, Hall.npart);
        /* DEBUG */
        /* snprintf(fname, 255, "H_part%d.csr", rank); */
        /* csr_write(fname, &Hpart); */
    } else {
        Hpart = Hall;
    }

    // read the individual Hamiltonian matrices H0 and H1
    // 2 matrices for each value of 0:lmax-1
    // Note: these are global, with original node indices - no partitioning
    H = malloc(sizeof(sparse_csr_t)*lmax*2);
    cnt = 0;
    for(int l=0; l<lmax; l++){
        snprintf(fname, 255, "H0l%d.csr", l);
        csr_read(fname, H+cnt);
        cnt++;
        snprintf(fname, 255, "H1l%d.csr", l);
        csr_read(fname, H+cnt);
        cnt++;
    }

    // load the base g matrices
    // 4 matrices: g0, g1, g2, g3
    // alpha 0:6-1
    // l     0:lmax-1
    g = malloc(sizeof(sparse_csr_t)*6*lmax*4);
    cnt = 0;
    for(int a=0; a<6; a++){
        for(int l=0; l<lmax; l++){
            for(int gi=0; gi<4; gi++){
                snprintf(fname, 255, "g%da%dl%d.csr", gi, a, l);
                csr_read(fname, g+cnt);
                cnt++;
            }
        }
    }

    // compute conjugate transpose of all g matrices
    gt = malloc(sizeof(sparse_csr_t)*6*lmax*4);
    cnt = 0;
    for(int a=0; a<6; a++){
        for(int l=0; l<lmax; l++){
            for(int gi=0; gi<4; gi++){
                csr_copy(gt+cnt, g+cnt);
                csr_conj_transpose(gt+cnt, g+cnt);
                cnt++;
            }
        }
    }

    // create the full rank-local matrix blocked structure
    // same Ap and Ai as Hpart
    // Ax modified to store sub-matrices (g.nnz)
    int blkdim = csr_nrows(&g[0]);
    csr_copy(&Hfull_blk, &Hpart);
    csr_block_params(&Hfull_blk, blkdim, csr_nnz(&g[0]));

    // create the non-blocked rank-local Hfull matrix structure
    //  - each non-zero is converted to blkdim x blkdim submatrix
    //  - each row has row_nnz(Hpart)*row_nnz(G) non-zero entries
    //  - create Hfull.Ap and Hfull.Ai accordingly
    // Dimensions of Hfull and Hfull_blk are the same, and they have
    // the same number of non-zeros. However, their Ap and Ai differ:
    // Hfull_blk inherits Ap and Ai directly from Hpart - one non-zero per entire submatrix.
    // Hfull stores all non-zeros independently in a native, non-blocked csr storage
    csr_unblock_matrix(&Hfull, &Hfull_blk, g);

    // Hfull_blk has the comm info copied from Hpart,
    // but Hfull doesnt - it has to be modified by the block size.
    // At this point there is no comm info in Hfull - so copy it.
    csr_unblock_comm_info(&Hfull, &Hfull_blk, rank, nranks);

    for(int r=0; r<nranks; r++){
        if(rank == r){
            printf("%d: All matrices read correctly. System info:\n", rank);
            printf(" - H dim:         %d x %d\n", csr_nrows(&H[0]), csr_ncols(&H[0]));
            printf(" - Hall dim:      %d x %d\n", csr_nrows(&Hall), csr_ncols(&Hall));
            printf(" - Hall nnz:      %d\n", csr_nnz(&Hall));
            printf(" - Hpart dim:     %d x %d\n", csr_nrows(&Hpart), csr_ncols(&Hpart));
            printf(" - Hpart nnz:     %d\n", csr_nnz(&Hpart));
            printf(" - Hfull dim:     %d x %d\n", csr_nrows(&Hfull_blk), csr_ncols(&Hfull_blk));
            printf(" - Hfull nnz:     %d\n", csr_nnz(&Hfull_blk));
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // allocate x and y vectors for SpMV: y = spA*x
    csr_data_t *x, *yblk, *yfull;
    x     = (csr_data_t *)calloc(csr_ncols(&Hfull_blk), sizeof(csr_data_t));
    yblk  = (csr_data_t *)calloc(csr_nrows(&Hfull_blk), sizeof(csr_data_t));
    yfull = (csr_data_t *)calloc(csr_nrows(&Hfull_blk), sizeof(csr_data_t));

    // allocate y vector for GPU SpMV
    csr_data_t *yblk_gpu;
    yblk_gpu = (csr_data_t *)calloc(csr_nrows(&Hfull_blk), sizeof(csr_data_t));

    { // Compute
        csr_index_t row, col, colp;

        // each submatrix will have the same non-zero structure as the base g matrices
        sparse_csr_t submatrix;
        csr_copy(&submatrix, g);

        tic();
        // for all rows
        for(row = 0; row < csr_nrows(&Hpart); row++){

            // for non-zeros in each row
            for(colp = Hpart.Ap[row]; colp < Hpart.Ap[row+1]; colp++){

                // NOTE: rows and cols in Hpart are remapped wrt. the original numbering in H
                col = Hpart.Ai[colp];

                // apply node renumbering - if available
                csr_index_t orig_row = row;
                csr_index_t orig_col = col;
                if(Hpart.perm) {
                    orig_row = Hpart.perm[csr_local_offset(&Hpart) + row];
                    orig_col = Hpart.perm[col];
                }

                // calculate kappa and mu parameters from row/col indices
                // see spnrbasis::bdpalphsigmaXmat
                int ki = (int)ik(orig_row); // k'
                int kj = (int)ik(orig_col); // k

                // double mui = imu(orig_row); // mu'
                // double muj = imu(orig_col); // mu

                sparse_csr_t *pg0, *pg1, *pg2, *pg3;
                sparse_csr_t *pgt0, *pgt1, *pgt2, *pgt3;
                csr_data_t H0[lmax], H1[lmax];
                csr_data_t ft[6];

                // TODO get the time-dependent f(a,t)
                for(int a=0; a<6; a++){
                    ft[a] = CMPLX(1,0);
                }

                // prefetch the Hamiltonian values H0(l) and H1(l)
                for(int l=0; l<lmax; l++){
                    H0[l] = csr_get_value(H + 2*l + 0, orig_row, orig_col);
                    H1[l] = csr_get_value(H + 2*l + 1, orig_row, orig_col);
                }

                csr_zero(&submatrix);
                for(int l=0; l<lmax; l++){
                    if(H0[l] != CMPLX(0,0)){

                        for(int a=0; a<6; a++){

                            pg0 = g + a*4*lmax + l*4;
                            pg1 = pg0 + 1;
                            pg2 = pg0 + 2;
                            pg3 = pg0 + 3;

                            // g matrices all have the same nnz pattern,
                            // so we can operate directly on the internal storage Ax
                            for(csr_index_t i=0; i<csr_nnz(&submatrix); i++){
                                submatrix.Ax[i] +=
                                    ft[a]*H0[l]*(pg0->Ax[i]        +
                                                 pg1->Ax[i]*ki     +
                                                 pg2->Ax[i]*kj     +
                                                 pg3->Ax[i]*ki*kj) ;
                            }
                        }
                    }
                }

                for(int l=0; l<lmax; l++){
                    if(H1[l] != CMPLX(0,0)){

                        for(int a=0; a<6; a++){

                            pgt0 = gt + a*4*lmax + l*4;
                            pgt1 = pgt0 + 1;
                            pgt2 = pgt0 + 2;
                            pgt3 = pgt0 + 3;

                            // g matrices all have the same nnz pattern,
                            // so we can operate directly on the internal storage Ax
                            for(csr_index_t i=0; i<csr_nnz(&submatrix); i++){
                                submatrix.Ax[i] -=
                                    ft[a]*H1[l]*(pgt0->Ax[i]       +
                                                 pgt1->Ax[i]*ki    +
                                                 pgt2->Ax[i]*kj    +
                                                 pgt3->Ax[i]*ki*kj);
                            }
                        }
                    }
                }

                // store the submatrix in the global Hfull_blk
                csr_block_insert(&Hfull_blk, row, col, submatrix.Ax);
            }
        }
        toc();

        // convert blocked Hfull_blk to non-blocked Hfull
        // could be done immediately above, but we do it here for timing purposes
        tic();
        for(row = 0; row < Hfull_blk.nrows; row++){

            // for non-zeros in each row
            for(colp = Hfull_blk.Ap[row]; colp < Hfull_blk.Ap[row+1]; colp++){

                // NOTE: rows and cols are remapped wrt. the original numbering in H
                col = Hfull_blk.Ai[colp];

                csr_block_link(&submatrix, &Hfull_blk, row, col);

                // insert into non-blocked Hfull matrix
                csr_index_t row_blk, col_blk, colp_blk;
                csr_index_t row_dst, col_dst;
                csr_index_t valp = 0;
                for(row_blk=0; row_blk<blkdim; row_blk++){

                    row_dst = row*blkdim + row_blk;
                    for(colp_blk=submatrix.Ap[row_blk]; colp_blk<submatrix.Ap[row_blk+1]; colp_blk++){
                        col_blk = submatrix.Ai[colp_blk];
                        col_dst = col*blkdim + col_blk;

                        csr_set_value(&Hfull, row_dst, col_dst, submatrix.Ax[valp]);
                        valp++;
                    }
                }
            }
        }
        toc();

        // set up communication buffers
        csr_init_communication(&Hfull_blk, x, rank, nranks);
        csr_init_communication(&Hfull, x, rank, nranks);

        // The Hfull_blk matrix contains all computed submatrices.
        // The submatrices are stored as a sub-block in the csr storage
        // meaning that the relevant Ax parts can be used directly
        // as Ax arrays in a template submatrix csr structure, e.g.
        // csr_block_link(&submatrix, &Hfull_blk, row, col);

        // initialize input vector. non-local parts are set to nan to verify communication:
        // all non-local entries are received from peers, hence set to non-nan during communication
        for(int i=0; i<csr_ncols(&Hfull_blk); i++) x[i] = CMPLX(NAN,NAN);
        for(int i=0; i<csr_nrows(&Hfull_blk); i++)
            x[csr_local_offset(&Hfull_blk) + i] = CMPLX(Hfull_blk.row_beg*blkdim + i, Hfull_blk.row_beg*blkdim + i);

        tic();
        csr_comm(&Hfull_blk, rank, nranks);
        for(row = 0; row < Hfull_blk.nrows; row++){

            // for non-zero blocks in each row
            for(colp = Hfull_blk.Ap[row]; colp < Hfull_blk.Ap[row+1]; colp++){

                // NOTE: rows and cols are remapped wrt. the original numbering in H
                col = Hfull_blk.Ai[colp];

                csr_block_link(&submatrix, &Hfull_blk, row, col);

                csr_data_t *xin, *yout;
                xin  = x + col*blkdim;
                yout = yblk + row*blkdim;

                csr_data_t *yout_gpu;
                yout_gpu = yblk_gpu + row*blkdim;

                // perform spmv
                csr_spmv(0, csr_nrows(&submatrix), &submatrix, xin, yout);

                // remember that at this stage xin and yout are renumbered wrt. the original node numbering
            }
        }
        toc();

        // initialize input vector. non-local parts are set to nan to verify communication:
        // all non-local entries are received from peers, hence set to non-nan during communication
        for(int i=0; i<csr_ncols(&Hfull); i++) x[i] = CMPLX(NAN,NAN);
        for(int i=0; i<csr_nrows(&Hfull); i++)
            x[csr_local_offset(&Hfull) + i] = CMPLX(Hfull.row_beg + i, Hfull.row_beg + i);

        // perform spmv for the non-blocked Hfull matrix (native CSR storage)
        tic();
        csr_comm(&Hfull, rank, nranks);
        csr_spmv(0, csr_nrows(&Hfull), &Hfull, x, yfull);
        toc();

        // validate - compare yblk and yfull results
        compare_vectors(yfull, yblk, csr_nrows(&Hfull));

        // DEBUG: write out the result vectors for comparison with single-rank result
        /*
        for(int r=0; r<nranks; r++){
            if(rank == r){
                for(row = 0; row < csr_nrows(&Hfull_blk); row++) fprintf(stderr, "%e %e\n", creal(yblk[row]), cimag(yblk[row]));
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        */
    }

#if defined USE_CUDA | defined USE_HIP
    gpu_spmv_test(Hfull, x, yfull);
    gpu_spmb_block_test(Hfull_blk, x, yfull, g);
#endif

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
#endif
}

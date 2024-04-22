#include "cuda_translate.h"
#include "hip_translate.h"

#include <unistd.h>

extern "C" {
#include "csr.h"
#include "gpu_sparse.h"
#include "solver_common.h"
#include "../src/tictoc.h"
}

extern int rank;
extern size_t total_bytes;

extern "C" {

    static int submatrix_nnz;
    static __device__ gpu_sparse_csr_t Hfull;
    static __device__ gpu_sparse_csr_t *g = NULL;
    static __device__ gpu_sparse_csr_t *gt = NULL;
    static __device__ gpu_sparse_csr_t *h0 = NULL;
    static __device__ csr_data_t *pgsum = NULL;
    static __device__ csr_data_t *pgtsum = NULL;
    static __device__ csr_index_t *Ai_orig, *Ki;
    static __device__ csr_index_t *Aj_orig, *Kj;
    static __device__ csr_data_t *H0, *H1;

    __global__ void device_compute_pgsums(csr_data_t cihdt , int lmax, int subsize,
                                          csr_data_t ft0, csr_data_t ft1, csr_data_t ft2,
                                          csr_data_t ft3, csr_data_t ft4, csr_data_t ft5)
    {
        const uint ix = blockIdx.x*blockDim.x + threadIdx.x;
        const uint gx = blockIdx.y;
        const uint lx = blockIdx.z;
        csr_data_t ft[6] = {ft0, ft1, ft2, ft3, ft4, ft5};
        csr_data_t pgtmp, pgttmp;
        const csr_data_t *pg, *pgt;
        uint ax;

        ax = (lx%2+1)%2;
        pg = g[ax*4*lmax + lx*4 + gx].Ax;
        pgtmp = gpuCmul(pg[ix], gpuCmul(cihdt, ft[ax]));
        pgt = gt[ax*4*lmax + lx*4 + gx].Ax;
        pgttmp = gpuCmul(pgt[ix], gpuCmul(cihdt, ft[ax]));

        ax += 2;
        pg = g[ax*4*lmax + lx*4 + gx].Ax;
        pgtmp = gpuCadd(pgtmp, gpuCmul(pg[ix], gpuCmul(cihdt, ft[ax])));
        pgt = gt[ax*4*lmax + lx*4 + gx].Ax;
        pgttmp = gpuCadd(pgttmp, gpuCmul(pgt[ix], gpuCmul(cihdt, ft[ax])));

        ax += 2;
        pg = g[ax*4*lmax + lx*4 + gx].Ax;
        pgtmp = gpuCadd(pgtmp, gpuCmul(pg[ix], gpuCmul(cihdt, ft[ax])));
        pgt = gt[ax*4*lmax + lx*4 + gx].Ax;
        pgttmp = gpuCadd(pgttmp, gpuCmul(pgt[ix], gpuCmul(cihdt, ft[ax])));

        pgsum[lx*4*subsize + gx*subsize + ix] = pgtmp;
        pgtsum[lx*4*subsize + gx*subsize + ix] = pgttmp;
    }

    __global__ void device_compute_timedep_matrices(int lmax, gpu_complex_t ihdt, csr_index_t submatrix_nnz, csr_index_t subsize)
    {
        const uint thrid  = threadIdx.x;
        const uint nthr   = blockDim.x;
        const uint nnzid  = blockIdx.x; // non-zero entry in the Hfull_blk matrix
        const uint subblk = blockIdx.y; // part of the submatrix this thread works on
        const uint submatrix_loc = subblk*nthr + thrid;  // id of the submatrix non-zero entry this thread computes
        const uint map_loc = nnzid*submatrix_nnz + submatrix_loc; // id of the global non-zero entry this thread computes

        // multipliers
        const gpu_complex_t ki = gpuMakeComplex(Ki[nnzid], 0);
        const gpu_complex_t kj = gpuMakeComplex(Kj[nnzid], 0);
        const gpu_complex_t ki2 = gpuCmul(ki, ki);
        const gpu_complex_t ki3 = gpuCmul(ki2, ki);
        const gpu_complex_t kikj = gpuCmul(ki, kj);

        gpu_complex_t result, tmp;

        // deactivate threads outside of submatrix range
        if(submatrix_loc < submatrix_nnz){

            // diagonal part
            result = gpuMakeComplex(0.0, 0.0);
            if(Ai_orig[nnzid] == Aj_orig[nnzid]){
                tmp = gpuCadd(gpuCmul(h0[1].Ax[submatrix_loc], ki), h0[0].Ax[submatrix_loc]);
                tmp = gpuCadd(gpuCmul(h0[2].Ax[submatrix_loc], ki2), tmp);
                tmp = gpuCadd(gpuCmul(h0[3].Ax[submatrix_loc], ki3), tmp);
                result = gpuCmul(ihdt, tmp);
            }

            for(int l=0; l<lmax; l++){
                gpu_complex_t H0l = H0[nnzid*lmax + l];
                tmp = pgsum[l*4*subsize + 0*subsize + submatrix_loc];
                tmp = gpuCadd(tmp, gpuCmul(pgsum[l*4*subsize + 1*subsize + submatrix_loc], ki));
                tmp = gpuCadd(tmp, gpuCmul(pgsum[l*4*subsize + 2*subsize + submatrix_loc], kj));
                tmp = gpuCadd(tmp, gpuCmul(pgsum[l*4*subsize + 3*subsize + submatrix_loc], kikj));
                tmp = gpuCmul(tmp, H0l);
                result = gpuCadd(result, tmp);

                gpu_complex_t H1l = H1[nnzid*lmax + l];
                tmp = pgtsum[l*4*subsize + 0*subsize + submatrix_loc];
                tmp = gpuCadd(tmp, gpuCmul(pgtsum[l*4*subsize + 1*subsize + submatrix_loc], kj));
                tmp = gpuCadd(tmp, gpuCmul(pgtsum[l*4*subsize + 2*subsize + submatrix_loc], ki));
                tmp = gpuCadd(tmp, gpuCmul(pgtsum[l*4*subsize + 3*subsize + submatrix_loc], kikj));
                tmp = gpuCmul(tmp, H1l);
                result = gpuCadd(result, tmp);
            }

            uint dest = Hfull.Ai_sub_map[map_loc];
            Hfull.Ax[dest] = result;
        }
    }

    void gpu_compute_row_col(int lmax, const sparse_csr_t *H,
                             sparse_csr_t *Hfull_blk, sparse_csr_t *Hfull)
    {
        csr_index_t nrows = csr_nrowblocks(Hfull_blk);
        csr_index_t row, col, colp;

        csr_index_t *cpu_Ai_orig, *cpu_Ki;
        csr_index_t *cpu_Aj_orig, *cpu_Kj;
        csr_data_t *cpu_H0, *cpu_H1;
        csr_index_t Hfull_nnz = Hfull_blk->nnz;

        // there arrays have O(Hfull_blk.nnz) entries
        cpu_Ai_orig = (csr_index_t*)malloc(sizeof(csr_index_t)*Hfull_nnz);
        cpu_Aj_orig = (csr_index_t*)malloc(sizeof(csr_index_t)*Hfull_nnz);
        cpu_Ki = (csr_index_t*)malloc(sizeof(csr_index_t)*Hfull_nnz);
        cpu_Kj = (csr_index_t*)malloc(sizeof(csr_index_t)*Hfull_nnz);
        cpu_H0 = (csr_data_t*)malloc(sizeof(csr_data_t)*Hfull_nnz*lmax);
        cpu_H1 = (csr_data_t*)malloc(sizeof(csr_data_t)*Hfull_nnz*lmax);

        for(row = 0; row < nrows; row++){

            // for non-zeros in each row
            for(colp = Hfull_blk->Ap[row]; colp < Hfull_blk->Ap[row+1]; colp++){

                // NOTE: rows and cols in Hfull_blk are remapped wrt. the original numbering in H
                col = Hfull_blk->Ai[colp];

                // apply node renumbering - if available
                csr_index_t orig_row = row;
                csr_index_t orig_col = col;
                if(Hfull_blk->perm) {
                    orig_row = Hfull_blk->perm[csr_local_rowoffset(Hfull_blk) + row];
                    orig_col = Hfull_blk->perm[col];
                }

                // calculate kappa and mu parameters from row/col indices
                // see spnrbasis::bdpalphsigmaXmat
                int ki = (int)ik(orig_row); // k'
                int kj = (int)ik(orig_col); // k

                cpu_Ai_orig[colp] = orig_row;
                cpu_Aj_orig[colp] = orig_col;
                cpu_Ki[colp] = ki;
                cpu_Kj[colp] = kj;

                // prefetch the Hamiltonian values H0(l) and H1(l)
                for(int l=0; l<lmax; l++){
                    cpu_H0[colp*lmax + l] = csr_get_value(H + 2*l + 0, orig_row, orig_col);
                    cpu_H1[colp*lmax + l] = csr_get_value(H + 2*l + 1, orig_row, orig_col);
                }
            }
        }

        // copy the auxiliary arrays to device
        {
            void *devp;
            size_t bytes = 0;

            // Ai_orig
            CHECK_GPU(gpuMalloc((void**) &devp, sizeof(csr_index_t)*Hfull_nnz));
            CHECK_GPU(gpuMemcpyToSymbol(Ai_orig, &devp, sizeof(csr_index_t*), 0, gpuMemcpyHostToDevice));
            CHECK_GPU(gpuMemcpy(devp, cpu_Ai_orig, sizeof(csr_index_t)*Hfull_nnz, gpuMemcpyHostToDevice));
            bytes += sizeof(csr_index_t)*Hfull_nnz;

            // Aj_orig
            CHECK_GPU(gpuMalloc((void**) &devp, sizeof(csr_index_t)*Hfull_nnz));
            CHECK_GPU(gpuMemcpyToSymbol(Aj_orig, &devp, sizeof(csr_index_t*), 0, gpuMemcpyHostToDevice));
            CHECK_GPU(gpuMemcpy(devp, cpu_Aj_orig, sizeof(csr_index_t)*Hfull_nnz, gpuMemcpyHostToDevice));
            bytes += sizeof(csr_index_t)*Hfull_nnz;

            // Ki
            CHECK_GPU(gpuMalloc((void**) &devp, sizeof(csr_index_t)*Hfull_nnz));
            CHECK_GPU(gpuMemcpyToSymbol(Ki, &devp, sizeof(csr_index_t*), 0, gpuMemcpyHostToDevice));
            CHECK_GPU(gpuMemcpy(devp, cpu_Ki, sizeof(csr_index_t)*Hfull_nnz, gpuMemcpyHostToDevice));
            bytes += sizeof(csr_index_t)*Hfull_nnz;

            // Kj
            CHECK_GPU(gpuMalloc((void**) &devp, sizeof(csr_index_t)*Hfull_nnz));
            CHECK_GPU(gpuMemcpyToSymbol(Kj, &devp, sizeof(csr_index_t*), 0, gpuMemcpyHostToDevice));
            CHECK_GPU(gpuMemcpy(devp, cpu_Kj, sizeof(csr_index_t)*Hfull_nnz, gpuMemcpyHostToDevice));
            bytes += sizeof(csr_index_t)*Hfull_nnz;

            // H0
            CHECK_GPU(gpuMalloc((void**) &devp, sizeof(csr_data_t)*Hfull_nnz*lmax));
            CHECK_GPU(gpuMemcpyToSymbol(H0, &devp, sizeof(csr_data_t*), 0, gpuMemcpyHostToDevice));
            CHECK_GPU(gpuMemcpy(devp, cpu_H0, sizeof(csr_data_t)*Hfull_nnz*lmax, gpuMemcpyHostToDevice));
            bytes += sizeof(csr_data_t)*Hfull_nnz*lmax;

            // H1
            CHECK_GPU(gpuMalloc((void**) &devp, sizeof(csr_data_t)*Hfull_nnz*lmax));
            CHECK_GPU(gpuMemcpyToSymbol(H1, &devp, sizeof(csr_data_t*), 0, gpuMemcpyHostToDevice));
            CHECK_GPU(gpuMemcpy(devp, cpu_H1, sizeof(csr_data_t)*Hfull_nnz*lmax, gpuMemcpyHostToDevice));
            bytes += sizeof(csr_data_t)*Hfull_nnz*lmax;

            total_bytes += bytes;
            PDEBUG("gpu_compute_row_col: allocated %li bytes on GPU (total %li)\n", bytes, total_bytes);

            printf("GPU copied auxiliary arrays to GPU\n");
        }
    }

    static int is_timedep_initialized = 0;
    void gpu_compute_timedep_matrices(double h, double dt, csr_data_t *ft, int lmax,
                                      const sparse_csr_t *cpu_Hfull_blk, sparse_csr_t *cpu_Hfull,
                                      gpu_sparse_csr_t *gpu_Hfull)
    {
        int nthreads = GPU_THREADS_PER_BLOCK;
        int nblocks = submatrix_nnz/nthreads;
        if(nblocks*nthreads<submatrix_nnz) nblocks++;
        int subsize = nblocks*nthreads;

        // one-time init - memory allocation
        if(!is_timedep_initialized){

            size_t bytes = 0;
            
            // copy the matrix structure
            gpu_put_csr(gpu_Hfull, cpu_Hfull);
            CHECK_GPU(gpuMemcpyToSymbol(Hfull, gpu_Hfull, sizeof(gpu_sparse_csr_t), 0, gpuMemcpyHostToDevice));

            csr_data_t *devp;
            CHECK_GPU(gpuMalloc((void**) &devp, lmax*4*subsize*sizeof(csr_data_t)));
            bytes += lmax*4*subsize*sizeof(csr_data_t);
            CHECK_GPU(gpuMemcpyToSymbol(pgsum, &devp, sizeof(csr_data_t*), 0, gpuMemcpyHostToDevice));

            CHECK_GPU(gpuMalloc((void**) &devp, lmax*4*subsize*sizeof(csr_data_t)));
            bytes += lmax*4*subsize*sizeof(csr_data_t);
            CHECK_GPU(gpuMemcpyToSymbol(pgtsum, &devp, sizeof(csr_data_t*), 0, gpuMemcpyHostToDevice));

            is_timedep_initialized = 1;

            total_bytes += bytes;
            PDEBUG("gpu_compute_timedep_matrices: allocated %li bytes on GPU (total %li)\n", bytes, total_bytes);
        }

        // precompute G-sums and Gt-sums
        {
            dim3 blocks(nblocks, 4, lmax);
            csr_data_t cihdt = gpuMakeComplex(0.0, SoL*h*dt/2);
            device_compute_pgsums<<<blocks, nthreads>>>(cihdt, lmax, subsize, ft[0], ft[1], ft[2], ft[3], ft[4], ft[5]);
        }

        // assemble Ht
        {
            dim3 blocks(cpu_Hfull_blk->nnz, nblocks);
            csr_data_t ihdt = gpuMakeComplex(0.0, h*dt/2);
            device_compute_timedep_matrices<<<blocks, nthreads>>>(lmax, ihdt, submatrix_nnz, subsize);
        }
        gpuDeviceSynchronize();
    }

    void gpu_init_model_matrices(int nmatrices, const sparse_csr_t *cpu_g, const sparse_csr_t *cpu_gt, const sparse_csr_t *cpu_h0)
    {
        gpu_sparse_csr_t gputemp = {};
        gpu_sparse_csr_t *devp;
        size_t bytes = 0;

        submatrix_nnz = cpu_g[0].nnz;

        CHECK_GPU(gpuMalloc((void**) &devp, nmatrices*sizeof(gpu_sparse_csr_t)));
        bytes += nmatrices*sizeof(gpu_sparse_csr_t);
        CHECK_GPU(gpuMemcpyToSymbol(g, &devp, sizeof(gpu_sparse_csr_t*), 0, gpuMemcpyHostToDevice));
        for(int i=0; i<nmatrices; i++){
            memset(&gputemp, 0, sizeof(gpu_sparse_csr_t));
            gpu_put_csr(&gputemp, cpu_g+i);
            CHECK_GPU(gpuMemcpy(devp+i, &gputemp, sizeof(gpu_sparse_csr_t), gpuMemcpyHostToDevice));
        }

        CHECK_GPU(gpuMalloc((void**) &devp, nmatrices*sizeof(gpu_sparse_csr_t)));
        bytes += nmatrices*sizeof(gpu_sparse_csr_t);
        CHECK_GPU(gpuMemcpyToSymbol(gt, &devp, sizeof(gpu_sparse_csr_t*), 0, gpuMemcpyHostToDevice));
        for(int i=0; i<nmatrices; i++){
            memset(&gputemp, 0, sizeof(gpu_sparse_csr_t));
            gpu_put_csr(&gputemp, cpu_gt+i);
            CHECK_GPU(gpuMemcpy(devp+i, &gputemp, sizeof(gpu_sparse_csr_t), gpuMemcpyHostToDevice));
        }

        CHECK_GPU(gpuMalloc((void**) &devp, 4*sizeof(gpu_sparse_csr_t)));
        bytes += nmatrices*sizeof(gpu_sparse_csr_t);
        CHECK_GPU(gpuMemcpyToSymbol(h0, &devp, sizeof(gpu_sparse_csr_t*), 0, gpuMemcpyHostToDevice));
        for(int i=0; i<4; i++){
            memset(&gputemp, 0, sizeof(gpu_sparse_csr_t));
            gpu_put_csr(&gputemp, cpu_h0+i);
            CHECK_GPU(gpuMemcpy(devp+i, &gputemp, sizeof(gpu_sparse_csr_t), gpuMemcpyHostToDevice));
        }

        total_bytes += bytes;
        PDEBUG("gpu_init_model_matrices: allocated %li bytes on GPU (total %li)\n", bytes, total_bytes);
        
        printf("GPU model matrices initialized\n");
    }
}

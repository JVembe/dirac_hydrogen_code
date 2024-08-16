#include <string.h>
#include <math.h>
#include "gpu_bicgstab.h"
#include "gpu_sparse.h"

#if defined USE_CUDA | defined USE_HIP

gpublasHandle_t handle;
void gpu_blas_init()
{
    CHECK_GPU_BLAS(gpublasCreate(&handle));
}


#define ALLOC_VECTOR(wsp, v, n)                                         \
    if(NULL == wsp->v.x) {                                              \
        gpu_put_vec(&wsp->v, NULL, n);                                  \
    }                                                                   \
    v = &wsp->v;

#define ALLOC_VECTOR2(wsp, v, n, nrow, local_offset)                    \
    if(NULL == wsp->v.x) {                                              \
        gpu_put_vec(&wsp->v, NULL, n);                                  \
        gpu_vec_local_part(&wsp->v, nrow, local_offset);                \
    }                                                                   \
    v = &wsp->v;

static inline double squarednorm(int n, const gpu_dense_vec_t *v, int incx)
{
    gpu_complex_t result;
    CHECK_GPU_BLAS(_gpuZdotc(handle, n, v->x, incx, v->x, incx, &result));
    return gpuCreal(result);
}

/* static inline  */
/*   CHECK_GPU_BLAS(cublasZcopy(handle, nrow, rhs->x, 1, r->x, 1)); */
/*   spmv(mat, x, r0, CMPLX(1,0), CMPLX(0,0)); */
/*   CHECK_GPU_BLAS(cublasZaxpy(handle, nrow, &gpuMone, r0->x, 1, r->x, 1)); */



#ifdef USE_MPI

#include <mpi.h>
#include "utils.h"
extern int rank, nranks;

static double dreduce(double val)
{
    double retval;
    if(nranks==1) return val;
    CHECK_MPI(MPI_Allreduce(&val, &retval, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
    return retval;
}

static gpu_complex_t zreduce(gpu_complex_t val)
{
    gpu_complex_t retval;
    if(nranks==1) return val;
    CHECK_MPI(MPI_Allreduce(&val, &retval, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
    return retval;
}

static void rankprint(char *fname, gpu_complex_t *v, int n)
{
    FILE *fd;
    for(int r=0; r<nranks; r++){
        if(rank==r){
            if(0==rank)
                fd = fopen(fname, "w+");
            else
                fd = fopen(fname, "a+");                
            for(int i=0; i<n; i++){
                double r, j;
                r = gpuCreal(v[i]);
                if(r == -0.0) r = 0.0;
                j = gpuCimag(v[i]);
                if(j == -0.0) j = 0.0;
                fprintf(fd, "%.10e %.10e\n", r, j);
            }
            fclose(fd);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

#else

static const double dreduce(double val)
{
    return val;
}

static const gpu_complex_t zreduce(gpu_complex_t val)
{
    return val;
}

#endif


/** \internal Low-level bi conjugate gradient stabilized algorithm
  * \param mat The matrix A
  * \param rhs The right hand side vector b
  * \param x On input and initial solution, on output the computed solution.
  * \param precond A preconditioner being able to efficiently solve for an
  *                approximation of Ax=b (regardless of b)
  * \param iters On input the max number of iteration, on output the number of performed iterations.
  * \param tol_error On input the tolerance error, on output an estimation of the relative error.
  * \return false in the case of numerical issue, for example a break down of BiCGSTAB.
  */
void gpu_bicgstab(gpu_spmv_fun spmv, const void *mat, const gpu_dense_vec_t *rhs, gpu_dense_vec_t *x, 
                  int nrow, int ncol, int local_col_beg,
                  gpu_precond_fun psolve, const void *precond, gpu_solver_workspace_t *wsp, int *iters, double *tol_error)
{

  double tol = *tol_error;
  int maxIters = *iters;
  int incx = 1;
  
  gpu_complex_t gpuOne   = gpuMakeComplex(1,0);
  gpu_complex_t gpuMone  = gpuMakeComplex(-1,0);
  gpu_complex_t gpuZero  = gpuMakeComplex(0,0);

  gpu_complex_t rho    = gpuOne;
  gpu_complex_t alpha  = gpuOne;
  gpu_complex_t w      = gpuOne;

  /* VectorType v = VectorType::Zero(n), p = VectorType::Zero(n); */
  gpu_dense_vec_t *v, *p;
  ALLOC_VECTOR(wsp, v, nrow);
  ALLOC_VECTOR(wsp, p, nrow);

  /* VectorType y(n),  z(n); */
  gpu_dense_vec_t *y, *z;
  ALLOC_VECTOR2(wsp, y, ncol, nrow, local_col_beg);
  ALLOC_VECTOR2(wsp, z, ncol, nrow, local_col_beg);

  /* VectorType s(n),  t(n); */
  gpu_dense_vec_t *s, *t;
  ALLOC_VECTOR(wsp, s, nrow);
  ALLOC_VECTOR(wsp, t, nrow);

  gpu_dense_vec_t *r, *r0;
  ALLOC_VECTOR(wsp, r, nrow);
  ALLOC_VECTOR(wsp, r0, nrow);

  /* VectorType r  = rhs - mat * x; */
  /* VectorType r0 = r; */
  gpuZcopy(nrow, rhs->x, 1, r->x, 1);
  spmv(mat, x, r0, CMPLX(1,0), CMPLX(0,0));
  gpuZaxpy(nrow, &gpuMone, r0->x, 1, r->x, 1);
  gpuZcopy(nrow, r->x, 1, r0->x, 1);
  
  /* RealScalar r0_sqnorm = r0.squaredNorm(); */
  double r0_sqnorm = dreduce(squarednorm(nrow, r0, incx));
  /* printf("initial norm %e\n", sqrt(r0_sqnorm)); */

  /* RealScalar rhs_sqnorm = rhs.squaredNorm(); */
  double rhs_sqnorm = dreduce(squarednorm(nrow, rhs, incx));
  
  if(rhs_sqnorm == 0){
      // x.setZero();
      gpuZscal(ncol, &gpuZero, x->x, incx);
      *iters = 0;
      *tol_error = 0;
      return;
  }

  /* RealScalar tol2 = tol*tol*rhs_sqnorm; */
  double tol2 = tol*tol*rhs_sqnorm;

  /* RealScalar eps2 = NumTraits<Scalar>::epsilon()*NumTraits<Scalar>::epsilon(); */
  double eps2 = (double)1e-12*(double)1e-12;

  if(r0_sqnorm < tol2){
      *iters = 0;
      *tol_error = sqrt(r0_sqnorm/rhs_sqnorm);
      return;
  }
  
  /* Index i = 0; */
  int i = 0;

  /* Index restarts = 0; */
  int restarts = 0;

  /* while ( r.squaredNorm() > tol2 && i<maxIters ) */
  #ifdef SCALING_MODE
  while ( i<20 ){
  #else
  while ( i<maxIters ){
  #endif
      gpu_complex_t blasa, blasb;
      gpu_complex_t rho_old = rho;

      /*   rho = r0.dot(r); */
      rho = zreduce(gpuZdotc(nrow, r0->x, incx, r->x, incx));

      if (gpuCabs(rho) < eps2*r0_sqnorm) {
          /* printf("restart %e %e\n", fabs(rho), eps2*r0_sqnorm); */

          /*
            The new residual vector became too orthogonal to the arbitrarily chosen direction r0
            Let's restart with a new r0:
          */
          /*     r  = rhs - mat * x; */
          /*     r0 = r; */
          gpuZcopy(nrow, rhs->x, 1, r->x, 1);
          spmv(mat, x, r0, CMPLX(1,0), CMPLX(0,0));
          gpuZaxpy(nrow, &gpuMone, r0->x, 1, r->x, 1);
          gpuZcopy(nrow, r->x, 1, r0->x, 1);

          /*     rho = r0_sqnorm = r.squaredNorm(); */
          r0_sqnorm = dreduce(squarednorm(nrow, r, incx));
          rho = gpuMakeComplex(r0_sqnorm, 0);
          /* if(restarts++ == 0) */
          /*     i = 0; */
      }

      /*   Scalar beta = (rho/rho_old) * (alpha / w); */
      gpu_complex_t beta = gpuCmul(gpuCdiv(rho,rho_old), gpuCdiv(alpha,w));

      /*   p = r + beta * (p - w * v); */
      blasa = gpuCmul(w, gpuMakeComplex(-1,0));
      gpuZaxpy(nrow, &blasa, v->x, 1, p->x, 1);
      gpuZaxpby(nrow, gpuOne, r->x, incx, beta, p->x, incx);

      /*   y = precond.solve(p); */
      psolve(precond, p, y);

      /*   v.noalias() = mat * y; */
      spmv(mat, y, v, CMPLX(1,0), CMPLX(0,0));

      /*   alpha = rho / r0.dot(v); */
      alpha = gpuCdiv(rho, zreduce(gpuZdotc(nrow, r0->x, incx, v->x, incx)));

      /*   s = r - alpha * v; */
      gpuZcopy(nrow, v->x, 1, s->x, 1);
      blasa = gpuMakeComplex(1,0);
      blasb = gpuCmul(alpha, gpuMakeComplex(-1,0));
      gpuZaxpby(nrow, blasa, r->x, incx, blasb, s->x, incx);

      /*   z = precond.solve(s); */
      psolve(precond, s, z);

      /*   t.noalias() = mat * z; */
      spmv(mat, z, t, CMPLX(1,0), CMPLX(0,0));

      /*   RealScalar tmp = t.squaredNorm(); */
      double tmp = dreduce(squarednorm(nrow, t, incx));

      /*   if(tmp>RealScalar(0)) */
      /*     w = t.dot(s) / tmp; */
      /*   else */
      /*     w = Scalar(0); */
      if(tmp > 0)
          w = gpuCdiv(zreduce(gpuZdotc(nrow, t->x, incx, s->x, incx)), gpuMakeComplex(tmp, 0));
      else
          w = gpuMakeComplex(0,0);

      /*   x += alpha * y + w * z; */
      gpuZaxpby(nrow, alpha, y->x+local_col_beg, incx, w, z->x+local_col_beg, incx);
      gpuZaxpy(nrow, &gpuOne, z->x+local_col_beg, 1, x->x+local_col_beg, 1);

      /*   r = s - w * t; */
      gpuZcopy(nrow, t->x, 1, r->x, 1);
      blasb = gpuCmul(w, gpuMakeComplex(-1,0));
      gpuZaxpby(nrow, gpuOne, s->x, incx, blasb, r->x, incx);

      tmp = dreduce(squarednorm(nrow, r, incx));
      //#ifdef DEBUG
      //if(rank==0) printf("%e\n", sqrt(tmp));
      //#endif
      #ifndef SCALING_MODE
	  if(tmp < tol2) break;
	  #endif
      ++i;
  }

  /* tol_error = sqrt(r.squaredNorm()/rhs_sqnorm); */
  *tol_error = sqrt(dreduce(squarednorm(nrow, r, incx))/rhs_sqnorm);
  *iters = i;
  return;
}

#endif

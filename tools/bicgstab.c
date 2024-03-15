#include "bicgstab.h"
#include <math.h>
#include <cblas-netlib.h>

int incx = 1;
#define squarednorm(v, n) creal(zdotc_(&n, v, &incx, v, &incx))

/* const double squarednorm(const cdouble_t *v, int n) */
/* { */
/*     double sum = 0; */
/*     for(int i=0; i<n; i++){ */
/*         sum += v[i]*conj(v[i]); */
/*     } */
/*     return sum; */
/* } */

const void pvec(const char *hdr, const cdouble_t *v, int n)
{
    printf("---- %s\n", hdr);
    for(int i=0; i<n; i++)
        printf("%e %e\n", creal(v[i]), cimag(v[i]));
}

// BLAS1
complex double zdotc_(const int *n, const double complex *x, const int *incx, const double complex *y, const int *incy);
void zaxpy_(const int *n, const double complex *alpha, const double complex *x, const int *incx, double complex *y, const int *incy);
void zaxpby_(const int *n, const double complex *alpha, const double complex *x, const int *incx, const double complex *beta, double complex *y, const int *incy);

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
int bicgstab(spmv_fun spmv, const void *mat, const cdouble_t *rhs, cdouble_t *x, int n,             
             precond_fun psolve, const void *precond, int *iters, double *tol_error)
{
  double tol = *tol_error;
  int maxIters = *iters;

  cdouble_t *r, *r0;
  r  = ALLOC_VECTOR(n);
  r0 = ALLOC_VECTOR(n);

  /* VectorType r  = rhs - mat * x; */
  /* VectorType r0 = r; */
  // TODO: comm
  spmv(mat, x, r);
  for(int i=0; i<n; i++) {
      r[i] = rhs[i] - r[i];
      r0[i] = r[i];
  }
  
  /* RealScalar r0_sqnorm = r0.squaredNorm(); */
  double r0_sqnorm = squarednorm(r0, n);
  
  /* RealScalar rhs_sqnorm = rhs.squaredNorm(); */
  double rhs_sqnorm = squarednorm(rhs, n);
  
  if(rhs_sqnorm == 0){
      // x.setZero();
      for(int i=0; i<n; i++)
          x[i] = 0;
      return 1;
  }
  cdouble_t rho    = 1;
  cdouble_t alpha  = 1;
  cdouble_t w      = 1;
  
  /* VectorType v = VectorType::Zero(n), p = VectorType::Zero(n); */
  cdouble_t *v = ALLOC_VECTOR(n);
  cdouble_t *p = ALLOC_VECTOR(n);
  
  /* VectorType y(n),  z(n); */
  cdouble_t *y = ALLOC_VECTOR(n);
  cdouble_t *z = ALLOC_VECTOR(n);
  
  /* VectorType s(n),  t(n); */
  cdouble_t *s = ALLOC_VECTOR(n);
  cdouble_t *t = ALLOC_VECTOR(n);

  /* RealScalar tol2 = tol*tol*rhs_sqnorm; */
  double tol2 = tol*tol*rhs_sqnorm;
  
  /* RealScalar eps2 = NumTraits<Scalar>::epsilon()*NumTraits<Scalar>::epsilon(); */
  double eps2 = (double)1e-12*(double)1e-12;
  
  /* Index i = 0; */
  int i = 0;
  
  /* Index restarts = 0; */
  int restarts = 0;
 
  /* while ( r.squaredNorm() > tol2 && i<maxIters ) */
  while ( squarednorm(r, n) > tol2 && i<maxIters ){

      cdouble_t blasa, blasb;
      cdouble_t rho_old = rho;

      /*   rho = r0.dot(r); */      
      rho = zdotc_(&n, r0, &incx, r, &incx);
      
      if (fabs(rho) < eps2*r0_sqnorm) {
          printf("restart %e %e\n", fabs(rho), eps2*r0_sqnorm);

          /*
            The new residual vector became too orthogonal to the arbitrarily chosen direction r0
            Let's restart with a new r0:
          */
          /*     r  = rhs - mat * x; */
          /*     r0 = r; */
          // TODO: comm
          spmv(mat, x, r);
          for(int i=0; i<n; i++) {
              r[i] = rhs[i] - r[i];
              r0[i] = r[i];
          }
          /*     rho = r0_sqnorm = r.squaredNorm(); */
          rho = r0_sqnorm = squarednorm(r, n);
          /* if(restarts++ == 0) */
          /*     i = 0; */
      }
      
      /*   Scalar beta = (rho/rho_old) * (alpha / w); */
      cdouble_t beta = (rho/rho_old) * (alpha / w);
      
      /*   p = r + beta * (p - w * v); */
      blasa = -w;
      zaxpy_(&n, &blasa, v, &incx, p, &incx);
      blasa = 1;
      zaxpby_(&n, &blasa, r, &incx, &beta, p, &incx);
    
      /*   y = precond.solve(p); */
      psolve(precond, p, y);
    
      /*   v.noalias() = mat * y; */
      // TODO: comm
      bzero(v, sizeof(cdouble_t)*n);
      spmv(mat, y, v);

      /*   alpha = rho / r0.dot(v); */
      alpha = rho / zdotc_(&n, r0, &incx, v, &incx);
      
      /*   s = r - alpha * v; */
      memcpy(s, v, sizeof(cdouble_t)*n);
      blasa = 1;
      blasb = -alpha;
      zaxpby_(&n, &blasa, r, &incx, &blasb, s, &incx);

      /*   z = precond.solve(s); */
      psolve(precond, s, z);
      
      /*   t.noalias() = mat * z; */
      // TODO: comm
      bzero(t, sizeof(cdouble_t)*n);
      spmv(mat, z, t);

      /*   RealScalar tmp = t.squaredNorm(); */
      double tmp = squarednorm(t, n); //zdotc_(&n, t, &incx, t, &incx);
      
      /*   if(tmp>RealScalar(0)) */
      /*     w = t.dot(s) / tmp; */
      /*   else */
      /*     w = Scalar(0); */
      if(tmp > 0)
          w = zdotc_(&n, t, &incx, s, &incx) / tmp;
      else
          w = 0;

      /*   x += alpha * y + w * z; */
      blasa = alpha;
      blasb = w;
      zaxpby_(&n, &blasa, y, &incx, &blasb, z, &incx);
      blasa = 1;
      zaxpy_(&n, &blasa, z, &incx, x, &incx);      

      /*   r = s - w * t; */
      memcpy(r, t, sizeof(cdouble_t)*n);
      blasa = 1;
      blasb = -w;
      zaxpby_(&n, &blasa, s, &incx, &blasb, r, &incx);

      printf("%e\n", sqrt(squarednorm(r, n)));
      ++i;
  }
  
  /* tol_error = sqrt(r.squaredNorm()/rhs_sqnorm); */
  *tol_error = sqrt(squarednorm(r, n)/rhs_sqnorm);
  *iters = i;
  return 1;
}

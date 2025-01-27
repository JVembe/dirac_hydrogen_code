#include "bicgstab.h"
#include <string.h>
#include <math.h>


#define ALLOC_VECTOR(wsp, v, n)                                         \
    if(NULL == wsp->v)                                                  \
        wsp->v = (cdouble_t*)calloc(n, sizeof(cdouble_t));              \
    v = wsp->v;

#define squarednorm(v, n) creal(zdotc_(&n, v, &incx, v, &incx))

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

static double complex zreduce(double complex val)
{
    double complex retval;
    if(nranks==1) return val;
    CHECK_MPI(MPI_Allreduce(&val, &retval, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
    return retval;
}

static void rankprint(char *fname, cdouble_t *v, int n)
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
                r = creal(v[i]);
                if(r == -0.0) r = 0.0;
                j = cimag(v[i]);
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

static const double complex zreduce(double complex val)
{
    return val;
}

#endif

void pvec(const char *hdr, const cdouble_t *v, int n)
{
    printf("---- %s\n", hdr);
    for(int i=0; i<n; i++)
        printf("%e %e\n", creal(v[i]), cimag(v[i]));
}

// BLAS1
complex double zdotc_(const int *n, const double complex *x, const int *incx, const double complex *y, const int *incy);
void zaxpy_(const int *n, const double complex *alpha, const double complex *x, const int *incx, double complex *y, const int *incy);
void zscal_(const int *n, const double complex *alpha, const double complex *x, const int *incx);
void zaxpby_(const int *n, const double complex *alpha, const double complex *x, const int *incx, const double complex *beta, double complex *y, const int *incy);

/* if zaxpby is not available */
/* #define zaxpby_(n, alpha, x, incx, beta, y, incy)   \ */
/*     {                                               \ */
/*         zscal_(n, beta, y, incy);                   \ */
/*         zaxpy_(n, alpha, x, incx, y, incy);         \ */
/*     } */


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
void bicgstab(spmv_fun spmv, const void *mat, const cdouble_t *rhs, cdouble_t *x, int nrow, int ncol, int local_col_beg,
              precond_fun psolve, const void *precond, solver_workspace_t *wsp, int *iters, double *tol_error)
{
  double tol = *tol_error;
  int maxIters = *iters;
  int incx = 1;
  
  cdouble_t rho    = 1;
  cdouble_t alpha  = 1;
  cdouble_t w      = 1;

  /* VectorType v = VectorType::Zero(n), p = VectorType::Zero(n); */
  cdouble_t *v, *p;
  ALLOC_VECTOR(wsp, v, nrow);
  ALLOC_VECTOR(wsp, p, nrow);

  /* VectorType y(n),  z(n); */
  cdouble_t *y, *z;
  ALLOC_VECTOR(wsp, y, ncol);
  ALLOC_VECTOR(wsp, z, ncol);

  /* VectorType s(n),  t(n); */
  cdouble_t *s, *t;
  ALLOC_VECTOR(wsp, s, nrow);
  ALLOC_VECTOR(wsp, t, nrow);

  cdouble_t *r, *r0;
  ALLOC_VECTOR(wsp, r, nrow);
  ALLOC_VECTOR(wsp, r0, nrow);

  /* VectorType r  = rhs - mat * x; */
  /* VectorType r0 = r; */
  for(int i=0; i<nrow; i++) r[i] = 0;
  spmv(mat, x, r);
  for(int i=0; i<nrow; i++) {
      r[i] = rhs[i] - r[i];
      r0[i] = r[i];
  }
  /* rankprint("cpur.txt", r, nrow); */
  /* return; */

  /* RealScalar r0_sqnorm = r0.squaredNorm(); */
  double r0_sqnorm = dreduce(squarednorm(r0, nrow));
  /* printf("initial norm %e\n", sqrt(r0_sqnorm)); */

  /* RealScalar rhs_sqnorm = rhs.squaredNorm(); */
  double rhs_sqnorm = dreduce(squarednorm(rhs, nrow));
  
  if(rhs_sqnorm == 0){
      // x.setZero();
      for(int i=0; i<nrow; i++) x[i] = 0;
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
  while ( i<maxIters ){

      cdouble_t blasa, blasb;
      cdouble_t rho_old = rho;

      /*   rho = r0.dot(r); */
      rho = zreduce(zdotc_(&nrow, r0, &incx, r, &incx));

      if (fabs(rho) < eps2*r0_sqnorm) {
          /* printf("restart %e %e\n", fabs(rho), eps2*r0_sqnorm); */

          /*
            The new residual vector became too orthogonal to the arbitrarily chosen direction r0
            Let's restart with a new r0:
          */
          /*     r  = rhs - mat * x; */
          /*     r0 = r; */
          spmv(mat, x, r);
          for(int i=0; i<nrow; i++) {
              r[i] = rhs[i] - r[i];
              r0[i] = r[i];
          }
          /*     rho = r0_sqnorm = r.squaredNorm(); */
          rho = r0_sqnorm = dreduce(squarednorm(r, nrow));
          /* if(restarts++ == 0) */
          /*     i = 0; */
      }

      /*   Scalar beta = (rho/rho_old) * (alpha / w); */
      cdouble_t beta = (rho/rho_old) * (alpha / w);

      /*   p = r + beta * (p - w * v); */
      blasa = -w;
      zaxpy_(&nrow, &blasa, v, &incx, p, &incx);
      blasa = 1;
      zaxpby_(&nrow, &blasa, r, &incx, &beta, p, &incx);

      /*   y = precond.solve(p); */
      psolve(precond, p, y+local_col_beg);

      /*   v.noalias() = mat * y; */
      bzero(v, sizeof(cdouble_t)*nrow);
      spmv(mat, y, v);

      /*   alpha = rho / r0.dot(v); */
      alpha = rho / zreduce(zdotc_(&nrow, r0, &incx, v, &incx));

      /*   s = r - alpha * v; */
      memcpy(s, v, sizeof(cdouble_t)*nrow);
      blasa = 1;
      blasb = -alpha;
      zaxpby_(&nrow, &blasa, r, &incx, &blasb, s, &incx);

      /*   z = precond.solve(s); */
      psolve(precond, s, z+local_col_beg);

      /*   t.noalias() = mat * z; */
      bzero(t, sizeof(cdouble_t)*nrow);
      spmv(mat, z, t);

      /*   RealScalar tmp = t.squaredNorm(); */
      double tmp = dreduce(squarednorm(t, nrow)); //zdotc_(&nrow, t, &incx, t, &incx);

      /*   if(tmp>RealScalar(0)) */
      /*     w = t.dot(s) / tmp; */
      /*   else */
      /*     w = Scalar(0); */
      if(tmp > 0)
          w = zreduce(zdotc_(&nrow, t, &incx, s, &incx)) / tmp;
      else
          w = 0;

      /*   x += alpha * y + w * z; */
      zaxpby_(&nrow, &alpha, y+local_col_beg, &incx, &w, z+local_col_beg, &incx);
      blasa = 1;
      zaxpy_(&nrow, &blasa, z+local_col_beg, &incx, x+local_col_beg, &incx);

      /*   r = s - w * t; */
      memcpy(r, t, sizeof(cdouble_t)*nrow);
      blasa = 1;
      blasb = -w;
      zaxpby_(&nrow, &blasa, s, &incx, &blasb, r, &incx);

      tmp = dreduce(squarednorm(r, nrow));
      //#ifdef DEBUG
      //if(rank==0) printf("%e\n", sqrt(tmp));
      //#endif
      if(tmp < tol2) break;
      ++i;
  }

  /* tol_error = sqrt(r.squaredNorm()/rhs_sqnorm); */
  *tol_error = sqrt(dreduce(squarednorm(r, nrow))/rhs_sqnorm);
  *iters = i;
  return;
}

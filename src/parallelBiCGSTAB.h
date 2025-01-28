#ifndef PARALLEL_BICGSTAB_H
#define PARALLEL_BICGSTAB_H


//This file is modified from Eigen source code distributed under the Mozilla Public License v 2.0
//I'm not sure what the licensing implications of this are.

 //
 // Copyright (C) 2011-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
 // Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
 //
 // This Source Code Form is subject to the terms of the Mozilla
 // Public License v. 2.0. If a copy of the MPL was not distributed
 // with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 
 // I think so at least? I'm a physicist not a copyright lawyer

 // Modifications to the BiCGSTAB algorithm are sourced from https://arxiv.org/pdf/2206.14780.pdf to assist in parallelization

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <mpi.h>
#include "mpiFuncs.h"

 namespace Eigen { 
  
 namespace internal {
  
 template<typename MatrixType, typename Rhs, typename Dest, typename Preconditioner>
 bool par_bicgstab(const MatrixType& mat, const Rhs& rhs, Dest& x,
               const Preconditioner& precond, Index& iters,
               typename Dest::RealScalar& tol_error)
 {
   using std::sqrt;
   using std::abs;
   typedef typename Dest::RealScalar RealScalar;
   typedef typename Dest::Scalar Scalar;
   typedef Matrix<Scalar,Dynamic,1> VectorType;
   RealScalar tol = tol_error;
   Index maxIters = iters;
  
   Index n = mat.cols();
   VectorType r  = rhs - mat * x;
   VectorType r0 = r;
   
   RealScalar r0_sqnorm_local = r0.squaredNorm();
   RealScalar rhs_sqnorm_local = rhs.squaredNorm();
   
   RealScalar r0_sqnorm; 
   RealScalar rhs_sqnorm;
   
   MPI_Allreduce(&r0_sqnorm_local, &r0_sqnorm, 1, mpi_get_type<RealScalar>(),MPI_SUM,MPI_COMM_WORLD);
   MPI_Allreduce(&rhs_sqnorm_local, &rhs_sqnorm, 1, mpi_get_type<RealScalar>(),MPI_SUM,MPI_COMM_WORLD);
   
   if(rhs_sqnorm == 0)
   {
     x.setZero();
     return true;
   }
   Scalar rho    = 1;
   Scalar alpha  = 1;
   Scalar w      = 1;
   
   VectorType v = VectorType::Zero(n), p = VectorType::Zero(n);
   VectorType y(n),  z(n);
   VectorType kt(n), ks(n);
  
   VectorType s(n), t(n);
  
   RealScalar tol2 = tol*tol*rhs_sqnorm;
   RealScalar eps2 = NumTraits<Scalar>::epsilon()*NumTraits<Scalar>::epsilon();
   Index i = 0;
   Index restarts = 0;
  
   // cout << "Parallel BiCGSTAB implementation running" << endl;
   
   
   RealScalar r_sqnorm_local = r.squaredNorm();
   RealScalar r_sqnorm;
   MPI_Allreduce(&r_sqnorm_local, &r_sqnorm, 1, mpi_get_type<RealScalar>(),MPI_SUM,MPI_COMM_WORLD);
   
   
   while ( r_sqnorm > tol2 && i<maxIters )
   {
     Scalar rho_old = rho;
	 Scalar rho_local = r0.dot(r);
	 // cout << "rho_local = " << rho_local << endl;
	 MPI_Allreduce(&rho_local,&rho,1,mpi_get_type<Scalar>(),MPI_SUM,MPI_COMM_WORLD);
	 
	 // cout << "rho = " << rho << endl;
     if (abs(rho) < eps2*r0_sqnorm)
     {
       // The new residual vector became too orthogonal to the arbitrarily chosen direction r0
       // Let's restart with a new r0:
       r  = rhs - mat * x;
       r0 = r;
       rho = r0_sqnorm = r.squaredNorm();
       if(restarts++ == 0)
         i = 0;
     }
     Scalar beta = (rho/rho_old) * (alpha / w);
	 // cout << "beta = " << beta << endl;
	 
     p = r + beta * (p - w * v);
     
     y = precond.solve(p);
     
     v.noalias() = mat * y;
	 
	 Scalar delta_local = r0.dot(v);
	 Scalar delta;
	 MPI_Allreduce(&delta_local,&delta,1,mpi_get_type<Scalar>(),MPI_SUM,MPI_COMM_WORLD);
	 
     alpha = rho / delta;
	 // cout << "alpha = " << alpha << endl;
     s = r - alpha * v;
  
     z = precond.solve(s);
     t.noalias() = mat * z;
  
     RealScalar tmp_local = t.squaredNorm();
	 RealScalar tmp;
	 MPI_Allreduce(&tmp_local,&tmp,1,mpi_get_type<RealScalar>(),MPI_SUM,MPI_COMM_WORLD);
	 
	 Scalar w_local;
     if(tmp>RealScalar(0))
       w_local = t.dot(s) / tmp;
     else
       w_local = Scalar(0);
	 
	 MPI_Allreduce(&w_local,&w,1,mpi_get_type<Scalar>(),MPI_SUM,MPI_COMM_WORLD);
	 
	 // cout << "w = " << w << endl;
   
     x += alpha * y + w * z;
     r = s - w * t;
	 r_sqnorm_local = r.squaredNorm();
	 MPI_Allreduce(&r_sqnorm_local, &r_sqnorm, 1, mpi_get_type<RealScalar>(),MPI_SUM,MPI_COMM_WORLD);
	 
     ++i;
     // cout << "Current # iterations = " << i << endl;
   }
   tol_error = sqrt(r.squaredNorm()/rhs_sqnorm);
   iters = i;
   return true; 
 }
 /*{
   using std::sqrt;
   using std::abs;
   typedef typename Dest::RealScalar RealScalar;
   typedef typename Dest::Scalar Scalar;
   typedef Matrix<Scalar,Dynamic,1> VectorType;
   RealScalar tol = tol_error;
   Index maxIters = iters;
  
   Index n = mat.cols();
   VectorType r  = rhs - mat * x;
   VectorType r0 = r;
   
   RealScalar r0_sqnorm_local = r0.squaredNorm();
   RealScalar rhs_sqnorm_local = rhs.squaredNorm();
   
   RealScalar r0_sqnorm;
   RealScalar rhs_sqnorm;
   
   MPI_Allreduce(&r0_sqnorm_local,&r0_sqnorm,1,mpi_get_type<RealScalar>(),MPI_SUM,MPI_COMM_WORLD);
   MPI_Allreduce(&rhs_sqnorm_local,&rhs_sqnorm,1,mpi_get_type<RealScalar>(),MPI_SUM,MPI_COMM_WORLD);
   
   if(rhs_sqnorm == 0)
   {
     x.setZero();
     return true;
   }
   Scalar rho    = 1;
   Scalar alpha  = 1;
   Scalar w      = 1;
   
   VectorType v = VectorType::Zero(n), p = VectorType::Zero(n);
   VectorType y(n),  z(n);
   VectorType kt(n), ks(n);
  
   VectorType s(n), t(n);
  
   RealScalar tol2 = tol*tol*rhs_sqnorm;
   RealScalar eps2 = NumTraits<Scalar>::epsilon()*NumTraits<Scalar>::epsilon();
   Index i = 0;
   Index restarts = 0;
   
   Scalar rho_old = rho;
  
   Scalar rho_local = r0.dot(r);
   
   MPI_Allreduce(&rho_local,&rho,1,mpi_get_type<Scalar>(),MPI_SUM,MPI_COMM_WORLD);
   Scalar beta = (rho/rho_old) * (alpha / w);
   
   while ( r.squaredNorm() > tol2 && i<maxIters )
   {
     
     p = r + beta * (p - w * v);
     
     y = precond.solve(p);
     
     v.noalias() = mat * y;
	
	 Scalar delta_local = r0.dot(v);
	 Scalar delta;
	 
	 MPI_Allreduce(&delta_local,&delta,1,mpi_get_type<RealScalar>(),MPI_SUM,MPI_COMM_WORLD);
	
	 cout << "delta_local = " << delta_local << endl;
	 cout << "delta = " << delta << endl;
	
     alpha = rho / delta;
	 
	 cout << "alpha = " << alpha << endl;
	 
     s = r - alpha * v;
  
     z = precond.solve(s);
     t.noalias() = mat * z;
  
     RealScalar tmp_local = t.squaredNorm();
	 Scalar ts_local = t.dot(s);
	 
	 RealScalar tmp;
	 Scalar ts;
	 
	 MPI_Allreduce(&tmp_local,&tmp,1,mpi_get_type<RealScalar>(),MPI_SUM,MPI_COMM_WORLD);
	 MPI_Allreduce(&ts_local,&ts,1,mpi_get_type<Scalar>(),MPI_SUM,MPI_COMM_WORLD);
	 
	 cout << "tmp_local = " << tmp_local << endl;
	 cout << "ts_local = " << ts_local << endl;
	 cout << "tmp = " << tmp << endl;
	 cout << "ts = " << ts << endl;
	 
	 
	 
     if(tmp>RealScalar(0))
       w = ts / tmp;
     else
       w = Scalar(0);
   
	 cout << "w = " << w << endl;
   
     x += alpha * y + w * z;
     r = s - w * t;
	 
	 rho_old = rho;
	 rho_local = r0.dot(r);
	 
	 cout << "rho_local = " << rho_local << endl;
	 
	 MPI_Allreduce(&rho_local,&rho,1,mpi_get_type<Scalar>(),MPI_SUM,MPI_COMM_WORLD);
	 
	 cout << "rho = " << rho << endl;
	 
	 if (abs(rho) < eps2*r0_sqnorm)
     {
       // The new residual vector became too orthogonal to the arbitrarily chosen direction r0
       // Let's restart with a new r0:
       r  = rhs - mat * x;
       r0 = r;
       rho = r0_sqnorm = r.squaredNorm();
       if(restarts++ == 0)
         i = 0;
     }
	 
	 alpha = rho/delta;
	 
	 beta = (alpha / w);
	 
     ++i;
   }
   tol_error = sqrt(r.squaredNorm()/rhs_sqnorm);
   iters = i;
   return true; 
 }*/
 
 }
  
 template< typename MatrixType_,
           typename Preconditioner_ = DiagonalPreconditioner<typename MatrixType_::Scalar> >
 class ParBiCGSTAB;
  
 namespace internal {
  
 template< typename MatrixType_, typename Preconditioner_>
 struct traits<ParBiCGSTAB<MatrixType_,Preconditioner_> >
 {
   typedef MatrixType_ MatrixType;
   typedef Preconditioner_ Preconditioner;
 };
  
 }
  
 template< typename MatrixType_, typename Preconditioner_>
 class ParBiCGSTAB : public IterativeSolverBase<ParBiCGSTAB<MatrixType_,Preconditioner_> >
 {
   typedef IterativeSolverBase<ParBiCGSTAB> Base;
   using Base::matrix;
   using Base::m_error;
   using Base::m_iterations;
   using Base::m_info;
   using Base::m_isInitialized;
 public:
   typedef MatrixType_ MatrixType;
   typedef typename MatrixType::Scalar Scalar;
   typedef typename MatrixType::RealScalar RealScalar;
   typedef Preconditioner_ Preconditioner;
  
 public:
  
   ParBiCGSTAB() : Base() {}
  
   template<typename MatrixDerived>
   explicit ParBiCGSTAB(const EigenBase<MatrixDerived>& A) : Base(A.derived()) {}
  
   template<typename Rhs,typename Dest>
   void _solve_vector_with_guess_impl(const Rhs& b, Dest& x) const
   {    
     m_iterations = Base::maxIterations();
     m_error = Base::m_tolerance;
     
     bool ret = internal::par_bicgstab(matrix(), b, x, Base::m_preconditioner, m_iterations, m_error);
  
     m_info = (!ret) ? NumericalIssue
            : m_error <= Base::m_tolerance ? Success
            : NoConvergence;
   }
  
 protected:
  
 };
 
 namespace internal {
 
 template<typename MatrixType, typename Rhs, typename Dest, typename Preconditioner>
 bool debugbicgstab(const MatrixType& mat, const Rhs& rhs, Dest& x,
               const Preconditioner& precond, Index& iters,
               typename Dest::RealScalar& tol_error)
 {
   using std::sqrt;
   using std::abs;
   typedef typename Dest::RealScalar RealScalar;
   typedef typename Dest::Scalar Scalar;
   typedef Matrix<Scalar,Dynamic,1> VectorType;
   RealScalar tol = tol_error;
   Index maxIters = iters;
  
   Index n = mat.cols();
   VectorType r  = rhs - mat * x;
   VectorType r0 = r;
   
   RealScalar r0_sqnorm = r0.squaredNorm();
   RealScalar rhs_sqnorm = rhs.squaredNorm();
   if(rhs_sqnorm == 0)
   {
     x.setZero();
     return true;
   }
   Scalar rho    = 1;
   Scalar alpha  = 1;
   Scalar w      = 1;
   
   VectorType v = VectorType::Zero(n), p = VectorType::Zero(n);
   VectorType y(n),  z(n);
   VectorType kt(n), ks(n);
  
   VectorType s(n), t(n);
  
   RealScalar tol2 = tol*tol*rhs_sqnorm;
   RealScalar eps2 = NumTraits<Scalar>::epsilon()*NumTraits<Scalar>::epsilon();
   Index i = 0;
   Index restarts = 0;
  
   // cout << "Original BiCGSTAB implementation running" << endl;

   while ( r.squaredNorm() > tol2 && i<maxIters )
   {
     Scalar rho_old = rho;
  
     rho = r0.dot(r);
	 // cout << "rho = " << rho << endl;
     if (abs(rho) < eps2*r0_sqnorm)
     {
       // The new residual vector became too orthogonal to the arbitrarily chosen direction r0
       // Let's restart with a new r0:
       r  = rhs - mat * x;
       r0 = r;
       rho = r0_sqnorm = r.squaredNorm();
       if(restarts++ == 0)
         i = 0;
     }
     Scalar beta = (rho/rho_old) * (alpha / w);
	 // cout << "beta = " << beta << endl;
	 
     p = r + beta * (p - w * v);
     
     y = precond.solve(p);
     
     v.noalias() = mat * y;
  
     alpha = rho / r0.dot(v);
	 // cout << "alpha = " << alpha << endl;
     s = r - alpha * v;
  
     z = precond.solve(s);
     t.noalias() = mat * z;
  
     RealScalar tmp = t.squaredNorm();
     if(tmp>RealScalar(0))
       w = t.dot(s) / tmp;
     else
       w = Scalar(0);
	 
	 // cout << "w = " << w << endl;
   
     x += alpha * y + w * z;
     r = s - w * t;
     ++i;
     // cout << "Current # iterations = " << i << endl;
   }
   tol_error = sqrt(r.squaredNorm()/rhs_sqnorm);
   iters = i;
   return true; 
 }
 
 }
 
 template< typename MatrixType_,
           typename Preconditioner_ = DiagonalPreconditioner<typename MatrixType_::Scalar> >
 class debugBiCGSTAB;
  
 namespace internal {
  
 template< typename MatrixType_, typename Preconditioner_>
 struct traits<debugBiCGSTAB<MatrixType_,Preconditioner_> >
 {
   typedef MatrixType_ MatrixType;
   typedef Preconditioner_ Preconditioner;
 };
  
 }
  
 template< typename MatrixType_, typename Preconditioner_>
 class debugBiCGSTAB : public IterativeSolverBase<debugBiCGSTAB<MatrixType_,Preconditioner_> >
 {
   typedef IterativeSolverBase<debugBiCGSTAB> Base;
   using Base::matrix;
   using Base::m_error;
   using Base::m_iterations;
   using Base::m_info;
   using Base::m_isInitialized;
 public:
   typedef MatrixType_ MatrixType;
   typedef typename MatrixType::Scalar Scalar;
   typedef typename MatrixType::RealScalar RealScalar;
   typedef Preconditioner_ Preconditioner;
  
 public:
  
   debugBiCGSTAB() : Base() {}
  
   template<typename MatrixDerived>
   explicit debugBiCGSTAB(const EigenBase<MatrixDerived>& A) : Base(A.derived()) {}
  
   ~debugBiCGSTAB() {}
  
   template<typename Rhs,typename Dest>
   void _solve_vector_with_guess_impl(const Rhs& b, Dest& x) const
   {    
     m_iterations = Base::maxIterations();
     m_error = Base::m_tolerance;
     
     bool ret = internal::debugbicgstab(matrix(), b, x, Base::m_preconditioner, m_iterations, m_error);
  
     m_info = (!ret) ? NumericalIssue
            : m_error <= Base::m_tolerance ? Success
            : NoConvergence;
   }
  
 protected:
  
 };
 
 }
#endif
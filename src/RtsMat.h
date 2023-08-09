/*
	This header file contains the class RtsMat, a wrapper class whose only purpose is to pass the action of an arbitrary Hamiltonian matrix H on a coefficient vector to Eigen's linear solvers
	
	It also contains the class SubmatPreconditioner, a specialization of the IncompleteLUT preconditioner which exploits the block structure of the Hamiltonian matrix to save memory and precondition the system in parallel.
	
	Other classes in here have previously been used for testing purposes.
*/

#ifndef RtsMat_H
#define RtsMat_H

#include "defs.h"
#include "hamiltonian.h"

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <mpi.h>
#include "mpiFuncs.h"
//using Eigen::SparseMatrix;


template<typename Htype>
class RtsMat;
namespace Eigen {
	namespace internal {
		template<typename Htype>
		struct traits<RtsMat<Htype> > : public Eigen::internal::traits<csmat >
		{};
	}
}
 

// Reverse propagator part of crank nicholson propagator

template<typename Htype>
class RtsMat : public Eigen::EigenBase<RtsMat<Htype> > {
private:
	const Htype* H = NULL;
	
	double t;
	double dt;
public:
	// Required typedefs, constants, and method:
	typedef double Scalar;
	typedef double RealScalar;
	typedef int StorageIndex;
	enum {
		ColsAtCompileTime = Eigen::Dynamic,
		MaxColsAtCompileTime = Eigen::Dynamic,
		IsRowMajor = true
	};
 
	Eigen::Index rows() const { return H->Dim(); }
	Eigen::Index cols() const { return H->Dim(); }
 
	template<typename Rhs>
	Eigen::Product<RtsMat,Rhs,Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs>& x) const {
		return Eigen::Product<RtsMat,Rhs,Eigen::AliasFreeProduct>(*this, x.derived());
	}

 
	// Custom API:
	RtsMat() {}
 
 /*void attachMats(const csmat &mat1, const csmat& mat2) {
		this->mat1 = &mat1;
		this->mat2 = &mat2;
	}
	*/
	
	cvec Svec(const cvec& v) const { return H->S(v); }
	
	cvec Hvec(const cvec& v) const { return H->H(t,v); }
	
	void setTime(double t) {
		this->t = t;
	}
	
	inline double getTime() const {
		return t;
	}
	
	void setDt(double t) {
		this->dt = t;
	}
	
	inline double getDt() const {
		return dt;
	}

	void attachHamiltonian(Htype& H) {
		this->H = &H;
	}
};


// Implementation of RtsMat * Eigen::DenseVector though a specialization of internal::generic_product_impl:
namespace Eigen {
	namespace internal {
		
		template<typename Rhs,typename Htype>
		struct generic_product_impl<RtsMat<Htype>, Rhs, SparseShape, DenseShape, GemvProduct> // GEMV stands for matrix-vector
		: generic_product_impl_base<RtsMat<Htype>,Rhs,generic_product_impl<RtsMat<Htype>,Rhs> >
			{
				typedef typename Product<RtsMat<Htype>,Rhs>::Scalar Scalar;
				
				
				template<typename Dest>
				static void scaleAndAddTo(Dest& dst, const RtsMat<Htype>& lhs, const Rhs& rhs, const Scalar& alpha)
				{
					int wrank;
					MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
					// cout << "rtsMatProd" << endl;
					const Eigen::IOFormat outformatLine(Eigen::FullPrecision,Eigen::DontAlignCols,", "," ","(","),"," = npy.array((\n","\n))\n",' ');
				
					// cout << "rhs_mpi" << rhs.format(outformatLine) << endl;
					
					
					dst.noalias() += alpha * (lhs.Svec(rhs) + cdouble(0,0.5) * lhs.getDt() * lhs.Hvec(rhs));
					
					// cout << "dst_mpi" << dst.format(outformatLine) << endl;
			}
		};
	 
	}
}

//Wrapper for Incomplete LUT solver, in which the compute step is only performed once for a static matrix, so as to save memory
template<typename _Scalar,typename _StorageIndex = int>
class CranknichPreconditioner: public Eigen::IncompleteLUT<_Scalar,_StorageIndex> {
	public:
		typedef typename Eigen::NumTraits<_Scalar>::Real RealScalar;
		
		CranknichPreconditioner():Eigen::IncompleteLUT<_Scalar,_StorageIndex>() { }
		
		template<typename MatrixType>
		explicit CranknichPreconditioner(const MatrixType& mat, const RealScalar& droptol=Eigen::NumTraits<_Scalar>::dummy_precision(), int fillfactor = 10):Eigen::IncompleteLUT<_Scalar,_StorageIndex>(mat,droptol,fillfactor) { }
		
		template<typename MatrixType>
		CranknichPreconditioner& setup(const MatrixType& amat)
		{
		  this->analyzePattern(amat);
		  this->factorize(amat);
		  return *this;
		}
		
		template<typename MatrixType>
		CranknichPreconditioner& compute(const MatrixType& amat) {
			return *this;
		}
};


template<typename _Scalar, typename _StorageIndex = int>
class SubmatPreconditioner: public Eigen::IncompleteLUT<_Scalar,_StorageIndex> {
	typedef Eigen::IncompleteLUT<_Scalar,_StorageIndex> ILUT;
	private:
		int Nmat;
		bool negAng;
		std::vector<csmat> amats;
		std::vector<ILUT*> solvers;
		
		std::vector<int> angids;
	public:
	
		typedef typename Eigen::NumTraits<_Scalar>::Real RealScalar;
		
		SubmatPreconditioner():ILUT() { }
		
		template<typename MatrixType>
		explicit SubmatPreconditioner(const MatrixType& mat, const RealScalar& droptol = Eigen::NumTraits<_Scalar>::dummy_precision(),int fillfactor = 10):ILUT(mat,droptol,fillfactor) { }
		
		
		
		template<typename Htype, typename basistype> 
		SubmatPreconditioner& setup(const DiracBase<Htype,basistype>& H,double dt) {
			int angMax = H.angMax();
			negAng = true;
			cout << "Setting up preconditioner...\nKappamax used: " << angMax << std::endl;
			for(int i = 1; i <= angMax; i++) {
				H.getBasis().getRadial().setState(ki(i),ki(i),1);
				
				
				
				csmat mat1 = H.template S<radial>() - dt * cdouble(0,0.5)  * H.template H0<radial>();
				amats.push_back(mat1);
				
				ILUT* subsolver = new ILUT(mat1); 
				subsolver->analyzePattern(mat1);
				subsolver->factorize(mat1);
				
				solvers.push_back(subsolver);
				
				// cout << i << ": " << solvers.size() - 1 << std::endl;
				
				Nmat = mat1.rows();
				
				H.getBasis().getRadial().setState(ki(-i),ki(-i),1);
				
				csmat mat2 = H.template S<radial>() + dt * cdouble(0,0.5)  * H.template H0<radial>();
				amats.push_back(mat2);
				ILUT* subsolver2 = new ILUT(mat2);
				
				subsolver2->analyzePattern(mat2);
				subsolver2->factorize(mat2);
				
				solvers.push_back(subsolver2);
				
				// cout << -i << ": " << solvers.size() - 1 << std::endl;
				
				Nmat = mat2.rows();
				cout << "setup loop: " << i  << " of " << angMax << std::endl;
			}
			
			setup(amats[0]);
			
			angids = std::vector<int>(H.getBasis().angqN());
			for(int i = 0; i < H.getBasis().angqN(); i++) {
				angids[i] = H.getBasis().indexTransform(i);
			}
			
			return *this;
		}
		
		template<typename Htype, typename basistype> 
		SubmatPreconditioner& setup(const SchrodingerBase<Htype,basistype>& H,double dt) {
			int angMax = H.angMax();
			negAng = false;
			//cout << "Lmax used: " << angMax << std::endl;
			for(int i = 0; i <= angMax; i++) {
				//cout << "Subsolver setup for l = " << i << std::endl;
				H.getBasis().getRadial().setState(li(i),li(i),1);
				
				
				
				csmat mat1 = H.template S<radial>() - dt * cdouble(0,0.5)  * H.template H0<radial>();
				amats.push_back(mat1);
				
				ILUT* subsolver = new ILUT(mat1); 
				subsolver->analyzePattern(mat1);
				subsolver->factorize(mat1);
				
				solvers.push_back(subsolver);
				
				
				Nmat = mat1.rows();
				/*
				H.getBasis().getRadial().setState(ki(-i),ki(-i),1);
				
				csmat mat2 = H.template S<radial>() + dt * cdouble(0,0.5)  * H.template H0<radial>();
				amats.push_back(mat2);
				ILUT* subsolver2 = new ILUT(mat2);
				
				subsolver2->analyzePattern(mat2);
				subsolver2->factorize(mat2);
				
				solvers.push_back(subsolver2);
				
				Nmat = mat2.rows();
				//cout << "setup loop: " << i << std::endl;*/
			}
			
			setup(amats[0]);
			
			angids = std::vector<int>(H.getBasis().angqN());
			for(int i = 0; i < H.getBasis().angqN(); i++) {
				angids[i] = H.getBasis().indexTransform(i);
			}
			
			return *this;
		}
		
		template<typename MatrixType>
		SubmatPreconditioner& setup(const MatrixType& amat)
		{	
			this->analyzePattern(amat);
			this->factorize(amat);
			Nmat = amat.rows();
			return *this;
		}
		
		template<typename MatrixType>
		SubmatPreconditioner& compute(const MatrixType& amat) 
		{
			return *this;
		}
		
		template<typename Rhs>
		Rhs solve(const Rhs& b) const
		{	
			return MPIsolve(b);
			//cout << "Solvers size: " << solvers.size() << std::endl;
				
						   
   
										
   
									  
   
					
   
										 
										
   
		   
   
			if(this->solvers.size() == 0) {
				//cout << "Hello, the solver list size is " << solvers.size() << "and we are in the 0 case" << std::endl;
				Rhs out (b.size());
				int Nblock = b.rows()/Nmat;
				for(int i = 0; i < Nblock; i++) {
					out.segment(i*Nmat,Nmat) = this->ILUT::solve(b.segment(i*Nmat,Nmat));
				}
				
				return out;
				
			}
			else {
				//cout << "Hello, the solver list size is " << solvers.size() << " and we are in the else case" << std::endl;

				Rhs out (b.size());
				int Nblock = b.rows()/Nmat;
				for(int i = 0; i < Nblock; i++) {
					if(negAng) {
						int kappa = ik(angids[i]);
						
						int iid = 2*abs(kappa) - (kappa>0) - 1;
						
						// cout << kappa << " : " << iid << std::endl;
						out.segment(i*Nmat,Nmat) = this->solvers[iid]->solve(b.segment(i*Nmat,Nmat));
					}
					else {
						int l = il(angids[i]);
						
						// cout << l << " : " << i << std::endl;
						
						out.segment(i*Nmat,Nmat) = this->solvers[l]->solve(b.segment(i*Nmat,Nmat));
					}
					
					
				}
				
				return out;
			}
		}
		
		template<typename Rhs>
		Rhs MPIsolve(const Rhs& b) const
		{	
			//cout << "Solvers size: " << solvers.size() << std::endl;
			cmat bblk;
			pseudoscatterVec(b,bblk,Nmat);
			
			// cout << "bblk:\n" << bblk << endl;
			
			cmat outs(bblk.rows(),bblk.cols());
			
			int wrank, wsize;
			
			MPI_Comm_size(MPI_COMM_WORLD, &wsize);
			MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
			
			Rhs out;
			
			if(this->solvers.size() == 0) {
				//cout << "Hello, the solver list size is " << solvers.size() << "and we are in the 0 case" << std::endl;
				// Rhs out (b.size());
				int Nblock = b.rows()/Nmat;
				for(int i = 0; i < Nblock/wsize; i++) {
					outs.col(i) = this->ILUT::solve(bblk.col(i));
				}
				
			}
			else {
				//cout << "Hello, the solver list size is " << solvers.size() << " and we are in the else case" << std::endl;

				// Rhs out (b.size());
				int Nblock = bblk.cols();
				for(int i = 0; i < Nblock; i++) {
					int rankind = wrank * Nblock + i;
					if(negAng) {
						
						int kappa = ik(angids[rankind]);
						
						int iid = 2*abs(kappa) - (kappa>0) - 1;
						
						// cout << kappa << " : " << iid << std::endl;
						outs.col(i) = this->solvers[iid]->solve(bblk.col(i));
					}
					else {
						int l = il(angids[rankind]);
						
						// cout << l << " : " << i << std::endl;
						
						outs.col(i) = this->solvers[l]->solve(bblk.col(i));
					}
					
					
				}
				
			}
			// cout << "outs:\n" << outs << endl;
			allgatherVec(outs,out);
			
			return out;
		}
		
		
};

template <typename slvType>
class SubmatSolver: public slvType {
	private:
		int Nmat;
		
		bool negAng;
		
		std::vector<csmat> amats;
		std::vector<slvType*> solvers;
		std::vector<int> angids;
	public:
	
		typedef double RealScalar;
		
		SubmatSolver():slvType() { }
		
		
		template<typename MatrixType>
		explicit SubmatSolver(const MatrixType& mat, const RealScalar& droptol = Eigen::NumTraits<cdouble>::dummy_precision(),int fillfactor = 10):slvType(mat,droptol,fillfactor) { }
		
		
		template<typename MatrixType>
		SubmatSolver& setup(const MatrixType& amat)
		{	
			this->analyzePattern(amat);
			this->factorize(amat);
			Nmat = amat.rows();
			return *this;
		}
		
		template<typename Htype, typename basistype> 
		SubmatSolver& setup(const DiracBase<Htype,basistype>& H,double dt) {
			int angMax = H.angMax();
			int nmat = H.template S<radial>().rows();
			negAng = true;
			
			//csmat Imat = csmat(nmat,nmat);
			//Imat.setIdentity();
			
			//cout << "Kappamax used: " << angMax << std::endl;
			for(int i = 1; i <= angMax; i++) {
				
				H.getBasis().getRadial().setState(ki(i),ki(i),1);
				
				
				
				csmat mat1 = H.template S<radial>() + dt * cdouble(0,0.5)  * H.template H0<radial>();
				amats.push_back(mat1);
				
				slvType* subsolver = new slvType(mat1); 
				subsolver->analyzePattern(mat1);
				subsolver->factorize(mat1);
				
				solvers.push_back(subsolver);
				
				
				Nmat = mat1.rows();
				
				H.getBasis().getRadial().setState(ki(-i),ki(-i),1);
				
				csmat mat2 = H.template S<radial>() + dt * cdouble(0,0.5)  * H.template H0<radial>();
				amats.push_back(mat2);
				slvType* subsolver2 = new slvType(mat2);
				
				subsolver2->analyzePattern(mat2);
				subsolver2->factorize(mat2);
				
				solvers.push_back(subsolver2);
				
				Nmat = mat2.rows();
				//cout << "setup loop: " << i << std::endl;
			}
			
			setup(amats[0]);
			
			angids = std::vector<int>(H.getBasis().angqN());
			for(int i = 0; i < H.getBasis().angqN(); i++) {
				angids[i] = H.getBasis().indexTransform(i);
			}
			
			return *this;
		}
		
		
		
		template<typename MatrixType>
		SubmatSolver& compute(const MatrixType& amat) 
		{
			return *this;
		}
		
		template<typename Rhs>
		Rhs solve(const Rhs& b) const
		{	
			//cout << "Solvers size: " << solvers.size() << std::endl;
				
			if(this->solvers.size() == 0) {
				//cout << "Hello, the solver list size is " << solvers.size() << "and we are in the 0 case" << std::endl;
				Rhs out (b.size());
				int Nblock = b.rows()/Nmat;
				for(int i = 0; i < Nblock; i++) {
					out.segment(i*Nmat,Nmat) = this->slvType::solve(b.segment(i*Nmat,Nmat));
				}
				
				return out;
				
			}
			else {
				//cout << "Hello, the solver list size is " << solvers.size() << " and we are in the else case" << std::endl;

				Rhs out (b.size());
				int Nblock = b.rows()/Nmat;
				for(int i = 0; i < Nblock; i++) {
					
					int kappa = ik(angids[i]);
					
					int iid = 2*abs(kappa) - (kappa>0) - 1;
					
					//cout << kappa << " : " << iid << std::endl;
					out.segment(i*Nmat,Nmat) = this->solvers[iid]->solve(b.segment(i*Nmat,Nmat));
				}
				
				return out;
			}
		}
		
		
};

template<typename _Scalar>
class SubmatDiagPreconditioner: public Eigen::DiagonalPreconditioner<_Scalar> {
	typedef Eigen::DiagonalPreconditioner<_Scalar> ILUT;
	private:
		int Nmat;
		bool negAng;
		
		std::vector<csmat> amats;
		std::vector<ILUT*> solvers;
	public:
	
		typedef typename Eigen::NumTraits<_Scalar>::Real RealScalar;
		
		SubmatDiagPreconditioner():ILUT() { }
		
		template<typename MatrixType>
		explicit SubmatDiagPreconditioner(const MatrixType& mat, const RealScalar& droptol = Eigen::NumTraits<_Scalar>::dummy_precision(),int fillfactor = 10):ILUT(mat,droptol,fillfactor) { }
		
		template<typename MatrixType>
		SubmatDiagPreconditioner& setup(const MatrixType& amat)
		{	
			this->analyzePattern(amat);
			this->factorize(amat);
			Nmat = amat.rows();
			return *this;
		}
		
		template<typename Htype, typename basistype> 
		SubmatDiagPreconditioner& setup(const DiracBase<Htype,basistype>& H,double dt) {
			int angMax = H.angMax();
			//cout << "Kappamax used: " << angMax << std::endl;
			
			for(int i = 1; i <= angMax; i++) {
				H.getBasis().getRadial().setState(ki(i),ki(i),1);
				
				csmat mat1 = H.template S<radial>() + dt * cdouble(0,0.5)  * H.template H0<radial>();
				amats.push_back(mat1);
				
				ILUT* subsolver = new ILUT(mat1); 
				subsolver->analyzePattern(mat1);
				subsolver->factorize(mat1);
				
				solvers.push_back(subsolver);
				
				
				Nmat = mat1.rows();
				
				H.getBasis().getRadial().setState(ki(-i),ki(-i),1);
				
				csmat mat2 = H.template S<radial>() + dt * cdouble(0,0.5)  * H.template H0<radial>();
				amats.push_back(mat2);
				ILUT* subsolver2 = new ILUT(mat2);
				
				subsolver2->analyzePattern(mat2);
				subsolver2->factorize(mat2);
				
				solvers.push_back(subsolver2);
				
				Nmat = mat2.rows();
				//cout << "setup loop: " << i << std::endl;
			}
			
			setup(amats[0]);
			
			return *this;
		}
		
		
		
		template<typename MatrixType>
		SubmatDiagPreconditioner& compute(const MatrixType& amat) 
		{
			return *this;
		}
		
		template<typename Rhs>
		Rhs solve(const Rhs& b) const
		{	
			//cout << "Solvers size: " << solvers.size() << std::endl;
				
			if(this->solvers.size() == 0) {
				//cout << "Hello, the solver list size is " << solvers.size() << "and we are in the 0 case" << std::endl;
				Rhs out (b.size());
				int Nblock = b.rows()/Nmat;
				for(int i = 0; i < Nblock; i++) {
					out.segment(i*Nmat,Nmat) = this->ILUT::solve(b.segment(i*Nmat,Nmat));
				}
				
				return out;
				
			}
			else {
				//cout << "Hello, the solver list size is " << solvers.size() << " and we are in the else case" << std::endl;

				Rhs out (b.size());
				int Nblock = b.rows()/Nmat;
				for(int i = 0; i < Nblock; i++) {
					
					int kappa = ik(i);
					
					int iid = 2*abs(kappa) - (kappa>0) - 1;
					
					//cout << kappa << " : " << iid << std::endl;
					out.segment(i*Nmat,Nmat) = this->solvers[iid]->solve(b.segment(i*Nmat,Nmat));
				}
				
				return out;
			}
		}
		
		
};

#endif
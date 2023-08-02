#ifndef PROPAGATOR_H
#define PROPAGATOR_H

#include "defs.h"
#include "RtsMat.h"
#include "cWavefunc.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
//#include <Eigen/unsupported/IterativeSolvers>
#include <mpi.h>

#include<chrono>

Eigen::IOFormat outformat(Eigen::FullPrecision,Eigen::DontAlignCols,", ","\n","(","),"," = npy.array((\n","\n))\n",' ');

//Eigen::IOFormat sparseOutformat(Eigen::FullPrecision,Eigen::DontAlignCols,", ","\n","(","),"," = spy.sparse.csr_array(npy.array((\n","\n)))\n",' ');

//The class Propagator was originally going to be used for everything
//In practice I only ever use Cranknich<Htype,basistype,matfree=true>, which has its own entire definition


template<typename chronoType,typename ratio>
void hmsPrint(std::chrono::duration<chronoType, ratio> dt) {
	auto hrs  = std::chrono::duration_cast<std::chrono::hours  >(dt             );
	auto mins = std::chrono::duration_cast<std::chrono::minutes>(dt - hrs       );
	auto secs = std::chrono::duration_cast<std::chrono::seconds>(dt - hrs - mins);
	
	if(hrs.count()>0) cout << hrs.count() << " hours, " << mins.count() << " minutes and " << secs.count() << " seconds";
	else if(mins.count()>0) cout << mins.count() << " minutes and " << secs.count() << " seconds";
	else cout << secs.count() << " seconds";
}

template <typename Derived, typename Htype, typename basistype > 
class Propagator {
	protected:
		Htype* H;
		double t;
		double dt;
	public:
		void doTimeStep() { };
		void doSetup() { };
		bool matfree = false;
		
		wavefunc<basistype> wf;
		wavefunc<basistype> wft;
		Propagator();
		Propagator(Htype& H) {
			this->H = &H;
		}
		
		string dumpfilename;
		
		void propagate(wavefunc<basistype>& psi, double dt, int tmax, int nSave) {
			int Nt = tmax;
			int ncoef = H->S().rows();
			cmat wfcoefs = cmat(ncoef,Nt);
			cmat wfct = cmat(ncoef,1);
			
			wfcoefs.setZero();
			wfct.setZero();
			
			cout << "psi dimensions: " << psi.coefs.rows() << ", " << psi.coefs.cols() << "\n";
			cout << "wfcoefs dimensions: " << wfcoefs.rows() << ", " << wfcoefs.cols() << "\n";
			
			wfcoefs.col(0) = psi.coefs.col(0);
			wf = wavefunc<basistype>(H->getBasis(),wfcoefs);
			wfct = psi.coefs;
			
			wft = wavefunc<basistype>(H->getBasis(),wfct);
			this->dt = dt;
			t = 0;
			cout << "Number of openMP threads at start: " << omp_get_num_threads() << std::endl;
			//Initialises preconditioner for propagation
			setup();
			auto starttime = std::chrono::system_clock::now();
			
			for(int i = 1; i < Nt; i++) {
				t = i*dt;
				if (i%10 == 0) {
					cdouble iprod = (psi*wft)(0,0);
					
					auto currentTime = std::chrono::system_clock::now();
					auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(currentTime - starttime);
					
					auto ETA = (currentTime - starttime)/i * (Nt-i + 1) * (2 * ((Nt - i) / Nt + 1));
					
					cout <<"step: " << i <<", t: " << t << ", <psi0|psi"<<i<<">=" << abs(iprod) << ", Elapsed: ";
					hmsPrint(elapsed); 
					cout << ", ETA: "; 
					hmsPrint(ETA);
					cout <<std::endl << std::endl;
					if (abs(iprod) < 0.5) {
						i = Nt;
					}
				}		
				try {			
					this->timeStep();
					
					wf.coefs.col(i) = wft.coefs;
				}
				catch(...) {
					cout << "Simulation failed, dumping end-of..." << std::endl;
					
					cout << "psitFull" << wf.coefs.format(outformat);
					
					break;
				}
			}
			
			//wf = wavefunc<basistype>(H->getBasis(),wfcoefs);
		}
		
		void setDumpfile(string name) {
			this->dumpfilename = name;
		}
		
		void timeStep()
			{ static_cast<Derived*>(this)->doTimeStep(); }
		void setup()
			{ static_cast<Derived*>(this)->doSetup(); }
		
};


//typedef Eigen::BiCGSTAB<RtsMat<Htype>,CranknichPreconditioner<cdouble> > cranknichSolver;
typedef Eigen::BiCGSTAB<csmat,CranknichPreconditioner<cdouble> > cranknichSolver;
//typedef Eigen::BiCGSTAB<csmat> cranknichSolver;

template <typename Htype,typename basistype,bool matfree = false>
class Cranknich: public Propagator<Cranknich<Htype, basistype,matfree> ,Htype,basistype> {
	friend class Propagator<Cranknich<Htype,basistype>, Htype, basistype>;
	protected:
		csmat precondMat;
		
		cranknichSolver solver;
		csmat A0;
		csmat B0;
		
	public:
	
		void doTimeStep();
		void doSetup();
		Cranknich():Propagator<Cranknich,Htype,basistype>() { };
		Cranknich(Htype& H):Propagator<Cranknich,Htype,basistype>(H) {
		};
};

template <typename Htype, typename basistype,bool matfree>
void Cranknich<Htype,basistype,matfree>::doSetup() {
	csmat precondMat = this->H->S() - cdouble(0,0.5) * this->dt * this->H->H0();
	solver.preconditioner().setup(precondMat);
	this->A0 = this->H->S() + cdouble(0,0.5) * this->dt * this->H->H0();
	this->B0 = this->H->S() - cdouble(0,0.5) * this->dt * this->H->H0();

}

template <typename Htype, typename basistype,bool matfree>
void Cranknich<Htype,basistype,matfree>::doTimeStep() {
	csmat dtHt = cdouble(0,0.5) * this->dt * this->H->Ht(this->t);
	
	//csmat Bt = *(this->S) + cdouble(0,0.5) * this->dt * this->H->H(this->t);
	
	
	//cout << At << "\n" << Bt << "\n";
	
	solver.compute(this->A0 + dtHt);
	
	this->wft.coefs = solver.solveWithGuess(this->B0 * this->wft.coefs + dtHt * this->wft.coefs,this->wft.coefs);
	
	cout <<  "iterations: " << solver.iterations() << std::endl;
	
	//cout << *(this->wft->coefs) << "\n";
	//cout << x << "\n";
	
	this->wft.normalize();
}

//Time propagator using only the "matrix-free" interface of Hamiltonian

template <typename Htype,typename basistype>
class Cranknich<Htype,basistype,true>/*: public Propagator<Cranknich<Htype, basistype,true> ,Htype,basistype>*/ {
	//friend class Propagator<Cranknich<Htype,basistype,true>, Htype, basistype>;
	protected:
		//Pointer to Hamiltonian instance to be propagated
		Htype* H;
		
		//csmat precondMat;
		
		//At some point there was a clear reason for this to be a class member, probably vestigial now
		cvec b;
		
		//time and delta time
		double t;
		double dt;
	public:
	
		Eigen::BiCGSTAB<RtsMat<Htype>,SubmatPreconditioner<cdouble> >* solver;
		//Eigen::GMRES<RtsMat<Htype>,SubmatPreconditioner<cdouble> >* solver;
		SubmatSolver<Eigen::SparseLU<csmat> > Sinv_solver;
		
		RtsMat<Htype>* proptest;
		
		bool matfree = false;
		
		wavefunc<basistype> wf;
		wavefunc<basistype> wft;
		
		//void doTimeStep();
		//void doSetup();
		Cranknich();
		Cranknich(Htype& H) {
			this->H = &H;
		}
		
		string dumpfilename;
		
		void timeStep() {
			
			proptest->setTime(t);
			
			// cvec Hpsi = H->H(this->t,this->wft.coefs);
			
			// b = H->S(this->wft.coefs) - dt * cdouble(0,0.5) * Hpsi;
			
			b = H->S(this->wft.coefs) - dt * cdouble(0,0.5) * H->H(this->t,this->wft.coefs);
			
			//cvec wftEst = wft.coefs  - dt * cdouble(0,1.0) * Sinv_solver.solve(Hpsi);// - pow(dt,2) * 0.5 * Sinv_solver.solve(H->H(t,Sinv_solver.solve(H->H(this->t,this->wft.coefs))));
			
			//wftEst = wftEst/wftEst.dot(H->S(wftEst));
			
			//Compute step prepares BiCGSTAB through black magic I don't know the details of, through the RtsMat class
			solver->compute(*proptest);
			
			//This is the actual solve part and gives the next time step
			// cvec c = solver->solveWithGuess(b,cvec(wft.coefs));
			
			
			//cvec c = solver->solveWithGuess(b,wftEst);
			// cvec c = solver->solve(b);
			wft.coefs = cvec(solver->solve(b));
			
			//Do something if the solver doesn't converge. Right now, it just says that happened
			if(solver->info()!=Eigen::ComputationInfo::Success) {
				cout << "Solver failed to converge, took "<< solver->iterations() << " iterations \n";
				
				// cout << "wft " << wft.coefs.format(outformat);
				// cout << "wf " << wf.coefs.format(outformat);
				
				// wft.coefs = c;
				// wft.normalize();
				
				// double dt_old = dt;
				// double t_old = t;
				// dt = dt/2;
				// if(dt < 10e-15) {
					// c = cvec::Zero(c.size());
					// cout << "Smaller time steps failed. Aborting..." << std::endl;
				// }
				// else {
					// cout << "BiCGSTAB Failed to converge within iterations limit at time " << t_old << ". Attempting smaller time step of " << dt << std::endl;
					// timeStep();
					// t = t + dt;
					// timeStep();
					// dt = dt_old;
					// t = t_old;
				// }
			}
			
			
			// else {
				// wft.coefs = c;
				// wft.normalize();
			// }
			
			wft.normalize();
			
		}
		
		void setup() {
			
			//precondMat = H->S() - dt * cdouble(0,0.5)  * H->H0();
			
			//proptest.attachHamiltonian(H);
			//proptest.setDt(dt);
			//proptest.setTime(dt);
			//solver.preconditioner().setup(precondMat);
			//Sinv_solver.setup(*H,0);
		}
		
		void propagate(wavefunc<basistype>& psi, double dt, int tmax, int nSave = 1) {
			int Nt = tmax;
			int ncoef = H->Dim();
			cmat wfcoefs = cmat(ncoef,Nt/nSave);
			cmat wfct = cmat(ncoef,1);
			
			wfcoefs.setZero();
			wfct.setZero();
			
			int wrank;
			MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
			
			if(wrank==0) {
				cout << "psi dimensions: " << psi.coefs.rows() << ", " << psi.coefs.cols() << "\n";
				cout << "wfcoefs dimensions: " << wfcoefs.rows() << ", " << wfcoefs.cols() << "\n";
			}
			
			wfcoefs.col(0) = psi.coefs.col(0);
			wf = wavefunc<basistype>(H->getBasis(),wfcoefs);
			wfct = psi.coefs;
			
			wft = wavefunc<basistype>(H->getBasis(),wfct);
			this->dt = dt;
			double Tmax = dt * Nt;
			t = 0;
			
			//Initialises preconditioner for propagation
			this->setup();
			auto starttime = std::chrono::system_clock::now();
			cout << "OpenMP max threads at rank " << wrank << ": " << omp_get_max_threads() << endl;
			cout << "Eigen OpenMP max threads at rank " << wrank << ": " << Eigen::nbThreads() << endl;
			for(int i = 1; i < Nt; i++) {
				t = i*dt;
				if (i%10 == 0) {
					cdouble iprod = (psi*wft)(0,0);
					
					auto currentTime = std::chrono::system_clock::now();
					auto elapsed = (currentTime - starttime);
					
					auto ETA = (currentTime - starttime)/i * (Nt-i + 1) * (2 * ((Nt - i) / Nt + 1));
					
					if(wrank == 0) {
						cout <<"step: " << i <<",\nt: " << t << ",\n<psi0|psi"<<i<<">=" << abs(iprod) << ",\nElapsed: ";
						hmsPrint(elapsed); 
						cout << ",\nETA: "; 
						hmsPrint(ETA);
						cout << "\nIterations: " << solver->iterations() << ", error estimate: " << solver->error() << std::endl;																					 
					}
				}
				// try {			
				this->timeStep();
				
				
				if(i%nSave == 0 && i < Nt && wrank == 0) {
					int saveId = i/nSave;
					
					cout << "Step " << i << " complete, Saving wft to wf col " << saveId << ", wf size" << wf.coefs.cols() << std::endl;
					
					wf.coefs.col(saveId) = wft.coefs;
					
					if(!dumpfilename.empty())
						wft.save(dumpfilename,saveId);
					else {
						cout << "Warning: No dumpfile name!" << std::endl;
					}
				}
				
				
			}
			
		}
		
		void setDumpfile(string name) {
			this->dumpfilename = name;
		}
};


template <typename Htype,typename Htype2,typename basistype>
class CranknichApprox/*: public Propagator<Cranknich<Htype, basistype,true> ,Htype,basistype>*/ {
	//friend class Propagator<Cranknich<Htype,basistype,true>, Htype, basistype>;
	protected:
		Htype* H;
		Htype2* H2;
		double t;
		double dt;
		csmat precondMat;
		
		cvec b;
	public:
	
		Eigen::BiCGSTAB<RtsMat<Htype>,SubmatPreconditioner<cdouble> >* solver;
		Eigen::BiCGSTAB<RtsMat<Htype2>,SubmatPreconditioner<cdouble> >* solver2;
		RtsMat<Htype>* proptest;
		RtsMat<Htype2>* proptest2;
		
		
		wavefunc<basistype> wf;
		wavefunc<basistype> wft;
		
		//void doTimeStep();
		//void doSetup();
		CranknichApprox();
		CranknichApprox(Htype& H, Htype2& H2) {
			this->H = &H;
			this->H2 = &H2;
		}
		
		void propagate(const wavefunc<basistype>& psi, double dt, int tmax) {
			int Nt = tmax;
			int ncoef = H->Dim();
			cmat wfcoefs = cmat(ncoef,Nt);
			cmat wfct = cmat(ncoef,1);
			
			wfcoefs.setZero();
			wfct.setZero();
			
			cout << "psi dimensions: " << psi.coefs.rows() << ", " << psi.coefs.cols() << "\n";
			cout << "wfcoefs dimensions: " << wfcoefs.rows() << ", " << wfcoefs.cols() << "\n";
			
			wfcoefs.col(0) = psi.coefs.col(0);
			wf = wavefunc<basistype>(H->getBasis(),wfcoefs);
			wfct = psi.coefs;
			
			wft = wavefunc<basistype>(H->getBasis(),wfct);
			this->dt = dt;
			t = 0;
			
			//Initialises preconditioner for propagation
			this->setup();
			
			for(int i = 1; i < Nt; i++) {
				t = i*dt;
				if (i%10 == 0) cout <<"step: " << i <<", t: " << t << ", dt: " << dt<<std::endl;
				try {			
					this->timeStep();
				
					wf.coefs.col(i) = wft.coefs;
				}
				catch(...) {
					cout << "Simulation failed, dumping end-of..." << std::endl;
					
					cout << "psitFull" << wf.coefs.format(outformat);
					
					break;
				}
			}
			
		}
		
		void timeStep() {
			
			proptest->setTime(t);
			proptest2->setTime(t);
			
			b = H->S(this->wft.coefs) - dt * cdouble(0,0.5) * H->H(this->t,this->wft.coefs);
			cvec b2 = H2->S(this->wft.coefs) - dt * cdouble(0,0.5) * H2->H(this->t,this->wft.coefs);
			
			solver->compute(*proptest);
			solver2->compute(*proptest2);
			
			cvec cApprox = solver2->solveWithGuess(b2,cvec(wft.coefs));
			cout << "approximation iterations: " << solver2->iterations() << std::endl;
			cvec c = solver->solveWithGuess(cApprox,cvec(wft.coefs));
			wft.coefs = c;
			wft.normalize();
			cout <<  "iterations: " << solver->iterations() << std::endl;
			
		}
		
		void setup() {
			
			//precondMat = H->S() - dt * cdouble(0,0.5)  * H->H0();
			
			//proptest.attachHamiltonian(H);
			//proptest.setDt(dt);
			//proptest.setTime(dt);
			//solver.preconditioner().setup(precondMat);
			
		}
};


template <typename Htype, typename basistype>
class Taylor: public Propagator<Taylor<Htype, basistype> ,Htype,basistype> {
	protected:
		csmat precondMat;
		RtsMat<Htype> rStep;
		
		//Eigen::BiCGSTAB<RtsMat<Htype>,CranknichPreconditioner<cdouble> > solver;
		csmat A0;
		csmat B0;
		
		cvec b;
		
		Eigen::IncompleteLUT<cdouble> LUsolver;
	public:
		void doTimeStep();
		void doSetup();
		Taylor():Propagator<Taylor,Htype,basistype>() { };
		Taylor(Htype& H):Propagator<Taylor,Htype,basistype>(H) {
		};
};




//This REALLY REALLY does not work, do not use.

template <typename Htype, typename basistype>
void Taylor<Htype,basistype>::doTimeStep() {
	
	cvec b;
	cvec c;
	cvec d;
	b.noalias() = this->wft.coefs; 
	c.noalias() = this->wft.coefs;
	d.noalias() = this->wft.coefs;
	
	int deg = 5;
	for(int i = 1; i <= deg; i++) {
		c = -cdouble(0,1.0) * this->dt * (this->H->H(this->t,c)) / i;
		
		//d = LUsolver.solve(c);
		
		b.noalias() += c;
		
	}
	
	
	
	this->wft.coefs = b;
	
	this->wft.normalize();
	
	//cout << std::endl;
}
//Seriously, you'll just suffer a lot if you use this, do not

template <typename Htype, typename basistype>
void Taylor<Htype,basistype>::doSetup() {
	LUsolver.compute(this->H->S());
}

//Gonna try krylov subspace expansion... Sometime.

template <typename Htype, typename basistype>
class Krylov: public Propagator<Krylov<Htype,basistype>, Htype, basistype> {
	
};

template <typename Htype, typename basistype>
class Lancszos: public Propagator<Lancszos<Htype,basistype>,Htype,basistype> {
	protected:
		SubmatSolver<Eigen::SparseLU<csmat> > solver;
		cmat Q;
		cmat T;
	public:
		double t;
		double dt;
		
		Lancszos(Htype& H):Propagator<Lancszos<Htype,basistype>,Htype,basistype>(H) {
			
		}
			
		void doSetup() {
			//As we only need S^-1, we set dt=0 s.t. S - i dt/2 H -> S in submatPreconditioner!
			solver.setup(*(this->H),0);
		}
		
		cvec krylovvec(cvec& v, int n) {
			cvec a;
			if (n == 0) return v;
			else if (n == 1) {
				cvec Hv = this->H->H(t,v);
				
				cvec sHv = solver.solve(Hv);
				
				cdouble nrm = sqrt(abs((Hv.adjoint()*sHv)[0]));
				//cout << n <<", " << nrm << std::endl;
				
				a = sHv/nrm;
				nrm =  a.adjoint() * this->H->S(a);
				a = a/nrm;
				//cout << a.adjoint() * this->H->S(a) << std::endl;
			}
			else {
				cvec Hv = this->H->H(t,krylovvec(v,n-1));
				
				cvec sHv = solver.solve(Hv);
				
				cdouble nrm = sqrt(abs((Hv.adjoint()*sHv)[0]));
				//cout << n <<", " << nrm << std::endl;
				
				
				a = sHv/nrm;
				nrm =  sqrt(abs((a.adjoint() * this->H->S(a))(0,0)));
				a = a/nrm;
				//cout << a.adjoint() * this->H->S(a) << std::endl;
			}
			
			return a;
		}
		
		cmat krylovspace(cvec v, int n) {
			Q = cmat::Zero(v.size(),n);
			T = cmat::Zero(n,n);
			
			//IMPORTANT LANCSZOS TIP: S AND H DO NOT COMMUTE!!
			
			//cdouble vnrm = sqrt(abs((v.adjoint()*this->H->S(v))[0]));
			cdouble vnrm = sqrt(abs((v.adjoint()*v)[0]));
			
			cvec q = v/vnrm;
			
			Q.col(0)=q;
			
			cvec Hq = this->H->H(t,q);
			
			
			cvec z = solver.solve(Hq);
			//cvec z = Hq;
			
			//cout << "z0" << z.format(outformat);
			
			cdouble alpha(0,0);
			cdouble beta(0,0);
			
			for(int i = 0; i < n-1; i++) {
				
				cdouble alpha = (q.adjoint() * Hq)[0];
				
				z -= alpha*q;
				//z -= q.dot(z) * q;
				z -= q.dot(this->H->S(z)) * q;
				//z -= q.dot(this->H->S(z)) * q;
				
				cdouble beta = sqrt(abs((z.adjoint() * this->H->S(z))[0]));
				//cdouble beta = sqrt(abs((z.adjoint() * z)[0]));
				
				q = z/beta;
				
				Q.col(i+1) = q;
				
				T(i,i) = alpha;
				T(i,i+1) = beta;
				T(i+1,i) = beta;
				
				Hq = this->H->H(t,q) - beta * this->H->S(Q.col(i));
				
				//cout << "Hq" << i << " " << Hq;
				
				z = solver.solve(Hq);
				//z = Hq;
				//z = solver.solve(Hq);// - beta * Q.col(i-1);
				
				//cout << "z" << i << " " << z;
				//z = Hq - beta * Q.col(i-1);
			}
			
			cout << "Q" << Q.format(outformat);
			
			cout << "T" << T.format(outformat);
			
			return Q;
		}
		
		cmat& krylovQ() {
			return Q;
		}
		
		cmat& krylovT() {
			return T;
		}
};

template <typename Htype, typename basistype>
class MatItr:  public Propagator<MatItr<Htype, basistype>, Htype,basistype> {
	protected:
		SubmatPreconditioner<cdouble> solver;
	public: 
		MatItr(Htype& H): Propagator<MatItr<Htype,basistype>, Htype, basistype> (H) {
		}
		
		void doSetup() {
			this->solver.setup(*(this->H),this->dt);
		}
		
		void doTimeStep() {
			//First part of crank-nicholson approximant
			cvec b = this->H->S(this->wft.coefs) - this->dt * cdouble(0,0.5) * this->H->H(this->t,this->wft.coefs);
			
			//Matrix iteration method
			
			cvec c = this->solver.solve(b);
			/*cvec d = c;
			for(int i = 1; i < 3; i++) {
				cvec e = this->dt * cdouble(0,0.5) * this->H->Ht(this->t,d);
				d = solver.solve(e);
				c.noalias() += pow(-1,i) * d;
			}*/
			
			this->wft.coefs = c;
			this->wft.normalize();
			cout << this->wft * this->wf[0] << std::endl;
		}
};

#endif

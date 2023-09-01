/*
This class generates matrix elements corresponding to the projection of various operators into a dual kinetic balance basis.
It is also responsible for matrix-vector products performed in this basis. The .h file is pretty big, as a lot of the code would've gotten very repetitive without the use of templates
*/

#ifndef DKBBASIS_H
#define DKBBASIS_H

#include "defs.h"
#include "mpiFuncs.h"
#include "basis.h"
#include "splinehandler.h"
// #include <omp.h>
#include <mpi.h>

class dkbbasis: public basis<dkbbasis>, public splineHandler {
	friend class basis<dkbbasis>;
	//Much of this mess is from the old state machine approach to quantum numbers
	int kappa = -1;
	int kappa1 = 0;
	int kappa2 = 0;
	int matvecul = 0;
	int bdpl = 0;
	int bdpalpha = 0;
	
	//clsmat kappaCache = clsmat(0,0);
	//clsmat dKappaCache = clsmat(0,0);
	
	//Not actually used for anything
	long double (*lastV)(long double);
	
	//Stores dual kinetic balance basis vectors P_0, P_kappa, Q_0, Q_kappa
	clsmat dkbU0 = clsmat(0,0);
	clsmat dkbUk = clsmat(0,0);
	clsmat ddkbU0 = clsmat(0,0);
	clsmat ddkbUk = clsmat(0,0);
	
	//U/L variable names indicate upper and lower components
	clsmat dkbL0 = clsmat(0,0);
	clsmat dkbLk = clsmat(0,0);
	clsmat ddkbL0 = clsmat(0,0);
	clsmat ddkbLk = clsmat(0,0);
	
	//As the basis funcitons here depend on kappa and we are using the formulation described in the paper, we need to store four block matrices for each matrix
	csmat mats0[N_MATS][2];
	csmat matsk0[N_MATS][2];
	csmat mats0k[N_MATS][2];
	csmat matskk[N_MATS][2];
	
	//std::vetor used to store bdpalph mats as the maximum required l is not known at compile time
	std::vector<csmat> bdpalph[2][4];
	
	//For optimization purposes, if current time == prevTime, skip some calculations
	double prevTime = -1.0;
	int prevL = -1;
	
	
	std::vector<std::vector<csmat> > bdpmats[dkbpart::Npts];
	
	std::vector<std::vector<cvec> > bdpvecs[dkbpart::Npts][2];
	
	
	template <dkbpart pp>
	csmat& getbdpmat(int l, int alpha) {
		//cout <<"l: " << l << ", size of bdp0mats: " << bdp0mats.size() << "\n";
		if(alpha >= bdpmats[pp].size()) {
			for(int i = bdpmats[pp].size(); i < alpha+1; i++) {
				bdpmats[pp].push_back(std::vector<csmat>(0));
			}
		}
		
		if(l >= bdpmats[pp][alpha].size()) {
			for(int i = bdpmats[pp][alpha].size(); i < l+1; i++) {
				bdpmats[pp][alpha].push_back(csmat(0,0));
			}
		}
		//cout <<"l: " << l << ", size of bdpmat: " << bdpmats[pp][alpha][l].size() << "\n";
		return bdpmats[pp][alpha][l];
	}
	
	csmat& getbdpmat(int pp, int l, int alpha) {
		//cout <<"l: " << l << ", size of bdp0mats: " << bdp0mats.size() << "\n";
		if(alpha >= bdpmats[pp].size()) {
			for(int i = bdpmats[pp].size(); i < alpha+1; i++) {
				bdpmats[pp].push_back(std::vector<csmat>(0));
			}
		}
		
		if(l >= bdpmats[pp][alpha].size()) {
			for(int i = bdpmats[pp][alpha].size(); i < l+1; i++) {
				bdpmats[pp][alpha].push_back(csmat(0,0));
			}
		}
		//cout <<"l: " << l << ", size of bdpmat: " << bdpmats[pp][alpha][l].size() << "\n";
		return bdpmats[pp][alpha][l];
	}
	
	/*csmat& getbdpmat(int kappa,int l, int alpha) {
		
	}*/
	
	
	
	template <dkbpart pp>
	cvec& getbdpvec(int ul, int cacheId) {
		//cout <<"l: " << l << ", size of bdp0mats: " << bdp0mats.size() << "\n";
		int uli = (ul+1)/2;
		//cout << uli;
		
		if(cacheId >= bdpvecs[pp][uli].size()) {
			for(int i = bdpvecs[pp][uli].size(); i < cacheId+1; i++) {
				bdpvecs[pp][uli].push_back(std::vector<cvec>(1));
				bdpvecs[pp][uli][i][0]=cvec(0);
			}
		}
		//cout <<"cacheId: " << cacheId << ", size of bdpvec: " << bdpvecs[uli][cacheId].size() << "\n";
		return bdpvecs[pp][uli][cacheId][0];
	}
	
	
	template <dkbpart pp>
	cvec& getbdpvec(int ul, int cacheId, int l) {
		//cout <<"l: " << l << ", size of bdp0mats: " << bdp0mats.size() << "\n";
		int uli = (ul+1)/2;
		//cout << uli;
		//cout << cacheId << " " << l << "vecvec size " << bdpvecs[pp][uli].size() << std::endl;
		
		
		if(cacheId >= bdpvecs[pp][uli].size()) {
			for(int i = bdpvecs[pp][uli].size(); i < cacheId+1; i++) {
				bdpvecs[pp][uli].push_back(std::vector<cvec> (0));
				for(int j = 0; j < l+1; j++) {
					//cout << i << " " << j << "\n";
					bdpvecs[pp][uli][cacheId].push_back(cvec(0));
				}
				
				//cout << "new vecvec size: " << bdpvecs[pp][uli].size() << ", new vector size: " << bdpvecs[pp][uli][cacheId].size() << std::endl;
				
			}
		}
		
		//cout << "sizeof vector cache: " << bdpvecs[pp][uli][cacheId].size() << "\n";
		
		if(l >= bdpvecs[pp][uli][cacheId].size()) {
			
				
			for(int i = bdpvecs[pp][uli][cacheId].size(); i < l+1; i++) {
				bdpvecs[pp][uli][cacheId].push_back(cvec(0));
			}
		}	
		
	
		//cout <<"cacheId: " << cacheId << ", l: " << l << "\n";
		return bdpvecs[pp][uli][cacheId][l];
	}
	
	
	
	void clearbdpmats() {
		for(int p = 0; p < dkbpart::Npts; p++) {
			for(int i = 0; i < bdpmats[p].size(); i++) {
				for(int j = 0; j < bdpmats[p][i].size(); j++) {
					bdpmats[p][i][j] = csmat(0,0);
				}
			}
		}
	}
	void clearbdpvecs() {
		for(int p = 0; p < dkbpart::Npts; p++) {
			for(int i = 0; i < 2; i++) {
				for(int j = 0; j < bdpvecs[p][i].size(); j++) {
					for(int k = 0; k < bdpvecs[p][i][j].size(); k++) {
						bdpvecs[p][i][j][k] = cvec(0);
					}
				}
			}
		}
	}
	
	bdpft Calph;
	cvec sumCalph;
	
	public:

		const Potential<beyondDipolePulse>* vExt;								   
		void clearDkbCache() {
			dkbU0 = clsmat(0,0);
			dkbUk = clsmat(0,0);
			ddkbU0 = clsmat(0,0);
			ddkbUk = clsmat(0,0);
			
			dkbL0 = clsmat(0,0);
			dkbLk = clsmat(0,0);
			ddkbL0 = clsmat(0,0);
			ddkbLk = clsmat(0,0);
		}
		
		void cldkbm(int l) {
			for(int alpha = 0; alpha < 6; alpha++) {
				for(int pp = 0; pp < 4; pp++) {
					getbdpmat(pp, l, alpha) = csmat(0,0);
				}
			}
		}
	
		void prepbdpvecs(int cacheSize, int lmax) {
			for(int p = 0; p < dkbpart::Npts; p++) {
				for(int uli = 0; uli < 2; uli++) {
					bdpvecs[p][uli] = std::vector<std::vector<cvec> >(cacheSize);
					for(int cid = 0; cid < cacheSize; cid++) {
						bdpvecs[p][uli][cid] = std::vector<cvec>(lmax);
					}
				}
			}
		}
		
		csmat angidmat;
		csmat kappasmat;
		
		dkbbasis(vec& t, int splinedeg);
		void setst(int i) {
			this->kappa = ik(i);
			// cout << "State set with index " << i <<", kappa equals " << this->kappa << std::endl;
		}

		void setst(int kappa1, int kappa2, int ul) {
			this->kappa1 = ik(kappa1);
			this->kappa = this->kappa1;
			this->kappa2 = ik(kappa2);
			this->matvecul = ul;
		}
		
		
		void setst(int kappa1, int kappa2, int ul,int l) {
			this->kappa1 = ik(kappa1);
			this->kappa2 = ik(kappa2);
			this->matvecul = ul;
			//if(l!=this->bdpl) this->sumCalph = cvec(0);
			this->bdpl = l;
		}
		
		void setst(int kappa1, int kappa2, int ul,int l, int alpha) {
			this->kappa1 = ik(kappa1);
			this->kappa2 = ik(kappa2);
			this->matvecul = ul;
			if(l!=this->bdpl) this->Calph = bdpft(6,0);
			this->bdpl = l;
			this->bdpalpha = alpha;
		}
		
		clsmat dkbCache(lvec& x, int ul);
		clsmat& dkb0Cache(lvec& x, int ul);
		clsmat& dkbkCache(lvec& x, int ul);
		
		clsmat dDkbCache(lvec& x, int ul);
		clsmat& dDkb0Cache(lvec& x, int ul);
		clsmat& dDkbkCache(lvec& x, int ul);
		
		csmat& olmat();
		csmat& ddmat(int dn);
		csmat& Emat(long double (*V)(long double));
		csmat& p1mat();
		// csmat& H0mat();
		
		csmat& kappamat();
		csmat& ulcmat();
		csmat& dpalphmat(int k1,int k2, int ul);
		csmat& bdpalphmat(int k1,int k2, int ul);
		 
		csmat& bdpalphmat(int k1,int k2, int ul,int l, int alpha, const Potential<beyondDipolePulse>& vExt);
		csmat& bdpalphmat(int k1,int k2, int ul,int l, const Potential<beyondDipolePulse>& vExt);
		
		csmat& componentOlmat(int k1, int k2, int comp);
			
		template<matname M>		
		csmat& get0mat(int ul = 0) {
			if(ul == LOWER) return mats0[M][1];
			return mats0[M][0]; 
		}
		
		template<matname M>		
		csmat& getk0mat(int ul = 0) {
			if(ul == LOWER) return matsk0[M][1];
			return matsk0[M][0];
		}
		
		template<matname M>
		csmat& get0kmat(int ul = 0) {
			if(ul == LOWER) return mats0k[M][1];
			return mats0k[M][0];
		}
		template<matname M>
		csmat& getkmat(int ul = 0) {
			return getk0mat<M>(ul);
		}
		template<matname M>		
		csmat& getkkmat(int ul = 0) {
			if(ul == LOWER) return matskk[M][1];
			return matskk[M][0];
		}
		
		template <matname M>
		csmat& getafm(int k) {
			switch(k) {
				case 0:
					return get0mat<M>();
					break;
				case 1:
					return getkmat<M>();
					break;
				case 2:
					return getkkmat<M>();
					break;
				default:
					return getmat<M>();
					break;
			}
		}
		
		template <matname M>
		csmat& getafm(int k, int ul,bool asym=false) {
			if(!asym) {
				switch(k) {
					case 0:
						return get0mat<M>(ul);
						break;
					case 1:
						return getkmat<M>(ul);
						break;
					case 2:
						return getkkmat<M>(ul);
						break;
					default:
						return getmat<M>();
						break;
				}
			} else {
				switch(k) {
					case 0:
						return get0mat<M>(ul);
						break;
					case 1:
						return get0kmat<M>(ul);
						break;
					case 2:
						return getk0mat<M>(ul);
						break;
					case 3:
						return getkkmat<M>(ul);
						break;
					default:
						return getmat<M>();
						break;
				}
			}
		}
		
		
		template <matname M>
		void bcastDkbMats(int origin, bool asym) {
			
			int wsize;
			int wrank;
			MPI_Comm_size(MPI_COMM_WORLD, &wsize);
			MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
			
			
			if(!asym) {
				for(int k = 0; k < 3; k++) {
					csmat& dkbmat = this->getafm<M>(k);
					bcastSparseMat(dkbmat,origin,wsize);
				}
			}
			else {
				for(int k = 0; k < 4; k++) {
					csmat& dkbmatU = this->getafm<M>(k,UPPER,true);
					// if(wrank!=origin) cout << "is dkbmatU cached? " << isCached(dkbmatU) << endl;
					
					cout << "dkbmatU" <<k << " before bcast " << dkbmatU << endl;
					cout << "Memory address: " << &dkbmatU << endl;
					bcastSparseMat(dkbmatU,origin,wsize);
					cout << "dkbmatU" <<k << " after bcast " << dkbmatU << endl;
					cout << "Memory address: " << &dkbmatU << endl;
					
					csmat& dkbmatL = this->getafm<M>(k,LOWER,true);
					if(wrank!=origin) cout << "is dkbmatL cached? " << isCached(dkbmatU) << endl;
					
					cout << "dkbmatL" <<k << " before bcast " << dkbmatL << endl;
					cout << "Memory address: " << &dkbmatL << endl;
					bcastSparseMat(dkbmatL,origin,wsize);
					cout << "dkbmatL" <<k << " after bcast " << dkbmatL << endl;
					cout << "Memory address: " << &dkbmatL << endl;
				}
			}
		}
		
		template<matname M>
		cvec matvec(const cvec& v);
		
		template<matname M>
		inline void cacheMatvec(const cvec& v, cvec& w, int cacheId, cdouble coef);
		/*cvec matvec<dpa>(const cvec& v) {
			csmat& dpa00 = get0mat<dpa>();
			csmat& dpa0k = dpalph0k;
			csmat& dpak0 = dpalphk0;
			csmat& dpakk = getkkmat<dpa>();
			
			return dpa00*v + kappa1 * dpa0k * v + kappa2 * dpak0 * v + kappa1 * kappa2 * dpakk * v;
		}*/
		
		template<matname M>
		inline void cacheMatvec(const cvec& v, cvec& w, int cacheId, cdouble coef, int k);
		
		
		template<matname M>
		inline void cacheMatvec(const cvec& v, cvec& w, int cacheId, cdouble coef, int k1, int k2, int ul);
		
		template<matname M>
		inline void cacheMatvec(const cvec& v, cvec& w, int cacheId, cdouble coef, int k1, int k2, int ul, int l);
		
		
		template<matname M>
		void precacheMatvec(const cvec& v, int cacheSize);
		
		template<matname M>
		void precacheMatvec(const cvec& v, int cacheSize, int lmax);
		
		void matmat_impl(std::integral_constant<matname,matname::H0> c, const cmat& m, cmat& w, const cvec& params) {
	
			csmat& H0_0 = getH0mat(0);
			csmat& H0_k = getH0mat(1);
			csmat& H0_k2 = getH0mat(2);
			csmat& H0_k3 = getH0mat(3);
			
			
			iarray kappas = ik(params.real().cast<int>());
			
			// cout << "H0_0 dims: (" << H0_0.rows() << ", " << H0_0.cols() << ")\n";
			// cout << "H0_k dims: (" << H0_k.rows() << ", " << H0_k.cols() << ")\n";
			// cout << "H0_k2 dims: (" << H0_k2.rows() << ", " << H0_k2.cols() << ")\n";
			// cout << "H0_k3 dims: (" << H0_k3.rows() << ", " << H0_k3.cols() << ")\n";
			
			// cout << "m dims: (" << m.rows() << ", " << m.cols() << ")\n";
			
			// cout << "params dims: (" << params.rows() << ", " << params.cols() << ")\n";
			
				
			///WHY does this not parallelize with OpenMP?
			// Eigen::initParallel();

			// cout << "H0_0.nonZeros() * m.cols() = " <<  H0_0.nonZeros() * m.cols() << endl;
			// cout << "H0_k.nonZeros() * m.cols() = " <<  H0_k.nonZeros() * m.cols() << endl;
			// cout << "H0_k2.nonZeros() * m.cols() = " << H0_k2.nonZeros() * m.cols() << endl;
			// cout << "H0_k3.nonZeros() * m.cols() = " << H0_k3.nonZeros() * m.cols() << endl;
			
			// #ifdef EIGEN_HAS_OPENMP
			// cout << "EIGEN_HAS_OPENMP TRUE" << endl;
			// #endif
			
			// cout << "number of Eigen threads " << Eigen::nbThreads() << endl;
			
			
			w.noalias() += H0_0 * m;
			w.noalias() += ((H0_k * m).array().rowwise() * kappas.cast<cdouble>().transpose()).matrix();
			w.noalias() += ((H0_k2 * m).array().rowwise() * kappas.pow(2).cast<cdouble>().transpose()).matrix();
			w.noalias() += ((H0_k3 * m).array().rowwise() * kappas.pow(3).cast<cdouble>().transpose()).matrix();

		}
		
		// void matmat_MPI(std::integral_constant<matname,matname::H0> c, const cmat& m, cmat& w, const cvec& params)  {
			// int wrank, wsize;
			
			// MPI_Comm_size(MPI_COMM_WORLD, &wsize);
			// MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
			
			
			
		// }
		
		void matmat_impl(std::integral_constant<matname,matname::dpa> c, const cmat& m, cmat& w, const csmat& thmat, const cvec& params, int l, int ul) {
			int wrank;
			MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
			
			csmat& dpalph0  = get0mat<dpa>(ul);
			csmat& dpalph0k = get0kmat<dpa>(ul);
			csmat& dpalphk0 = getk0mat<dpa>(ul);
			csmat& dpalphkk = getkkmat<dpa>(ul);
			
			int localNth = thmat.rows();
			
			// cout << "l array: " << larray << std::endl;
			
			// if(!isCached(angidmat)) {
				// angidmat = csmat(thmat.rows(),params.rows());
				// for(int i = 0; i < params.rows(); i++) {
					// angidmat.insert(params.real().cast<int>()(i),i) = 1;
				// }
			// }
			
			if(!isCached(kappasmat)) {
				iarray kappas = ik(params.real().cast<int>());
				kappasmat = csmat(kappas.rows(),kappas.rows());
				kappasmat.setIdentity();
				kappasmat.diagonal() = kappas.matrix().cast<cdouble>();
			}
			
			iarray localKappas = ik(params.segment(wrank*localNth,localNth).real().cast<int>());
			csmat localKappasmat = csmat(localNth,localNth);
			localKappasmat.setIdentity();
			localKappasmat.diagonal() = localKappas.matrix().cast<cdouble>();
			
			// cout << "kappasmat\n" << kappasmat << endl;
			
			// cout << "kappasmat local\n" << localKappasmat << endl;
			
			// cout << "ul: " << ul << std::endl;
			
			// cout << "dpalph0  dims: (" << dpalph0 .rows() << ", " << dpalph0 .cols() << ")\n";
			// cout << "dpalph0k dims: (" << dpalph0k.rows() << ", " << dpalph0k.cols() << ")\n";
			// cout << "dpalphk0 dims: (" << dpalphk0.rows() << ", " << dpalphk0.cols() << ")\n";
			// cout << "dpalphkk dims: (" << dpalphkk.rows() << ", " << dpalphkk.cols() << ")\n";
			
			// cout << "m dims: (" << m.rows() << ", " << m.cols() << ")\n";
			// cout << "w dims: (" << w.rows() << ", " << w.cols() << ")\n";
			// cout << "kappasmat dims: (" << kappasmat.rows() << ", " << kappasmat.cols() << ")\n";
			
			// cout << "thmat dims: (" << thmat.rows() << ", " << thmat.cols() << ")\n";
			
			// cout << std::endl;
			
			// cout << "dpalph0 :\n" << dpalph0  << std::endl;
			// cout << "dpalph0k:\n" << dpalph0k << std::endl;
			// cout << "dpalphk0:\n" << dpalphk0 << std::endl;
			// cout << "dpalphkk:\n" << dpalphkk << std::endl;
			
			
			// w.transposeInPlace();

			// cout << "dpa00*m dims: (" << (thmat * (dpalph0 * m).transpose()).rows() << ", "  << (thmat * (dpalph0 * m).transpose()).cols() << ")" << endl;
			// cout << "dpak0*m dims: (" << ((kappasmat * thmat * (dpalphk0 * m).transpose()).transpose()).rows() 
									  // << ", "  << ((kappasmat * thmat * (dpalphk0 * m).transpose()).transpose()).cols() << ")" << endl;
			// cout << "dpa0k*m dims: (" << ((thmat * kappasmat.middleRows(wrank*thmat.rows(),thmat.rows()) * (dpalph0k * m).transpose()).transpose()).rows() 
									  // << ", "  << ((thmat * kappasmat.middleRows(wrank*thmat.rows(),thmat.rows()) * (dpalph0k * m).transpose()).transpose()).cols() << ")" << endl;
			// cout << "dpakk*m dims: (" << ((kappasmat * thmat * kappasmat.middleRows(wrank*thmat.rows(),thmat.rows()))).rows() 
									  // << ", "  << ((kappasmat * thmat * kappasmat.middleRows(wrank*thmat.rows(),thmat.rows()))).cols() << ")" << endl;

			// cmat mat1 = dpalph0k * m;
			
			// cout << "mat1 dims: (" << mat1.rows() << ", " << mat1.cols() << ")" << endl; 
			
			// cmat mat2 = kappasmat * mat1.transpose();
			
			// cout << "mat2 dims: (" << mat2.rows() << ", " << mat2.cols() << ")" << endl; 
			
			// cmat mat3 = thmat * mat2;
			
			// cout << "mat3 dims: (" << mat3.rows() << ", " << mat3.cols() << ")" << endl; 
		
			if(ul == LOWER) {
				w.noalias() += (thmat * (dpalph0 * m).transpose());
				w.noalias() += (localKappasmat * thmat * (dpalphk0 * m).transpose());
				
				w.noalias() += (thmat * kappasmat * (dpalph0k * m).transpose()); // (thmat * ((kappasmat.middleRows(wrank*thmat.cols(),thmat.cols()) * (dpalph0k * m)))).transpose();
				w.noalias() += ((localKappasmat * thmat * kappasmat) * (dpalphkk * m).transpose());
			}

			if(ul == UPPER) {
				w.noalias() += (thmat * (dpalph0 * m).transpose());
				w.noalias() += (localKappasmat * thmat * (dpalphk0 * m).transpose());
				w.noalias() += (thmat * kappasmat * (dpalph0k * m).transpose());
				w.noalias() += (localKappasmat * thmat * kappasmat * (dpalphkk * m).transpose());
			}

			// w.transposeInPlace();
		}
		
		void matmat_impl_reduce(const cmat& m, cmat& w, const csmat& thmat, const cvec& params, int l, int ul) {
			int wrank;
			MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
			
			csmat& dpalph0  = get0mat<dpa>(ul);
			csmat& dpalph0k = get0kmat<dpa>(ul);
			csmat& dpalphk0 = getk0mat<dpa>(ul);
			csmat& dpalphkk = getkkmat<dpa>(ul);
			
			
			// cout << "l array: " << larray << std::endl;
			
			// if(!isCached(angidmat)) {
				// angidmat = csmat(thmat.rows(),params.rows());
				// for(int i = 0; i < params.rows(); i++) {
					// angidmat.insert(params.real().cast<int>()(i),i) = 1;
				// }
			// }
			
			if(!isCached(kappasmat)) {
				iarray kappas = ik(params.real().cast<int>());
				kappasmat = csmat(kappas.rows(),kappas.rows());
				kappasmat.setIdentity();
				kappasmat.diagonal() = kappas.matrix().cast<cdouble>();
			}
			
			// cout << "localth0, localNth: " << localth0 << ", " << localNth << endl;
			
			iarray localKappas = ik(params.segment(localth0,localNth-localth0).real().cast<int>());
			csmat localKappasmat = csmat(localNth-localth0,localNth-localth0);
			localKappasmat.setIdentity();
			localKappasmat.diagonal() = localKappas.matrix().cast<cdouble>();
			
			// cout << "kappasmat\n" << kappasmat << endl;
			
			// cout << "kappasmat local\n" << localKappasmat << endl;
			
			// cout << "ul: " << ul << std::endl;
			
			// cout << "dpalph0  dims: (" << dpalph0 .rows() << ", " << dpalph0 .cols() << ")\n";
			// cout << "dpalph0k dims: (" << dpalph0k.rows() << ", " << dpalph0k.cols() << ")\n";
			// cout << "dpalphk0 dims: (" << dpalphk0.rows() << ", " << dpalphk0.cols() << ")\n";
			// cout << "dpalphkk dims: (" << dpalphkk.rows() << ", " << dpalphkk.cols() << ")\n";
			
			// cout << "m dims: (" << m.rows() << ", " << m.cols() << ")\n";
			// cout << "w dims: (" << w.rows() << ", " << w.cols() << ")\n";
			// cout << "kappasmat dims: (" << kappasmat.rows() << ", " << kappasmat.cols() << ")\n";
			// cout << "localKappasmat dims: (" << localKappasmat.rows() << ", " << localKappasmat.cols() << ")\n";
			
			// cout << "thmat dims: (" << thmat.rows() << ", " << thmat.cols() << ")\n";
			
			// cout << std::endl;
			
			// // cout << "dpalph0 :\n" << dpalph0  << std::endl;
			// // cout << "dpalph0k:\n" << dpalph0k << std::endl;
			// // cout << "dpalphk0:\n" << dpalphk0 << std::endl;
			// // cout << "dpalphkk:\n" << dpalphkk << std::endl;
			
			
			
			// cout << "dpa00*m dims: (" << (thmat * (dpalph0 * m).transpose()).rows() << ", "  
									  // << (thmat * (dpalph0 * m).transpose()).cols() << ")" << endl;
									  
			// cout << "dpak0*m dims: (" << ((kappasmat * thmat * (dpalphk0 * m).transpose())).rows() << ", "  
									  // << ((kappasmat * thmat * (dpalphk0 * m).transpose())).cols() << ")" << endl;
									  
			// cout << "dpa0k*m dims: (" << (thmat * localKappasmat * (dpalph0k * m).transpose()).rows() << ", "  
									  // << (thmat * localKappasmat * (dpalph0k * m).transpose()).cols() << ")" << endl;
			
			// cout << "dpakk*m dims: (" << ((kappasmat * thmat * localKappasmat) * (dpalphkk * m).transpose()).rows() << ", "  
									  // << ((kappasmat * thmat * localKappasmat) * (dpalphkk * m).transpose()).cols() << ")" << endl;

			// cmat mat1 = dpalph0k * m;
			
			// cout << "mat1 dims: (" << mat1.rows() << ", " << mat1.cols() << ")" << endl; 
			
			// cmat mat2 = localKappasmat * mat1.transpose();
			
			// cout << "mat2 dims: (" << mat2.rows() << ", " << mat2.cols() << ")" << endl; 
			
			// cmat mat3 = thmat * mat2;
			
			// cout << "mat3 dims: (" << mat3.rows() << ", " << mat3.cols() << ")" << endl; 
			// w.transposeInPlace();

			if(ul == LOWER) {
				w.noalias() += (thmat * (dpalph0 * m).transpose());
				w.noalias() += (kappasmat * thmat * (dpalphk0 * m).transpose());
				
				w.noalias() += (thmat * localKappasmat * (dpalph0k * m).transpose()); // (thmat * ((kappasmat.middleRows(wrank*thmat.cols(),thmat.cols()) * (dpalph0k * m)))).transpose();
				w.noalias() += ((kappasmat * thmat * localKappasmat) * (dpalphkk * m).transpose());
			}

			if(ul == UPPER) {
				w.noalias() += (thmat * (dpalph0 * m).transpose());
				w.noalias() += (kappasmat * thmat * (dpalphk0 * m).transpose());
				w.noalias() += (thmat * localKappasmat * (dpalph0k * m).transpose());
				w.noalias() += (kappasmat * thmat * localKappasmat * (dpalphkk * m).transpose());
			}

			// w.transposeInPlace();
		}
		
		int locall0,localNl,localth0,localNth;
		
		void slp(int a1,int a2, int b1, int b2) {
			localth0 = a1;
			localNth = a2;
			locall0 = b1;
			localNl = b2;
		}
		
		int prevul = 0;
		
		void matmat_impl(std::integral_constant<matname,matname::bdpa> c, const cmat& m, cmat& w, const csmat& thmat, const cvec& params, int l, int ul) {
			int wrank;
			MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
			
			// cout << "l array: " << larray << std::endl;
			
			// if(!isCached(angidmat)) {
				// angidmat = csmat(thmat.rows(),params.rows());
				// for(int i = 0; i < params.rows(); i++) {
					// angidmat.insert(params.real().cast<int>()(i),i) = 1;
				// }
			// }
			
			// cout << "Current time: " << vExt->getTime() << endl;
			
			if(vExt->getTime() != prevTime) {
				for(int pp = 0; pp < dkbpart::Npts; pp++) {
					bdpalph[0][pp] = std::vector<csmat>(localNl - locall0);
					bdpalph[1][pp] = std::vector<csmat>(localNl - locall0);
					for(int ll = locall0; ll < localNl; ll++) {
						bdpalph[0][pp][ll-locall0] = csmat(radqN(),radqN());
						bdpalph[1][pp][ll-locall0] = csmat(radqN(),radqN());
					}
				}
			
		//		prevL = bdpl;
				bdpft pT = vExt->template axialPart<axis::t>(vExt->getTime());
				#pragma omp parallel for collapse(2)
				for(int ll = locall0; ll < localNl; ll++) {
					for(int pp = 0; pp < dkbpart::Npts; pp++) {
						for(int alpha = 0; alpha < 6; alpha++) {
							if(ll%2 != alpha%2) {
								// cout << "ll: " << ll << ", alpha: " << alpha << endl;
								
								csmat& dpalph = getbdpmat(pp,ll,alpha);
								
								
								bdpalph[0][pp][ll-locall0] += (pT(alpha,0).real() * dpalph);
								
							}
						}
						bdpalph[1][pp][ll-locall0] = csmat(bdpalph[0][pp][ll-locall0].adjoint());
					
					}
				}
			}
			
			
			// if(vExt->getTime()!=prevTime || bdpl!=prevL || ul!=prevul) {
				// prevL = bdpl;
				// prevul = ul;
				
				// bdpalph00 = csmat(radqN(),radqN());
				// bdpalph0k = csmat(radqN(),radqN());
				// bdpalphk0 = csmat(radqN(),radqN());
				// bdpalphkk = csmat(radqN(),radqN());
				
				// for(int alpha = 0; alpha < 6; alpha++) {
					// if(bdpl%2 != alpha%2) {
						// csmat& dpalph0  = getbdpmat<p00>(bdpl,alpha);
						// csmat& dpalph0k = getbdpmat<p0k>(bdpl,alpha);
						// csmat& dpalphk0 = getbdpmat<pk0>(bdpl,alpha);
						// csmat& dpalphkk = getbdpmat<pkk>(bdpl,alpha);
						
						// double pT = (double) vExt->template axialPart<axis::t>(vExt->getTime())(alpha,0).real();
						
						// bdpalph00 += (pT *  dpalph0);
						// bdpalph0k += (pT * dpalphk0);
						// bdpalphk0 += (pT * dpalph0k);
						// bdpalphkk += (pT * dpalphkk);
					// }
				// }
			// }
			
			if(!isCached(kappasmat)) {
				iarray kappas = ik(params.real().cast<int>());
				kappasmat = csmat(kappas.rows(),kappas.rows());
				kappasmat.setIdentity();
				kappasmat.diagonal() = kappas.matrix().cast<cdouble>();
			}
			
			iarray localKappas = ik(params.segment(localth0,localNth-localth0).real().cast<int>());
			csmat localKappasmat = csmat(localNth-localth0,localNth-localth0);
			localKappasmat.setIdentity();
			localKappasmat.diagonal() = localKappas.matrix().cast<cdouble>();
			
			// cout << "kappasmat\n" << kappasmat << endl;
			
			// cout << "kappasmat local\n" << localKappasmat << endl;
			
			// cout << "ul: " << ul << std::endl;
			
			
			// if(ul==UPPER) {
				// cout << "bdpalph00 dims: (" << bdpalph[0][0][l-locall0].rows() << ", " << bdpalph[0][0][l-locall0].cols() << ")\n";
				// cout << "bdpalph0k dims: (" << bdpalph[0][1][l-locall0].rows() << ", " << bdpalph[0][1][l-locall0].cols() << ")\n";
				// cout << "bdpalphk0 dims: (" << bdpalph[0][2][l-locall0].rows() << ", " << bdpalph[0][2][l-locall0].cols() << ")\n";
				// cout << "bdpalphkk dims: (" << bdpalph[0][3][l-locall0].rows() << ", " << bdpalph[0][3][l-locall0].cols() << ")\n";
			// }
			
			// if(ul==LOWER) {
				// cout << "bdpalph00 dims: (" << bdpalph[1][0][l-locall0].rows() << ", " << bdpalph[1][0][l-locall0].cols() << ")\n";
				// cout << "bdpalph0k dims: (" << bdpalph[1][1][l-locall0].rows() << ", " << bdpalph[1][1][l-locall0].cols() << ")\n";
				// cout << "bdpalphk0 dims: (" << bdpalph[1][2][l-locall0].rows() << ", " << bdpalph[1][2][l-locall0].cols() << ")\n";
				// cout << "bdpalphkk dims: (" << bdpalph[1][3][l-locall0].rows() << ", " << bdpalph[1][3][l-locall0].cols() << ")\n";
			// }
			
			// cout << "w dims: (" << w.rows() << ", " << w.cols() << ")\n";
			// cout << "kappasmat dims: (" << kappasmat.rows() << ", " << kappasmat.cols() << ")\n";
			
			// cout << "thmat dims: (" << thmat.rows() << ", " << thmat.cols() << ")\n";
			
			// cout << std::endl;
			
			// cout << "bdpalph00:\n" << bdpalph00 << std::endl;
			// cout << "bdpalph0k:\n" << bdpalph0k << std::endl;
			// cout << "bdpalphk0:\n" << bdpalphk0 << std::endl;
			// cout << "bdpalphkk:\n" << bdpalphkk << std::endl;
			
			

			// cout << "bdpa00*m dims: (" << (thmat * (dpalph0 * m).transpose()).rows() << ", "  << (thmat * (dpalph0 * m).transpose()).cols() << ")" << endl;
			// cout << "bdpak0*m dims: (" << ((kappasmat * thmat * (dpalphk0 * m).transpose()).transpose()).rows() 
									  // << ", "  << ((kappasmat * thmat * (dpalphk0 * m).transpose()).transpose()).cols() << ")" << endl;
			// cout << "bdpa0k*m dims: (" << ((thmat * kappasmat.middleRows(wrank*thmat.rows(),thmat.rows()) * (dpalph0k * m).transpose()).transpose()).rows() 
									  // << ", "  << ((thmat * kappasmat.middleRows(wrank*thmat.rows(),thmat.rows()) * (dpalph0k * m).transpose()).transpose()).cols() << ")" << endl;
			// cout << "bdpakk*m dims: (" << ((kappasmat * thmat * kappasmat.middleRows(wrank*thmat.rows(),thmat.rows()))).rows() 
									  // << ", "  << ((kappasmat * thmat * kappasmat.middleRows(wrank*thmat.rows(),thmat.rows()))).cols() << ")" << endl;

			// cmat mat1 = bdpalph[0][1][l-locall0] * m;
			
			// cout << "mat1 dims: (" << mat1.rows() << ", " << mat1.cols() << ")" << endl; 
			
			// cmat mat2 = localKappasmat * mat1.transpose();
			
			// cout << "mat2 dims: (" << mat2.rows() << ", " << mat2.cols() << ")" << endl; 
			
			// cmat mat3 = thmat * mat2;
			
			// cout << "mat3 dims: (" << mat3.rows() << ", " << mat3.cols() << ")" << endl; 
			// // w.transposeInPlace();
			
			// cmat mI = cmat::Constant(m.cols(),m.rows(),1.0);
			// #pragma omp parallel cout << "
			
			// m dims: (" << m.rows() << ", " << m.cols() << ")\n";
			{
				if(ul == LOWER) {
					w.noalias() += (thmat * (bdpalph[1][0][l-locall0] * m).transpose());
					w.noalias() += (thmat * localKappasmat * (bdpalph[1][1][l-locall0] * m).transpose());
					w.noalias() += (kappasmat * thmat * (bdpalph[1][2][l-locall0] * m).transpose());
					w.noalias() += ((kappasmat * thmat * localKappasmat) * (bdpalph[1][3][l-locall0] * m).transpose());
				}

				if(ul == UPPER) {
					w.noalias() += (thmat * (bdpalph[0][0][l-locall0] * m).transpose());
					w.noalias() += (kappasmat * thmat * (bdpalph[0][1][l-locall0] * m).transpose());
					w.noalias() += (thmat * localKappasmat * (bdpalph[0][2][l-locall0] * m).transpose());
					w.noalias() += ((kappasmat * thmat * localKappasmat) * (bdpalph[0][3][l-locall0] * m).transpose());
				}
			}
			// w.transposeInPlace();

			// w.transposeInPlace();
		}
		
		template<matname M>
		void matmat_impl(std::integral_constant<matname,M> c, const cmat& m, cmat& w, const cvec& params) {
			
			csmat& M0 = get0mat<M>();
			csmat& Mk = getkmat<M>();
			csmat& Mkk = getkkmat<M>();
			
			
			iarray kappas = ik(params.real().cast<int>());
			// cout << "l array: " << larray << std::endl;
			
			// cout << "H0_0 dims: (" << H0_0.rows() << ", " << H0_0.cols() << ")\n";
			// cout << "H0_k dims: (" << H0_k.rows() << ", " << H0_k.cols() << ")\n";
			// cout << "H0_k2 dims: (" << H0_k2.rows() << ", " << H0_k2.cols() << ")\n";
			// cout << "H0_k3 dims: (" << H0_k3.rows() << ", " << H0_k3.cols() << ")\n";
			
			// cout << "m dims: (" << m.rows() << ", " << m.cols() << ")\n";
			
			// cout << "params dims: (" << params.rows() << ", " << params.cols() << ")\n";
			
			// cmat Hlml = Hlm.array().rowwise() * params.array().transpose();
			
			
			// cout << "M0v\n" <<  M0 * m << std::endl 
			// << "Mkv\n" << ((Mk * m).array().rowwise() * kappas.cast<cdouble>().transpose()).matrix() << std::endl 
			// << "Mkkv\n" << ((Mkk * m).array().rowwise() * kappas.pow(2).cast<cdouble>().transpose()).matrix() << std::endl;
			
			// cout << "kappas\n" << (kappas.cast<cdouble>().transpose()).matrix() << std::endl;
			// cout << "kappas^2\n" << (kappas.pow(2).cast<cdouble>().transpose()).matrix() << std::endl;
			
			
			
			// #pragma omp parallel 
			// {
				w.noalias() += M0 * m;
				w.noalias() += ((Mk * m).array().rowwise() * kappas.cast<cdouble>().transpose()).matrix();
				w.noalias() += ((Mkk * m).array().rowwise() * kappas.pow(2).cast<cdouble>().transpose()).matrix();
			// }
		}
		
		int rqn();
		
		void clvcs() {
			clearbdpvecs();
		}
		
		template <dkbpart pp>
		void dumpBdpVecs() {
			Eigen::IOFormat outformat(Eigen::FullPrecision,Eigen::DontAlignCols,", ","\n","(","),"," = npy.array((\n","\n))\n",' ');
			for(int uli = 0; uli < 2; uli++) {
				for(int cacheId = 0; cacheId < bdpvecs[pp][uli].size(); cacheId++) {
					for(int l = 0; l < bdpvecs[pp][uli][cacheId].size(); l++) {
						cout << "vcache[" <<uli << "][" <<cacheId<<"][" << l << "] " << bdpvecs[pp][uli][cacheId][l].format(outformat) << std::endl;
					}
				}
			}
		}
};

template<matname M>
inline cvec dkbbasis::matvec(const cvec& v) {
	csmat& M0 = get0mat<M>();
	csmat& Mk = getkmat<M>();
	csmat& Mkk = getkkmat<M>();
	
	return M0 * v + kappa * Mk * v + pow(kappa,2) * Mkk * v;
}


template <>
inline cvec dkbbasis::matvec<ulc>(const cvec& v) {
	csmat& M0 = get0mat<ulc>();
	csmat& Mk = getkmat<ulc>();
	csmat& Mkk = getkkmat<ulc>();
	
	return -2 * pow(SoL,2) * (M0 * v + kappa * Mk * v + pow(kappa,2) * Mkk * v);

}

template <>
inline cvec dkbbasis::matvec<k>(const cvec& v) {
	csmat& M0 = get0mat<k>();
	csmat& Mk = getkmat<k>();
	csmat& Mkk = getkkmat<k>();
	
	return kappa * (M0 * v + kappa * Mk * v + pow(kappa,2) * Mkk * v);

}

template <>
inline cvec dkbbasis::matvec<p1>(const cvec& v) {

	return matvec<dd>(v) + matvec<k>(v);

}

template <>
inline cvec dkbbasis::matvec<dpa>(const cvec& v) {
	csmat& dpalph0 = get0mat<dpa>();
	csmat& dpalph0k = get0kmat<dpa>();
	csmat& dpalphk0 = getk0mat<dpa>();
	csmat& dpalphkk = getkkmat<dpa>();
	
	switch(matvecul) {
		case LOWER:
			return dpalph0.adjoint() * v + kappa2 * dpalphk0.adjoint() * v + kappa1 * dpalph0k.adjoint() * v + kappa1 * kappa2 * dpalphkk.adjoint() * v;
			
			break;
		case UPPER:
			return dpalph0 * v + kappa1 * dpalphk0 * v + kappa2 * dpalph0k * v + kappa1 * kappa2 * dpalphkk * v;
			
			break;
		default:
			return v;
	}
}


template <>
inline cvec dkbbasis::matvec<bdpa>(const cvec& v) {
	
	/*
	csmat& dpalph0  = getbdpmat<p00>(bdpl,bdpalpha);
	csmat& dpalph0k = getbdpmat<p0k>(bdpl,bdpalpha);
	csmat& dpalphk0 = getbdpmat<pk0>(bdpl,bdpalpha);
	csmat& dpalphkk = getbdpmat<pkk>(bdpl,bdpalpha);
	*/
	cvec vOut = cvec::Zero(v.size());
	switch(matvecul) {
		case -1:
			for(int alpha = 0; alpha < 6; alpha++) {
				if(bdpl%2 != alpha%2) {
					vOut.noalias() += (double) vExt->template axialPart<axis::t>(vExt->getTime())(alpha,0).real()
					* (
									   getbdpmat<p00>(bdpl,alpha).adjoint() * v
							+ kappa2 * getbdpmat<p0k>(bdpl,alpha).adjoint() * v
							+ kappa1 * getbdpmat<pk0>(bdpl,alpha).adjoint() * v
				   + kappa1 * kappa2 * getbdpmat<pkk>(bdpl,alpha).adjoint() * v
				   );
				}
			}	
			break;
		case 1:
			for(int alpha = 0; alpha < 6; alpha++) {
				if(bdpl%2 != alpha%2) {
					vOut.noalias() += (double) vExt->template axialPart<axis::t>(vExt->getTime())(alpha,0).real()
					* ( 
									   getbdpmat<p00>(bdpl,alpha) * v
							+ kappa1 * getbdpmat<p0k>(bdpl,alpha) * v
							+ kappa2 * getbdpmat<pk0>(bdpl,alpha) * v
				   + kappa1 * kappa2 * getbdpmat<pkk>(bdpl,alpha) * v
					);
				}
			}
			break;
	}
	return vOut;
}

#define MATRICIZE_V false

template<>
inline void dkbbasis::precacheMatvec<dpa>(const cvec& v, int cacheSize, int lmax) {
	

	
	//int kappamax = (lmax - 1)/2;
	
	if(MATRICIZE_V) {
		csmat* alph[4];
		
		alph[0] = &get0mat<dpa>();
		alph[1] = &get0kmat<dpa>();
		alph[2] = &getk0mat<dpa>();
		alph[3] = &getkkmat<dpa>();
		
		v.reshaped(radqN(),radqN());
		
		cmat vcaches[4][2];
		
		#pragma omp for collapse(2)
		for(int i = 0; i < 4; i++) {
			for(int j = 0; j < 2; j++) {
				if(j==0)
					vcaches[i][j] = alph[i]->adjoint() * v.reshaped(radqN(),cacheSize);
				else 
					vcaches[i][j] = *(alph[i]) * v.reshaped(radqN(),cacheSize);
			}
		}
		
		for(int i = 0; i < 4; i++) {
			for(int j = 0; j < 2; j++) {
				cout << vcaches[i][j] << std::endl << std::endl;
			}
		}
	}
	
	//cout << "Cache size: " << cacheSize << std::endl;
	
	else{
		
	csmat& alph00 = get0mat<dpa>();
	csmat& alph0k = get0kmat<dpa>();
	csmat& alphk0 = getk0mat<dpa>();
	csmat& alphkk = getkkmat<dpa>();
		
	#pragma omp for collapse(2)
		for(int cacheId = 0; cacheId < cacheSize; cacheId++) {
			for(int ul = -1; ul < 2; ul+=2) {
				//if(abs(abs(ik(cacheId) + 0.5) - kappamax - 0.5) <= 0 <= abs(ik(cacheId) + 0.5) + kappamax + 0.5) {
	//				cout << "cacheId: " << cacheId << std::endl;
				cvec& vcache00 = getbdpvec<p00>(ul,cacheId);
				cvec& vcache0k = getbdpvec<p0k>(ul,cacheId);
				cvec& vcachek0 = getbdpvec<pk0>(ul,cacheId);
				cvec& vcachekk = getbdpvec<pkk>(ul,cacheId);
				
				if(!isCached(vcache00)) {
					vcache00 = cvec::Zero(radqN());
					vcachek0 = cvec::Zero(radqN());
					vcache0k = cvec::Zero(radqN());
					vcachekk = cvec::Zero(radqN());


					if(ul == LOWER) {
						vcache00.noalias() += (alph00.adjoint() * v.segment(cacheId*radqN(),radqN()));
						vcachek0.noalias() += (alphk0.adjoint() * v.segment(cacheId*radqN(),radqN()));
						vcache0k.noalias() += (alph0k.adjoint() * v.segment(cacheId*radqN(),radqN()));
						vcachekk.noalias() += (alphkk.adjoint() * v.segment(cacheId*radqN(),radqN()));
					}
					if(ul == UPPER) {
						vcache00.noalias() += (alph00 * v.segment(cacheId*radqN(),radqN()));
						vcachek0.noalias() += (alphk0 * v.segment(cacheId*radqN(),radqN()));
						vcache0k.noalias() += (alph0k * v.segment(cacheId*radqN(),radqN()));
						vcachekk.noalias() += (alphkk * v.segment(cacheId*radqN(),radqN()));							
					}
				}
			}
		}
	}
}



template <>
inline void dkbbasis::cacheMatvec<dpa>(const cvec& v, cvec& w, int cacheId, cdouble coef, int k1, int k2, int ul) {
	cvec& vcache00 = getbdpvec<p00>(ul,cacheId);
	cvec& vcache0k = getbdpvec<p0k>(ul,cacheId);
	cvec& vcachek0 = getbdpvec<pk0>(ul,cacheId);
	cvec& vcachekk = getbdpvec<pkk>(ul,cacheId);
	
	int kk1 = ik(k1);
	int kk2 = ik(k2);
	
	
	//cout << "REEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE" << std::endl;
	if(ul == LOWER) {
		w.noalias() += coef * (vcache00 + kk2 * vcachek0 + kk1 * vcache0k + kk1 * kk2 * vcachekk);
	}
	if(ul == UPPER) {
		w.noalias() += coef * (vcache00 + kk1 * vcachek0 + kk2 * vcache0k + kk1 * kk2 * vcachekk);
	}
	
	//cout << w;
}

/*
template <>
inline void dkbbasis::cacheMatvec<dpa>(const cvec& v, cvec& w, int cacheId, cdouble coef, int k1, int k2, int ul) {
	csmat& dpalph0 = get0mat<dpa>();
	csmat& dpalph0k = get0kmat<dpa>();
	csmat& dpalphk0 = getk0mat<dpa>();
	csmat& dpalphkk = getkkmat<dpa>();
	
	int kk1 = ik(k1);
	int kk2 = ik(k2);
	
		 
	switch(ul) {
		case -1:
			w.noalias() += coef * (dpalph0.adjoint() * v + kk2 * dpalphk0.adjoint() * v + kk1 * dpalph0k.adjoint() * v + kk1 * kk2 * dpalphkk.adjoint() * v);
			
			break;
		case 1:
			w.noalias() += coef * (dpalph0 * v + kk1 * dpalphk0 * v + kk2 * dpalph0k * v + kk1 * kk2 * dpalphkk * v);
			
			break;
	}
}*/




template<>
inline void dkbbasis::precacheMatvec<bdpa>(const cvec& v, int cacheId) {
	cvec& vcache00 = getbdpvec<p00>(matvecul,cacheId);
	cvec& vcache0k = getbdpvec<p0k>(matvecul,cacheId);
	cvec& vcachek0 = getbdpvec<pk0>(matvecul,cacheId);
	cvec& vcachekk = getbdpvec<pkk>(matvecul,cacheId);
	
	//Consider: What if we do the time dependent part only once?
	
	csmat& bdpalph00  = get0mat <bdpa>();
	csmat& bdpalph0k = get0kmat<bdpa>();
	csmat& bdpalphk0 = getk0mat<bdpa>();
	csmat& bdpalphkk = getkkmat<bdpa>();
	
	if(vExt->getTime()!=prevTime || bdpl!=prevL) {
		prevL = bdpl;
		
		bdpalph00 = csmat(radqN(),radqN());
		bdpalph0k = csmat(radqN(),radqN());
		bdpalphk0 = csmat(radqN(),radqN());
		bdpalphkk = csmat(radqN(),radqN());
		
		for(int alpha = 0; alpha < 6; alpha++) {
			if(bdpl%2 != alpha%2) {
				csmat& dpalph0  = getbdpmat<p00>(bdpl,alpha);
				csmat& dpalph0k = getbdpmat<p0k>(bdpl,alpha);
				csmat& dpalphk0 = getbdpmat<pk0>(bdpl,alpha);
				csmat& dpalphkk = getbdpmat<pkk>(bdpl,alpha);
				
				double pT = (double) vExt->template axialPart<axis::t>(vExt->getTime())(alpha,0).real();
				
				bdpalph00 += (pT *  dpalph0);
				bdpalph0k += (pT * dpalphk0);
				bdpalphk0 += (pT * dpalph0k);
				bdpalphkk += (pT * dpalphkk);
			}
		}
	}
	
	if(!isCached(vcache00)) {
		vcache00.noalias() = cvec::Zero(v.size());
		vcachek0.noalias() = cvec::Zero(v.size());
		vcache0k.noalias() = cvec::Zero(v.size());
		vcachekk.noalias() = cvec::Zero(v.size());


			switch(matvecul) {
				case LOWER:
					vcache00.noalias() += (bdpalph00.adjoint() * v);
					vcachek0.noalias() += (bdpalph0k.adjoint() * v);
					vcache0k.noalias() += (bdpalphk0.adjoint() * v);
					vcachekk.noalias() += (bdpalphkk.adjoint() * v);
					
					break;
				case UPPER:
					vcache00.noalias() += (bdpalph00 * v);
					vcachek0.noalias() += (bdpalph0k * v);
					vcache0k.noalias() += (bdpalphk0 * v);
					vcachekk.noalias() += (bdpalphkk * v);
					
					break;
		}
	}
	
}


template<>
inline void dkbbasis::precacheMatvec<bdpa>(const cvec& v, int cacheSize, int lmax) {
	
	//Consider: What if we do the time dependent part only once?
	
	// std::vector<csmat> bdpalph00(lmax);
	// std::vector<csmat> bdpalph0k(lmax);
	// std::vector<csmat> bdpalphk0(lmax);
	// std::vector<csmat> bdpalphkk(lmax);
	
	
	// std::vector<csmat> bdpalph[2][4];
	
	//std::vector<cvec> vparts(cacheSize);
	
	if(vExt->getTime() != prevTime) {
		#pragma omp for
		for(int pp = 0; pp < dkbpart::Npts; pp++) {
			bdpalph[0][pp] = std::vector<csmat>(lmax);
			bdpalph[1][pp] = std::vector<csmat>(lmax);
			for(int l = 0; l < lmax; l++) {
				bdpalph[0][pp][l] = csmat(radqN(),radqN());
				bdpalph[1][pp][l] = csmat(radqN(),radqN());
			}
		}
	
//		prevL = bdpl;
		bdpft pT = vExt->template axialPart<axis::t>(vExt->getTime());
		#pragma omp for collapse(2)
		for(int l = 0; l < lmax; l++) {
			for(int pp = 0; pp < dkbpart::Npts; pp++) {
				for(int alpha = 0; alpha < 6; alpha++) {
					if(l%2 != alpha%2) {
						
						csmat& dpalph = getbdpmat(pp,l,alpha);
						
						// double pT = (double) vExt->template axialPart<axis::t>(vExt->getTime())(alpha,0).real();
						
						bdpalph[0][pp][l] += (pT(alpha,0).real() * dpalph);
						
					}
				}
				bdpalph[1][pp][l] += csmat(bdpalph[0][pp][l].adjoint());
			
			// bdpalph00[l] = csmat(radqN(),radqN());
			// bdpalph0k[l] = csmat(radqN(),radqN());
			// bdpalphk0[l] = csmat(radqN(),radqN());
			// bdpalphkk[l] = csmat(radqN(),radqN());
			}
			// for(int alpha = 0; alpha < 6; alpha++) {
				// if(l%2 != alpha%2) {
					// csmat& dpalph0  = getbdpmat<p00>(l,alpha);
					// csmat& dpalph0k = getbdpmat<p0k>(l,alpha);
					// csmat& dpalphk0 = getbdpmat<pk0>(l,alpha);
					// csmat& dpalphkk = getbdpmat<pkk>(l,alpha);
					
					// // double pT = (double) vExt->template axialPart<axis::t>(vExt->getTime())(alpha,0).real();
					
					// bdpalph00[l] += (pT(alpha,0).real() *  dpalph0);
					// bdpalph0k[l] += (pT(alpha,0).real() * dpalphk0);
					// bdpalphk0[l] += (pT(alpha,0).real() * dpalph0k);
					// bdpalphkk[l] += (pT(alpha,0).real() * dpalphkk);
				// }
			// }
		}
	}
	
	prevTime = vExt->getTime();
	
	int kappamax = (lmax - 1)/2;
	
	#pragma omp for collapse(3)
	for(int cacheId = 0; cacheId < cacheSize; cacheId++) {
		for(int l = 0; l < lmax; l++) {
			for(int ul = -1; ul < 2; ul+=2) {
				if(abs(abs(ik(cacheId) + 0.5) - kappamax - 0.5) <= l <= abs(ik(cacheId) + 0.5) + kappamax + 0.5) {
					cvec& vcache00 = getbdpvec<p00>(ul,cacheId,l);
					cvec& vcache0k = getbdpvec<p0k>(ul,cacheId,l);
					cvec& vcachek0 = getbdpvec<pk0>(ul,cacheId,l);
					cvec& vcachekk = getbdpvec<pkk>(ul,cacheId,l);
					
					if(!isCached(vcache00)) {
						// vcache00.noalias() = cvec::Zero(radqN());
						// vcachek0.noalias() = cvec::Zero(radqN());
						// vcache0k.noalias() = cvec::Zero(radqN());
						// vcachekk.noalias() = cvec::Zero(radqN());


						if(ul == LOWER) {
							vcache00 = (bdpalph[1][0][l] * v.segment(cacheId*radqN(),radqN()));
							vcache0k = (bdpalph[1][1][l] * v.segment(cacheId*radqN(),radqN()));
							vcachek0 = (bdpalph[1][2][l] * v.segment(cacheId*radqN(),radqN()));
							vcachekk = (bdpalph[1][3][l] * v.segment(cacheId*radqN(),radqN()));
						}
						if(ul == UPPER) {
							vcache00 = (bdpalph[0][0][l] * v.segment(cacheId*radqN(),radqN()));
							vcache0k = (bdpalph[0][1][l] * v.segment(cacheId*radqN(),radqN()));
							vcachek0 = (bdpalph[0][2][l] * v.segment(cacheId*radqN(),radqN()));
							vcachekk = (bdpalph[0][3][l] * v.segment(cacheId*radqN(),radqN()));
						}
					}
				}
			}
		}
	}
}


template <>
inline void dkbbasis::cacheMatvec<bdpa>(const cvec& v, cvec& w, int cacheId, cdouble coef) {
	cvec& vcache00 = getbdpvec<p00>(matvecul,cacheId);
	cvec& vcache0k = getbdpvec<p0k>(matvecul,cacheId);
	cvec& vcachek0 = getbdpvec<pk0>(matvecul,cacheId);
	cvec& vcachekk = getbdpvec<pkk>(matvecul,cacheId);
	
	
	
	if(!isCached(vcache00)) {
		precacheMatvec<bdpa>(v,cacheId);
	}
	
	if(matvecul == LOWER) {
		w.noalias() += coef * (vcache00 + kappa2 * vcachek0 + kappa1 * vcache0k + kappa1 * kappa2 * vcachekk);
	}
	if(matvecul == UPPER) {
		w.noalias() += coef * (vcache00 + kappa1 * vcachek0 + kappa2 * vcache0k + kappa1 * kappa2 * vcachekk);
	}
}

template <>
inline void dkbbasis::cacheMatvec<bdpa>(const cvec& v, cvec& w, int cacheId, cdouble coef, int k1, int k2, int ul) {
	cvec& vcache00 = getbdpvec<p00>(ul,cacheId);
	cvec& vcache0k = getbdpvec<p0k>(ul,cacheId);
	cvec& vcachek0 = getbdpvec<pk0>(ul,cacheId);
	cvec& vcachekk = getbdpvec<pkk>(ul,cacheId);
	
	int kk1 = ik(k1);
	int kk2 = ik(k2);
	
	
	if(ul == LOWER) {
		w.noalias() += coef * (vcache00 + kk2 * vcachek0 + kk1 * vcache0k + kk1 * kk2 * vcachekk);
	}
	if(ul == UPPER) {
		w.noalias() += coef * (vcache00 + kk1 * vcachek0 + kk2 * vcache0k + kk1 * kk2 * vcachekk);
	}
}

template <>
inline void dkbbasis::cacheMatvec<bdpa>(const cvec& v, cvec& w, int cacheId, cdouble coef, int k1, int k2, int ul, int l) {
	cvec& vcache00 = getbdpvec<p00>(ul,cacheId,l);
	cvec& vcache0k = getbdpvec<p0k>(ul,cacheId,l);
	cvec& vcachek0 = getbdpvec<pk0>(ul,cacheId,l);
	cvec& vcachekk = getbdpvec<pkk>(ul,cacheId,l);
	
	int kk1 = ik(k1);
	int kk2 = ik(k2);
	
	//cout << cacheId << ", " << coef << ", " << k1 << ", " << k2 << ", " << ul << ", " << l << std::endl;
	
	if(ul == LOWER) {
		w.noalias() += coef * (vcache00 + kk2 * vcachek0 + kk1 * vcache0k + kk1 * kk2 * vcachekk);
	}
	if(ul == UPPER) {
		w.noalias() += coef * (vcache00 + kk1 * vcachek0 + kk2 * vcache0k + kk1 * kk2 * vcachekk);
	}
}

template<>
inline void dkbbasis::cacheMatvec<H0>(const cvec& v, cvec& w, int cacheId, cdouble coef, int k) {
	int kappa = ik(k);
	
	csmat& H0_0 = getH0mat(0);
	csmat& H0_k = getH0mat(1);
	csmat& H0_k2 = getH0mat(2);
	csmat& H0_k3 = getH0mat(3);
	
	w = H0_0 * v + kappa * H0_k * v + pow(kappa,2) * H0_k2 * v + pow(kappa,3) * H0_k3 * v;
}


template<matname M>
inline void dkbbasis::cacheMatvec(const cvec& v, cvec& w, int cacheId, cdouble coef) {
	w.noalias() += matvec<M>(v);
}

template<matname M>
inline void dkbbasis::cacheMatvec(const cvec& v, cvec& w, int cacheId, cdouble coef, int k1, int k2, int ul) {
	csmat& M0 = get0mat<M>();
	csmat& M0k = get0kmat<M>();
	csmat& Mk0 = getk0mat<M>();
	csmat& Mkk = getkkmat<M>();
	
	int kappa1 = ik(k1);
	int kappa2 = ik(k2);
	
	// cvec v00 = M0 * v;
	// cvec v0k = M0k * v;
	
	if(ul == LOWER) {
		w.noalias() += coef * (M0 * v - kappa2 * Mk0 * v - kappa1 * M0k * v + kappa1 * kappa2 * Mkk * v);
	}
	if(ul == UPPER) {
		w.noalias() += coef * (M0.adjoint() * v + kappa1 * Mk0.adjoint() * v + kappa2 * M0k.adjoint() * v + kappa1 * kappa2 * Mkk.adjoint() * v);
	}
}

template<matname M>
inline void dkbbasis::precacheMatvec(const cvec& v, int cacheId) {
	
}

csmat factorisedglintmat(csmat& M0, csmat& Mk, csmat& Mkk,
						const clsmat& aP0,const clsmat& aQ0,const clsmat& bP0,const clsmat& bQ0,
						const clsmat& aPk,const clsmat& aQk,const clsmat& bPk,const clsmat& bQk,
						int k, const clsmat& wts, int kappa, csmat(*func)(const clsmat& aP,const clsmat& aQ,const clsmat& bP,const clsmat& bQ,const clsmat& wts,int k, int ul, int sumcoef), int ul = 0, int sumcoef = 1);

csmat dkbglintmat(const clsmat& a0, const clsmat& b0, const clsmat& ws, int k);

csmat dkbglintmat(const clsmat& a0,const clsmat&  a1,const clsmat& b0,const clsmat& b1,const clsmat& ws,int k, int ul = 0, int sumcoef = 1);

#endif
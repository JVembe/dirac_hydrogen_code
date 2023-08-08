#ifndef BASIS_H
#define BASIS_H

#include "defs.h"
#include "potential.h"


//Matname enum is used in template functions that act on specific matrices
//It seemed clever at the time
enum matname {
	S,  		//0  Overlap matrix
	dd, 		//1  Projection of derivative operator
	E,  		//2  Projection of potential energy operator
	Lsq,		//3  Projection of L^2-operator
	L,  		//4  Projection of L-operator
	psq,		//5  Projection of momentum operator squared
	pz, 		//6  Projection of momentum operator z component
	bdpApx,		//7  A*px beyond dipole
	bdpApz,		//8  A*pz beyond dipole
	k,  		//9  Projection of kappa operator
	ulc,		//10 Projection of rest energy in dirac equation, gap of 2 mc^2 between upper and lower states
	p1, 		//11 Wait this doesn't actually need caching to be honest??
	dpa,		//12 Time independent part of couplings in dipole approximation
	bdpa,		//13
	bdpAsq,		//14 A(x,t)^2-operator 
	H0,			//15 Store the H0-matrix
	Ht,			//16 Store the Ht-matrix
	SU,			//17 Overlap matrix of upper component only
	SL,			//18 Overlap matrix of lower component only
	compS,		//19 Component-wise overlap matrix (for matvec)
	N_MATS		//NUMBER OF NAMED MATRICES, MUST BE LAST ELEMENT
};


//The basis class is a monster of ad-hoc solutions which at one point was supposed to be pretty general
//Child classes of basis are <Derived>
template <typename Derived>
class basis;

template <typename Derived>
struct basistraits {
	typedef basis<Derived> radial_type;
	typedef basis<Derived> angular_type;
};

template <typename Derived>
class basis {
	protected:
		//The idea here is to store the matrices that will be reused on construction
		csmat mats[N_MATS];
		
		//Refer to enum comments for what all these are
		csmat& olmat() 
			{ return getmat<S>(); }
		csmat& componentOlmat(int k1, int k2, int comp) {
			{ return getmat<compS>(); }
		}
		csmat& componentOlmat(int comp) {
			{ return getmat<compS>(); }
		}
		csmat& ddmat(int dn) 
			{ return getmat<dd>(); }
		csmat& Emat(long double (*V)(long double)) 
			{ return getmat<E>(); }
		csmat& Emat(long double (*V)(long double),bool recache) 
			{ getmat<E>() = csmat(0,0); return Emat(V); }
		csmat& Lsqmat() 
			{ return getmat<Lsq>(); }
		csmat& Lmat(double ll) 
			{ return getmat<L>(); }
		csmat& psqmat() {
			csmat& psqm = getmat<psq>();
			if(isCached(psqm)) return psqm;
			else {
				psqm = -0.5 * ddm(2);
			}
			return psqm;
		};
		csmat& pzmat() 
			{ return getmat<pz>(); };
		csmat& bdppzmat()
			{ return getmat<bdppzm>();}
		csmat& bdppzmat(int l)
			{ return getmat<bdppzm>();}
		csmat& kappamat() 
			{ return getmat<k>(); }
		csmat& ulcmat() 
			{ return getmat<ulc>(); }
		csmat& p1mat() 
			{ return getmat<p1>(); }
		csmat& dpalphmat(int ul) 
			{ return getmat<dpa>(); }
		csmat& dpalphmat() 
			{ return getmat<dpa>(); }
		csmat& dpalphmat(int k1,int k2, int ul) 
			{ return getmat<dpa>(); }
		csmat& bdpalphmat(int ul) 
			{ return getmat<bdpa>(); }
		csmat& H0mat() {
			return getmat<H0>();
		}
		csmat& Htmat() {
			return getmat<H0>();
		}
		
		//"Get angular factor matrix", ad hoc approach to solve an issue with the DKB basis
		template <matname M>
		csmat& getafm(int k) {
			return getmat<M>();
		}
		
		//rqn and aqn, radial and angular quantum numbers. The intent was that the protected functions would be defined in child classes while public angqN() and radqN() would be the interface. I forgot.
		int rqn() 
			{ return 1; }
		int aqn() 
			{ return 1; }
			
		//amax, angular maximum, is not always angqN and so got a separate function
		int amax() 
			{ return angqN(); }
		
		//Default imlementation of matrix-vector product to feed to solvers just wraps the relevant matrix multiplication
		template<matname M>
		cvec matvec(const cvec& v) {
			//cout << getmat<M>().rows() << std::endl;
			//cout << v.size() << std::endl;
			
			return getmat<M>() * v;
		}
		
		//As cacheMatvec is only useful for some bases, it has a default implementation that's just matvec
		template<matname M>
		cvec cacheMatvec(const cvec& v, int cacheId) {
			return static_cast<Derived*>(this)->matvec(v);
		}
		
		//Two variants of matmat_impl needed for parallelized matrix-vector products. In practice the second version is what sees use
		template<matname M>
		cmat matmat_impl(std::integral_constant<matname,M> c,const cmat& m) {
			//cout << getmat<M>().rows() << std::endl;
			//cout << v.size() << std::endl;
			
			return getmat<M>() * m;
		}
		
		template<matname M>
		void matmat_impl(std::integral_constant<matname,M> c,const cmat& m, cmat& w, cvec& params) {
			//cout << getmat<M>().rows() << std::endl;
			//cout << v.size() << std::endl;
			
			w.noalias() += (getmat<M>() * m).colwise() * params;
		}
		
		//For bases with both radial and angular components, returns the relevant basis
		typename basistraits<Derived>::radial_type& radial() const { return *this; }
		typename basistraits<Derived>::radial_type& angular() const { return *this; }
		
		//Clear vecs. Used for bases with a cache
		void clvcs() {}
		
		//This was never implemented
		template <typename basisType>
		cvec interpolate(basis<basisType>& otherBasis,cvec& v);
		
		int idxtfrm(int i) {
			return i;
		}
		
		void slp(int a1,int a2, int b1, int b2) { }
	public:
		
		//Default constructor and destructor
		basis() { }
		~basis() { }
		
		
		//Interface for explicit-matrix mode. Return the relevant matrices
		csmat& Sm() 
			{ return static_cast<Derived*>(this)->olmat();}
		csmat& olcm(int comp)
			{ return static_cast<Derived*>(this)->componentOlmat(comp);}
		csmat& olcm(int k1, int k2, int comp)
			{ return static_cast<Derived*>(this)->componentOlmat(k1, k2, comp);}
		csmat& ddm(int dn) 
			{ return static_cast<Derived*>(this)->ddmat(dn); }
		csmat& Em(long double (*V)(long double)) 
			{ return static_cast<Derived*>(this)->Emat(V); }
		csmat& Em(long double (*V)(long double),bool recache)
			{ return static_cast<Derived*>(this)->Emat(V,recache); }
		csmat& Lsqm() 
			{ return static_cast<Derived*>(this)->Lsqmat(); }
		csmat& Lm() 
			{ return static_cast<Derived*>(this)->Lmat(); }
		csmat& psqm() 
			{ return static_cast<Derived*>(this)->psqmat(); }
		csmat& pzm() 
			{ return static_cast<Derived*>(this)->pzmat(); }
		csmat& pzm(int ul) 
			{ return static_cast<Derived*>(this)->pzmat(ul); }
		csmat& bdppzm()
			{ return static_cast<Derived*>(this)->bdppzmat(); }
		csmat& bdppzm(int l)
			{ return static_cast<Derived*>(this)->bdppzmat(l); }
		csmat& km() 
			{ return static_cast<Derived*>(this)->kappamat(); }
		csmat& ulcm() 
			{ return static_cast<Derived*>(this)->ulcmat(); }
		csmat& p1m() 
			{ return static_cast<Derived*>(this)->p1mat(); }
		csmat& dpam() 
			{ return static_cast<Derived*>(this)->dpalphmat(); } 
		csmat& dpam(int ul) 
			{ return static_cast<Derived*>(this)->dpalphmat(ul); }
		csmat& dpam(int k1,int k2, int ul) 
			{ return static_cast<Derived*>(this)->dpalphmat(k1,k2,ul); }
		csmat& bdpam(int param) 
			{ return static_cast<Derived*>(this)->bdpalphmat(param); }
		csmat& bdpam(int param1, int param2) 
			{ return static_cast<Derived*>(this)->bdpalphmat(param1,param2); }
		
		csmat& bdpam(const Potential<beyondDipolePulse>& vExt) 
			{ return static_cast<Derived*>(this)->bdpalphmat(vExt); }
			
		csmat& bdpam(int param1, const Potential<beyondDipolePulse>& vExt) 
			{ return static_cast<Derived*>(this)->bdpalphmat(param1,vExt); }
			
		csmat& bdpam(int param1, int param2, const Potential<beyondDipolePulse>& vExt) 
			{ return static_cast<Derived*>(this)->bdpalphmat(param1, param2, vExt); }
		csmat& bdpam(int k1,int k2, int ul) 
			{ return static_cast<Derived*>(this)->bdpalphmat(k1,k2,ul); }
		
		csmat& bdpam(int k1,int k2, int ul, int l, int alpha, const Potential<beyondDipolePulse>& vExt) 
			{ return static_cast<Derived*>(this)->bdpalphmat(k1,k2,ul,l, alpha, vExt); }
		
		csmat& bdpam(int k1,int k2, int ul, int l, const Potential<beyondDipolePulse>& vExt) 
			{ return static_cast<Derived*>(this)->bdpalphmat(k1,k2,ul,l, vExt); }
		
		
		csmat& bdpAsqm(const int l) 
			{ return static_cast<Derived*>(this)->bdpAsqmat(l); }
		
		csmat& bdpAsqm(const int l, const Potential<beyondDipolePulse>& vExt) 
			{ return static_cast<Derived*>(this)->bdpAsqmat(l, vExt); }
		
		
		csmat& bdpApxm(const int l, const int ul) 
			{ return static_cast<Derived*>(this)->bdpApxmat(l,ul); }
		
		csmat& bdpApxm(const int l) 
			{ return static_cast<Derived*>(this)->bdpApxmat(l); }
			
		csmat& bdpApzm(const int l, const int ul) 
			{ return static_cast<Derived*>(this)->bdpApzmat(l,ul); }
		
		csmat& bdpApzm(const int l) 
			{ return static_cast<Derived*>(this)->bdpApzmat(l); }
		
		
		csmat& bdpApzm(const int l, const int ll, const int ul, const Potential<beyondDipolePulse>& vExt) 
			{ return static_cast<Derived*>(this)->bdpApzmat(l,ll,ul, vExt); }
		
		csmat& H0m() {
			return static_cast<Derived*>(this)->H0mat();
		}
		
		csmat& Htm() {
			return static_cast<Derived*>(this)->Htmat();
		}
		
		//I have no idea. Don't delete, might break something
		std::vector<csmat> H0mats;
		std::vector<csmat> Htmats;
		
		csmat& getH0mat(int l) {
			if(l >= H0mats.size()) {
				for(int i = H0mats.size(); i <= l; i++) {
					H0mats.push_back(csmat(0,0));
				}
			}
			
			return H0mats[l];
		}
		
		csmat& getHtmat(int l) {
			if(l >= Htmats.size()) {
				for(int i = Htmats.size(); i <= l; i++) {
					Htmats.push_back(csmat(0,0));
				}
			}
			
			return Htmats[l];
		}
		
		//So many of these are just public-facing aliases to private functions, which makes a lot more sense when looking at child classes
		template<matname M>		
		csmat& getmat() { return mats[M]; }
		
		template<matname M>
		csmat& getAngFacMat(int k) 
			{ return static_cast<Derived*>(this)->template getafm<M>(k); }
		
		//Interface for matrix-free mode
		template<matname M>
		cvec matfree(const cvec& v ) 
			{ return static_cast<Derived*>(this)->template matvec<M>(v); }
		
		template<matname M>
		cvec matfree(const cvec& v, int param) 
			{ return static_cast<Derived*>(this)->template matvec<M>(v,param); }
		
		template<matname M>
		cmat matmat(const cmat& m) {
			{ return static_cast<Derived*>(this)->template matmat_impl(std::integral_constant<matname,M> {}, m); }
		}
		
		template<matname M>
		void matmat(const cmat& m, cmat& w, cvec& params) {
			{ static_cast<Derived*>(this)->template matmat_impl(std::integral_constant<matname,M> {}, m, w, params); }
		}
		
		template<matname M>
		void matmat(const cmat& m, cmat& w, cvec& params, int l, int ul) {
			{ static_cast<Derived*>(this)->template matmat_impl(std::integral_constant<matname,M> {}, m, w, params,l,ul); }
		}
		
		template<matname M>
		void matmat(const cmat& m, cmat& w, csmat& thmat, cvec& params, int l, int ul) {
			{ static_cast<Derived*>(this)->template matmat_impl(std::integral_constant<matname,M> {}, m, w, thmat, params,l,ul); }
		}
		
		template<matname M,typename... ArgTs>
		inline void cacheMatfree(const cvec& v, cvec& w, int cacheId, ArgTs... Args) 
			{ static_cast<Derived*>(this)->template cacheMatvec<M>(v,w,cacheId,Args...); }
		
		template<matname M>
		inline void cacheMatfree(const cvec& v,int cacheId) 
			{ static_cast<Derived*>(this)->template cacheMatvec<M>(v,cacheId); }
		
		
		template<matname M>
		inline void precacheMatfree(const cvec& v,int cacheSize) 
			{ static_cast<Derived*>(this)->template precacheMatvec<M>(v,cacheSize); }
		
		template<matname M, typename... ArgTs>
		inline void precacheMatfree(const cvec& v,int cacheSize, ArgTs... Args) 
			{ static_cast<Derived*>(this)->template precacheMatvec<M>(v,cacheSize,Args...); }
		
		
		csmat& angCouplings(bool bdp = false) 
			{ return static_cast<Derived*>(this)->angcp(bdp); }
		int angqN() 
			{ return static_cast<Derived*>(this)->aqn(); }
		int radqN() 
			{ return static_cast<Derived*>(this)->rqn(); }
		
		int angMax()
			{ return static_cast<Derived*>(this)->amax(); }
		
		//Sets the return of aqn() to a different number from what would usually be returned. This is mainly used in rthphbasis
		int angOverride = 0;
		void angovr(int i) { angOverride = i; }
		
		//For radial bases that have an associated angular state (bsplinebasis, dkbbasis) setstate was originally used to determine the quantum number used.
		//This function has basically been phased out but I'm afraid of removing it
		template <typename... ArgTs>
		void setState(ArgTs... Args) 
			{ static_cast<Derived*>(this)->setst(Args...); }
		void setst(int i) { };
		
		//Interface
		typename basistraits<Derived>::radial_type& getRadial() const { return static_cast<const Derived*>(this)->radial(); }
		typename basistraits<Derived>::angular_type& getAngular() const { return static_cast<const Derived*>(this)->angular(); }
		
		//I have no idea why this is here
		void setst(int i, int j, int k) { };
		
		//In some bases the indexing 
		int indexTransform(int i) {
			return static_cast<Derived*>(this)->idxtfrm(i);
		}
		
		
		//It's here in case you want to clear all the matrices I guess
		void clearMats() {
			for(int i = 0; i < matname::N_MATS; i++) {
				mats[i] = csmat(0,0);
			}
		}
		
		//This is used in the nondipole case to clear the vector cache actually
		void clearVecs() {
			return static_cast<Derived*>(this)->clvcs();
		}
		
		//In the nondipole case you need a cap on the l variable for the Bessel function expansion, and this is that
		int bdplmax() {
			return static_cast<Derived*>(this)->bdplmax();
		}
		
		//Never got implemented
		template <typename basistype>
		cvec interpFrom(basis<basistype>& otherBasis, cvec& v) {
			return static_cast<Derived*>(this)->interpolate(otherBasis,v);
		}
		
		void setLocalParams(int a1, int a2, int b1, int b2) {
			return static_cast<Derived*>(this)->slp(a1,a2,b1,b2);
		}
};

//This never got far because finite difference is a terrible basis for this system

class fdbasis: public basis<fdbasis> {
	friend class basis<fdbasis>;
	int ncomps;
	int nprecs;
	
	double dr;
	double rmax;

	public:
		fdbasis(vec &t, int nprec,int ncomps);
		void setst(int i);
		
		csmat& olmat();
		csmat& ddmat(int dn);
		csmat& Emat(long double (*V)(long double));
		csmat& p1mat();
		
		csmat& kappamat();
		csmat& ulcmat();
		csmat& dpalphmat(int k1,int k2, int ul);
		
		int rqn();
};

//Spherical harmonic basis with no m quantum number. Sort of useless for anything not dipole
class spharmbasis: public basis<spharmbasis> {
	friend class basis<spharmbasis>;
	int lmax = 0;
	
	public:
		spharmbasis(int lmax);
		int aqn();
		csmat& Lsqmat();
		csmat& pzmat();
		csmat& angcp(bool bdp) { return pzmat(); }
		
		csmat& pzmat(int ul);
};


//Spherical harmonic basis with variable maximum l and m, far more useful
class spharmbasisLM: public basis<spharmbasisLM> {
	friend class basis<spharmbasisLM>;
	int lmax = 0;
	int mmax = 0;
	
	int bdplovr = 0;
	
	vector<csmat> bdppzmats[2];
	vector<csmat> bdpsqmats;
	csmat bdpacp;
	
	csmat& getbdppzmat(int l, int ul) {
		switch(ul) {
			case 1:
				return bdppzmats[0][l];
				break;
			case -1:
				return bdppzmats[1][l];
				break;
		}
	}
	csmat& getbdpsqmat(int l) {
		return bdpsqmats[l];
	}
	
	public:
		spharmbasisLM(int lmax, int mmax = 0);
		int aqn();
		csmat& Lsqmat();
		csmat& pzmat();
		csmat& pzmat(int ul);
		csmat& bdppzmat(int l);
		csmat& bdpApzmat(int l,int ul);
		csmat& bdpApxmat(int l,int ul);
		csmat& bdpAsqmat(int l);
		
		int bdplmax() {
			if(this->bdplovr == 0) 
				return 2*lmax + 1;
			else return bdplovr;
		}
		
		int amax() {
			return lmax;
		}
		
		csmat& angcp(bool bdp) { 
			switch(bdp) {
				case false:
					return pzmat(); 
				break;
				case true:
					if(isCached(bdpacp)) return bdpacp;
					else {
						bdpacp = csmat(angqN(),angqN());
						for(int l = 0; l < bdplmax(); l++) {
							bdpacp += bdpApzmat(l,1);
							bdpacp += bdpApzmat(l,-1);
							bdpacp += bdpAsqmat(l);
						}
					}
					return bdpacp;
				break;
			}
		}
		
		void bdplOverride(int i) {
			this->bdplovr = i;
		}
};

//Spinor basis using Dirac bispinors
class spnrbasis: public basis<spnrbasis> {
	friend class basis<spnrbasis>;
	int kmax = 0;
	int mumax = 0;			 
	int matvecul = 0;
	
	int bdplovr = 0;
	
	csmat dpamats[2];
	vector<csmat> bdplmats;
	
	private:
		cdouble bdpK(int i, int j, int l);
		cdouble bdpL(int i, int j, int l);
		cdouble bdpM(int i, int j, int l);
		cdouble bdpN(int i, int j, int l);
		
		csmat& getbdpmat(int ul, int l) {
			switch(ul) {
				case -1:
					return bdplmats[l];
				case 1:
					return bdplmats[l+bdplmax()];
				default:
					cout << "ul should always be 1 or -1";
					return bdplmats[2*bdplmax()];
			}
		}
		
		csmat acp;
		
	public:
		spnrbasis(int kmax,int mumax);		
		spnrbasis(int kmax):spnrbasis(kmax,0) { }
		int aqn();
		
		csmat& componentOlmat(int comp);
		csmat& dpalphmat(int ul);
		csmat& bdpalphmat(int ul,int l);
		csmat& bdpalphsigmaXmat(int ul,int l);
		
		void setst(int ul) {
			this->matvecul = ul;
		}
		int bdplmax() {
			if(bdplovr==0)
				return 2*(kmax) + 1;
			else return bdplovr;
		}
		
		int amax() {
			//cout << "We made it! spnrbasis::amax() was called and returns " << kmax << "!\n";
			return kmax;
		}
		csmat& angcp(bool bdp) { 
			if(!bdp) {
				acp = dpalphmat(0);
				return acp; 
				}
			else {
				acp = csmat(angqN(),angqN());
				for(int l = 0; l < bdplmax(); l++) {
					acp += bdpalphmat(-1,l) + bdpalphmat(1,l);
				}
				return acp;
			}
		}
		
		
		void bdplOverride(int i) {
			this->bdplovr = i;
		}
};

#endif
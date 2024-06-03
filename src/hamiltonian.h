/*
	This file contains the interface class template Hamiltonian, as well as specializations for dipole and nondipole Dirac equation Hamiltonians.
	
	There were also Schrödinger implementations that have been moved out for the time being.
	
	The Hamiltonian class is responsible for high-level management of the dynamics of the system.
*/

#ifndef HAMILTONIAN_H
#define HAMILTONIAN_H
#include "defs.h"
#include "wavefunc.h"
#include <mpi.h>
#include "mpiFuncs.h"

vec linInterpolate(const vec& x,const vec& y, const vec& pts) {
	int j = 0;
	vec out(pts.size());
	for(int i = 0; i < pts.size(); i++) {
		double xi = pts[i];
		
		if(xi < x[0]) out[i] = 0.0;
		else {
			while (j < x.size()-1) {
				if(xi >= x[j+1]) j++;
				else break;
			}
			//cout << j << ":\n";
			//cout <<"xi " << pts[i] << ", xj " << x[j] << "\n";
			if(j < x.size() - 1) {
				double x0 = x[j];
				double x1 = x[j+1];
				
				double y0 = y[j];
				double y1 = y[j+1];
				
				//cout << "x0: " << x0 << ", x1: " << x1 << ", y0: " << y0 << ", y1" << y1 << "\n";
				
				out[i] = (y1*(pts[i] - x0) + y0*(x1 - pts[i]))/(x1-x0);
			}
			else out[i] = 0.0;
		}
	}
	return out;
}



template <typename Derived, typename basistype>
class Hamiltonian {
	protected:
		
		bool matfreePrepped;
		
		eigSolver geigs;
		
		bool angSep = false;
		
		vector<dsmat> eigvecs;
		vector<vec> eigvals;

		csmat* Sm = new csmat;
		csmat* H0m = new csmat;
		
		basis<basistype>* bs;
		
		const csmat& Smat() const {
			if(isCached(Sm)) return *Sm;
			else {
				//cout << "First call to S(), computing\n";
				*Sm = this->bs->Sm();
				return *Sm;
			}
		}
		
		const csmat& H0mat() const {
			cout << "Default hamiltonian called\n";
			if(isCached(H0m)) return *H0m;
			else {
				*H0m = this->bs->Sm();
				return *H0m;
			}
		}

		cvec Svec(const cvec& v) const {
			//if(matfreePrepped) {
			return this->bs->template matfree<matname::S>(v);
			//}
			//else return v;
		}

		cvec H0vec(const cvec& v) const {
			//if(matfreePrepped) {
			return this->bs->template matfree<matname::S>(v);
			//}
			//else return v;
		}
		
		cvec H0radvec(const cvec& v) const {
			//if(matfreePrepped) {
			return this->bs->getRadial().template matfree<matname::S>(v);
			//}
		}
		
		csmat Htmat(double t) { return csmat(0,0); }
		
		int angMax_impl() const {
			return this->bs->angMax();
		}
	public:
		long double (*Vfunc)(long double);
		
		
		basis<basistype>& getBasis() const { return *bs; };
		
		Hamiltonian() { };
		Hamiltonian(basis<basistype>& bs) {
			this->bs = &bs;
		}
		
		const csmat& S() const {
			return static_cast<const Derived*>(this)->Smat();
		}
		
		const csmat Simpl(std::integral_constant<axis,radial> c) const {
			return this->bs->getRadial().Sm();
		}
		
		template<axis Ax>
		const csmat Simpl(std::integral_constant<axis,Ax> c) const {
			return S();
		}
		
		template<axis Ax>
		const csmat S() const {
			return static_cast<const Derived*>(this)->Simpl(std::integral_constant<axis,Ax>{} );
		}
		
		cvec S(const cvec& v) const {
			return static_cast<const Derived*>(this)->Svec(v);
		}
		
		// cmat S(const cmat& v) const {
			// return static_cast<const Derived*>(this)->Svec(v);
		// }
		
		const csmat& H0() const {
			return static_cast<const Derived*>(this)->H0mat();
		}
		
		template<axis Ax>
		csmat H0() const {
			return H0impl(std::integral_constant<axis,Ax>{} );
		}
		
		csmat H0impl(std::integral_constant<axis,radial> c) const {
			return static_cast<const Derived*>(this)->H0radmat();
		}
		
		template<axis Ax>
		csmat H0impl(std::integral_constant<axis, Ax> c) const {
			return this->H0();
		}
		
		
		template<axis Ax  = axis::all>
		cvec H0(const cvec& v) const {
			return H0impl(std::integral_constant<axis, Ax> {},v);
		}
		
		cvec H0impl(std::integral_constant<axis, radial> c, const cvec& v) const {
			return static_cast<const Derived*>(this)->H0radvec(v);
		}
		template<axis Ax>
		cvec H0impl(std::integral_constant<axis,Ax> c, const cvec& v) const {
			return static_cast<const Derived*>(this)->H0vec(v);
		}
		
		csmat Ht(double t) {
			return static_cast<Derived*>(this)->Htmat(t);
		}
		
		cvec Ht(double t, const cvec& v) const {
			return static_cast<const Derived*>(this)->HtVec(t, v);
		}
		
		csmat H(double t) {
			return static_cast<Derived*>(this)->Ht(t) + static_cast<Derived*>(this)->H0();
		}
		
		cvec H(double t, const cvec& v) const {
			//cvec H0v = static_cast<const Derived*>(this)->H0vec(v);
			//cvec Htv = static_cast<const Derived*>(this)->HtVec(t, v);
			
			//cout << "H0v dims: (" << H0v.rows() << ", " << H0v.cols() << ")\nHtv dims: " << Htv.rows() << ", " << Htv.cols() << ")" << endl;
			
			// int wrank;
			// MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
			
			
			cvec Hv;// = H0v + Htv;
			
			Hv = static_cast<const Derived*>(this)->HtVec(t, v) + static_cast<const Derived*>(this)->H0vec(v);
			
			// bcastMat(Hv);
			// if(wrank!=0)
				// Hv = cvec::Zero(this->bs->angqN()*this->bs->radqN());
			// cout << "Hv dims: (" << Hv.rows() << ", " << Hv.cols() << ")"<<endl;
			return Hv;
		}
		
		// cmat H(double t, const cmat& v) const {
			
			
			// cmat Hv;// = H0v + Htv;
			
			// Hv = static_cast<const Derived*>(this)->HtVec(t, v) + static_cast<const Derived*>(this)->H0vec(v);
			
			// return Hv;
		// }
		
		
		double sigmaE(const cvec& v) {
			cvec Hv = this->H0(v);
			
			cvec Hsqv = this->H0(Hv);
			
			
			double E = abs((v.adjoint()*Hv)(0,0));
			
			double Esq = abs((v.adjoint()*Hsqv)(0,0));
			
			cout << E << ", " << Esq << "\n";
			
			return sqrt(Esq - pow(E,2));
		}
		cmat eigenstateProject(const wavefunc<basistype> psi) {
			return static_cast<Derived*>(this)->eigProj(psi);
		}
		
		/*
		void prepeigs(int nev, int ncv, bool angSep = true) {
			this->angSep = angSep;
			if(!angSep){
				dsmat H0 = this->H0().real();
				dsmat S = this->S().real();
				
				geigs.compute(H0,S);
				
				eigvecs = vector<dsmat>(1);
				eigvals = vector<vec>(1);
				
				eigvecs[0] = geigs.eigenvectors().sparseView();
				eigvals[0] = geigs.eigenvalues();
				
			}
			else {
				//Separate H0 and S by angular quantum number
				eigvecs = vector<dsmat>(this->bs->angqN());
				eigvals = vector<vec>(this->bs->angqN());
				for(int j = 0; j < this->bs->angqN(); j++) {
					//cout << j << "\n";
					dsmat imat = dsmat(this->bs->angqN()*this->bs->radqN(),this->bs->radqN());
					
					std::vector<dT> triplets;
					triplets.reserve(this->bs->radqN());
				
					for(int i = 0; i < this->bs->radqN(); i++) {
						triplets.push_back(dT(
							i + j*this->bs->radqN(),
							i,
							1
							)
						);
					}
					
					
					imat.setFromTriplets(triplets.begin(),triplets.end());
					
					this->getBasis().getRadial().setState(this->getBasis().indexTransform(j));
					
					
					dsmat H0r = this->template H0<axis::radial>().real();
					dsmat Sr = this->template S<axis::radial>().real();
					
					//dsmat H0 = this->H0().real().middleCols(j*this->bs->radqN(),this->bs->radqN());
					//dsmat S = this->S().real().middleCols(j*this->bs->radqN(),this->bs->radqN());
					
					geigs.compute(H0r,Sr);
					
					dsmat eigstates =  imat * dsmat(this->geigs.eigenvectors().real().sparseView());
					
					eigvecs[j] = eigstates;
					eigvals[j] = geigs.eigenvalues();
					
				}
				
			}
		}
		*/
		
		void prepeigs(int nev, int ncv, bool angSep = true) {
			//this->H0();
			//this->S();
			
			this->angSep = angSep;
			if(!angSep){
				dsmat H0 = this->H0().real();
				dsmat S = this->S().real();
				
				this->geigs.compute(H0,S);
				
				this->eigvecs = vector<dsmat>(1);
				this->eigvals = vector<vec>(1);
				
				this->eigvecs[0] = geigs.eigenvectors().sparseView();
				this->eigvals[0] = geigs.eigenvalues();
			}
			else {
				//Separate H0 and S by angular quantum number
				int Nr = this->bs->radqN();
				int Nang = this->bs->angqN();
				
				int Ntot = Nr*Nang;
				
				this->eigvecs = vector<dsmat>(Nang);
				this->eigvals = vector<vec>(Nang);
				
				

				for(int j = 0; j < Nang; j++) {
					//cout << j << "\n";
					dsmat imat = dsmat(Nr*Nang,Nr);
					
					std::vector<dT> triplets;
					triplets.reserve(Nr);
				
				
					for(int i = 0; i < Nr; i++) {
						triplets.push_back(dT(
							i + j*Nr,
							i,
							1
							)
						);
					}
					
					
					imat.setFromTriplets(triplets.begin(),triplets.end());
					
					
					//dsmat H0r = imat.transpose() * this->H0().real() * imat;
					//dsmat Sr = imat.transpose() * this->S().real() * imat;
					
					this->bs->getRadial().setState(this->getBasis().indexTransform(j));
					
					dsmat H0r = this->template H0<axis::radial>().real();
					dsmat Sr = this->template S<axis::radial>().real();
					
					
					//cout << "imat dimensions: (" << Ntot << "," << Nr << ")\n";
					//dsmat H0 = this->H0().real().middleCols(j*Nr,Nr);
					//dsmat S = this->S().real().middleCols(j*Nr,Nr);
					
					//cout << "geigs declared \n";
					
					// Initialize and compute
					//this->geigs->init();
					
					this->geigs.compute(H0r,Sr);
					
					cvec evals = geigs.eigenvalues();
					int eigN = evals.size();
					
					dsmat eigstates =  (imat * this->geigs.eigenvectors().real()).sparseView();
					/*
					//Iterate over all columns of eigstates and ensure positive
					for(int i = 0; i < dsmat.cols(); i++) {
						//Determine if nonzero 
						double vmax = eigstates.col(i).maxCoeff();
						double vmin = abs(eigstates.col(i).minCoeff());
						
						double vv;
						
						if vmin>vmax vv = vmin;
						else vv = vmax;
						
						int j = 0;
						while(true) {
							if(abs(eigstates(i,j)*100) > vv) {
								break;
							}
							j++;
						}
						
						if(eigstates(i,j) < 0) eigstates.col(i) = -eigstates.col(i);
					}
					*/
					this->eigvecs[j] = eigstates;
					//cout << this->eigvecs.size() << "\n";
					
					this->eigvals[j] = vec(geigs.eigenvalues().real()/*.middleRows(eigN/2,nev)*/ );
					

				}
				
			}
		}
				
		virtual vector<vec> &getevals(){
			return eigvals;
		}
		
		virtual vector<dsmat> &getevecs(){
			return eigvecs;
		}
		
		/*vector<wavefunc<basistype> > getEigenstates(){
			vector<mat> ecoefs = getevecs();
			vector<wavefunc<basistype> > eigstates;
			for(int i = 0; i < ecoefs.size(); i++) {
				eigstates.push_back(wavefunc<basistype> (*bs,ecoefs[i]));
			}
			
			return eigstates;
		}*/
		
		mat dPdE(wavefunc<basistype>& psi,int gridres = 10000, double Emin = -0.500007, double Emax = 300){
			if(!this->angSep) {
				cout << "Error: AngSep off\n";
				return mat(0,0);
			}
			vector<vec> la = getevals();
			vector<dsmat > eigvecs = getevecs();
			
			
			vec x(gridres);
			vec y(gridres);
			y.setZero();
			vector<int> xind(this->bs->angqN());
			
			vec ey(gridres);
			
			for(int i = 0; i < xind.size(); i++) {
				xind[i] = 0;
			}
			
			double dx = (Emax - Emin) / gridres;
			for(int i = 0; i < gridres; i++) {
				x[i] = Emin + i*dx;
			}
			
			//cout << "Step 1:  la and eigstates by angular then radial quantum number\n";
			
			for(int i = 0; i < this->bs->angqN();i++) {
				//cout << i << "\n";
				vec dPdEl = vec(la[i].size());
				dPdEl[0] = 1.0;
				dPdEl[la[i].size()-1] = 1.0;
			
				for(int j = 1; j < eigvecs[i].cols() - 1; j++) {
				
					double p = abs((psi * eigvecs[i].col(j))(0,0));
					
					double pde = 2 * pow(p,2)/(la[i][j+1] - la[i][j-1]);
					
					dPdEl[j] = pde;
				}
			
				y.noalias() += linInterpolate(la[i],dPdEl,x);
			}
			
			
			mat xy(gridres,2);
			xy.col(0) = x;
			xy.col(1) = y;
			
			return xy;
		}
		vector<vector<double> > angsepla();
		
		int Dim() const {
			// cout << "Dim() call returns: " << this->bs->angqN() * this->bs->radqN() << endl;
			return this->bs->angqN() * this->bs->radqN();
		}
		
		int angMax() const {
			return static_cast<const Derived*>(this)->angMax_impl();
		}
};

template <typename DiracType, typename basistype>

class DiracBase: public Hamiltonian<DiracType,basistype> {
	friend class Hamiltonian<DiracType, basistype >;
	
	
	friend class Hamiltonian<DiracType, basistype >;
	diracEigSolver gseigs;
	
	protected:
		
	public:
	
		int angMax_impl() const {
			return this->bs->getAngular().angMax();
		}
	
		long double (*extPot)(long double);
		DiracBase(basis<basistype>& bs):Hamiltonian<DiracType,basistype> (bs) { };
		
		
		const csmat& H0mat() const {
			if(isCached(this->H0m)) {
				return *(this->H0m);
			}
			else {
				//cout << "First call to H0(), computing\n";
				
				*(this->H0m) = (SoL * this->bs->p1m()) + this->bs->Em(this->Vfunc) + this->bs->ulcm();
				
				return *(this->H0m);
			}
		}
		
		
		cvec H0vec(const cvec& v) const {
			// cout << "h0mat dims = (" << this->bs->getRadial().getH0mat(0).rows()  << ", " << this->bs->getRadial().getH0mat(0).cols() << ")\n";
			
			if(!isCached(this->bs->getRadial().getH0mat(0)))
				return SoL * (this->bs->template matfree<dd>(v) + this->bs->template matfree<k>(v)) + (this->bs->template matfree<E>(v)) + (this->bs->template matfree<ulc>(v));
			else
				return this->bs->template matfree<H0>(v);
		}
		
		cmat H0vec(const cmat& v) const {
			if(!isCached(this->bs->getRadial().getH0mat(0)))
				return SoL * (this->bs->template matfree<dd>(v) + this->bs->template matfree<k>(v)) + (this->bs->template matfree<E>(v)) + (this->bs->template matfree<ulc>(v));
			else
				return static_cast<basistype*>(this->bs)->h0matvecMPIblock(v);
		}
	
		
		cvec H0radvec(const cvec& v) const {
			if(this->matfreePrepped) {
				return SoL * (this->bs->getRadial().template matfree<dd>(v) + this->bs->getRadial().template matfree<k>(v)) + (this->bs->getRadial().template matfree<E>(v)) + (this->bs->getRadial().template matfree<ulc>(v));
			}
			else return v;
		}
		
		csmat H0radmat() const {
			// if(isCached(this->bs->getRadial().getH0mat(0)))
				// return this->bs->getRadial().H0m();
			// else
				return SoL * (this->bs->getRadial().ddm(1) + this->bs->getRadial().km()) + (this->bs->getRadial().Em(this->Vfunc)) + (this->bs->getRadial().ulcm());
		}
		
		void H0radprep() {
			csmat H0_0 = SoL * (this->bs->getRadial()).template getAngFacMat<dd>(0) 
							 + (this->bs->getRadial()).template getAngFacMat<E>(0) 
			  -2 * pow(SoL,2) *(this->bs->getRadial()).template getAngFacMat<ulc>(0);
			
			csmat H0_kappa = SoL *((this->bs->getRadial()).template getAngFacMat<dd>(1)
								 + (this->bs->getRadial()).template getAngFacMat<k>(0))
								 + (this->bs->getRadial()).template getAngFacMat<E>(1)
				  -2 * pow(SoL,2) * (this->bs->getRadial()).template getAngFacMat<ulc>(1);
			
			csmat H0_kappa2 = SoL *((this->bs->getRadial()).template getAngFacMat<dd>(2)
								  + (this->bs->getRadial()).template getAngFacMat<k>(1)) 
								  + (this->bs->getRadial()).template getAngFacMat<E>(2) 
				  -2 * pow(SoL,2) * (this->bs->getRadial()).template getAngFacMat<ulc>(2);
		
			csmat H0_kappa3 = SoL * (this->bs->getRadial()).template getAngFacMat<k>(2);
			
			this->bs->getRadial().getH0mat(0) = H0_0;//.selfadjointView<Eigen::Upper>();
			this->bs->getRadial().getH0mat(1) = H0_kappa;//.selfadjointView<Eigen::Upper>();
			this->bs->getRadial().getH0mat(2) = H0_kappa2;//.selfadjointView<Eigen::Upper>();
			this->bs->getRadial().getH0mat(3) = H0_kappa3;//.selfadjointView<Eigen::Upper>();
		}
		
		
		void prepeigs(int nev, int ncv, bool angSep = true) {
			// this->H0();
			// this->S();
			
			this->angSep = angSep;
			if(!angSep){
				dsmat H0 = this->H0().real();
				dsmat S = this->S().real();
				
				this->gseigs.compute(H0,S);
				
				this->eigvecs = vector<dsmat>(1);
				this->eigvals = vector<vec>(1);
				
				this->eigvecs[0] = gseigs.eigenvectors().sparseView();
				this->eigvals[0] = gseigs.eigenvalues();
			}
			else {
				//Separate H0 and S by angular quantum number
				int Nr = this->bs->radqN();
				int Nang = this->bs->angqN();
				
				int Ntot = Nr*Nang;
				
				this->eigvecs = vector<dsmat>(Nang);
				this->eigvals = vector<vec>(Nang);
				
				
				std::vector<mat> kappaevecs;
				std::vector<vec> kappaevals;
				std::vector<int> kappas;
				
				
				int kappamax = this->angMax();
				
				for(int kappa = 1; kappa <= kappamax; kappa++) {
					int iL = ki(-kappa);
					
					cout << "(" << -kappa << "," << iL << ","<< ik(iL) << ")" << std::endl;
					
					kappas.push_back(-kappa);
					
					this->bs->getRadial().setState(iL);
					
					dsmat H0rL = this->template H0<axis::radial>().real();
					dsmat SrL = this->template S<axis::radial>().real();
					
					this->gseigs.compute(H0rL,SrL);
					
					vec evalsL = gseigs.eigenvalues().real();
					
					kappaevals.push_back(evalsL);
					int eigNL = evalsL.size();
					
					kappaevecs.push_back(this->gseigs.eigenvectors().real());
					
					int iU = ki(kappa);
					
					cout << "(" << kappa << ", " << iU << "," << ik(iU) << ")" << std::endl;
					
					kappas.push_back(kappa);
					
					this->bs->getRadial().setState(iU);
					
					dsmat H0rU = this->template H0<axis::radial>().real();
					dsmat SrU = this->template S<axis::radial>().real();
					
					this->gseigs.compute(H0rU,SrU);
					
					vec evalsU = gseigs.eigenvalues().real();
					
					kappaevals.push_back(evalsU);
					int eigNU = evalsU.size();
					
					kappaevecs.push_back(this->gseigs.eigenvectors().real());
					
					
				}
				
				for(int k = 0; k < kappaevecs.size(); k++) {
					for(int i = 0; i < kappaevecs[k].cols(); i++) {
						//Determine if nonzero 
						double vmax = abs(kappaevecs[k].col(i).maxCoeff());
						double vmin = abs(kappaevecs[k].col(i).minCoeff());
						
						double vv;
						
						if(vmin>vmax) vv = vmin;
						else vv = vmax;
						
						int j = 0;
						while(true) {
							if(abs(kappaevecs[k](j,i)*100) > vv) {
								break;
							}
							j++;
						}
						
						if(kappaevecs[k](j,i) < 0) kappaevecs[k].col(i) = -kappaevecs[k].col(i);
					}
				}
				if(!USEMPI) {
					//#pragma omp parallel for
					for(int j = 0; j < Nang; j++) {
						// cout << j << "\n";
						dsmat imat = dsmat(Nr*Nang,Nr);
						
						std::vector<dT> triplets;
						triplets.reserve(Nr);
					
					
						for(int i = 0; i < Nr; i++) {
							triplets.push_back(dT(
								i + j*Nr,
								i,
								1
								)
							);
						}
						
						
						imat.setFromTriplets(triplets.begin(),triplets.end());
						
						int kappa = ik(this->getBasis().indexTransform(j));
						
						int idx;
						
						for(int i = 0; i < kappas.size(); i++) {
							if(kappas[i] == kappa) {
								idx = i;
								break;
							}
						}
						
						// cout << "Index determined for current kappa: " << idx << ", kappa: " << kappa << std::endl;
						
						
						//dsmat H0r = imat.transpose() * this->H0().real() * imat;
						//dsmat Sr = imat.transpose() * this->S().real() * imat;
						/*
						this->bs->getRadial().setState(this->getBasis().indexTransform(j));
						
						dsmat H0r = this->template H0<axis::radial>().real();
						dsmat Sr = this->template S<axis::radial>().real();
						
						
						//cout << "imat dimensions: (" << Ntot << "," << Nr << ")\n";
						//dsmat H0 = this->H0().real().middleCols(j*Nr,Nr);
						//dsmat S = this->S().real().middleCols(j*Nr,Nr);
						
						//cout << "geigs declared \n";
						
						// Initialize and compute
						//this->geigs->init();
						
						this->gseigs.compute(H0r,Sr);
						
						cvec evals = gseigs.eigenvalues();
						int eigN = evals.size();
						
						*/
						dsmat eigstates =  (imat * kappaevecs[idx].real()).sparseView();
						
						
						//Iterate over all columns of eigstates and ensure positive
						
						this->eigvecs[j] = eigstates;
						//cout << this->eigvecs.size() << "\n";
						
						this->eigvals[j] = vec(kappaevals[idx].real());
						
					}
				} else if(USEMPI) {
					
					for(int j = 0; j < Nang; j++) {
						// cout << j << "\n";
						dsmat imat = dsmat(Nr*Nang,Nr);
						
						std::vector<dT> triplets;
						triplets.reserve(Nr);
					
					
						for(int i = 0; i < Nr; i++) {
							triplets.push_back(dT(
								i + j*Nr,
								i,
								1
								)
							);
						}
						
						
						imat.setFromTriplets(triplets.begin(),triplets.end());
						
						int kappa = ik(this->getBasis().indexTransform(j));
						
						int idx;
						
						for(int i = 0; i < kappas.size(); i++) {
							if(kappas[i] == kappa) {
								idx = i;
								break;
							}
						}
						
						cout << "Index determined for current kappa: " << idx << ", kappa: " << kappa << std::endl;
						
						
						//dsmat H0r = imat.transpose() * this->H0().real() * imat;
						//dsmat Sr = imat.transpose() * this->S().real() * imat;
						
						dsmat eigstates =  (imat * kappaevecs[idx].real()).sparseView();
						
						
						//Iterate over all columns of eigstates and ensure positive
						
						this->eigvecs[j] = eigstates;
						//cout << this->eigvecs.size() << "\n";
						
						this->eigvals[j] = vec(kappaevals[idx].real());
						
					}
				}
			}
		}
		
		
	std::vector<mat> kappaevecs;
	std::vector<vec> kappaevals;
	std::vector<int> kappas;
	
	void prepeigsLowMem(int nev, int ncv, bool localOnly = false,int kchoice = 0) {
		cout << "Preparing eigenvalues" << endl;
		//this->H0();
		//this->S();
		
		this->angSep = true;
		
		//Separate H0 and S by angular quantum number
		int Nr = this->bs->radqN();
		int Nang = this->bs->angqN();
		
		int Ntot = Nr*Nang;
		
		this->eigvecs = vector<dsmat>(Nang);
		this->eigvals = vector<vec>(Nang);
		
		cout << "Preparing eigenvalues and eigenvectors for ";
		if(kchoice == 0) cout << "all values of kappa" << endl;
		else cout << "kappa = " << kchoice << endl;
		
		int kappamax = this->angMax();
		if(kchoice != 0) {
		     int iL = ki(kchoice);

                                cout << "(" << kchoice << "," << iL << ","<< ik(iL) << ")" << std::endl;

                                kappas.push_back(kchoice);

                                this->bs->getRadial().setState(iL);

                                dsmat H0rL = this->template H0<axis::radial>().real();
                                dsmat SrL = this->template S<axis::radial>().real();

                                this->gseigs.compute(H0rL,SrL);

                                vec evalsL = gseigs.eigenvalues().real();

                                kappaevals.push_back(evalsL);
                                int eigNL = evalsL.size();

                                kappaevecs.push_back(this->gseigs.eigenvectors().real());
		} 
		else {
		if(!localOnly) {
			for(int kappa = 1; kappa <= kappamax; kappa++) {
				int iL = ki(-kappa);
				
				cout << "(" << -kappa << "," << iL << ","<< ik(iL) << ")" << std::endl;
				
				kappas.push_back(-kappa);
				
				this->bs->getRadial().setState(iL);
				
				dsmat H0rL = this->template H0<axis::radial>().real();
				dsmat SrL = this->template S<axis::radial>().real();
				
				this->gseigs.compute(H0rL,SrL);
				
				vec evalsL = gseigs.eigenvalues().real();
				
				kappaevals.push_back(evalsL);
				int eigNL = evalsL.size();
				
				kappaevecs.push_back(this->gseigs.eigenvectors().real());
				
				int iU = ki(kappa);
				
				cout << "(" << kappa << ", " << iU << "," << ik(iU) << ")" << std::endl;
				
				kappas.push_back(kappa);
				
				this->bs->getRadial().setState(iU);
				
				dsmat H0rU = this->template H0<axis::radial>().real();
				dsmat SrU = this->template S<axis::radial>().real();
				
				this->gseigs.compute(H0rU,SrU);
				
				vec evalsU = gseigs.eigenvalues().real();
				
				kappaevals.push_back(evalsU);
				int eigNU = evalsU.size();
				
				kappaevecs.push_back(this->gseigs.eigenvectors().real());
				
				
			}
			
			for(int k = 0; k < kappaevecs.size(); k++) {
				cout << "Applying sign convention to eigenvectors for kappa " << kappas[k] << "..." << endl; 
				for(int i = 0; i < kappaevecs[k].cols(); i++) {
					// cout << i << " ";
					
					//Determine if nonzero 
					double vmax = abs(kappaevecs[k].col(i).maxCoeff());
					double vmin = abs(kappaevecs[k].col(i).minCoeff());
					
					double vv;
					
					if(vmin>vmax) vv = vmin;
					else vv = vmax;
					
					int j = 0;
					while(true) {
						if(abs(kappaevecs[k](j,i)*100) > vv) {
							break;
						}
						j++;
					}
					
					if(kappaevecs[k](j,i) < 0) kappaevecs[k].col(i) = -kappaevecs[k].col(i);
				}
			}
			
		}
		else {
			int lth0, lNth, ll0, lNl;
			
			this->bs->getLocalParams(lth0,lNth,ll0,lNl);
			
			
			int firstkappa = ik(this->bs->indexTransform(lth0));
			int lastkappa = ik(this->bs->indexTransform(lNth-1));
			
			int id0 = ki(firstkappa);
			int id1 = ki(lastkappa);
			cout << "first kappa: " << firstkappa << ", last kappa: " << lastkappa << endl;
			
			cout << "id0: " << id0 << ", id1: " << id1 << endl;
			
			for(int kappa = 1; kappa <= kappamax; kappa++) {
				int iL = ki(-kappa);
				
				
				if(((iL >= id0) && (iL <= id1)) || (ik(iL) == ik(id1)) ) {
					cout << "(" << -kappa << "," << iL << ","<< ik(iL) << ")" << std::endl;
					
					kappas.push_back(-kappa);
					
					this->bs->getRadial().setState(iL);
					
					dsmat H0rL = this->template H0<axis::radial>().real();
					dsmat SrL = this->template S<axis::radial>().real();
					
					this->gseigs.compute(H0rL,SrL);
					
					vec evalsL = gseigs.eigenvalues().real();
					
					kappaevals.push_back(evalsL);
					int eigNL = evalsL.size();
					
					kappaevecs.push_back(this->gseigs.eigenvectors().real());
				}
				int iU = ki(kappa);
				
				if(((iU >= id0) && (iU <= id1)) || (ik(iU) == ik(id1)) ) {
					cout << "(" << kappa << ", " << iU << "," << ik(iU) << ")" << std::endl;
					
					kappas.push_back(kappa);
					
					this->bs->getRadial().setState(iU);
					
					dsmat H0rU = this->template H0<axis::radial>().real();
					dsmat SrU = this->template S<axis::radial>().real();
					
					this->gseigs.compute(H0rU,SrU);
					
					vec evalsU = gseigs.eigenvalues().real();
					
					kappaevals.push_back(evalsU);
					int eigNU = evalsU.size();
					
					kappaevecs.push_back(this->gseigs.eigenvectors().real());
				}
				
			}
			
			for(int k = 0; k < kappaevecs.size(); k++) {
				cout << "Applying sign convention to eigenvectors for kappa " << kappas[k] << "..." << endl; 
				for(int i = 0; i < kappaevecs[k].cols(); i++) {
					// cout << i << " ";
					
					//Determine if nonzero 
					double vmax = abs(kappaevecs[k].col(i).maxCoeff());
					double vmin = abs(kappaevecs[k].col(i).minCoeff());
					
					double vv;
					
					if(vmin>vmax) vv = vmin;
					else vv = vmax;
					
					int j = 0;
					while(true) {
						if(abs(kappaevecs[k](j,i)*100) > vv) {
							break;
						}
						j++;
					}
					
					if(kappaevecs[k](j,i) < 0) kappaevecs[k].col(i) = -kappaevecs[k].col(i);
				}
			}
		}
		}

	}
	
	std::vector<cmat> eigProj(const wavefunc<basistype>& psi) {
		// cout << psi.coefs << endl;
		// cout << "kappaevecs size = " << kappaevecs.size() << endl;
		int kappasize = kappaevecs.size();
		
		int lth0, lNth, ll0,lNl;
		
		int Nr = this->bs->radqN();
		
		this->bs->getLocalParams(lth0,lNth,ll0,lNl);
		
		//Need to get evecs corresponding to current kappa
		cout << "Nr = " << Nr << endl;
		cout << "local th0, Nth = " << lth0 << ", " << lNth << endl;
		
		std::vector<cmat> psievs(lNth - lth0);
		
		for(int th = lth0; th < lNth; th++) {
			cout << "Index: " << th 
				 << "\nTransformed index: " << this->bs->indexTransform(th) 
				 << "\nKappa: " << ik(this->bs->indexTransform(th)) << endl;
			int i = -1;
			
			
			
			for(int j = 0; j < kappas.size(); j++) {
				if(kappas[j] == ik(this->bs->indexTransform(th))) i = j;
			}
			
			// cout << "kappa index: " << i << endl;
			
			// cout << psi.coefs.size() << endl;
			
			this->bs->getRadial().setState(this->bs->indexTransform(th));
			// cvec psiseg = this->bs->getRadial().template matfree<S>(psi.coefs.reshaped(Nr,(lNth - lth0)).col(th-lth0));
			cvec psiseg = psi.coefs.reshaped(Nr,(lNth - lth0)).col(th-lth0);
			// cout << psiseg.rows() << ", " << psiseg.cols() << endl;
			
			cmat skev(Nr,Nr);
			
			#pragma omp parallel for
			for(int j = 0; j < kappaevecs[i].cols(); j++) {
				skev.col(j) = this->bs->getRadial().template matfree<S>(kappaevecs[i].col(j));
			}
			
			// cout << skev.rows() << ", " << skev.cols() << endl;
			Eigen::IOFormat outformat(Eigen::FullPrecision,Eigen::DontAlignCols,", ","\n","(","),"," = npy.array((\n","\n))\n",' ');
			
			cmat psiev = (psiseg.adjoint() * skev);
			
			psievs[th-lth0] = psiev;
			
		}
		
		return psievs;
	}
	
	void savePsievs(const wavefunc<basistype>& psi, std::string filename) {
		int wsize, wrank;
		
		MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
		MPI_Comm_size(MPI_COMM_WORLD, &wsize);
		Eigen::IOFormat outformat(Eigen::FullPrecision,Eigen::DontAlignCols,", ","\n","(","),"," = npy.array((\n","\n))\n",' ');
			
		cout << "Projecting onto eigenstates and saving...";
		
		std::vector<cmat> psievs = eigProj(psi);
		
		int lth0, lNth, ll0,lNl;
		this->bs->getLocalParams(lth0,lNth,ll0,lNl);
		
		
		for(int i = 0; i < wsize; i++) {
			ofstream psievf(filename,ofstream::app);
			if(wrank==i) {
				
				for(int i = 0; i < psievs.size(); i++) {
					int j = -1;
					
					for(int k = 0; k < kappas.size(); k++) {
						if(kappas[k] == ik(this->bs->indexTransform(lth0 + i))) j = k;
					}
					psievf << "evl[" << lth0 + i << "]" << kappaevals[j].format(outformat) << endl;
					psievf << "psiev[" << lth0 + i << "]" << psievs[i].format(outformat) << endl;
				}
			}
			
			psievf.close();
			
			MPI_Barrier(MPI_COMM_WORLD);
		}
	}
	
	cvec getevec(int N, int kappa, double mu) {
		int i = -1;
		for(int j = 0; j < kappas.size(); j++) {
			if(kappas[j] == kappa) i = j;
		}
		
		int lth0, lNth, ll0,lNl;
		
		this->bs->getLocalParams(lth0,lNth,ll0,lNl);
		
		cvec evec;
		
		if(i != -1) evec = kappaevecs[i].col(N);
		
		//Pad evec
		
		cmat evecpd = cmat::Zero(this->bs->radqN(), lNth - lth0);
		
		cout << "kappa, mu = " << kappa << ", " << mu << endl;
		
		int colid = this->bs->indexQn(kappa,mu);
		
		if(i != -1) evecpd.col(colid) = evec;
		
		return evecpd.reshaped(this->bs->radqN() * (lNth - lth0),1);
	}
	
	vec getevals(int kappa) {
		int i = -1;
		for(int j = 0; j < kappas.size(); j++) {
			if(kappas[j] == kappa) i = j;
		}
		
		// int lth0, lNth, ll0,lNl;
		
		// this->bs->getLocalParams(lth0,lNth,ll0,lNl);
		
		vec evl;
		
		if(i != -1) evl = kappaevals[i];
		
		
		return evl;
	}
};

template <typename basistype>
class Dirac: public DiracBase<Dirac<basistype>,basistype> {
	
	friend class DiracBase<Dirac<basistype>, basistype >;
	diracEigSolver gseigs;
	
	protected:
		
	public:
		Potential<dipolePulse>* vExt = NULL;
		
		Dirac(basis<basistype>& bs, Potential<dipolePulse>& vExt):DiracBase<Dirac<basistype>, basistype> (bs) {
			this->vExt = &vExt;
		}
		
		csmat Htmat(double t) {
			vExt->setTime(t);
			
			double Et = vExt->template axialPart<axis::t>(vExt->getTime());
			
			return Et * SoL * this->bs->dpam();;
		}

		cvec HtVec(double t, const cvec& v) const {
			vExt->setTime(t);
			
			double Et = vExt->template axialPart<axis::t>(vExt->getTime());
			
			return Et * SoL * this->bs->template matfree<dpa>(v);
		}
};

//Effectively a wrapper class for Dirac, but implementing beyond-dipole

template <typename basistype>
class DiracBDP: public DiracBase<DiracBDP<basistype>, basistype> {
	
	friend class DiracBase<DiracBDP<basistype>,basistype>;
	
	protected:
		//bdpft (*extPotT) (long double);
		
	public:
		Potential<beyondDipolePulse>* vExt = NULL;
	
		DiracBDP(basis<basistype>& bs, Potential<beyondDipolePulse>& vExt):DiracBase<DiracBDP<basistype>,basistype>(bs) {
			this->vExt = &vExt;
		}
		cvec HtVec(double t, const cvec& v) const {
			vExt->setTime(t);
			
			// bdpft vt = vExt->template axialPart<axis::t>(t);
			// cout << "f("<<t<<"):\n" << vt;
			
			cvec out = cvec::Zero(v.size());
			
			// cout << "out size: " << out.size() << std::endl;
			// cout << "matfree size: " << this->bs->template matfree<bdpa>(v,0).size() << std::endl;
			
			//for(int alpha = 0; alpha < 6; alpha++) {
			out = SoL * this->bs->template matfree<bdpa>(v);
			//}
			
			return out;
		}
		
		cmat HtVec(double t, const cmat& v) const {
			vExt->setTime(t);
			
			//bdpft vt = vExt->template axialPart<axis::t>(t);
			
			cmat out;// = cmat::Zero(v.size());
			
			// cout << "out size: " << out.size() << std::endl;
			// cout << "matfree size: " << this->bs->template matfree<bdpa>(v,0).size() << std::endl;
			
			//for(int alpha = 0; alpha < 6; alpha++) {
			out = SoL * static_cast<basistype*>(this->bs)->bdpamatvecMPIblock(v);
			//}
			
			return out;
		}
		
		csmat Htmat(double t) {
			vExt->setTime(t);
			
			bdpft vt = vExt->template axialPart<axis::t>(t);
			
			csmat out = csmat(this->S().rows(),this->S().cols());
			
			for(int alpha = 0; alpha < 6; alpha++) {
				out += (cdouble)vt(alpha,0) * this->bs->bdpam(alpha,*vExt);
			}
			
			return out;
		}
};

#endif

#ifndef BSBASIS_H
#define BSBASIS_H

#include "defs.h"
#include "basis.h"
#include "splinehandler.h"
#include "potential.h"
#include <omp.h>

class bsbasis: public basis<bsbasis>, public splineHandler {
	friend class basis<bsbasis>;
	
	protected:
		int lastdn = 0;
		
		int mfl = 0;
		
		csmat& olmat();
		csmat& ddmat(int dn);
		csmat& Lmat();
		csmat& Lsqmat();
		csmat& Emat(long double (*V)(long double));
		csmat& bdppzmat(int l, const Potential<beyondDipolePulse>& vExt);
		csmat& bdpApzmat(int l, int ll, int ul, const Potential<beyondDipolePulse>& vExt);
		csmat& bdpAsqmat(int l, const Potential<beyondDipolePulse>& vExt);
		csmat& H0mat();
		int rqn();
		
		
		enum bdpMatTypes {
			AL,
			Adr,
			Asq,
			N_BDPMATS
		};
		
		std::vector<std::vector<csmat> > bdpmats[N_BDPMATS];
		
		std::vector<std::vector<cvec> > cachevecs[2];
		
		
		
		template<bdpMatTypes M>
		csmat& getbdpmat(int l, int alpha) {
			//out <<"l: " << l << ", size of bdpmats: " << bdpmats.size() << "\n";
			if(alpha >= bdpmats[M].size()) {
				for(int i = bdpmats[M].size(); i < alpha+1; i++) {
					bdpmats[M].push_back(std::vector<csmat>(0));
				}
			}
			
			if(l >= bdpmats[M][alpha].size()) {
				for(int i = bdpmats[M][alpha].size(); i < l+1; i++) {
					bdpmats[M][alpha].push_back(csmat(0,0));
				}
			}
			//cout <<"l: " << l << ", size of bdpmat: " << bdpmats[alpha][l].size() << "\n";
			return bdpmats[M][alpha][l];
		}
		
		//const Potential<beyondDipolePulse>* vExt;
	public:
		cvec& getcachevec(int cacheId, int l, int cacheN = 0) {
			
			
			if(cacheId >= cachevecs[cacheN].size()) {
				for(int i = cachevecs[cacheN].size(); i < cacheId+1; i++) {
					cachevecs[cacheN].push_back(std::vector<cvec> (0));
					for(int j = 0; j < l+1; j++) {
						//cout << i << " " << j << "\n";
						cachevecs[cacheN][i].push_back(cvec(0));
					}
					
				}
			}
				
			if(l >= cachevecs[cacheN][cacheId].size()) {
				
					
				for(int i = cachevecs[cacheN][cacheId].size(); i < l+1; i++) {
					cachevecs[cacheN][cacheId].push_back(cvec(0));
				}
			}	
			
			
			//cout <<"cacheId: " << cacheId << ", l: " << l << "\n";
			return cachevecs[cacheN][cacheId][l];
		}
		
		
		const Potential<beyondDipolePulse>* vExt = NULL;
		
		bsbasis(vec& t, int splinedeg);
		bsbasis() {t = vec(0); }
		
		void setst(int i) {
			this->mfl = il(i);
		}
		
		void setst(int i, int j, int ul) {
			this->mfl = il(i);
		}
		
		template<matname M, typename... ArgTs>
		inline void cacheMatvec(const cvec& v, cvec& w, int cacheId, cdouble coef, ArgTs... Args);
		
		
		template<matname M>
		inline void precacheMatvec(const cvec& v, int cacheSize, int lmax);
		
		void clvcs() {
			for(int i = 0; i < 2; i++) {
				for(int j = 0; j < cachevecs[i].size(); j++) {
					for(int k = 0; k < cachevecs[i][j].size(); k++) {
						cachevecs[i][j][k] = cvec(0);
					}
				}
			}
		}
		
		void matmat_impl(std::integral_constant<matname,matname::H0> c, const cmat& m, cmat& w, cvec params) {
	
			csmat& H0_0 = getH0mat(0);
			csmat& H0_l = getH0mat(1);
			
			// cout << "l array: " << larray << std::endl;
			
			// cout << "Hlm dims: (" << Hlm.rows() << ", " << Hlm.cols() << ")\n";
			
			// cout << "params dims: (" << params.rows() << ", " << params.cols() << ")\n";
			
			// cmat Hlml = Hlm.array().rowwise() * params.array().transpose();
			
			iarray ls = il(params.real().cast<int>());
			
			w.noalias() += H0_0.selfadjointView<Eigen::Upper>() * m;
			
			w.noalias() += ((H0_l * m).array().rowwise() * (ls * (ls+1)).cast<cdouble>().transpose()).matrix();
			
		}
		
		void matmat_impl(std::integral_constant<matname,matname::pz> c, const cmat& m, cmat& w, cvec params, int l, int ul) {
	
			iarray ls = il(params.real().cast<int>());
			
			csmat& drm = getmat<dd>();
			csmat& Lm = getmat<L>();
			
			if(ul == 1)
				w.noalias() += (drm * m + (Lm * (m.array().rowwise() *      ls.transpose().cast<cdouble>()).matrix()));
			if(ul == -1)
				w.noalias() += (drm * m + (Lm * (m.array().rowwise() * (-ls-1).transpose().cast<cdouble>()).matrix()));

		}
		
		void matmat_impl(std::integral_constant<matname,matname::bdpApx> c, const cmat& m, cmat& w, cvec params, int l, int ul) {
	
			
			// cout << "l array: " << larray << std::endl;
			
			// cout << "Hlm dims: (" << Hlm.rows() << ", " << Hlm.cols() << ")\n";
			
			// cout << "params dims: (" << params.rows() << ", " << params.cols() << ")\n";
			
			// cmat Hlml = Hlm.array().rowwise() * params.array().transpose();
			
			iarray ls = il(params.real().cast<int>());
	
			bdpft pT = vExt->template axialPart<axis::t>(vExt->getTime());
			
			// cmat wl = cmat::Zero(w.rows(),w.cols());
			
			// for(int halpha = 0; halpha < 3; halpha++) {  //Half Alpha or halpha, haha geddit? ;)
				// int alpha = 2*halpha + l%2;
				
				// csmat& ALl = getbdpmat<AL>(l,alpha);
				// double AT = pT(alpha,0).real();
				
				// wl.noalias() += AT * ALl * m;
			// }
			
			// if(ul == 1)
			// w += (wl.array().rowwise() * ls.cast<cdouble>().transpose()).matrix();
			// if(ul == -1)
			// w += -(wl.array().rowwise() * (ls+1).cast<cdouble>().transpose()).matrix();
			
			// for(int halpha = 0; halpha < 3; halpha++) {
				// int alpha = 2*halpha + l%2;
				// csmat& Adrl = getbdpmat<Adr>(l,alpha);
				
				// double AT = pT(alpha,0).real();
				
				// w.noalias() += AT * Adrl * m;
			// }
			
			// csmat lsm(ls.size(),ls.size());
			// lsm.setIdentity();
			
			// cmat ml;
			
			// if(ul == 1)
				// ml = (m.array().rowwise() * ls.transpose().cast<cdouble>()).matrix();
				// // lsm.diagonal() = ls.matrix().cast<cdouble>();
			// if(ul ==-1)
				// ml = (m.array().rowwise() * (-ls-1).transpose().cast<cdouble>()).matrix();
				// // lsm.diagonal() = (-ls-1).matrix().cast<cdouble>();
			
			// cmat ml = m * lsm;
			
			cmat wpart[3];
			
			//#pragma omp parallel for
			for(int alpha = l%2; alpha < 6; alpha+=2) {
				
				csmat& Adrl = getbdpmat<Adr>(l,alpha);
				csmat& ALl = getbdpmat<AL>(l,alpha);
				
				// cout << "Adrl: " << Adrl.rows() << ", " << Adrl.cols() << std::endl;
				// cout << "ALl: " << ALl.rows() << ", " << ALl.cols() << std::endl;
				// cout << "lsm: " << lsm.rows() << ", " << lsm.cols() << std::endl;
				// cout << "m: " << m.rows() << ", " << m.cols() << std::endl;
				// wpart[(alpha - l%2)/2] = pT(alpha,0).real() * (Adrl * m + ALl * ml);
				if(ul == 1)
					w.noalias() += pT(alpha,0).real() * (Adrl * m + (ALl * (m.array().rowwise() *      ls.transpose().cast<cdouble>()).matrix()));
				if(ul == -1)
					w.noalias() += pT(alpha,0).real() * (Adrl * m + (ALl * (m.array().rowwise() * (-ls-1).transpose().cast<cdouble>()).matrix()));
			}
			
			// for(int i = 0; i < 3; i++) {
				// w.noalias() += wpart[i];
			// }
		}
		
		
		void matmat_impl(std::integral_constant<matname,matname::bdpAsq> c, const cmat& m, cmat& w, cvec params, int l, int ul) {
			
			// iarray ls = il(params.real().cast<int>());
	
			bdpsqft pT = vExt->template axialPartSq<axis::t>(vExt->getTime(),l);
			
			for(int alpha = 0; alpha < 8; alpha++) {
				
				csmat& Asqm = getbdpmat<Asq>(l,alpha);
				
				double AT = pT(alpha,0).real();
				
				w.noalias() += AT * Asqm * m;
			}
		}
		
		template <matname M>
		void matmat_impl(std::integral_constant<matname,M> c, const cmat& m, cmat& w, cvec params) {
	
			csmat& M0 = getmat<M>();
			
			// cout << "l array: " << larray << std::endl;
			
			// cout << "Hlm dims: (" << Hlm.rows() << ", " << Hlm.cols() << ")\n";
			
			// cout << "params dims: (" << params.rows() << ", " << params.cols() << ")\n";
			
			// cmat Hlml = Hlm.array().rowwise() * params.array().transpose();
			
			
			w.noalias() += M * m;
			
		}
};


template<>
inline void bsbasis::precacheMatvec<bdpAsq>(const cvec& v, int cacheSize, int lmax) { 
	
	//Consider: What if we do the time dependent part only once?
	
	std::vector<csmat> bAsq(lmax);
	
	//std::vector<cvec> vparts(cacheSize);
	
	//cout << "Address of vExt: " << &(this->vExt) << std::endl;

	/*for(int l = 0; l < lmax; l++) {
		bAsq[l] = csmat(radqN(),radqN());
		
		for(int alpha = 0; alpha < 8; alpha++) {
			//if(l%2 != alpha%2) {
				//cout << "l: " << l <<", alpha: " << alpha << "\n";
				csmat& Asql = getbdpmat<Asq>(l,alpha);
				
				double pT = (double) this->vExt->template axialPartSq<axis::t>(this->vExt->getTime(),l)(alpha,0).real();
				
				//cout << pT << ",\n";
				
				bAsq[l].noalias() += (pT *  Asql);
			//}
		}
		//cout << bAsq[l] << std::endl;
	}

	*/

	int kappamax = (lmax - 1)/2;
	
	#pragma omp parallel for collapse(2)
	for(int cacheId = 0; cacheId < cacheSize; cacheId++) {
		for(int l = 0; l < lmax; l++) { 
			cvec& vcache = getcachevec(cacheId,l);
			if(!isCached(vcache)) {
				vcache = cvec::Zero(radqN());
				for(int alpha = 0; alpha < 8; alpha++) {
					
					csmat& Asql = getbdpmat<Asq>(l,alpha);
				
					double pT = (double) this->vExt->template axialPartSq<axis::t>(this->vExt->getTime(),l)(alpha,0).real();
				
					vcache.noalias() += (pT * Asql * v.segment(cacheId*radqN(),radqN()));
					
				}
			}
			//cout << vcache << "\n";
		}
	}
}


template<>
inline void bsbasis::cacheMatvec<bdpAsq>(const cvec& v, cvec& w, int cacheId, cdouble coef, int l) { 
	cvec& vcache = getcachevec(cacheId,l);
	
	//cout << "vcache:\n " <<vcache; 
	
	w.noalias() += coef * vcache;
	
}

template<>
inline void bsbasis::precacheMatvec<bdpApz>(const cvec& v, int cacheSize, int lmax) { 
	
	//Consider: What if we do the time dependent part only once?
	
	//std::vector<cvec> vparts(cacheSize);
	/*
	for(int l = 0; l < lmax; l++) {
		bAdr[l] = csmat(radqN(),radqN());
		bAL[l] = csmat(radqN(),radqN());
		
		for(int alpha = 0; alpha < 6; alpha++) {
			if(l%2 != alpha%2) {
				//cout << "l: " << l <<", alpha: " << alpha << "\n";
				csmat& Adrl = getbdpmat<Adr>(l,alpha);
				csmat& ALl = getbdpmat<AL>(l,alpha);
				
				double pT = (double) vExt->template axialPart<axis::t>(vExt->getTime())(alpha,0).real();
				
				//cout << "Adrl: " << Adrl.rows() << std::endl;
				//cout << "ALl: " << ALl.rows() << std::endl;
				
				bAdr[l] += (pT *  Adrl);
				bAL[l] += (pT * ALl);
			}
		}
	}
	*/
	bdpft pT = vExt->template axialPart<axis::t>(vExt->getTime());
	// #pragma omp parallel for collapse(2)
	for(int cacheId = 0; cacheId < cacheSize; cacheId++) {
		for(int l = 0; l < lmax; l++) { 
			
			cvec& vcachedr = getcachevec(cacheId,l);
			cvec& vcacheL  = getcachevec(cacheId,l,1);
			
			if(!isCached(vcachedr)) {
				vcachedr = cvec::Zero(radqN());
				vcacheL = cvec::Zero(radqN());
				// cout << "A(t) = " << dplA<15,50,10>(vExt->getTime())<<std::endl;
				for(int alpha = 0; alpha < 6; alpha++) {
					if(l%2 == alpha%2) {							
						csmat& Adrl = getbdpmat<Adr>(l,alpha);
						csmat& ALl = getbdpmat<AL>(l,alpha);
						
						double At = (double) pT(alpha,0).real();
						
						// cout << "At("<<alpha<<") = " << At<<std::endl;
						
						vcachedr.noalias() += (At * Adrl * v.segment(cacheId*radqN(),radqN()));
						vcacheL.noalias()  += (At * ALl * v.segment(cacheId*radqN(),radqN()));
					}
				}
			}
		}
	}
}

template<>
inline void bsbasis::cacheMatvec<bdpApx>(const cvec& v, cvec& w, int cacheId, cdouble coef, int l, int ll, int ul) { 
	cvec& vcachedr = getcachevec(cacheId,l);
	cvec& vcacheL  = getcachevec(cacheId,l,1);
	
	// cout << "vdr " << vcachedr.size() << std::endl;
	// cout << "vL  " << vcacheL.size() << std::endl;

	
	switch(ul) {
		case 1:
			w.noalias() += coef * ((vcachedr) + (il(ll)) * vcacheL);
			
			//cout << "Col: " << ll << ", l: " << il(ll) << std::endl;
			
			break;
		case -1:
			w.noalias() += coef  * ((vcachedr) - (il(ll) + 1) * vcacheL);
			
			//cout << "Col: " << ll << ", l: " << il(ll) << std::endl;
			
			break;
	}
}

template<>
inline void bsbasis::cacheMatvec<bdpApz>(const cvec& v, cvec& w, int cacheId, cdouble coef, int l, int ll, int ul) { 
	cacheMatvec<bdpApx>(v,w,cacheId,coef,l,ll,ul);
	return;
	
	cvec& vcachedr = getcachevec(cacheId,l);
	cvec& vcacheL  = getcachevec(cacheId,l,1);
	
	// cout << "vdr " << vcachedr.size() << std::endl;
	// cout << "vL  " << vcacheL.size() << std::endl;

	
	switch(ul) {
		case 1:
			w.noalias() += coef * ((vcachedr) + (il(ll)) * vcacheL);
			
			//cout << "Col: " << ll << ", l: " << il(ll) << std::endl;
			
			break;
		case -1:
			w.noalias() += coef  * ((vcachedr) - (il(ll) + 1) * vcacheL);
			
			//cout << "Col: " << ll << ", l: " << il(ll) << std::endl;
			
			break;
	}
	
}



template<>
inline void bsbasis::cacheMatvec<H0>(const cvec& v, cvec& w, int cacheId, cdouble coef, int k) {
	int l = il(k);
	
	csmat& H0_0 = getH0mat(0);
	csmat& H0_l = getH0mat(1);
	
	w.noalias() += H0_0.selfadjointView<Eigen::Upper>() * v + H0_l.selfadjointView<Eigen::Upper>() * ((l*(l+1)) * v);
}



#endif
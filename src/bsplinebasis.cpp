#include "bsplinebasis.h"




vec padt(vec t, int splinedeg) {
	vec tpad(t.size() + splinedeg*2);
	
	for(int i = 0; i < splinedeg; i++) {
		tpad[i] = t[0];
	}
	for(int i = 0; i < t.size(); i++) {
		tpad[i+splinedeg] = t[i];
	}
	for(int i = 0; i < splinedeg; i++) {
		tpad[i+t.size()+splinedeg] = t[t.size()-1];
	}
	
	return tpad;
}



clsmat initSplineCache(lvec x,vec t,int splinedeg,bool glmode=true) {
	//SplineCache is a sparse matrix where the nth column corresponds to the nth b-spline's numerical values in the chosen x-coordinates
	
	vec tpad = padt(t,splinedeg);
	
	//cout << tpad << "\n";
	
	int Nx = x.size();
	int Nspl = tpad.size() - 1 - splinedeg;
	
	//initialise SplineCache through de Boor's Algorithm
	clsmat *splineCache = new clsmat(Nx,Nspl);
	clsmat *tempSplines;
	
	std::vector<ldT> tripletsInit;
	tripletsInit.reserve((splinedeg + 1) * Nspl);
	
	for(int i = 0; i < Nspl + 1; i++){
		for(int j = 0; j < Nx; j++) {
			if( x[j] >= tpad[i] && x[j] < tpad[i+1]) {
				tripletsInit.push_back(ldT(j,i,1.0));
				//splineCache->insert(j,i) = 1.0;
			}
		}
	}
	
	splineCache->setFromTriplets(tripletsInit.begin(),tripletsInit.end());
	
	for(int k = 1; k < splinedeg+1; k++) {
		//std::vector<ldT> tripletsItr;
		//tripletsItr.reserve((splinedeg+1)*k*Nspl);
		
		tempSplines = new clsmat(Nx,Nspl);
		for(int i = 0; i + k + 1 < tpad.size(); i++) {
			//cout << i << ":\n";
			//cout << tpad[i] <<" , " << tpad[i+k+1] << "\n";
			//if(i < tempSplines->cols())	cout << i << ": " << tempSplines->col(i).rows() << ", " << tempSplines->col(i).cols() << std::endl;
			
			if(i + k + 1 < tpad.size()) {
				if (tpad[i + k + 1] != tpad[i + 1]) {
					lvec xshift = lvec::Constant(Nx,1,tpad[i + k + 1]);
					long double xmul = 1.0 / (tpad[i + k + 1] - tpad[i+1]);
					tempSplines->col(i)+= splineCache->col(i+1).cwiseProduct(xmul * (xshift - x));
				}
			}
			
			if(i + k < tpad.size()) {
				if(tpad[i] != tpad[i+k]) {
					lvec xshift = lvec::Constant(Nx,1,tpad[i]);
					long double xmul = 1.0 / (tpad[i + k] - tpad[i]);
					tempSplines->col(i)+= splineCache->col(i).cwiseProduct(xmul * (x - xshift));
				}
			}
					
			/*for(int j = 0; j < Nx; j++) {
				//cout <<"("<< i << "," << j << ") ";
				if (x[j] >= tpad[i] && x[j] < tpad[i + k + 1]) {
					long double splvl = 0;
					//tempSplines->insert(j,i) = 0;
					
					
					if(i + k + 1 < tpad.size()) {
						if (tpad[i + k + 1] != tpad[i + 1]) {
							//tempSplines -> coeffRef(j,i) += (tpad[i + k + 1] - x[j]) / (tpad[i + k + 1] - tpad[i+1]) * splineCache->coeffRef(j,i+1);
							splvl += ((tpad[i + k + 1] - x[j]) / (tpad[i + k + 1] - tpad[i+1]) * splineCache->coeffRef(j,i+1).real());
						}
					}
					
					if(i + k < tpad.size()) {
						if(tpad[i] != tpad[i+k]) {
							//tempSplines -> coeffRef(j,i) += (x[j] - tpad[i]) / (tpad[i + k] - tpad[i]) * splineCache->coeffRef(j,i);
							splvl += ((x[j] - tpad[i]) / (tpad[i + k] - tpad[i]) * splineCache->coeffRef(j,i).real());
						}
					}
					if(splvl != 0.0) {
						tripletsItr.push_back(ldT(j,i,splvl));						
					}
				}*/
			//}
			//cout << "\n";
		}
		//tempSplines->setFromTriplets(tripletsItr.begin(),tripletsItr.end());
		
		delete(splineCache);
		splineCache = tempSplines;
		//cout << *splineCache << "\n";
		
	}
	
	clsmat out = *splineCache;
	return out;
}

clsmat dSplineCache(lvec x,vec t,int splinedeg,int dn, bool glmode = true) {
	vec tpad = padt(t,splinedeg);
	
	int Nx = x.size();
	int Nspl = tpad.size() - 1-splinedeg;
	
	//initialise SplineCache through de Boor's Algorithm
	clsmat *splineCache = new clsmat(Nx,Nspl);
	clsmat *tempSplines;
	
	std::vector<ldT> tripletsInit;
	tripletsInit.reserve((splinedeg + 1) * Nspl);
	
	for(int i = 0; i < Nspl + 1; i++) {
		for(int j = 0; j < Nx; j++) {
			if( x[j] >= tpad[i] && x[j] < tpad[i+1]) {
				tripletsInit.push_back(ldT(j,i,1.0));
				//splineCache->insert(j,i) = 1.0;
			}
		}
	}
	
	splineCache->setFromTriplets(tripletsInit.begin(),tripletsInit.end());
	
	for(int k = 1; k < splinedeg+1 - dn; k++) {
		
		std::vector<ldT> tripletsItr;
		tripletsItr.reserve((splinedeg+1)*k*Nspl);
		
		tempSplines = new clsmat(Nx,Nspl);
		for(int i = 0; i + k + 1 < tpad.size(); i++) {
			//cout << i << ":\n";
			//cout << tpad[i] <<" , " << tpad[i+k+1] << "\n";
			for(int j = 0; j < Nx; j++) {
				//cout <<"("<< i << "," << j << ") ";
				if (x[j] >= tpad[i] && x[j] < tpad[i + k + 1]) {
					
					long double splvl = 0;
					//tempSplines->insert(j,i) = 0;
					
					if(i + k + 1 < tpad.size()) {
						if (tpad[i + k + 1] != tpad[i + 1]) {
							//tempSplines -> coeffRef(j,i) += (tpad[i + k + 1] - x[j]) / (tpad[i + k + 1] - tpad[i+1]) * splineCache->coeffRef(j,i+1);
							splvl += ((tpad[i + k + 1] - x[j]) / (tpad[i + k + 1] - tpad[i+1]) * splineCache->coeffRef(j,i+1).real());
						}
					}
					
					if(i + k < tpad.size()) {
						if(tpad[i] != tpad[i+k]) {
							//tempSplines -> coeffRef(j,i) += (x[j] - tpad[i]) / (tpad[i + k] - tpad[i]) * splineCache->coeffRef(j,i);
							splvl += ((x[j] - tpad[i]) / (tpad[i + k] - tpad[i]) * splineCache->coeffRef(j,i).real());
						}
					}
					
					if(splvl != 0.0) {
						tripletsItr.push_back(ldT(j,i,splvl));						
					}
				}
			}
			//cout << "\n";
		}
		tempSplines->setFromTriplets(tripletsItr.begin(),tripletsItr.end());
		
		delete(splineCache);
		splineCache = tempSplines;
		//cout << *splineCache << "\n";
		
	}
	
	for(int k = splinedeg + 1 - dn; k < splinedeg+1; k++) {
		
		std::vector<ldT> tripletsItr;
		tripletsItr.reserve((splinedeg+1)*k*Nspl);
		
		tempSplines = new clsmat(Nx,Nspl);
		for(int i = 0; i + k + 1 < tpad.size(); i++) {
			//cout << i << ":\n";
			//cout << tpad[i] <<" , " << tpad[i+k+1] << "\n";
			for(int j = 0; j < Nx; j++) {
				//cout <<"("<< i << "," << j << ") ";
				if (x[j] >= tpad[i] && x[j] < tpad[i + k + 1]) {

					long double splvl = 0;
					//tempSplines->insert(j,i) = 0;
					
					
					if(i + k + 1 < tpad.size()) {
						if (tpad[i + k + 1] != tpad[i + 1]) {
							//tempSplines -> coeffRef(j,i) -= (long double)(k / (tpad[i + k + 1] - tpad[i+1])) * splineCache->coeffRef(j,i+1);
							splvl -= ((k / (tpad[i + k + 1] - tpad[i+1])) * splineCache->coeffRef(j,i+1).real());
						}
					}
					
					if(i + k < tpad.size()) {
						if(tpad[i] != tpad[i+k]) {
							//tempSplines -> coeffRef(j,i) += (long double)(k / (tpad[i + k] - tpad[i])) * splineCache->coeffRef(j,i);
							splvl += ((k / (tpad[i + k] - tpad[i])) * splineCache->coeffRef(j,i).real());
						}
					}
					
					if(splvl != 0.0) {
						tripletsItr.push_back(ldT(j,i,splvl));						
					}
				}
			}
			//cout << "\n";
		}
		tempSplines->setFromTriplets(tripletsItr.begin(),tripletsItr.end());
		
		delete(splineCache);
		splineCache = tempSplines;
		//cout << *splineCache << "\n";
		
	}
	
	clsmat out = *splineCache;
	delete(splineCache);
	return out;
}

clsmat initdSplineCache(lvec x,vec t,int splinedeg,int dn) {
	return dSplineCache(x,t,splinedeg,dn);
}

lvec glpts(vec t, int l) {
	vector<long double> glzeros = boost::math::legendre_p_zeros<long double>(l);
	
	lvec glpts(l*(t.size()-1));
	lvec w(l);
	
	/*
	wi = 2/((1-pow(xi,2))*pow(npy.polynomial.legendre.Legendre(c).deriv()(xi),2))
	
	*/
	
	int odd = l%2;
	int midpt = int(l/2);
	
	
	
	for(int i = 0; i < t.size() - 1;i++) {
		if(odd) {
			glpts[i*l + midpt] = (glzeros[0] + 1.0)*((t[i+1] - t[i]))/2 + t[i];
			for(int j = 1; j < glzeros.size(); j++) {
				glpts[i*l + midpt + j] = (1.0 + glzeros[j])*((t[i+1] - t[i]))/2 + t[i];
				glpts[i*l + midpt - j] = (1.0 - glzeros[j])*((t[i+1] - t[i]))/2 + t[i];
			}
		}
		else {
			for(int j = 0; j < glzeros.size(); j++) {
				glpts[i*l + midpt + j] = (1.0 + glzeros[j])*((t[i+1] - t[i]))/2 + t[i];
				glpts[i*l + midpt - j - 1] = (1.0 - glzeros[j])*((t[i+1] - t[i]))/2 + t[i];
			}
		}
		
	}
	return glpts;
}

template <class T>
T dLegendre(T x, int l) {
	T P0 = boost::math::legendre_p<T>(l-1,x);
	T P1 = boost::math::legendre_p<T>(l,x);
	T dP = l/(T(pow(double(x),2)) - 1) * (x*P1 - P0);
	
	return dP;
}

template <class T>
T glwt(T x, int l) {
	return 2/((1-pow(x,2)) * pow(dLegendre(x,l),2));
}

template <class T>
std::vector<T> glwts(int l) {
	std::vector<T> out(l);
	std::vector<T> zeros = boost::math::legendre_p_zeros<T>(l);
	
	int odd = l%2;
	int midpt = int(l/2);
	
	if(odd) {
		out[midpt] = glwt<T>(zeros[0],l);
		for(int j = 1; j < zeros.size(); j++) {
			out[midpt + j] = glwt<T>(zeros[j],l);
			out[midpt - j] = glwt<T>(zeros[j],l);
		}
	}
	else {
		for(int j = 0; j < zeros.size(); j++) {
			out[midpt + j] = glwt<T>(zeros[j],l);
			out[midpt - j - 1] = glwt<T>(zeros[j],l);
		}
	}
	
	return out;
}

clsmat wtmat(Eigen::VectorXd t, int l) {
	int N = l * (t.size()-1);
	clsmat wtmat(N,N);
	std::vector<long double> glwtlist = glwts<long double>(l);
	for(int i = 0; i < N; i++) {
		int j = int(i/l);
		wtmat.insert(i,i) = glwtlist[i%l] * (t[j+1] - t[j])/2 ;
	}
	return wtmat;
}

csmat glintmat(const clsmat& a,const clsmat& b,const clsmat& ws,int k) {
	int N = a.cols();
	
	clsmat M(N,N);
	
	M = a.adjoint() * ws * b;
	
	return M.cast<complex<double> >();
}



splineHandler::splineHandler(vec& t, int splinedeg) {
	this->t = t;
	this->splinedeg = splinedeg;
	this->splineCaches = vector<clsmat>(splinedeg);
	for(int i = 0; i < splinedeg; i++) {
		this->splineCaches[i] = clsmat(0,0);
	}
}



void splineHandler::checkx(const lvec& x) {
	if(&lastx != &x) {
		
		for(int i = 0; i < splinedeg; i++) {
			if (isCached(splineCaches[i])) { 
				splineCaches[i] = clsmat(0,0);
			}
		}
		
		lastx = x;
	}
}

lvec& splineHandler::glpts() {
	if (isCached(glpch)) return glpch;
	else {
		
		int l = splinedeg + 1;
		
		std::vector<long double> glzeros = boost::math::legendre_p_zeros<long double>(l);
	
		glpch = lvec(l*(t.size()-1));
		
		int odd = l%2;
		int midpt = int(l/2);
		
		for(int i = 0; i < t.size() - 1;i++) {
			if(odd) {
				glpch[i*l + midpt] = (glzeros[0] + 1.0)*((t[i+1] - t[i]))/2 + t[i];
				for(int j = 1; j < glzeros.size(); j++) {
					glpch[i*l + midpt + j] = (1.0 + glzeros[j])*((t[i+1] - t[i]))/2 + t[i];
					glpch[i*l + midpt - j] = (1.0 - glzeros[j])*((t[i+1] - t[i]))/2 + t[i];
				}
			}
			else {
				for(int j = 0; j < glzeros.size(); j++) {
					glpch[i*l + midpt + j] = (1.0 + glzeros[j])*((t[i+1] - t[i]))/2 + t[i];
					glpch[i*l + midpt - j - 1] = (1.0 - glzeros[j])*((t[i+1] - t[i]))/2 + t[i];
				}
			}
			
		}
		
		return glpch;
	}
}

clsmat& splineHandler::wtmat() {
	if (isCached(wtmt)) return wtmt;
	else {
		int l = this->splinedeg + 1;
		
		int N = l * (t.size()-1);
		wtmt = clsmat(N,N);
		std::vector<long double> glwtlist = glwts<long double>(l);
		for(int i = 0; i < N; i++) {
			int j = int(i/l);
			wtmt.insert(i,i) = glwtlist[i%l] * (t[j+1] - t[j])/2 ;
		}
		
		return wtmt;
	}
}

clsmat& splineHandler::splineCache(const lvec& x) {
	checkx(x);
	if(isCached(splineCaches[0])) {
		return splineCaches[0];
	}
		
	else {
		clsmat scraw = initSplineCache(x, t, splinedeg);
		splineCaches[0] = scraw.middleCols(dropfirst,scraw.cols() - (dropfirst + droplast));
		
		return splineCaches[0];
	}
}

clsmat& splineHandler::splineCache() {
	if(isCached(splineCaches[0]))
		return splineCaches[0];
	
	else 
		return splineCache(glpts());
}

clsmat& splineHandler::dSplineCache(const lvec& x, int dn) {
	checkx(x);
	
	if(isCached(splineCaches[dn])) {
		return splineCaches[dn];
	}
	else {
		
		clsmat dscraw = initdSplineCache(x, t, splinedeg,dn);
		splineCaches[dn] = dscraw.middleCols(dropfirst, dscraw.cols() - (dropfirst + droplast));
		return splineCaches[dn];
	}
}

bsbasis::bsbasis(vec& t, int splinedeg): splineHandler(t,splinedeg) {
	
}

csmat &bsbasis::olmat() {
	csmat& Sm = getmat<S>();
	
	if (isCached(Sm)) return Sm;
	else {
		clsmat a = splineCache(glpts());
		clsmat wt = wtmat();
		
		//cout << *a << '\n' << *wt << '\n';
		
		Sm= glintmat(a,a,wt,splinedeg+1);
		return Sm;
	}
}

csmat &bsbasis::ddmat(int dn) {
	csmat& ddm = getmat<dd>();
	csmat& pzm = getmat<pz>();
	if (isCached(ddm) && dn == 2) return ddm;
	if (isCached(pzm) && dn == 1) return pzm;
	
	clsmat a = splineCache(glpts());
	clsmat b = dSplineCache(glpts(),dn);
	clsmat wt = wtmat();
	
	lastdn = dn;
	
	if(dn == 2) {
		ddm = glintmat(a,b,wt,splinedeg+1);
		return ddm;
	}
	if(dn == 1) {
		pzm = glintmat(a,b,wt,splinedeg+1);
		return pzm;
	}
	
	ddm = glintmat(a,b,wt,splinedeg+1);
	return ddm;
}

csmat &bsbasis::Emat(long double (*V)(long double)) {
	csmat& Em = getmat<E>();
	if (isCached(Em)) return Em;
	else {
		lvec gls = glpts();
		clsmat splcch = splineCache(gls);
		int N = splcch.rows();
		clsmat vdiag(N,N);
		
		for(int i = 0; i < N; i++) {
			vdiag.insert(i,i) = V(gls[i]);
		}
		
		clsmat Vsplcch = vdiag * splcch;
		clsmat wt = wtmat();
		Em = glintmat(splcch,Vsplcch,wt,splinedeg+1);
		
		return Em;
	}
}

csmat &bsbasis::Lmat() {
	csmat& Lm = getmat<L>();
	
	if(isCached(Lm)) return Lm;
	lvec gls = glpts();
	clsmat splcch = splineCache(gls);
	int N = splcch.rows();
	clsmat vdiag(N,N);
	
	std::vector<T> triplets;
	triplets.reserve(N);
	
	for(int i = 0; i < N; i++) {
		triplets.push_back(T(
			i,
			i,
			1/gls[i]
		));
	}
	
	vdiag.setFromTriplets(triplets.begin(),triplets.end());
	
	clsmat Vsplcch = vdiag * splcch;
	clsmat wt = this->wtmat();
	Lm = glintmat(splcch,Vsplcch,wt,splinedeg+1);
	
	return Lm;
}

csmat& bsbasis::Lsqmat() {
	csmat& lsqm = getmat<Lsq>();
	if(isCached(lsqm)) return lsqm;
	lvec gls = glpts();
	clsmat splcch = splineCache(gls);
	int N = splcch.rows();
	clsmat vdiag(N,N);
	
	std::vector<T> triplets;
	triplets.reserve(N);
	
	for(int i = 0; i < N; i++) {
		triplets.push_back(T(
			i,
			i,
			0.5 * 1/pow(gls[i],2)
		));
	}
	
	vdiag.setFromTriplets(triplets.begin(),triplets.end());
	
	clsmat Vsplcch = vdiag * splcch;
	clsmat wt = this->wtmat();
	lsqm = glintmat(splcch,Vsplcch,wt,splinedeg+1);
	
	return lsqm;
}

csmat& bsbasis::bdpApzmat(const int l, const int ll, const int ul, const Potential<beyondDipolePulse>& vExt) {
	this->vExt = &vExt;
	csmat& dpalphm  =  getmat<bdpApz>();
	/*csmat& dpalph0  = getbdpmat<p00>(l,0);
	csmat& dpalph0k = getbdpmat<p0k>(l,0);
	csmat& dpalphk0 = getbdpmat<pk0>(l,0);
	csmat& dpalphkk = getbdpmat<pkk>(l,0);
	*/
	if(!isCached(getbdpmat<Adr>(l,0))) {
		lvec x = glpts();
		
		clsmat& wt = wtmat();
		
		//cout << "Size of x coord: " << x.size() << std::endl;
		//Explicitly put time dependence in bdpalphmat!
		bdpft Calph = vExt.template axialPart<axis::radial>(x,l);
		
		clsmat wtC[6];
		for(int alpha = 0; alpha < 6; alpha++) {
			wtC[alpha] = wt * Calph.row(alpha).asDiagonal();
		}
		
			
		clsmat vdiag(x.size(),x.size());
		
		std::vector<T> triplets;
		triplets.reserve(x.size());
		
		for(int i = 0; i < x.size(); i++) {
			triplets.push_back(T(
				i,
				i,
				1/x[i]
			));
		}
		
		vdiag.setFromTriplets(triplets.begin(),triplets.end());
		
		clsmat P = splineCache(x);
		clsmat dP = dSplineCache(x,1);
		
		
		// cout << "products\n";
		for(int alpha = 0; alpha < 6; alpha++) {
			
			getbdpmat<Adr>(l,alpha) = P.adjoint() * wtC[alpha] * dP;
			getbdpmat<AL>(l,alpha) = P.adjoint() * (wtC[alpha] * vdiag) * P;
		}
	}
	
	// cout << "summations\n";
	dpalphm = csmat(getbdpmat<Adr>(l,0).rows(), getbdpmat<Adr>(l,0).cols());

	for(int alpha = 0; alpha < 6; alpha++) {
		if(l%2 == alpha%2) {
			switch(ul) {
				case 1:
					dpalphm += ((double) vExt.template axialPart<axis::t>(vExt.getTime())(alpha,0).real() )
					* (getbdpmat<Adr>(l,alpha) + il(ll) * getbdpmat<AL>(l,alpha));
					break;
				case -1:
					dpalphm += ((double) vExt.template axialPart<axis::t>(vExt.getTime())(alpha,0).real() )
					* (getbdpmat<Adr>(l,alpha) - (il(ll) + 1) * getbdpmat<AL>(l,alpha));
					break;
				default:
					cout << "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAARGHGHGAEGAKJDHAJHWURHWOURAHFJLHEOIHYRAOWHRIØWAHROWQ" << std::endl;
			}
		}
	}

	// cout << "return statement\n";
	return dpalphm;
}

csmat& bsbasis::bdpAsqmat(int l, const Potential<beyondDipolePulse>& vExt) {
	this->vExt = &vExt;
	csmat& bdpAsqm  =  getmat<bdpAsq>();
	/*csmat& dpalph0  = getbdpmat<p00>(l,0);
	csmat& dpalph0k = getbdpmat<p0k>(l,0);
	csmat& dpalphk0 = getbdpmat<pk0>(l,0);
	csmat& dpalphkk = getbdpmat<pkk>(l,0);
	*/
	if(!isCached(getbdpmat<Asq>(l,0))) {
		lvec x = glpts();
		
		clsmat& wt = wtmat();
		
		//cout << "Size of x coord: " << x.size() << std::endl;
		//Explicitly put time dependence in bdpalphmat!
		
		
		//cout << "Address of vExt: " << &vExt << std::endl;
		
		bdpsqft Calph = this->vExt->template axialPartSq<axis::radial>(x,l);
		
		clsmat wtC[8];
		for(int alpha = 0; alpha < 8; alpha++) {
			wtC[alpha] = wt * Calph.row(alpha).asDiagonal();
		}
		
		
		clsmat P = splineCache(x);
		clsmat dP = dSplineCache(x,1);
		
		
		// cout << "products\n";
		for(int alpha = 0; alpha < 8; alpha++) {
			//cout << "\n" << P.rows() << ", " << P.cols()<<"\n" << wtC[alpha].rows() << ", " << wtC[alpha].cols() << "\n";
			
			
			getbdpmat<Asq>(l,alpha) = P.adjoint() * wtC[alpha] * P;
		}
	}
	
	// cout << "summations\n";
	bdpAsqm = csmat(getbdpmat<Asq>(l,0).rows(), getbdpmat<Asq>(l,0).cols());

	for(int alpha = 0; alpha < 8; alpha++) {
		bdpAsqm += ((double) this->vExt->template axialPartSq<axis::t>(this->vExt->getTime(), l)(alpha,0).real())
		* ( getbdpmat<Asq>(l,alpha) );
	}

	// cout << "return statement\n";
	return bdpAsqm;
}

csmat& bsbasis::H0mat() {
	csmat& H0_0 = getH0mat(0);
	csmat& H0_l = getH0mat(1);
	
	csmat& H0 = getmat<matname::H0>();
	
	H0 = H0_0 + mfl*(mfl+1)*H0_l;
	
	return H0;
}

int bsbasis::rqn() {
	return splineCache().cols();
}
/*
csmat& bsbasis::bdpalphmat(int l, const Potential<beyondDipolePulse>& vExt) {
	this->vExt = &vExt;
	csmat& dpalphm  =  getmat<bdpa>();
	/*csmat& dpalph0  = getbdpmat<p00>(l,0);
	csmat& dpalph0k = getbdpmat<p0k>(l,0);
	csmat& dpalphk0 = getbdpmat<pk0>(l,0);
	csmat& dpalphkk = getbdpmat<pkk>(l,0);
	
	if(!isCached(getbdpmat(l,0))) {
		lvec x = glpts();
		
		clsmat& wt = wtmat();
		
		//cout << "Size of x coord: " << x.size() << std::endl;
		//Explicitly put time dependence in bdpalphmat!
		bdpft Calph = vExt.template axialPart<axis::radial>(x,l);
		
		clsmat wtC[6];
		for(int alpha = 0; alpha < 6; alpha++) {
			wtC[alpha] = wt * Calph.row(alpha).asDiagonal();
		}
		
		
		clsmat& P = splineCache(x);
		clsmat& dP = dSplineCache(x,1);
		
		
		// cout << "products\n";
		for(int alpha = 0; alpha < 6; alpha++) {
			
			getbdpmat(l,alpha) = P.adjoint() * (wtC[alpha] * (dP + x.diagonal() * /*l, but the other l *  P));
		}
	}
	
	// cout << "summations\n";
	dpalphm = csmat(getbdpmat(l,0).rows(), getbdpmat(l,0).cols());

	for(int alpha = 0; alpha < 6; alpha++) {
		if(l%2==alpha%2) {
			dpalphm += ((double) vExt.template axialPart<axis::t>(vExt.getTime())(alpha,0).real() )
			* ( getbdpmat(l,alpha) );
		}
	}

	// cout << "return statement\n";
	return dpalphm;
}
*/
/*
PYTHON CODE TO PORT
 def pzmat(self,lmax = 0):
        pzm = sprs.lil_matrix(((lmax+1)*self.N,(lmax+1)*self.N),dtype=complex)
        for l in range(0,lmax+1):
            lpot1=nltc.coloumb(-(l+1))
            lpot2=nltc.coloumb(-l)
            dd = sprs.lil_matrix(self.ddmat(1),dtype = complex)
            E1 = sprs.lil_matrix(self.Emat(lpot1),dtype = complex)
            E2 = sprs.lil_matrix(self.Emat(lpot2),dtype = complex)
            
            
            ddd = sprs.dia_matrix(dd.shape,dtype = complex)
            E1d = sprs.dia_matrix(E1.shape,dtype = complex)
            E2d = sprs.dia_matrix(E2.shape,dtype = complex)
            
            for i in range(0,self.k+1): 
                ddd.setdiag(dd.diagonal(i),i)
                ddd.setdiag(dd.diagonal(-i),-i)
                
                E1d.setdiag(E1.diagonal(i),i)
                E1d.setdiag(E1.diagonal(-i),-i)
                
                E2d.setdiag(E2.diagonal(i),i)
                E2d.setdiag(E2.diagonal(-i),-i)
                
            # plt.imshow(ddd.toarray().real)
            # plt.imshow(E1d.toarray().real)
            
            if l-1 >= 0 and l+1 < self.N:
                pzm[l*self.N:(l+1)*self.N,(l-1)*self.N:l*self.N] = ddd + E1d
                pzm[(l-1)*self.N:l*self.N,l*self.N:(l+1)*self.N] = ddd - E2d
        return sprs.dia_matrix(pzm)
*/
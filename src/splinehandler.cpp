/*

*/
#include "splinehandler.h"
#include <boost/math/special_functions/legendre.hpp>

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

lvec& splineHandler::glpts(int ll) {
	if (isCached(glpch)) return glpch;
	else {
		
		if(ll == -1) l = splinedeg + 1;
		else l = l;
		
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
		// int l = this->splinedeg + 1;
		
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
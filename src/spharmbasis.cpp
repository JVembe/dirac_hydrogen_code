#include "basis.h"
#include <wignerSymbols.h>
#include <boost/math/special_functions/spherical_harmonic.hpp>
spharmbasis::spharmbasis(int l) {
	this->lmax = l;
	
	for(int i = 0; i < matname::N_MATS; i++) {
		if(i!=matname::dpa && i!= matname::bdpa && i!= matname::pz && i!=matname::Lsq) {
			mats[i] = cmat::Identity(angqN(),angqN()).sparseView();
		}
	}
}

int spharmbasis::aqn() {
	return lmax + 1;
}
/*
int l(int i) {
	int il = int(sqrt(i));
	return il;
}

int m(int i) {
	int il = l(i);
	int im = i - (il*il+1);
	
	return im;
}*/

int il(int i) {
	//int ii = i/2;
	
	return floor(sqrt(i));
}


iarray il(iarray i) {
	return floor(sqrt(i));
}

int li(int l) {
	return pow(l,2);
}

int im(int i) {
	int l = il(i);
	
	int ii = i - pow(il(i),2);
	

	return (ii % (il(i)*2 + 1)) - il(i);
}

cdouble alm(double l, double m) {
	return sqrt(cdouble(((l + m)*(l + m - 1))/(4 * (2 * l - 1)*(2 * l + 1)),0));
}

cdouble blm(double l,double m) { //They sure do
	return sqrt(cdouble(((l + m) * (l - m))/((2*l + 1) * (2* l - 1)),0));
}

csmat& spharmbasis::Lsqmat() {
	csmat& Lsqm = getmat<Lsq>();
	
	if (isCached(Lsqm)) return Lsqm;
	else {
		int N = angqN();
		
		Lsqm = csmat(N,N);
		
		for(int i = 0; i < N; i++) {
			Lsqm.insert(i,i) = i*(i+1);
		}
		return Lsqm;
	}
}

csmat& spharmbasis::pzmat() {
	csmat& pzm = getmat<pz>();
	if (isCached(pzm)) return pzm;
	else {
		int N = angqN();
		
		pzm = csmat(N,N);
		
		for(int i = 0; i < N; i++) {
			if(i - 1 >=0) pzm.insert(i,i-1) = -blm(i,0);
			if(i + 1 < N) pzm.insert(i,i+1) = -blm(i+1,0);
		}
		
		return pzm;
	}
}

csmat& spharmbasisLM::pzmat(int ul) {
	csmat& pzm = getmat<pz>();
	// if (isCached(pzm)) return pzm;
	// else {


	int N = angqN();
	
	cmat nspzm = cmat(N,N);
	
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			int li = il(i);
			int lj = il(j);
			int mi = im(i);
			int mj = im(j);
			if(li == lj - 1 && mi == mj && i!=j && (ul == 1)) nspzm(i,j) = cdouble(0.,-1.) * blm(lj , mj);
			if(li == lj + 1 && mi == mj && i!=j && (ul ==-1)) nspzm(i,j) = cdouble(0.,-1.) * blm(lj + 1 , mj);
		}
	}
	
	pzm = nspzm.sparseView();
	
	return pzm;
	// }
}

spharmbasisLM::spharmbasisLM(int l, int m) {
	this->lmax = l;
	if(m == 0) this->mmax = l;
	else this->mmax = m;
	
	for(int i = 0; i < matname::N_MATS; i++) {
		if(i!=matname::dpa && i!= matname::bdpa && i!= matname::pz && i!=matname::Lsq) {
			mats[i] = cmat::Identity(angqN(),angqN()).sparseView();
		}
	}
	
	this->bdppzmats[0] = std::vector<csmat>(bdplmax());
	this->bdppzmats[1] = std::vector<csmat>(bdplmax());
	this->bdpsqmats = std::vector<csmat>(bdplmax());
}

int spharmbasisLM::aqn() {
	return pow(lmax+1,2);
}
/*
int l(int i) {
	int il = int(sqrt(i));
	return il;
}

int m(int i) {
	int il = l(i);
	int im = i - (il*il+1);
	
	return im;
}*/


csmat& spharmbasisLM::Lsqmat() {
	csmat& Lsqm = getmat<Lsq>();
	
	if (isCached(Lsqm)) return Lsqm;
	else {
		int N = angqN();
		
		Lsqm = csmat(N,N);
		
		for(int i = 0; i < N; i++) {
			Lsqm.insert(i,i) = il(i)*(il(i)+1);
		}
		return Lsqm;
	}
}

csmat& spharmbasisLM::pzmat() {
	csmat& pzm = getmat<pz>();
	if (isCached(pzm)) return pzm;
	else {
	
	
		int N = angqN();
		
		cmat nspzm = cmat::Zero(N,N);
		
		for(int i = 0; i < N; i++) {
			for(int j = 0; j < N; j++) {
				int li = il(i);
				int lj = il(j);
				int mi = im(i);
				int mj = im(j);
				if(li == lj - 1 && mi == mj && i!=j) nspzm(i,j) = cdouble(0.,-1.) * blm(lj , mj);
				if(li == lj + 1 && mi == mj && i!=j) nspzm(i,j) = cdouble(0.,-1.) * blm(lj + 1 , mj);
			}
		}
		
		pzm = nspzm.sparseView();
		
		return pzm;
	}
}


using namespace WignerSymbols;
namespace bmath = boost::math;

csmat& spharmbasisLM::bdpApzmat(int l = 0, int ul = 0) {
	csmat& bdpApzm = getbdppzmat(l,ul);
	
	
	//cout << "bdpalphm dims: " << bdpalphm.rows() << ", "<<bdpalphm.cols() << std::endl;
	if (isCached(bdpApzm)) return bdpApzm;
	else {
	
		int N = angqN();
		
		bdpApzm = csmat(N,N);
		
		cmat apzm = cmat::Zero(N,N);
		for(int i = 0; i < N; i++) {
			for(int j = 0; j < N; j++) {
				int li = il(i);
				int lj = il(j);
				int mi = im(i);
				int mj = im(j);
				
				
				//cout << l << ", ";
				/*
				THIS IS THE OLD VERSION, IN WHICH THE PULSE IS ASSUMED TO PROPAGATE IN THE Z DIRECTION
				
				cdouble YYYU = pow(-1,mj) * sqrt(cdouble((2*l+1),0)) * sqrt(cdouble((2*li+1)*(2*l+1)*(2*lj+3),0)) * wigner3j(li,l,lj+1,0,0,0) * wigner3j(li,l,lj+1,mi,0,-mj);
				
				*/
				if((abs(mi) <= mmax) && (abs(mj) <= mmax )) {
					cdouble msum(0,0);
					
					for(int m = -l; m <= l; m++) {
						if((abs(bmath::spherical_harmonic(l,m,PI/2,0)) > 10e-15) && (abs(m) <= mmax))
						msum += bmath::spherical_harmonic(l,m,PI/2,0) * cdouble(pow(-1,m) * wigner3j(li,l,lj+1,-mi,-m,mj),0);
					}
					
					cdouble YYYU = pow(-1,mj) * sqrt(cdouble(4*PI*(2*li+1)*(2*l+1)*(2*lj+3),0)) * wigner3j(li,l,lj+1,0,0,0) * msum;
					
					//cdouble YYYL = pow(-1,mj) * sqrt(cdouble((2*l+1)/(4*PI),0)) * sqrt(cdouble((2*li+1)*(2*l+1)*(2*lj-1)/(4*PI),0)) * wigner3j(li,l,lj-1,0,0,0) * wigner3j(li,l,lj-1,mi,0,-mj);
					
					
					if(YYYU!=cdouble(0,0) /*|| YYYL!=cdouble(0,0)*/) {
						//bdpApzm.insert(i,j) = cdouble(0,0);
						//bdpApzm.insert(j,i) = cdouble(0,0);
						
						switch(ul) {
							case 1:
								if(ul != -1) apzm(i,j) += cdouble(0.,1.) * -blm(lj + 1, mj) * YYYU;
							break;
							case -1:
								if(ul !=  1) apzm(j,i) += cdouble(0.,1.) * -blm(lj + 1, mj) * YYYU;
							break;
						}
						
						//if(YYYL!=cdouble(0,0) && ul != 1 ) bdpApzm.coeffRef(i,j) += cdouble(0,1) * -blm(lj, mj)    * YYYL;
					}
					
					//bdpApzm.insert(i,j) = cdouble(mi,mj);
				}
			}
			//cout << std::endl;
		}
		bdpApzm = apzm.sparseView();
		
		return bdpApzm;
	}
}

csmat& spharmbasisLM::bdpApxmat(int l = 0, int ul = 0) {
	csmat& bdpApxm = getbdppzmat(l,ul);
	
	
	//cout << "bdpalphm dims: " << bdpalphm.rows() << ", "<<bdpalphm.cols() << std::endl;
	if (isCached(bdpApxm)) return bdpApxm;
	else {
	
		int N = angqN();
		
		bdpApxm = csmat(N,N);
		
		cmat apxm = cmat::Zero(N,N);
		for(int i = 0; i < N; i++) {
			for(int j = 0; j < N; j++) {
				int li = il(i);
				int lj = il(j);
				int mi = im(i);
				int mj = im(j);
				
				
				//cout << l << ", ";
				//THIS IS THE OLD VERSION, IN WHICH THE PULSE IS ASSUMED TO PROPAGATE IN THE Z DIRECTION
				if((abs(mi) <= mmax) && (abs(mj) <= mmax )) {
					cdouble YYY1 = pow(-1,mi) * sqrt(cdouble((2*l+1),0)) * sqrt(cdouble((2*li+1)*(2*l+1)*(2*lj+3),0)) * wigner3j(li,l,lj+1,0,0,0) * wigner3j(li,l,lj+1,-mi,0,mj+1);
					cdouble YYY2 = pow(-1,mi) * sqrt(cdouble((2*l+1),0)) * sqrt(cdouble((2*li+1)*(2*l+1)*(2*lj+3),0)) * wigner3j(li,l,lj+1,0,0,0) * wigner3j(li,l,lj+1,-mi,0,mj-1);
					//cdouble YYY3 = pow(-1,mj) * sqrt(cdouble((2*l+1),0)) * sqrt(cdouble((2*li+1)*(2*l+1)*(2*lj-1),0)) * wigner3j(li,l,lj-1,0,0,0) * wigner3j(li,l,lj-1,-mi,0,mj+1);
					//cdouble YYY4 = pow(-1,mj) * sqrt(cdouble((2*l+1),0)) * sqrt(cdouble((2*li+1)*(2*l+1)*(2*lj-1),0)) * wigner3j(li,l,lj-1,0,0,0) * wigner3j(li,l,lj-1,-mi,0,mj-1);
					
					switch(ul) {
						case 1:
							apxm(i,j)  	= cdouble(0.,1.) * alm(lj + 1, mj + 1) * YYY1
										- cdouble(0.,1.) * alm(lj + 1,-mj + 1) * YYY2;
						break;
						case -1:
							apxm(j,i)  	= cdouble(0.,1.) * alm(lj + 1, mj + 1) * YYY1
										- cdouble(0.,1.) * alm(lj + 1,-mj + 1) * YYY2;
						break;
					}
					//bdpApzm.insert(i,j) = cdouble(mi,mj);
				}
				
			}
			//cout << std::endl;
		}
		bdpApxm = apxm.sparseView();
		
		return bdpApxm;
	}
}

csmat& spharmbasisLM::bdpAsqmat(int l = 0) {
	csmat& bdpAsqm = getbdpsqmat(l);
	
	
	//cout << "bdpalphm dims: " << bdpalphm.rows() << ", "<<bdpalphm.cols() << std::endl;
	if (isCached(bdpAsqm)) return bdpAsqm;
	else {
	
		int N = angqN();
		
		bdpAsqm = csmat(N,N);
		
		for(int i = 0; i < N; i++) {
			for(int j = 0; j < N; j++) {
				int li = il(i);
				int lj = il(j);
				int mi = im(i);
				int mj = im(j);
				
				if((abs(mi) <= mmax) && (abs(mj) <= mmax )) {
					cdouble msum(0,0);
					
					//TODO: Make the propagation direction an option
					for(int m = -l; m <= l; m++) {
						if((abs(bmath::spherical_harmonic(l,m,0,0)) > 10e-15) && (abs(m) <= mmax))
						msum += bmath::spherical_harmonic(l,m,0,0) * cdouble(pow(-1,m) * wigner3j(li,l,lj,-mi,-m,mj),0);
					}
					//cout << l << ", ";
					
					cdouble YYY = pow(-1,mj) * sqrt(cdouble(4*PI*(2*li+1)*(2*l+1)*(2*lj+1),0))*wigner3j(li,l,lj,0,0,0)*msum;
					
					if(YYY!=cdouble(0,0) && YYY==YYY) {
						bdpAsqm.insert(i,j) = YYY;
					}
				}
				
				//bdpApzm.insert(i,j) = cdouble(mi,mj);
			}
			//cout << std::endl;
		}
		
		
		return bdpAsqm;
	}
}
/*
csmat& spharmbasisLM::bdpAsqmat(int l = 0) {
	csmat& bdpAsqm = getbdpsqmat(l);
	
	
	//cout << "bdpalphm dims: " << bdpalphm.rows() << ", "<<bdpalphm.cols() << std::endl;
	if (isCached(bdpAsqm)) return bdpAsqm;
	else {
	
		int N = angqN();
		
		bdpAsqm = csmat(N,N);
		
		for(int i = 0; i < N; i++) {
			for(int j = 0; j < N; j++) {
				int li = il(i);
				int lj = il(j);
				int mi = im(i);
				int mj = im(j);
				
				if((abs(mi) <= mmax) && (abs(mj) <= mmax )) {
					cdouble msum(0,0);
					
					//TODO: Make the propagation direction an option
					for(int m = -l; m <= l; m++) {
						if((abs(bmath::spherical_harmonic(l,m,0,0)) > 10e-15) && (abs(m) <= mmax))
						msum += bmath::spherical_harmonic(l,m,0,0) * cdouble(pow(-1,m) * wigner3j(li,l,lj,-mi,-m,mj),0);
					}
					//cout << l << ", ";
					
					cdouble YYY = pow(-1,mj) * sqrt(cdouble(4*PI*(2*li+1)*(2*l+1)*(2*lj+1),0))*wigner3j(li,l,lj,0,0,0)*msum;
					
					if(YYY!=cdouble(0,0) && YYY==YYY) {
						bdpAsqm.insert(i,j) = YYY;
					}
				}
				
				//bdpApzm.insert(i,j) = cdouble(mi,mj);
			}
			//cout << std::endl;
		}
		
		
		return bdpAsqm;
	}
}
*/
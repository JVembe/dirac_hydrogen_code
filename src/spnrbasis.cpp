#include "basis.h"
#include <wignerSymbols.h>
/*
def ikmu(i):
    #Block kappa pattern of -1, 1, -2, 2, -3, 3
    #Mu pattern of -|k| + 1/2,...., |k| - 1/2
    ii = i//4
    
    abskappa = int(0.5*(npy.sqrt(8*ii + 1) - 1)) + 1
    
    kmod = max(2,(abskappa)*2)
    
    sgnmod = max(4,(abskappa)*4)
    sgnfloor = 2 * abskappa * (abskappa-1)
    sgnkappa = 2*int((i-sgnfloor)/sgnmod >=0.5) - 1
    
    
    mu = i%kmod - abskappa + 1/2
    return abskappa*sgnkappa,mu
*/

template <typename T> int getsgn(T val) {
    return (T(0) < val) - (val < T(0));
}



spnrbasis::spnrbasis(int kmax, int mumax) {
	this->kmax = kmax;
	this->mumax = mumax;
	
	
	int Nth = aqn();
	//Initialise matrices to identity
	std::vector<T> triplets;
	triplets.reserve(Nth);
	
	for(int i = 0; i < Nth; i++) {
		triplets.push_back(T(i,i,1.0));
	}
	
	for(int i = 0; i < matname::N_MATS; i++) {
		if(i!=matname::dpa && i!= matname::bdpa) {
			mats[i] = csmat(Nth,Nth);
			mats[i].setFromTriplets(triplets.begin(),triplets.end());
		}
	}
	
	//reserve cache matrices for l up to lmax
	this->bdplmats = std::vector<csmat>(bdplmax()*2);
	for(int i = 0; i < bdplmax(); i++) {
		bdplmats[i] = csmat(0,0);
	}
	
	//Setup index-kappa array and index-mu array
	kapparr = vector<int>(Nth);
	muarr = vector<int>(Nth);
	
	for(int i = 0; i < Nth; i++) {
		kapparr[i] = ik(i);
		muarr[i] = imu(i);
	}
}

int spnrbasis::aqn() {
	if (kmax == 1)
            return 4;
        else
            //return 2*(kmax+1)*(kmax);
            return 2*(kmax+1)*(kmax);
}


using namespace WignerSymbols;

csmat& spnrbasis::componentOlmat(int comp) {
	
	
	cmat outmat = cmat::Zero(angqN(),angqN());
	
	csmat& olc = getmat<compS>();
	
	olc = csmat(angqN(),angqN());
	
	for(int i = 0; i < angqN(); i++) {
		for(int j = 0; j < angqN(); j++) {
			int ki = (int)ik(i);
			int kj = (int)ik(j);
			
			double ji = abs(ki) - 0.5;
			double jj = abs(kj) - 0.5;
			
			double mui = imu(i);
			double muj = imu(j);
			
			int lpi, lqi, lpj, lqj;
			
			lpi = abs( ki + 0.5) - 0.5;
			lpj = abs( kj + 0.5) - 0.5;
			lqi = abs(-ki + 0.5) - 0.5;
			lqj = abs(-kj + 0.5) - 0.5;
			
			// if(i==j) outmat(i,j) = 1;
			
			if(((lpi == lpj) || (lqi == lqj)) && (mui == muj)) {
				switch(comp) {
					case 1:
						outmat(i,j) = clebschGordan(lpi,0.5,ji,mui-0.5,0.5,mui) * clebschGordan(lpj,0.5,jj,muj-0.5,0.5,muj) * (lpi == lpj);
						continue;
					case 0:
						outmat(i,j) = clebschGordan(lpi,0.5,ji,mui+0.5,-0.5,mui) * clebschGordan(lpj,0.5,jj,muj+0.5,-0.5,muj) * (lpi == lpj);
						continue;
					case 3:
						outmat(i,j) = clebschGordan(lqi,0.5,ji,mui-0.5,0.5,mui) * clebschGordan(lqj,0.5,jj,muj-0.5,0.5,muj) * (lqi == lqj);
						continue;
					case 2:
						outmat(i,j) = clebschGordan(lqi,0.5,ji,mui+0.5,-0.5,mui) * clebschGordan(lqj,0.5,jj,muj+0.5,-0.5,muj) * (lqi == lqj);
						continue;
				}
			}
		}
	}
	
	//cout << outmat << std::endl;
	olc = outmat.sparseView();
	return olc;
}

csmat& spnrbasis::dpalphmat(int ul = 1) {
	
	csmat* alphp;
	
	switch(ul) {
		case -1:
			alphp = &dpamats[0];
			break;
		case 1:
			alphp = &dpamats[1];
			break;
		default:
			alphp = &getmat<dpa>();
			break;
	}
	
	csmat& dpalphm = *alphp;
	
	if(!isCached(dpalphm)) {
		// cout << "Constructing dpalphmat " << ul;
		dpalphm = csmat(angqN(),angqN());
		for(int i = 0; i < dpalphm.rows(); i++) {
			for(int j = 0; j < dpalphm.cols(); j++) {
				
				int ki = (int)ik(i);
				int kj = (int)ik(j);
				
				double ji = abs(ki) - 0.5;
				double jj = abs(kj) - 0.5;
				
				double mui = imu(i);
				double muj = imu(j);
				
				double lp, lq;
				
				if(ki > 0) {
					lp = ki;
					lq = ki - 1;
				}
				else {
					lp = -ki - 1;
					lq = -ki;
				}
				
				if((ki == -kj || kj == ki + 1 || ki - 1 == kj)  && (mui == muj)) {
					dpalphm.insert(i,j) = cdouble(0,0);
					if((ki == -kj || kj == ki + 1) && (ul == 1 || ul == 0)) {
						dpalphm.coeffRef(i,j) += cdouble(0, 
													clebschGordan(lp,0.5,ji,mui-0.5,0.5,mui) * clebschGordan(lp,0.5,jj,mui-0.5,0.5,mui) 
													- clebschGordan(lp,0.5,ji,mui+0.5,-0.5,mui) * clebschGordan(lp,0.5,jj,mui+0.5,-0.5,mui)
													);
					}
					
					if((ki == -kj || ki - 1 == kj) && (ul == -1 || ul == 0)) {
						dpalphm.coeffRef(i,j) -= cdouble(0,
													  clebschGordan(lq,0.5,ji,mui-0.5,0.5,mui) * clebschGordan(lq,0.5,jj,mui-0.5,0.5,mui) 
													- clebschGordan(lq,0.5,ji,mui+0.5,-0.5,mui) * clebschGordan(lq,0.5,jj,mui+0.5,-0.5,mui)
													);
					}
					
				}
				
				/*
				if(mumax == 0 || (abs(mui) < mumax && abs(muj) < mumax)) {
					switch(ul) {
						case 1:
							if(abs(2*ki + 1) == abs(2*kj - 1) && (mui == muj + 1 || mui == muj - 1)) {
								cdouble Gij(0,0);
								
								if(mui == muj + 1) {
									Gij += cdouble(sgn(-ki) * sqrt((ki + 0.5 - mui)*(kj - 0.5 - muj)/((2*ki + 1)*(2*kj - 1))),0);
								}
								if(mui == muj - 1) {
									Gij += cdouble(sgn(kj) * sqrt((ki + 0.5 + mui)*(kj - 0.5 + muj)/((2*ki + 1)*(2*kj - 1))),0);
								}
								
								dpalphm.insert(i,j) = cdouble(0,1) * Gij;
							}
						
							break;
						case -1:
							if(abs(2*ki - 1) == abs(2*kj + 1) && (mui == muj + 1 || mui == muj - 1)) {
								cdouble Gij(0,0);
								
								if(mui == muj + 1) {
									Gij += cdouble(sgn(ki) * sqrt((ki - 0.5 + mui)*(kj + 0.5 + muj)/((2*ki - 1)*(2*kj + 1))),0);
								}
							if(mui == muj - 1) {
								Gij += cdouble(sgn(-kj) * sqrt((ki - 0.5 - mui)*(kj + 0.5 - muj)/((2*ki - 1)*(2*kj + 1))),0);
							}
							
							dpalphm.insert(i,j) = cdouble(0,-1) * Gij;
						}
						break;
				
					default:
						csmat out(angqN(),angqN());
						out += dpalphmat(1);
						out += dpalphmat(-1);
						dpalphm = out;
						break;
					}
				}
				*/
			}
		}
		
		dpalphm.prune(cdouble(0,0));
		
	}
	return dpalphm;
}

csmat& spnrbasis::bdpalphmat(int ul = 1, int l = 1) {
	return bdpalphsigmaXmat(ul,l);
	
	csmat& bdpalphm = getbdpmat(ul,l);
	
	
	//cout << "bdpalphm dims: " << bdpalphm.rows() << ", "<<bdpalphm.cols() << std::endl;
	if(isCached(bdpalphm) && (l < this->bdplmax())) return bdpalphm;
	else {
		//cout << ul << ", " << l << std::endl;
		bdpalphm = csmat(angqN(),angqN());
		for(int i = 0; i < bdpalphm.rows(); i++) {
			for(int j = 0; j < bdpalphm.cols(); j++) {
				//int mui = imu(i);
				//int muj = imu(j);
				
				int ki = (int)ik(i);
				int kj = (int)ik(j);
				
				double ji = abs(ki) - 0.5;
				double jj = abs(kj) - 0.5;
				
				double mui = imu(i);
				double muj = imu(j);
				
				if(((abs(mui)>mumax) || abs(muj) > mumax) && (mumax!=0)) continue;
				
				double lpi, lqi, lpj, lqj;
				
				if(ki > 0) {
					lpi = ki;
					lqi = ki - 1;
				}
				else {
					lpi = -ki - 1;
					lqi = -ki;
				}
				
				if(kj > 0) {
					lpj = kj;
					lqj = kj - 1;
				}
				else {
					lpj = -kj - 1;
					lqj = -kj;
				}
				
				cdouble musum1(0,0);
				cdouble musum2(0,0);
				
				cdouble YYY1;
				cdouble YYY2;
				
				
				switch(ul) {
					case 1:
						
						for(int m = -l; m <= l; m++) {
							if((abs(bmath::spherical_harmonic(l,m,PI/2,0)) > 10e-15) && ((abs(m) <= mumax) || (mumax == 0))) {
								cdouble musum = cdouble(pow(-1,int(m + mui - 0.5)) * wigner3j(lpi,l,lqj,-mui+0.5,-m,muj-0.5),0);
								
								if(true){//(musum == musum) {
									musum1 += bmath::spherical_harmonic(l,m,PI/2,0) * musum;
								}
							}
						}
						
						for(int m = -l; m <= l; m++) {
							if((abs(bmath::spherical_harmonic(l,m,PI/2,0)) > 10e-15) && ((abs(m) <= mumax) || (mumax == 0))){
								cdouble musum = cdouble(pow(-1,int(m + mui + 0.5)) * wigner3j(lpi,l,lqj,-mui-0.5,-m,muj+0.5),0);
							
								if(true){//(musum == musum) {
									musum2 += bmath::spherical_harmonic(l,m,PI/2,0) * musum;
								}
							}
						}
						
						YYY1 = sqrt(cdouble((2*lpi+1)*(2*l+1)*(2*lqj+1)*(4*PI),0)) * wigner3j(lpi,l,lqj,0,0,0) * musum1;
						YYY2 = sqrt(cdouble((2*lpi+1)*(2*l+1)*(2*lqj+1)*(4*PI),0)) * wigner3j(lpi,l,lqj,0,0,0) * musum2;
						
						
						if(((YYY1 != cdouble(0,0)) && (YYY1 == YYY1)) || ((YYY2 != cdouble(0,0)) && (YYY2 == YYY2))) {
							bdpalphm.insert(i,j) =  cdouble(0,1) * 
													( 
													 YYY1 * clebschGordan(lpi,0.5,ji,mui-0.5, 0.5,mui) * clebschGordan(lqj,0.5,jj,muj-0.5, 0.5,muj) 
													-YYY2 * clebschGordan(lpi,0.5,ji,mui+0.5,-0.5,mui) * clebschGordan(lqj,0.5,jj,muj+0.5,-0.5,muj)
													);
						}
						break;
						
					case -1:
						
						for(int m = -l; m <= l; m++) {
							if((abs(bmath::spherical_harmonic(l,m,PI/2,0)) > 10e-15) && ((abs(m) <= mumax) || (mumax == 0))) {
								cdouble musum = cdouble(pow(-1,int(m + mui - 0.5)) * wigner3j(lqi,l,lpj,-mui+0.5,-m,muj-0.5),0);
							
								if(true){//(musum == musum) 
									musum1 += bmath::spherical_harmonic(l,m,PI/2,0) * musum;
								}
							}
						}
						
						for(int m = -l; m <= l; m++) {
							if((abs(bmath::spherical_harmonic(l,m,PI/2,0)) > 10e-15) && ((abs(m) <= mumax) || (mumax == 0))) {
								cdouble musum = cdouble(pow(-1,int(m + mui + 0.5)) * wigner3j(lqi,l,lpj,-mui-0.5,-m,muj+0.5),0);
								if(true){//(musum == musum) 
									musum2 += bmath::spherical_harmonic(l,m,PI/2,0) * musum;
								}
							}
						}
						
						YYY1 = sqrt(cdouble((2*lqi+1)*(2*l+1)*(2*lpj+1)*(4*PI),0)) * wigner3j(lqi,l,lpj,0,0,0) * musum1;
						YYY2 = sqrt(cdouble((2*lqi+1)*(2*l+1)*(2*lpj+1)*(4*PI),0)) * wigner3j(lqi,l,lpj,0,0,0) * musum2;
						
						
						if((((YYY1 != cdouble(0,0)) && (YYY1 == YYY1)) || ((YYY2 != cdouble(0,0)) && (YYY2 == YYY2)))) {
							bdpalphm.insert(i,j) =  cdouble(0,1) * 
													( 
													-YYY1 * clebschGordan(lqi,0.5,ji,mui-0.5, 0.5,mui) * clebschGordan(lpj,0.5,jj,muj-0.5, 0.5,muj) 
													+YYY2 * clebschGordan(lqi,0.5,ji,mui+0.5,-0.5,mui) * clebschGordan(lpj,0.5,jj,muj+0.5,-0.5,muj)
													);
						}
						break;
				}
				
			}
		}
		return bdpalphm;
	}
}

csmat& spnrbasis::bdpalphsigmaXmat(int ul, int l) {
		csmat& bdpalphm = getbdpmat(ul,l);
	
	
	//cout << "bdpalphm dims: " << bdpalphm.rows() << ", "<<bdpalphm.cols() << std::endl;
	if(isCached(bdpalphm) && (l < this->bdplmax())) return bdpalphm;
	else {
		//cout << ul << ", " << l << std::endl;
		bdpalphm = csmat(angqN(),angqN());
		for(int i = 0; i < bdpalphm.rows(); i++) {
			for(int j = 0; j < bdpalphm.cols(); j++) {
				//int mui = imu(i);
				//int muj = imu(j);
				
				// int ki = (int)ik(i);
				// int kj = (int)ik(j);
				int ki = kapparr[i];
				int kj = kapparr[j];
				
				
				double ji = abs(ki) - 0.5;
				double jj = abs(kj) - 0.5;
				
				// double mui = imu(i);
				// double muj = imu(j);
				
				int mui = muarr[i];
				int muj = muarr[j];
				
				if(((abs(mui)>mumax) || abs(muj) > mumax) && (mumax!=0)) continue;
				
				double lpi, lqi, lpj, lqj;
				
				if(ki > 0) {
					lpi = ki;
					lqi = ki - 1;
				}
				else {
					lpi = -ki - 1;
					lqi = -ki;
				}
				
				if(kj > 0) {
					lpj = kj;
					lqj = kj - 1;
				}
				else {
					lpj = -kj - 1;
					lqj = -kj;
				}
				if(mumax == 0 || (abs(mui) < mumax && abs(muj) < mumax)) {
					switch(l%2) {
						case 1:
							switch(ul) {
								case 1:
									if (bdpK(i,j,l)!= cdouble(0,0) || bdpL(i,j,l) != cdouble(0,0)) {
										bdpalphm.insert(i,j) = cdouble(0,1) * (bdpK(i,j,l) + bdpL(i,j,l));
									}
									break;
								case -1:
									if(bdpM(i,j,l)!= cdouble(0,0) || bdpN(i,j,l) != cdouble(0,0)) {
										bdpalphm.insert(i,j) = cdouble(0,1) * (bdpM(i,j,l) + bdpN(i,j,l));
									}
									break;
							}
							
							break;
						
						case 0:
							switch(ul) {
								case 1:
									if (bdpK(i,j,l)!= cdouble(0,0) || bdpL(i,j,l) != cdouble(0,0)) {
										bdpalphm.insert(i,j) = cdouble(-1,0) * (bdpK(i,j,l) + bdpL(i,j,l));
									}
									break;
								case -1:
									if(bdpM(i,j,l)!= cdouble(0,0) || bdpN(i,j,l) != cdouble(0,0)) {
										bdpalphm.insert(i,j) = cdouble(-1,0) * (bdpM(i,j,l) + bdpN(i,j,l));
									}
									break;
							}
							
							break;
					}
				}
			}
		}
		// bdpalphm.setIdentity();
		return bdpalphm;
	}
}

using cint = std::complex<int>;
cint ipow(int n) {
	

	switch(n%4) {
		case -3:
			return cint(0,1);
		case -2:
			return cint(-1,0);
		case -1:
			return cint(0,-1);
		case 0:
			return cint(1,0);
		case 1:
			return cint(0,1);
		case 2:
			return cint(-1,0);
		case 3:
			return cint(0,-1);
		default:
			return 0;
	}
}

cdouble spnrbasis::bdpK(int i, int j, int l) {
	// double ki = ik(i);
	// double kj = ik(j);
	
	// double mui = imu(i);
	// double muj = imu(j);
	
	double ki = kapparr[i];
	double kj = kapparr[j]; 
	
	double mui = muarr[i];
	double muj = muarr[j];
	
	//return cdouble(mui,muj);
	
	if(((int)(2*muj) == (int)(2*mui)- 2) && (abs(abs(ki + 0.5) - abs(kj - 0.5)) <= l) && (l <= (abs(ki + 0.5) + abs(kj - 0.5) - 1))) {
		//return cdouble(1,0);
		return cast<double,int>((getsgn(ki) * ipow(2*mui + l)))
			   * (2.0*l + 1.0) * sqrt(abs(ki + 0.5 - mui) * abs(kj - 0.5 - muj))
			   * wigner3j(abs(ki + 0.5) - 0.5, abs(kj - 0.5) - 0.5, l, 0, 0, 0)
			   * wigner3j(abs(ki + 0.5) - 0.5, abs(kj - 0.5) - 0.5, l, - mui + 0.5, muj + 0.5, 0);
	}
	else return cdouble(0,0);
}

cdouble spnrbasis::bdpL(int i, int j, int l) {
	// double ki = ik(i);
	// double kj = ik(j);
	
	// double mui = imu(i);
	// double muj = imu(j);
	
	double ki = kapparr[i];
	double kj = kapparr[j]; 
	
	double mui = muarr[i];
	double muj = muarr[j];
	
	//return cdouble(mui,muj);
	
	if(((int)(2*muj) == (int)(2*mui) + 2 ) && (abs(abs(ki + 0.5) - abs(kj - 0.5)) <= l) && (l <= (abs(ki + 0.5) + abs(kj - 0.5) - 1))) {
		//return cdouble(0,1);
		return cast<double,int>((getsgn(kj) * ipow(2*mui + l)))
			   * (2.0*l + 1.0) * sqrt(abs(ki + 0.5 + mui) * abs(kj - 0.5 + muj))
			   * wigner3j(abs(ki + 0.5) - 0.5, abs(kj - 0.5) - 0.5, l, 0, 0, 0)
			   * wigner3j(abs(ki + 0.5) - 0.5, abs(kj - 0.5) - 0.5, l, - mui - 0.5, muj - 0.5, 0);
	}
	else return cdouble(0,0);
}

cdouble spnrbasis::bdpM(int i, int j, int l) {
	// double ki = ik(i);
	// double kj = ik(j);
	
	// double mui = imu(i);
	// double muj = imu(j);
	
	double ki = kapparr[i];
	double kj = kapparr[j]; 
	
	double mui = muarr[i];
	double muj = muarr[j];
	//return cdouble(mui,muj);
	
	if(((int)(2*muj) == (int)(2*mui) - 2) && (abs(abs(ki - 0.5) - abs(kj + 0.5)) <= l) && (l <= (abs(ki - 0.5) + abs(kj + 0.5) - 1))) {
		//return cdouble(1,0);
		return cast<double,int>((getsgn(ki) * ipow(2*mui + l)))
			   * (2.0*l + 1.0) * sqrt(abs(ki - 0.5 + mui) * abs(kj + 0.5 + muj))
			   * wigner3j(abs(ki - 0.5) - 0.5, abs(kj + 0.5) - 0.5, l, 0, 0, 0)
			   * wigner3j(abs(ki - 0.5) - 0.5, abs(kj + 0.5) - 0.5, l, - mui + 0.5, muj + 0.5, 0);
	}
	else return cdouble(0,0);
}

cdouble spnrbasis::bdpN(int i, int j, int l) {
	// double ki = ik(i);
	// double kj = ik(j);
	
	// double mui = imu(i);
	// double muj = imu(j);
	
	double ki = kapparr[i];
	double kj = kapparr[j]; 
	
	double mui = muarr[i];
	double muj = muarr[j];
	
	//return cdouble(mui,muj);
	
	if(((int)(2*muj) == (int)(2*mui) + 2 ) && (abs(abs(ki - 0.5) - abs(kj + 0.5)) <= l) && (l <= (abs(ki - 0.5) + abs(kj + 0.5) - 1))) {
		//return cdouble(0,1);
		return cast<double,int>((getsgn(kj) * ipow(2*mui + l)))
			   * (2.0*l + 1.0) * sqrt(abs(ki - 0.5 - mui) * abs(kj + 0.5 - muj))
			   * wigner3j(abs(ki - 0.5) - 0.5, abs(kj + 0.5) - 0.5, l, 0, 0, 0)
			   * wigner3j(abs(ki - 0.5) - 0.5, abs(kj + 0.5) - 0.5, l, - mui - 0.5, muj - 0.5, 0);
	}
	else return cdouble(0,0);
}

int ik(int i) {
	int ii = i/4;
	int abskappa = int(0.5*(sqrt(8.0*ii+1.0) - 1.0)) + 1;
	
	int sgnmod = max(4,abskappa*4);
	double sgnfloor = 2*abskappa * (abskappa - 1);
	//cout << " sgnfloor = " << sgnfloor; 
	int sgnkappa = ((i-sgnfloor)/sgnmod >= 0.5) - ((i-sgnfloor)/sgnmod < 0.5);
	
	return abskappa * sgnkappa;
}

iarray ik(iarray i) {
	iarray abskappa = (sqrt(8 * (i/4) + 1) - 1)/2 + 1;
	
	iarray sgnmod = (abskappa*4).max(4);
	iarray sgnfloor = 2*abskappa * (abskappa - 1);
	
	iarray sgnkappa = (2*(i - sgnfloor)/sgnmod >= 1).cast<int>() - (2*(i-sgnfloor)/sgnmod < 1).cast<int>();
	
	return abskappa * sgnkappa;
}

int ki(int kappa) {
	int sgnkappa = sgn(kappa);
	
	int abskappa = abs(kappa);
	
	int i0 = (pow(2*(abskappa-1)+1,2)-1)/2;
	return i0 + (sgnkappa+1) * max(1,abskappa);
}

double imu(int i) {
	int abskappa = abs(ik(i));
	
	int kmod = max(2,2*abskappa);
	
	
	
	double mu = i%kmod - abskappa + 0.5;
	
	return mu;
}

using namespace std;

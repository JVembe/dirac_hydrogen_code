/*
	This implements parts of the dual kinetic balance basis, see dkbbasis.h for the rest
*/

#include "defs.h"
#include "dkbbasis.h"
#include "potential.h"


int dkbbasis::rqn() {
	return this->Nsplines;
}

csmat dkbglintmat(const clsmat& a0,const clsmat&  a1,const clsmat& b0,const clsmat& b1,const clsmat& ws,int k, int ul, int sumcoef) {
	int N = a0.cols();
	
	clsmat M(N,N);
	
	switch(ul) {
		case 1:
		M = a0.adjoint() * ws * b0;
		break;
		case -1:
		M = a1.adjoint() * ws * b1;
		break;
		default:
		M = a0.adjoint() * ws * b0 
		+ sumcoef * a1.adjoint() * ws * b1;
		break;
		
	}
	
	return M.cast<cdouble>();
}

csmat dkbglintmat(const clsmat& a0, const clsmat& b0, const clsmat& ws, int k) {
	int N = a0.cols();
	
	clsmat M(N,N);
	
	M = a0.adjoint() * ws * b0;
		/*+ (a0->transpose() * (*ws) * (*b1))
		+ (a1->transpose() * (*ws) * (*b0))*/
	
	return M.cast<cdouble>();
}

csmat dkbcrossglintmat(const clsmat& aP,const clsmat& aQ,const clsmat& bP,const clsmat& bQ,const clsmat& wts,int k, int ul, int sumcoef) {
	
	switch(ul) {
		case 1:
			return dkbglintmat(aQ,bP,wts,k);
		case -1:
			return dkbglintmat(aP,bQ,wts,k);
		default:
			return dkbglintmat(aQ,bP,wts,k) + sumcoef * dkbglintmat(aP,bQ,wts,k);
	}
}
//Horrifying, I know.
csmat factorisedglintmat(csmat& M0, csmat& Mk, csmat& Mkk,
						const clsmat& aP0,const clsmat& aQ0,const clsmat& bP0,const clsmat& bQ0,
						const clsmat& aPk,const clsmat& aQk,const clsmat& bPk,const clsmat& bQk,
						int k, const clsmat& wts, int kappa, csmat(*func)(const clsmat& aP,const clsmat& aQ,const clsmat& bP,const clsmat& bQ,const clsmat& wts,int k, int ul, int sumcoef), int ul, int sumcoef) {
						
	if(!isCached(M0)) M0 = func(aP0, aQ0, bP0, bQ0, wts, k, ul, sumcoef);
	if(!isCached(Mk)) Mk = func(aP0, aQ0, bPk, bQk, wts, k, ul, sumcoef) + func(aPk, aQk, bP0, bQ0, wts, k, ul, sumcoef);
	if(!isCached(Mkk)) Mkk = func(aPk, aQk, bPk, bQk, wts, k, ul, sumcoef);
	
	return M0 + kappa * Mk + pow(kappa,2) * Mkk;
}

//For use when all are already cached

csmat cachedglintmat(const csmat& M0, const csmat& Mk, const csmat& Mkk, int kappa) {
	return M0 + kappa * Mk + pow(kappa,2) * Mkk;
}


dkbbasis::dkbbasis(vec &t, int splinedeg): splineHandler(t,splinedeg) {
	dropfirst = 1; //More than 1 makes for very inaccurate energy levels
	droplast = 1; //Less than 2 causes boundary issues with derivatives
	//dkbCache(glpts(),1);
	//cout << radqN() << "\n";
}

/*
	NOTE FOR FUTURE SELF ABOUT THE DEFINITION OF DUAL KINETIC BALANCE
	
	
	TYSSE, 2018 SAYS IT'S d/dr +- k/r 
	
	HOWEVER, ORIGINAL DUAL KINETIC BALANCE PAPER SAYS k/r +- d/dr INSTEAD
	
	WE DEFER TO THE ORIGINAL PAPER IN THIS CASE
	
	ALSO, IT IS THE UPPER COMPONENT OF DDMAT THAT IS NEGATIVE
	
*/



clsmat dkbbasis::dkbCache(lvec& x,int ul) {
	//cout  << "pointer used in dkbCache " << &x << std::endl;
	
	clsmat splines = splineCache(x);
	clsmat dsplines = dSplineCache(x,1);
	
	//cout << "splines dimensions: " << splines.rows() << "," << splines.cols() <<"\n";
	//cout << "dsplines dimensions: " << dsplines.rows() << "," << dsplines.cols() <<"\n";
	
	int Nx = splines.rows();
	int Ns = splines.cols();
	
	
	if(!isCached(dkbUk) || !isCached(dkbLk) ) {
		
	}
	
	//cout << "KappaCache dimensions: " << kappaCache.rows() << "," << kappaCache.cols() <<"\n";
	//cout << "dsplines dimensions: " << dsplines.rows() << "," << dsplines.cols() <<"\n";
	
	clsmat out(Nx,2*Ns);
	switch(ul) {
		case 1:
			if(!isCached(dkbU0)) {
				dkbU0 = clsmat(Nx,2*Ns);
				dkbU0.leftCols(Ns) = splines;
				dkbU0.rightCols(Ns) = -1/(2*SoL) *  dsplines;
			}
			
			if(!isCached(dkbUk)) {
				clsmat kappapot(Nx,Nx);
			
				for(int i = 0; i < Nx; i++) {
					kappapot.insert(i,i) = 1.0 / x[i];
				}
				
				dkbUk = clsmat(Nx,2*Ns);
				dkbUk.rightCols(Ns) = 1/(2*SoL) *  (kappapot * (splines));
			}
			
			out = dkbU0 + kappa * dkbUk;
			break;
		case -1:
			if(!isCached(dkbL0)) {
				dkbL0 = clsmat(Nx,2*Ns);
				dkbL0.rightCols(Ns) = splines;
				dkbL0.leftCols(Ns) = 1/(2*SoL) *  dsplines;
			}
			
			if(!isCached(dkbLk)) {
				clsmat kappapot(Nx,Nx);
			
				for(int i = 0; i < Nx; i++) {
					kappapot.insert(i,i) = 1.0 / x[i];
				}
				
				dkbLk = clsmat(Nx,2*Ns);
				dkbLk.leftCols(Ns) = 1/(2*SoL) *  (kappapot * (splines));
			}
			
			out = dkbL0 + kappa * dkbLk;
			break;
			
		default:
			cout << "no.\n";
			break;
	}
	
	
	Nsplines = out.cols();
	return out;
}

clsmat& dkbbasis::dkb0Cache(lvec& x, int ul) {
	//cout << "pointer passed by dkb0Cache " << &x << std::endl;
	switch(ul) {
		case 1:
			if(isCached(dkbU0)) return dkbU0;
			else {
				dkbCache(x,1);
				return dkbU0;
			}
		case -1:
			if(isCached(dkbL0)) return dkbL0;
			else {
				dkbCache(x,-1);
				return dkbL0;
			}
		default:
			cout << "no.\n";
			return dkbU0;
	}
}

clsmat& dkbbasis::dkbkCache(lvec& x, int ul) {
	//cout << "pointer passed by dkbkCache " << &x << std::endl;
	switch(ul) {
		case 1:
			if(isCached(dkbUk)) return dkbUk;
			else {
				dkbCache(x,1);
				return dkbUk;
			}
		case -1:
			if(isCached(dkbLk)) return dkbLk;
			else {
				dkbCache(x,-1);
				return dkbLk;
			}
		default:
			cout << "no.\n";
			return dkbUk;
	}
}


/*
	THIS IMPLEMENTATION CONTRADICTS TYSSE
	
	BUT SHOULD BE CORRECT ACCORDING TO SHABAEV ET AL
	
	AS OF 13/10/21
*/

clsmat dkbbasis::dDkbCache(lvec& x,int ul) {
	clsmat splines = splineCache(x);
	clsmat dsplines = dSplineCache(x,1);
	clsmat d2splines = dSplineCache(x,2);
	
	int Nx = splines.rows();
	int Ns = splines.cols();
	
	
	clsmat out(Nx,2*Ns);
	
	
	switch(ul) {
		case 1:
			if(!isCached(ddkbU0)) {
				ddkbU0 = clsmat(Nx,2*Ns);
				
				ddkbU0.leftCols(Ns) = dsplines;
				ddkbU0.rightCols(Ns) = -  1/(2*SoL) * d2splines;
			}
			
			if(!isCached(ddkbUk)) {
				ddkbUk = clsmat(Nx,2*Ns);
				
				clsmat kappapot(Nx,Nx);
				clsmat kappasqpot(Nx,Nx);
				
				for(int i = 0; i < Nx; i++) {
					kappapot.insert(i,i) = 1.0 / x[i];
					kappasqpot.insert(i,i) = 1.0 / pow(x[i],2);
				}
				ddkbUk.rightCols(Ns) = 1/(2*SoL) * (kappapot * dsplines - kappasqpot * splines);
			}
			
			out = ddkbU0 + ddkbUk;
			break;
		case -1:
		
			if(!isCached(ddkbL0)) {
				ddkbL0 = clsmat(Nx,2*Ns);
				
				ddkbL0.rightCols(Ns) = dsplines;
				ddkbL0.leftCols(Ns) =  1/(2*SoL) * d2splines;
			}
			
			if(!isCached(ddkbLk)) {
				ddkbLk = clsmat(Nx,2*Ns);
				
				clsmat kappapot(Nx,Nx);
				clsmat kappasqpot(Nx,Nx);
				
				for(int i = 0; i < Nx; i++) {
					kappapot.insert(i,i) = 1.0 / x[i];
					kappasqpot.insert(i,i) = 1.0 / pow(x[i],2);
				}
				ddkbLk.leftCols(Ns) =  1/(2*SoL) * (kappapot * dsplines - kappasqpot * splines);
			}
			
			out = ddkbL0 + ddkbLk;
			break;
		default:
			cout << "no.\n";
			break;
	}
	
	return out;
}

clsmat& dkbbasis::dDkb0Cache(lvec& x, int ul) {
	switch(ul) {
		case 1:
			if(isCached(ddkbU0)) return ddkbU0;
			else {
				dDkbCache(x,1);
				return ddkbU0;
			}
		case -1:
			if(isCached(ddkbL0)) return ddkbL0;
			else {
				dDkbCache(x,-1);
				return ddkbL0;
			}
		default:
			cout << "no.\n";
			return ddkbU0;
	}
}

clsmat& dkbbasis::dDkbkCache(lvec& x, int ul) {
	switch(ul) {
		case 1:
			if(isCached(ddkbUk)) return ddkbUk;
			else {
				dDkbCache(x,1);
				return ddkbUk;
			}
		case -1:
			if(isCached(ddkbLk)) return ddkbLk;
			else {
				dDkbCache(x,-1);
				return ddkbLk;
			}
		default:
			cout << "no.\n";
			return ddkbUk;
	}
}


csmat& dkbbasis::olmat() {
	csmat& Sm = getmat<S>();
	csmat& S0 = get0mat<S>();
	csmat& Sk = getk0mat<S>();
	csmat& Skk = getkkmat<S>();
	
	lvec& x = glpts();
	
	clsmat wts = wtmat();
	/*
	clsmat P = dkbCache(x,1);
	clsmat Q = dkbCache(x,-1);
	
	Sm= dkbglintmat(P, Q, P, Q, wts, splinedeg + 1);
	*/
	
	clsmat& P0 = dkb0Cache(x,1);
	clsmat& Q0 = dkb0Cache(x,-1);
	
	clsmat& Pk = dkbkCache(x,1);
	clsmat& Qk = dkbkCache(x,-1);
	
	Sm = factorisedglintmat(S0, Sk, Skk,
							P0, Q0, P0, Q0, Pk, Qk, Pk, Qk,
							splinedeg+1, wts, kappa, &dkbglintmat);
	
	return Sm;
}

csmat& dkbbasis::componentOlmat(int k1, int k2, int comp) {
	csmat& SUm = getmat<SU>();
	csmat& SLm = getmat<SL>();
	
	csmat& SU0 = get0mat<SU>();
	csmat& SL0 = get0mat<SL>();
	
	csmat& SU0k = getk0mat<SU>();
	csmat& SL0k = getk0mat<SL>();
	
	csmat& SUk0 = get0kmat<SU>();
	csmat& SLk0 = get0kmat<SL>();
	
	csmat& SUkk = getkkmat<SU>();
	csmat& SLkk = getkkmat<SL>();
	
	lvec x = glpts();
	
	clsmat& wts = wtmat();
	
	clsmat& P0 = dkb0Cache(x,1);
	clsmat& Q0 = dkb0Cache(x,-1);
	
	clsmat& Pk = dkbkCache(x,1);
	clsmat& Qk = dkbkCache(x,-1);
	
	kappa1 = ik(k1);
	kappa2 = ik(k2);
	
	if(comp < 2) {
		if(!isCached(SUm)) {
			SU0 =  P0.adjoint() * wts * P0;
			SU0k = P0.adjoint() * wts * Pk;
			SUk0 = Pk.adjoint() * wts * P0;
			SUkk = Pk.adjoint() * wts * Pk;
		}
		
		SUm = SU0 + kappa1 * SUk0 + kappa2 * SU0k + kappa1 * kappa2 * SUkk;
		
		return SUm;
	}
	else {
		if(!isCached(SLm)) {
			SL0 =  Q0.adjoint() * wts * Q0;
			SL0k = Q0.adjoint() * wts * Qk;
			SLk0 = Qk.adjoint() * wts * Q0;
			SLkk = Qk.adjoint() * wts * Qk;
		}
		
		SLm = SL0 + kappa1 * SLk0 + kappa2 * SL0k + kappa1 * kappa2 * SLkk;
		
		return SLm;
	}
}

csmat& dkbbasis::ddmat(int dn) {
	csmat& ddm = getmat<dd>();
	csmat& dd0 = get0mat<dd>();
	csmat& ddk = getk0mat<dd>();
	csmat& ddkk = getkkmat<dd>();
	
	
	
	lvec x = glpts();
	clsmat wts = wtmat();
	
	/*clsmat dkbU = dkbCache(x,1);
	clsmat dkbL = dkbCache(x,-1);
	
	clsmat ddkbU = dDkbCache(x,1);
	clsmat ddkbL = dDkbCache(x,-1);
	
	ddm = dkbcrossglintmat(dkbU,dkbL,ddkbU,ddkbL,wts,splinedeg + 1,0);
	*/
	
	clsmat& P0 = dkb0Cache(x,1);
	clsmat& Q0 = dkb0Cache(x,-1);
	
	clsmat& Pk = dkbkCache(x,1);
	clsmat& Qk = dkbkCache(x,-1);
	
	clsmat& dP0 = dDkb0Cache(x,1);
	clsmat& dQ0 = dDkb0Cache(x,-1);
	
	clsmat& dPk = dDkbkCache(x,1);
	clsmat& dQk = dDkbkCache(x,-1);
	
	ddm = factorisedglintmat(dd0, ddk, ddkk,
							P0, Q0, dP0, dQ0, Pk, Qk, dPk, dQk,
							splinedeg+1, wts, kappa, &dkbcrossglintmat, 0, -1);
    /*
	Constructing these matrices properly results in an
	undesirable asymmetry caused by the boundary B-splines.
	To work aroun this we simply cheat a little and take the 
	average of each matrix with its adjoint
	*/
	
	dd0 = (dd0 + csmat(dd0.adjoint()))/2;
	ddk = (ddk + csmat(ddk.adjoint()))/2;
	ddkk = (ddkk + csmat(ddkk.adjoint()))/2;
	
	return ddm;
}

csmat& dkbbasis::Emat(long double (*V)(long double)) {
	csmat& Em = getmat<E>();
	csmat& E0 = get0mat<E>();
	csmat& Ek = getk0mat<E>();
	csmat& Ekk = getkkmat<E>();
	
	if(V != lastV){
		E0 = csmat(0,0);
		Ek = csmat(0,0);
		Ek = csmat(0,0);
	}
	
	if(!isCached(E0) || !isCached(Ek) || !isCached(Ekk)) {
		lvec& x = glpts();
		
		
		
		/*
		clsmat dkbU = dkbCache(x,1);
		clsmat dkbL = dkbCache(x,-1);
		
		Em = dkbglintmat(dkbU,dkbL,dkbU,dkbL,wt,splinedeg + 1);
		*/
		
		clsmat& P0 = dkb0Cache(x,1);
		clsmat& Q0 = dkb0Cache(x,-1);
		
		clsmat& Pk = dkbkCache(x,1);
		clsmat& Qk = dkbkCache(x,-1);
		
		clsmat vdiag(P0.rows(),P0.rows());
		
		for(int i = 0; i < P0.rows(); i++) {
			vdiag.insert(i,i) = V(x[i]);
		}
		
		clsmat wt = wtmat() * vdiag;
		
		Em = factorisedglintmat(E0, Ek, Ekk,
								P0, Q0, P0, Q0, Pk, Qk, Pk, Qk,
								splinedeg+1, wt, kappa, &dkbglintmat);
	}
	
	else {
		Em = cachedglintmat(E0,Ek,Ekk,kappa);
	}
	
	return Em;

}

csmat& dkbbasis::kappamat() {
	csmat& km = getmat<k>();
	csmat& k0 = get0mat<k>();
	csmat& kk = getk0mat<k>();
	csmat& k_2k = getkkmat<k>();
	
	if(!isCached(k0) || !isCached(kk) || !isCached(k_2k) ) {
		
		//Factorising out kappa allows for more rapid calculation
		
		
		/*
		clsmat dkbU = dkbCache(x,1);
		clsmat dkbL = dkbCache(x,-1);
		
		
		km = dkbcrossglintmat(dkbU, dkbL, dkbU, dkbL, kwt, splinedeg+1,1) + dkbcrossglintmat(dkbU, dkbL, dkbU, dkbL, kwt, splinedeg + 1, -1);
		*/
		lvec& x = glpts();

		clsmat& P0 = dkb0Cache(x,1);
		clsmat& Q0 = dkb0Cache(x,-1);
		
		clsmat& Pk = dkbkCache(x,1);
		clsmat& Qk = dkbkCache(x,-1);
		
		
		clsmat kdiag(P0.rows(),P0.rows());
		for(int i = 0; i < P0.rows(); i++) {
			//cout << i << " ";
			
			kdiag.insert(i,i) = 1.0/x[i];
		}
		
		clsmat kwt = wtmat() * kdiag;
		
		km = kappa * factorisedglintmat(k0,kk,k_2k,
										P0, Q0, P0, Q0, Pk, Qk, Pk, Qk,
										splinedeg+1, kwt, kappa, &dkbcrossglintmat, 0, 1);
		
		return km;
	}
	
	else {
		km = kappa * cachedglintmat(k0,kk,k_2k,kappa);
		return km;
	}
}

csmat& dkbbasis::ulcmat() {
	csmat& ulcm = getmat<ulc>();
	csmat& ulc0 = get0mat<ulc>();
	csmat& ulck = getk0mat<ulc>();
	csmat& ulckk = getkkmat<ulc>();
	
	lvec x = glpts();
	
	clsmat& wts = wtmat();
	/*
	clsmat dkbU = dkbCache(x,1);
	clsmat dkbL = dkbCache(x,-1);
	//csmat dkb0 = 0*dkbU;
	
	ulcm =// dkbglintmat(&dkbU,&dkbL,&dkbU,&dkbL,&cwt,splinedeg + 1,1) 
		- dkbglintmat(dkbU,dkbL,dkbU,dkbL,cwt,splinedeg + 1, -1);
	*/
	
	clsmat& P0 = dkb0Cache(x,1);
	clsmat& Q0 = dkb0Cache(x,-1);
	
	clsmat& Pk = dkbkCache(x,1);
	clsmat& Qk = dkbkCache(x,-1);

	ulcm =  -2 * pow(SoL,2) * factorisedglintmat(ulc0,ulck,ulckk,
											  P0, Q0, P0, Q0, Pk, Qk, Pk, Qk,
											  splinedeg+1, wts, kappa, &dkbglintmat, -1);
	
	return ulcm;

}

csmat& dkbbasis::p1mat() {
	csmat& p1m = getmat<p1>();
	p1m = ddmat(1) + kappamat();
	return p1m;
}

// csmat& dkbbasis::H0mat() {
	// csmat& H0m = getmat<H0>();
	
	// csmat& H0_0 = getH0mat(0);
	// csmat& H0_k = getH0mat(1);
	// csmat& H0_k2 = getH0mat(2);
	// csmat& H0_k3 = getH0mat(3);
	
	// H0m = H0_0 + kappa * H0_k + pow(kappa,2) * H0_k2 + pow(kappa,3) * H0_k3;
	
	// return H0m;
// }

csmat& dkbbasis::dpalphmat(int k1,int k2, int ul) {
	csmat& dpalphm = getmat<dpa>();
	csmat& dpalph0 = get0mat<dpa>(ul);
	csmat& dpalphk0 = getk0mat<dpa>(ul);
	csmat& dpalph0k = get0kmat<dpa>(ul);
	csmat& dpalphkk = getkkmat<dpa>(ul);
	
	lvec x = glpts();
	/*setState(k1);
	clsmat dkbU1 = dkbCache(x,1);
	clsmat dkbL1 = dkbCache(x,-1);
	
	setState(k2);
	clsmat dkbU2 = dkbCache(x,1);
	clsmat dkbL2 = dkbCache(x,-1);
	*/
	
	int kappa1 = ik(k1);
	int kappa2 = ik(k2);
	
	//cout << "kappa1: " << kappa1 <<", kappa2: " << kappa2 << "\n";
	
	clsmat& wt = wtmat();
	
	clsmat& P0 = dkb0Cache(x,1);
	clsmat& Q0 = dkb0Cache(x,-1);
	
	clsmat& Pk = dkbkCache(x,1);
	clsmat& Qk = dkbkCache(x,-1);
	
	if(!isCached(dpalph0) || !isCached(dpalphk0) || !isCached(dpalph0k) || !isCached(dpalphkk)) {
		
		// csmat alph0 =(P0.adjoint() * wt * Q0).cast<cdouble>();
		// dpalph0 = alph0.selfadjointView<Eigen::Upper>();
		
		switch(ul) {
			case -1:
				dpalph0  = Q0.adjoint() * wt * P0;
				dpalphk0 = Qk.adjoint() * wt * P0;
				dpalph0k = Q0.adjoint() * wt * Pk;
				dpalphkk = Qk.adjoint() * wt * Pk;
				
				break;
			
			case 1:
				dpalph0  = P0.adjoint() * wt * Q0;
				dpalphk0 = Pk.adjoint() * wt * Q0;
				dpalph0k = P0.adjoint() * wt * Qk;
				dpalphkk = Pk.adjoint() * wt * Qk;
				
				break;
		}
		
		
		// csmat alphkk = (Pk.adjoint() * wt * Qk).cast<cdouble>();
		// dpalphkk = alphkk.selfadjointView<Eigen::Upper>();
		
	}
	
	switch(ul) {
		case -1:
			dpalphm = dpalph0 + kappa2 * dpalphk0 + kappa1 * dpalph0k + kappa1 * kappa2 * dpalphkk;
			
			break;
		case 1:
			dpalphm = dpalph0 + kappa1 * dpalphk0 + kappa2 * dpalph0k + kappa1 * kappa2 * dpalphkk;
			
			break;
		default:
			return dpalphm;
	}
	
	//Have to use -ul for sorcerous, difficult to comprehend reasons
	//dpalphm = dkbcrossglintmat(dkbU1, dkbL1, dkbU2, dkbL2, wt, splinedeg + 1, -ul);
	return dpalphm;
}

csmat& dkbbasis::bdpalphmat(int k1, int k2, int ul, int l, int alpha, const Potential<beyondDipolePulse>& vExt) {
	csmat& dpalphm  =   getmat<bdpa>();
	csmat& dpalph0  = getbdpmat<p00>(l,alpha);
	csmat& dpalph0k = getbdpmat<p0k>(l,alpha);
	csmat& dpalphk0 = getbdpmat<pk0>(l,alpha);
	csmat& dpalphkk = getbdpmat<pkk>(l,alpha);
	
	if(!isCached(dpalph0)) {
		
		lvec x = glpts();
		
		int kappa1 = ik(k1);
		int kappa2 = ik(k2);
		
		clsmat& wt = wtmat();
		
		//cout << "Size of x coord: " << x.size() << std::endl;
		
		//if(!isCached(Calph)) {
		Calph = vExt.template axialPart<axis::radial>(x,l);
		
		clsmat wtC = wt * Calph.row(alpha).asDiagonal();
		/*if(l%2 == alpha%2) { //both odd or both even
			wtC = clsmat(wtC.rows(),wtC.cols());
		}*/
		
		clsmat& P0 = dkb0Cache(x,1);
		clsmat& Q0 = dkb0Cache(x,-1);
		
		clsmat& Pk = dkbkCache(x,1);
		clsmat& Qk = dkbkCache(x,-1);
		
		
		// cout << "products\n";
		
		dpalph0 = P0.adjoint() * wtC * Q0;
		
		dpalphk0 = Pk.adjoint() * wtC * Q0;
		dpalph0k = P0.adjoint() * wtC * Qk;
		
		dpalphkk = Pk.adjoint() * wtC * Qk;
	}
	
	// cout << "summations\n";
	
	switch(ul) {
		case -1:
			dpalphm = dpalph0.adjoint() + kappa2 * dpalphk0.adjoint() + kappa1 * dpalph0k.adjoint() + kappa1 * kappa2 * dpalphkk.adjoint();
			
			break;
		case 1:
			dpalphm = dpalph0 + kappa1 * dpalphk0 + kappa2 * dpalph0k + kappa1 * kappa2 * dpalphkk;
			
			break;
		default:
			return dpalphm;
	}
	// cout << "return statement\n";
	return dpalphm;
}

csmat& dkbbasis::bdpalphmat(int k1, int k2, int ul, int l, const Potential<beyondDipolePulse>& vExt) {
	this->vExt = &vExt;
	csmat& dpalphm  =   getmat<bdpa>();
	/*csmat& dpalph0  = getbdpmat<p00>(l,0);
	csmat& dpalph0k = getbdpmat<p0k>(l,0);
	csmat& dpalphk0 = getbdpmat<pk0>(l,0);
	csmat& dpalphkk = getbdpmat<pkk>(l,0);
	*/
	if(!isCached(getbdpmat<p00>(l,0))) {
		lvec x = glpts();
		
		int kappa1 = ik(k1);
		int kappa2 = ik(k2);
		
		clsmat& wt = wtmat();
		
		//cout << "Size of x coord: " << x.size() << std::endl;
		//Explicitly put time dependence in bdpalphmat!
		Calph = vExt.template axialPart<axis::radial>(x,l);
		
		clsmat wtC[6];
		for(int alpha = 0; alpha < 6; alpha++) {
			wtC[alpha] = wt * Calph.row(alpha).asDiagonal();
		}
		
		
		clsmat& P0 = dkb0Cache(x,1);
		clsmat& Q0 = dkb0Cache(x,-1);
		
		clsmat& Pk = dkbkCache(x,1);
		clsmat& Qk = dkbkCache(x,-1);
		
		
		// cout << "products\n";
		for(int alpha = 0; alpha < 6; alpha++) {
			
			getbdpmat<p00>(l,alpha) = P0.adjoint() * wtC[alpha] * Q0;
			getbdpmat<p0k>(l,alpha) = Pk.adjoint() * wtC[alpha] * Q0;
			getbdpmat<pk0>(l,alpha) = P0.adjoint() * wtC[alpha] * Qk;
			getbdpmat<pkk>(l,alpha) = Pk.adjoint() * wtC[alpha] * Qk;
		}
	}
	
	// cout << "summations\n";
	dpalphm = csmat(getbdpmat<p00>(l,0).rows(), getbdpmat<p00>(l,0).cols());
	switch(ul) {
		case -1:
			for(int alpha = 0; alpha < 6; alpha++) {
				if(l%2==alpha%2) {
					dpalphm += ((double) vExt.template axialPart<axis::t>(vExt.getTime())(alpha,0).real())
					* csmat(
							     		getbdpmat<p00>(l,alpha).adjoint() 
							 + kappa2 * getbdpmat<p0k>(l,alpha).adjoint() 
							 + kappa1 * getbdpmat<pk0>(l,alpha).adjoint() 
				    + kappa1 * kappa2 * getbdpmat<pkk>(l,alpha).adjoint()
				    );
				}
			}
			break;
		case 1:
			for(int alpha = 0; alpha < 6; alpha++) {
				if(l%2==alpha%2) {
					dpalphm += ((double) vExt.template axialPart<axis::t>(vExt.getTime())(alpha,0).real() )
					* (
							 		    getbdpmat<p00>(l,alpha)
							 + kappa1 * getbdpmat<p0k>(l,alpha)
							 + kappa2 * getbdpmat<pk0>(l,alpha)
					+ kappa1 * kappa2 * getbdpmat<pkk>(l,alpha)
					);
				}
			}
			break;
		default:
			return dpalphm;
	}
	// cout << "return statement\n";
	return dpalphm;
}
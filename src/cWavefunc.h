#ifndef CWAVEFUNC_H
#define CWAVEFUNC_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Eigenvalues>
#include <iostream>
#include <boost/math/special_functions/legendre.hpp>

#include "defs.h"
#include "potential.h"
//#include "dkbbasis.h"

using namespace std;

//Dirac solver typedefs

void evilprod(csmat *out, csmat *a, csmat *b);




/*
template<matname M>
inline cvec bsbasis::matvec(const cvec& v) {
	csmat& M0 = getmat<M>();
	
	return M0 * v;
}

template<>
inline cvec bsbasis::matvec<L>(const cvec& v) {
	csmat& M0 = getmat<L>();
	
	return cdouble(mfl,0) * M0 * v;
}
*/






template <typename D, typename S> 
std::complex<D> cast(const std::complex<S> s)
{
    return std::complex<D>(s.real(), s.imag());
}



template <int Z>
long double coloumb(long double r) {
	if(r == 0.0) return 0;
	else return -(long double)Z/r;
}

template <int N, int omega, int E0>
long double dplA(long double t) {
	
	long double T = (long double)N*2*PI / omega;
	
	return (long double)E0/(long double)omega * pow(sin(PI*t/(long double)T),2) * sin((long double)omega*t) * (t < (long double)T);
}


template <class scalar>
Eigen::Matrix<scalar,Eigen::Dynamic,Eigen::Dynamic> expm(Eigen::Matrix<scalar,Eigen::Dynamic,Eigen::Dynamic> in, int n) {
	int N = in.rows();
	Eigen::Matrix<scalar,Eigen::Dynamic,Eigen::Dynamic> out(N,N);
	Eigen::Matrix<scalar,Eigen::Dynamic,Eigen::Dynamic> powm(N,N);
	powm.setZero();
	for(int i = 0; i < N; i++) {
		powm(i,i) = 1;
	}
	
	out = powm;
	
	for(int i = 1; i < n; i++) {
		powm = (in * powm)/i;
		out = out + powm;
	}
	
	return out;
}


vec& linInterpolate(const vec& x, const vec& y, const vec& pts, vec& out);


csmat factorisedglintmat(csmat& M0, csmat& Mk, csmat& Mkk,
						const clsmat& aP0,const clsmat& aQ0,const clsmat& bP0,const clsmat& bQ0,
						const clsmat& aPk,const clsmat& aQk,const clsmat& bPk,const clsmat& bQk,
						int k, const clsmat& wts, int kappa, csmat(*func)(const clsmat& aP,const clsmat& aQ,const clsmat& bP,const clsmat& bQ,const clsmat& wts,int k, int ul, int sumcoef), int ul = 0, int sumcoef = 1);

csmat dkbglintmat(const clsmat& a0, const clsmat& b0, const clsmat& ws, int k);

csmat dkbglintmat(const clsmat& a0,const clsmat&  a1,const clsmat& b0,const clsmat& b1,const clsmat& ws,int k, int ul = 0, int sumcoef = 1);
#endif
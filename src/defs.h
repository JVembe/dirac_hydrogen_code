/*
defs.h is for typedefs, enums and functions that are used throughout the code.
A lot of Eigen types are given shorthand names here.
*/

#ifndef DEFS_H
#define DEFS_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>

#define SoL 137.035999084
#define PI 3.141592653589793238462643383279
#define UPPER 1
#define LOWER -1
#define USEMPI 0
//#define USE_MY_OPENMP
#define EIGEN_DONT_PARALLELIZE 1

using namespace std;

typedef Eigen::Vector<double, Eigen::Dynamic> vec;
typedef Eigen::Vector<long double,Eigen::Dynamic> lvec;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mat;
typedef Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic> lmat;

typedef complex<double> cdouble;
typedef complex<long double> cldouble;
typedef Eigen::Matrix<cdouble, Eigen::Dynamic,Eigen::Dynamic> cmat;
typedef Eigen::Matrix<cldouble, Eigen::Dynamic,Eigen::Dynamic> clmat;
typedef Eigen::Vector<cdouble, Eigen::Dynamic> cvec;


typedef Eigen::SparseMatrix<double,Eigen::RowMajor> dsmat;
typedef Eigen::SparseMatrix<cdouble,Eigen::RowMajor> csmat;
typedef Eigen::SparseMatrix<cldouble> clsmat;

//typedef Eigen::SparseMatrix<cdouble,RowMajor> rmcsmat;

typedef Eigen::GeneralizedSelfAdjointEigenSolver<mat> eigSolver;
typedef Eigen::GeneralizedSelfAdjointEigenSolver<mat> diracEigSolver;

typedef Eigen::Triplet<cdouble> T;
typedef Eigen::Triplet<double> dT;
typedef Eigen::Triplet<long double> ldT;

typedef Eigen::Matrix<cldouble,6,Eigen::Dynamic> bdpft;
typedef Eigen::Matrix<cldouble,8,Eigen::Dynamic> bdpsqft;

typedef Eigen::Array<int,Eigen::Dynamic,1> iarray;

enum axis {
	all,
	radial,
	azimuth,
	polar,
	angular, //Both axes in cases of spherical harmonics
	x,
	y,
	z,
	t
};

enum dkbpart {
	p00,
	p0k,
	pk0,
	pkk,
	Npts
};

//These functions serve to enumerate the quantum numbers kappa,mu and l,m.

int ik(int i);
int ki(int kappa);
iarray ik(iarray i);
double imu(int i);

inline int il(int i) {
	//int ii = i/2;
	
	return floor(sqrt(i));
}


inline iarray il(iarray i) {
	return floor(sqrt(i));
}

inline int li(int l) {
	return pow(l,2);
}

inline int im(int i) {
	int l = il(i);
	
	int ii = i - pow(il(i),2);
	

	return (ii % (il(i)*2 + 1)) - il(i);
}


template <typename Derived>
bool isCached(const Eigen::EigenBase<Derived>& obj) {
	if(obj.rows() == 0 || obj.cols() == 0) {
		//cout << "Returning false\n";
		return false;
	}
	else 
	{
		//cout << "Returning true\n";
		return true;
	}
}

template <typename Derived>
bool isCached(const Eigen::EigenBase<Derived>* obj) {
	if(obj == NULL) {
		//cout << "Returning false\n";
		return false;
	}
	else 
	{
		//cout << "Returning true\n";
		return isCached(*obj);
	}
}

template <typename D, typename S> 
std::complex<D> cast(const std::complex<S> s)
{
    return std::complex<D>(s.real(), s.imag());
}

//Simple potentials are implemented through function pointers. This does not work that well for the nondipole potential, which has its own handling.

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
#endif
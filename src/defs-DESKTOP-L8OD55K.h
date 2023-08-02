#ifndef DEFS_H
#define DEFS_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>

#define SoL 137.035999084
#define PI 3.141592653589793238462643383279

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

int ik(int i);
int ki(int kappa);
double imu(int i);
int il(int i);
int li(int i);
int im(int i);

iarray ik(iarray i);
iarray il(iarray i);

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


#endif
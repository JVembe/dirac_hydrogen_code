/*
	Class for potentials, primarily the nondipole laser pulse potential.
	
	There's a lot of stuff in here that's questionable in terms of how useful it actually is, as I wrote it with some intention of making it a lot more expandable than it needs to be right now 
*/

#ifndef POTENTIAL_H
#define POTENTIAL_H
#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/spherical_harmonic.hpp>
#include "defs.h"



template<typename Derived>
class Coordinate {
	
	
	public:
		Coordinate();
		
		template <axis Ax>
		double axisProj() const {
			return static_cast<Derived*>(this)->axis_impl(std::integral_constant<axis,Ax>{} );
		}
};

class sphCoord: public Coordinate<sphCoord> {
	friend class Coordinate<sphCoord>;
	protected:
		const double r;
		const double th;
		const double ph;
	
	public:
		sphCoord(double r, double th, double ph):r(r),th(th),ph(ph) {
		}
		
		double axis_impl(std::integral_constant<axis,axis::radial> c) const {
			return r;
		}
		
		double axis_impl(std::integral_constant<axis,axis::azimuth> c) const {
			return th;
		}
		
		double axis_impl(std::integral_constant<axis,axis::polar> c) const {
			return ph;
		}
		
		double axis_impl(std::integral_constant<axis,axis::x> c) const {
			return r*sin(th)*sin(ph);
		}
		
		double axis_impl(std::integral_constant<axis,axis::y> c) const {
			return r*cos(th)*sin(ph);
		}
		
		double axis_impl(std::integral_constant<axis,axis::z> c) const {
			return r*cos(ph);
		}
};

template<typename Derived>
struct traits {
	bool isTimeDependent;
	bool isSpaceDependent;
	typedef double PotentialType;
	typedef mat MatrixReturnType;
	const PotentialType unity = 1.0;
	typedef double PotentialSqType;
	typedef mat MatrixSqReturnType;
	const PotentialSqType unitySq = 1.0;
};

template<typename Derived>
class Potential	{
	protected:
		template <typename coordType, axis Ax>
		typename traits<Derived>::PotentialType axialPart_impl(std::integral_constant<axis,Ax> c,const Coordinate<coordType>& x) {
			return traits<Derived>::unity;
		}
		
		template <typename MatrixType>
		typename traits<Derived>::MatrixReturnType axialPart_impl(const Eigen::MatrixBase<MatrixType> x) const {
			return typename traits<Derived>::MatrixReturnType::Constant(x.rows(),x.cols(),traits<Derived>::unity);
		}
		
		template <typename Scalar, axis Ax>
		typename traits<Derived>::PotentialType axialPart_impl(std::integral_constant<axis,Ax> c, const Scalar x) const {
			return traits<Derived>::unity;
		}
		
		template <typename coordType, axis Ax>
		typename traits<Derived>::PotentialSqType axialPartSq_impl(std::integral_constant<axis,Ax> c,const Coordinate<coordType>& x) {
			return traits<Derived>::unitySq;
		}
		
		template <typename MatrixType>
		typename traits<Derived>::MatrixSqReturnType axialPartSq_impl(const Eigen::MatrixBase<MatrixType> x) const {
			return typename traits<Derived>::MatrixSqReturnType::Constant(x.rows(),x.cols(),traits<Derived>::unitySq);
		}
		
		template <typename Scalar, axis Ax>
		typename traits<Derived>::PotentialSqType axialPartSq_impl(std::integral_constant<axis,Ax> c, const Scalar x) const {
			return traits<Derived>::unitySq;
		}
		
		double* t = new double;
	public:
		Potential() { };
		
		template <typename coordType>
		typename traits<Derived>::PotentialType evalAt(const Coordinate<coordType>& x) {
			return static_cast<Derived*>(this)->evalAt_impl(x); 
		}
		
		template <axis Ax, typename coordType>
		typename traits<Derived>::PotentialType axialPart(const coordType& x) const {
			return static_cast<const Derived*>(this)->axialPart_impl(std::integral_constant<axis,Ax> {}, x);
		}
		
		
		template <axis Ax, typename MatrixType>
		typename traits<Derived>::MatrixReturnType axialPart(const Eigen::MatrixBase<MatrixType>& x) const {
			return static_cast<const Derived*>(this)->template axialPart_impl(std::integral_constant<axis,Ax> {}, x);
		}
		
		template <axis Ax, typename MatrixType>
		typename traits<Derived>::MatrixReturnType axialPart(const Eigen::MatrixBase<MatrixType>& x,int param) const {
			return static_cast<const Derived*>(this)->template axialPart_impl(std::integral_constant<axis,Ax> {}, x, param);
		}
		
		template <axis Ax, typename coordType, typename... ArgTs>
		typename traits<Derived>::PotentialSqType axialPartSq(const coordType& x, ArgTs... Args) const {
			return static_cast<const Derived*>(this)->axialPartSq_impl(std::integral_constant<axis,Ax> {}, x, Args...);
		}
		
		
		template <axis Ax, typename MatrixType>
		typename traits<Derived>::MatrixSqReturnType axialPartSq(const Eigen::MatrixBase<MatrixType>& x) const {
			return static_cast<const Derived*>(this)->template axialPartSq_impl(std::integral_constant<axis,Ax> {}, x);
		}
		
		template <axis Ax, typename MatrixType>
		typename traits<Derived>::MatrixSqReturnType axialPartSq(const Eigen::MatrixBase<MatrixType>& x,int param) const {
			return static_cast<const Derived*>(this)->template axialPartSq_impl(std::integral_constant<axis,Ax> {}, x, param);
		}
		
		template<typename coordType>
		typename traits<Derived>::PotentialType operator()(const coordType& x) const {
			return evalAt(x);
		}
		
		double getTime() const {
			return *(this->t);
		}
		
		void setTime(double t) const {
			*(this->t) = t;
		}
		
		
		
};

template<int Z>
class Coloumb;

template<int Z>
struct traits<Coloumb<Z> > {
	bool isTimeDependent = false;
	bool isSpaceDependent = true;
	bool isRadSeparable = true;
	typedef double PotentialType;
	typedef mat MatrixReturnType;
	typedef double PotentialSqType;
	typedef mat MatrixSqReturnType;
};

template <int Z>
class Coloumb: public Potential<Coloumb<Z> > {
	Coloumb() : Potential<Coloumb<Z > >() {};
	template <typename coordType>
	double evalAt_impl(const coordType& x) {
		return this->axialPart<radial>(x.template axisProj<radial>(x));
	}
	
	template <typename Scalar>
	double axialPart_impl(std::integral_constant<axis,radial> c, const Scalar x) const {
		if(x==0.0d) return 0.0d;
		else return Z/x;
	}
	
	template <typename MatrixType>
	typename traits<Coloumb<Z> >::MatrixReturnType axialPart_impl(std::integral_constant<axis,radial> c, const Eigen::MatrixBase<MatrixType>& x) const {
		
		typename traits<Coloumb<Z> >::MatrixReturnType out = Z * x.cwiseInvert();
		return out;
	}
};

class dipolePulse;

template<>
struct traits<dipolePulse> {
	bool isTimeDependent = true;
	bool isSpaceDependent = false;
	bool isRadSeparable = false;
	typedef double PotentialType;
	typedef mat MatrixReturnType;
	const PotentialType unity = 1.0;
	typedef double PotentialSqType;
	typedef mat MatrixSqReturnType;
	const PotentialSqType unitySq = 1.0;
};

class dipolePulse:public Potential<dipolePulse> {
	
	double E0;
	double omega; 
	double T;
	
	public:
		dipolePulse():Potential<dipolePulse>() {
			this->E0 = 0;
			this->omega = 0;
			this->T = 0;
		}
		
		dipolePulse(double E0, double omega, double N):Potential<dipolePulse>() {
			this->E0 = E0;
			this->omega = omega;
			this->T = (double)N*2*PI / omega;
		}

		double axialPart_impl(std::integral_constant<axis,axis::t> c, double t) const {
			return E0/omega * pow(sin(PI*t/T),2) * sin(omega*t) * (t < T);
		}
};


class beyondDipolePulse;

template<>
struct traits<beyondDipolePulse> {
	bool isTimeDependent = true;
	bool isSpaceDependent = true;
	bool isRadSeparable = true;
	typedef bdpft PotentialType;
	typedef bdpft MatrixReturnType;
	typedef bdpsqft PotentialSqType;
	typedef bdpsqft MatrixSqReturnType;
};

namespace bmath = boost::math;


class beyondDipolePulse:public Potential<beyondDipolePulse> {
	
	protected:
		double E0;
		double omega;
		double T;
		static int l;
		
		double dirtheta;
		double dirphi;
	
	public:
		
		static long double besselJ(long double x) {
			int ll = l;
			return bmath::sph_bessel(ll,x);		
		}
		beyondDipolePulse():Potential<beyondDipolePulse>() {};
		beyondDipolePulse(double E0, double omega, double N):Potential<beyondDipolePulse>() {
			this->E0 = E0;
			this->omega = omega;
			this->T = (double)N*2*PI / omega;
		}
		
		// #pragma message("Compiling with modified beyond dipole potential, remember to restore full BDP functionality")
		
		// bdpft axialPart_impl(std::integral_constant<axis,axis::t> c, double t) const {
			// bdpft out = bdpft::Zero(6,1);
			// using Scalar = double;
			// cdouble phi = cdouble(0,0);
			
			// //Remaining rows are redundant and can be optimized out later
			// out(0,0) = ((Scalar)  E0/(omega) * pow( sin(PI * t / T), 2) * cos((phi + t) * (Scalar) omega));
			// out(1,0) = ((Scalar)  E0/(omega) * pow( sin(PI * t / T), 2) * sin((phi + t) * (Scalar) omega));
			
			// return out;
		// }
		
		// template <typename MatrixType>
		// bdpft axialPart_impl(std::integral_constant<axis,axis::t> c, const Eigen::MatrixBase<MatrixType>& t) const {
			// using Scalar = cldouble; //Easiest fix for earlier mistake
			// bdpft out = bdpft::Zero(6,t.size());
			
			// auto tt = t.col(0).template cast<cdouble>();
			
			// cdouble phi = cdouble(0,0);
			
			// out.row(0) = ((Scalar) E0/(cldouble(omega,0)) * pow( sin(PI * t / T), 2) * cos((phi + tt) * (Scalar) omega));
			// out.row(1) = ((Scalar) E0/(cldouble(omega,0)) * pow( sin(PI * t / T), 2) * sin((phi + tt) * (Scalar) omega));
			
			// return out;
		// }
		
		// bdpft axialPart_impl(std::integral_constant<axis,axis::x> c, double x) const {
			// bdpft out = bdpft::Zero(6,1);
			
			// out(0,0) = cos(omega/SoL*x);
			// out(1,0) = sin(omega/SoL*x);
			
			// return out;
		// }
		
		// template <typename MatrixType>
		// bdpft axialPart_impl(std::integral_constant<axis,axis::x> c, const Eigen::MatrixBase<MatrixType>& x) const {
			// bdpft out = bdpft::Zero(6,x.size());
			
			// cldouble k = omega/SoL;
			
			// out.row(0) = cos(k * x);
			// out.row(1) = sin(k * x);
			
			// return out;
		// }
		
		// template <typename MatrixType>
		// bdpft axialPart_impl(std::integral_constant<axis,axis::radial> c, const Eigen::MatrixBase<MatrixType>& x,int l) const {
			// bdpft out = bdpft::Zero(6,x.size());
			
			// double k = omega/SoL;
			
			// //Bessel function must be evaluated on x for each x, but this is expensive. Need to consider options.
			// beyondDipolePulse::l = l;
			
			// //This feels unhinged
			
			// out.row(0) =(( k * x.real()).unaryExpr(std::ref(besselJ))).template cast<cldouble>();
			// out.row(1) =(( k * x.real()).unaryExpr(std::ref(besselJ))).template cast<cldouble>();
			
			// //cout << "Radial part matrix dimensions: " << out.rows() << ", " << out.cols() << std::endl;
			
			// return out;
		// }
		
		bdpft axialPart_impl(std::integral_constant<axis,axis::t> c, double t) const {
			bdpft out(6,1);
			using Scalar = double;
			cdouble phi = cdouble(-PI/2,0);
			// cdouble phi = cdouble(0,0);//cdouble(-PI/2,0);
			
			//Row 3 originally -E0. Negated to counter issue in Bessel function expansion
			
			// out(0,0) = ((Scalar) (E0 / (omega) ) * pow(sin(PI / T * t),2) * sin(omega * t))/3;
			// out(1,0) = 0;
			// out(2,0) = ((Scalar) (E0 / (omega) ) * pow(sin(PI / T * t),2) * sin(omega * t))/3;
			// out(3,0) = 0;
			// out(4,0) = ((Scalar) (E0 / (omega) ) * pow(sin(PI / T * t),2) * sin(omega * t))/3;
			// out(5,0) = 0;
			
			
			out(0,0) = ((Scalar) -E0/(4*omega) * cos( phi + t * (Scalar) (2 * PI / T + omega)));
			out(1,0) = ((Scalar) -E0/(4*omega) * sin( phi + t * (Scalar) (2 * PI / T + omega)));
			out(2,0) = ((Scalar) -E0/(4*omega) * cos(-phi + t * (Scalar) (2 * PI / T - omega)));
			out(3,0) = ((Scalar)  E0/(4*omega) * sin(-phi + t * (Scalar) (2 * PI / T - omega)));
			out(4,0) = ((Scalar)  E0/(2*omega) * cos( phi + t * (Scalar) omega));
			out(5,0) = ((Scalar)  E0/(2*omega) * sin( phi + t * (Scalar) omega));
			
			return out;
		}
		
		template <typename MatrixType>
		bdpft axialPart_impl(std::integral_constant<axis,axis::t> c, const Eigen::MatrixBase<MatrixType>& t) const {
			using Scalar = cldouble; //Easiest fix for earlier mistake
			bdpft out(6,t.size());
			
			auto tt = t.col(0).template cast<cdouble>();
			
			cdouble phi = cdouble(0,0);
			
			
			// out.row(0) = ((Scalar) (E0 / (omega) ) * pow(sin(PI / T * tt),2) * cos(omega * tt));
			// out.row(1) = MatrixType::Constant(t.size(),0.0).real();
			// out.row(2) = MatrixType::Constant(t.size(),0.0).real();
			// out.row(3) = MatrixType::Constant(t.size(),0.0).real();
			// out.row(4) = MatrixType::Constant(t.size(),0.0).real();
			// out.row(5) = MatrixType::Constant(t.size(),0.0).real();
			
			//Row 3 originally -E0. Negated to counter issue in Bessel function expansion
			
			out.row(0) = ((Scalar) (-E0 / (4*omega) ) * cos(( phi + tt) * (Scalar) (2 * PI / T + omega)));
			out.row(1) = ((Scalar) (-E0 / (4*omega) ) * sin(( phi + tt) * (Scalar) (2 * PI / T + omega)));
			out.row(2) = ((Scalar) (-E0 / (4*omega) ) * cos((-phi + tt) * (Scalar) (2 * PI / T - omega)));
			out.row(3) = ((Scalar) ( E0 / (4*omega) ) * sin((-phi + tt) * (Scalar) (2 * PI / T - omega)));
			out.row(4) = ((Scalar) ( E0 / (2*omega) ) * cos(( phi + tt) * (Scalar) omega));
			out.row(5) = ((Scalar) ( E0 / (2*omega) ) * sin(( phi + tt) * (Scalar) omega));
			
			
			/*out.col(1) =  E0 / (4*omega) * cos(omega * t.col(0) + 2 * PI * t.col(0) / T);
			out.col(2) = -E0 / (4*omega) * sin(omega * t.col(0) - 2 * PI * t.col(0) / T);
			out.col(3) =  E0 / (4*omega) * cos(omega * t.col(0) - 2 * PI * t.col(0) / T);
			out.col(4) = -E0 / (4*omega) * sin(omega * t.col(0));
			out.col(5) =  E0 / (4*omega) * cos(omega * t.col(0));*/
			
			return out;
		}
		
		bdpft axialPart_impl(std::integral_constant<axis,axis::x> c, double x) const {
			bdpft out(6,1);
			
			out(0,0) = cos(omega/SoL*x + 2*PI*x/(SoL*T));
			out(1,0) = sin(omega/SoL*x + 2*PI*x/(SoL*T));
			out(2,0) = cos(omega/SoL*x - 2*PI*x/(SoL*T));
			out(3,0) = sin(omega/SoL*x - 2*PI*x/(SoL*T));
			out(4,0) = cos(omega/SoL*x);
			out(5,0) = sin(omega/SoL*x);
			
			return out;
		}
		
		template <typename MatrixType>
		bdpft axialPart_impl(std::integral_constant<axis,axis::x> c, const Eigen::MatrixBase<MatrixType>& x) const {
			bdpft out(6,x.size());
			
			cldouble k = omega/SoL;
			
			out.row(0) = cos(k * x + 2*PI * x / (SoL * T));
			out.row(1) = sin(k * x + 2*PI * x / (SoL * T));
			out.row(2) = cos(k * x - 2*PI * x / (SoL * T));
			out.row(3) = sin(k * x - 2*PI * x / (SoL * T));
			out.row(4) = cos(k * x);
			out.row(5) = sin(k * x);
			
			return out;
		}
		
		template <typename MatrixType>
		bdpft axialPart_impl(std::integral_constant<axis,axis::radial> c, const Eigen::MatrixBase<MatrixType>& x,int l) const {
			bdpft out(6,x.size());
			
			double k = omega/SoL;
			
			//Bessel function must be evaluated on x for each x, but this is expensive. Need to consider options.
			beyondDipolePulse::l = l;
			
			//cout << "l: " << l << std::endl;
			
			//Rows 2 and 3 must be negated to avoid issues with bessel function expansion
			
			
			// out.row(0) = MatrixType::Constant(x.size(),1.0).real() * (l==0);
			// out.row(1) = MatrixType::Constant(x.size(),1.0).real() * (l==0);
			// out.row(2) = MatrixType::Constant(x.size(),1.0).real() * (l==0);
			// out.row(3) = MatrixType::Constant(x.size(),1.0).real() * (l==0);
			// out.row(4) = MatrixType::Constant(x.size(),1.0).real() * (l==0);
			// out.row(5) = MatrixType::Constant(x.size(),1.0).real() * (l==0);
			
			
			out.row(0) = /*(long double) sqrt(4*PI) * /*(2*l+1)) * pow(cldouble(0,1),l  ) * */ (( (2*PI/(SoL * T) + k ) * x.real()/* MatrixType::Constant(x.size(),0.0).real()*/).unaryExpr(std::ref(besselJ))).template cast<cldouble>();
			out.row(1) = /*(long double) sqrt(4*PI) * /*(2*l+1)) * pow(cldouble(0,1),l-1) * */ (( (2*PI/(SoL * T) + k ) * x.real()/* MatrixType::Constant(x.size(),0.0).real()*/).unaryExpr(std::ref(besselJ))).template cast<cldouble>();
			out.row(2) = /*(long double) sqrt(4*PI) * /*(2*l+1)) * pow(cldouble(0,1),l  ) * */ ((-(2*PI/(SoL * T) - k ) * x.real()/* MatrixType::Constant(x.size(),0.0).real()*/).unaryExpr(std::ref(besselJ))).template cast<cldouble>();
			out.row(3) = /*(long double) sqrt(4*PI) * /*(2*l+1)) * pow(cldouble(0,1),l-1) * */ ((-(2*PI/(SoL * T) - k ) * x.real()/* MatrixType::Constant(x.size(),0.0).real()*/).unaryExpr(std::ref(besselJ))).template cast<cldouble>();
			out.row(4) = /*(long double) sqrt(4*PI) * /*(2*l+1)) * pow(cldouble(0,1),l  ) * */ ((                   k   * x.real()/* MatrixType::Constant(x.size(),0.0).real()*/).unaryExpr(std::ref(besselJ))).template cast<cldouble>();
			out.row(5) = /*(long double) sqrt(4*PI) * /*(2*l+1)) * pow(cldouble(0,1),l-1) * */ ((                   k   * x.real()/* MatrixType::Constant(x.size(),0.0).real()*/).unaryExpr(std::ref(besselJ))).template cast<cldouble>();
			
			//cout << "Radial part matrix dimensions: " << out.rows() << ", " << out.cols() << std::endl;
			
			return out;
		}
		
		bdpsqft axialPartSq_impl(std::integral_constant<axis,axis::t> c, double t, int l) const {
			bdpsqft out(8,1);
			using Scalar = double;
			cdouble phi = cdouble(0,0);
			int sgn;
			switch(l%2) {
				case 0:
					sgn = (1-2*((l/2)%2));
				case 1:
					sgn = (1-2*(((l-1)/2)%2));
			}
			
			//cout << "Call to axialPartSq<time>, t = " << t << ", l = " << l << ", sgn: " << sgn << ", l%2: " << l%2 << "Ca t: " << ((phi + t) * (Scalar) (4 * PI / T - 2* omega)) << "Ca: " << (Scalar) (4 * PI / T - 2* omega) << "\n"
			//<< "T: " << T << "omega: " << omega << "\n";
			//#pragma message("Using dipolized beyond dipole pulse")
			
			
			switch(l%2) {
				case 0:
					out(0,0) =  3./16. * sgn * ((Scalar) (pow(E0/omega,2)) * (l==0)								   		      );
					out(1,0) =  1./16. * sgn * ((Scalar) (pow(E0/omega,2)) * cos((phi + t) * (Scalar) (4 * PI / T)           ));
					out(2,0) = -1./4.  * sgn * ((Scalar) (pow(E0/omega,2)) * cos((phi + t) * (Scalar) (2 * PI / T)           ));
					out(3,0) =  1./32. * sgn * ((Scalar) (pow(E0/omega,2)) * cos((phi + t) * (Scalar) (4 * PI / T - 2* omega)));
					out(4,0) = -1./8.  * sgn * ((Scalar) (pow(E0/omega,2)) * cos((phi + t) * (Scalar) (2 * PI / T - 2* omega)));
					out(5,0) =  3./16. * sgn * ((Scalar) (pow(E0/omega,2)) * cos((phi + t) * (Scalar)               2* omega ));
					out(6,0) = -1./8.  * sgn * ((Scalar) (pow(E0/omega,2)) * cos((phi + t) * (Scalar) (2 * PI / T + 2* omega)));
					out(7,0) =  1./32. * sgn * ((Scalar) (pow(E0/omega,2)) * cos((phi + t) * (Scalar) (4 * PI / T + 2* omega)));
					break;
				case 1:
					out(0,0) =  0																				   		       ;
					out(1,0) =  1./16. * sgn * ((Scalar) (pow(E0/omega,2)) * sin((phi + t) * (Scalar) (4 * PI / T)           ));
					out(2,0) = -1./4.  * sgn * ((Scalar) (pow(E0/omega,2)) * sin((phi + t) * (Scalar) (2 * PI / T)           ));
					out(3,0) =  1./32. * sgn * ((Scalar) (pow(E0/omega,2)) * sin((phi + t) * (Scalar) (4 * PI / T - 2* omega)));
					out(4,0) = -1./8.  * sgn * ((Scalar) (pow(E0/omega,2)) * sin((phi + t) * (Scalar) (2 * PI / T - 2* omega)));
					out(5,0) =  3./16. * sgn * ((Scalar) (pow(E0/omega,2)) * sin((phi + t) * (Scalar)               2* omega ));
					out(6,0) = -1./8.  * sgn * ((Scalar) (pow(E0/omega,2)) * sin((phi + t) * (Scalar) (2 * PI / T + 2* omega)));
					out(7,0) =  1./32. * sgn * ((Scalar) (pow(E0/omega,2)) * sin((phi + t) * (Scalar) (4 * PI / T + 2* omega)));
					break;
			}
			/*
			switch(l%2) {
				case 0:
					out(0,0) =  0; //((Scalar) (pow(E0/omega,2)) * pow(sin(PI/T * t),4) * pow(cos(omega * t),2) * (l==0));
					out(1,0) =  0; // 1./16. * sgn * ((Scalar) (pow(E0/omega,2)) * cos((phi + t) * (Scalar) (4 * PI / T)           ));
					out(2,0) =  0; //-1./4.  * sgn * ((Scalar) (pow(E0/omega,2)) * cos((phi + t) * (Scalar) (2 * PI / T)           ));
					out(3,0) =  0; // 1./32. * sgn * ((Scalar) (pow(E0/omega,2)) * cos((phi + t) * (Scalar) (4 * PI / T - 2* omega)));
					out(4,0) =  0; //-1./8.  * sgn * ((Scalar) (pow(E0/omega,2)) * cos((phi + t) * (Scalar) (2 * PI / T - 2* omega)));
					out(5,0) =  0; // 3./16. * sgn * ((Scalar) (pow(E0/omega,2)) * cos((phi + t) * (Scalar)             - 2* omega ));
					out(6,0) =  0; //-1./8.  * sgn * ((Scalar) (pow(E0/omega,2)) * cos((phi + t) * (Scalar) (2 * PI / T + 2* omega)));
					out(7,0) =  0; // 1./32. * sgn * ((Scalar) (pow(E0/omega,2)) * cos((phi + t) * (Scalar) (4 * PI / T + 2* omega)));
					break;
				case 1:
					out(0,0) =  0																				   		       ;
					out(1,0) =  0; // 1./16. * sgn * ((Scalar) (pow(E0/omega,2)) * sin((phi + t) * (Scalar) (4 * PI / T)           ));
					out(2,0) =  0; //-1./4.  * sgn * ((Scalar) (pow(E0/omega,2)) * sin((phi + t) * (Scalar) (2 * PI / T)           ));
					out(3,0) =  0; // 1./32. * sgn * ((Scalar) (pow(E0/omega,2)) * sin((phi + t) * (Scalar) (4 * PI / T - 2* omega)));
					out(4,0) =  0; //-1./8.  * sgn * ((Scalar) (pow(E0/omega,2)) * sin((phi + t) * (Scalar) (2 * PI / T - 2* omega)));
					out(5,0) =  0; // 3./16. * sgn * ((Scalar) (pow(E0/omega,2)) * sin((phi + t) * (Scalar)             - 2* omega ));
					out(6,0) =  0; //-1./8.  * sgn * ((Scalar) (pow(E0/omega,2)) * sin((phi + t) * (Scalar) (2 * PI / T + 2* omega)));
					out(7,0) =  0; // 1./32. * sgn * ((Scalar) (pow(E0/omega,2)) * sin((phi + t) * (Scalar) (4 * PI / T + 2* omega)));
					break;
			}
			*/
			
			//cout << out;

			return out;
		}
		
		template <typename MatrixType>
		bdpsqft axialPartSq_impl(std::integral_constant<axis,axis::radial> c, const Eigen::MatrixBase<MatrixType>& x,int l) const {
			bdpsqft out(8,x.size());
			
			double k = omega/SoL;
			
			//Bessel function must be evaluated on x for each x, but this is expensive. Need to consider options.
			beyondDipolePulse::l = l;
			
			//Bessel function j(x) is only defined for x>0, causing issues when 2k > 2PI/cT. Solution: sgnfix 
			int sgnfix2 = (2*PI/(SoL * T) > 2*k) - (2*PI/(SoL * T) < 2*k);
			int sgnfix4 = (4*PI/(SoL * T) > 2*k) - (4*PI/(SoL * T) < 2*k);
			
			int sgnc2 = 1;
			int sgnc4 = 1;
			
			//Sin(x) is antisymmetric and so when sgnfix == -1, the bessel function must also be negated
			if(sgnfix2 == -1 && l%2 == 1) {
				sgnc2 = -1;
			}
			
			if(sgnfix4 == -1 && l%2 == 1) {
				sgnc4 = -1;
			}
			
			//cout << "l: " << l << std::endl;
			
			/*
			out.row(0) = MatrixType::Constant(x.size(),1.0) * (l==0);
			out.row(1) = MatrixType::Constant(x.size(),0.0);// 		((			 (4*PI/(SoL * T)       ) * x.real()).unaryExpr(std::ref(besselJ))).template cast<cldouble>();
			out.row(2) = MatrixType::Constant(x.size(),0.0);// 		((			 (2*PI/(SoL * T)       ) * x.real()).unaryExpr(std::ref(besselJ))).template cast<cldouble>();
			out.row(3) = MatrixType::Constant(x.size(),0.0);//sgnc4 *	((sgnfix4 *  (4*PI/(SoL * T) - 2*k ) * x.real()).unaryExpr(std::ref(besselJ))).template cast<cldouble>();
			out.row(4) = MatrixType::Constant(x.size(),0.0);//sgnc2 * ((sgnfix2 *  (2*PI/(SoL * T) - 2*k ) * x.real()).unaryExpr(std::ref(besselJ))).template cast<cldouble>();
			out.row(5) = MatrixType::Constant(x.size(),0.0);// 		((			 (                 2*k ) * x.real()).unaryExpr(std::ref(besselJ))).template cast<cldouble>();
			out.row(6) = MatrixType::Constant(x.size(),0.0);// 		((			 (2*PI/(SoL * T) + 2*k ) * x.real()).unaryExpr(std::ref(besselJ))).template cast<cldouble>();
			out.row(7) = MatrixType::Constant(x.size(),0.0);// 		((			 (4*PI/(SoL * T) + 2*k ) * x.real()).unaryExpr(std::ref(besselJ))).template cast<cldouble>();
			*/
			
			
			out.row(0) = MatrixType::Constant(x.size(),1.0) * (l==0);
			out.row(1) = 		((			 (4*PI/(SoL * T)       ) * x.real()).unaryExpr(std::ref(besselJ))).template cast<cldouble>();
			out.row(2) = 		((			 (2*PI/(SoL * T)       ) * x.real()).unaryExpr(std::ref(besselJ))).template cast<cldouble>();
			out.row(3) =sgnc4 *	((sgnfix4 *  (4*PI/(SoL * T) - 2*k ) * x.real()).unaryExpr(std::ref(besselJ))).template cast<cldouble>();
			out.row(4) =sgnc2 * ((sgnfix2 *  (2*PI/(SoL * T) - 2*k ) * x.real()).unaryExpr(std::ref(besselJ))).template cast<cldouble>();
			out.row(5) = 		((			 (                 2*k ) * x.real()).unaryExpr(std::ref(besselJ))).template cast<cldouble>();
			out.row(6) = 		((			 (2*PI/(SoL * T) + 2*k ) * x.real()).unaryExpr(std::ref(besselJ))).template cast<cldouble>();
			out.row(7) = 		((			 (4*PI/(SoL * T) + 2*k ) * x.real()).unaryExpr(std::ref(besselJ))).template cast<cldouble>();
			
			
			//cout << "Radial part matrix dimensions: " << out.rows() << ", " << out.cols() << std::endl;
			
			return out;
		}
		
		template <typename MatrixType>
		bdpft axialPart_impl(std::integral_constant<axis,axis::angular> c, const Eigen::MatrixBase<MatrixType>& m, int l) const {
			bdpft out(6,m.size());
			
			for(int i = 0; i < 6; i++) {
				for(int j = 0; j < m.size(); j++) {
					int mm = (int)m(j);
					if (abs(mm) <= l) {
						cdouble ylm = bmath::spherical_harmonic(l,mm,dirtheta,dirphi);
						if(ylm != cdouble(0,0)) {
							out(i,j) = ylm;
						}
					}
				}
			}
			return out;
		}
		
		template <axis Ax>
		bdpft axialPart_impl(std::integral_constant<axis,Ax> c, double a) const {
			return bdpft::Constant(6,1,1.0);
		}
		
		template <axis Ax, typename MatrixType>
		bdpft axialPart_impl(std::integral_constant<axis,Ax> c, const Eigen::MatrixBase<MatrixType>& a) const {
			return bdpft::Constant(6,a.size(),1.0);
		}
		
};

class beyondDipoleCounterPulse:public beyondDipolePulse {
	
	public:
		
		beyondDipoleCounterPulse(double E0, double omega, double N):beyondDipolePulse(E0, omega, N) {
			
		}
		
		bdpft axialPart_impl(std::integral_constant<axis,axis::t> c, double t) const {
			bdpft out(6,1);
			using Scalar = double;
			cdouble phi = cdouble(-PI/2,0);
			// cdouble phi = cdouble(0,0);//cdouble(-PI/2,0);
			
			//Row 3 originally -E0. Negated to counter issue in Bessel function expansion
			
			// out(0,0) = ((Scalar) (E0 / (omega) ) * pow(sin(PI / T * t),2) * sin(omega * t))/3;
			// out(1,0) = 0;
			// out(2,0) = ((Scalar) (E0 / (omega) ) * pow(sin(PI / T * t),2) * sin(omega * t))/3;
			// out(3,0) = 0;
			// out(4,0) = ((Scalar) (E0 / (omega) ) * pow(sin(PI / T * t),2) * sin(omega * t))/3;
			// out(5,0) = 0;
			
			
			out(0,0) = ((Scalar) -E0/(2*omega) * cos( phi + t * (Scalar) (2 * PI / T + omega)));
			out(1,0) = 0;//((Scalar) -E0/(4*omega) * sin( phi + t * (Scalar) (2 * PI / T + omega)));
			out(2,0) = ((Scalar) -E0/(2*omega) * cos(-phi + t * (Scalar) (2 * PI / T - omega)));
			out(3,0) = 0;//((Scalar)  E0/(4*omega) * sin(-phi + t * (Scalar) (2 * PI / T - omega)));
			out(4,0) = ((Scalar)  E0/(  omega) * cos( phi + t * (Scalar) omega));
			out(5,0) = 0;//((Scalar)  E0/(2*omega) * sin( phi + t * (Scalar) omega));
			
			return out;
		}
		
		template <typename MatrixType>
		bdpft axialPart_impl(std::integral_constant<axis,axis::t> c, const Eigen::MatrixBase<MatrixType>& t) const {
			using Scalar = cldouble; //Easiest fix for earlier mistake
			bdpft out(6,t.size());
			
			auto tt = t.col(0).template cast<cdouble>();
			
			cdouble phi = cdouble(0,0);
			
			
			// out.row(0) = ((Scalar) (E0 / (omega) ) * pow(sin(PI / T * tt),2) * cos(omega * tt));
			// out.row(1) = MatrixType::Constant(t.size(),0.0).real();
			// out.row(2) = MatrixType::Constant(t.size(),0.0).real();
			// out.row(3) = MatrixType::Constant(t.size(),0.0).real();
			// out.row(4) = MatrixType::Constant(t.size(),0.0).real();
			// out.row(5) = MatrixType::Constant(t.size(),0.0).real();
			
			//Row 3 originally -E0. Negated to counter issue in Bessel function expansion
			
			out.row(0) = ((Scalar) (-E0 / (2*omega) ) * cos(( phi + tt) * (Scalar) (2 * PI / T + omega)));
			out.row(1) = 0;//((Scalar) (-E0 / (4*omega) ) * sin(( phi + tt) * (Scalar) (2 * PI / T + omega)));
			out.row(2) = ((Scalar) (-E0 / (2*omega) ) * cos((-phi + tt) * (Scalar) (2 * PI / T - omega)));
			out.row(3) = 0;//((Scalar) ( E0 / (4*omega) ) * sin((-phi + tt) * (Scalar) (2 * PI / T - omega)));
			out.row(4) = ((Scalar) ( E0 / (  omega) ) * cos(( phi + tt) * (Scalar) omega));
			out.row(5) = 0;//((Scalar) ( E0 / (2*omega) ) * sin(( phi + tt) * (Scalar) omega));
			
			
			/*out.col(1) =  E0 / (4*omega) * cos(omega * t.col(0) + 2 * PI * t.col(0) / T);
			out.col(2) = -E0 / (4*omega) * sin(omega * t.col(0) - 2 * PI * t.col(0) / T);
			out.col(3) =  E0 / (4*omega) * cos(omega * t.col(0) - 2 * PI * t.col(0) / T);
			out.col(4) = -E0 / (4*omega) * sin(omega * t.col(0));
			out.col(5) =  E0 / (4*omega) * cos(omega * t.col(0));*/
			
			return out;
		}
		
		bdpft axialPart_impl(std::integral_constant<axis,axis::x> c, double x) const {
			bdpft out(6,1);
			
			out(0,0) = cos(omega/SoL*x + 2*PI*x/(SoL*T));
			out(1,0) = 0;//sin(omega/SoL*x + 2*PI*x/(SoL*T));
			out(2,0) = cos(omega/SoL*x - 2*PI*x/(SoL*T));
			out(3,0) = 0;//sin(omega/SoL*x - 2*PI*x/(SoL*T));
			out(4,0) = cos(omega/SoL*x);
			out(5,0) = 0;//sin(omega/SoL*x);
			
			return out;
		}
		
		template <typename MatrixType>
		bdpft axialPart_impl(std::integral_constant<axis,axis::x> c, const Eigen::MatrixBase<MatrixType>& x) const {
			bdpft out(6,x.size());
			
			cldouble k = omega/SoL;
			
			out.row(0) = cos(k * x + 2*PI * x / (SoL * T));
			out.row(1) = 0;//sin(k * x + 2*PI * x / (SoL * T));
			out.row(2) = cos(k * x - 2*PI * x / (SoL * T));
			out.row(3) = 0;//sin(k * x - 2*PI * x / (SoL * T));
			out.row(4) = cos(k * x);
			out.row(5) = 0;//sin(k * x);
			
			return out;
		}
		
		template <typename MatrixType>
		bdpft axialPart_impl(std::integral_constant<axis,axis::radial> c, const Eigen::MatrixBase<MatrixType>& x,int l) const {
			bdpft out(6,x.size());
			
			double k = omega/SoL;
			
			//Bessel function must be evaluated on x for each x, but this is expensive. Need to consider options.
			beyondDipolePulse::l = l;
			
			//cout << "l: " << l << std::endl;
			
			//Rows 2 and 3 must be negated to avoid issues with bessel function expansion
			
			
			// out.row(0) = MatrixType::Constant(x.size(),1.0).real() * (l==0);
			// out.row(1) = MatrixType::Constant(x.size(),1.0).real() * (l==0);
			// out.row(2) = MatrixType::Constant(x.size(),1.0).real() * (l==0);
			// out.row(3) = MatrixType::Constant(x.size(),1.0).real() * (l==0);
			// out.row(4) = MatrixType::Constant(x.size(),1.0).real() * (l==0);
			// out.row(5) = MatrixType::Constant(x.size(),1.0).real() * (l==0);
			
			
			out.row(0) = /*(long double) sqrt(4*PI) * /*(2*l+1)) * pow(cldouble(0,1),l  ) * */ (( (2*PI/(SoL * T) + k ) * x.real()/* MatrixType::Constant(x.size(),0.0).real()*/).unaryExpr(std::ref(besselJ))).template cast<cldouble>();
			out.row(1) = 0;///*(long double) sqrt(4*PI) * /*(2*l+1)) * pow(cldouble(0,1),l-1) * */ (( (2*PI/(SoL * T) + k ) * x.real()/* MatrixType::Constant(x.size(),0.0).real()*/).unaryExpr(std::ref(besselJ))).template cast<cldouble>();
			out.row(2) = /*(long double) sqrt(4*PI) * /*(2*l+1)) * pow(cldouble(0,1),l  ) * */ ((-(2*PI/(SoL * T) - k ) * x.real()/* MatrixType::Constant(x.size(),0.0).real()*/).unaryExpr(std::ref(besselJ))).template cast<cldouble>();
			out.row(3) = 0;///*(long double) sqrt(4*PI) * /*(2*l+1)) * pow(cldouble(0,1),l-1) * */ ((-(2*PI/(SoL * T) - k ) * x.real()/* MatrixType::Constant(x.size(),0.0).real()*/).unaryExpr(std::ref(besselJ))).template cast<cldouble>();
			out.row(4) = /*(long double) sqrt(4*PI) * /*(2*l+1)) * pow(cldouble(0,1),l  ) * */ ((                   k   * x.real()/* MatrixType::Constant(x.size(),0.0).real()*/).unaryExpr(std::ref(besselJ))).template cast<cldouble>();
			out.row(5) = 0;///*(long double) sqrt(4*PI) * /*(2*l+1)) * pow(cldouble(0,1),l-1) * */ ((                   k   * x.real()/* MatrixType::Constant(x.size(),0.0).real()*/).unaryExpr(std::ref(besselJ))).template cast<cldouble>();
			
			//cout << "Radial part matrix dimensions: " << out.rows() << ", " << out.cols() << std::endl;
			
			return out;
		}
		
		
		template <typename MatrixType>
		bdpft axialPart_impl(std::integral_constant<axis,axis::angular> c, const Eigen::MatrixBase<MatrixType>& m, int l) const {
			bdpft out(6,m.size());
			
			for(int i = 0; i < 6; i++) {
				for(int j = 0; j < m.size(); j++) {
					int mm = (int)m(j);
					if (abs(mm) <= l) {
						cdouble ylm = bmath::spherical_harmonic(l,mm,dirtheta,dirphi);
						if(ylm != cdouble(0,0)) {
							out(i,j) = ylm;
						}
					}
				}
			}
			return out;
		}
		
		template <axis Ax>
		bdpft axialPart_impl(std::integral_constant<axis,Ax> c, double a) const {
			return bdpft::Constant(6,1,1.0);
		}
		
		template <axis Ax, typename MatrixType>
		bdpft axialPart_impl(std::integral_constant<axis,Ax> c, const Eigen::MatrixBase<MatrixType>& a) const {
			return bdpft::Constant(6,a.size(),1.0);
		}
		
};

class dipolizedBeyondDipolePulse: public beyondDipolePulse {
	public:
		dipolizedBeyondDipolePulse(double E0, double omega, double N):beyondDipolePulse(E0, omega, N) {
			
		}
		template <typename MatrixType>
			bdpft axialPart_impl(std::integral_constant<axis,axis::radial> c, const Eigen::MatrixBase<MatrixType>& x,int l) const {
				bdpft out(6,x.size());
				
				double k = omega/SoL;
				
				//Bessel function must be evaluated on x for each x, but this is expensive. Need to consider options.
				beyondDipolePulse::l = l;
				
				//cout << "l: " << l << std::endl;
				
				out.row(0) = /*(long double) sqrt(4*PI) * /*(2*l+1)) * pow(cldouble(0,1),l  ) * */ (((2*PI/(SoL * T) + k ) * MatrixType::Constant(x.size(),0.0).real()).unaryExpr(std::ref(besselJ))).template cast<cldouble>();
				out.row(1) = /*(long double) sqrt(4*PI) * /*(2*l+1)) * pow(cldouble(0,1),l-1) * */ (((2*PI/(SoL * T) + k ) * MatrixType::Constant(x.size(),0.0).real()).unaryExpr(std::ref(besselJ))).template cast<cldouble>();
				out.row(2) = /*(long double) sqrt(4*PI) * /*(2*l+1)) * pow(cldouble(0,1),l  ) * */ (((2*PI/(SoL * T) - k ) * MatrixType::Constant(x.size(),0.0).real()).unaryExpr(std::ref(besselJ))).template cast<cldouble>();
				out.row(3) = /*(long double) sqrt(4*PI) * /*(2*l+1)) * pow(cldouble(0,1),l-1) * */ (((2*PI/(SoL * T) - k ) * MatrixType::Constant(x.size(),0.0).real()).unaryExpr(std::ref(besselJ))).template cast<cldouble>();
				out.row(4) = /*(long double) sqrt(4*PI) * /*(2*l+1)) * pow(cldouble(0,1),l  ) * */ ((                  k   * MatrixType::Constant(x.size(),0.0).real()).unaryExpr(std::ref(besselJ))).template cast<cldouble>();
				out.row(5) = /*(long double) sqrt(4*PI) * /*(2*l+1)) * pow(cldouble(0,1),l-1) * */ ((                  k   * MatrixType::Constant(x.size(),0.0).real()).unaryExpr(std::ref(besselJ))).template cast<cldouble>();
				
				//cout << "Radial part matrix dimensions: " << out.rows() << ", " << out.cols() << std::endl;
				
				return out;
			}
			
			
			bdpft axialPart_impl(std::integral_constant<axis,axis::t> c, double t) const {
				bdpft out(6,1);
				using Scalar = double;
				cdouble phi = cdouble(0,0);
				
				out(0,0) = ((Scalar) SoL * -E0/(4*omega) * cos((phi + t) * (Scalar) (2 * PI / T + omega )));
				out(1,0) = ((Scalar) SoL * -E0/(4*omega) * sin((phi + t) * (Scalar) (2 * PI / T + omega )));
				out(2,0) = ((Scalar) SoL * -E0/(4*omega) * cos((phi + t) * (Scalar) (2 * PI / T - omega )));
				out(3,0) = ((Scalar) SoL * -E0/(4*omega) * sin((phi + t) * (Scalar) (2 * PI / T - omega )));
				out(4,0) = ((Scalar) SoL *  E0/(2*omega) * cos((phi + t) * (Scalar) omega));
				out(5,0) = ((Scalar) SoL *  E0/(2*omega) * sin((phi + t) * (Scalar) omega));
				
				return out;
			}
			
			template <typename MatrixType>
			bdpft axialPart_impl(std::integral_constant<axis,axis::t> c, const Eigen::MatrixBase<MatrixType>& t) const {
				using Scalar = cldouble; //Easiest fix for earlier mistake
				bdpft out(6,t.size());
				
				auto tt = t.col(0).template cast<cdouble>();
				
				cdouble phi = cdouble(0,0);
				
				out.row(0) = ((Scalar) (SoL * -E0 / (4*omega) ) * cos((phi + tt) * (Scalar) (2 * PI / T + omega )));
				out.row(1) = ((Scalar) (SoL * -E0 / (4*omega) ) * sin((phi + tt) * (Scalar) (2 * PI / T + omega )));
				out.row(2) = ((Scalar) (SoL * -E0 / (4*omega) ) * cos((phi + tt) * (Scalar) (2 * PI / T - omega )));
				out.row(3) = ((Scalar) (SoL * -E0 / (4*omega) ) * sin((phi + tt) * (Scalar) (2 * PI / T - omega )));
				out.row(4) = ((Scalar) (SoL *  E0 / (2*omega) ) * cos((phi + tt) * (Scalar) omega));
				out.row(5) = ((Scalar) (SoL *  E0 / (2*omega) ) * sin((phi + tt) * (Scalar) omega));
				
				/*out.col(1) =  E0 / (4*omega) * cos(omega * t.col(0) + 2 * PI * t.col(0) / T);
				out.col(2) = -E0 / (4*omega) * sin(omega * t.col(0) - 2 * PI * t.col(0) / T);
				out.col(3) =  E0 / (4*omega) * cos(omega * t.col(0) - 2 * PI * t.col(0) / T);
				out.col(4) = -E0 / (4*omega) * sin(omega * t.col(0));
				out.col(5) =  E0 / (4*omega) * cos(omega * t.col(0));*/
				
				return out;
			}
};

class beyondDipoleCarrierPulse:public beyondDipolePulse {
	
	protected:
		double E0;
		double omega;
		double T;
		static int l;
		
		double dirtheta;
		double dirphi;
	
	public:
		
		static long double besselJ(long double x) {
			int ll = l;
			return bmath::sph_bessel(ll,x);		
		}
		beyondDipoleCarrierPulse():beyondDipolePulse() {};
		beyondDipoleCarrierPulse(double E0, double omega, double N):beyondDipolePulse(E0,omega,N) {
			this->E0 = E0;
			this->omega = omega;
			this->T = (double)N*2*PI / omega;
		}
		
		bdpft axialPart_impl(std::integral_constant<axis,axis::t> c, double t) const {
			bdpft out = bdpft::Zero(6,1);
			using Scalar = double;
			cdouble phi = cdouble(0,0);
			
			//Remaining rows are redundant and can be optimized out later
			out(0,0) = ((Scalar)  E0/(omega) * pow( sin(PI * t / T), 2) * cos((phi + t) * (Scalar) omega));
			out(1,0) = ((Scalar)  E0/(omega) * pow( sin(PI * t / T), 2) * sin((phi + t) * (Scalar) omega));
			
			return out;
		}
		
		template <typename MatrixType>
		bdpft axialPart_impl(std::integral_constant<axis,axis::t> c, const Eigen::MatrixBase<MatrixType>& t) const {
			using Scalar = cldouble; //Easiest fix for earlier mistake
			bdpft out = bdpft::Zero(6,t.size());
			
			auto tt = t.col(0).template cast<cdouble>();
			
			cdouble phi = cdouble(0,0);
			
			out.row(0) = ((Scalar) E0/(cldouble(omega,0)) * pow( sin(PI * t / T), 2) * cos((phi + tt) * (Scalar) omega));
			out.row(1) = ((Scalar) E0/(cldouble(omega,0)) * pow( sin(PI * t / T), 2) * sin((phi + tt) * (Scalar) omega));
			
			return out;
		}
		
		bdpft axialPart_impl(std::integral_constant<axis,axis::x> c, double x) const {
			bdpft out = bdpft::Zero(6,1);
			
			out(0,0) = cos(omega/SoL*x);
			out(1,0) = sin(omega/SoL*x);
			
			return out;
		}
		
		template <typename MatrixType>
		bdpft axialPart_impl(std::integral_constant<axis,axis::x> c, const Eigen::MatrixBase<MatrixType>& x) const {
			bdpft out = bdpft::Zero(6,x.size());
			
			cldouble k = omega/SoL;
			
			out.row(0) = cos(k * x);
			out.row(1) = sin(k * x);
			
			return out;
		}
		
		template <typename MatrixType>
		bdpft axialPart_impl(std::integral_constant<axis,axis::radial> c, const Eigen::MatrixBase<MatrixType>& x,int l) const {
			bdpft out = bdpft::Zero(6,x.size());
			
			double k = omega/SoL;
			
			//Bessel function must be evaluated on x for each x, but this is expensive. Need to consider options.
			beyondDipolePulse::l = l;
			
			//This feels unhinged
			
			out.row(0) =(( k * x.real()).unaryExpr(std::ref(besselJ))).template cast<cldouble>();
			out.row(1) =(( k * x.real()).unaryExpr(std::ref(besselJ))).template cast<cldouble>();
			
			//cout << "Radial part matrix dimensions: " << out.rows() << ", " << out.cols() << std::endl;
			
			return out;
		}
		
		
		template <typename MatrixType>
		bdpft axialPart_impl(std::integral_constant<axis,axis::angular> c, const Eigen::MatrixBase<MatrixType>& m, int l) const {
			bdpft out(6,m.size());
			
			for(int i = 0; i < 6; i++) {
				for(int j = 0; j < m.size(); j++) {
					int mm = (int)m(j);
					if (abs(mm) <= l) {
						cdouble ylm = bmath::spherical_harmonic(l,mm,dirtheta,dirphi);
						if(ylm != cdouble(0,0)) {
							out(i,j) = ylm;
						}
					}
				}
			}
			return out;
		}
		
		template <axis Ax>
		bdpft axialPart_impl(std::integral_constant<axis,Ax> c, double a) const {
			return bdpft::Constant(6,1,1.0);
		}
		
		template <axis Ax, typename MatrixType>
		bdpft axialPart_impl(std::integral_constant<axis,Ax> c, const Eigen::MatrixBase<MatrixType>& a) const {
			return bdpft::Constant(6,a.size(),1.0);
		}
		
};

#endif
/*
Glorified wrapper for an Eigen::vector object with complex coefficients. Mostly used for slight optimizations of inner products in non-orthogonal bases.
*/

#ifndef WAVEFUNC_H
#define WAVEFUNC_H

#include "defs.h"
#include "basis.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include "mpiFuncs.h"

Eigen::IOFormat saveformat(Eigen::FullPrecision,Eigen::DontAlignCols,", ","\n","(","),"," = npy.array((\n","\n))\n",' ');

using namespace std;
template <typename basistype>
class wavefunc {
	basis<basistype> * bs;
	
	public:
		bool cached = false;
		cmat coefs;
		cmat lcoefs;
		wavefunc();
		wavefunc(basis<basistype>& basis, const cmat& coefs);
		cmat operator*(const wavefunc& psi);
		wavefunc operator*(cdouble c);
		
		template <typename matrixType>
		cmat operator*(const Eigen::MatrixBase<matrixType>& psi) {
				if(!cached) {
		
				lcoefs = coefs.adjoint() * bs->Sm();
				return  lcoefs * psi;
			}
			else {
				return  lcoefs * psi;
			}
		}
		
		template <typename matrixType>
		cmat operator*(const Eigen::SparseMatrixBase<matrixType>& psi) {
				if(!cached) {
		
				lcoefs = coefs.adjoint() * bs->Sm();
				return  lcoefs * psi;
			}
			else {
				return  lcoefs * psi;
			}
		}
		
		wavefunc operator+(const wavefunc& psi);
		wavefunc operator[](int index);
		
		void normalize();
		csmat invnorm();
		
		
		void save(string filename, int t) {
			std::ofstream savefile(filename,ios::app);
			
			savefile << "\n wft[" << t << "]";
			savefile << coefs.format(saveformat);
			savefile.close();
		}
		
		
};


template <typename basistype>
wavefunc<basistype>::wavefunc(basis<basistype>& basis, const cmat& coefs) {
	bs = &basis;
	this->coefs = coefs;
}


template <typename basistype>
wavefunc<basistype>::wavefunc() {
	bs = NULL;
	coefs = cmat(0,0);
}


//Get individual wavefunc<basistype> from multi-state wavefunc

template <typename basistype>
wavefunc<basistype> wavefunc<basistype>::operator[](int index) {
	//cout << "retrieving wavefunc<basistype> state no. " << index << "\n";
	if(coefs.cols() == 1) {
		//cout << "Index operator of wavefunc<basistype> class does not work on single-state wavefunc<basistype> object\n";
		return *this;
	}
	else {
	
		return wavefunc(*bs, coefs.col(index));
	}
}


template <typename basistype>
cmat wavefunc<basistype>::operator*(const wavefunc& psi) {
	if(!cached) {
		
		lcoefs = bs->template matfree<S>(coefs).adjoint();
		// cout << "lcoefs dims = (" << lcoefs.rows() << ", " << lcoefs.cols() << ")" << endl;
		
		// cout << "psi.coefs dims = (" << psi.coefs.rows() << ", " << psi.coefs.cols() << ")" << endl;
		
		// cout << "lcoefs * psi.coefs = " << lcoefs * psi.coefs << endl;
		
		int lth0, lNth, ll0, lNl;
		
		bs->getLocalParams(lth0,lNth,ll0,lNl);
		
		// cout << "lth0, lNth = " << lth0 << ", " << lNth << endl;
		// cout << "bs->radqN() * (lNth - lth0) = " << bs->radqN() * (lNth - lth0) << endl;
		
		if((lcoefs.cols() == psi.coefs.rows()) && (lcoefs.cols() == (bs->radqN() * bs->angqN()))) {
			// cout << "Nonparallel dot product" << endl;
			return  lcoefs * psi.coefs;
		}
		else if((lcoefs.cols() == psi.coefs.rows()) && (lcoefs.cols() == (bs->radqN() * (lNth - lth0)))) {
			// cout << "Parallel dot product" << endl;
			cmat localDot = lcoefs * psi.coefs;
			// cout << "localDot = " << lcoefs * psi.coefs << endl;
			cmat dot;
			
			allreduceMat(localDot,dot);
			
			// cout << "dot = " << dot << endl;
			
			return dot;
		}
		else {
			cout << "wtf??"<< endl;
			cout << lcoefs.cols() << endl << psi.coefs.rows() << endl <<  (bs->radqN() * (lNth - lth0)) << endl;
			return  lcoefs * psi.coefs;
		}
	}
	else {
		return  lcoefs * psi.coefs;
	}
}


template <typename basistype>
wavefunc<basistype> wavefunc<basistype>::operator*(cdouble c) {
	cmat newcoefs = c * coefs;
	
	return wavefunc(*bs, newcoefs);
}


template <typename basistype>
wavefunc<basistype> wavefunc<basistype>::operator+(const wavefunc& psi) {
	cmat newcoefs = this->coefs + psi.coefs;
	
	return wavefunc(*bs, newcoefs);
}


template <typename basistype>
csmat wavefunc<basistype>::invnorm() {
	cmat psipsi = (*this)*(*this);
	
	if(psipsi.size() == 1) {
		csmat invnrm(1,1);
		invnrm.insert(0,0) = 1/sqrt(abs(psipsi(0,0)));
		
		return invnrm;
	}
	else {
		csmat invnrm(psipsi.rows(),psipsi.cols());
		
		for(int i = 0; i < psipsi.rows(); i++) {
			//cout << i << ": norm = " << psipsi(i,i) << ", inv norm = " << 1/sqrt(abs(psipsi(i,i))) << "\n";
			invnrm.insert(i,i) = 1/sqrt(abs(psipsi(i,i)));
		}
		
		return invnrm;
	}
}


template <typename basistype>
void wavefunc<basistype>::normalize() {
	csmat inrm = invnorm();
	
	if(inrm.coeffRef(0,0) == inrm.coeffRef(0,0)) {
		coefs = (inrm * coefs.transpose()).transpose();
	}
	else {
		cout << "Warning: Normalization failed. Using un-normalized wfunc \n";
	}
}

#endif
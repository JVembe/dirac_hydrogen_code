/*
The splineHandler class makes more sense when the Schrödinger code is also included, as it is a parent class of both the pure b-spline basis used in the Schrödinger picture and the dual kinetic balance basis used in the Dirac picture

The class contains functionality directly dealing with generating B-splines for use by other bases.
*/

#ifndef SPLINEHANDLER_H
#define SPLINEHANDLER_H

#include "defs.h"
#include <boost/math/special_functions/legendre.hpp>

class splineHandler {
	protected:
		int splinedeg;
		
		vec t;
		lvec lastx;
		
		int l;
		
		int dropfirst = 1;
		int droplast = 1;
		
		vector<clsmat> splineCaches;
		
		lvec glpch;
		clsmat wtmt;
		
		
		int Nsplines;
		
		void checkx(const lvec& x);
		//constructor and destructor
		
	public:
		splineHandler(vec& t, int splinedeg);
		splineHandler() {
			t = vec(0);
			splinedeg = 0;
		}
		
		lvec& glpts(int ll = -1);
		clsmat& splineCache(const lvec& x);
		clsmat& splineCache();
		clsmat& dSplineCache(const lvec& x, int dn);
		
		clsmat& wtmat();
		
		cvec fit(const vec& x, const vec& y) {
			
		}
		
		void clearCaches() {
			for(int i = 0; i < splineCaches.size(); i++) {
				splineCaches[i] = clsmat(0,0);
				wtmt = clsmat(0,0);
			}
		}
};


#endif
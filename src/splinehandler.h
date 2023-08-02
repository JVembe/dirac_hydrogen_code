#ifndef SPLINEHANDLER_H
#define SPLINEHANDLER_H

#include "defs.h"

class splineHandler {
	protected:
		int splinedeg;
		
		vec t;
		lvec lastx;
		
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
		
		lvec& glpts();
		clsmat& splineCache(const lvec& x);
		clsmat& splineCache();
		clsmat& dSplineCache(const lvec& x, int dn);
		
		clsmat& wtmat();
		
		cvec fit(const vec& x, const vec& y) {
			
		}
};


#endif
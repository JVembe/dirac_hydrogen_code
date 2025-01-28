/*
	GaDE - Gpu-accelerated solver for the time dependent Dirac Equation
	
    Copyright (C) 2025  Johanne Elise Vembe <johannevembe@gmail.com>
    Copyright (C) 2025  Marcin Krotkiewski <marcink@uio.no>
	Copyright (C) 2025  Hicham Agueny <hicham.agueny@uib.no>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

/*
The splineHandler class makes more sense when the Schrödinger code is also included, as it is a parent class of both the pure 
b-spline basis used in the Schrödinger picture and the dual kinetic balance basis used in the Dirac picture

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
#include "cwavefunc.h"

class splineInterpolator {
	splineHandler* splinesA;
	splineHandler* splinesB;
	
	public:
		splineInterpolator(splineHandler& A, splineHandler& B) {
			splinesA = &A;
			splinesB = &B;
		}
		
		void genInterpMat
};
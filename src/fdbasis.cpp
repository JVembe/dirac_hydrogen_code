#include "cWavefunc.h" 

fdbasis::fdbasis(double rmax, double dr, int nprec, std::vector<int> &brkpts) {
	this->rmax = rmax;
	this->dr = dr;
	
	
	this->nprec = nprec;
	this->ncomps = ncomps;
	
	this->brkpts = brkpts;

	this->nR = ncomps * (int)rmax/dr; 
}

int fdbasis::radqN() {
	return this->nR;
}

csmat* fdbasis::olmat() {
	if (Sm!=NULL) return Sm;
	else {
		Sm= new csmat;
		
		
		*Sm= dr * csmat.identity(nR,nR);
	}
}
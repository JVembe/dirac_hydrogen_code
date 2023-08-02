#include "dkbbasis.h"

int beyondDipolePulse::l = 0;

int main() {
	
	
	int Nsplines = 200;
	int Ntime = 8000;
	int Nkappa = 8;
	int Nmu = 0;
	int Nl = 0;
	double rBox = 30.0;
	Eigen::IOFormat outformat(Eigen::FullPrecision,Eigen::DontAlignCols,", ","\n","(","),"," = npy.array((\n","\n))\n",' ');
	
	vec t(Nsplines+1);
	vec y(Nsplines+1);
	for(int i = 0; i < Nsplines+1; i++) {
		t[i] = rBox/(Nsplines)*i;
	}
	
	
	
	dkbbasis dkbb(t,7);
	dkbb.setState(-1);
	lvec x = dkbb.glpts();
	
	
	clsmat& splch = dkbb.splineCache(x);
	clsmat& dsplch = dkbb.dSplineCache(x,1);
	clsmat& ddsplch = dkbb.dSplineCache(x,2);
	
	
	
	double dt = (0.6*PI)/Ntime;
	double T = 0;
	dkbb.prepbdpvecs(30, 6);
	dkbb.clvcs();
	
	
	
	//return 0;
	dkbb.dpam(1,1,1);
	dkbb.dpam(-1,1,1);
	
	
	cvec testvec =  cvec::Random(30*dkbb.radqN());
	
	cout << testvec << std::endl;
	
	dkbb.template precacheMatvec<dpa>(testvec,30,0);
}
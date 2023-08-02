#include "RtsMat.h"
#include "cWavefunc.h"
#include "bsplinebasis.h"
#include "dkbbasis.h"
#include "rthphbasis.h"
#include "wavefunc.h"
#include "hamiltonian.h"
#include "propagator.h"

#define INTENSITY 1000

int beyondDipolePulse::l = 1;

int main() {
	using dirbs = rthphbasis<dkbbasis,spnrbasis>;
	using dirwf = wavefunc<dirbs>;
	
	
	int Nsplines = 1400;
	int Ntime = 8000;
	int Nkappa = 25;
	int Nmu = 0;
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
	
	
	spnrbasis spnrb(Nkappa,Nmu);
	
	cmat bdpams = cmat::Zero(spnrb.angqN(),spnrb.angqN());
	
	beyondDipolePulse bdpp(INTENSITY,50,15);
	//dipolizedBeyondDipolePulse bdpp(10,50,15);
	
	bdpp.setTime(1.75);
	
	
	double dt = (0.6*PI)/Ntime;
	double T = 0;
	
	dirbs rthphb(dkbb,spnrb);
	

	vec angInit = vec::Zero(rthphb.angqN());
	angInit[0] = 1.0;
	
	rthphb.pruneUncoupled(angInit,true);
	
	dkbb.prepbdpvecs(rthphb.angqN(), spnrb.bdplmax());
	dkbb.clvcs();
	
	for(int i = 0; i < rthphb.angids.size(); i++) {
		cout << "(" << i << ", " << rthphb.angids[i] << ", " << ik(rthphb.angids[i])<< ", " << imu(rthphb.angids[i]) << ")," << std::endl;
	}
	
	//return 0;
	dkbb.dpam(1,1,1);
	dkbb.dpam(-1,1,1);
	for(int l = 0; l < spnrb.bdplmax(); l++) {
		dkbb.bdpam(1,1,1,l,bdpp);
		dkbb.bdpam(1,1,-1,l,bdpp);
	}
	
	using Htype = DiracBDP<dirbs>;
	//using Htype = Dirac<dirbs>;
	Htype H(rthphb,bdpp);
	//Htype H(rthphb,&dplA<15,50,INTENSITY>);
	H.Vfunc = &coloumb<1>;
	H.prepeigs(Nsplines,Nsplines/2);
	
	vector<vec>& evals = H.getevals();
	vector<dsmat>& evecs = H.getevecs();
	
	cout << "eigs prepped\n";
	
	dirwf psi1 = dirwf(rthphb,evecs[0].col(Nsplines + 5));
	psi1.normalize();
	cout << "psi1 " << psi1.coefs.format(outformat)<<"\n";
	//cout << "E1 = " << evals[0][Nsplines+5] << std::endl;;
	
	//return 0;
	
	RtsMat<Htype> proptest;
	proptest.setDt(dt);
	proptest.setTime(T);
	proptest.attachHamiltonian(H);

	
	cvec b = H.S(psi1.coefs) - dt * cdouble(0,0.5) * H.H(T,psi1.coefs);
	
	Eigen::BiCGSTAB<RtsMat<Htype >,SubmatPreconditioner<cdouble> > solver;
	
	
	H.getBasis().getRadial().setState(0);
	
	solver.preconditioner().setup(H,dt);

	solver.compute(proptest);
	
	Cranknich<Htype,dirbs,true> cnp(H);
	
	//Cranknich<Htype,dirbs> cnp(H);
	
	
	cnp.proptest = &proptest;
	cnp.solver = &solver;
	
	
	
	cnp.propagate(psi1,(0.6*PI)/Ntime,Ntime);
	
	dirwf wft = cnp.wf[Ntime - 1];
	
	cout << "psit " << wft.coefs.format(outformat) << "\n";
	
	for(int i = 0; i < evecs.size(); i++) {
		cout << "evl["<<i<<"]" << evals[i].format(outformat) << std::endl;
		cout << "psievBDP["<<i<<"] " << cmat(wft * evecs[i]).format(outformat) << std::endl;
	}
	
	cmat projt = psi1 * cnp.wf;
	cout << "projt " << projt.format(outformat) << "\n";
	
	mat dPdE = H.dPdE(wft,5000,-0.500007,500);
	
	cout << "dPdEbdp_E" << INTENSITY << "R" << Nsplines << "K" << Nkappa << "M" << Nmu << "r" << rBox << "T" << Ntime << dPdE.format(outformat) << "\n";
	
	return 0;
}
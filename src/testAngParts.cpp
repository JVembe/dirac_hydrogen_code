#include "RtsMat.h"
#include "cWavefunc.h"
#include "rthphbasis.h"
#include "wavefunc.h"
#include "hamiltonian.h"
#include "propagator.h"

int beyondDipolePulse::l = 1;

int main() {
	using dirbs = rthphbasis<dkbbasis,spnrbasis>;
	using dirwf = wavefunc<dirbs>;
	
	int Nsplines = 200;
	int Ntime = 4000;
	double rBox = 40.0;
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
	
	
	spnrbasis spnrb(2);
	
	cmat bdpams = cmat::Zero(spnrb.angqN(),spnrb.angqN());
	
	beyondDipolePulse bdpp(10,50,15);
	
	bdpp.setTime(1.75);
	
	
	double dt = (0.6*PI)/Ntime;
	double T = 0;
	
	dirbs rthphb(dkbb,spnrb);
	

	vec angInit(rthphb.angqN());
	angInit.setZero();
	angInit[0] = 1.0;

	dkbb.dpam(1,1,1);
	dkbb.dpam(1,1,-1);
	
	for(int l = 0; l < spnrb.bdplmax(); l++) {
		dkbb.bdpam(1,1,1,l,bdpp);
		dkbb.bdpam(1,1,-1,l,bdpp);
	}
	
	using HBDPtype = DiracBDP<dirbs>;
	HBDPtype HBDP(rthphb,bdpp);
	HBDP.Vfunc = &coloumb<1>;
	HBDP.prepeigs(Nsplines,Nsplines/2);

	using HDPtype = Dirac<dirbs>;
	HDPtype HDP(rthphb,&dplA<15,50,10>);
	HDP.Vfunc = &coloumb<1>;
	HDP.prepeigs(Nsplines,Nsplines/2);	
	
	vector<vec>& evals = HDP.getevals();
	vector<dsmat>& evecs = HDP.getevecs();
	
	
	dirwf psi1 = dirwf(rthphb,evecs[0].col(Nsplines + 5));
	psi1.normalize();
	
	
	/*
	RtsMat<HDPtype> proptestDP;
	proptestDP.setDt(dt);
	proptestDP.setTime(T);
	proptestDP.attachHamiltonian(HDP);

	
	Eigen::BiCGSTAB<RtsMat<HDPtype >,SubmatPreconditioner<cdouble> > solverDP;
	
	
	HDP.getBasis().getRadial().setState(0);
	
	solverDP.preconditioner().setup(HDP,dt);

	solverDP.compute(proptestDP);
	
	Cranknich<HDPtype,dirbs,true> cnpDP(HDP);
	
	cnpDP.proptest = &proptestDP;
	cnpDP.solver = &solverDP;
	
	*/
	RtsMat<HBDPtype> proptestBDP;
	proptestBDP.setDt(dt);
	proptestBDP.setTime(T);
	proptestBDP.attachHamiltonian(HBDP);

	
	Eigen::BiCGSTAB<RtsMat<HBDPtype >,SubmatPreconditioner<cdouble> > solverBDP;
	
	
	HBDP.getBasis().getRadial().setState(0);
	
	solverBDP.preconditioner().setup(HBDP,dt);

	solverBDP.compute(proptestBDP);
	
	Cranknich<HBDPtype,dirbs,true> cnpBDP(HBDP);
	
	cnpBDP.proptest = &proptestBDP;
	cnpBDP.solver = &solverBDP;
	
	
	
	/*
	cnpDP.propagate(psi1,(0.6*PI)/Ntime,Ntime);
	
	
	dirwf wftDP = cnpDP.wf[Ntime - 1];
	
	
	
	cout << "psitDP " << wftDP.coefs.format(outformat) << "\n";
	
	cmat projtDP = psi1 * cnpDP.wf;
	cout << "projtDP " << projtDP.format(outformat) << "\n";
	
	for(int i = 0; i < evecs.size(); i++) {
		cout << "psievDP["<<i<<"] " << cmat(wftDP * evecs[i]).format(outformat) << std::endl;
	}
	
	mat dPdEDP = HDP.dPdE(wftDP,3000,-0.500007,300);
	
	cout << "dPdEDP " << dPdEDP.format(outformat) << "\n";
	
	return 0;
	*/
	
	cnpBDP.propagate(psi1,(0.6*PI)/Ntime,Ntime);
	
	dirwf wftBDP = cnpBDP.wf[Ntime - 1];
	
	cout << "psitBDP " << wftBDP.coefs.format(outformat) << "\n";
	
	cmat projtBDP = psi1 * cnpBDP.wf;
	cout << "projtBDP " << projtBDP.format(outformat) << "\n";
	
	for(int i = 0; i < evecs.size(); i++) {
		cout << "evl["<<i<<"]" << evals[i].format(outformat) << std::endl;
		cout << "psievBDP["<<i<<"] " << cmat(wftBDP * evecs[i]).format(outformat) << std::endl;
	}
	
	mat dPdEBDP = HBDP.dPdE(wftBDP,3000,-0.500007,300);
	
	cout << "dPdEBDP " << dPdEBDP.format(outformat) << "\n";
	
	return 0;
}
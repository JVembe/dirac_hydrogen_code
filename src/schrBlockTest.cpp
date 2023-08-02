#include "RtsMat.h"
#include "bsplinebasis.h"
#include "dkbbasis.h"
#include "rthphbasis.h"
#include "wavefunc.h"
#include "hamiltonian.h"
#include "propagator.h"

#include <chrono>
#include <sstream>
#include <fstream>

#define INTENSITY 10

int beyondDipolePulse::l = 1;

int main() {
	using rthbtype = rthphbasis<bsbasis,spharmbasisLM>;
	using wftype = wavefunc<rthbtype>;
	using dirbs = rthphbasis<dkbbasis,spnrbasis>;
	using dirwf = wavefunc<dirbs>;
	
	int Nsplines = 250;
	int Ntime = 4000;
	int NL = 6;
	int NM = 0;
	double rBox = 30.0;
	
	Eigen::initParallel();
	
	vec t(Nsplines+1);
	vec y(Nsplines+1);
	for(int i = 0; i < Nsplines+1; i++) {
		t[i] = rBox/(Nsplines)*i;
	}
	
	std::stringstream fnpstream;
	
	// fnpstream << "schrDp_E" << INTENSITY << "R" << Nsplines << "L" << NL << "M" << NM << "r" << rBox << "T" << Ntime;
	
	string filenamePrefix = fnpstream.str();
	
	// spharmbasisLM sphb(NL);
	spharmbasisLM sphb(NL);
	spnrbasis spnrb(NL);
	
	sphb.pzmat();
	spnrb.dpam( 1);
	spnrb.dpam(-1);
	
	// sphb.bdplOverride(10);
	//cout << sphb.Sm();
	
	//cout << "pzmL" << cmat(sph.pzm()).format(outformat);
	//cout << "pzm" << cmat(sphb.pzm()).format(outformat);
	//cout << "lsqm" << cmat(sphb.Lsqm()).format(outformat);
	// cout << "pzm" << cmat(sphb.pzm()).format(outformat);
	//cout << "dpam" << cmat(spnrb.dpam(1) + spnrb.dpam(-1)).format(outformat);
	/*
	beyondDipolePulse bdpp(10,50,15);
	
	for(int i = 0; i < sphb.bdplmax(); i++) {
		sphb.bdpAsqm(i);
		cout << i << "\n";
	}
	//return 0;
	*/
	
	cmat bdpam = cmat::Identity(spnrb.angqN(),spnrb.angqN());
	
	// for(int l = 0; l < spnrb.bdplmax(); l++) {
		// bdpam += spnrb.bdpam(1,l);
		// bdpam += spnrb.bdpam(-1,l);
	// }
	
	// cout << "bdpathm " << bdpam.format(outformat);
	
	// return 0;
	
	bsbasis bsb(t,7);
	bsb.setState(0);
	lvec x = bsb.glpts();
	// cout << x;
	
	// cout << "S" << cmat(bsb.Sm()).format(outformat);
	
	
	clsmat& splch = bsb.splineCache(x);
	clsmat& dsplch = bsb.dSplineCache(x,1);
	clsmat& ddsplch = bsb.dSplineCache(x,2);
	
	dkbbasis dkbb(t,7);
	dkbb.setState(-1);
	lvec dkbx = dkbb.glpts();
	
	
	clsmat& dkbsplch = dkbb.splineCache(x);
	clsmat& dkbdsplch = dkbb.dSplineCache(x,1);
	clsmat& dkbddsplch = dkbb.dSplineCache(x,2);
	
	
	cmat bdpams = cmat::Zero(sphb.angqN(),sphb.angqN());
	
	beyondDipolePulse bdpp(10,50,15);
	
	bdpp.setTime(1.75);
	
	
	double dt = (0.6*PI)/Ntime;
	double T = 0;
	
	rthbtype rthphb(bsb,sphb);
	
	dirbs rthphb2(dkbb,spnrb);
	
	
	//bsb.prepbdpvecs(rthphb.angqN(), spnrb.bdplmax());
	
	vec angInit = vec::Zero(rthphb.angqN());
	angInit[0] = 1.0;
	// cout << "aqn pre prune: " << rthphb.angqN();
	// rthphb.pruneUncoupled(angInit);

	// cout << "aqn post prune: " << rthphb.angqN();
	
	iarray angids = iarray::Zero(rthphb.angids.size());
	
	for(int i = 0; i < rthphb.angids.size(); i++) {
		// cout << rthphb.angids[i] << std::endl;
		angids[i] = rthphb.angids[i];
	}
	
	// cout << angids << std::endl;
	
	// cout << il(angids) << std::endl;
	
	
	vec angInit2 = vec::Zero(rthphb2.angqN());
	angInit2[0] = 1.0;
	
	rthphb2.pruneUncoupled(angInit2);
	
	iarray angids2 = iarray::Zero(rthphb2.angids.size());
	
	for(int i = 0; i < rthphb2.angids.size(); i++) {
		// cout << rthphb.angids[i] << std::endl;
		angids2[i] = rthphb2.angids[i];
	}
	
	// cout << ik(angids2) << std::endl;
	
	// return 0;
	// for(int l = 0; l < spnrb.bdplmax(); l++) {
		// bsb.bdpam(1,1,1,l,bdpp);
		// bsb.bdpam(1,1,-1,l,bdpp);
	// }
	//cout << bsb.splineCache(bsb.glpts());
	/*
	for(int i = 0; i < sphb.bdplmax(); i++) {
		cout << "bdpAsqm["<<i<<"]" << cmat(bsb.bdpAsqm(i,bdpp)).format(outformat);
	}
	*/
	for(int i = 0; i < sphb.bdplmax(); i++) {
		bsb.bdpApzm(i,4,1,bdpp);
	}
	
	bsb.Lm();
	bsb.Lsqm();
	bsb.ddm(1);
	bsb.ddm(2);
	bsb.Em(&coloumb<1>);
	//cout << ((bsbasis*)(&rthphb.radial())) -> vExt << std::endl;
	//cout << bsb.vExt << std::endl;
	
	//return 0;
	/*
	for(int i = 0; i < sphb.bdplmax(); i++) {
		cout << bdpp.template axialPartSq<axis::t>(dt,i) << "\n\n";
	}
	*/
	//bsb.getcachevec(sphb.angqN(),sphb.bdplmax());
	
	//bsb.getcachevec(sphb.angqN(),sphb.bdplmax(),1);
	dkbb.prepbdpvecs(rthphb.angqN(), spnrb.bdplmax());
	dkbb.clvcs();
	
	dkbb.dpam(1,1,1);
	dkbb.dpam(1,1,-1);
	
	dkbb.prepbdpvecs(rthphb.angqN(), spnrb.bdplmax());
	dkbb.clvcs();
	
	cvec testvec = cvec::Random(rthphb.radqN()*rthphb.angqN(),1.0);
	
	using Htype = Schrodinger<rthbtype>;
	// using Htype = Dirac<dirbs>;
	// Htype H(rthphb,bdpp);
	// Htype H(rthphb,bdpp);
	Htype H(rthphb,&dplA<15,50,INTENSITY>);
	// Htype H(rthphb2,&dplA<15,50,INTENSITY>);
	H.Vfunc = &coloumb<1>;
	//H2.Vfunc = &coloumb<1>;
	
	// cout << "OpenMP max threads: " << omp_get_max_threads() << std::endl;
	
	H.H0radprep();
	
	// cout << "H0r " << cmat(H.template H0<axis::radial>()).format(outformat);
	// return 0;
	H.prepeigs(Nsplines,Nsplines/2);
	
	
	cvec a;
	cvec b;
	auto t0 = std::chrono::high_resolution_clock::now();
	for(int i = 0; i < 10; i++) {
		cout << i << std::endl;
		a = rthphb.template matvec<H0>(testvec);
	}
	auto t1 = std::chrono::high_resolution_clock::now();
	for(int i = 0; i < 10; i++) {
		cout << i << std::endl;
		b = rthphb.h0matvecBlock(testvec);
	}
	auto t2 = std::chrono::high_resolution_clock::now();
	
	cout << "Apzv_old" << a.format(outformat) << std::endl;
	cout << "Apzv_block" << b.format(outformat) << std::endl;
	
	cout << "Time of old function call " << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() << std::endl;
	cout << "Time of new function " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;
	
	
	
	return 0;
}
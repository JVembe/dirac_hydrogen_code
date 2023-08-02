#include "RtsMat.h"
#include "bsplinebasis.h"
#include "rthphbasis.h"
#include "wavefunc.h"
#include "hamiltonian.h"
#include "propagator.h"

#include <chrono>
#include <sstream>
#include <fstream>

#define INTENSITY 400

int beyondDipolePulse::l = 1;

void printSparseNonZeros(const dsmat& mat) {
	cout << " = npy.array((\n";
	
	for(int i = 0; i < mat.outerSize(); i++) {
		for(dsmat::InnerIterator j(mat,i); j; ++j) {
			cout << "(" << j.value();
			
			//if(j.value().imag() >= 0)
			//	cout << " + ";
			
			cout //<< j.value().imag() 
			<< "," << j.row() << ", " << j.col() << "),\n";
		}
	}
	
	cout << "))\n";
}

void printSparseNonZeros(const csmat& mat) {
	cout << " = npy.array((\n";
	
	for(int i = 0; i < mat.outerSize(); i++) {
		for(csmat::InnerIterator j(mat,i); j; ++j) {
			cout << "(" << j.value().real();
			
			if(j.value().imag() >= 0)
				cout << " + ";
			
			cout << j.value().imag() << "j," << j.row() << ", " << j.col() << "),\n";
		}
	}
	
	cout << "))\n";
}

int main() {
	using rthbtype = rthphbasis<bsbasis,spharmbasisLM>;
	using wftype = wavefunc<rthbtype>;
	
	int Nsplines = 500;
	int Ntime = 6000;
	int NL = 30;
	int NM = 0;
	double rBox = 50.0;
	
	vec t(Nsplines+1);
	vec y(Nsplines+1);
	for(int i = 0; i < Nsplines+1; i++) {
		t[i] = rBox/(Nsplines)*i;
	}
	
	std::stringstream fnpstream;
	
	fnpstream << "schrDp_E" << INTENSITY << "R" << Nsplines << "L" << NL << "M" << NM << "r" << rBox << "T" << Ntime;
	
	string filenamePrefix = fnpstream.str();
	
	spharmbasisLM sphb(NL);
	spharmbasis sph(4);
	spnrbasis spnrb(4);
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
	bsbasis bsb(t,7);
	bsb.setState(0);
	lvec x = bsb.glpts();
	// cout << x;
	
	// cout << "S" << cmat(bsb.Sm()).format(outformat);
	
	clsmat& splch = bsb.splineCache(x);
	clsmat& dsplch = bsb.dSplineCache(x,1);
	clsmat& ddsplch = bsb.dSplineCache(x,2);
	
	
	cmat bdpams = cmat::Zero(sphb.angqN(),sphb.angqN());
	
	//beyondDipolePulse bdpp(10,50,15);
	
	//bdpp.setTime(1.75);
	
	
	double dt = (0.6*PI)/Ntime;
	double T = 0;
	
	rthbtype rthphb(bsb,sphb);
	
	sphb.pzm();
	
	//bsb.prepbdpvecs(rthphb.angqN(), spnrb.bdplmax());
	
	// cout << sphb.angCouplings(false);
	
	vec angInit = vec::Zero(rthphb.angqN());
	angInit[0] = 1.0;
	
	
	
	cout << "aqn_pre_prune = " << rthphb.angqN() << std::endl;
	rthphb.pruneUncoupled(angInit);

	cout << "aqn_post_prune = " << rthphb.angqN() << std::endl;
	
	
	cout << "angidsSchrDp_L" << NL << " = npy.array((";
	for(int i = 0; i < rthphb.angids.size(); i++) {
		cout << "(" << i << ", " << rthphb.angids[i] << ", " << il(rthphb.angids[i])<< ", " << im(rthphb.angids[i]) << ")," << std::endl;
	}
	
	cout << "))\n";
	
	cout << "angidsReverseSchrDp_L" << NL << " = npy.array((";
	for(int i = 0; i < rthphb.angidsReverse.size(); i++) {
		cout << "(" << i << ", " << rthphb.angidsReverse[i] << "),\n";// " << il(rthphb.angidsReverse[i])<< ", " << im(rthphb.angidsReversed[i]) << ")," << std::endl;
	}
	
	cout << "))\n";
	
	/*
	for(int l = 0; l < spnrb.bdplmax(); l++) {
		bsb.bdpam(1,1,1,l,bdpp);
		bsb.bdpam(1,1,-1,l,bdpp);
	}
	*/
	//cout << bsb.splineCache(bsb.glpts());
	/*
	for(int i = 0; i < sphb.bdplmax(); i++) {
		cout << "bdpAsqm["<<i<<"]" << cmat(bsb.bdpAsqm(i,bdpp)).format(outformat);
	}
	
	for(int i = 0; i < sphb.bdplmax(); i++) {
		cout << "bdpApzm["<<i<<"]" << cmat(bsb.bdpApzm(i,4,1,bdpp)).format(outformat);
	}
	*/
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
	
	cvec testvec = cvec::Constant(rthphb.radqN()*rthphb.angqN(),1.0);
	
	cvec rtestvec = cvec::Constant(rthphb.radqN(),1.0);
	
	
	cvec testOut = cvec::Zero(rthphb.radqN());
	
	//bsb.template precacheMatfree<bdpApz>(testvec,rthphb.angqN(),sphb.bdplmax());
	//bsb.template cacheMatfree<bdpApz>(rtestvec,testOut,0,cdouble(1.0,0),1,0,1);
	
	
	//cout << "bR " << (dplA<15,50,10>(10) * bsb.template matfree<pz>(rtestvec) + bsb.template matfree<L>(rtestvec)).format(outformat);
	//cout << "b2R " << testOut.format(outformat);
	//return 0;
	
	//bsb.template precacheMatfree<bdpAsq>(testvec,rthphb.angqN(),sphb.bdplmax());
	//
	//bsb.template cacheMatfree<bdpAsq>(testvec.segment(0,bsb.radqN()),rtestvec,0,1.0,0);
	
	//cout << "Avr " << rtestvec << std::endl;
	
	//cout << "Av " << rthphb.template matfree<bdpAsq>(testvec).format(outformat);
	//cout << "Av " << rthphb.template matfree<bdpApz>(testvec).format(outformat);
	
	//return 0;
	//using Htype = SchrodingerBDP<rthbtype>;
	using Htype = Schrodinger<rthbtype>;
	// Htype H(rthphb,bdpp);
	//Htype H(rthphb,bdpp);
	Htype H(rthphb,&dplA<15,50,INTENSITY>);
	H.Vfunc = &coloumb<1>;
	//H2.Vfunc = &coloumb<1>;
	
	H.H0radprep();
	
	// csmat H0sadj = H.template H0<axis::radial>().selfadjointView<Eigen::Upper>();
	
	// cout << "Hv_adj" << H0sadj*rtestvec << std::endl;
	// cout << "Hv_old" << H.template H0<axis::radial>() * rtestvec << std::endl;
	
	// for(int i = 0; i < 24; i++) {
		// bsb.setState(i);
		// cout << bsb.H0m() << std::endl;
		
	// }
	//return 0;
	//return 0;
	// auto t0 = std::chrono::system_clock::now();
	
	// for(int i = 0; i < 100; i++) {
		// cvec v_old = H.H0(testvec);
	// }
	
	// auto t1 = std::chrono::system_clock::now();
	
	// for(int i = 0; i < 100; i++) {
		// cvec v_opt = rthphb.template matvec<H0>(testvec);
	// }
	
	// auto t2 = std::chrono::system_clock::now();
	
	// cout << "v_old execution time: " << std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count() << "\n";
	// cout << "v_opt execution time: " << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() << "\n";
	
	//return 0;
	
	H.prepeigs(Nsplines,Nsplines/2);
	vector<vec>& evals = H.getevals();
	vector<dsmat>& evecs = H.getevecs();
	
	// for(int i = 0; i <= sphb.angMax(); i++) {
		// // cout << i << std::endl;
		// int idx = li(i);
		
		// cout << "evl["<<i<<"]" << evals[i].real().format(outformat) << std::endl;
		// cout << "evc["<<i<<"]";
		// printSparseNonZeros(evecs[i]);
		// cout << std::endl;
	// }
	
	// return 0;
	
	// cout << "eigs prepped\n";
	
	wftype psi1 = wftype(rthphb,evecs[0].col(0));
	psi1.normalize();
	//cout << "psi1 " << psi1.coefs.format(outformat)<<"\n";
	
	int Nr = rthphb.radqN();
	//cout << bsb.template matfree<L>(psi1.coefs.col(0).segment(0,Nr));
	
	//return 0;
	
	
	
	//cout << psi1.coefs.size() << std::endl;
	
	//cout << rthphb.psqm().rows() << ", " << rthphb.psqm().cols() << std::endl;
	
	bsb.setState(1);
	//cout << sphb.Lsqm() << std::endl;
	//cout << "tv " << testvec.format(outformat);
	
	//cout << "bm" << ((/*H.S() - cdouble(0,1.75/2) **/H.H0() * testvec)).format(outformat);

	
	//cvec b = H2.Ht(1.75,testvec);
	cvec b2 = H.H(1.75,testvec);
	
	//cout << "b " << b.format(outformat);
	
	cout << "b2 " << b2.format(outformat);
	
	//return 0;
	//cout << "thb" << cmat(sphb.pzm()).format(outformat);
	
	//cout << "bv" << b.format(outformat);
	
	
	//cout << "E1 = " << evals[0][Nsplines+5] << std::endl;;
	
	//cout << "DE estimate: " << H.sigmaE(psi1.coefs);
	
	//return 0;
	
	RtsMat<Htype> proptest;
	proptest.setDt(dt);
	proptest.setTime(T);
	proptest.attachHamiltonian(H);

	Eigen::BiCGSTAB<RtsMat<Htype >,SubmatPreconditioner<cdouble> > solver;
	//Eigen::GMRES<RtsMat<Htype >,SubmatPreconditioner<cdouble> > solver;
	
	
	H.getBasis().getRadial().setState(0);
	
	solver.preconditioner().setup(H,dt);

	solver.compute(proptest);
	
	Cranknich<Htype,rthbtype,true> cnp(H);
	
	cnp.setDumpfile((filenamePrefix + "_dump"));
	
	cnp.proptest = &proptest;
	cnp.solver = &solver;
	
	cvec psi2 = solver.solve(b2);
	cout << "psi2 " << psi2.format(outformat);
	// return 0;
	
	cnp.propagate(psi1,(0.6*PI)/Ntime,Ntime,10);
	
	wftype wft = cnp.wf[Ntime/10 - 1];
	
	//cout << "psitt " << cnp.wf.coefs.format(outformat) << "\n";
	
	// cout << "psit " << wft.coefs.format(outformat) << "\n";
	
	ofstream psievf(filenamePrefix + "_psiev");
	
	psievf << "psit" << filenamePrefix << wft.coefs.format(outformat) << std::endl;
	
	psievf << "evl"   << filenamePrefix << " = npy.zeros((" << evecs.size() << "),dtype = object)\n";
	psievf << "psiev" << filenamePrefix << " = npy.zeros((" << evecs.size() << "),dtype = object)\n";
	
	for(int i = 0; i < evecs.size(); i++) {
		psievf << "evl" << filenamePrefix << "["<<i<<"]" << evals[i].format(outformat) << std::endl;
		psievf << "psiev" << filenamePrefix << "["<<i<<"] " << cmat(wft * evecs[i]).format(outformat) << std::endl;
	}
	
	psievf.close();
	
	cmat projt = psi1 * cnp.wf;
	cout << "projt " << projt.format(outformat) << "\n";
	
	ofstream dPdEf(filenamePrefix + "_dPdE");
	
	mat dPdE = H.dPdE(wft,5000,-0.500007,500);
	
	dPdEf << "dPdE" << filenamePrefix << dPdE.format(outformat);
	
	dPdEf.close();
	
	return 0;
}
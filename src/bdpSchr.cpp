#include "RtsMat.h"
#include "bsplinebasis.h"
#include "rthphbasis.h"
#include "wavefunc.h"
#include "hamiltonian.h"
#include "propagator.h"

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
	
	int Nsplines = 150;
	int Ntime = 4000;
	int NL = 4;
	int NM = 3;
	double rBox = 30.0;
	
	vec t(Nsplines+1);
	vec y(Nsplines+1);
	for(int i = 0; i < Nsplines+1; i++) {
		t[i] = rBox/(Nsplines)*i;
	}
	
	std::stringstream fnpstream;
	
	fnpstream << "schrBdp_E" << INTENSITY << "R" << Nsplines << "L" << NL << "M" << NM << "r" << rBox << "T" << Ntime;
	
	string filenamePrefix = fnpstream.str();
		
	spharmbasisLM sphb(NL,NM);
	//spharmbasis sph(NL);
	//spnrbasis spnrb(NL);
	//cout << sphb.Sm();
	
	
	// sphb.bdplOverride(10);
	
	//cout << "pzmL" << cmat(sph.pzm()).format(outformat);
	//cout << "pzm" << cmat(sphb.pzm()).format(outformat);
	//cout << "lsqm" << cmat(sphb.Lsqm()).format(outformat);
	//cout << "p1m" << cmat(sphb.p1m()).format(outformat);
	// cout << "dpam" << cmat(spnrb.dpam(1) + spnrb.dpam(-1)).format(outformat);
	
	beyondDipolePulse bdpp(INTENSITY,50,15);
	
	
	for(int i = 0; i < sphb.bdplmax(); i++) {
		sphb.bdpAsqm(i);
		sphb.bdpApxm(i,1);
		sphb.bdpApxm(i,-1);
	}
	
	// return 0;
	
	bsbasis bsb(t,7);
	bsb.setState(0);
	lvec x = bsb.glpts();
	//cout << x;
	
	//cout << bsb.Sm();
	
	clsmat& splch = bsb.splineCache(x);
	clsmat& dsplch = bsb.dSplineCache(x,1);
	clsmat& ddsplch = bsb.dSplineCache(x,2);
	
	
	cmat bdpams = cmat::Zero(sphb.angqN(),sphb.angqN());
	
	//beyondDipolePulse bdpp(10,50,15);
	
	bdpp.setTime(1.75);
	
	
	
	double dt = (0.6*PI)/Ntime;
	double T = 0;
	
	// cout << "Adp = npy.array((\n";
	// for(int i = 0; i < Ntime; i++) {
		// cout << dplA<15,50,INTENSITY>(i*dt) << ",";
	// }
	// cout << "))" << std::endl;
	
	// cout << "Abdp = npy.array((\n";
	// for(int i = 0; i < Ntime; i++) {
		// bdpft pT = bdpp.template axialPart<axis::t>(i*dt);
		// double at = pT(0,0).real() + pT(2,0).real() + pT(4,0).real();
		// cout << at << ",";
	// }
	// cout << "))";
	
	// return 0;
	
	rthbtype rthphb(bsb,sphb);
	
	vec angInit = vec::Zero(rthphb.angqN());
	angInit[0] = 1.0;
	
	rthphb.pruneUncoupled(angInit,true);
	
	cout << "angidsSchrBdp_L" << NL << "M" << NM << " = npy.array((";
	for(int i = 0; i < rthphb.angids.size(); i++) {
		cout << "(" << i << ", " << rthphb.angids[i] << ", " << il(rthphb.angids[i])<< ", " << im(rthphb.angids[i]) << ")," << std::endl;
	}
	
	cout << "))";
	
	cout << "angidsReverseSchrBdp_L" << NL << "M" << NM << " = npy.array((";
	for(int i = 0; i < rthphb.angidsReverse.size(); i++) {
		cout << "(" << i << ", " << rthphb.angidsReverse[i] << ")\n";// " << il(rthphb.angidsReverse[i])<< ", " << im(rthphb.angidsReversed[i]) << ")," << std::endl;
	}
	
	cout << "))";
	
	//return 0;
	
	rthphb.pzm();
	//bsb.prepbdpvecs(rthphb.angqN(), spnrb.bdplmax());

	
	//cout << bsb.splineCache(bsb.glpts());
	
	bsb.Em(&coloumb<1>);
	bsb.Lm();
	bsb.Lsqm();
	bsb.ddm(1);
	bsb.ddm(2);
	//cout << ((bsbasis*)(&rthphb.radial())) -> vExt << std::endl;
	//cout << bsb.vExt << std::endl;
	
	
	
	//return 0;
	//for(int i = 0; i < sphb.bdplmax(); i++) {
	//	cout << bdpp.template axialPartSq<axis::t>(dt,i) << "\n\n";
	//}
	
	bsb.getcachevec(sphb.angqN(),sphb.bdplmax());
	
	bsb.getcachevec(sphb.angqN(),sphb.bdplmax(),1);
	
	cvec testvec = cvec::Constant(rthphb.radqN()*rthphb.angqN(),1.0);
	
	cvec rtestvec = cvec::Constant(rthphb.radqN(),1.0);
	
	for(int i = 0; i < sphb.bdplmax(); i++) {
		//cout << "bdpAsqm["<<i<<"]" << cmat(bsb.bdpAsqm(i,bdpp)).format(outformat);
	}
	
	//return 0;
	
	for(int i = 0; i < sphb.bdplmax(); i++) {
		bsb.bdpApzm(i,4,1,bdpp);
		bsb.bdpApzm(i,4,-1,bdpp);
		bsb.bdpAsqm(i,bdpp);
	}
	
	
	// for(int i = 0; i < sphb.angqN(); i++) {
		// cout << "bdpApzvL["<<i<<"]" << (bsb.bdpApzm(0,i,-1,bdpp) * rtestvec).format(outformat);
		// cout << "dppzvL["<<i<<"]" << (dplA<15,50,INTENSITY>(1.75) * (bsb.ddm(1) + ((il(i)) * bsb.Lm())) * rtestvec).format(outformat);
		// cout << "bdpApzvU["<<i<<"]" << (bsb.bdpApzm(0,i,1,bdpp) * rtestvec).format(outformat);
		// cout << "dppzvU["<<i<<"]" << (dplA<15,50,INTENSITY>(1.75) * (bsb.ddm(1) - ((il(i) + 1) * bsb.Lm())) * rtestvec).format(outformat);
		
	// }
	
	// return 0;
	
	
	cvec testOut = cvec::Zero(rthphb.radqN());
	
	bsb.template precacheMatfree<bdpApz>(testvec,rthphb.angqN(),sphb.bdplmax());
	bsb.template cacheMatfree<bdpApz>(rtestvec,testOut,0,cdouble(1.0,0),1,0,1);
	

	bsb.template precacheMatfree<bdpAsq>(testvec,rthphb.angqN(),sphb.bdplmax());
	
	bsb.template cacheMatfree<bdpAsq>(testvec.segment(0,bsb.radqN()),rtestvec,0,1.0,0);
	//cout << "Avr " << rtestvec << std::endl;
	
	//return 0;
	
	//cout << "Asqv " << rthphb.template matfree<bdpAsq>(testvec).format(outformat);
	//cout << "Apzv " << rthphb.template matfree<bdpApz>(testvec).format(outformat);
	//cout << "Apzvdp " << rthphb.template matfree<pz>(testvec).format(outformat);
	
	//return 0;
	using Htype = SchrodingerBDP<rthbtype>;
	Htype H(rthphb,bdpp);
	//using HtypeDP = Schrodinger<rthbtype>;
	//HtypeDP HDP(rthphb,&dplA<15,50,INTENSITY>);
	
	H.Vfunc = &coloumb<1>;
	H.H0radprep();
	
	H.prepeigs(Nsplines,Nsplines/2);
	
	vector<vec>& evals = H.getevals();
	vector<dsmat>& evecs = H.getevecs();
	
	//cout << "eigs prepped\n";
	// cout << "S";
	// printSparseNonZeros(H.S());
	
	// cout << "Sdim = (" << rthphb.angqN() * rthphb.radqN() << ", " << rthphb.angqN() * rthphb.radqN() << ")" << std::endl;
	
	// cout << sphb.angMax() << std::endl;
	
	// for(int i = 0; i <= sphb.angMax(); i++) {
		// cout << i << std::endl;
		// int idx = li(i);
		
		// cout << "evl["<<i<<"]" << evals[idx].real().format(outformat) << std::endl;
		// cout << "evc["<<i<<"]";
		// printSparseNonZeros(evecs[idx]);
		// cout << std::endl;
	// }
	
	// cout << "evcdim = (" << rthphb.radqN() << ", " << rthphb.radqN() << ")" << std::endl;
	
	// return 0;
	
	wftype psi1 = wftype(rthphb,evecs[0].col(0));
	psi1.normalize();
	
	int Nr = rthphb.radqN();
	
	bsb.setState(1);
	
	cvec b2 = H.Ht(1.75,psi1.coefs);
	
	cout << "b2_dipolized4 " << b2.format(outformat);
	// return 0;
	//cout << "psi"<<1900<<"BDP"<< H.Ht(1900*dt,testvec).format(outformat) << std::endl;
	//cout << "psi"<<1900<<"DP"<< HDP.Ht(1900*dt,testvec).format(outformat) << std::endl;

	//return 0;
	//cout << "thb" << cmat(sphb.pzm()).format(outformat);
	
	//cout << "bv" << b.format(outformat);
	
	
	//cout << "E1 = " << evals[0][Nsplines+5] << std::endl;;
	
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
	
	cout << "psi2 " << solver.solve(b2).format(outformat);
	// return 0;
	//solver.setMaxIterations(20);
	
	Cranknich<Htype,rthbtype,true> cnp(H);
	
	cnp.proptest = &proptest;
	cnp.solver = &solver;
	
	cnp.setDumpfile((filenamePrefix + "_dump"));
	
	int NSkip = 10;
	
	cnp.propagate(psi1,(0.6*PI)/Ntime,Ntime,NSkip);
	
	wftype wft = cnp.wf[Ntime/NSkip - 1];
	
	//cout << "psitt " << 
	
	cout << "psit " << wft.coefs.format(outformat) << "\n";
	
	ofstream psievf(filenamePrefix + "_psiev");
	
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
	
	
	return 0;
}
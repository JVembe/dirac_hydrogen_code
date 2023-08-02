#include "RtsMat.h"
#include "cWavefunc.h"
#include "dkbbasis.h"
#include "rthphbasis.h"
#include "wavefunc.h"
#include "hamiltonian.h"
#include "propagator.h"
#include <wignerSymbols.h>

#define INTENSITY 100

int beyondDipolePulse::l = 1;

int main() {
	using dirbs = rthphbasis<dkbbasis,spnrbasis>;
	using dirwf = wavefunc<dirbs>;
	
	
	int Nsplines = 200;
	int Ntime = 16000;
	int Nkappa = 20;
	int Nmu = 0;
	int Nl = 0;
	double rBox = 30.0;
	Eigen::IOFormat outformat(Eigen::FullPrecision,Eigen::DontAlignCols,", ","\n","(","),"," = npy.array((\n","\n))\n",' ');
	Eigen::IOFormat outformatLine(Eigen::FullPrecision,Eigen::DontAlignCols,", "," ","(","),"," = npy.array((\n","\n))\n",' ');
	
	vec t(Nsplines+1);
	vec y(Nsplines+1);
	for(int i = 0; i < Nsplines+1; i++) {
		t[i] = rBox/(Nsplines)*i;
	}
	
	std::stringstream fnpstream;
	
	fnpstream << "dirBdp_E" << INTENSITY << "R" << Nsplines << "K" << Nkappa << "Mu" << Nmu << "r" << rBox << "T" << Ntime;
	
	string filenamePrefix = fnpstream.str();
	
	dkbbasis dkbb(t,7);
	dkbb.setState(-1);
	lvec x = dkbb.glpts();
	
	
	clsmat& splch = dkbb.splineCache(x);
	clsmat& dsplch = dkbb.dSplineCache(x,1);
	clsmat& ddsplch = dkbb.dSplineCache(x,2);
	
	
	spnrbasis spnrb(Nkappa,Nmu);
	// spnrb.bdplOverride(Nl);
	
	// cout << "dpaU" << cmat(spnrb.dpam(1)).format(outformat);
	// cout << "dpaL" << cmat(spnrb.dpam(-1)).format(outformat);
	
	// for(int l = 0; l < spnrb.bdplmax(); l++) {
		// cout << "bdpaU["<<l<<"]"<<cmat(spnrb.bdpam(1,l)).format(outformat);
		// cout << "bdpaL["<<l<<"]"<<cmat(spnrb.bdpam(-1,l)).format(outformat);
	// }
	
	//return 0;
	
	cmat bdpams = cmat::Zero(spnrb.angqN(),spnrb.angqN());
	
	//beyondDipoleCarrierPulse bdpcp(INTENSITY,50,15);
	beyondDipolePulse bdpp(INTENSITY,50,15);
	//dipolizedBeyondDipolePulse bdpp(10,50,15);
	
	//cout << bdpp.template axialPart<axis::radial>(x,0) << std::endl;
	//cout << bdpp.template axialPart_impl(std::integral_constant<axis,axis::radial> {}, x,0) << std::endl;
	
	// cout << bdpcp.template axialPart<axis::radial>(x,0) << std::endl;
	// cout << bdpcp.template axialPart_impl(std::integral_constant<axis,axis::radial> {}, x,0) << std::endl;
	
	//return 0;
	
	bdpp.setTime(1.75);
	
	
	double dt = (0.6*PI)/Ntime;
	double T = 0;
	
	dirbs rthphb(dkbb,spnrb);
	

	vec angInit = vec::Zero(rthphb.angqN());
	angInit[0] = 1.0;
	
	cout << rthphb.olmat().nonZeros();
	
	return 0;
	
	rthphb.pruneUncoupled(angInit,true);
	
	dkbb.prepbdpvecs(rthphb.angqN(), spnrb.bdplmax());
	dkbb.clvcs();
	
	for(int i = 0; i < rthphb.angids.size(); i++) {
		cout << "(" << i << ", " << rthphb.angids[i] << ", " << ik(rthphb.angids[i])<< ", " << imu(rthphb.angids[i]) << ")," << std::endl;
	}
	
	// cout << "Sr_data";
	// printSparseNonZeros(dkbb.Sm());
	
	//return 0;
	dkbb.dpam(1,1,1);
	dkbb.dpam(-1,1,1);
	for(int l = 0; l < spnrb.bdplmax(); l++) {
		dkbb.bdpam(1,1,1,l,bdpp);
		dkbb.bdpam(1,1,-1,l,bdpp);
	}
	
	dkbb.dpam(1,1,-1);
	dkbb.dpam(1,1, 1);
	
	int Nv = rthphb.angqN() * rthphb.radqN();
	
	cvec testvec = cvec::Constant(Nv,1.0);
	cvec rtestvec = cvec::Constant(dkbb.radqN(),1.0);
	cvec routvec = cvec::Zero(dkbb.radqN());
	
	int k1 = 10;
	int k2 = 14;
	int ul = -1;
	
	
	dkbb.precacheMatfree<bdpa>(testvec,rthphb.angqN(),spnrb.bdplmax());
	
	dkbb.dumpBdpVecs<dkbpart::pk0>();
	
	// dkbb2.cacheMatfree<bdpa>(testvec,routvec,k2,1.0,k1,k2,ul);
	
	return 0;
	
	// dkbb2.setState(k1,k2,ul);
	// //cout << "dpavLin" << (dkbb2.template matfree<dpa>(rtestvec)).format(outformat);
	// //cout << "dpavPar" << (routvec).format(outformat);
	
	// //return 0;
	
	// //cout << "vLin " << rthphb2.template matfree<dpa>(testvec).format(outformat);
	// //cout << "vPar " << rthphb2.parallel_dpa(testvec).format(outformat);
	
	
	
	
	// //return 0;
	
	
	// using Htype = DiracBDP<dirbs>;
	// Htype H(rthphb,bdpp);
	// using Htype2 = Dirac<dirbs>;
	// Htype2 H2(rthphb2,&dplA<15,50,400>);
	
	
	
	// H.Vfunc = &coloumb<1>;
	// H.prepeigs(Nsplines,Nsplines/2);
	// H2.Vfunc = &coloumb<1>;
	// H2.prepeigs(Nsplines,Nsplines/2);
	
	
	
	// //cout << "HtBDPv " << H.Ht(1.75,testvec);
	// //cout << "HtDPv " << H2.Ht(1.75,testvec).format(outformat);
	// for(int tt = 0; tt < Ntime/100; tt++) {
		// cout << "HtBDPv[" << tt << "] " << H.Ht(dt*100*tt,testvec).format(outformat);
	// }
	

	// return 0;

	// for(int i = 0; i < Ntime; i++) {
		// cout << i << std::endl;
		// H.H(i*dt,testvec);
	// }
	
	// return 0;

	// vector<vec>& evals = H.getevals();
	// vector<dsmat>& evecs = H.getevecs();
	
	// vector<vec>& evals2 = H2.getevals();
	// vector<dsmat>& evecs2 = H2.getevecs();
	
	// //cout << "eigs prepped\n";
	
	// dirwf psi1 = dirwf(rthphb,evecs[0].col(Nsplines + 5));
	// psi1.normalize();
	// cout << "psi1 " << psi1.coefs.format(outformat)<<"\n";
	
	// dirwf psi2 = dirwf(rthphb2,evecs2[0].col(Nsplines2 + 5));
	// psi2.normalize();
	// cout << "psi2 " << psi2.coefs.format(outformat)<<"\n";
	
	// return 0;
	
	
	
	// //cout << "NSv " << cmat(H.S(testvec)).format(outformat);
	
	
	// RtsMat<Htype> proptest;
	// proptest.setDt(dt);
	// proptest.setTime(T);
	// proptest.attachHamiltonian(H);

	// RtsMat<Htype2> proptest2;
	// proptest2.setDt(dt);
	// proptest2.setTime(T);
	// proptest2.attachHamiltonian(H2);

	
	// cvec b = H.S(psi1.coefs) - dt * cdouble(0,0.5) * H.H(T,psi1.coefs);
	// //cout << b;
	
	// cvec b2 = H2.S(psi1.coefs) - dt * cdouble(0,0.5) * H2.H(T,psi1.coefs);
	
	
	// Eigen::BiCGSTAB<RtsMat<Htype >,SubmatPreconditioner<cdouble> > solver;
	// Eigen::BiCGSTAB<RtsMat<Htype2 >,SubmatPreconditioner<cdouble> > solver2;
	
	
	// H.getBasis().getRadial().setState(0);
	
	// solver.preconditioner().setup(H,dt);

	// solver.compute(proptest);
	
	// H2.getBasis().getRadial().setState(0);
	
	// solver2.preconditioner().setup(H2,dt);

	// solver2.compute(proptest2);
	
	// CranknichApprox<Htype,Htype2,dirbs> cnp(H,H2);
	
	// cnp.proptest = &proptest;
	// cnp.solver = &solver;
	// cnp.proptest2 = &proptest2;
	// cnp.solver2 = &solver2;
	
	// cnp.propagate(psi1,(0.6*PI)/Ntime,20);
	
	
	// return 0;
}

int a() {
	using dirbs = rthphbasis<dkbbasis,spnrbasis>;
	using dirwf = wavefunc<dirbs>;
	
	int Nsplines = 250;
	int Ntime = 4000;
	double rBox = 40.0;
	Eigen::IOFormat outformat(Eigen::FullPrecision,Eigen::DontAlignCols,", ","\n","(","),"," = npy.array((\n","\n))\n",' ');
	
	vec t(Nsplines+1);
	vec y(Nsplines+1);
	for(int i = 0; i < Nsplines+1; i++) {
		t[i] = rBox/(Nsplines)*i;
	}
	
	/*
	for(int i = 0; i < Nsplines+1; i++) {
		y[i] = sin(50.0/(Nsplines)*i);
	}*/
	
	/*vec ptx(100000);
	for(int i = 0; i < 100000; i++) {
		ptx[i] = 50.0/100000 * i;
	}*/
	
	//cout << "x " << t.format(outformat) << "y " << y.format(outformat);
	//vec yx = linInterpolate(t,y,ptx);
	
	//cout << "xx " << ptx.format(outformat) << "yy " << yx.format(outformat);
	
	//return 0;
	
	
	dkbbasis dkbb(t,7);
	dkbb.setState(-1);
	lvec x = dkbb.glpts();
	
	//cout << "glpts " << x->format(outformat) << "\n";
	
	clsmat& splch = dkbb.splineCache(x);
	clsmat& dsplch = dkbb.dSplineCache(x,1);
	clsmat& ddsplch = dkbb.dSplineCache(x,2);
	
	/*
	cout << "dkbU " << lmat(dkbb.dkbCache(x,1).real()).format(outformat) << "dkbL " << lmat(dkbb.dkbCache(x,-1).real()).format(outformat);
	cout << "ddkbU " << lmat(dkbb.dDkbCache(x,1).real()).format(outformat) << "ddkbL " << lmat(dkbb.dDkbCache(x,-1).real()).format(outformat);
	return 0;*/
	//mat ulc = mat(dkbb.ulcmat().real());
	
	//cout << "ulcmat " << ulc.format(outformat);
	//return 0;
	
	spnrbasis spnrb(4);
	// for(int i = 0; i < spnrb.angqN(); i++) {
		// cout << ik(i) << ", " << imu(i) << std::endl;
	// }
	// return 0;
	
	// cmat bdpams = cmat::Zero(spnrb.angqN(),spnrb.angqN());
	
	// /*for(int i = 0; i < 10; i++) {
		// bdpams += spnrb.bdpam(1,i) + spnrb.bdpam(-1,i);
	// }
	// cout << "bdpams " << bdpams.format(outformat);
	// */
	
	// for(int i = 0; i < spnrb.bdplmax(); i++) {
		// //cout << ik(i) << std::endl;
		// cout << "bdpaU"<<i << cmat(spnrb.bdpam(1,i)).format(outformat);
		// cout << "bdpaL"<<i << cmat(spnrb.bdpam(-1,i)).format(outformat);
	// }
	// cout << "dpaU " << (cmat(spnrb.dpam(1))).format(outformat) <<std::endl << "dpaL " << cmat(spnrb.dpam(-1)).format(outformat);
	// return 0;
	
	
	//cout << "dkbb dpalphmat: " <<dkbb.dpalphmat(1,-1,1) << "\n" << "spnrb dpalphmat: " << spnrb.dpalphmat(1) << "\n";
	/*
	cout << "ddmat" << mat(dkbb.ddmat(1).real()).format(outformat) << "\n";
	cout << "Emat" << mat(dkbb.Emat(&coloumb<1>).real()).format(outformat) << "\n";
	cout << "kappamat" << mat(dkbb.kappamat().real()).format(outformat) << "\n";
	cout << "ulcmat" << mat(dkbb.ulcmat().real()).format(outformat) << "\n";
	*/
	
	double dt = (0.6*PI)/Ntime;
	double T = 0;
	beyondDipolePulse bdpp(10,50,15);
	
	/*cout << "At = npy.array((";
	for(int i = 0; i < Ntime; i++) {
		auto At = bdpp.template axialPart<axis::t>(dt*i);
		cout << "(" << At(0,0).real() + At(2,0).real() + At(4,0).real() << ", " << At(1,0).real() + At(3,0).real() + At(5,0).real() << "),\n";
	}
	cout << "))\n";
	return 0;
	*/
	bdpp.setTime(0.8);
	
	/*cout << "dprU"  << cmat(dplA<15,50,10>(1.75)*dkbb.dpam(1,1,1)).format(outformat);
	cout << "dprL"  << cmat(dplA<15,50,10>(1.75)*dkbb.dpam(1,1,-1)).format(outformat);
	
	
	for(int i = 0; i < spnrb.bdplmax(); i++) {
		cout << "bdprU"<<i << cmat(dkbb.bdpam(1,1,1,i,bdpp)).format(outformat);
		cout << "bdprL"<<i << cmat(dkbb.bdpam(1,1,-1,i,bdpp)).format(outformat);
	}
	return 0;
	*/
	
	/*cout << "Adp = npy.array((";
	for(int t = 0; t < Ntime; t++) {
		cout << "(" << SoL * dplA<15,50,10>(t*dt) << "),\n";
	}
	
	cout << "))";

	cout << "Abdp = npy.array((";
	for(int t = 0; t < Ntime; t++) {
		cout << "(" << bdpp.template axialPart<axis::t>(t*dt)(0).real() + bdpp.template axialPart<axis::t>(t*dt)(2).real() + bdpp.template axialPart<axis::t>(t*dt)(4).real() << 
		"," << bdpp.template axialPart<axis::t>(t*dt)(1).real() + bdpp.template axialPart<axis::t>(t*dt)(3).real() + bdpp.template axialPart<axis::t>(t*dt)(5).real()<< "),\n";
	}
	
	cout << "))";
	
	return 0;
	*/
	dirbs rthphb(dkbb,spnrb);
	
	dkbb.prepbdpvecs(rthphb.angqN(), spnrb.bdplmax());
	
	vec angInit(rthphb.angqN());
	angInit.setZero();
	//cout << spnrb.angCouplings();
	angInit[0] = 1.0;
	
	//cout << cmat(rthphb.bdpalphmat(bdpp)).imag();
	//return 0;
	//rthphb.dpalphmat() = csmat(0,0);
	for(int l = 0; l < spnrb.bdplmax(); l++) {
		dkbb.bdpam(1,1,1,l,bdpp);
		dkbb.bdpam(1,1,-1,l,bdpp);
		
		spnrb.bdpam(1,l);
		spnrb.bdpam(-1,l);
	}
	
	
	int Nrad = dkbb.bdpam(-1,-1,1,0,bdpp).rows();
	
	cvec rtestvec = cvec::Constant(Nrad,1.0);
	
	
	int k1 = 6;
	int k2 = 10;

	//cout << "dpamU " << cmat(dkbb.dpam(k1,k2, 1)).format(outformat);
	//cout << "dpamL " << cmat(dkbb.dpam(k1,k2,-1)).format(outformat);
	/*
	for(int l = 0; l < spnrb.bdplmax(); l++) {
		/*
		cvec vU = cvec::Zero(Nrad);
		cvec vL = cvec::Zero(Nrad);
		
		dkbb.setState(k1,k2,1,l);
		dkbb.template cacheMatfree<bdpa>(rtestvec,vU,0,1.0);
		dkbb.setState(k1,k2,-1,l);
		dkbb.template cacheMatfree<bdpa>(rtestvec,vL,0,1.0);
		
		cout << "bdpavU["<<l<<"] " << vU.format(outformat);
		cout << "bdpavL["<<l<<"] " << vL.format(outformat);
		
		dkbb.clearVecs();
		*/
	/*	
		csmat mvU(Nrad,Nrad);
		csmat mvL(Nrad,Nrad);
		bdpft At = bdpp.template axialPart<axis::t>(0.80);
		
		for(int alpha = 0; alpha < 6; alpha++) {
			if(l%2!=alpha%2) {
				mvU += (double) At(alpha,0).real() * dkbb.bdpam(k1,k2,1,l,alpha,bdpp);
				mvL += (double) At(alpha,0).real() * dkbb.bdpam(k1,k2,-1,l,alpha,bdpp);
			}
		}
		
		cout << "bdpamU["<<l<<"] " << cmat(mvU).format(outformat);
		cout << "bdpamL["<<l<<"] " << cmat(mvL).format(outformat);
		
	}
	
	
	return 0;
	*/
	
	// using Htype = DiracBDP<dirbs>;
	// using Htype2 = Dirac<dirbs>;
	
	// Htype H(rthphb,bdpp);
	// Htype2 H2(rthphb,&dplA<15,50,10>);
	// H.Vfunc = &coloumb<1>;
	// H2.Vfunc = &coloumb<1>;
	// int Nv = H.H0().cols();
	// H.S();
	// cvec testvec = cvec::Constant(Nv,1.0);
	// //cout <<"v " << testvec.format(outformat) << std::endl;
	// //cout << "Htv" << H.Ht(1.75,testvec).format(outformat);
	
	// /*for(int i = 0; i < 100; i++) {
		// cout << i << std::endl;
		// H.H(i*0.01,testvec);
	// }
	
	// return 0;
	// */
	// /*cout << "Hbdpm" << (cmat(*/ H.Ht(0.80) *testvec/*)).format(outformat)*/;
	// /*                         */                    /*                    */
	// /*cout << "Hdpm"  << (cmat(*/ H2.Ht(0.80)*testvec/*)).format(outformat)*/;
	// /*                         */ 
	// /*cout << "Hbdp"  << (cmat( H.Ht(0.80,testvec))).format(outformat); */
	// /*                                                                  */
	// /*cout << "Hdp"   << (cmat( H2.Ht(0.80,testvec))).format(outformat);*/
	
	
	// cmat Hvt = cmat::Zero(Nv,Ntime);
	// cmat Hvtdp = cmat::Zero(Nv,Ntime);
	// for(int i = 0; i < Ntime; i++) {
		// double tt = i*dt;
		// Hvt.col(i) = H.Ht(tt,testvec);
// //		Hvtdp.col(i) = H2.Ht(tt,testvec);
		// cout << i << std::endl;
	// }
	
	// cout << Hvt.imag() << std::endl; //<< Hvtdp.imag();
	
	// //cout << bdpp.getTime() << std::endl;
	
	// return 0;
	
	//cout << dplA<15,50,10>(0) << ", " << dplA<15,50,10>(0.25) << ", " << dplA<15,50,10>(0.5) << ", " << dplA<15,50,10>(0.75) << ", " << dplA<15,50,10>(1.0) << ", " << dplA<15,50,10>(1.25) << ", " << dplA<15,50,10>(1.5) << ", " << dplA<15,50,10>(1.75) << std::endl;
	//return 0;
	//cout << "S " << mat(H.S().real()).format(outformat);
	/*
	cout << "dpathU " << cmat(spnrb.dpalphmat(1)).format(outformat);
	cout << "dpathL " << cmat(spnrb.dpalphmat(-1)).format(outformat);
	
	cout << "dparU " << cmat(dkbb.dpalphmat(1,1,1)).format(outformat);
	cout << "dparL " << cmat(dkbb.dpalphmat(1,1,-1)).format(outformat);
	*/
	/*cout << "dpam " << cmat(rthphb.dpalphmat()).format(outformat);
	cout << "cAt =" << SoL * dplA<15,50,10>(0.879) << std::endl;
	cout << "Htdp " << mat(H2.Ht(0.911).imag()).format(outformat);
	cout << "Htbdp " << mat(H.Ht(0.911).imag()).format(outformat);
	return 0;
	*/
	//cout << "propagator " << cmat(*H.S() - cdouble(0,0.5) *((0.6*PI)/Ntime) * H.H(1.75)).format(outformat);
	
	// cout << *dkbb.ddmat(1);
	//return 0;
	
	
	//H.prepeigs(Nsplines,Nsplines/2);
	/*
	bsbasis bsb(t,7);
	
	// //cout << *(bsb.olmat());
	
	
	// //cout << "Declaring thb\n";
	spharmbasis thb(4);
	
	// cout << mat(thb.pzmat()->imag()).format(outformat);
	
	// //cout << *(thb.olmat());
	
	
	
	// //cout << "Declaring rthphb\n";
	rthphbasis rthphb(&bsb,&thb);
	
	
	
	// //cout << cmat(*rthphb.pzmat()).format(outformat);
	// //return 0;
	
	
	Schrodinger H(&rthphb,&dplA<15,50,400>);
	
	H.Vfunc = &coloumb<1>;
	
	//cout << "dirHt" << mat(H.Ht(1.75).imag()).format(outformat);
	//cout << "schHt " << mat(scH.Ht(1.75).imag()).format(outformat);
	//return 0;
	*/
	
	//cout << "U " << cmat(H.S() - dt * cdouble(0,0.5) * H.H(dt)).format(outformat) << std::endl;
	
	//return 0;
	

	/*cout << "cLa = npy.array((";
	for(int j = 0; j < evals.size(); j++) {
		for(int i = 0; i < evals[j].rows(); i++) {
			cout << evals[j][i] << ",";
			
		}
		
	}
	
	
	
	cout << "))\n\n";
	
	//return 0;
	//cout << "cEvecs " << estates[0].coefs.real().format(outformat)<<"\n";
	
	//return 0;
	
	cout << estates[0].coefs.rows() << "," << estates[0].coefs.cols() << "\n";
	*/
	
	
	//cout << mat(dkbb.dpalphmat(0,1)->real()).format(outformat);
	
	//cout << "Hpsi1 " << H.H(1.75,psi1.coefs).format(outformat)<<"\n";
	
	//return 0;
	
	
	for(int l = 0; l < spnrb.bdplmax(); l++) {
		dkbb.bdpam(1,1,1,l,bdpp);
		dkbb.bdpam(1,1,-1,l,bdpp);
	}
	
	dkbb.dpam(1,1, 1);
	dkbb.dpam(1,1,-1);
	
	using Htype = DiracBDP<dirbs>;
	Htype H(rthphb,bdpp);
	H.Vfunc = &coloumb<1>;
	H.prepeigs(Nsplines,Nsplines/2);
	
	vector<vec>& evals = H.getevals();
	vector<dsmat>& evecs = H.getevecs();
	
	cout << "eigs prepped\n";
	
	dirwf psi1 = dirwf(rthphb,evecs[0].col(Nsplines + 5));
	psi1.normalize();
	cout << "psi1 " << psi1.coefs.format(outformat)<<"\n";
	
	
	
	RtsMat<Htype> proptest;
	proptest.setDt(dt);
	proptest.setTime(T);
	proptest.attachHamiltonian(H);

	
	cvec b = H.S(psi1.coefs) - dt * cdouble(0,0.5) * H.H(T,psi1.coefs);
	
	Eigen::BiCGSTAB<RtsMat<Htype >,SubmatPreconditioner<cdouble> > solver;
	
	H.getBasis().getRadial().setState(0);
	
	solver.preconditioner().setup(H,dt);

	solver.compute(proptest);
	
	proptest.setTime(T);
	
	b = H.S(psi1.coefs) - dt * cdouble(0,0.5) * H.H(T,psi1.coefs);
	
	solver.compute(proptest);
	cvec c = solver.solveWithGuess(b,cvec(psi1.coefs));
	
	cout <<  "iterations: " << solver.iterations() << std::endl;
	
	return 0;
	//csmat precondMat = H.S<radial>() - dt * cdouble(0,0.5)  * H.H0<radial>();
	
	//cout << precondMat << std::endl;
	
	// solver.preconditioner().setup(H,dt);
	// solver2.preconditioner().setup(H2,dt);
	// solver.preconditioner().setup(precondMat);
	
	
	
	// cvec c = solver.preconditioner().solve(b);
	
	// cout << "b " << b.format(outformat) << "\n";
	
	// solver.compute(proptest);
	
	// cvec psi2 = solver.solveWithGuess(b,psi1.coefs);
	// cout << "iterations = " << solver.iterations();
	
	// cout << "psi2 " << psi2.format(outformat) << "\n";

	//return 0;
	/*
	for(int i = 1; i < 4; i++) {
		T = i * dt;
		proptest.setTime(dt);
		
		b.noalias() = H.S(psi2) + dt * cdouble(0,0.5) * H.H(T,psi2);
		
		solver.compute(proptest);
		//if(i == 9) cout << "psi2Precond " << cvec(solver.preconditioner().solve(b)).format(outformat);
		
		psi2 = solver.solveWithGuess(b,cvec(psi2));
		cout <<  i << ": iterations: " << solver.iterations() << std::endl;
		
	}
	cout << "psi2 " << psi2.format(outformat)<<"\n";
	
	std::cout << "BiCGSTAB: #iterations: " << solver.iterations() << ", estimated error: " << solver.error() << std::endl;


	//std::cout << "BiCGSTAB mat: #iterations: " << matsolver.iterations() << ", estimated error: " << matsolver.error() << std::endl;
	return 0;
	*/
	
	
	Cranknich<Htype,dirbs,true> cnp(H);
	
	//cnp.matfree = true;
	cnp.proptest = &proptest;
	cnp.solver = &solver;
	
	// cnp.proptest2 = &proptest2;
	// cnp.solver2 = &solver2;
	
	
	cnp.propagate(psi1,(0.6*PI)/Ntime,Ntime);
	
	//cout << "psit " << cnp.wf.coefs.format(outformat) << "\n";
	
	dirwf wft = cnp.wf[Ntime - 1];
	
	cout << "psit " << wft.coefs.format(outformat) << "\n";
	//dirwf wft = dirwf(rthphb,psi2);
	/*for(int i = 0; i < evals.size(); i++) {
		cout <<"la"<<i<<" "<< evals[i].format(outformat) << "dPdE"<<i<<" " << (wft*estates[i]).format(outformat);
	}*/
	
	cmat projt = psi1 * cnp.wf;
	cout << "projt " << projt.format(outformat) << "\n";
	
	mat dPdE = H.dPdE(wft,3000,-0.500007,300);
	
	cout << "dPdE " << dPdE.format(outformat) << "\n";
	
	return 0;
}
#include "potential.h"
#include "RtsMat.h"
#include "dkbbasis.h"
#include "rthphbasis.h"
#include "wavefunc.h"
#include "hamiltonian.h"
#include "propagator.h"

#define INTENSITY 10
#define Z 1


int beyondDipolePulse::l = 1;
int beyondDipoleCarrierPulse::l = 1;

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
	using dirbs = rthphbasis<dkbbasis,spnrbasis>;
	using dirwf = wavefunc<dirbs>;
	
	
	int Nsplines = 150;
	int Ntime = 8000;
	int Nkappa = 6;
	int Nmu = 4;
	int Nl = 10;
	
	double rBox = 30.0;
	Eigen::IOFormat outformat(Eigen::FullPrecision,Eigen::DontAlignCols,", ","\n","(","),"," = npy.array((\n","\n))\n",' ');
	Eigen::IOFormat outformatLine(Eigen::FullPrecision,Eigen::DontAlignCols,", "," ","(","),"," = npy.array((\n","\n))\n",' ');
	
	
	cout<< "Simulation run parameters:\nSpline knots:" << Nsplines << "\nTime steps: " << Ntime 
		<< "\nkappa max quantum number: " << Nkappa << "\nmu max quantum number: " << Nmu 
		<< "\nBessel function l max: " << Nl << "\nBox radius: " << rBox << "\nIntensity: "  << INTENSITY << std::endl;
	
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
	
	spnrb.bdplOverride(Nl);
	
	// cout << "dpaU" << cmat(spnrb.dpam(1)).format(outformat);
	// cout << "dpaL" << cmat(spnrb.dpam(-1)).format(outformat);
	
	// for(int l = 0; l < spnrb.bdplmax(); l++) {
		// cout << "Nonzero elements of bdpaU["<<l<<"]"<<spnrb.bdpam(1,l).nonZeros() << std::endl;
		// cout << "Fill-in ratio of bdpaU["<<l<<"]: "<<(double)spnrb.bdpam(1,l).nonZeros() / spnrb.bdpam(1,l).size() << std::endl;
		// // cout << "bdpaL["<<l<<"]"<<cmat(spnrb.bdpam(-1,l)).format(outformat);
	// }
	
	// return 0;
	
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
	
	using Htype = DiracBDP<dirbs>;
	//using Htype = Dirac<dirbs>;
	Htype H(rthphb,bdpp);
	//Htype H(rthphb,&dplA<15,50,INTENSITY>);
	H.Vfunc = &coloumb<Z>;
	
	
	cvec testvec = cvec::Constant(rthphb.radqN()*rthphb.angqN(),1.0);
	
	
	H.prepeigs(Nsplines,Nsplines/2);
	H.H0radprep();
	
	vector<vec>& evals = H.getevals();
	vector<dsmat>& evecs = H.getevecs();
	
	// cout << "eigs prepped\n";
	
	//Extract bound state eigenvectors
	// int k = 0;
	// for(int i = 0; i < evals.size(); i++) {
		// for(int j = 0; j < evals[i].rows(); j++) {	
			// if((evals[i][j] > -pow(SoL,2)) && (evals[i][j] < 0)) {
				// // cout << "evlBound[" << k << "] = " << evals[i][j] << std::endl;
				// // cout << "psiBound[" << k << "] " << cmat(evecs[i].col(j)).format(outformatLine) << std::endl;
				// cout << "SpsiBound[" << k << "] " << H.S(evecs[i].col(j)).format(outformatLine) << std::endl;
				// // cout << "idxBound[" << k << "] = (" << i << ", " << j << ")" << std::endl;
				// k++;
			// }
		// }
	// }
	// return 0;
	
	dirwf psi1 = dirwf(rthphb,evecs[0].col(Nsplines + 5));
	psi1.normalize();
	cout << "psi1 " << psi1.coefs.format(outformat)<<"\n";
	//cout << "E1 = " << evals[0][Nsplines+5] << std::endl;;
	
	cvec b2 = H.Ht(1.75,testvec);
	
	// cout << "b2_bdp " << b2.format(outformat);
	// return 0;
	
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
	
	// cnp.setDumpfile((filenamePrefix + "_dump"));
	
	//Cranknich<Htype,dirbs> cnp(H);
	solver.setMaxIterations(100);
	
	cnp.proptest = &proptest;
	cnp.solver = &solver;
	
	
	cout << "psi2 " << solver.solve(b2).format(outformat);
	// return 0;
	
	cnp.propagate(psi1,(0.6*PI)/Ntime,Ntime,100);
	
	dirwf wft = cnp.wf[(Ntime/100) - 1];
	
	ofstream psievf(filenamePrefix + "_psiev");
	psievf << "psit" << filenamePrefix <<" " << wft.coefs.format(outformat) << "\n";
	
	
	psievf << "evl"   << filenamePrefix << " = npy.zeros((" << evecs.size() << "),dtype = object)\n";
	psievf << "psiev" << filenamePrefix << " = npy.zeros((" << evecs.size() << "),dtype = object)\n";
	
	for(int i = 0; i < evecs.size(); i++) {
		psievf << "evl" << filenamePrefix << "["<<i<<"]" << evals[i].format(outformat) << std::endl;
		psievf << "psiev" << filenamePrefix << "["<<i<<"] " << cmat(wft * evecs[i]).format(outformat) << std::endl;
	}
	
	// cmat projt = psi1 * cnp.wf;
	// psievf << "projt" << filenamePrefix << " " << projt.format(outformat) << "\n";
	
	psievf.close();
	
	ofstream dPdEf(filenamePrefix + "_dPdE");
	
	mat dPdE = H.dPdE(wft,5000,-0.500007,500);
	
	dPdEf << "dPdE" << filenamePrefix << dPdE.format(outformat);
	
	dPdEf.close();
	
	return 0;
}

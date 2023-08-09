/*
This is the old file for dipole runs. Keeping it around in case it becomes useful.
*/

#include "RtsMat.h"
#include "dkbbasis.h"
#include "rthphbasis.h"
#include "wavefunc.h"
#include "hamiltonian.h"
#include "propagator.h"

#define INTENSITY 400
#define Z 1
int beyondDipolePulse::l = 1;

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

void writeSparseNonZeros(const dsmat& mat,ofstream& file) {
	file << " = npy.array((\n";
	
	for(int i = 0; i < mat.outerSize(); i++) {
		for(dsmat::InnerIterator j(mat,i); j; ++j) {
			file << "(" << j.value();
			
			//if(j.value().imag() >= 0)
			//	cout << " + ";
			
			file //<< j.value().imag() 
			<< "," << j.row() << ", " << j.col() << "),\n";
		}
	}
	
	file << "))\n";
}


int main() {
	using dirbs = rthphbasis<dkbbasis,spnrbasis>;
	using dirwf = wavefunc<dirbs>;
	
	
	int Nsplines = 250;
	int Ntime = 8000;
	int Nkappa = 8;
	int Nmu = 0;
	double rBox = 30.0;
	Eigen::IOFormat outformat(Eigen::FullPrecision,Eigen::DontAlignCols,", ","\n","(","),"," = npy.array((\n","\n))\n",' ');
	
	vec t(Nsplines+1);
	vec y(Nsplines+1);
	for(int i = 0; i < Nsplines+1; i++) {
		t[i] = rBox/(Nsplines)*i;
	}
	
	std::stringstream fnpstream;
	
	fnpstream << "dirdp_E" << INTENSITY << "R" << Nsplines << "K" << Nkappa << "Mu" << Nmu << "r" << rBox << "T" << Ntime;
	
	string filenamePrefix = fnpstream.str();
	
	
	dkbbasis dkbb(t,7);
	dkbb.setState(-1);
	lvec x = dkbb.glpts();
	
	
	clsmat& splch = dkbb.splineCache(x);
	clsmat& dsplch = dkbb.dSplineCache(x,1);
	clsmat& ddsplch = dkbb.dSplineCache(x,2);
	
	
	spnrbasis spnrb(Nkappa,Nmu);
	
	// cout << "dpaU ";
	spnrb.dpam(1);
	// cout << "dpaL ";
	spnrb.dpam(-1);
	
	cmat bdpams = cmat::Zero(spnrb.angqN(),spnrb.angqN());
	
	beyondDipolePulse bdpp(INTENSITY,50,15);
	//dipolizedBeyondDipolePulse bdpp(10,50,15);
	
	bdpp.setTime(1.75);
	
	
	double dt = (0.6*PI)/Ntime;
	double T = 0;
	
	dirbs rthphb(dkbb,spnrb);
	

	vec angInit = vec::Zero(rthphb.angqN());
	angInit[0] = 1.0;
	cout << "aqn pre prune: " << rthphb.angqN();
	rthphb.pruneUncoupled(angInit);

	cout << "aqn post prune: " << rthphb.angqN();
	
	dkbb.prepbdpvecs(rthphb.angqN(), spnrb.bdplmax());
	dkbb.clvcs();
	
	
	// cout << "angidsDirDp_K" << Nkappa << "Mu" << Nmu << " = npy.array((";
	// for(int i = 0; i < rthphb.angids.size(); i++) {
		// cout << "(" << i << ", " << rthphb.angids[i] << ", " << ik(rthphb.angids[i])<< ", " << imu(rthphb.angids[i]) << ")," << std::endl;
	// }
	
	// cout << "))";
	
	// cout << "angidsReverseDirDp_K" << Nkappa << "Mu" << Nmu << " = npy.array((";
	// for(int i = 0; i < rthphb.angidsReverse.size(); i++) {
		// cout << "(" << i << ", " << rthphb.angidsReverse[i] << ")\n";// " << il(rthphb.angidsReverse[i])<< ", " << im(rthphb.angidsReversed[i]) << ")," << std::endl;
	// }
	
	// cout << "))";
	
	//return 0;
	// cout << "dpaUr ";
	dkbb.dpam(1,1,1);
	// cout << "dpaLr ";
	dkbb.dpam(1,1,-1);
	
	//return 0;
	
	
	using Htype = Dirac<dirbs>;
	Htype H(rthphb,&dplA<15,50,INTENSITY>);
	H.Vfunc = &coloumb<1>;
	H.H0radprep();
	
	H.prepeigs(Nsplines,Nsplines/2);
	
	cvec testvec = cvec::Constant(rthphb.radqN()*rthphb.angqN(),1.0);
	
	cvec b2 = H.H(1.75,testvec);
	
	//cout << "b " << b.format(outformat);
	
	cout << "b2 " << b2.format(outformat);
	// return 0;
	
	//cout << "H0_0 " << dkbb.getH0mat(0);
	//cout << "H0_kappa " << dkbb.getH0mat(1);
	//cout << "H0_kappa2 " << dkbb.getH0mat(2);
	//cout << "H0_kappa3 " << dkbb.getH0mat(3);
	//return 0;
	//auto t0 = std::chrono::system_clock::now();
	//
	//cvec v_old;
	//cvec v_opt;
	//
	//for(int i = 0; i < 100; i++) {
	//	v_old = H.H0(testvec);
	//}
	//
	//auto t1 = std::chrono::system_clock::now();
	//
	//for(int i = 0; i < 100; i++) {
	//	v_opt = rthphb.template matvec<H0>(testvec);
	//}
	//
	//auto t2 = std::chrono::system_clock::now();
	//
	//cout << "v_old execution time: " << std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count() << "\n";
	//cout << "v_opt execution time: " << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() << "\n";
	//
	//cout << "v_old " << v_old.format(outformat) << std::endl;
	//cout << "v_opt " << v_opt.format(outformat) << std::endl;
	
	//return 0;
	
	vector<vec>& evals = H.getevals();
	vector<dsmat>& evecs = H.getevecs();
	
	// for(int i = 0; i < evecs.size(); i++) {
		// cout << "evcnrm[" << i << "] = npy.array((";
		// for(int j = 0; j < evecs[i].cols(); j++) {
			// cout << evecs[i].col(j).adjoint() * H.S(evecs[i].col(j)) << ", ";
		// }
		// cout << ")" << std::endl;
	// }
	
	// return 0;
	
	// for(int i = 0; i < evals.size(); i++) {
		// stringstream fname;
		// fname << "evlEvcK20R500r60kappa" << i << "";
		// ofstream evlf(fname.str());
		// evlf << "evl[" <<i << "] " << evals[i].format(outformat);
		// evlf << "evc_data[" <<i << "]";
		// writeSparseNonZeros(evecs[i],evlf);
		// evlf.close();
	// }
	
	// return 0;
	cout << "eigs prepped\n";
	
	//return 0;
	
	dirwf psi1 = dirwf(rthphb,evecs[0].col(Nsplines + 5));
	psi1.normalize();
	cout << "psi1 " << psi1.coefs.format(outformat)<<"\n";
	//cout << "E1 = " << evals[0][Nsplines+5] << std::endl;;
	
	
	
	
	RtsMat<Htype> proptest;
	proptest.setDt(dt);
	proptest.setTime(T);
	proptest.attachHamiltonian(H);

	
	cvec b = H.S(psi1.coefs) - dt * cdouble(0,0.5) * H.H(T,psi1.coefs);
	
	Eigen::BiCGSTAB<RtsMat<Htype >,SubmatPreconditioner<cdouble> > solver;
	
	//solver.setMaxIterations(1000);
	
	H.getBasis().getRadial().setState(0);
	
	solver.preconditioner().setup(H,dt);

	solver.compute(proptest);
	
	cvec psi2 = solver.solve(b);
	
	cout << "psi2 " << psi2.format(outformat);
	// return 0;
	
	Cranknich<Htype,dirbs,true> cnp(H);
	
	
	
	//Cranknich<Htype,dirbs> cnp(H);
	
	
	cnp.proptest = &proptest;
	cnp.solver = &solver;
	
	// cnp.setDumpfile((filenamePrefix + "_dump"));
	
	cnp.propagate(psi1,(0.6*PI)/Ntime,Ntime,10);
	
	dirwf wft = cnp.wf[Ntime/10 - 1];
	
	ofstream psievf(filenamePrefix + "_psiev");
	psievf << "psit " << wft.coefs.format(outformat) << "\n";
	
	psievf << "evl"   << filenamePrefix << " = npy.zeros((" << evecs.size() << "),dtype = object)\n";
	psievf << "psiev" << filenamePrefix << " = npy.zeros((" << evecs.size() << "),dtype = object)\n";
	
	for(int i = 0; i < evecs.size(); i++) {
		psievf << "evl" << filenamePrefix << "["<<i<<"]" << evals[i].format(outformat) << std::endl;
		psievf << "psiev" << filenamePrefix << "["<<i<<"] " << cmat(wft * evecs[i]).format(outformat) << std::endl;
	}
	
	cmat projt = psi1 * cnp.wf;
	psievf << "projt " << projt.format(outformat) << "\n";
	
	psievf.close();
	
	ofstream dPdEf(filenamePrefix + "_dPdE");
	
	mat dPdE = H.dPdE(wft,5000,-0.500007,500);
	
	dPdEf << "dPdE" << filenamePrefix << dPdE.format(outformat);
	
	dPdEf.close();
	
	return 0;
}
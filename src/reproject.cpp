#include "potential.h"
#include "RtsMat.h"
#include "dkbbasis.h"
#include "rthphbasis.h"
#include "wavefunc.h"
#include "hamiltonian.h"
#include "propagator.h"

#include <fstream>

#define INTENSITY 300
#define Z 1

int beyondDipolePulse::l = 0;

using namespace std;

cvec loadData(std::string filename) {
	vector<double> rawData;
	
	ifstream psiFile(filename);
	
	string row;
	
	string entry;
	
	int Nrow = 0;
	
	while(getline(psiFile,row,'\r')) {
		stringstream rowstream(row);
		
		while(getline(rowstream,entry,' ')) {
			// cout << stod(entry);
			
			rawData.push_back(stod(entry));
			// cout << " ";
		}
		// cout << "\n";
		Nrow ++;
	}
	psiFile.close();
	
	mat realCoeffs = Eigen::Map<mat>(rawData.data(),rawData.size() / Nrow,Nrow);
	
	cvec outvec = cdouble(1,0) * realCoeffs.row(0) + cdouble(0,1) * realCoeffs.row(1);
	
	
	return outvec;
}

int main(int argc, char** argv) {
	using dirbs = rthphbasis<dkbbasis,spnrbasis>;
	using dirwf = wavefunc<dirbs>;
	//Load wavefunction from txt
	
	int Nsplines = 250;
	int Ntime = 8000;
	int Nkappa = 20;
	int Nmu = 0;
	int Nl = 0;
	double rBox = 30.0;
	
	Eigen::IOFormat outformat(Eigen::FullPrecision,Eigen::DontAlignCols,", ","\n","(","),"," = npy.array((\n","\n))\n",' ');
	Eigen::IOFormat partformat(Eigen::FullPrecision,Eigen::DontAlignCols,", ","\n","(","),","","",' ');
	Eigen::IOFormat outformatLine(Eigen::FullPrecision,Eigen::DontAlignCols,", "," ","(","),"," = npy.array((\n","\n))\n",' ');
	
	string fname = argv[1];
	
	

	vec t(Nsplines+1);
	vec y(Nsplines+1);
	for(int i = 0; i < Nsplines+1; i++) {
		t[i] = rBox/(Nsplines)*i;
	}
	
	beyondDipolePulse bdpp(INTENSITY,50,15);
	
	dkbbasis dkbb(t,7);
	dkbb.setState(-1);
	lvec x = dkbb.glpts();
	
	
	clsmat& splch = dkbb.splineCache(x);
	clsmat& dsplch = dkbb.dSplineCache(x,1);
	clsmat& ddsplch = dkbb.dSplineCache(x,2);
	
	
	spnrbasis spnrb(Nkappa,Nmu);
	spnrb.bdplOverride(Nl);
	
	// for(int i = 0; i < 4; i++) {
		// cout << "S" << i << " " << cmat(spnrb.olcm(i)).format(outformat);
	// }
	
	// return 0;
	
	bdpp.setTime(1.75);
	
	
	double dt = (0.6*PI)/Ntime;
	double T = 0;
	
	dirbs rthphb(dkbb,spnrb);
	
	vec angInit = vec::Zero(rthphb.angqN());
	angInit[0] = 1.0;
	
	rthphb.pruneUncoupled(angInit,true);
	
	cvec psi = loadData(fname);
	
	dirwf psi1 = dirwf(rthphb,psi);
	psi1.normalize();
	// cout << "psi " << psi1.coefs.format(outformat)<<"\n";
	// psi1.save("savetest");
	// return 0;
	
	dkbb.Sm();
	dkbb.dpam(1,1,1);
	dkbb.dpam(-1,1,1);
	for(int l = 0; l < spnrb.bdplmax(); l++) {
		dkbb.bdpam(1,1,1,l,bdpp);
		dkbb.bdpam(1,1,-1,l,bdpp);
	}
	
	dkbb.setState(-1);
	
	
	dkbb.olcm(1,1,0);
	dkbb.olcm(1,1,2);
	
	cvec rtestvec = cvec::Constant(dkbb.radqN(),1.0);
	
	cout << "Sv "  << dkbb.template matfree<S>(rtestvec).format(outformat) << std::endl;
	
	
	// return 0;
	
	// cout << "Sm " << cmat(dkbb.Sm()).format(outformat);
	// cout << "Pm " << cmat(dkbb.olcm(0,0,0)).format(outformat);
	// cout << "Qm " << cmat(dkbb.olcm(0,0,2)).format(outformat);
	
	cvec SUv = cvec::Zero(dkbb.radqN());
	cvec SLv = cvec::Zero(dkbb.radqN());
	
	dkbb.template cacheMatfree<SU>(rtestvec,SUv,0,1.0,0,0,1);
	dkbb.template cacheMatfree<SL>(rtestvec,SLv,0,1.0,0,0,1);
	
	cout << "SUv " << SUv.format(outformat) << std::endl;
	cout << "SLv " << SLv.format(outformat) << std::endl;
	// return 0;
	
	using Htype = DiracBDP<dirbs>;
	
	Htype H(rthphb,bdpp);
	
	H.Vfunc = &coloumb<Z>;
	
	H.prepeigs(Nsplines,Nsplines/2);
	
	vector<vec>& evals = H.getevals();
	vector<dsmat>& evecs = H.getevecs();
	
	cout << "psi" << psi1.coefs.format(outformat) << std::endl;
	
	// cout << "evc0" << cmat(evecs[0]).format(outformat) << std::endl;
	
	// cout << "Filename: " << argv[1] << "\n";
	cout << "psi: (" << psi1.coefs.rows() << ", " << psi1.coefs.cols() << ")\n";
	cout << "evecs: (" <<  evecs[0].rows() << ", " << evecs[0].cols() << ")" << std::endl;
	
	cout << "evlDirBdp_R"<<Nsplines<<"K"<<Nkappa<<"Mu"<<Nmu<<"l"<<Nl<<"r"<<int(rBox)<<"= npy.zeros(("<<evecs.size()<<"),dtype=object)" << std::endl;
	cout << "psievDirBdp_R"<<Nsplines<<"K"<<Nkappa<<"Mu"<<Nmu<<"l"<<Nl<<"r"<<int(rBox)<<"= npy.zeros(("<<evecs.size()<<"),dtype=object)" << std::endl;
	cout << "psievDirBdp4_R"<<Nsplines<<"K"<<Nkappa<<"Mu"<<Nmu<<"l"<<Nl<<"r"<<int(rBox)<<"= npy.zeros(("<<evecs.size()<<","<<4<<"),dtype=object)" << std::endl;
	
	for(int i = 0; i < evecs.size(); i++) {
		cout << "evlDirBdp_R"<<Nsplines<<"K"<<Nkappa<<"Mu"<<Nmu<<"l"<<Nl<<"r"<<int(rBox)<<"["<<i<<"]" << evals[i].format(outformat) << std::endl;
		cout << "psievDirBdp_R"<<Nsplines<<"K"<<Nkappa<<"Mu"<<Nmu<<"l"<<Nl<<"r"<<int(rBox)<<"["<<i<<"]" << cmat(psi1 * evecs[i]).format(outformat) << std::endl;
		for(int j = 0; j < 4; j++) {
			
			cout << "psievDirBdp4_R"<<Nsplines<<"K"<<Nkappa<<"Mu"<<Nmu<<"l"<<Nl<<"r"<<int(rBox)<<"["<<i<<"]["<<j<<"]" << (cmat(rthphb.template matfree<compS>(psi1.coefs,j).transpose() * evecs[i])).format(outformat) << std::endl;
		}
	}
	
	return 0;
}
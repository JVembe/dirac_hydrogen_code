#include "potential.h"
#include "RtsMat.h"
#include "dkbbasis.h"
#include "rthphbasis.h"
#include "wavefunc.h"
#include "hamiltonian.h"
#include "propagator.h"

#include <fstream>

#define INTENSITY 400
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
			cout << Nrow << " ";
			
			cout << stod(entry);
			
			rawData.push_back(stod(entry));
			cout << " ";
		}
		cout << "\n";
		Nrow ++;
	}
	psiFile.close();
	
	mat realCoeffs = Eigen::Map<mat>(rawData.data(),rawData.size() / Nrow,Nrow);
	
	cvec outvec = cdouble(1,0) * realCoeffs.row(0) + cdouble(0,1) * realCoeffs.row(1);
	
	
	return outvec;
}

cmat loadPsievs(std::string filename) {
	vector<double> rawData;
	// cout << "opening " << filename << endl;
	ifstream psievFile(filename);
	
	string row;
	
	string entry;
	
	int Nrow = 0;
	
	int NAng;
	bool record = true;
	while(getline(psievFile,row,'\n')) {
		stringstream rowstream(row);
		
		// if(record) cout << "row " << Nrow << ": ";
		//Identify what type of row we are in
		// cout << row[0];
		//If fist two rows, 
		if(Nrow < 2) {
			// cout << row << "\n";
			size_t pos = row.find("npy.zeros(");
			size_t pos2 = row.find(", dtype");
			// cout << "array decl at " << pos << endl;
			// cout << "number ends at " << pos2 << endl;
			
			string numstring = row.substr(pos + 10, pos2-(pos+10));
			
			// cout << numstring << endl;
			
			NAng = stod(numstring);
		}
		else {
			//Need to identify if we are reading psiev or evl data, skip evl data
			
			// if(record) cout << row[0];
			
			if (row[0] == 'p') {
				record = true;
				// cout << "recording psievs";
			}
			else {
				if(record) {
					while(getline(rowstream,entry,',')) {
						//Special handling of start and end of array
						size_t start = entry.find("((");
						if(start!=-1) {
							// cout << entry.substr(start+2) << endl;
							rawData.push_back(stod(entry.substr(start+2)));
						}
						else {
							// cout << entry[0] << endl;
							if(entry[0] == ' ') {
								// cout << entry.substr(2) << endl;
								rawData.push_back(stod(entry.substr(2)));
							}
							else {
								size_t brkpos = entry.find(")");
								if(brkpos!=-1) {
									// cout << brkpos << endl;
									// cout << entry.substr(0,brkpos) << endl;
									rawData.push_back(stod(entry.substr(0,brkpos)));
								}
							}
						}
					}
				}
				record = false;
			}
		}
		// while(getline(rowstream,entry,' ')) {
			// cout << Nrow << " ";
			
			// cout << stod(entry);
			
			// rawData.push_back(stod(entry));
			// cout << " ";
		// }
		// if(record) cout << "\n";
		Nrow ++;
	}
	// cout << std::endl;
	psievFile.close();
	
	// cout << "rawData size: " << rawData.size();
	
	mat realCoeffs = Eigen::Map<mat>(rawData.data(),2, rawData.size() / 2);
	
	// cout << realCoeffs << endl;
	
	cvec outvec = cdouble(1,0) * realCoeffs.row(0) + cdouble(0,1) * realCoeffs.row(1);
	// cout << outvec << endl;;
	cmat psievmat = outvec.reshaped(outvec.size()/NAng,NAng);
	// cout << psievmat << endl;
	return psievmat;
}

int main(int argc, char** argv) {
	using dirbs = rthphbasis<dkbbasis,spnrbasis>;
	using dirwf = wavefunc<dirbs>;
	//Load wavefunction from txt
	
	int Nsplines = 250;
	int Ntime = 16000;
	int Nkappa = 20;
	int Nmu = 10;
	int Nl = 10;
	double rBox = 30.0;
	
	Eigen::IOFormat outformat(Eigen::FullPrecision,Eigen::DontAlignCols,", ","\n","(","),"," = npy.array((\n","\n))\n",' ');
	Eigen::IOFormat partformat(Eigen::FullPrecision,Eigen::DontAlignCols,", ","\n","(","),","","",' ');
	Eigen::IOFormat outformatLine(Eigen::FullPrecision,Eigen::DontAlignCols,", "," ","(","),"," = npy.array((\n","\n))\n",' ');
	
	string fname = argv[1];
	
	cmat psievs = loadPsievs(fname);
	// cout << psievs;
	cvec psi;
	// cout << endl;
	// return 0;
	MPI_Init(NULL,NULL);
	int wrank;
	int wsize;
	MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
	MPI_Comm_size(MPI_COMM_WORLD,&wsize);
	
	std::stringstream fnpstream;
	
	fnpstream << "dirBdp_E" << INTENSITY << "R" << Nsplines << "K" << Nkappa << "Mu" << Nmu << "r" << rBox << "T" << Ntime;
	
	string filenamePrefix = fnpstream.str();

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
	
	// psi1.save("savetest");
	// return 0;
	
	dkbb.Sm();
	// for(int l = 0; l < spnrb.bdplmax(); l++) {
		// dkbb.bdpam(1,1,1,l,bdpp);
		// dkbb.bdpam(1,1,-1,l,bdpp);
	// }
	
	dkbb.setState(-1);
	
	
	dkbb.olcm(1,1,0);
	dkbb.olcm(1,1,2);
	
	// cvec rtestvec = cvec::Constant(dkbb.radqN(),1.0);
	rthphb.blockDistribute2();	
	dkbb.dpam(1,1,1);
	dkbb.dpam(1,1,-1);
	
	
	// return 0;
	
	// cout << "Sm " << cmat(dkbb.Sm()).format(outformat);
	// cout << "Pm " << cmat(dkbb.olcm(0,0,0)).format(outformat);
	// cout << "Qm " << cmat(dkbb.olcm(0,0,2)).format(outformat);
	
	// cvec SUv = cvec::Zero(dkbb.radqN());
	// cvec SLv = cvec::Zero(dkbb.radqN());
	
	// dkbb.template cacheMatfree<SU>(rtestvec,SUv,0,1.0,0,0,1);
	// dkbb.template cacheMatfree<SL>(rtestvec,SLv,0,1.0,0,0,1);
	
	// cout << "SUv " << SUv.format(outformat) << std::endl;
	// cout << "SLv " << SLv.format(outformat) << std::endl;
	// // return 0;
	
	using Htype = DiracBDP<dirbs>;
	
	Htype H(rthphb,bdpp);
	
	H.Vfunc = &coloumb<Z>;
	
	H.prepeigsLowMem(Nsplines,Nsplines/2, true);
	
	// vector<vec>& evals = H.getevals();
	// vector<dsmat>& evecs = H.getevecs();
	
	
	cvec psit = cvec(psievs.rows()*psievs.cols());
	vector<vec> evls = H.kappaevals;
	
	for(int i = 0; i < psievs.cols(); i++) {
		int kappa =  ik(rthphb.angids[i]);
		int kappaid = 2*abs(kappa) + (sgn(kappa) - 1)/2 - 1;
		cout << i <<"," << kappaid << endl;
		cmat evcs = H.getevecs(ik(rthphb.angids[i]),imu(rthphb.angids[i]));
		cvec freeEvlPos = cvec(((evls[kappaid].array() > 0) /* (evls[kappaid].array() < 75)*/).cast<double>());
		cvec freeEvlNeg = 0*cvec((evls[kappaid].array() < -pow(SoL,2)).cast<double>());
		cmat evcsFree = evcs.array().rowwise()*(freeEvlPos.array() + freeEvlNeg.array()).array().transpose();
		
		// cout << "i" << evcs << "\n\n" << freeEvlPos.format(outformat) << "\n\n" << freeEvlNeg.format(outformat) << "\n\n" << evcsFree << endl;
		
		cvec psii =  evcsFree * psievs.col(i);
		psit.segment(i*psievs.rows(),psievs.rows()) = psii;
	}
	// return 0;
	
	cout << "psit" << psit.format(outformat);
	
	
	dirwf psi1 = dirwf(rthphb,psit);
	// psi1.normalize();
	// cout << "psi " << psi1.coefs.format(outformat)<<"\n";
	
	cvec azv = rthphb.template matfree<dpa>(psi1.coefs);
	
	cout << "azv "  << azv.format(outformat) << std::endl;
	cout << "az" << SoL * cvec(psi1.coefs).dot(azv);
	
	cout << "pIon" << abs((psi1 * psi1)(0,0));
	
	// cout << "E0" << H.getevecs(-1,-0.5).format(outformat) << endl;
	
	// cout << "psi" << psi1.coefs.format(outformat) << std::endl;
	
	// cout << "evc0" << cmat(evecs[0]).format(outformat) << std::endl;
	
	// cout << "Filename: " << argv[1] << "\n";
	// cout << "psi: (" << psi1.coefs.rows() << ", " << psi1.coefs.cols() << ")\n";
	// cout << "evecs: (" <<  evecs[0].rows() << ", " << evecs[0].cols() << ")" << std::endl;
	
	// ofstream psievf("psievReproject");
	// psievf << "psit " << psi1.coefs.format(outformat) << "\n";

	// psievf << "evl"   << filenamePrefix << " = npy.zeros((" << evecs.size() << "),dtype = object)\n";
	// psievf << "psiev" << filenamePrefix << " = npy.zeros((" << evecs.size() << "),dtype = object)\n";

	// for(int i = 0; i < evecs.size(); i++) {
		// psievf << "evl" << filenamePrefix << "["<<i<<"]" << evals[i].format(outformat) << std::endl;
		// psievf << "psiev" << filenamePrefix << "["<<i<<"] " << cmat(psi1 * evecs[i]).format(outformat) << std::endl;
	// }

	// psievf.close();
	
	// cout << "evlDirBdp_R"<<Nsplines<<"K"<<Nkappa<<"Mu"<<Nmu<<"l"<<Nl<<"r"<<int(rBox)<<"= npy.zeros(("<<evecs.size()<<"),dtype=object)" << std::endl;
	// cout << "psievDirBdp_R"<<Nsplines<<"K"<<Nkappa<<"Mu"<<Nmu<<"l"<<Nl<<"r"<<int(rBox)<<"= npy.zeros(("<<evecs.size()<<"),dtype=object)" << std::endl;
	// // cout << "psievDirBdp4_R"<<Nsplines<<"K"<<Nkappa<<"Mu"<<Nmu<<"l"<<Nl<<"r"<<int(rBox)<<"= npy.zeros(("<<evecs.size()<<","<<4<<"),dtype=object)" << std::endl;
	
	// for(int i = 0; i < evecs.size(); i++) {
		// cout << "evlDirBdp_R"<<Nsplines<<"K"<<Nkappa<<"Mu"<<Nmu<<"l"<<Nl<<"r"<<int(rBox)<<"["<<i<<"]" << evals[i].format(outformat) << std::endl;
		// cout << "psievDirBdp_R"<<Nsplines<<"K"<<Nkappa<<"Mu"<<Nmu<<"l"<<Nl<<"r"<<int(rBox)<<"["<<i<<"]" << cmat(psi1 * evecs[i]).format(outformat) << std::endl;
		// // for(int j = 0; j < 4; j++) {
			
			// // cout << "psievDirBdp4_R"<<Nsplines<<"K"<<Nkappa<<"Mu"<<Nmu<<"l"<<Nl<<"r"<<int(rBox)<<"["<<i<<"]["<<j<<"]" << (cmat(rthphb.template matfree<compS>(psi1.coefs,j).transpose() * evecs[i])).format(outformat) << std::endl;
		// // }
	// }
	
	return 0;
}

// Design goals: Modularity
// Note: Effectiveness is debatable as only a few configurations are commonly used

// Standard Library Headers
#include <iostream>  // for std::cerr
#include <fstream>   // for std::ifstream
#include <chrono>    // for timing
#include <omp.h>             // OpenMP for parallelization

// Third-Party Libraries
#include "nlohmann/json.hpp" // for JSON parsing

// Custom Headers for Physics and Simulation
#include "potential.h"   // Defines all laser pulse potentials
#include "RtsMat.h"      // Helper class for preconditioners
#include "dkbbasis.h"    // Dual kinetic balance basis
#include "rthphbasis.h"  // Combined basis for spherical coordinates
#include "wavefunc.h"    // Wrapper for Eigen vector
#include "hamiltonian.h" // Physics in Hamiltonian class and derived classes
#include "propagator.h"  // Time evolution code

// Headers for MPI and Parallel Computing
#include "mpiFuncs.h"    // Functions for MPI and Eigen matrix operations

#define Z 1

int beyondDipolePulse::l = 0;

cvec loadData(std::string filename) {
	vector<double> rawData;
	
	ifstream psiFile(filename);
	
	string row;
	
	string entry;
	
	int Nrow = 0;
	
	while(getline(psiFile,row,'\r')) {
		// cout << row << endl;
		stringstream rowstream(row);
		
		while(getline(rowstream,entry,' ')) {
			cout << Nrow << " ";
			try{
				cout << stod(entry);
			
				rawData.push_back(stod(entry));
				cout << " ";
			}
			catch(...) {
				break;
			}
		}
		cout << "\n";
		Nrow ++;
	}
	psiFile.close();
	
	mat realCoeffs = Eigen::Map<mat>(rawData.data(),rawData.size() / Nrow,Nrow);
	
	cvec outvec = cdouble(1,0) * realCoeffs.row(0) + cdouble(0,1) * realCoeffs.row(1);
	
	
	return outvec;
}

cvec loadBinaryData(std::string filename) {
	//vector<double> rawData;
	
	ifstream psiFile(filename,ios::binary|ios::in|ios::ate);
	int32_t Nrow;
	int32_t Ncol;
	
	streampos size = psiFile.tellg();
	
	psiFile.seekg(0,ios::beg);
	
	psiFile.read(reinterpret_cast<char*>(&Nrow),sizeof(Nrow));
	psiFile.read(reinterpret_cast<char*>(&Ncol),sizeof(Ncol));
	
	
	cout << "rows: " << Nrow << ", cols: " << Ncol << endl;
	
	int dataSize = ((long int)(size)-2*(long int)(sizeof(int32_t)))/sizeof(double);
	
	cout << "size(char): " << size << ", size(double): " << dataSize << endl;
	
	
	double* rawData = new double[dataSize];
	psiFile.read(reinterpret_cast<char*>(rawData),dataSize*sizeof(double));
	
	for(int i = 0; i < 10; i++) {
		cout << rawData[i] << endl;
	}
	
	psiFile.close();
	
	mat realCoeffs = Eigen::Map<mat>(rawData,dataSize/Nrow,Nrow);
	
	//cout << realCoeffs << endl;
	cvec outvec = cdouble(1,0) * realCoeffs.row(0) + cdouble(0,1) * realCoeffs.row(1);
	
	
	return outvec;
}

// template <typename MatrixType>
// MatrixType loadMatrix(std::string filename) {
	// using idx_t = typename MatrixType::Index;
	// using scalar_t = typename MatrixType::Scalar;

	// ifstream psiFile(filename,ios::binary|ios::in|ios::ate);
	// idx_t Nrow;
	// idx_t Ncol;
	
	// streampos size = psiFile.tellg();
	
	// psiFile.seekg(0,ios::beg);
	
	// psiFile.read(reinterpret_cast<char*>(&Nrow),sizeof(Nrow));
	// psiFile.read(reinterpret_cast<char*>(&Ncol),sizeof(Ncol));
	
	
	// cout << "rows: " << Nrow << ", cols: " << Ncol << endl;
	
	// int dataSize = ((long int)(size)-2*(long int)(sizeof(idx_t)))/sizeof(scalar_t);
	
	// cout << "size(char): " << size << ", size(scalar): " << dataSize << endl;
	
	
	// scalar_t* rawData = new scalar_t[dataSize];
	// psiFile.read(reinterpret_cast<char*>(rawData),dataSize*sizeof(scalar_t));
	
	// for(int i = 0; i < 10; i++) {
		// cout << rawData[i] << endl;
	// }
	
	// psiFile.close();
	
	// MatrixType outmat = Eigen::Map<MatrixType>(rawData,dataSize/Nrow,Nrow);
	
	// return outmat;
// }


int main(int argc, char** argv) {
	using dirbs = rthphbasis<dkbbasis,spnrbasis>;
	using dirwf = wavefunc<dirbs>;
	// Check argument count
	if(argc != 3) {
		std::cerr << "Usage: " << argv[0] << " <path/to/parameters.json> <path/to/xN.out>\n";
		return 1;
	}

	// Get the JSON file path from command line arguments
	std::string json_file_path = argv[1];
	std::string fname = argv[2];
	// Open and read the JSON file
	std::ifstream input_file(json_file_path);
	if (!input_file.is_open()) {
		std::cerr << "Could not open the file " << json_file_path << "\n";
		return 1;
	}
	// cout << loadMatrix<cmat>("evc-1.mat");
	
	// return 0;
	

	// Parse JSON parameters
	nlohmann::json json_params;
	input_file >> json_params;
	//Define types for basis and wave function from templates
	//rthphbasis "R, theta, phi basis" combines dkbbasis and spnrbasis to model system in spherical coordinates
	using dirbs = rthphbasis<dkbbasis,spnrbasis>;
	using dirwf = wavefunc<dirbs>;

	// Simulation parameters
	int Nsplines = json_params["Nsplines"]; // Radial resolution, typical values: 200-250
	int Ntime = json_params["Ntime"]; // Number of time steps, typical values: 8000-20000
	int Nkappa = json_params["Nkappa"]; // Maximum absolute value of kappa quantum number, typical values: >= 16
	int Nmu = json_params["Nmu"]; // Typically 10 for nondipole simulations, as higher values of mu are suppressed by orders of magnitude
	int Nl = json_params["Nl"]; // Optimal value depends on radius of the box, typically 10 is sufficient
	double rBox = json_params["rBox"]; // Radius of the system in atomic units; 30 is usually sufficient
	int Intensity = json_params["Intensity"]; //Intensity of the laser pulse in atomic units: 10-500
	double omega = json_params["Omega"]; //Frequency of the laser pulse in atomic units: 50
	int cycles = json_params["Cycles"]; //Number of cycles for the laser pulse: 15
	
	
	//Formats for outputting matrices
	Eigen::IOFormat outformat(Eigen::FullPrecision,Eigen::DontAlignCols,", ","\n","(","),"," = npy.array((\n","\n))\n",' ');
	Eigen::IOFormat outformatLine(Eigen::FullPrecision,Eigen::DontAlignCols,", "," ","(","),"," = npy.array((\n","\n))\n",' ');

	//Initialize MPI
	MPI_Init(NULL,NULL);
	int wrank;
	int wsize;
	MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
	MPI_Comm_size(MPI_COMM_WORLD,&wsize);

	cerr<< "Simulation run parameters:\nSpline knots:" << Nsplines << "\nTime steps: " << Ntime
			<< "\nkappa max quantum number: " << Nkappa << "\nmu max quantum number: " << Nmu
			<< "\nBessel function l max: " << Nl << "\nBox radius: " << rBox << "\nIntensity: "  << Intensity << std::endl;

	cerr << "MPI world size: " << wsize << endl;
	cerr << "MPI rank: " << wrank << endl;
	cerr << "OpenMP threads: " << omp_get_max_threads() << endl;

	std::stringstream fnpstream;

	fnpstream << "dirBdp_E" << Intensity << "R" << Nsplines << "K" << Nkappa << "Mu" << Nmu << "r" << rBox << "T" << Ntime;
	
	string filenamePrefix = fnpstream.str();

	cvec psi = loadBinaryData(fname);
	// cout << psi;
	// return 0;
	
	//t: List of spline knots, here evenly distributed

	vec t(Nsplines+1);
	vec y(Nsplines+1);
	for(int i = 0; i < Nsplines+1; i++) {
		t[i] = rBox/(Nsplines)*i;
	}
	
	beyondDipolePulse bdpp(Intensity,50,15);
	
	dkbbasis dkbb(t,7);
	dkbb.setState(-1);
	lvec x = dkbb.glpts();
	
	
	clsmat& splch = dkbb.splineCache(x);
	clsmat& dsplch = dkbb.dSplineCache(x,1);
	clsmat& ddsplch = dkbb.dSplineCache(x,2);
	
	
	spnrbasis spnrb(Nkappa,Nmu);
	spnrb.bdplOverride(Nl);
	
	// // for(int i = 0; i < 4; i++) {
		// // cout << "S" << i << " " << cmat(spnrb.olcm(i)).format(outformat);
	// // }
	
	// // return 0;
	
	// bdpp.setTime(1.75);
	
	
	// double dt = (0.6*PI)/Ntime;
	// double T = 0;
	
	dirbs rthphb(dkbb,spnrb);
	
	vec angInit = vec::Zero(rthphb.angqN());
	angInit[0] = 1.0;
	
	rthphb.pruneUncoupled(angInit,true);
	
	
	dirwf psi1 = dirwf(rthphb,psi);
	// psi1.normalize();
	// cout << "psi " << psi1.coefs.format(outformat)<<"\n";
	// // psi1.save("savetest");
	// // return 0;
	
	// dkbb.Sm();
	// // for(int l = 0; l < spnrb.bdplmax(); l++) {
		// // dkbb.bdpam(1,1,1,l,bdpp);
		// // dkbb.bdpam(1,1,-1,l,bdpp);
	// // }
	
	// dkbb.setState(-1);
	
	
	// dkbb.olcm(1,1,0);
	// dkbb.olcm(1,1,2);
	
	// // cvec rtestvec = cvec::Constant(dkbb.radqN(),1.0);
	// rthphb.blockDistribute2();	
	// dkbb.dpam(1,1,1);
	// dkbb.dpam(1,1,-1);
	
	// // cvec azv = rthphb.template matfree<dpa>(psi1.coefs);
	
	// // cout << "azv "  << azv.format(outformat) << std::endl;
	// // cout << "az" << cvec(psi1.coefs).dot(azv);
	
	// // return 0;
	
	// // cout << "Sm " << cmat(dkbb.Sm()).format(outformat);
	// // cout << "Pm " << cmat(dkbb.olcm(0,0,0)).format(outformat);
	// // cout << "Qm " << cmat(dkbb.olcm(0,0,2)).format(outformat);
	
	// // cvec SUv = cvec::Zero(dkbb.radqN());
	// // cvec SLv = cvec::Zero(dkbb.radqN());
	
	// // dkbb.template cacheMatfree<SU>(rtestvec,SUv,0,1.0,0,0,1);
	// // dkbb.template cacheMatfree<SL>(rtestvec,SLv,0,1.0,0,0,1);
	
	// // cout << "SUv " << SUv.format(outformat) << std::endl;
	// // cout << "SLv " << SLv.format(outformat) << std::endl;
	// // // return 0;
	
	using Htype = DiracBDP<dirbs>;
	
	Htype H(rthphb,bdpp);
	
	H.Vfunc = &coloumb<Z>;
	
	dkbb.p1m();
	dkbb.Sm();
	dkbb.ulcm();
	dkbb.km();
	dkbb.Em(&coloumb<Z>);
	H.H0radprep();
	
	H.loadEigs();
	H.savePsievs(psi1,"psiev_processtest");
	
	// H.prepeigsLowMem(Nsplines,Nsplines/2, true);
	
	// H.savePsievs(psi1,"psiev");
	
	// // vector<vec>& evals = H.getevals();
	// // vector<dsmat>& evecs = H.getevecs();
	
	// // cout << "psi" << psi1.coefs.format(outformat) << std::endl;
	
	// // // cout << "evc0" << cmat(evecs[0]).format(outformat) << std::endl;
	
	// // // cout << "Filename: " << argv[1] << "\n";
	// // cout << "psi: (" << psi1.coefs.rows() << ", " << psi1.coefs.cols() << ")\n";
	// // cout << "evecs: (" <<  evecs[0].rows() << ", " << evecs[0].cols() << ")" << std::endl;
	
	// // ofstream psievf("psievReproject");
	// // psievf << "psit " << psi1.coefs.format(outformat) << "\n";

	// // psievf << "evl"   << filenamePrefix << " = npy.zeros((" << evecs.size() << "),dtype = object)\n";
	// // psievf << "psiev" << filenamePrefix << " = npy.zeros((" << evecs.size() << "),dtype = object)\n";

	// // for(int i = 0; i < evecs.size(); i++) {
		// // psievf << "evl" << filenamePrefix << "["<<i<<"]" << evals[i].format(outformat) << std::endl;
		// // psievf << "psiev" << filenamePrefix << "["<<i<<"] " << cmat(psi1 * evecs[i]).format(outformat) << std::endl;
	// // }

	// // psievf.close();
	
	// // cout << "evlDirBdp_R"<<Nsplines<<"K"<<Nkappa<<"Mu"<<Nmu<<"l"<<Nl<<"r"<<int(rBox)<<"= npy.zeros(("<<evecs.size()<<"),dtype=object)" << std::endl;
	// // cout << "psievDirBdp_R"<<Nsplines<<"K"<<Nkappa<<"Mu"<<Nmu<<"l"<<Nl<<"r"<<int(rBox)<<"= npy.zeros(("<<evecs.size()<<"),dtype=object)" << std::endl;
	// // // cout << "psievDirBdp4_R"<<Nsplines<<"K"<<Nkappa<<"Mu"<<Nmu<<"l"<<Nl<<"r"<<int(rBox)<<"= npy.zeros(("<<evecs.size()<<","<<4<<"),dtype=object)" << std::endl;
	
	// // for(int i = 0; i < evecs.size(); i++) {
		// // cout << "evlDirBdp_R"<<Nsplines<<"K"<<Nkappa<<"Mu"<<Nmu<<"l"<<Nl<<"r"<<int(rBox)<<"["<<i<<"]" << evals[i].format(outformat) << std::endl;
		// // cout << "psievDirBdp_R"<<Nsplines<<"K"<<Nkappa<<"Mu"<<Nmu<<"l"<<Nl<<"r"<<int(rBox)<<"["<<i<<"]" << cmat(psi1 * evecs[i]).format(outformat) << std::endl;
		// // // for(int j = 0; j < 4; j++) {
			
			// // // cout << "psievDirBdp4_R"<<Nsplines<<"K"<<Nkappa<<"Mu"<<Nmu<<"l"<<Nl<<"r"<<int(rBox)<<"["<<i<<"]["<<j<<"]" << (cmat(rthphb.template matfree<compS>(psi1.coefs,j).transpose() * evecs[i])).format(outformat) << std::endl;
		// // // }
	// // }
	
	return 0;
}

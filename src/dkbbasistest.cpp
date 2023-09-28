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
#include "defs.h"
#include "potential.h"   // Defines all laser pulse potentials
#include "dkbbasis.h"    // Dual kinetic balance basis


#define Z 1

//These global variables are the consequence of unfortunate silliness in how Bessel functions are applied during the construction of the interaction Hamiltonian. They stop being relevant once matrix elements are constructed
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


int main(int argc, char* argv[]) {
  // Check argument count
  if(argc != 2) {
      std::cerr << "Usage: " << argv[0] << " <path/to/parameters.json>\n";
      return 1;
  }

  // Get the JSON file path from command line arguments
  std::string json_file_path = argv[1];

  // Open and read the JSON file
  std::ifstream input_file(json_file_path);
  if (!input_file.is_open()) {
      std::cerr << "Could not open the file " << json_file_path << "\n";
      return 1;
  }

  // Parse JSON parameters
  nlohmann::json json_params;
  input_file >> json_params;
	
	// Simulation parameters
	int Nsplines = json_params["Nsplines"]; // Radial resolution, typical values: 200-250
	int Ntime = json_params["Ntime"]; // Number of time steps, typical values: 8000-20000
	int Nkappa = json_params["Nkappa"]; // Maximum absolute value of kappa quantum number, typical values: >= 16
	int Nmu = json_params["Nmu"]; // Typically 10 for nondipole simulations, as higher values of mu are suppressed by orders of magnitude
	int Nl = json_params["Nl"]; // Optimal value depends on radius of the box, typically 10 is sufficient
	double rBox = json_params["rBox"]; // Radius of the system in atomic units; 30 is usually sufficient
	int Intensity = json_params["Intensity"]; //Intensity of the laser pulse in atomic units: 10-500
	int omega = json_params["Omega"]; //Frequency of the laser pulse in atomic units: 50
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

	cout<< "Simulation run parameters:\nSpline knots:" << Nsplines << "\nTime steps: " << Ntime
		<< "\nkappa max quantum number: " << Nkappa << "\nmu max quantum number: " << Nmu
		<< "\nBessel function l max: " << Nl << "\nBox radius: " << rBox << "\nIntensity: "  << Intensity << std::endl;

	cout << "MPI world size: " << wsize << endl;
	cout << "MPI rank: " << wrank << endl;
	cout << "OpenMP threads: " << omp_get_max_threads() << endl;

	//t: List of spline knots, here evenly distributed

	std::stringstream fnpstream;

	fnpstream << "dirBdp_E" << Intensity << "R" << Nsplines << "K" << Nkappa << "Mu" << Nmu << "r" << rBox << "T" << Ntime;

	string filenamePrefix = fnpstream.str();

	vec t(Nsplines+1);
	vec y(Nsplines+1);
	for(int i = 0; i < Nsplines+1; i++) {
		t[i] = rBox/(Nsplines)*i;
	}

	//dkbbasis: 2-compoent basis P,Q-doublet of r-dependent Bspline fucntions
	dkbbasis dkbb(t,7);
	dkbb.setState(-1);
	lvec x = dkbb.glpts();

	//Cache splines, first derivatives of splines and second derivatives of splines
	clsmat& splch = dkbb.splineCache(x);
	clsmat& dsplch = dkbb.dSplineCache(x,1);
	clsmat& ddsplch = dkbb.dSplineCache(x,2);

	//spnrbasis, spinor basis of X_{kappa,mu},X_{-kappa,mu}-doublets
	spnrbasis spnrb(Nkappa,Nmu);

	//bdplOverride limits the number of l terms in the Bessel expansion of the interaction Hamiltonian
	spnrb.bdplOverride(Nl);

	int Nth = spnrb.angqN();


	cmat bdpams = cmat::Zero(spnrb.angqN(),spnrb.angqN());

	//Construct laser pulse with desired parameters
	beyondDipolePulse bdpp(Intensity,omega,cycles);

	//This is for verifying that things work before starting the simulation
	double dt = (0.6*PI)/Ntime;
	double T = 1.75;
	bdpp.setTime(T);
	
	//This would normnally be done by rthphbasis::blockDistribute2()
	dkbb.slp(0,spnrb.angqN(),0,Nl);

	//Construct nondipole alpha matrices, both upper and lower versions
	for(int l = 0; l < Nl; l++) {
		if (wrank == 0) cout << l << endl;
		dkbb.bdpam(1,1,1,l,bdpp);
		dkbb.bdpam(1,1,-1,l,bdpp);
	}


	//params list, needed to cache kappasmat in dkbbasis, for this test we just use an array that counts from 0 to spnrb.angqN()
	
	cvec params = cvec::Zero(spnrb.angqN());
	for(int i = 0; i < spnrb.angqN(); i++) {
		params[i] = i;
	}

	cmat outm = cmat::Zero(dkbb.radqN(), spnrb.angqN());
	
	//We just use Ntime for the number of test iterations, because anarchy reigns
	for(int i = 0; i < Ntime; i++) {
		cmat inmat = cmat::Random(dkbb.radqN(), spnrb.angqN());
		for(int l = 0; l < Nl; l++) {
			dkbb.template matmat<bdpa>(inmat,outm,spnrb.bdpam(LOWER,l),params,l,LOWER);
			dkbb.template matmat<bdpa>(inmat,outm,spnrb.bdpam(UPPER,l),params,l,UPPER);
		}
		cout << i << "/" << Ntime << endl;
	}
	
	MPI_Finalize();
	return 0;
}

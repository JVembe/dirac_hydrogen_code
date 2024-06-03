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

//These global variables are the consequence of unfortunate silliness in how Bessel functions are applied during the construction of the interaction Hamiltonian. They stop being relevant once matrix elements are constructed
int beyondDipolePulse::l = 1;
int beyondDipoleCarrierPulse::l = 1;

template <typename MatrixType>
void dense_write(const char* fname, MatrixType& mat) {
	const typename MatrixType::Scalar *Ax;
	
	typename MatrixType::Index rows = mat.rows();
	typename MatrixType::Index cols = mat.cols();
	
	FILE *fd;
	fd = fopen(fname, "w+");
	
	Ax = mat.data();
	
    if(!fd) {
        fprintf(stderr, "cant open %s\n", fname);
        exit(0);
    }
	
	size_t fsize, nread;
	//Storage format: rows, cols, data
	fwrite(&rows, sizeof(typename MatrixType::Index), 1, fd);
	fwrite(&cols, sizeof(typename MatrixType::Index), 1, fd);
    fwrite(Ax, sizeof(csmat::Scalar), rows*cols, fd);
	
    fclose(fd);
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
	double dt = 0.125;
	double T = 1.00;
	bdpp.setTime(T);

	//Construct basis for Hamiltonian
	dirbs rthphb(dkbb,spnrb);

	//Prune uncoupled states, i.e. states that are excluded by selection rules
	//This involves an expensive estimation of the matrix exponential which could probably be replaced by something else
	vec angInit = vec::Zero(rthphb.angqN());
	angInit[0] = 1.0;
	// vec angInit = vec::Constant(rthphb.angqN(),1.0);

	rthphb.pruneUncoupled(angInit,true); //true means using nondipole couplings
	
	
	//Dump enumeration of angular momentum states, this is needed in postprocessing of data
	for(int i = 0; i < rthphb.angids.size(); i++) {
		cout << "(" << i << ", " << rthphb.angids[i] << ", " << ik(rthphb.angids[i])<< ", " << imu(rthphb.angids[i]) << ")," << std::endl;
	}
	
	// for(int i = 0; i < rthphb.angidsReverse.size(); i++) {
		// cout << "(" << i << ", " << rthphb.angidsReverse[i] << ")," << std::endl;
	// }

	//Construct nondipole alpha matrices, both upper and lower versions
	for(int l = 0; l < spnrb.bdplmax(); l++) {
		if (wrank == 0) cout << l << endl;
		dkbb.bdpam(1,1,1,l,bdpp);
		dkbb.bdpam(1,1,-1,l,bdpp);
	}


	using Htype = DiracBDP<dirbs>;
	//Initialize Hamiltonian and set Coulomb potential
	Htype H(rthphb,bdpp);
	H.Vfunc = &coloumb<Z>;

	
	//Assemble H0 for propagation


	//blockDistribute2 is a sort of hacked together attempt to load balance the matrix-vector product for the time evolution operator
	//see rthphbasis.h for code. Currenlty definitely far from optimal.
	rthphb.blockDistribute2();

	// cout << "coefsE0:" << endl << coefsE0;
	//Get eigenvalues and eigenvectors
	H.prepeigsLowMem(Nsplines,Nsplines/2, true);

	H.H0radprep();
	//psi1 set from ground state and normalized
	
	cvec coefsE0 = H.getevec(Nsplines+5,-1,-0.5);
	

	dirwf psi1 = dirwf(rthphb,coefsE0);
	psi1.normalize();
	
	// H.eigProj(psi1);
	
	
	
	
	cvec testvec = cvec::Constant(rthphb.radqN()*rthphb.angqN(),1.0);

	cvec b;

	int Nr = rthphb.radqN();

	b = H.S(testvec) - dt * cdouble(0,0.5) * H.H(T,testvec);

	// cvec Htv = rthphb.bdpamatvecMPIblock(testvec);
	// cvec H0v = H.H0(testvec);
	// cvec Sv  = H.S(testvec);
	
	// cout << "Htvtst " << Htv.format(outformat);
	// cout << "H0vtst " << H0v.format(outformat);
	// cout << "Svtst "  << Sv.format(outformat);
	// return 0;

	//Due to a quirk in Eigen I haven't been able to figure out, if I don't initializze the solver here and attach it to the propagator myself, it takes more iterations
	Eigen::ParBiCGSTAB<RtsMat<Htype >,SubmatPreconditioner<cdouble> > solver;

	//RtsMat is a helper class needed by the preconditioner, and functions as a wrapper to the Hamiltonian class that masks the time step formula as a matrix-vector product.
	//RtsMat.h is quite p ossibly the most haunted file in this code. I do not understand how it works.
	RtsMat<Htype> proptest;

	//RtsMat needs setup
	proptest.setDt(dt);
	proptest.setTime(T);
	proptest.attachHamiltonian(H);
	proptest.overrideRows(psi1.coefs.rows());
	proptest.overrideCols(psi1.coefs.rows());

	//Typically, if it fails to converge within 1000 iterations there's trouble



	//Holdover from old code. Either does nothing or essential. I don't dare to remove it.
	H.getBasis().getRadial().setState(0);

	//Preconditioner setup performs ILUT factorizations of individual blocks of H0
	solver.preconditioner().setup(H,dt);
	//Eigen solvers need a compute step to be performed before solving
	solver.compute(proptest);

	// solver.setMaxIterations(1000);

	cvec psi2 = solver.solve(b);
	
	dense_write("presolve",b);
	dense_write("postsolve",psi2);
	
	// cvec psi2_mpi = solver.preconditioner().MPIsolve(b);

	// cout << "psi2" << psi2.format(outformat) << endl;
	// cout << "psi2_mpi" << psi2_mpi.format(outformat) << endl;


	// cout << "iterations: "  << solver.iterations();

	// cout << "Rtsv" << (proptest * testvec).format(outformat) << endl;

	// MPI_Finalize();
	// return 0;
	// //Initialize crank-nicholson propagator
	Cranknich<Htype,dirbs,true> cnp(H);



	cnp.proptest = &proptest;
	cnp.solver = &solver;

	cnp.setDumpfile((filenamePrefix + "_dump"));

	cnp.propagate(psi1,dt,Ntime,1);

	dirwf wft = cnp.wft;

	if(wrank==0)
	// cout << "wft" << wft.coefs.format(outformat);
	wft.save(filenamePrefix + "_psit",0);
	H.savePsievs(wft,filenamePrefix + "_psiev");
	
	MPI_Finalize();
	return 0;
}
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


//These global variables are the consequence of unfortunate silliness in how Bessel functions are applied during the construction of the interaction Hamiltonian. They stop being relevant once matrix elements are constructed
int beyondDipolePulse::l = 1;
int beyondDipoleCarrierPulse::l = 1;

void csr_write(const char*fname, csmat& mat)
{
    const csmat::StorageIndex *Ap, *Ai;
    const csmat::Scalar *Ax;

    csmat::StorageIndex nnz = mat.nonZeros();
    csmat::StorageIndex dim = mat.rows();

    // make sure we have compressed storage
    mat.makeCompressed();

    Ap = mat.outerIndexPtr();
    Ai = mat.innerIndexPtr();
    Ax = mat.valuePtr();

    FILE *fd;
    fd = fopen(fname, "w+");
    if(!fd) {
        fprintf(stderr, "cant open %s\n", fname);
        exit(0);
    }
    
    size_t fsize, nread;

    // storage format: dim, nnz, Ap, Ai, Ax
    fwrite(&dim, sizeof(csmat::StorageIndex), 1, fd);
    fwrite(&nnz, sizeof(csmat::StorageIndex), 1, fd);
    fwrite(Ap, sizeof(csmat::StorageIndex), dim+1, fd);
    fwrite(Ai, sizeof(csmat::StorageIndex), nnz, fd);
    fwrite(Ax, sizeof(csmat::Scalar), nnz, fd);

    fclose(fd);
}

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
		fwrite(Ax, sizeof(typename MatrixType::Scalar), rows*cols, fd);
        
    fclose(fd);
}

void printSparseNonZeros(const csmat& mat) {
    cerr << mat.nonZeros() << "\n";
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
  if(argc < 2 || argc > 3) {
      std::cout << "Usage: " << argv[0] << "<path/to/parameters.json>\n";
      return 1;
  }

  // Get the JSON file path from command line arguments
  std::string json_file_path;
  json_file_path = argv[1];

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
        double omega = json_params["Omega"]; //Frequency of the laser pulse in atomic units: 50
        int cycles = json_params["Cycles"]; //Number of cycles for the laser pulse: 15
        int Z = json_params["Z"]; //Nuclear charge
        
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

        //t: List of spline knots, here evenly distributed

        std::stringstream fnpstream;

        fnpstream << "dirBdp_E" << Intensity << "R" << Nsplines << "K" << Nkappa << "Mu" << Nmu << "r" << rBox << "T" << Ntime;

        string filenamePrefix = fnpstream.str();

        vec t(Nsplines+1);
        vec y(Nsplines+1);
        for(int i = 0; i < Nsplines+1; i++) {
                t[i] = rBox/(Nsplines)*i;
        }


        //spnrbasis, spinor basis of X_{kappa,mu},X_{-kappa,mu}-doublets
        spnrbasis spnrb(Nkappa,Nmu);
        //bdplOverride limits the number of l terms in the Bessel expansion of the interaction Hamiltonian
        spnrb.bdplOverride(Nl);

        {
            
            dkbbasis dkbb(t,7);
            dkbb.setState(-1);
            lvec x = dkbb.glpts();

            //Cache splines, first derivatives of splines and second derivatives of splines
            clsmat& splch = dkbb.splineCache(x);
            clsmat& dsplch = dkbb.dSplineCache(x,1);
            clsmat& ddsplch = dkbb.dSplineCache(x,2);
                
            //Construct laser pulse with desired parameters
            beyondDipolePulse bdpp(Intensity,omega,cycles);

            
            // Bear with me here, the scheme for assembling the h-matrices is a bit silly
            dirbs rthphb(dkbb,spnrb);
            //pruneUncoupled is slow and unnecessary for this
			cout << "Constructing Hamiltonian" << endl;
            using Htype = DiracBDP<dirbs>;
            Htype H(rthphb,bdpp);
			coulomb clp(Z);
            H.Vfunc = &clp;
            //cout << "Diagonalizing Hamiltonian" << endl;
			dkbb.p1m();
			dkbb.Sm();
			dkbb.ulcm();
			dkbb.km();
			dkbb.Em(&clp);
			H.H0radprep();
	    
			H.prepeigsLowMem(Nsplines,Nsplines/2, true,0,true);
            // return 0;
			//Write all eigenvalues and eigenvectors to files
			for(int i = 0; i < H.kappas.size(); i++) {
				int kappa = H.kappas[i];
				char fevl[256];
				snprintf(fevl, 255,"evl%i.mat", kappa);
				char fevc[256];
				snprintf(fevc,255,"evc%i.mat",kappa);
				vec evls = H.getevals(kappa);
				cmat evcs = H.getevecs(kappa,-0.5);
				dense_write(fevl,evls);
				dense_write(fevc,evcs);
			}
            // vec evls0 = H.getevals(-1);
            // //To find the ground state we identify the index of the eigenvalue closest to the ground state energy, -0.500007
                        
            // vec::Index e0idx;
                        
            // double e0 = vec(evls0+vec::Constant(evls0.rows(),0.500007)).array().abs().minCoeff(&e0idx);
                        
            // // cout << evls0;
            // cout << "e0: " << evls0[e0idx] << ", idx: " << e0idx << endl;
                        
            // cvec evc0 = H.getevec(e0idx,-1,-0.5);
                        
            // dense_write("psi0",evc0);
        }
        MPI_Finalize();
        return 0;
}

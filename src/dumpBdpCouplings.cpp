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
    fwrite(Ax, sizeof(csmat::Scalar), rows*cols, fd);
        
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
            for(int l = 0; l < spnrb.bdplmax(); l++) {
                char fname[256];
                snprintf(fname, 255, "H0l%d.csr", l);
                csr_write(fname, spnrb.bdpam(1,l));
                snprintf(fname, 255, "H1l%d.csr", l);
                csr_write(fname, spnrb.bdpam(-1,l));
            }

            //dkbbasis: 2-compoent basis P,Q-doublet of r-dependent Bspline fucntions
            dkbbasis dkbb(t,7);
            dkbb.setState(-1);
            lvec x = dkbb.glpts();

            //Cache splines, first derivatives of splines and second derivatives of splines
            clsmat& splch = dkbb.splineCache(x);
            clsmat& dsplch = dkbb.dSplineCache(x,1);
            clsmat& ddsplch = dkbb.dSplineCache(x,2);
                
            //Construct laser pulse with desired parameters
            beyondDipolePulse bdpp(Intensity,omega,cycles);

            //Compute bdp-alpha matrices, this stores g0...g3 in memory
            for(int l = 0; l < spnrb.bdplmax(); l++) {
                if (wrank == 0) cout << l << endl;
                dkbb.bdpam(1,1,1,l,bdpp);
                dkbb.bdpam(1,1,-1,l,bdpp);
            }
                        
            // for efficiency of assembly we want the g matrices to have the same non-zero pattern,
            // even at the expense of storing some zero entries
            int init = 0;
            csmat base_nnz_pattern;
            for(int n = 0; n < 4; n++) {
                
                for(int alpha = 0; alpha < 6; alpha++) {
                    for(int l = 0; l < spnrb.bdplmax(); l++) {
                        char fname[256];
                        snprintf(fname, 255, "g%da%dl%d.csr", n, alpha, l);

                        // we assume that the first g matrix has the non-zero pattern
                        // that includes all subsequent matrices. It seems g0* have such property
                        if(!init) {
                            init = 1;
                            base_nnz_pattern = 0*dkbb.getbdpmat(n,l,alpha);
                        }
                        
                        // add base nnz pattern
                        csmat gmat = dkbb.getbdpmat(n,l,alpha) + base_nnz_pattern;
                        csr_write(fname, gmat);

			printf("Writing g matrix g%da%dl%d out of %d,%d,%d\n", n, alpha, l,4,6,spnrb.bdplmax());
		        cout << endl;	
                    }
                }
            }
            cout << "All g matrices written, moving on" << endl;
            // Bear with me here, the scheme for assembling the h-matrices is a bit silly
            dirbs rthphb(dkbb,spnrb);
            //pruneUncoupled is slow and unnecessary for this
	    //vec angInit = vec::Zero(rthphb.angqN());
            //angInit[0] = 1.0;
            //rthphb.pruneUncoupled(angInit,true);
	    cout << "Constructing Hamiltonian" << endl;
            using Htype = DiracBDP<dirbs>;
            Htype H(rthphb,bdpp);
            H.Vfunc = &coloumb<Z>;
            //cout << "Diagonalizing Hamiltonian" << endl;
	    dkbb.p1m();
	    dkbb.Sm();
	    dkbb.ulcm();
	    dkbb.km();
	    dkbb.Em(&coloumb<Z>);
	    H.H0radprep();
	    
            
                        
            //Dump time-independent part
                        
            for(int n = 0; n < 4; n++) {
                char fname[256];
                snprintf(fname,255,"h0%d.csr",n);
                csmat hmat = dkbb.getH0mat(n) + base_nnz_pattern;
                csr_write(fname,hmat);
		printf("Writing h matrix %d\n",n);
            }
        
            //And finally the overlap matrix blocks as well, s0...s2
            csmat s0m = dkbb.get0mat<S>()  + base_nnz_pattern;
            csmat s1m = dkbb.getkmat<S>()  + base_nnz_pattern;
            csmat s2m = dkbb.getkkmat<S>() + base_nnz_pattern;
            csr_write("s0.csr",s0m);
            csr_write("s1.csr",s1m);
            csr_write("s2.csr",s2m);
            cout << "Diagonalizing Hamiltonian" << endl;
	    H.prepeigsLowMem(Nsplines,Nsplines/2, true,-1);
                
            vec evls0 = H.getevals(-1);;
            //To find the ground state we identify the index of the eigenvalue closest to the ground state energy, -0.500007
                        
            vec::Index e0idx;
                        
            double e0 = vec(evls0+vec::Constant(evls0.rows(),0.500007)).array().abs().minCoeff(&e0idx);
                        
            // cout << evls0;
            cout << "e0: " << evls0[e0idx] << ", idx: " << e0idx << endl;
                        
            cvec evc0 = H.getevec(e0idx,-1,-0.5);
                        
            dense_write("psi0",evc0);
        }
        MPI_Finalize();
        return 0;
}

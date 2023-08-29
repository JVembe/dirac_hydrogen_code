//The design of this code is intended to be modular

//The success of this is debatable as only a small number of configurations ever see use

#include "potential.h"   //Defines all laser pulse potentials used in these simulations
#include "RtsMat.h"      //Helper class needed for preconditioners
#include "dkbbasis.h"    //Dual kinetic balance basis
#include "rthphbasis.h"  //Combined basis for spherical coordinates
#include "wavefunc.h"    //Effectively a wrapper class for the Eigen vector class
#include "hamiltonian.h" //Contains the physics in the Hamiltonian class and derived classes
#include "propagator.h"  //Time evolution code
#include <omp.h>		 //This is openMP
#include "mpiFuncs.h"    //Functions for sending and receiving Eigen matrices with MPI
#include <chrono>

#define INTENSITY 10
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

int main() {
	//Define types for basis and wave function from templates
	//rthphbasis "R, theta, phi basis" combines dkbbasis and spnrbasis to model system in spherical coordinates
	using dirbs = rthphbasis<dkbbasis,spnrbasis>;
	using dirwf = wavefunc<dirbs>;
	
	//Simulation parameters
	
	int Nsplines = 250; 
	//Technically not the number of splines, but the number of knots. In practice this acts as the radial resolution of the simulation.
	//Typical values used to get sufficient accuracy have been in the 200-250 range
	
	int Ntime = 10; 
	//Number of time steps to do. Typical values have been in the 8000-20000 range
	
	int Nkappa = 20;
	//Maximum absolute value of the kappa quantum number, which ranges from -Nkappa to +Nkappa, not including 0. 
	//Typically need this to be 16 or higher.
	
	int Nmu = 10;
	//If 0, the mu quantum number follows selection rules and may only take half integer values in the interval{-|kappa|,|kappa|}
	//High values typically aren't important, so the option of a cutoff has been implemented to reduce the problem size
	
	int Nl = 10;
	//For spatial dependence we rely on a Bessel function expansion, this parameter limits the number of terms.
	
	double rBox = 30.0;
	//Radius of the system, in atomic units. 30 is very small, but works for our purposes
	
	//Formats for outputting matrices
	Eigen::IOFormat outformat(Eigen::FullPrecision,Eigen::DontAlignCols,", ","\n","(","),"," = npy.array((\n","\n))\n",' ');
	Eigen::IOFormat outformatLine(Eigen::FullPrecision,Eigen::DontAlignCols,", "," ","(","),"," = npy.array((\n","\n))\n",' ');
	
	//Initialize MPI
	MPI_Init(NULL,NULL);
	int wrank;
	MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
	
	cout<< "Simulation run parameters:\nSpline knots:" << Nsplines << "\nTime steps: " << Ntime 
		<< "\nkappa max quantum number: " << Nkappa << "\nmu max quantum number: " << Nmu 
		<< "\nBessel function l max: " << Nl << "\nBox radius: " << rBox << "\nIntensity: "  << INTENSITY << std::endl;
	
	//t: List of spline knots, here evenly distributed
	
	std::stringstream fnpstream;
	
	fnpstream << "dirBdp_E" << INTENSITY << "R" << Nsplines << "K" << Nkappa << "Mu" << Nmu << "r" << rBox << "T" << Ntime;
	
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
	// spnrb.bdplOverride(Nl);
	
	int Nth = spnrb.angqN();
	
	
	cmat bdpams = cmat::Zero(spnrb.angqN(),spnrb.angqN());
	
	//Construct laser pulse with desired parameters
	beyondDipolePulse bdpp(INTENSITY,50,15);

	//This is for verifying that things work before starting the simulation
	double dt = (0.6*PI)/Ntime;
	double T = 1.75;
	bdpp.setTime(T);
	
	//Construct basis for Hamiltonian
	dirbs rthphb(dkbb,spnrb);
	
	//Prune uncoupled states, i.e. states that are excluded by selection rules
	//This involves an expensive estimation of the matrix exponential which could probably be replaced by something else
	vec angInit = vec::Zero(rthphb.angqN());
	angInit[0] = 1.0;
	// vec angInit = vec::Constant(rthphb.angqN(),1.0);
	
	rthphb.pruneUncoupled(angInit,true);
	
	//Dump enumeration of angular momentum states, this is needed in postprocessing of data
	for(int i = 0; i < rthphb.angids.size(); i++) {
		cout << "(" << i << ", " << rthphb.angids[i] << ", " << ik(rthphb.angids[i])<< ", " << imu(rthphb.angids[i]) << ")," << std::endl;
	}
	
	
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
	
	
	
	//Eigenvalue solution of time-independent part of Hamiltonian
	H.prepeigs(Nsplines,Nsplines/2);
	//Assemble H0 for propagation
	H.H0radprep();
	
	//Get eigenvalues and eigenvectors
	vector<vec>& evals = H.getevals();
	vector<dsmat>& evecs = H.getevecs();
	
	//psi1 set from ground state and normalized
	//blockDistribute2 is a sort of hacked together attempt to load balance the matrix-vector product for the time evolution operator
	//see rthphbasis.h for code. Currenlty definitely far from optimal.
	cvec coefsE0 = rthphb.blockDistribute2(evecs[0].col(Nsplines+5));
	
	dirwf psi1 = dirwf(rthphb,coefsE0);
	psi1.normalize();
	
	cvec testvec = cvec::Constant(rthphb.radqN()*rthphb.angqN(),1.0);
	
	cvec b;

	int Nr = rthphb.radqN();
	
	b = H.S(testvec) - dt * cdouble(0,0.5) * H.H(T,testvec);

	
	//Due to a quirk in Eigen I haven't been able to figure out, if I don't initializze the solver here and attach it to the propagator myself, it takes more iterations
	Eigen::ParBiCGSTAB<RtsMat<Htype >,SubmatPreconditioner<cdouble> > solver;
	
	//RtsMat is a helper class needed by the preconditioner, and functions as a wrapper to the Hamiltonian class that masks the time step formula as a matrix-vector product.
	//RtsMat.h is quite p ossibly the most haunted file in this code. I do not understand how it works.
	RtsMat<Htype> proptest;
	
	//RtsMat needs setup
	proptest.setDt(dt);
	proptest.setTime(T);
	proptest.attachHamiltonian(H);

	//Typically, if it fails to converge within 1000 iterations there's trouble

	
	
	//Holdover from old code. Either does nothing or essential. I don't dare to remove it.
	H.getBasis().getRadial().setState(0);
	
	//Preconditioner setup performs ILUT factorizations of individual blocks of H0
	solver.preconditioner().setup(H,dt);
	//Eigen solvers need a compute step to be performed before solving
	solver.compute(proptest);
	
	solver.setMaxIterations(1000);
	
	cvec psi2 = solver.solve(b);
	// cvec psi2_mpi = solver.preconditioner().MPIsolve(b);
	
	// cout << "psi2" << psi2.format(outformat) << endl;
	// cout << "psi2_mpi" << psi2_mpi.format(outformat) << endl;
	
	
	cout << "iterations: "  << solver.iterations();
	
	// cout << "Rtsv" << (proptest * testvec).format(outformat) << endl;
	
	// MPI_Finalize();
	// return 0;
	// //Initialize crank-nicholson propagator
	Cranknich<Htype,dirbs,true> cnp(H);
	
	
	
	cnp.proptest = &proptest;
	cnp.solver = &solver;
	
	// cnp.setDumpfile((filenamePrefix + "_dump"));
	
	cnp.propagate(psi1,(0.6*PI)/8000,Ntime,1);
	
	dirwf wft = cnp.wf[Ntime/1 - 1];
	
	if(wrank==0)
	cout << "wft" << wft.coefs.format(outformat);
	
	MPI_Finalize();
	return 0;
}
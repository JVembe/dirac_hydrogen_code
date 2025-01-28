/*
	GaDE - Gpu-accelerated solver for the time dependent Dirac Equation
	
    Copyright (C) 2025  Johanne Elise Vembe <johannevembe@gmail.com>
    Copyright (C) 2025  Marcin Krotkiewski <marcink@uio.no>
	Copyright (C) 2025  Hicham Agueny <hicham.agueny@uib.no>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/


//The design of this code is intended to be modular

//The success of this is debatable as only a small number of configurations ever see use

#include "potential.h"   //Defines all laser pulse potentials used in these simulations
#include "RtsMat.h"      //Helper class needed for preconditioners
#include "dkbbasis.h"    //Dual kinetic balance basis
#include "rthphbasis.h"  //Combined basis for spherical coordinates
#include "wavefunc.h"    //Effectively a wrapper class for the Eigen vector class
#include "hamiltonian.h" //Contains the physics in the Hamiltonian class and derived classes
#include "propagator.h"  //Time evolution code
// #include <omp.h>		 //This is openMP
#include "mpiFuncs.h"    //Functions for sending and receiving Eigen matrices with MPI
#include <chrono>
#include "parallelBiCGSTAB.h"

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
	int Nsplines = 200;
	int Ntime = 8000;
	int Nkappa = 20;
	int Nmu = 10;
	int Nl = 10;
	double rBox = 30.0;
	
	
	//Formats for outputting matrices
	Eigen::IOFormat outformat(Eigen::FullPrecision,Eigen::DontAlignCols,", ","\n","(","),"," = npy.array((\n","\n))\n",' ');
	Eigen::IOFormat outformatLine(Eigen::FullPrecision,Eigen::DontAlignCols,", "," ","(","),"," = npy.array((\n","\n))\n",' ');
	
	//Initialize MPI
	MPI_Init(NULL,NULL);
	Eigen::initParallel();
	int wrank;
	MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
	
	if(wrank == 0) {
		cout<< "Simulation run parameters:\nSpline knots:" << Nsplines << "\nTime steps: " << Ntime 
		<< "\nkappa max quantum number: " << Nkappa << "\nmu max quantum number: " << Nmu 
		<< "\nBessel function l max: " << Nl << "\nBox radius: " << rBox << "\nIntensity: "  << INTENSITY << std::endl;
	}
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
	spnrb.bdplOverride(Nl);
	
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
	
	//Allocate memory for cached matrix-vector products
	dkbb.prepbdpvecs(rthphb.angqN(), spnrb.bdplmax());
	dkbb.clvcs();
	
	//Dump enumeration of angular momentum states, this is needed in postprocessing of data
	if(wrank == 0) {
		for(int i = 0; i < rthphb.angids.size(); i++) {
			cout << "(" << i << ", " << rthphb.angids[i] << ", " << ik(rthphb.angids[i])<< ", " << imu(rthphb.angids[i]) << ")," << std::endl;
		}
	}
	
	//Construct dipole alpha matrix, both upper and lower versions
	// dkbb.dpam(1,1,1);
	// dkbb.dpam(1,1,-1);
	MPI_Barrier(MPI_COMM_WORLD);
	if(wrank == 0) cout << "Initializing matrices..." << endl;
	for(int l = 0; l < spnrb.bdplmax(); l++) {
		if (wrank == 0) cout << l << endl;
		dkbb.bdpam(1,1,1,l,bdpp);
		dkbb.bdpam(1,1,-1,l,bdpp);
	}
	
	// cout << dkbb.bdpam(1,1,1,0,bdpp).nonZeros() << endl;
	
	// MPI_Finalize();
	// return 0;
	using Htype = DiracBDP<dirbs>;
	//using Htype = Dirac<dirbs>;
	Htype H(rthphb,bdpp);
	//Initialize Hamiltonian and set Coulomb potential
	// Htype H(rthphb,&dplA<15,50,INTENSITY>);
	H.Vfunc = &coulomb<Z>;
	
	cvec testvec = cvec::Constant(rthphb.radqN()*rthphb.angqN(),1.0);
	cvec testsegs = rthphb.blockDistribute3(testvec);
	MPI_Finalize();
	return 0;
	
	//Eigenvalue solution of time-independent part of Hamiltonian
	
	MPI_Barrier(MPI_COMM_WORLD);
	if(wrank == 0) cout << "Diagonalizing..." << endl;
	H.prepeigs(Nsplines,Nsplines/2);
	//Assemble H0 for propagation
	H.H0radprep();
	//Get eigenvalues and eigenvectors
	vector<vec>& evals = H.getevals();
	vector<dsmat>& evecs = H.getevecs();
	
	//psi1 set from ground state and normalized
	
	
	// cout << "testvec: " << testvec << endl;
	
	cvec b;
	
	// cout << "Sv:\n" << H.S(testvec) << endl;
	
	// cout << "Hv:\n" << H.H(T,testvec) << endl;

	// MPI_Finalize();
	// return 0;

	int Nr = rthphb.radqN();
	
	// cout << "Sv" << H.S(testvec).format(outformat) << endl;
	// cout << "Sv_mpi" << rthphb.smatvecMPIblock(testvec).format(outformat) << endl;
	// cout << "H0v" << H.H0(testvec).format(outformat) << endl;
	// cout << "H0v_mpi" << rthphb.h0matvecMPIblock(testvec).format(outformat) << endl;
	
	MPI_Barrier(MPI_COMM_WORLD);
	if(wrank == 0) cout << "Distributing matrix blocks..." << endl;
	
	cvec coefsE0 = rthphb.blockDistribute3(evecs[0].col(Nsplines+5));
	MPI_Finalize();
	return 0;
	cvec coefsE0 = rthphb.blockDistribute2(evecs[0].col(Nsplines+5));
	
	dirwf psi1 = dirwf(rthphb,coefsE0);
	psi1.normalize();
	
	// cout << psi1.coefs << endl;
	
	// cmat testblks; 
	// testblks = testblocks.reshaped(Nr,testblocks.size()/Nr);
	
	// cout << "Htv" << rthphb.template matfree<bdpa>(testvec).format(outformat) << endl;
	
	// cout << "Htv_MPI " << rthphb.bdpamatvecMPIblock(testblks).format(outformat);
	
	
	// MPI_Finalize();
	// return 0;
	
	
	// cvec Htv = rthphb.template matfree<bdpa>(psi1.coefs);
	// cout << "Htv_mpi" << Htv.format(outformat) << endl;
	// cvec H0v = rthphb.template matfree<H0>(psi1.coefs);
	// cout << "H0v_mpi" << H0v.format(outformat) << endl;
	// cvec Sv  = rthphb.template matfree<S>(psi1.coefs);
	// cout << "Sv_mpi"  << Sv.format(outformat)  << endl;
	
	// cout << "|Htv| = " << Htv.squaredNorm() << endl;
	// cout << "|H0v| = " << H0v.squaredNorm() << endl;
	// cout << "|Sv| = " << Sv.squaredNorm() << endl;
	// MPI_Finalize();
	// return 0;
	RtsMat<Htype> proptest;
	// RtsMat<Htype> oldproptest;
	
	//RtsMat needs setup
	proptest.setDt(dt);
	proptest.setTime(T);
	proptest.attachHamiltonian(H);
	proptest.overrideRows(psi1.coefs.rows());
	proptest.overrideCols(psi1.coefs.rows());
	
	// oldproptest.setDt(dt);
	// oldproptest.setTime(T);
	// oldproptest.attachHamiltonian(H);
	// cout << "H.H(v)\n" << H.H(1.75,testblocks);
	// cout << "proptestv\n" << proptest * testblocks;
	
	
	
	
	bdpp.setTime(1.25);
	
	// for(int i = 0; i < 1000; i++) {
	// cout << "HIv_mpi" << rthphb.bdpamatvecMPIblock(testvec).format(outformat);
	// cout << "HIv_old" << rthphb.template matfree<bdpa>(testvec).format(outformat);
	// // }
	// // auto t2 = std::chrono::system_clock::now();
	
	// MPI_Finalize();
	// return 0;
	// cout << "calculating b..." << endl;
	// for(int i = 0; i < Ntime; i++) {	
	b = H.S(psi1.coefs) - dt * cdouble(0,0.5) * H.H(T,psi1.coefs);
		//b = H.Ht(T,psi1.coefs);
		// cmat outv;
		
		// allreduceMat(vblocks,outv);
		// cout << i << endl;
	
	// cout << "b"<<wrank<<" " << b.format(outformat) << endl;
	
	MPI_Barrier(MPI_COMM_WORLD);
	// cout << endl;
	// cmat bblock = b.reshaped(Nr,b.size()/Nr);
	
	
	
	// cvec bFull;
	
	// allgatherVec(bblock,bFull);
	
	// cout << "bFull " << bFull.format(outformat) << endl;
	
	
	// MPI_Finalize();
	// return 0;
	
	//Due to a quirk in Eigen I haven't been able to figure out, if I don't initializze the solver here and attach it to the propagator myself, it takes more iterations
	// Eigen::debugBiCGSTAB<RtsMat<Htype >, SubmatPreconditioner_old<cdouble> > oldSolver;
	Eigen::ParBiCGSTAB<RtsMat<Htype >,SubmatPreconditioner<cdouble> > solver;
	
	//RtsMat is a helper class needed by the preconditioner, and functions as a wrapper to the Hamiltonian class that masks the time step formula as a matrix-vector product.
	//RtsMat.h is quite p ossibly the most haunted file in this code. I do not understand how it works.
	

	//Typically, if it fails to converge within 1000 iterations there's trouble

	
	
	//Holdover from old code. Either does nothing or essential. I don't dare to remove it.
	H.getBasis().getRadial().setState(0);
	
	//Preconditioner setup performs ILUT factorizations of individual blocks of H0
	solver.preconditioner().setup(H,dt);
	// oldSolver.preconditioner().setup(H,dt);
	
	// cvec slvb = solver.preconditioner().solve(b);
	// cout << "slvb " << slvb.format(outformat) << endl;
	
	// cvec slvbFull = oldSolver.preconditioner().solve(bFull);
	// cout << "slvbFull " << slvbFull.format(outformat) << endl;
	// MPI_Finalize();
	// return 0;
	MPI_Barrier(MPI_COMM_WORLD);
	
	dkbb.clearOperatorMats();
	dkbb.clearCaches();
	// MPI_Finalize();
	// return 0;
	// //Eigen solvers need a compute step to be performed before solving
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	// cout << "Computing solvers" << endl;
	// oldSolver.compute(oldproptest);
	solver.compute(proptest);
	// cout << "setMaxIterations" << endl;
	// oldSolver.setMaxIterations(100);
	// solver.setMaxIterations(100);
	
	MPI_Barrier(MPI_COMM_WORLD);
	// cout << "solving oldSolver" << endl;
	// cvec psi2Full = oldSolver.solve(bFull);
	// cout << "solving solver" << endl;
	cvec psi2 = solver.solve(b);
	
	// // cvec psi2_mpi = solver.preconditioner().MPIsolve(b);
	MPI_Barrier(MPI_COMM_WORLD);
	
	// cout << "psi2" << psi2.format(outformat) << endl;
	
	// cout << "psi2Full" << psi2Full.format(outformat) << endl;
	
	// // cout << "psi2_mpi" << psi2_mpi.format(outformat) << endl;
	MPI_Barrier(MPI_COMM_WORLD);
	
	// cout << "oldSolver iterations: "  << oldSolver.iterations() << endl;
	cout << "iterations: "  << solver.iterations() << endl;
	
	// MPI_Finalize();
	// return 0;
	// // // cout << "Rtsv" << (proptest * testvec).format(outformat) << endl;
	
	// MPI_Finalize();
	// return 0;
	// // //Initialize crank-nicholson propagator
	Cranknich<Htype,dirbs,true> cnp(H);
	
	
	
	cnp.proptest = &proptest;
	cnp.solver = &solver;
	
	// cnp.setDumpfile((filenamePrefix + "_dump"));
	
	cnp.propagate(psi1,(0.6*PI)/Ntime,Ntime,1);
	
	cmat wft_local = cnp.wft.coefs.reshaped(Nr,cnp.wft.coefs.size()/Nr);//[Ntime - 1];
	
	// cout << "wft_local" << wft_local.format(outformat);
	
	cmat wftcoefs;
	
	allgatherVec(wft_local,wftcoefs);
	
	dirwf wft(rthphb,wftcoefs);
	
	// mat dPdE = H.dPdE(wft,5000,-0.500007,500);
	MPI_Barrier(MPI_COMM_WORLD);
	if(wrank == 0) {
		ofstream psievf("psiev");
		psievf << "psit " << wft.coefs.format(outformat) << "\n";

		psievf << "evl"   << filenamePrefix << " = npy.zeros((" << evecs.size() << "),dtype = object)\n";
		psievf << "psiev" << filenamePrefix << " = npy.zeros((" << evecs.size() << "),dtype = object)\n";

		for(int i = 0; i < evecs.size(); i++) {
			psievf << "evl" << filenamePrefix << "["<<i<<"]" << evals[i].format(outformat) << std::endl;
			psievf << "psiev" << filenamePrefix << "["<<i<<"] " << cmat(wft * evecs[i]).format(outformat) << std::endl;
		}

		psievf.close();

	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	MPI_Finalize();
	return 0;
}
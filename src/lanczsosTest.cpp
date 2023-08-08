#include "RtsMat.h"
#include "dkbbasis.h"
#include "rthphbasis.h"
#include "wavefunc.h"
#include "hamiltonian.h"
#include "propagator.h"
//#include <Eigen/unsupported/Eigen/MatrixFunctions>
#define INTENSITY 10

int beyondDipolePulse::l = 1;

int main() {
	using dirbs = rthphbasis<dkbbasis,spnrbasis>;
	using dirwf = wavefunc<dirbs>;
	
	
	int Nsplines = 100;
	int Ntime = 4000;
	int Nkappa = 4;
	int Nmu = 0;
	double rBox = 30.0;
	
	Eigen::IOFormat outformat(Eigen::FullPrecision,Eigen::DontAlignCols,", ","\n","(","),"," = npy.array((\n","\n))\n",' ');

	vec t(Nsplines+1);
	vec y(Nsplines+1);
	for(int i = 0; i < Nsplines+1; i++) {
		t[i] = rBox/(Nsplines)*i;
	}
	
	
	
	dkbbasis dkbb(t,7);
	dkbb.setState(-1);
	lvec x = dkbb.glpts();
	
	
	clsmat& splch = dkbb.splineCache(x);
	clsmat& dsplch = dkbb.dSplineCache(x,1);
	clsmat& ddsplch = dkbb.dSplineCache(x,2);
	
	
	spnrbasis spnrb(Nkappa,Nmu);
	
	cmat bdpams = cmat::Zero(spnrb.angqN(),spnrb.angqN());
	
	beyondDipolePulse bdpp(INTENSITY,50,15);
	//dipolizedBeyondDipolePulse bdpp(10,50,15);
	
	bdpp.setTime(1.75);
	
	
	double dt = (0.6*PI)/Ntime;
	double T = dt*10;
	
	dirbs rthphb(dkbb,spnrb);
	

	vec angInit = vec::Zero(rthphb.angqN());
	angInit[0] = 1.0;
	
	rthphb.pruneUncoupled(angInit);
	
	dkbb.prepbdpvecs(rthphb.angqN(), spnrb.bdplmax());
	dkbb.clvcs();
	
	for(int i = 0; i < rthphb.angids.size(); i++) {
		cout << "(" << i << ", " << rthphb.angids[i] << ", " << ik(rthphb.angids[i])<< ", " << imu(rthphb.angids[i]) << ")," << std::endl;
	}
	//return 0;
	dkbb.dpam(1,1,1);
	dkbb.dpam(-1,1,1);
	for(int l = 0; l < spnrb.bdplmax(); l++) {
		dkbb.bdpam(1,1,1,l,bdpp);
		dkbb.bdpam(1,1,-1,l,bdpp);
	}
	
	using Htype = Dirac<dirbs>;
	Htype H(rthphb,&dplA<15,50,INTENSITY>);
	H.Vfunc = &coloumb<1>;
	H.prepeigs(Nsplines,Nsplines/2);
	
	vector<vec>& evals = H.getevals();
	vector<dsmat>& evecs = H.getevecs();
	
	//cout << "eigs prepped\n";
	
	dirwf psi1 = dirwf(rthphb,evecs[0].col(Nsplines + 5));
	
	
	
	
	psi1.normalize();
	cout << "psi1 " << psi1.coefs.format(outformat)<<"\n";
	//cout << "E1 = " << evals[0][Nsplines+5] << std::endl;;
	
	//return 0;
	
	
	
	//return 0;
	
	RtsMat<Htype> proptest;
	proptest.setDt(dt);
	proptest.setTime(T);
	proptest.attachHamiltonian(H);

	
	cvec b = H.S(psi1.coefs) - dt * cdouble(0,0.5) * H.H(T,psi1.coefs);
	
	SubmatSolver<Eigen::SparseLU<csmat> > solver;
	solver.setup(H,0);
		
	//solver.setMaxIterations(1000);
	
	H.getBasis().getRadial().setState(0);
	
	
	Lancszos<Htype,dirbs> lp(H);
	
	//Cranknich<Htype,dirbs> cnp(H);
	
	lp.dt = dt;
	lp.t = dt*10;
	
	lp.setup();
	
	cvec testvec = cvec::Random(rthphb.angqN()*rthphb.radqN());
	/*
	cout << "Testing commutator [H,S]" << std::endl;
	
	cout << "[H,S^-1]v\n" << H.H(T,solver.solve(testvec))-solver.solve(H.H(T,testvec)) << std::endl;
	
	cout << "{H,S^-1}v\n" << H.H(T,solver.solve(testvec))+solver.solve(H.H(T,testvec)) << std::endl;
	
	cout << "HS^-1v" << H.H(T,solver.solve(testvec)) << std::endl;
	
	cout << "S^-1Hv" << solver.solve(H.H(T,testvec)) << std::endl;
	*/
	//return 0;
	
	cvec psi2 = psi1.coefs;
	
	int N = 40;
	
	cmat v(psi2.size(),N);
	v.col(0) = psi2;
	v.col(0).normalize();
	cvec vi = dt * solver.solve(H.H(1.,v.col(0)));
	
	csmat Tm(N,N);
	
	using cdT = Eigen::Triplet<cdouble>;
	
	std::vector<cdT> triplets(0);
	
	triplets.reserve(3*N);
	
	for(int i = 0; i < N-1; i++) {
		
		cdouble alpha = v.col(i).dot(vi);
		
		vi = vi - alpha * v.col(i);
		
		cdouble beta = vi.norm();
		
		//if(abs(beta) < 10e-15) vi = vi*0;
		vi.normalize();
		
		v.col(i+1) = vi;
		
		cout << "alpha: " << alpha << ", beta: " << beta << std::endl;
		
		triplets.push_back(cdT(i,i,beta));
		triplets.push_back(cdT(i+1,i,alpha));
		triplets.push_back(cdT(i,i+1,alpha));
		
		vi = dt * solver.solve(H.H(1.,v.col(i+1))) - beta * v.col(i);
	}
	
	cdouble alpha = v.col(39).dot(vi);
	
	triplets.push_back(cdT(39,39,alpha));
	
	Tm.setFromTriplets(triplets.begin(),triplets.end());
	
	cout << "v" << v.format(outformat) << std::endl;
	
	cout << "Tm" << cmat(Tm).format(outformat) << std::endl;
	
	
	return 0;
	/*
	lp.krylovspace(psi1.coefs,50);
	
	
	cmat Qm = lp.krylovQ();
	cmat Tm = lp.krylovT();
	
	
	cout << "S" << cmat(H.S()).format(outformat);
	
	cout << Qm.format(outformat);
	
	for(int i = 0; i < 50; i++) {
		cout << "(";
		for(int j = 0; j < 50; j++) {
			cout << Qm.col(i).adjoint() * Qm.col(j) << ", ";
		}
		cout << ")," << std::endl;
	}
	
	Eigen::SelfAdjointEigenSolver<cmat> eigenSolver(Tm);
	
	
	
	cout << eigenSolver.eigenvalues() << std::endl;
	
	cout << eigenSolver.eigenvectors() << std::endl;
	
	cmat expT = (-cdouble(0,1) * eigenSolver.eigenvalues()).array().exp().matrix().asDiagonal();
	cmat eiT = (eigenSolver.eigenvectors().adjoint() * expT * eigenSolver.eigenvectors());
	
	cout << "eiT" << eiT.format(outformat) << std::endl;
	
	cvec Spsi2 = (Qm*eiT).col(0);
	
	cvec psi2 = solver.solve(Spsi2);
	
	psi2 = psi2/(psi2.adjoint() * H.S(psi2));
	
	cout << "psi2 " << psi2.format(outformat);
	
	return 0;*/
	/*
	cnp.propagate(psi1,(0.6*PI)/Ntime,Ntime,10);
	
	dirwf wft = cnp.wf[Ntime - 1];
	
	cout << "psit " << wft.coefs.format(outformat) << "\n";
	
	for(int i = 0; i < evecs.size(); i++) {
		cout << "evl["<<i<<"]" << evals[i].format(outformat) << std::endl;
		cout << "psievBDP["<<i<<"] " << cmat(wft * evecs[i]).format(outformat) << std::endl;
	}
	
	cmat projt = psi1 * cnp.wf;
	cout << "projt " << projt.format(outformat) << "\n";
	
	mat dPdE = H.dPdE(wft,5000,-0.500007,500);
	
	cout << "dPdEdp_E" << INTENSITY << "R" << Nsplines << "K" << Nkappa << "M" << Nmu << "r" << rBox << "T" << Ntime << dPdE.format(outformat) << "\n";
	
	return 0;*/
}
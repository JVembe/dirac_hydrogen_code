#include "RtsMat.h"
#include "cWavefunc.h"
#include "rthphbasis.h"
#include "wavefunc.h"
#include "hamiltonian.h"
#include "propagator.h"

int beyondDipolePulse::l = 1;

int main() {
	using dirbs = rthphbasis<dkbbasis,spnrbasis>;
	using dirwf = wavefunc<dirbs>;
	
	int Nsplines = 400;
	int Ntime = 2000;
	double rBox = 40.0;
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
	
	spnrbasis spnrb(3);
	
	cmat bdpams = cmat::Zero(spnrb.angqN(),spnrb.angqN());
	
	double dt = (0.6*PI)/Ntime;
	double T = 0;
	
	dirbs rthphb(dkbb,spnrb);
	

	vec angInit(rthphb.angqN());
	angInit.setZero();
	angInit[0] = 1.0;
	
	/*for(int i = 0; i < 6; i++) {
		
		rthphb.bdpalphmat(i,bdpp);
	}*/
	
	
	
	using Htype = Dirac<dirbs>;
	
	Htype H(rthphb,&dplA<15,50,10>);
	rthphb.dpalphmat();
	H.Vfunc = &coloumb<1>;
	
	H.prepeigs(Nsplines,Nsplines/2);
	
	vector<vec>& evals = H.getevals();
	vector<dsmat>& evecs = H.getevecs();
	
	dirwf psi1 = dirwf(rthphb,evecs[0].col(Nsplines + 5));
	psi1.normalize();
	
	SubmatSolver<Eigen::SparseLU<csmat> > slvr;
	slvr.setup(H,dt);
	
	
	
	cvec b;
	
	cvec c;
	
	cvec d;
	
	b = H.S(psi1.coefs) - dt * cdouble(0,0.5) * H.H0(psi1.coefs);
	c = slvr.solve(b);
	dirwf psi2(rthphb,c);
	psi2.normalize();
	/*
	cout << "psi1 " << psi1.coefs.format(outformat)<<std::endl;
	cout << "psi2 " << psi2.coefs.format(outformat)<<std::endl;
	cout << "b " << b.format(outformat);
//	cout << psi1 * c;
//	cout << psi1.coefs.transpose() * c;
	*/
//	cout << psi1 * psi2;
	
	//return 0;
	
	
	cout << "psi2 " << psi2.coefs.format(outformat) << std::endl;
	
	cout << psi1 * psi2;
	
	//cvec propDiag = H.S().diagonal() + dt * cdouble(0,0.5) * H.H0().diagonal();
	
	psi2 = dirwf(rthphb,c);
	
	for(int i = 0; i < Ntime; i++) {
		if(i!=0) T += dt;
		b = H.S(psi2.coefs) - dt * cdouble(0,0.5) * H.H(T+dt/2,psi2.coefs);
		//cout << b << std::endl;
		
		c = slvr.solve(b);
		d = c;
		
		//if(i == 30) cout << std::endl << "psit"<<d.format(outformat);
		
		for(int j = 1; j < 100; j++) {
			double dnrmold = d.norm();
			
			cvec e = -dt * cdouble(0,0.5) * H.Ht(T+dt/2,d);// - propDiag.asDiagonal() * d;
			d = slvr.solve(e);
			
			cout <<"||d||/||(H^-1)d|| = " << d.norm()/dnrmold << std::endl;
			
			//if(i == 30) cout << std::endl << "Hpsi"<<j<< d.format(outformat);
			cvec cNew = c + d;
			
			double convCheck = abs(cNew.norm() - c.norm());
			
			cout << convCheck << std::endl;
			
			if(convCheck < 10e-14) {
				c = cNew;
				cout << "matrix iteration count: " << j;
				break;
			}
			c = cNew;
		}
			
		psi2 = dirwf(rthphb,c);
		psi2.normalize();
		
		cout << psi1 * psi2 << std::endl;
		
		//cout << "psi2 " << psi2.coefs.format(outformat) << std::endl;
		
		//cout << std::endl << i << ": " << psi1 * psi2 << std::endl;
	}/*
	return 0;
	
	MatItr<Htype,dirbs> cnp(H);
	cout << "Starting propagator \n";
	
	cnp.propagate(psi1,(0.6*PI)/Ntime,Ntime);
	*/
	dirwf wft = psi2;// = cnp.wf[Ntime - 1];
	
	//dirwf wft = dirwf(rthphb,psi2);
	cout << "psit " << wft.coefs.format(outformat) << "\n";
	
	/*for(int i = 0; i < evals.size(); i++) {
		cout <<"la"<<i<<" "<< evals[i].format(outformat) << "dPdE"<<i<<" " << (wft*estates[i]).format(outformat);
	}*/
	
	//cmat projt = psi1 * cnp.wf;
	//cout << "projt " << projt.format(outformat) << "\n";
	
	mat dPdE = H.dPdE(wft,3000,-0.500007,300);
	
	cout << "dPdE " << dPdE.format(outformat) << "\n";
	
	return 0;
}
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include "cwavefunc.h"
#include "RtsMat.h"
 
int main()
{
	int n = 10;
	csmat S = Eigen::MatrixXcd::Random(n,n).sparseView(0.5,1);
	S = S.transpose()*S;
	
	csmat H = Eigen::MatrixXcd::Random(n,n).sparseView(0.5,1);
	H = H.transpose()*H;
	
	rcp A;
	A.attachMats(S,H);
 
	cvec b(n), x;
	b.setRandom();
	

	{
		Eigen::BiCGSTAB<rcp, CranknichPreconditioner<cdouble> > bicg;
		bicg.preconditioner().setup(S);
		bicg.compute(A);
		x = bicg.solve(b);
		std::cout << "BiCGSTAB: #iterations: " << bicg.iterations() << ", estimated error: " << bicg.error() << std::endl;
	}
	return 0;
}
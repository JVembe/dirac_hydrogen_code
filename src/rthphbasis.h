/*
	This specialization of the basis class takes two other basis objects ("rbasis", "thphbasis") and combines them into a basis for representing a full 3D wavefunction.
	
	It implements matrix-vector products rather than explicitly calculating any matrix representation of the operators, as it's infeasible to store the full matrix representation in memory.
*/

#ifndef RTHPHBASIS_H
#define RTHPHBASIS_H

#include "defs.h"
// #include <omp.h>
#include "mpiFuncs.h"

long double lpot(long double r) {
	if(r == 0.0) return 0;
	else{
		return 1.0/r;
	}
}

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

cvec expmvec(const csmat& mat, const cvec& vec, int n) {
	int N = mat.rows();
	
	cvec out = vec;
	cvec Mv = vec;
	
	
	
	for(int i = 1; i < n; i++) {
		Mv = (mat * Mv) / i;
		out += Mv;
		
		// cout << i << endl << out << endl << endl;
	}
	
	return out;
}

template <typename rbtype, typename thphbtype>
class rthphbasis;

template <typename rbtype, typename thphbtype>
struct basistraits<rthphbasis<rbtype, thphbtype> > {
	typedef basis<rbtype> radial_type;
	typedef basis<thphbtype> angular_type;
};

template <typename rbtype, typename thphbtype>
class rthphbasis: public basis<rthphbasis<rbtype, thphbtype> > {
	friend class basis<rthphbasis<rbtype, thphbtype> >;
	friend class basis<rbtype>;
	friend class basis<thphbtype>;
	
	basis<rbtype>* rbasis;
	basis<thphbtype>* thphbasis;
	
	std::vector<csmat> bdpAsqth;
	
	std::vector<csmat> bdpApzthU;
	std::vector<csmat> bdpApzthL;

	cvec angidsvec;
	//Utility function for mapping indices in matvec function

	int localNl;
	int localNth;
	
	int locall0;
	int localth0;
	
	public:
		void glp(int& a1, int& a2, int& b1, int& b2) {
			a1 = localth0;
			a2 = localNth;
			b1 = locall0;
			b2 = localNl;
		}
		
		inline void indexMap(const int jr,const int jc, int& k1, int& k2, int& row, int& col) {
			
			row = angidsReverse[jr];
			col = angidsReverse[jc];
			
			
			if(row!=-1) k1 = jr;
			if(col!=-1) k2 = jc;
			
			
			// for(int m = 0; m < angids.size(); m++) {
				// if(jr == angids[m]) 
				// {
					// k1 = angids[m];
					// row = m;
				// }
				// if(jc == angids[m]) 
				// {
					// k2 = angids[m];
					// col = m;
				// }
			// }
			// cout << jr << ", " << jc << ", " << k1 << ", " << k2 << ", " << row << ", " << col << "\n";
		}
	
		std::vector<int> angids;
		std::vector<int> angidsReverse;
		
		int idxtfrm(int i) {
			return angids[i];
		}
		
		int iqn(int kappa, double mu) {
			
			int idl = ki(kappa);
			
			cout << idl << endl;
			
			if(abs(mu)<abs(kappa)) {
				idl += abs(kappa) + mu - 0.5;
				cout << idl << endl;
			}
			else return -1;
			
			return angidsReverse[idl];
		}
		
		rthphbasis(basis<rbtype>& rbasis, basis<thphbtype>& thphbasis):basis<rthphbasis<rbtype,thphbtype> >() {
			this->rbasis = &rbasis;
			this->thphbasis = &thphbasis;
			angids = std::vector<int>(this->angqN());
			angidsReverse = std::vector<int>(this->angqN());
			for(int i = 0; i < this->angqN(); i++) {
				angids[i] = i;
				angidsReverse[i] = i;
			}
			
			localth0 = 0;
			localNth = this->angqN();
		}
		
		csmat& olmat() {
			csmat& Sm = this->template getmat<S>();
		
			if (isCached(Sm)) return Sm;
			else {	
				csmat& S0 = rbasis->Sm();
				csmat& S1 = thphbasis->Sm();
				
				std::vector<T> triplets;
				
				triplets.reserve(S0.nonZeros() * S1.nonZeros());
				
				Sm = csmat(S0.rows() * this->angqN(),S0.cols() * this->angqN());	
				
				for(int i = 0; i < this->angqN(); i++) {
					rbasis->setState(angids[i]);
					
					S0 = rbasis->Sm();
					for(int j = 0; j < S0.outerSize(); j++) {
						for(csmat::InnerIterator k(S0,j); k; ++k) {
							triplets.push_back(T(
								i*S0.rows() + k.row(),
								i*S0.rows() + k.col(),
								k.value()
								)
								);
						}
					}
				}
				
				Sm.setFromTriplets(triplets.begin(),triplets.end());
				
				return Sm;
			}
		}

		csmat& psqmat() {
			csmat& psqm = this->template getmat<psq>();
			if (isCached(psqm)) return psqm;
			// cout << "psqmat called\n";
			csmat& dd = rbasis->ddm(2);
			csmat& lsq = thphbasis->Lsqm();
			csmat& Sth = thphbasis->Sm();
			
			// cout << "lsq-matrix: " << lsq << "\n";
			
			psqm = csmat(dd.rows() * lsq.rows(), dd.cols() * lsq.cols());
			
			std::vector<T> triplets;
			
			triplets.reserve(dd.nonZeros() * lsq.nonZeros());
			
			for(int i = 0; i < Sth.outerSize(); i++) {
				for(csmat::InnerIterator j(Sth,i); j; ++j) {
					
					int ll = lsq.coeffRef(j.row(),j.col()).real();
					
					csmat& lm = rbasis->Lsqm();
					for(int k = 0; k < dd.outerSize(); k++) {
						for(csmat::InnerIterator l(dd,k); l; ++l) {
							triplets.push_back(T(
								j.row() * dd.rows() + l.row(),
								j.col() * dd.cols() + l.col(),
								-0.5 * l.value() + cdouble(ll,0) * lm.coeffRef(l.row(),l.col())
								)
							);
						}
					}
				}
			}
			
			psqm.setFromTriplets(triplets.begin(),triplets.end());
			
			return psqm;
		}
		
		csmat& Emat(long double (*V)(long double)) {
			csmat& Em = this->template getmat<E>();
			// cout << "Emat called\n";
			csmat E0 = rbasis->Em(V);
			csmat S1 = thphbasis->Sm();
			
			
			std::vector<T> triplets;
			
			triplets.reserve(E0.nonZeros() * S1.nonZeros());
			
			Em = csmat(E0.rows() * this->angqN(),E0.cols() * this->angqN());	
			
			for(int i = 0; i < this->angqN(); i++) {

				rbasis->setState(angids[i]);
				
				E0 = rbasis->Em(V);
				for(int j = 0; j < E0.outerSize(); j++) {
					for(csmat::InnerIterator k(E0,j); k; ++k) {
						triplets.push_back(T(
							i*E0.rows() + k.row(),
							i*E0.rows() + k.col(),
							k.value()
							)
							);
					}
				}
			}
			
			Em.setFromTriplets(triplets.begin(),triplets.end());
			
			return Em;
		}
		
		csmat& pzmat(){
			csmat& pzm = this->template getmat<pz>();
			if(isCached(pzm)) return pzm;
			
			else {
				// cout << "First pzmat call\n";
				csmat& ddr = rbasis->ddm(1);
				csmat& pzang = thphbasis->pzm();
				csmat& Sth = thphbasis->Sm();
				
				// cout << "Declating pzm\n";
				pzm = csmat(ddr.rows() * pzang.rows(), ddr.cols() * pzang.cols());
				
				std::vector<Eigen::Triplet<cdouble> > triplets;
			
				triplets.reserve(ddr.nonZeros() * pzang.nonZeros());
				csmat& lm = rbasis->Lm();
				
				// cout << "Combining matrices in triplet list\n";
				for(int i = 0; i < pzang.outerSize(); i++) {
					for(csmat::InnerIterator j(pzang,i); j; ++j) {
						int prevl = (min(j.row(),j.col()) + 1) * sgn(j.col() - j.row());
						// cout << "("<<j.row()<<", " <<j.col() <<"): " << prevl << "\n";
						
						for(int k = 0; k < ddr.outerSize(); k++) {
							for(csmat::InnerIterator l(ddr,k); l; ++l) {
								// cout << j.row() <<" "<< ddr.rows() <<" "<< l.row() <<"\n"<<
									// j.col() <<" "<< ddr.cols() <<" "<< l.col()<<"\n"<<
									// j.value() <<" "<< l.value() <<" "<< lm.coeffRef(l.row(),l.col());
								triplets.push_back(Eigen::Triplet<cdouble>(
									j.row() * ddr.rows() + l.row(),
									j.col() * ddr.cols() + l.col(),
									j.value() * (l.value() + cdouble(prevl,0) * lm.coeffRef(l.row(),l.col()))
									)
								);
							}
						}
					}
				}
				
				pzm.setFromTriplets(triplets.begin(),triplets.end());
				
				return pzm;
			}
		}
		csmat& p1mat() {
			csmat& p1m = this->template getmat<p1>();
			// cout << "p1mat called\n";
			csmat& dd = rbasis->ddm(1);
			//csmat *alph = this->thphbasis->dpam();
			
			// cout << "Setting p1m \n";
			p1m = csmat(dd.rows() * this->angqN(), dd.cols() * this->angqN());
			// cout << "Declaring triplet vector\n";
			std::vector<T> triplets;
			//cout << "Reserving " << dd.nonZeros() * alph->nonZeros() << "triplets for vector \n";
			triplets.reserve(dd.nonZeros() * this->angqN());
			
			// cout << "Initiating loop \n";
			
			for(int i = 0; i < this->angqN(); i++) {
					// cout << i << "\n";
					rbasis->setState(angids[i]);
					
					dd = rbasis->ddm(1);
					csmat km = rbasis->km();
					
					for(int k = 0; k < dd.outerSize(); k++) {
						for(csmat::InnerIterator l(dd,k); l; ++l) {
							triplets.push_back(T(
								i * dd.rows() + l.row(),
								i * dd.cols() + l.col(),
								l.value() + km.coeffRef(l.row(),l.col())
								)
							);
						}
					}
				}
			
			p1m.setFromTriplets(triplets.begin(),triplets.end());
			
			return p1m;
		}
		
		csmat& dpalphmat() {
			csmat& dpalphm = this->template getmat<dpa>();
			// cout << "Cache check rthphb.dpalphmat: " << isCached(dpalphm) << "\n";
			if(isCached(dpalphm)) return dpalphm;
			else {
				// cout << "Dpalphmat not cached, constructing...\n";
				
				csmat ralphU = this->rbasis->dpam(0,0,1);
				csmat ralphL = this->rbasis->dpam(0,0,-1);
				csmat ralph = ralphU + ralphL;
				
				csmat thalphU = this->thphbasis->dpam(1);
				csmat thalphL = this->thphbasis->dpam(-1);
				
				csmat thalph = thalphU + thalphL;
				
				// cout << "ralph rows: " << ralph.rows() << ", cols: " << ralph.cols() << "\n";
				// cout << "thalph rows: " << thalph.rows() << ", cols: " << thalph.cols() << "\n";		
				std::vector<T> triplets;
			
				triplets.reserve(ralphU.nonZeros() * thalphU.nonZeros());
			
				dpalphm = csmat(ralphU.rows() * this->angqN(),ralphU.cols() * this->angqN());	
				
				// cout << thalph;
				
				for(int i = 0; i < thalph.outerSize(); i++) {
					for(csmat::InnerIterator j(thalph,i); j; ++j) {
						int k1 = -1;
						int k2 = -1;
						
						int row = -1;
						int col = -1;
						
						indexMap(j.row(),j.col(),k1,k2,row,col);
						
						// for(int m = 0; m < angids.size(); m++) {
							// if(j.row() == angids[m]) 
							// {
								// k1 = angids[m];
								// row = m;
							// }
							// if(j.col() == angids[m]) 
							// {
								// k2 = angids[m];
								// col = m;
							// }
						// }
						
						
						
						// cout << k1 << "," << k2 << std::endl;
						
						if(k1 !=-1 && k2 != -1) {
							// cout << "("<<row<<", " <<col << "): (" <<k1 << ", " <<k2 <<")\n";
							
							ralphU = rbasis->dpam(k1,k2,1);
							ralphL = rbasis->dpam(k1,k2,-1);
							
							
							
							// cout << "Finished\n";
							for(int k = 0; k < ralph.outerSize(); k++) {
								for(csmat::InnerIterator l(ralph,k); l; ++l) {
									// cout << j.row() << " " << l.row() << "\n" <<
										// j.col() << " " << l.col() << "\n" <<
										// j.value() << " " << l.value();// << " " << lm->coeffRef(l.row(),l.col());
									triplets.push_back(Eigen::Triplet<cdouble>(
										row * ralphU.rows() + l.row(),
										col * ralphU.cols() + l.col(),
										ralphU.coeffRef(l.row(),l.col()) * thalphU.coeffRef(j.row(),j.col()) + ralphL.coeffRef(l.row(),l.col()) * thalphL.coeffRef(j.row(),j.col())
										)
									);
								}
							}
						}
					}
				}
				
				dpalphm.setFromTriplets(triplets.begin(),triplets.end());
				
				return dpalphm;
			}
		}
		
		csmat& bdpalphmat(int alpha, const Potential<beyondDipolePulse>& vExt) {
			csmat& bdpalphm = this->template getmat<bdpa>();
			// cout << "Cache check rthphb.bdpalphmat: " << isCached(bdpalphm) << "\n";
			// if(isCached(bdpalphm)) return bdpalphm;
			// else {
				csmat ralphU = this->rbasis->bdpam(0,0,1,0,alpha,vExt);
				csmat ralphL = this->rbasis->bdpam(0,0,-1,0,alpha,vExt);
				//csmat ralph = this->rbasis->dpam(0,0,0);
				
				
				csmat thalph;
				csmat thalphU = this->thphbasis->bdpam(1,0);
				csmat thalphL = this->thphbasis->bdpam(-1,0);
				
				
				// cout << "ralph rows: " << ralph.rows() << ", cols: " << ralph.cols() << "\n";
				// cout << "thalph rows: " << thalph.rows() << ", cols: " << thalph.cols() << "\n";		
				
				bdpalphm = csmat(ralphU.rows() * this->angqN(),ralphU.cols() * this->angqN());	
				
				
				for(int l = 0; l < thphbasis->bdplmax(); l++) {
					std::vector<T> triplets;
				
					triplets.reserve(ralphU.nonZeros() * thalphU.nonZeros());
					
					if(l%2!=alpha%2) {
						csmat dpalphm(ralphU.rows() * this->angqN(),ralphU.cols() * this->angqN());
						thalphU = this->thphbasis->bdpam(1,l);
						thalphL = this->thphbasis->bdpam(-1,l);
						thalph = thalphU + thalphL;
						
						for(int i = 0; i < thalph.outerSize(); i++) {
							for(csmat::InnerIterator j(thalph,i); j; ++j) {
								int k1 = -1;
								int k2 = -1;
								
								int row = -1;
								int col = -1;
								
								indexMap(j.row(),j.col(),k1,k2,row,col);
						
								// for(int m = 0; m < angids.size(); m++) {
									// if(j.row() == angids[m]) 
									// {
										// k1 = angids[m];
										// row = m;
									// }
									// if(j.col() == angids[m]) 
									// {
										// k2 = angids[m];
										// col = m;
									// }
								// }
								if(k1 !=-1 && k2 != -1) {
									// cout << "("<<row<<", " <<col << "): (" <<k1 << ", " <<k2 <<")\n";
									
									ralphU = rbasis->bdpam(k1,k2,1,l,alpha,vExt);
									ralphL = rbasis->bdpam(k1,k2,-1,l,alpha,vExt);
									//ralphU = rbasis->dpam(k1,k2,1);
									//ralphL = rbasis->dpam(k1,k2,-1);

									
									// cout << "Finished\n";
									for(int k = 0; k < ralphU.outerSize(); k++) {
										for(csmat::InnerIterator m(ralphU,k); m; ++m) {
											// cout << j.row() <<" "<< ddr.rows() <<" "<< l.row() <<"\n"<<
											//	j.col() <<" "<< ddr.cols() <<" "<< l.col()<<"\n"<<
											//	j.value() <<" "<< l.value() <<" "<< lm->coeffRef(l.row(),l.col());
											
											// cdouble thU = thalphU.coeffRef(j.row(),j.col());
											// cdouble thL = thalphL.coeffRef(j.row(),j.col());
											
											
											
											// triplets.push_back(Eigen::Triplet<cdouble>(
												// row * ralphU.rows() + m.row(),
												// col * ralphU.cols() + m.col(),
												// ralphU.coeffRef(l.row(),l.col()) * cdouble(thU!=cdouble(0,0),0) + ralphL.coeffRef(l.row(),l.col()) * cdouble(thL!=cdouble(0,0),0)
												// ralphU.coeffRef(m.row(),m.col()) * thU + ralphL.coeffRef(m.row(),m.col()) * thL
												//(cdouble(k1,k2) * thU + cdouble(k1,k2) * thL) * cdouble(row * ralphU.rows() + m.row() == col * ralphU.cols() + m.col(),0)
												//)
											//);
										}
									}
								}
							}
						}
						dpalphm.setFromTriplets(triplets.begin(),triplets.end());
						
						bdpalphm += dpalphm;
					}
				}
				
				return bdpalphm;
			//}
		}
		/*
		csmat& bdpalphmat(const Potential<beyondDipolePulse>& vExt) {
			csmat& bdpalphm = this->template getmat<bdpa>();
			cout << "Cache check rthphb.dpalphmat: " << isCached(dpalphm) << "\n";
			// if(isCached(bdpalphm)) return bdpalphm;
			// else {
				csmat ralphU = this->rbasis->bdpam(0,0,1,0,vExt);
				csmat ralphL = this->rbasis->bdpam(0,0,-1,0,vExt);
				//csmat ralph = this->rbasis->dpam(0,0,0);
				
				csmat thalph;
				csmat thalphU = this->thphbasis->bdpam(1,0);
				csmat thalphL = this->thphbasis->bdpam(-1,0);
				
				
				cout << "ralph rows: " << ralph.rows() << ", cols: " << ralph.cols() << "\n";
				cout << "thalph rows: " << thalph.rows() << ", cols: " << thalph.cols() << "\n";		
				std::vector<T> triplets;
			
				triplets.reserve(ralphU.nonZeros() * thalphU.nonZeros());
				
				bdpalphm = csmat(ralphU.rows() * this->angqN(),ralphU.cols() * this->angqN());	
				
				for(int l = 0; l < thphbasis->bdplmax(); l++) {
					csmat dpalphm(ralphU.rows() * this->angqN(),ralphU.cols() * this->angqN());
					thalphU = this->thphbasis->bdpam(1,l);
					thalphL = this->thphbasis->bdpam(-1,l);
					thalph = thalphU + thalphL;
					
					for(int i = 0; i < thalph.outerSize(); i++) {
						for(csmat::InnerIterator j(thalph,i); j; ++j) {
							int k1 = -1;
							int k2 = -1;
							
							int row = -1;
							int col = -1;
							for(int m = 0; m < angids.size(); m++) {
								if(j.row() == angids[m]) 
								{
									k1 = angids[m];
									row = m;
								}
								if(j.col() == angids[m]) 
								{
									k2 = angids[m];
									col = m;
								}
							}
							if(k1 !=-1 && k2 != -1) {
								cout << "("<<row<<", " <<col << "): (" <<k1 << ", " <<k2 <<")\n";
								
								ralphU = rbasis->bdpam(k1,k2,1,l,vExt);
								ralphL = rbasis->bdpam(k1,k2,-1,l,vExt);
								
								//ralphU = rbasis->dpam(k1,k2,1);
								//ralphL = rbasis->dpam(k1,k2,-1);
								
								
								
								cout << "Finished\n";
								for(int k = 0; k < ralphU.outerSize(); k++) {
									for(csmat::InnerIterator l(ralphU,k); l; ++l) {
										cout << j.row() <<" "<< ddr.rows() <<" "<< l.row() <<"\n"<<
										//	j.col() <<" "<< ddr.cols() <<" "<< l.col()<<"\n"<<
										//	j.value() <<" "<< l.value() <<" "<< lm->coeffRef(l.row(),l.col());
										triplets.push_back(Eigen::Triplet<cdouble>(
											row * ralphU.rows() + l.row(),
											col * ralphU.cols() + l.col(),
											ralphU.coeffRef(l.row(),l.col()) * thalphU.coeffRef(j.row(),j.col()) + ralphL.coeffRef(l.row(),l.col()) * thalphL.coeffRef(j.row(),j.col())
											)
										);
									}
								}
							}
						}
					}
					dpalphm.setFromTriplets(triplets.begin(),triplets.end());
					
					bdpalphm += dpalphm;
				}
				
				return bdpalphm;
			//}
		}
		*/
		csmat& ulcmat() {
			csmat& ulcm = this->template getmat<ulc>();
			
			csmat ulc0 = rbasis->ulcm();
			csmat S1 = thphbasis->Sm();
			
			
			std::vector<T> triplets;
			
			triplets.reserve(ulc0.nonZeros() * S1.nonZeros());
			
			ulcm = csmat(ulc0.rows() * this->angqN(),ulc0.cols() * this->angqN());	
			
			for(int i = 0; i < this->angqN(); i++) {
				rbasis->setState(angids[i]);
				
				ulc0 = rbasis->ulcm();
				for(int j = 0; j < ulc0.outerSize(); j++) {
					for(csmat::InnerIterator k(ulc0,j); k; ++k) {
						triplets.push_back(T(
							i*ulc0.rows() + k.row(),
							i*ulc0.rows() + k.col(),
							k.value()
							)
							);
					}
				}
			}
			
			ulcm.setFromTriplets(triplets.begin(),triplets.end());
			
			return ulcm;
		}
		
		template <matname M>
		cvec matvec(const cvec& v) {
			return matvec_impl(std::integral_constant<matname, M> {}, v);
		}
		
		template <matname M>
		cvec matvec(const cvec& v, int param) {
			return matvec_impl(std::integral_constant<matname, M> {}, v, param);
		}
		
		cvec matvec_impl(std::integral_constant<matname,matname::compS> c, const cvec& v, int comp) {
			
			csmat olcomp = this->thphbasis->olcm(comp);
			
			cvec out = cvec::Zero(v.size());
			
			int Nr = this->radqN();
			int Nth = this->angqN();
			
			std::vector<cvec> outparts;
			for(int i = 0; i < Nth; i++) {
				outparts.push_back(cvec::Zero(Nr));
			}
			
			
			for(int i = 0; i < olcomp.outerSize(); i++) {
				for(csmat::InnerIterator j(olcomp,i); j; ++j) {
					int k1 = -1;
					int k2 = -1;
					int row = -1;
					int col = -1;
					
					indexMap(j.row(),j.col(),k1,k2,row,col);
					
					
					if(k1 !=-1 && k2 != -1) {
						this->rbasis->setState(k1,k2,1);
						// cout << "k1: " << k1 <<", k2: " << k2 << ", j: " << j.value() << std::endl;
						if(comp < 2)
							this->rbasis->template cacheMatfree<SU>(v.segment(Nr*col,Nr),outparts[row],col,j.value(),k1,k2,1);
						else
							this->rbasis->template cacheMatfree<SL>(v.segment(Nr*col,Nr),outparts[row],col,j.value(),k1,k2,1);
					}
				}
			}
			
			for(int i = 0; i < Nth; i++) {
				out.segment(Nr*i,Nr) += outparts[i];
			}
			
			return out;
		}
		
		cvec matvec_impl(std::integral_constant<matname,matname::dpa> c, const cvec& v) {
			return dpamatvecMPIblock(v);
			// return dpamatvecBlock(v);
			
			
			// cout << "Matvec<dpa> called successfully\n";
			csmat dpthU = this->thphbasis->dpam(1);
			csmat dpthL = this->thphbasis->dpam(-1);
			
			
			cvec out = cvec::Zero(v.size());
			
			
			int Nr = this->radqN();
			int Nth = this->angqN();
			
			
			// cout << "v size: " << v.size() << ", Nr: " << Nr << ", Nth: " << Nth <<"\n";
			
			for(int i = 0; i < dpthU.outerSize(); i++) {
				for(csmat::InnerIterator j(dpthU,i); j; ++j) {
					int k1 = -1;
					int k2 = -1;
					
					int row = -1;
					int col = -1;
					
					indexMap(j.row(),j.col(),k1,k2,row,col);
					
					// for(int m = 0; m < angids.size(); m++) {
						// if(j.row() == angids[m]) 
						// {
							// k1 = angids[m];
							// row = m;
						// }
						// if(j.col() == angids[m]) 
						// {
							// k2 = angids[m];
							// col = m;
						// }
					// }
					if(k1 !=-1 && k2 != -1) {
								
						this->rbasis->setState(k1,k2,1);
						
						// cout << "Setting block at " << Nr*row << " of size " << Nr << " From block at " << Nr*col << ", equivalent to matrix block at " << row << ", " << col << "\n";
						out.segment(Nr*row,Nr).noalias() += (this->rbasis->template matfree<dpa>(j.value() * v.segment(Nr*col,Nr)));
					}
				}
			}
			for(int i = 0; i < dpthL.outerSize(); i++) {
				for(csmat::InnerIterator j(dpthL,i); j; ++j) {
					int k1 = -1;
					int k2 = -1;
					
					int row = -1;
					int col = -1;
					
					indexMap(j.row(),j.col(),k1,k2,row,col);
					// for(int m = 0; m < angids.size(); m++) {
						// if(j.row() == angids[m]) 
						// {
							// k1 = angids[m];
							// row = m;
						// }
						// if(j.col() == angids[m]) 
						// {
							// k2 = angids[m];
							// col = m;
						// }
					// }
					if(k1 !=-1 && k2 != -1) {
								
						this->rbasis->setState(k1,k2,-1);
						
						// cout << "Setting block at " << Nr*row << " of size " << Nr << " From block at " << Nr*col << ", equivalent to matrix block at " << row << ", " << col << "\n";
						
						out.segment(Nr*row,Nr).noalias() += (this->rbasis->template matfree<dpa>(j.value() * v.segment(Nr*col,Nr)));
						
					}
				}
			}
			
			return out;
		}
		
		cvec parallel_dpa(const cvec& v) {
			// cout << "Matvec<dpa> called successfully\n";
			csmat dpthU = this->thphbasis->dpam(1);
			csmat dpthL = this->thphbasis->dpam(-1);
			
			
			cvec out = cvec::Zero(v.size());
			
			
			int Nr = this->radqN();
			int Nth = this->angqN();
			
			this->clearVecs();
			
			std::vector<cvec> outparts;
			for(int i = 0; i < Nth; i++) {
				outparts.push_back(cvec::Zero(Nr));
			}
			
			// cout << "Operating cache size: " << Nth << std::endl;
			
			this->rbasis->template precacheMatfree<dpa>(v,Nth,1);
			// cout << "v size: " << v.size() << ", Nr: " << Nr << ", Nth: " << Nth <<"\n";
			#ifdef USE_MY_OPENMP
			//#pragma omp parallel
			#endif
			{

				#ifdef USE_MY_OPENMP
				//#pragma omp for
				#endif
				for(int i = 0; i < dpthU.cols(); i++) {
					for(csmat::InnerIterator j(dpthU,i); j; ++j) {
						int k1 = -1;
						int k2 = -1;
						
						int row = -1;
						int col = -1;

						indexMap(j.row(),j.col(),k1,k2,row,col);
						
						// for(int m = 0; m < angids.size(); m++) {
							// if(j.row() == angids[m]) 
							// {
								// k1 = angids[m];
								// row = m;
							// }
							// if(j.col() == angids[m]) 
							// {
								// k2 = angids[m];
								// col = m;
							// }
						// }
						if(k1 !=-1 && k2 != -1) {
									
							//this->rbasis->setState(k1,k2,1);
							
							// cout << "Setting block at " << row << " of size " << Nr << " From block at " << col << ", equivalent to matrix block at " << j.row() << ", " << j.col() << "\n"
							// << "k1: " << k1 << ", k2: " << k2 << ", kappa1: " << ik(k1) << ", kappa2: " << ik(k2) << std::endl;
							// << "Outparts size: " << outparts[row].size() << ", v segment size: " << v.segment(Nr*col,Nr).size() << std::endl;
							
							this->rbasis->template cacheMatfree<dpa>(v.segment(Nr*col,Nr),outparts[row],col,j.value(),k1,k2,1);
							// this->rbasis->template cacheMatfree<dpa>(v.segment(Nr*col,Nr),outparts[row],col,1.0,k1,k2,1);
							// v.segment(Nr*col,Nr) = cvec::Constant(Nr,cdouble(k1,k2));
						}
					}
					for(csmat::InnerIterator j(dpthL,i); j; ++j) {
						int k1 = -1;
						int k2 = -1;
						
						int row = -1;
						int col = -1;
						
						indexMap(j.row(),j.col(),k1,k2,row,col);
							
						// for(int m = 0; m < angids.size(); m++) {
							// if(j.row() == angids[m]) 
							// {
								// k1 = angids[m];
								// row = m;
							// }
							// if(j.col() == angids[m]) 
							// {
								// k2 = angids[m];
								// col = m;
							// }
						// }
						
						if(k1 !=-1 && k2 != -1) {
									
							//this->rbasis->setState(k1,k2,-1);
							
							// cout << "Setting block at " << row << " of size " << Nr << " From block at " << col << ", equivalent to matrix block at " << j.row() << ", " << j.col() << "\n"
							// << "k1: " << k1 << ", k2: " << k2 << ", kappa1: " << ik(k1) << ", kappa2: " << ik(k2) << std::endl;
							
							this->rbasis->template cacheMatfree<dpa>(v.segment(Nr*col,Nr),outparts[row],col,j.value(),k1,k2,-1);
							// this->rbasis->template cacheMatfree<dpa>(v.segment(Nr*col,Nr),outparts[row],col,0.0,k1,k2,-1);
							
						}
					}
				}
				#ifdef USE_MY_OPENMP
				//#pragma omp for
				#endif
				for(int i = 0; i < Nth; i++) {
					out.segment(Nr*i,Nr) += outparts[i];
				}
			}
			return out;
		}
		
		cvec matvec_impl(std::integral_constant<matname,matname::pz> c, const cvec& v) {
			
			csmat& pzang = thphbasis->pzm();
			
			cvec out = cvec::Zero(v.size());
			
			
			int Nr = this->radqN();
			int Nth = this->angqN();
			
			// cout << "Call to pzm\n";
			//pzm = csmat(ddr.rows() * pzang.rows(), ddr.cols() * pzang.cols());
			
			// cout << "Combining matrices in triplet list\n";
			for(int i = 0; i < pzang.outerSize(); i++) {
				for(csmat::InnerIterator j(pzang,i); j; ++j) {
					int k1 = -1;
					int k2 = -1;
					
					int row = -1;
					int col = -1;
					
					indexMap(j.row(),j.col(),k1,k2,row,col);
					// for(int m = 0; m < angids.size(); m++) {
						// if(j.row() == angids[m]) 
						// {
							// k1 = angids[m];
							// row = m;
						// }
						// if(j.col() == angids[m]) 
						// {
							// k2 = angids[m];
							// col = m;
						// }
					// }
					
					if(k1 !=-1 && k2 != -1) {
						int prevl;
						if(k1<k2) prevl = il(k2);
						else prevl = -il(k2) - 1;
						
						// cout << "("<<j.row()<<", " <<j.col() <<"): " << prevl << "\n";
						
						//prevl = 0;
						
						cdouble pza = j.value();
						
						
						out.segment(Nr*row,Nr).noalias() += pza * (rbasis->template matfree<pz>(v.segment(Nr*col,Nr)) + cdouble(prevl,0) * rbasis->template matfree<L>(v.segment(Nr*col,Nr)));
					}
					//int prevl = (min(j.row(),j.col()) + 1) * sgn(j.col() - j.row());
				}
			}
			
			return out;
			
		}
		
		cvec matvec_impl(std::integral_constant<matname,matname::psq> c, const cvec& v) {
			/*
				csmat& psqm = this->template getmat<psq>();
				if (isCached(psqm)) return psqm;
				cout << "psqmat called\n";
				csmat& dd = rbasis->ddm(2);
				csmat& lsq = thphbasis->Lsqm();
				csmat& Sth = thphbasis->Sm();
				
				cout << "lsq-matrix: " << lsq << "\n";
				
				psqm = csmat(dd.rows() * lsq.rows(), dd.cols() * lsq.cols());
				
				std::vector<T> triplets;
				
				triplets.reserve(dd.nonZeros() * lsq.nonZeros());
				
				for(int i = 0; i < Sth.outerSize(); i++) {
					for(csmat::InnerIterator j(Sth,i); j; ++j) {
						int ll = lsq.coeffRef(j.row(),j.col()).real();
						csmat& lm = rbasis->Lsqm();
						for(int k = 0; k < dd.outerSize(); k++) {
							for(csmat::InnerIterator l(dd,k); l; ++l) {
								triplets.push_back(T(
									j.row() * dd.rows() + l.row(),
									j.col() * dd.cols() + l.col(),
									-0.5 * l.value() + cdouble(ll,0) * lm.coeffRef(l.row(),l.col())
									)
								);
							}
						}
					}
				}
				
				psqm.setFromTriplets(triplets.begin(),triplets.end());
				
				return psqm;
			}
			*/
			csmat& Lsqang = thphbasis->Lsqm();
			
			cvec out = cvec::Zero(v.size());
			
			
			int Nr = this->radqN();
			int Nth = this->angqN();
			
			// cout << "Declating pzm\n";
			//pzm = csmat(ddr.rows() * pzang.rows(), ddr.cols() * pzang.cols());
			
			// cout << "Combining matrices in triplet list\n";
			for(int i = 0; i < Lsqang.outerSize(); i++) {
				for(csmat::InnerIterator j(Lsqang,i); j; ++j) {
					int k1 = -1;
					int k2 = -1;
					
					int row = -1;
					int col = -1;

					indexMap(j.row(),j.col(),k1,k2,row,col);
					
					// for(int m = 0; m < angids.size(); m++) {
						// if(j.row() == angids[m]) 
						// {
							// k1 = angids[m];
							// row = m;
						// }
						// if(j.col() == angids[m]) 
						// {
							// k2 = angids[m];
							// col = m;
						// }
					// }
					
					if(k1 !=-1 && k2 != -1) {
						// cout << "k1: " << k1 << ", k2: " << k2 << ", row: " << row << ", col: " << col << std::endl;
							
						cdouble lsq = j.value();
						
						// cout << "matfree v size: " << v.segment(Nr*col,Nr).size();
						// cout << "matfree dd size: " << rbasis->template matfree<dd>(v.segment(Nr*col,Nr)).size();
						// cout << "matfree lsq size: " << rbasis->template matfree<Lsq>(v.segment(Nr*col,Nr)).size();
						
						out.segment(Nr*row,Nr).noalias() += -0.5 * rbasis->template matfree<dd>(v.segment(Nr*col,Nr)) + lsq * rbasis->template matfree<Lsq>(v.segment(Nr*col,Nr));
					}
				}
			}
			
			return out;
			
		}
		
		cvec matvec_impl(std::integral_constant<matname,matname::bdpa> c, const cvec& v,int alpha) {
			// cout << "Matvec<dpa> called successfully\n";
			csmat& bdpthU = this->thphbasis->bdpam(1,0);
			csmat& bdpthL = this->thphbasis->bdpam(-1,0);
			
			
			cvec out = cvec::Zero(v.size());
			
			int Nr = this->radqN();
			int Nth = this->angqN();
			
			std::vector<cvec> outparts;
			for(int i = 0; i < Nth; i++) {
				outparts.push_back(cvec::Zero(Nr));
			}
			// cout << "v size: " << v.size() << ", Nr: " << Nr << ", Nth: " << Nth <<"\n";
			
			//#pragma omp parallel for
			for(int l = 0; l < thphbasis->bdplmax(); l++) {
				bdpthU = this->thphbasis->bdpam(1,l);
				bdpthL = this->thphbasis->bdpam(-1,l);
				
				this->clearVecs();
				
				//#pragma omp parallel for
				for(int i = 0; i < bdpthU.outerSize(); i++) {
					//csmat::InnerIterator j = csmat::InnerIterator(bdpthU,i);
					
					//while (!j) {
					for(csmat::InnerIterator j(bdpthU,i); j; ++j) {
						int k1 = -1;
						int k2 = -1;
						
						int row = -1;
						int col = -1;
						
						indexMap(j.row(),j.col(),k1,k2,row,col);
						
						// for(int m = 0; m < angids.size(); m++) {
							// if(j.row() == angids[m]) 
							// {
								// k1 = angids[m];
								// row = m;
							// }
							// if(j.col() == angids[m]) 
							// {
								// k2 = angids[m];
								// col = m;
							// }
						// }
						if(k1 !=-1 && k2 != -1) {
									
							this->rbasis->setState(k1,k2,1,l,alpha);
							//dpalphm = dpalph0 + kappa1 * dpalphk0 + kappa2 * dpalph0k + kappa1 * kappa2 * dpalphkk;
							
							// cout << "Setting block at " << Nr*row << " of size " << Nr << " From block at " << Nr*col << ", equivalent to matrix block at " << row << ", " << col << "\n";
							
							outparts[row].noalias() += j.value() * (this->rbasis->template cacheMatfree<bdpa>(v.segment(Nr*col,Nr),col));
							//out.segment(Nr*row,Nr).noalias() += j.value() * (this->rbasis->template cacheMatfree<bdpa>(v.segment(Nr*col,Nr),col));
						
						}
						
						//++j;
					}
				}
				
				//#pragma omp parallel for
				for(int i = 0; i < bdpthL.outerSize(); i++) {
					//csmat::InnerIterator j = csmat::InnerIterator(bdpthL,i);
					for(csmat::InnerIterator j(bdpthL,i); j; ++j) {
					//while(!j) {
						int k1 = -1;
						int k2 = -1;
						
						int row = -1;
						int col = -1;
						
						indexMap(j.row(),j.col(),k1,k2,row,col);
						
						// for(int m = 0; m < angids.size(); m++) {
							// if(j.row() == angids[m]) 
							// {
								// k1 = angids[m];
								// row = m;
							// }
							// if(j.col() == angids[m]) 
							// {
								// k2 = angids[m];
								// col = m;
							// }
						// }
						if(k1 !=-1 && k2 != -1) {
									
							this->rbasis->setState(k1,k2,-1,l,alpha);
							
							// cout << "Setting block at " << Nr*row << " of size " << Nr << " From block at " << Nr*col << ", equivalent to matrix block at " << row << ", " << col << "\n";
							
							outparts[row].noalias()  += j.value() * (this->rbasis->template cacheMatfree<bdpa>(v.segment(Nr*col,Nr),col));
							//out.segment(Nr*row,Nr).noalias() += j.value() * (this->rbasis->template cacheMatfree<bdpa>(v.segment(Nr*col,Nr),col));
							
						}
						//++j;
					}
				}
			}
			
			for(int i = 0; i < Nth; i++) {
				out.segment(Nr*i,Nr) = outparts[i];
			}
			return out;
		}
		
		cvec matvec_impl(std::integral_constant<matname,matname::bdpa> c, const cvec& v) {
			return bdpamatvecMPIblock(v);
			// cout << "Matvec<dpa> called successfully\n";
			cvec out = cvec::Zero(v.size());
			
			int Nr = this->radqN();
			int Nth = this->angqN();
			int Nl = this->thphbasis->bdplmax();
			
			// cout << "Old bdpa Nth: " << Nth << endl;
			
			this->clearVecs();
			
			std::vector<std::vector<cvec> > outparts(Nth);
			for(int i = 0; i < Nth; i++) {
				outparts[i] = std::vector<cvec>(Nl);
				for(int l = 0; l < Nl; l++) {
					outparts[i][l] = cvec::Zero(Nr);
				}
			}
			
			std::vector<csmat> bdpthU(Nth);
			std::vector<csmat> bdpthL(Nth);
			
			for(int i = 0; i < Nth; i++) {
				bdpthU[i] = csmat(Nl,Nth);
				bdpthL[i] = csmat(Nl,Nth);
				
				for(int l = 0; l < Nl; l++) {
					bdpthU[i].row(l) = thphbasis->bdpam(1,l).row(i);
					bdpthL[i].row(l) = thphbasis->bdpam(-1,l).row(i);
				}
			}
			
			// for(int i = 0; i < Nth; i++) {
				// cout << bdpthU[i] << std::endl;
				
				// cout << bdpthL[i] << std::endl;
			
			// }
			
			
			std::vector<cvec> vecparts(Nth);
			
			// cout << "Operating cache size: " << Nth << std::endl;
			
			//#pragma omp parallel
				{
				this->rbasis->template precacheMatfree<bdpa>(v,Nth,Nl);
				
				
				//#pragma omp for collapse(2)
				for(int i = 0; i < Nth; i++) {
					for(int l = 0; l < Nl; l++) {
						for(csmat::InnerIterator j(bdpthU[i],l); j; ++j) {
							int k1 = -1;
							int k2 = -1;
							
							int row = -1;
							int col = -1;
							
							indexMap(j.col(),i,k1,k2,col,row);
							
							// for(int m = 0; m < angids.size(); m++) {
								// if(j.col() == angids[m]) 
								// {
									// k1 = angids[m];
									// col = m;
								// }
								// if(i == angids[m]) 
								// {
									// k2 = angids[m];
									// row = m;
								// }
							// }
							
							//k1 = j.col();
							//k2 = i;
							
							if(k1 !=-1 && k2 != -1) {
								// cout << "k1: " << k1 << ", k2: " << k2 << ", row: " << row << ", col: " << col << ", ul:  1" << std::endl;
								
								this->rbasis->template cacheMatfree<bdpa>(v.segment(Nr*col,Nr),outparts[row][l],col, j.value(),k1,k2,1,l);
								// this->rbasis->template cacheMatfree<bdpa>(v.segment(Nr*col,Nr),outparts[row][l],col, j.value(),0,0,1,l);
							}
						}
						
						for(csmat::InnerIterator j(bdpthL[i],l); j; ++j) {
							int k1 = -1;
							int k2 = -1;
							
							int row = -1;
							int col = -1;
							
							indexMap(j.col(),i,k1,k2,col,row);
							
							// for(int m = 0; m < angids.size(); m++) {
								// if(j.col() == angids[m]) 
								// {
									// k1 = angids[m];
									// col = m;
								// }
								// if(i == angids[m]) 
								// {
									// k2 = angids[m];
									// row = m;
								// }
							// }
							
							//k1 = j.col();
							//k2 = i;
							if(k1 !=-1 && k2 != -1) {
								// cout << "k1: " << k1 << ", k2: " << k2 << ", row: " << row << ", col: " << col << ", ul: -1" << std::endl;
								
								this->rbasis->template cacheMatfree<bdpa>(v.segment(Nr*col,Nr),outparts[row][l],col, j.value(),k1,k2,-1,l);
								//this->rbasis->template cacheMatfree<bdpa>(v.segment(Nr*col,Nr),outparts[row][l],col, j.value(),0,0,-1,l);
							}
						}
					}
				}
				
				
				//#pragma omp for
				for(int i = 0; i < Nth; i++) {
					cvec a = cvec::Zero(Nr);
					for(int j = 0; j < Nl; j++) {
						a.noalias() += outparts[i][j];
					}
					out.segment(Nr*i,Nr) = a;
				}
				
			}
			return out;
		}
		
  		cvec matvec_impl(std::integral_constant<matname,matname::bdpAsq> c, const cvec& v) {
			
			cvec out = cvec::Zero(v.size());
			
			int Nr = this->radqN();
			int Nth = this->angqN();
			int Nl = this->thphbasis->bdplmax();
			
			
			this->clearVecs();
			
			std::vector<std::vector<cvec> > outparts(Nth);
			for(int i = 0; i < Nth; i++) {
				outparts[i] = std::vector<cvec>(Nl);
				for(int l = 0; l < Nl; l++) {
					outparts[i][l] = cvec::Zero(Nr);
				}
			}
			
			if(bdpAsqth.size() != Nth) {
				bdpAsqth = std::vector<csmat>(Nth);
				for(int i = 0; i < Nth; i++) {
					bdpAsqth[i] = csmat(Nl,Nth);
					
					for(int l = 0; l < Nl; l++) {
						bdpAsqth[i].row(l) = thphbasis->bdpAsqm(l).row(i);
					}
				}
			}
			
			this->rbasis->template precacheMatfree<bdpAsq>(v,Nth,Nl);
			
			//#pragma omp parallel for collapse(2)
			for(int i = 0; i < Nth; i++) {
				for(int l = 0; l < Nl; l++) {
					for(csmat::InnerIterator j(bdpAsqth[i],l); j; ++j) {
						int k1 = -1;
						int k2 = -1;
						
						int row = -1;
						int col = -1;
						
						indexMap(j.col(),i,k1,k2,col,row);
						
						// for(int m = 0; m < angids.size(); m++) {
							// if(j.col() == angids[m]) 
							// {
								// k1 = angids[m];
								// col = m;
							// }
							// if(i == angids[m]) 
							// {
								// k2 = angids[m];
								// row = m;
							// }
						// }
						
						//k1 = j.col();
						//k2 = i;
						
						if(k1 !=-1 && k2 != -1) {
							// cout << j.col() << ", " << i << ", " << k1 << ", " << k2 << ", " << col << ", " << row << std::endl;
							this->rbasis->template cacheMatfree<bdpAsq>(v.segment(Nr*col,Nr),outparts[i][l],col, j.value(),l);
						}
						// cout << "outpart " << i << ", " << l << ": " << outparts[i][l] << "\n";
					}
					
				}
			}
			
			//#pragma omp parallel for
			for(int i = 0; i < Nth; i++) {
				cvec a = cvec::Zero(Nr);
				for(int j = 0; j < Nl; j++) {
					a.noalias() += outparts[i][j];
				}
				out.segment(Nr*i,Nr) = a;
			}
			
			return out;
		}
		
		cvec matvec_impl(std::integral_constant<matname,matname::bdpApz> c, const cvec& v) {
			
			cvec out = cvec::Zero(v.size());
			
			int Nr = this->radqN();
			int Nth = this->angqN();
			int Nl = this->thphbasis->bdplmax();
			
			
			this->clearVecs();
			
			std::vector<std::vector<cvec> > outparts(Nth);
			for(int i = 0; i < Nth; i++) {
				outparts[i] = std::vector<cvec>(Nl);
				for(int l = 0; l < Nl; l++) {
					outparts[i][l] = cvec::Zero(Nr);
				}
			}
			
			if(bdpApzthU.size() != Nth) {
				bdpApzthU = std::vector<csmat>(Nth);
				bdpApzthL = std::vector<csmat>(Nth);
				for(int i = 0; i < Nth; i++) {
					bdpApzthU[i] = csmat(Nl,Nth);
					bdpApzthL[i] = csmat(Nl,Nth);
					
					for(int l = 0; l < Nl; l++) {
						bdpApzthU[i].row(l) = thphbasis->bdpApxm(l, 1).row(i);
						bdpApzthL[i].row(l) = thphbasis->bdpApxm(l,-1).row(i);
					}
				}
			}
			
			
			this->rbasis->template precacheMatfree<bdpApz>(v,Nth,Nl);
			
			// cout << "Call to bdpApz matfree \n";
			//#pragma omp parallel for collapse(2)
			
			for(int i = 0; i < Nth; i++) {
				for(int l = 0; l < Nl; l++) {
					for(csmat::InnerIterator j(bdpApzthU[i],l); j; ++j) {
						int k1 = -1;
						int k2 = -1;
						
						int row = -1;
						int col = -1;
						
						indexMap(j.col(),i,k1,k2,col,row);
						
						// for(int m = 0; m < angids.size(); m++) {
							// if(j.col() == angids[m]) 
							// {
								// k1 = angids[m];
								// col = m;
							// }
							// if(i == angids[m]) 
							// {
								// k2 = angids[m];
								// row = m;
							// }
						// }
						
						//k1 = j.col();
						//k2 = i;
						
						if(k1 !=-1 && k2 != -1) {
						// cout << "(row: "<<k1<<", col: " <<k2 <<", l: " << l << "):" << il(k2) << "\n";
						
							this->rbasis->template cacheMatfree<bdpApz>(v.segment(Nr*col,Nr),outparts[i][l],col, j.value(),l,k1,-1);
						}
						// cout << "outpart " << i << ", " << l << ": " << outparts[i][l] << "\n";
					}
					
					for(csmat::InnerIterator j(bdpApzthL[i],l); j; ++j) {
						int k1 = -1;
						int k2 = -1;
						
						int row = -1;
						int col = -1;
						
						indexMap(j.col(),i,k1,k2,col,row);
						
						// for(int m = 0; m < angids.size(); m++) {
							// if(j.col() == angids[m]) 
							// {
								// k1 = angids[m];
								// col = m;
							// }
							// if(i == angids[m]) 
							// {
								// k2 = angids[m];
								// row = m;
							// }
						// }
						
						//k1 = j.col();
						//k2 = i;
						
						if(k1 !=-1 && k2 != -1) {
						// cout << "(row: "<<k1<<", col: " <<k2 <<", l: " << l << "): " << -(il(k2)+1) << "\n";
						
							this->rbasis->template cacheMatfree<bdpApz>(v.segment(Nr*col,Nr),outparts[i][l],col, j.value(),l,k1,1);
						
						}
						// cout << "outpart " << i << ", " << l << ": " << outparts[i][l] << "\n";
					}
				}
			}
			
			//#pragma omp parallel for
			for(int i = 0; i < Nth; i++) {
				cvec a = cvec::Zero(Nr);
				for(int j = 0; j < Nl; j++) {
					a.noalias() += outparts[i][j];
				}
				out.segment(Nr*i,Nr) = a;
			}
			
			return out;
		}
		
		cvec matvec_impl(std::integral_constant<matname,matname::H0> c, const cvec& v) {
			return h0matvecMPIblock(v);
			cvec out(v.size());
			
			int Nr = this->radqN();
			int Nth = this->angqN();
			
			
			std::vector<cvec> outparts;
			for(int i = 0; i < Nth; i++) {
				outparts.push_back(cvec::Zero(Nr));
			}
			
			
			//#pragma omp parallel 
			{
				//#pragma omp for
				for(int i = 0; i < Nth; i++) {
					int k = angids[i];
					this->rbasis->template cacheMatfree<H0>(v.segment(Nr*i,Nr),outparts[i],i,1.0,k);
				}
				
				//#pragma omp for
				for(int i = 0; i < Nth; i++) {
					out.segment(Nr*i,Nr) = outparts[i];
				}
			}
			return out;
		}
		
		//Watch this. We're going to try something.............. Un. Hinged.
		
		cvec h0matvecBlock(const cvec& v) {
			
			int Nr = this->radqN();
			int Nth = this->angqN();
			
			//Do the entire l-independent part as a single matrix-matrix product
			
			// csmat& H0_0 = this->rbasis->getH0mat(0);
			// csmat& H0_1 = this->rbasis->getH0mat(1);
			
			cmat out = cmat::Zero(Nr,Nth);
			
			// out.noalias() += (H0_0 * v.reshaped(Nr,Nth)).reshaped(Nr*Nth,1);
			
			cvec ilist = cvec::Zero(Nth);
			
			for(int i = 0; i < angids.size(); i++) {
				ilist[i] = angids[i];	
			}
			
			this->rbasis->template matmat<H0>(v.reshaped(Nr,Nth),out,ilist);
			
			return out.reshaped(Nr*Nth,1);
			
		}
		
		cvec h0matvecMPIblock(const cvec& v) {
			//v is only received at rank 0
			
			int wrank, wsize;
			
			MPI_Comm_size(MPI_COMM_WORLD, &wsize);
			MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
			
			int Nr = this->radqN();
			int Nth = this->angqN();
			
			
			// int lth0;
			// int lNth;
			
			// if(Nth%wsize == 0) {
				// lNth = Nth/wsize;
				// lth0 = wrank*lNth;
			// }
			// else {
				// float flNth = 1.*Nth/wsize;
				
				// lth0 = round(wrank * flNth);
				
				// lNth = round((wrank + 1) * flNth) - lth0;
				
			// }
			
			// cout << "lth0, lNth at wrank " << wrank << ": " << lth0 << ", " << lNth << endl;
			
			if(v.size() == Nr*Nth) {
				if(Nth <= localNth - localth0) {
					cmat vblocks = v.reshaped(Nr,Nth);
					return h0matvecMPIblock(vblocks).reshaped(Nr*Nth,1);
				}
				else {
					
					cmat vblocks = v.reshaped(Nr,Nth).middleCols(localth0,localNth-localth0);
					
					// cout << "vblocks dims = (" << vblocks.rows() << ", " << vblocks.cols() << ")" << endl;
					
					cmat svb = h0matvecMPIblock(vblocks);
					
					// cout << "svb dims = (" << svb.rows() << ", " << svb.cols() << ")" << endl;
					
					cvec outv;
			
					allgatherVec(svb,outv);
					
					return outv;
					
				}
			}
			else {
				cmat vblocks = v.reshaped(Nr,localNth - localth0);
				return h0matvecMPIblock(vblocks).reshaped(Nr*(localNth - localth0),1);
			}
			
			// cmat vblocks = v.reshaped(Nr,Nth).middleCols(lth0,lNth);
			
			// //Prepare ilist w/ only indices relevant to rank
			// cvec ilist = cvec::Zero(lNth);
			
			// for(int i = 0; i < lNth; i++) {
				// ilist[i] = angids[lth0 + i];
			// }
			
			// // cout << ilist;
			
			// cmat outs = cmat::Zero(Nr,lNth);
			
			// this->rbasis->template matmat<H0>(vblocks,outs,ilist);
			
			// // cout << outs;
			
			// cvec outv;
			
			// allgatherVec(outs,outv);
			
			// // outv = cvec::Zero(this->angqN()*this->radqN());
			
			// // cout << "h0matvec outv size: " << outv.size() << endl;
			
			// return outv;
		}
		
		cvec smatvecMPIblock(const cvec& v) {
			//v is only received at rank 0
			
			int wrank, wsize;
			
			MPI_Comm_size(MPI_COMM_WORLD, &wsize);
			MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
			
			int Nr = this->radqN();
			int Nth = this->angqN();
			
			// int lth0;
			// int lNth;
			
			// if(Nth%wsize == 0) {
				// lNth = Nth/wsize;
				// lth0 = wrank*lNth;
			// }
			// else {
				// float flNth = 1.*Nth/wsize;
				
				// lth0 = round(wrank * flNth);
				
				// lNth = round((wrank + 1) * flNth) - lth0;
				
			// }
			
			// cout << "v.size() = " << v.size() << endl;
			// cout << "Nr*Nth = " << Nr*Nth << endl;
			// cout << "Nr*(localNth - localth0) = " << Nr*(localNth - localth0) << endl;
			
			if(v.size() == Nr*Nth) {
				if(Nth <= localNth - localth0) {
					cmat vblocks = v.reshaped(Nr,Nth);
					
					// cout << "vblocks dims = (" << vblocks.rows() << ", " << vblocks.cols() << ")" << endl;
					
					cmat svb = smatvecMPIblock(vblocks);
					
					// cout << "svb dims = (" << svb.rows() << ", " << svb.cols() << ")" << endl;
					
					return svb.reshaped(Nr*Nth,1);
				}
				else {
					cmat vblocks = v.reshaped(Nr,Nth).middleCols(localth0,localNth-localth0);
					
					// cout << "vblocks dims = (" << vblocks.rows() << ", " << vblocks.cols() << ")" << endl;
					
					cmat svb = smatvecMPIblock(vblocks);
					
					// cout << "svb dims = (" << svb.rows() << ", " << svb.cols() << ")" << endl;
					
					cvec outv;
			
					allgatherVec(svb,outv);
					
					return outv;
					// return svb.transpose().reshaped(Nr*(localNth-localth0),1);
				}
			}
			else {
				cmat vblocks = v.reshaped(Nr,localNth - localth0);

				// cout << "vblocks dims = (" << vblocks.rows() << ", " << vblocks.cols() << ")" << endl;
				
				cmat svb = smatvecMPIblock(vblocks);
				
				// cout << "svb dims = (" << svb.rows() << ", " << svb.cols() << ")" << endl;
				
				return svb.reshaped(Nr*(localNth - localth0),1);
			}
			
			
			// // cout << "lth0, lNth at wrank " << wrank << ": " << lth0 << ", " << lNth << endl;
			
			// cmat vblocks = v.reshaped(Nr,Nth).middleCols(lth0,lNth);
			
			// //Prepare ilist w/ only indices relevant to rank
			// cvec ilist = cvec::Zero(lNth);
			
			// for(int i = 0; i < lNth; i++) {
				// ilist[i] = angids[lth0 + i];
			// }
			
			// // cout << ilist;
			
			// cmat outs = cmat::Zero(Nr,lNth);
			
			// // cout <<"vblocks\n" << vblocks << std::endl;
			
			// this->rbasis->template matmat<S>(vblocks,outs,ilist);
			
			// // cout <<"outblocks\n" << outs << std::endl;
			
			// cvec outv;
			
			// allgatherVec(outs,outv);
			// // outv = cvec::Zero(this->angqN()*this->radqN());
			
			// // cout << "smatvec outv size: " << outv.size() << endl;
			
			// return outv;
		}
		
		csmat angidmat;
		csmat dpthUR = csmat(0,0);
		csmat dpthLR = csmat(0,0);
		
		cvec dpamatvecMPIblock(const cvec& v) {
			return dpamatvecMPIblock_reduceScatter(v);
		}
		
		cvec dpamatvecMPIblock_gather(const cvec& v) {
			//v is only received at rank 0
			
			int wrank, wsize;
			
			MPI_Comm_size(MPI_COMM_WORLD, &wsize);
			MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
			
			int Nr = this->radqN();
			int Nth = this->angqN();
			
			csmat dpthU = this->thphbasis->dpam(1);
			csmat dpthL = this->thphbasis->dpam(-1);
			
			cvec ilist = cvec::Zero(Nth);
			
			for(int i = 0; i < angids.size(); i++) {
				ilist[i] = angids[i];
			}
			
			//Out mat bigger than in mat due to dpth matrix multiplication
			cmat outs = cmat::Zero(Nth/wsize,Nr);
			
			if(!isCached(angidmat)) {
				angidmat = csmat(dpthU.rows(),ilist.rows());
				// cout << "(" << Nth << ", " << ilist.rows() << ")" << endl;
				for(int i = 0; i < ilist.rows(); i++) {
					// cout << "(" << ilist.real().cast<int>()(i) << "," << i << ")" << endl;
					angidmat.insert(ilist.real().cast<int>()(i),i) = 1;
				}
			}
			
			if(!isCached(dpthUR)) {
				csmat tempmat = (angidmat.transpose() * dpthU * angidmat);
				
				dpthUR = tempmat.middleRows(wrank*Nth/wsize,Nth/wsize);
			}
			
			if(!isCached(dpthLR)) {
				csmat tempmat = (angidmat.transpose() * dpthL * angidmat);
				
				dpthLR = tempmat.middleRows(wrank*Nth/wsize,Nth/wsize);
			}
			
			cmat vblock = v.reshaped(Nr,Nth);
			
			//Previously implemented matmat code almost works in parallel case given correct input
			this->rbasis->template matmat<dpa>(vblock,outs,dpthUR,ilist,0,UPPER);
			this->rbasis->template matmat<dpa>(vblock,outs,dpthLR,ilist,0,LOWER);
			
			outs.transposeInPlace();
			
			cvec outv;
			
			allgatherVec(outs,outv);
			
			// outv.resize(Nr*Nth,1);
			
			return outv;
		}
		
		cvec dpamatvecMPIblock_reduce(const cvec& v) {
			int wrank, wsize;
			
			MPI_Comm_size(MPI_COMM_WORLD, &wsize);
			MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
			
			int Nr = this->radqN();
			int Nth = this->angqN();
			
			csmat dpthU = this->thphbasis->dpam(1);
			csmat dpthL = this->thphbasis->dpam(-1);
			
			cmat vblocks;
			pseudoscatterVec(v,vblocks,Nr);
			
			
			
			cvec ilist = cvec::Zero(Nth);
			
			for(int i = 0; i < angids.size(); i++) {
				ilist[i] = angids[i];
			}
			
			//Out mat bigger than in mat due to dpth matrix multiplication
			cmat outs = cmat::Zero(Nr,Nth);
			
			if(!isCached(angidmat)) {
				angidmat = csmat(dpthU.rows(),ilist.rows());
				// cout << "(" << Nth << ", " << ilist.rows() << ")" << endl;
				for(int i = 0; i < ilist.rows(); i++) {
					// cout << "(" << ilist.real().cast<int>()(i) << "," << i << ")" << endl;
					angidmat.insert(ilist.real().cast<int>()(i),i) = 1;
				}
			}
			
			//Due to sparse matrix storage issues, we need to do some silly transposition to get middle columns of dpth mats. Operation inefficient but only performed once.
			
			if(!isCached(dpthUR)) {
				csmat tempmat = (angidmat.transpose() * dpthU * angidmat).transpose();
				
				// cout << "tempmat.middleRows(wrank*Nth/wsize,Nth/wsize).transpose() dims: (" << tempmat.middleRows(wrank*Nth/wsize,Nth/wsize).transpose().rows() << "," << tempmat.middleRows(wrank*Nth/wsize,Nth/wsize).transpose().cols() << ")\n"; 
				// cout << "tempmat.middleRows(wrank*Nth/wsize,Nth/wsize) dims: (" << tempmat.middleRows(wrank*Nth/wsize,Nth/wsize).rows() << "," << tempmat.middleRows(wrank*Nth/wsize,Nth/wsize).cols() << ")\n"; 
				
				dpthUR = tempmat.middleRows(wrank*Nth/wsize,Nth/wsize).transpose();
			
			}
			
			if(!isCached(dpthLR)) {
				csmat tempmat = (angidmat.transpose() * dpthL * angidmat).transpose();
				
				// cout << "tempmat.middleRows(wrank*Nth/wsize,Nth/wsize).transpose() dims: (" << tempmat.middleRows(wrank*Nth/wsize,Nth/wsize).transpose().rows() << "," << tempmat.middleRows(wrank*Nth/wsize,Nth/wsize).transpose().cols() << ")\n"; 
				// cout << "tempmat.middleRows(wrank*Nth/wsize,Nth/wsize) dims: (" << tempmat.middleRows(wrank*Nth/wsize,Nth/wsize).rows() << "," << tempmat.middleRows(wrank*Nth/wsize,Nth/wsize).cols() << ")\n"; 
				
				dpthLR = tempmat.middleRows(wrank*Nth/wsize,Nth/wsize).transpose();
			}
			
			
			
			//Previously implemented matmat code almost works in parallel case given correct input
			((dkbbasis*)(this->rbasis))->matmat_impl_reduce(vblocks,outs,dpthUR,ilist,0,UPPER);
			((dkbbasis*)(this->rbasis))->matmat_impl_reduce(vblocks,outs,dpthLR,ilist,0,LOWER);
			
			
			cmat outv;
			
			allreduceMat(outs,outv);
			
			outv.resize(Nr*Nth,1);
			
			
			return outv;
		}
		
		cvec dpamatvecMPIblock_reduceScatter(const cvec& v) {
			int wrank, wsize;
			
			MPI_Comm_size(MPI_COMM_WORLD, &wsize);
			MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
			
			int Nr = this->radqN();
			int Nth = this->angqN();
			int Nl = this->thphbasis->bdplmax();
			
			if(v.size() == Nr*Nth) {
				if(Nth <= localNth - localth0) {
					cmat vblocks = v.reshaped(Nr,Nth);
					
					// cout << "vblocks dims = (" << vblocks.rows() << ", " << vblocks.cols() << ")" << endl;
					
					cmat svb = dpamatvecMPIblock(vblocks);
					
					// cout << "svb dims = (" << svb.rows() << ", " << svb.cols() << ")" << endl;
					
					return svb.reshaped(Nr*Nth,1);
				}
				else {
					cmat vblocks = v.reshaped(Nr,Nth).middleCols(localth0,localNth-localth0);
					
					// cout << "vblocks dims = (" << vblocks.rows() << ", " << vblocks.cols() << ")" << endl;
					
					cmat svb = dpamatvecMPIblock(vblocks);
					
					// cout << "svb dims = (" << svb.rows() << ", " << svb.cols() << ")" << endl;
					
					cvec outv;
			
					allgatherVec(svb,outv);
					
					return outv;
					// return svb.transpose().reshaped(Nr*(localNth-localth0),1);
				}
			}
			else {
				cmat vblocks = v.reshaped(Nr,localNth - localth0);

				// cout << "vblocks dims = (" << vblocks.rows() << ", " << vblocks.cols() << ")" << endl;
				
				cmat svb = dpamatvecMPIblock(vblocks);
				
				// cout << "svb dims = (" << svb.rows() << ", " << svb.cols() << ")" << endl;
				
				return svb.reshaped(Nr*(localNth - localth0),1);
			}
		}
		
		cmat dpamatvecMPIblock(const cmat& vblocks) {
			int wrank, wsize;
			
			MPI_Comm_size(MPI_COMM_WORLD, &wsize);
			MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
			
			int Nr = this->radqN();
			int Nth = this->angqN();
			int Nl = this->thphbasis->bdplmax();
			
			// cout << "New bdpa Nth: " << Nth << endl;
			
			csmat dpthU = this->thphbasis->dpam(UPPER);
			csmat dpthL = this->thphbasis->dpam(LOWER);
			
			cvec ilist = cvec::Zero(Nth);
			
			for(int i = 0; i < angids.size(); i++) {
				ilist[i] = angids[i];
			}
			
			//Local interval variables are set by the blockDistribute function
			
			// cout << "wsize: " << wsize <<", wrank: " << wrank << ", bdplmax: " <<  Nl << ", Nth: " << Nth << endl; 
			
			// cout << "Local th0: " << localth0 << "\n Local Nth: " << localNth << "\n local l0: " << locall0 << "\n local Nl: " << localNl << endl;
			
			if(!isCached(angidmat)) {
				angidmat = csmat(dpthU.rows(),ilist.rows());
				// cout << "(" << Nth << ", " << ilist.rows() << ")" << endl;
				for(int i = 0; i < ilist.rows(); i++) {
					// cout << "(" << ilist.real().cast<int>()(i) << "," << i << ")" << endl;
					angidmat.insert(ilist.real().cast<int>()(i),i) = 1;
				}
			}
			
			
			cmat outs = cmat::Zero(Nth,Nr);
			
			csmat tempmat;

		    if(!isCached(dpthUR)) {
				tempmat = (angidmat.transpose() * dpthU * angidmat).transpose();
				
				dpthUR = tempmat.middleRows(localth0,localNth-localth0).transpose();
				
				tempmat = (angidmat.transpose() * dpthL * angidmat).transpose();
				
				dpthLR = tempmat.middleRows(localth0,localNth-localth0).transpose();
			}
			
			((dkbbasis*)(this->rbasis))->matmat_impl_reduce(vblocks,outs,dpthLR,ilist,0,LOWER);
			((dkbbasis*)(this->rbasis))->matmat_impl_reduce(vblocks,outs,dpthUR,ilist,0,UPPER);
			
			Eigen::IOFormat outformat(Eigen::FullPrecision,Eigen::DontAlignCols,", ","\n","(","),"," = npy.array((\n","\n))\n",' ');
			
			// cout << "local intervals (" << localth0 << ", " << localNth << "), (" << locall0 << ", " << localNl << ")" << endl;
			// cout << "localOuts " << outs.format(outformat) << endl;
			
			outs.transposeInPlace();
			
			cmat outv(Nr,localNth - localth0);
			
			reduceScatterMat(outs,outv,localNths,localth0s);
			
			// cout << outv.rows() << ", " << outv.cols() << endl;
			
			return outv;
			
		}
		
		csmat* bdpthUR = NULL;
		csmat* bdpthLR = NULL;
		
		
		
		//This distribution algorithm is guaranteed not to be optimal. It's an ad-hoc solution that'll be used once per simulation run.
		
		void blockDistribute() {
			int wrank, wsize;
			
			MPI_Comm_size(MPI_COMM_WORLD, &wsize);
			MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
			
			
			int Nr = this->radqN();
			int Nth = this->angqN();
			int Nl = this->thphbasis->bdplmax();
			
			Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic> wtmat(this->thphbasis->bdplmax(),Nth);
			wtmat.setZero();
			
			cvec ilist = cvec::Zero(Nth);
			
			for(int i = 0; i < angids.size(); i++) {
				ilist[i] = angids[i];
			}
			
			if(!isCached(angidmat)) {
				angidmat = csmat(this->thphbasis->angqN(),ilist.rows());
				// cout << "(" << Nth << ", " << ilist.rows() << ")" << endl;
				for(int i = 0; i < ilist.rows(); i++) {
					// cout << "(" << ilist.real().cast<int>()(i) << "," << i << ")" << endl;
					angidmat.insert(ilist.real().cast<int>()(i),i) = 1;
				}
			}

			//Parallelize this just for fun
			
			//#define PAR_WTMAT
			#ifndef PAR_WTMAT
			for(int i = 0; i < this->thphbasis->bdplmax(); i++) {
				
				
				csmat bdpam = this->thphbasis->bdpam(UPPER,i) + this->thphbasis->bdpam(LOWER,i) ;
				
				// cout << "bdpam\n" << bdpam << endl;
				
				csmat tempmat = (angidmat.transpose() * bdpam * angidmat);
				
				// cout << "tempmat" << tempmat << endl;
				// cout << "l = " << i << endl;
				
				//As sparse matrices are compressed by default, we can't use innerNonZeroPtr for this.
				
				// cout << "outerIndexPtr: " << tempmat.outerIndexPtr() << endl;
				
				for(int j = 0; j < tempmat.outerSize(); j++) {
					// cout << j << ": (" << tempmat.outerIndexPtr()[j] << ", " << tempmat.outerIndexPtr()[j+1] - tempmat.outerIndexPtr()[j] << ")" << endl;
					wtmat(i,j) = tempmat.outerIndexPtr()[j+1] - tempmat.outerIndexPtr()[j];
				}
				
				// wtmat.row(i) = Eigen::Map<Eigen::Matrix<int,Eigen::Dynamic,1> >(tempmat.innerNonZeroPtr(),Nth,1);
			}
			#else
			for(int i = 0; i < this->thphbasis->bdplmax(); i++) {
				
				
				csmat bdpam = this->thphbasis->bdpam(UPPER,i) + this->thphbasis->bdpam(LOWER,i) ;
				
				// cout << "bdpam\n" << bdpam << endl;
				
				csmat tempmat = (angidmat.transpose() * bdpam * angidmat);
				
				// cout << "tempmat" << tempmat << endl;
				// cout << "l = " << i << endl;
				
				//As sparse matrices are compressed by default, we can't use innerNonZeroPtr for this.
				
				// cout << "outerIndexPtr: " << tempmat.outerIndexPtr() << endl;
				
				for(int j = 0; j < tempmat.outerSize(); j++) {
					// cout << j << ": (" << tempmat.outerIndexPtr()[j] << ", " << tempmat.outerIndexPtr()[j+1] - tempmat.outerIndexPtr()[j] << ")" << endl;
					wtmat(i,j) = tempmat.outerIndexPtr()[j+1] - tempmat.outerIndexPtr()[j];
				}
				
				// wtmat.row(i) = Eigen::Map<Eigen::Matrix<int,Eigen::Dynamic,1> >(tempmat.innerNonZeroPtr(),Nth,1);
			}
			#endif
			
			
			// cout << "wtmat\n\n" << wtmat << "\n\n";
			
			int W = wsize;
			
			int N = wtmat.sum();
			
			// cout << "W: " << W << endl;
			// cout << "N: " << N << endl;
			
			locall0 = 0;
			localth0 = 0;
			localNl = Nl;
			localNth = Nth;
			
			int wid = wrank;
			while(W>1) {
				int W0 = floor(W/2.);
				int W1 = ceil(W/2.);
				
				// cout << "wid: " << wid << endl;
				// cout << "W0: " << W0 <<"\nW1: " << W1 << endl;
				
				int sl = localNl - locall0;
				int sth = localNth - localth0;
				
				double W00 = 1.*W0/W;
				double W01 = 1.*W1/W;
				
				int n0, n1;
				int N0,N1;
				
				if(sl>sth) {
					
					n0 = floor(W00 * sl);
					n1 = ceil(W01 * sl);
					
					N0 = wtmat.block(locall0,localth0,n0,sth).sum();
					N1 = wtmat.block(locall0+n0,localth0,n1,sth).sum();
					
				}
				else
				{
					n0 = floor(W00 * sth);
					n1 = ceil(W01 * sth);
					
					N0 = wtmat.block(locall0,localth0,sl,n0).sum();
					N1 = wtmat.block(locall0,localth0 + n0,sl,n1).sum();
				}
				
				// cout << "n0, n1 at wrank " << wrank << ": " << n0 << ", " << n1 << endl;
				// cout << "N0, N1 at wrank " << wrank << ": " << N0 << ", " << N1 << endl;
				
				int N0target = floor(W00 * N);
				
				// cout << "Target N0: " << N0target << endl << "Difference: " << abs(N0 - N0target);
				
				//Adjust n0,n1 s.t. N0,N1 approximately equal
				
				int N0new = N0;
				
				while(true) {
					int n0new, n1new;
					
					if(N0new < N0target) {
						n0new = n0 + 1;
						n1new = n1 - 1;
					}
					else {
						n0new = n0 - 1;
						n1new = n1 + 1;
					}						
					
					if(sl>sth) {
						N0new = wtmat.block(locall0,localth0,n0new,sth).sum();
					}
					else {
						N0new = wtmat.block(locall0,localth0,sl,n0new).sum();
					}
					
					// cout << "n0new: "  << n0 <<", n1new: " << n1 << ", N0new: " << N0new << endl;
					
					// cout << "newDiff: " << abs(N0target - N0new) << ", oldDiff: " << abs(N0target - N0) << endl;
					
					if((abs(N0target - N0new) >= abs(N0target - N0)) && (N0new!= 0 && N0new!=N)) {
						break;
					}
					
					N0 = N0new;
					n0 = n0new;
					n1 = n1new;
				}
				
				
				if(wid < W0) { 
					W = W0;
					if(sl>sth) {
						localNl = locall0 + n0;
					} else {
						localNth = localth0 + n0;
					}
				}
				else {
					wid = wid - W0;
					W = W1;
					
					if(sl>sth) {
						locall0 += n0;
					} else {
						localth0 += n0;
					}
				}
				
				// cout << "W at wrank "<< wrank << ": " << W << endl;
				// cout << "theta,l intervals: {" << localth0 << ", " << localNth << "}, {" << locall0 << ", " << localNl << "}" << endl;
				// cout << "local mat\n" << wtmat.block(locall0,localth0,localNl-locall0,localNth-localth0) << endl;
				N = wtmat.block(locall0,localth0,localNl-locall0,localNth-localth0).sum();
				
				// cout << "N: " << N << endl;
			}
			
			for(int l = 0; l < Nl; l++) {
				if(l < locall0 || l>= localNl) {
					this->rbasis->clearDkbMats(l);
				}
			}
			
			this->rbasis->setLocalParams(localth0,localNth,locall0,localNl);
			
		}
		
		int* localNths;
		int* localth0s;
		
		//This version of blockDistribute also accounts for H0 and S, and distributes the input vector
		void blockDistribute2() {
			int wrank, wsize;
			
			MPI_Comm_size(MPI_COMM_WORLD, &wsize);
			MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
			
			
			int Nr = this->radqN();
			int Nth = this->angqN();
			int Nl = this->thphbasis->bdplmax();
			
			Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic> wtmat(this->thphbasis->bdplmax()+2,Nth);
			wtmat.setZero();
			
			cvec ilist = cvec::Zero(Nth);
			
			for(int i = 0; i < angids.size(); i++) {
				ilist[i] = angids[i];
			}
			
			if(!isCached(angidmat)) {
				angidmat = csmat(this->thphbasis->angqN(),ilist.rows());
				// cout << "(" << Nth << ", " << ilist.rows() << ")" << endl;
				for(int i = 0; i < ilist.rows(); i++) {
					// cout << "(" << ilist.real().cast<int>()(i) << "," << i << ")" << endl;
					angidmat.insert(ilist.real().cast<int>()(i),i) = 1;
				}
			}
			
			//DANGER: This makes it so localNl does not reflect the actual number of l-matrices, as it includes S,H0 as l=0,l=1
			
			for(int j = 0; j < Nth; j++) {
					// cout << j << ": (" << tempmat.outerIndexPtr()[j] << ", " << tempmat.outerIndexPtr()[j+1] - tempmat.outerIndexPtr()[j] << ")" << endl;
					wtmat(0,j) = 1;
					wtmat(1,j) = 1;
				}
			
			for(int i = 0; i < this->thphbasis->bdplmax(); i++) {
				
				
				csmat bdpam = this->thphbasis->bdpam(UPPER,i) + this->thphbasis->bdpam(LOWER,i) ;
				
				// cout << "bdpam\n" << bdpam << endl;
				
				csmat tempmat = (angidmat.transpose() * bdpam * angidmat);
				
				// cout << "tempmat" << tempmat << endl;
				// cout << "l = " << i << endl;
				
				//As sparse matrices are compressed by default, we can't use innerNonZeroPtr for this.
				
				// cout << "outerIndexPtr: " << tempmat.outerIndexPtr() << endl;
				
				for(int j = 0; j < tempmat.outerSize(); j++) {
					// cout << j << ": (" << tempmat.outerIndexPtr()[j] << ", " << tempmat.outerIndexPtr()[j+1] - tempmat.outerIndexPtr()[j] << ")" << endl;
					wtmat(i+2,j) = tempmat.outerIndexPtr()[j+1] - tempmat.outerIndexPtr()[j];
				}
				
			} 
			
			// cout << "wtmat\n\n" << wtmat << "\n\n";
			
			int W = wsize;
			
			int N = wtmat.sum();
			
			cout << "W: " << W << endl;
			cout << "N: " << N << endl;
			
			locall0 = 0;
			localth0 = 0;
			localNl = Nl+2;
			localNth = Nth;
			
			int wid = wrank;
			while(W>1) {
				int W0 = floor(W/2.);
				int W1 = ceil(W/2.);
				
				// cout << "wid: " << wid << endl;
				// cout << "W0: " << W0 <<"\nW1: " << W1 << endl;
				
				int sl = localNl - locall0;
				int sth = localNth - localth0;
				
				double W00 = 1.*W0/W;
				double W01 = 1.*W1/W;
				
				int n0, n1;
				int N0,N1;
				
				// if(sl>sth) {
					
					// n0 = floor(W00 * sl);
					// n1 = ceil(W01 * sl);
					
					// N0 = wtmat.block(locall0,localth0,n0,sth).sum();
					// N1 = wtmat.block(locall0+n0,localth0,n1,sth).sum();
					
				// }
				// else
				// {
					n0 = floor(W00 * sth);
					n1 = ceil(W01 * sth);
					
					N0 = wtmat.block(locall0,localth0,sl,n0).sum();
					N1 = wtmat.block(locall0,localth0 + n0,sl,n1).sum();
				// }
				
				// cout << "n0, n1 at wrank " << wrank << ": " << n0 << ", " << n1 << endl;
				// cout << "N0, N1 at wrank " << wrank << ": " << N0 << ", " << N1 << endl;
				
				int N0target = floor(W00 * N);
				
				// cout << "Target N0: " << N0target << endl << "Difference: " << abs(N0 - N0target);
				
				//Adjust n0,n1 s.t. N0,N1 approximately equal
				
				int N0new = N0;
				
				while(true) {
					int n0new, n1new;
					
					if(N0new < N0target) {
						n0new = n0 + 1;
						n1new = n1 - 1;
					}
					else {
						n0new = n0 - 1;
						n1new = n1 + 1;
					}						
					
					if(sl>sth) {
						N0new = wtmat.block(locall0,localth0,n0new,sth).sum();
					}
					else {
						N0new = wtmat.block(locall0,localth0,sl,n0new).sum();
					}
					
					// cout << "n0new: "  << n0 <<", n1new: " << n1 << ", N0new: " << N0new << endl;
					
					// cout << "newDiff: " << abs(N0target - N0new) << ", oldDiff: " << abs(N0target - N0) << endl;
					
					if((abs(N0target - N0new) >= abs(N0target - N0)) && (N0new!= 0 && N0new!=N)) {
						break;
					}
					
					N0 = N0new;
					n0 = n0new;
					n1 = n1new;
				}
				
				
				if(wid < W0) { 
					W = W0;
					// if(sl>sth) {
						// localNl = locall0 + n0;
					// } else {
						localNth = localth0 + n0;
					// }
				}
				else {
					wid = wid - W0;
					W = W1;
					
					// if(sl>sth) {
						// locall0 += n0;
					// } else {
						localth0 += n0;
					// }
				}
				
				cout << "W at wrank "<< wrank << ": " << W << endl;
				cout << "theta,l intervals: {" << localth0 << ", " << localNth << "}, {" << locall0 << ", " << localNl << "}" << endl;
				// cout << "local mat\n" << wtmat.block(locall0,localth0,localNl-locall0,localNth-localth0) << endl;
				N = wtmat.block(locall0,localth0,localNl-locall0,localNth-localth0).sum();
				
				cout << "N: " << N << endl;
			}
			
			
			locall0-=2;
			localNl-=2;
			
			
			for(int l = 0; l < Nl; l++) {
				if(l < locall0 || l>= localNl) {
					this->rbasis->clearDkbMats(l);
				}
			}
			
			if(locall0 < 0)
				this->rbasis->setLocalParams(localth0,localNth,0,localNl);
			else
				this->rbasis->setLocalParams(localth0,localNth,locall0,localNl);
			
			localNths = new int[wsize];
			localth0s = new int[wsize];
			
			int localN = (localNth - localth0)*Nr;
			int localDisp = localth0 * Nr;
			
			MPI_Allgather(&localN,1,MPI_INT,localNths,1,MPI_INT,MPI_COMM_WORLD);
			MPI_Allgather(&localDisp,1,MPI_INT,localth0s,1,MPI_INT,MPI_COMM_WORLD);
			
			if(wrank == 0) { 
				cout << "localNths:\n";
				for(int i = 0; i < wsize; i++) {
					cout << localNths[i] << endl;
				}
				
				
				cout << "localth0s:\n";
				for(int i = 0; i < wsize; i++) {
					cout << localth0s[i] << endl;
				}
			}
		}
		
		cvec blockDistribute2(const cvec& v) {
			blockDistribute2();
			int Nr = this->radqN();
			return v.segment(localth0*Nr,(localNth-localth0)*Nr);//.reshaped(Nr,(localNth-localth0)));
		}
		
		cvec bdpamatvecMPIblock(const cvec& v) {
			int wrank, wsize;
			
			MPI_Comm_size(MPI_COMM_WORLD, &wsize);
			MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
			
			int Nr = this->radqN();
			int Nth = this->angqN();
			int Nl = this->thphbasis->bdplmax();
			/*
			if(v.size() == Nr*Nth) {
				cmat vblocks = v.reshaped(Nr,Nth).middleCols(localth0,localNth-localth0);
				return bdpamatvecMPIblock(vblocks).reshaped(Nr*Nth,1);
			}
			else {
				cmat vblocks = v.reshaped(Nr,localNth - localth0);
				return bdpamatvecMPIblock(vblocks).reshaped(Nr*(localNth - localth0),1);
			}
			*/
			if(v.size() == Nr*Nth) {
				if(Nth <= localNth - localth0) {
					cmat vblocks = v.reshaped(Nr,Nth);
					
					// cout << "vblocks dims = (" << vblocks.rows() << ", " << vblocks.cols() << ")" << endl;
					
					cmat svb = bdpamatvecMPIblock(vblocks);
					
					// cout << "svb dims = (" << svb.rows() << ", " << svb.cols() << ")" << endl;
					
					return svb.reshaped(Nr*Nth,1);
				}
				else {
					cmat vblocks = v.reshaped(Nr,Nth).middleCols(localth0,localNth-localth0);
					
					// cout << "vblocks dims = (" << vblocks.rows() << ", " << vblocks.cols() << ")" << endl;
					
					cmat svb = bdpamatvecMPIblock(vblocks);
					
					// cout << "svb dims = (" << svb.rows() << ", " << svb.cols() << ")" << endl;
					
					cvec outv;
			
					allgatherVec(svb,outv);
					
					return outv;
					// return svb.transpose().reshaped(Nr*(localNth-localth0),1);
				}
			}
			else {
				cmat vblocks = v.reshaped(Nr,localNth - localth0);

				// cout << "vblocks dims = (" << vblocks.rows() << ", " << vblocks.cols() << ")" << endl;
				
				cmat svb = bdpamatvecMPIblock(vblocks);
				
				// cout << "svb dims = (" << svb.rows() << ", " << svb.cols() << ")" << endl;
				
				return svb.reshaped(Nr*(localNth - localth0),1);
			}
			// // cout << "New bdpa Nth: " << Nth << endl;
			
			// csmat bdpthU = this->thphbasis->bdpam(UPPER,0);
			// csmat bdpthL = this->thphbasis->bdpam(LOWER,0);
			
			// cvec ilist = cvec::Zero(Nth);
			
			// for(int i = 0; i < angids.size(); i++) {
				// ilist[i] = angids[i];
			// }
			
			// //Local interval variables are set by the blockDistribute function
			
			// // cout << "wsize: " << wsize <<", wrank: " << wrank << ", bdplmax: " <<  Nl << ", Nth: " << Nth << endl; 
			
			// // cout << "Local th0: " << localth0 << "\n Local Nth: " << localNth << "\n local l0: " << locall0 << "\n local Nl: " << localNl << endl;
			
			// if(!isCached(angidmat)) {
				// angidmat = csmat(bdpthU.rows(),ilist.rows());
				// // cout << "(" << Nth << ", " << ilist.rows() << ")" << endl;
				// for(int i = 0; i < ilist.rows(); i++) {
					// // cout << "(" << ilist.real().cast<int>()(i) << "," << i << ")" << endl;
					// angidmat.insert(ilist.real().cast<int>()(i),i) = 1;
				// }
			// }
			
			
			
			// cmat outs = cmat::Zero(Nth,Nr);
			
			// // if(!isCached(bdpthUR)) {
				// bdpthUR = new csmat[localNl - locall0];
				// bdpthLR = new csmat[localNl - locall0];
			// // }
			// csmat tempmat;
			
			// for(int l = locall0; l < localNl; l++) {
				// // if(!isCached(bdpthUR[l-locall0])) {
					// bdpthU = this->thphbasis->bdpam(UPPER,l);
					
					// tempmat = (angidmat.transpose() * bdpthU * angidmat).transpose();
					
					// bdpthUR[l-locall0] = tempmat.middleRows(localth0,localNth-localth0).transpose();
					
					// // cout << "bdpthUR[" << l << "]:\n" << bdpthUR[l-locall0] << endl;
				// // }
				
				
				// // if(!isCached(bdpthLR[l-locall0])) {
					// bdpthL = this->thphbasis->bdpam(LOWER,l);
					
					// tempmat = (angidmat.transpose() * bdpthL * angidmat).transpose();
					
					// bdpthLR[l-locall0] = tempmat.middleRows(localth0,localNth-localth0).transpose();
					
					// // cout << "bdpthLR[" << l << "]:\n" << bdpthLR[l-locall0] << endl;
				// // }
				
					
				// this->rbasis->template matmat<bdpa>(vblocks,outs,bdpthLR[l-locall0],ilist,l,LOWER);
				// this->rbasis->template matmat<bdpa>(vblocks,outs,bdpthUR[l-locall0],ilist,l,UPPER);
			// }
			
			// // cout << "local outv:\n" << outs << endl;
			
			
			// //Out mat bigger than in mat due to dpth matrix multiplication
			
			// //Previously implemented matmat code almost works in parallel case given correct input
			
			// // outs.transposeInPlace();
			
			// cmat outv;
		
			// allreduceMat(outs,outv);
			
			// // cout << outv.rows() << ", " << outv.cols() << endl;
			
			// return outv.transpose().reshaped(Nr*Nth,1);
			
		}
		
		cvec blockDistribute3(const cvec& v) {
			int wrank, wsize;
			
			MPI_Comm_size(MPI_COMM_WORLD, &wsize);
			MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
			
			
			int Nr = this->radqN();
			int Nth = this->angqN();
			int Nl = this->thphbasis->bdplmax();
			
			Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic> wtmat(Nth,Nth);
			wtmat.setIdentity();
			
			wtmat *=2;
			
			cvec ilist = cvec::Zero(Nth);
			
			for(int i = 0; i < angids.size(); i++) {
				ilist[i] = angids[i];
			}
			
			if(!isCached(angidmat)) {
				angidmat = csmat(this->thphbasis->angqN(),ilist.rows());
				// cout << "(" << Nth << ", " << ilist.rows() << ")" << endl;
				for(int i = 0; i < ilist.rows(); i++) {
					// cout << "(" << ilist.real().cast<int>()(i) << "," << i << ")" << endl;
					angidmat.insert(ilist.real().cast<int>()(i),i) = 1;
				}
			}
			
			for(int i = 0; i < this->thphbasis->bdplmax(); i++) {
				
				
				csmat bdpamU = this->thphbasis->bdpam(UPPER,i);
				csmat bdpamL = this->thphbasis->bdpam(LOWER,i);
				
				csmat tempmatU = (angidmat.transpose() * bdpamU * angidmat);
				csmat tempmatL = (angidmat.transpose() * bdpamL * angidmat);
				
				cdouble* iidU = tempmatU.valuePtr();
				cdouble* iidL = tempmatL.valuePtr();
				for(int i = 0; i < tempmatU.nonZeros(); i++) {
					iidU[i] = cdouble(1,0);
				}
				for(int i = 0; i < tempmatL.nonZeros(); i++) {
					iidL[i] = cdouble(1,0);
				}
				
				wtmat += tempmatU.real().cast<int>() + tempmatL.real().cast<int>();
				
			} 
			
			Eigen::IOFormat outformat(Eigen::FullPrecision,Eigen::DontAlignCols,", ","\n","(","),"," = npy.array((\n","\n))\n",' ');
			cout << "wtmat\n\n" << wtmat.format(outformat) << "\n\n";
			
			
			int W = wsize;
			
			int N = wtmat.sum();
			
			cout << "W: " << W << endl;
			cout << "N: " << N << endl;
			
			int th01 = 0,th02 = 0,thN1 = Nth,thN2 = Nth;
			
			int wid = wrank;
			while(W>1) {
				int W0 = floor(W/2.);
				int W1 = ceil(W/2.);
				
				cout << "wid: " << wid << endl;
				cout << "W0: " << W0 <<"\nW1: " << W1 << endl;
				
				int sth1 = thN1 - th01;
				int sth2 = thN2 - th02;
				
				double W00 = 1.*W0/W;
				double W01 = 1.*W1/W;
				
				cout << "W00: " << W00 << endl;
				cout << "W01: " << W01 << endl;
				
				int n0, n1;
				int N0,N1;
				
				if(sth1>sth2) {
					
					n0 = floor(W00 * sth1);
					n1 =  ceil(W01 * sth1);
					
					cout << "n0, n1: " << n0 << ", " << n1 << endl;
					
					cout << "(" << th01 << ", " << th02 << ", " << n0 <<", " << sth2 << ")" << endl;
					cout << "(" << th01+n0 << ", " << th02 << ", " << n1 <<", " << sth2 << ")" << endl;
					
					N0 = wtmat.block(th01,th02,n0,sth2).sum();
					N1 = wtmat.block(th01+n0,th02,n1,sth2).sum();
					
				}
				else
				{
					cout << "n0, n1: " << n0 << ", " << n1 << endl;
					
					n0 = floor(W00 * sth2);
					n1 =  ceil(W01 * sth2);
					
					cout << "(" << th01 << ", " << th02 << ", " << sth1 <<", " << n0 << ")" << endl;
					cout << "(" << th01 << ", " << th02+n0 <<", " << sth1 << ", " << n1  << ")" << endl;
					
					N0 = wtmat.block(th01,th02,sth1,n0).sum();
					N1 = wtmat.block(th01,th02+n0,sth1,n0).sum();
				}
				
				cout << "n0, n1 at wrank " << wrank << ": " << n0 << ", " << n1 << endl;
				cout << "N0, N1 at wrank " << wrank << ": " << N0 << ", " << N1 << endl;
				
				int N0target = floor(W00 * N);
				
				cout << "Target N0: " << N0target << endl << "Difference: " << abs(N0 - N0target);
				
				//Adjust n0,n1 s.t. N0,N1 approximately equal
				
				int N0new = N0;
				
				while(true) {
					int n0new, n1new;
					
					if(N0new < N0target) {
						n0new = n0 + 1;
						n1new = n1 - 1;
					}
					else {
						n0new = n0 - 1;
						n1new = n1 + 1;
					}						
					
					if(sth1>sth2) {
						N0new = wtmat.block(th01,th02,n0new,sth2).sum();
					}
					else {
						N0new = wtmat.block(th01,th02,sth1,n0new).sum();
					}
					
					// cout << "n0new: "  << n0 <<", n1new: " << n1 << ", N0new: " << N0new << endl;
					
					// cout << "newDiff: " << abs(N0target - N0new) << ", oldDiff: " << abs(N0target - N0) << endl;
					
					if((abs(N0target - N0new) >= abs(N0target - N0)) && (N0new!= 0 && N0new!=N)) {
						break;
					}
					
					N0 = N0new;
					n0 = n0new;
					n1 = n1new;
				}
				
				
				if(wid < W0) { 
					W = W0;
					if(sth1>sth2) {
						thN1 = th01 + n0;
					} else {
						thN2 = th02 + n0;
					}
				}
				else {
					wid = wid - W0;
					W = W1;
					
					if(sth1>sth2) {
						th01 += n0;
					} else {
						th02 += n0;
					}
				}
				
				cout << "W at wrank "<< wrank << ": " << W << endl;
				cout << "th1,th2 intervals: {" << th01 << ", " << thN1 << "}, {" << th02 << ", " << thN2 << "}" << endl;
				// cout << "local mat\n" << wtmat.block(locall0,localth0,localNl-locall0,localNth-localth0) << endl;
				N = wtmat.block(th01,th02,thN1-th01,thN2-th02).sum();
				
				cout << "N: " << N << endl;
			}
			
			
			// locall0-=2;
			// localNl-=2;
			
			
			// for(int l = 0; l < Nl; l++) {
				// if(l < locall0 || l>= localNl) {
					// this->rbasis->clearDkbMats(l);
				// }
			// }
			
			// if(locall0 < 0)
				// this->rbasis->setLocalParams(localth0,localNth,0,localNl);
			// else
			this->rbasis->setLocalParams(th01,thN1,0,Nl);
			
			// localNths = new int[wsize];
			// localth0s = new int[wsize];
			
			// int localN = (localNth - localth0)*Nr;
			// int localDisp = localth0 * Nr;
			
			// MPI_Allgather(&localN,1,MPI_INT,localNths,1,MPI_INT,MPI_COMM_WORLD);
			// MPI_Allgather(&localDisp,1,MPI_INT,localth0s,1,MPI_INT,MPI_COMM_WORLD);
			
			// if(wrank == 0) { 
				// cout << "localNths:\n";
				// for(int i = 0; i < wsize; i++) {
					// cout << localNths[i] << endl;
				// }
				
				
				// cout << "localth0s:\n";
				// for(int i = 0; i < wsize; i++) {
					// cout << localth0s[i] << endl;
				// }
			// }
			
			int* th01s = new int[wsize];
			int* th02s = new int[wsize];
			int* thN1s = new int[wsize];
			int* thN2s = new int[wsize];
			
			MPI_Allgather(&th01, 1, MPI_INT, th01s, 1, MPI_INT, MPI_COMM_WORLD);
			MPI_Allgather(&th02, 1, MPI_INT, th02s, 1, MPI_INT, MPI_COMM_WORLD);
			MPI_Allgather(&thN1, 1, MPI_INT, thN1s, 1, MPI_INT, MPI_COMM_WORLD);
			MPI_Allgather(&thN2, 1, MPI_INT, thN2s, 1, MPI_INT, MPI_COMM_WORLD);
			
			Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic> blockmat(Nth,Nth);
			
			for(int i = 0; i < wsize; i++) {
				blockmat.block(th01s[i],th02s[i],thN1s[i] - th01s[i], thN2s[i] - th02s[i]) 
					= Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic>::Constant(thN1s[i] - th01s[i], thN2s[i] - th02s[i],i);
			}
			
			cout << "blockmat" << blockmat.format(outformat);
			
			MPI_Barrier(MPI_COMM_WORLD);
			
			
			return cvec::Zero(0);//return v.segment(localth0*Nr,(localNth-localth0)*Nr);//.reshaped(Nr,(localNth-localth0)));
		}
		
		
		cmat bdpamatvecMPIblock(const cmat& vblocks) {
			int wrank, wsize;
			
			MPI_Comm_size(MPI_COMM_WORLD, &wsize);
			MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
			
			int Nr = this->radqN();
			int Nth = this->angqN();
			int Nl = this->thphbasis->bdplmax();
			
			// cout << "New bdpa Nth: " << Nth << endl;
			
			csmat bdpthU = this->thphbasis->bdpam(UPPER,0);
			csmat bdpthL = this->thphbasis->bdpam(LOWER,0);
			
			cvec ilist = cvec::Zero(Nth);
			
			for(int i = 0; i < angids.size(); i++) {
				ilist[i] = angids[i];
			}
			
			//Local interval variables are set by the blockDistribute function
			
			// cout << "wsize: " << wsize <<", wrank: " << wrank << ", bdplmax: " <<  Nl << ", Nth: " << Nth << endl; 
			
			// cout << "Local th0: " << localth0 << "\n Local Nth: " << localNth << "\n local l0: " << locall0 << "\n local Nl: " << localNl << endl;
			
			if(!isCached(angidmat)) {
				angidmat = csmat(bdpthU.rows(),ilist.rows());
				// cout << "(" << Nth << ", " << ilist.rows() << ")" << endl;
				for(int i = 0; i < ilist.rows(); i++) {
					// cout << "(" << ilist.real().cast<int>()(i) << "," << i << ")" << endl;
					angidmat.insert(ilist.real().cast<int>()(i),i) = 1;
				}
			}
			
			
			cmat outs = cmat::Zero(Nth,Nr);
			
			if(!isCached(bdpthUR)) {
				bdpthUR = new csmat[localNl - locall0];
				bdpthLR = new csmat[localNl - locall0];
			}
			csmat tempmat;
			
			for(int l = locall0 + 2; l < localNl; l++) {
				// if(l>=0) {
					if(!isCached(bdpthUR[l-(locall0+2)])) {
						bdpthU = this->thphbasis->bdpam(UPPER,l);
						
						tempmat = (angidmat.transpose() * bdpthU * angidmat).transpose();
						
						// cout << "l-(locall0+2)" << l-(locall0+2) << endl;
						
						bdpthUR[l-(locall0+2)] = tempmat.middleRows(localth0,localNth-localth0).transpose();
						
						// cout << "bdpthUR[" << l << "]:\n" << bdpthUR[l-locall0] << endl;
					}
					
					
					if(!isCached(bdpthLR[l-(locall0+2)])) {
						bdpthL = this->thphbasis->bdpam(LOWER,l);
						
						tempmat = (angidmat.transpose() * bdpthL * angidmat).transpose();
						
						// cout << "l-(locall0+2)" << l-(locall0+2) << endl;
						
						bdpthLR[l-(locall0+2)] = tempmat.middleRows(localth0,localNth-localth0).transpose();
						
						// cout << "bdpthLR[" << l << "]:\n" << bdpthLR[l-locall0] << endl;
					}
					
						
					this->rbasis->template matmat<bdpa>(vblocks,outs,bdpthLR[l-(locall0+2)],ilist,l,LOWER);
					this->rbasis->template matmat<bdpa>(vblocks,outs,bdpthUR[l-(locall0+2)],ilist,l,UPPER);
				// }
			}
			
			Eigen::IOFormat outformat(Eigen::FullPrecision,Eigen::DontAlignCols,", ","\n","(","),"," = npy.array((\n","\n))\n",' ');
			
			// cout << "local intervals (" << localth0 << ", " << localNth << "), (" << locall0 << ", " << localNl << ")" << endl;
			// cout << "localOuts " << outs.format(outformat) << endl;
			
			outs.transposeInPlace();
			
			cmat outv(Nr,localNth - localth0);
			
			reduceScatterMat(outs,outv,localNths,localth0s);
			
			// cout << outv.rows() << ", " << outv.cols() << endl;
			
			return outv;
			
		}
		
		cmat h0matvecMPIblock(const cmat& vblocks) {
			//v is only received at rank 0
			
			int wrank, wsize;
			
			MPI_Comm_size(MPI_COMM_WORLD, &wsize);
			MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
			
			int Nr = this->radqN();
			int Nth = this->angqN();
			
			// cout << "lth0, lNth at wrank " << wrank << ": " << lth0 << ", " << lNth << endl;
			
			//Prepare ilist w/ only indices relevant to rank
			cvec ilist = cvec::Zero(localNth-localth0);
			
			for(int i = 0; i < localNth-localth0; i++) {
				ilist[i] = angids[localth0 + i];
			}
			
			// cout << ilist;
			
			cmat outs = cmat::Zero(Nr,localNth-localth0);
			if(locall0 <= -1)
			this->rbasis->template matmat<H0>(vblocks,outs,ilist);
			
			return outs;
		}
		
		cmat smatvecMPIblock(const cmat& vblocks) {
			//v is only received at rank 0
			
			int wrank, wsize;
			
			MPI_Comm_size(MPI_COMM_WORLD, &wsize);
			MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
			
			int Nr = this->radqN();
			int Nth = this->angqN();
			
			
			// cout << "lth0, lNth at wrank " << wrank << ": " << lth0 << ", " << lNth << endl;
			
			//Prepare ilist w/ only indices relevant to rank
			cvec ilist = cvec::Zero(localNth-localth0);
			
			for(int i = 0; i < localNth-localth0; i++) {
				ilist[i] = angids[localth0 + i];
			}
			
			// cout << ilist;
			
			cmat outs = cmat::Zero(Nr,localNth-localth0);
			
			// cout <<"vblocks\n" << vblocks << std::endl;
			
			if(locall0 <= -2)
			this->rbasis->template matmat<S>(vblocks,outs,ilist);
			
			
			return outs;
		}
		
		
		cvec pzmatvecBlock(const cvec& v) {
			
			int Nr = this->radqN();
			int Nth = this->angqN();
			
			cvec ilist = cvec::Zero(Nth);
			
			
			for(int i = 0; i < angids.size(); i++) {
				ilist[i] = angids[i];	
			}
			
			cmat out = cmat::Zero(Nr,Nth);
			
			
			for(int ul = -1; ul < 2; ul+=2) {
				csmat& pzm = thphbasis->pzm(ul);
					
				cmat outm = cmat::Zero(Nr,Nth);
				
				this->rbasis->template matmat<pz>(v.reshaped(Nr,Nth),outm,ilist,0,ul);
				// outmats[uli][l] = (bdpApxm * outmats[uli][l].transpose()).transpose();
				out.noalias() += (pzm * outm.transpose()).transpose();
			}
			return out.reshaped(Nr*Nth,1);
		}
		
		cvec dpamatvecBlock(const cvec& v) {
			int Nr = this->radqN();
			int Nth = this->angqN();
			
			cvec ilist = cvec::Zero(Nth);
			
			
			for(int i = 0; i < angids.size(); i++) {
				ilist[i] = angids[i];	
			}
			
			
			cmat out = cmat::Zero(Nth,Nr);
			
			cmat inmat = v.reshaped(Nr,Nth);
			// #pragma omp parallel
			{
				for(int ul = -1; ul < 2; ul+=2) {
					csmat& dpam = thphbasis->dpam(ul);
						
					cmat outm = cmat::Zero(Nth,Nr);
					
					// cmat inm = dpam * v.reshaped(Nr,Nth)
					
					this->rbasis->template matmat<dpa>(inmat,outm,dpam,ilist,0,ul);
					// outmats[uli][l] = (bdpApxm * outmats[uli][l].transpose()).transpose();
					out.noalias() += outm;
				}
			}
			return out.transpose().reshaped(Nr*Nth,1);
		}
		
		cvec bdpApxmatvecBlock(const cvec& v) {
			
			int Nr = this->radqN();
			int Nth = this->angqN();
			int Nl = this->thphbasis->bdplmax();
			
			//Do the entire l-independent part as a single matrix-matrix product
			
			// csmat& H0_0 = this->rbasis->getH0mat(0);
			// csmat& H0_1 = this->rbasis->getH0mat(1);
			
			cmat out = cmat::Zero(Nr,Nth);
			
			// out.noalias() += (H0_0 * v.reshaped(Nr,Nth)).reshaped(Nr*Nth,1);
			
			cvec ilist = cvec::Zero(Nth);
			
			for(int i = 0; i < angids.size(); i++) {
				ilist[i] = angids[i];	
			}
			
			
			if(bdpApzthU.size() != Nth) {
				bdpApzthU = std::vector<csmat>(Nth);
				bdpApzthL = std::vector<csmat>(Nth);
				for(int i = 0; i < Nth; i++) {
					bdpApzthU[i] = csmat(Nl,Nth);
					bdpApzthL[i] = csmat(Nl,Nth);
					
					for(int l = 0; l < Nl; l++) {
						bdpApzthU[i].row(l) = thphbasis->bdpApxm(l, 1).row(i);
						bdpApzthL[i].row(l) = thphbasis->bdpApxm(l,-1).row(i);
					}
				}
			}
			// std::vector<vector<cmat>> outmats(2);
			// for(int ul = 0; ul < 2; ul++ ) {
				// outmats[ul] = std::vector<cmat>(Nl);
				// for(int l = 0; l < Nl; l++) {
					// outmats[ul][l] = cmat::Zero(Nr,Nth);
				// }
			// }
			
			for(int ul = -1; ul < 2; ul+=2) {
				cmat bdplout(Nr*Nth,Nl);
				
				for(int l = 0; l < Nl; l++) {
					int uli = (ul + 1)/2;
					
					cmat outl = cmat::Zero(Nr,Nth);
					
					this->rbasis->template matmat<bdpApx>(v.reshaped(Nr,Nth),outl,ilist,l, ul);
					
					csmat& bdpApxm = thphbasis->bdpApxm(l,-ul);
					
					
					// outmats[uli][l] = (bdpApxm * outmats[uli][l].transpose()).transpose();
					out.noalias() += (bdpApxm * outl.transpose()).transpose();
					
					// bdplout.col(l).noalias() = outl.reshaped(Nr*Nth,1);
				}
				
				// for(int i = 0; i < Nth; i++) {
					// if(ul == 1)
						// out.noalias() += bdpApzthU[i] * bdplout;
					// if(ul ==-1)
						// out.noalias() += bdpApzthL[i] * bdplout;
					
				// }
			}
			
			// for(int ul = 0; ul < 2; ul++ ) {
				// for(int l = 0; l < Nl; l++) {
					// out.noalias() += outmats[ul][l];
				// }
			// }
			
			return out.reshaped(Nr*Nth,1);
			
		}
		
		cvec bdpAsqmatvecBlock(cvec& v) {
			
			int Nr = this->radqN();
			int Nth = this->angqN();
			int Nl = this->thphbasis->bdplmax();
			
			//Do the entire l-independent part as a single matrix-matrix product
			
			// csmat& H0_0 = this->rbasis->getH0mat(0);
			// csmat& H0_1 = this->rbasis->getH0mat(1);
			cvec ilist = cvec::Zero(Nth);

			cmat out = cmat::Zero(Nr,Nth);
			
			// out.noalias() += (H0_0 * v.reshaped(Nr,Nth)).reshaped(Nr*Nth,1);
			
			for(int l = 0; l < Nl; l++) {
				csmat& bdpAsqth = thphbasis->bdpAsqm(l);
				
				cmat w = cmat::Zero(Nr,Nth);
				
				this->rbasis->template matmat<bdpAsq>(v.reshaped(Nr,Nth),w,ilist,l,0);
				
				out.noalias() += (w * bdpAsqth);
			}
			
			return out.reshaped(Nr*Nth,1);
			
		}
		
		template <matname M>
		cvec matvecBlock(const cvec& v) {
			int Nr = this->radqN();
			int Nth = this->angqN();
			
			cmat out = cmat::Zero(Nr,Nth);
			
			if(!isCached(angidsvec)) {
				angidsvec = cvec::Zero(Nth);
				for(int i = 0; i < angids.size(); i++) {
					angidsvec[i] = angids[i];	
				}
			}
			
			this->rbasis->template matmat<M>(v.reshaped(Nr,Nth),out,angidsvec);
			
			return out.reshaped(Nr*Nth,1);
		}
		
		cvec matvec_impl(std::integral_constant<matname,S> c, const cvec& v) {
			return smatvecMPIblock(v);
		}
		
		template <matname M>
		cvec matvec_impl(std::integral_constant<matname, M> c, const cvec& v) {
			// return matvecBlock<M>(v);
			csmat& Mth = this->thphbasis->template getmat<M>();
			cvec out(v.size());
			
			int Nr = this->radqN();
			int Nth = this->angqN();
			
			for(int i = 0; i < Mth.outerSize(); i++) {
				for(csmat::InnerIterator j(Mth,i); j; ++j) {
					
					int k1 = -1;
					int k2 = -1;
					
					int row = -1;
					int col = -1;
					
					indexMap(j.row(),j.col(),k1,k2,row,col);
					// for(int m = 0; m < angids.size(); m++) {
						// if(j.row() == angids[m]) 
						// {
							// k1 = angids[m];
							// row = m;
						// }
						// if(j.col() == angids[m]) 
						// {
							// k2 = angids[m];
							// col = m;
						// }
						
					// }
					
					if(k1 !=-1 && k2 != -1) {
						if(row == col) this->rbasis->setState(k1);
						else this->rbasis->setState(k1,k2,0);
						
						out.segment(Nr*row,Nr) = (this->rbasis->template matfree<M>(j.value() * v.segment(Nr*col,Nr)));
					}
				}
			}
			return out;
		}
		
		void pruneUncoupled(vec angStates,bool bdp = false) {
			if(isCached(angCouplings(bdp))) {
				
				csmat angcp = angCouplings(bdp);
				
				// cout << angcp << "\n";
				
				int rn = this->radqN();
				int an = this->angqN();
				
				// cout << rn << "," << an << "\n";
				
				// cmat eangc = expm(angcp,20);
				
				// cout << eangc << "\n";
				
				// cvec angslect = eangc * angStates;
				int nExpm = sqrt(an/2);
				
				cvec angslect = expmvec(angcp,angStates,nExpm);						   
				cout << angStates << "\n";
				cout << angslect << "\n";
				vector<int> slids(0);
				for(int i = 0; i < angslect.size(); i++) {
					if(angslect[i]!=cdouble(0.0,0.0)) slids.push_back(i);
				}	
				
				int nslect = slids.size();
				
				/*for(int i = 0; i < nslect; i++) {
					cout << slids[i] << ",";
				}
				*/
				if((an - nslect) > 10) {
					this->angovr(nslect);
					angids = slids;
				}
				else {
				        for(int i = 0; i < an; i++) {
						angids[i] = i;
					}
				}
				angidsReverse = vector<int>(an);
				for(int i = 0; i < an; i++) {
					angidsReverse[i] = -1;
					
					for(int j = 0; j < nslect; j++) {
						if(angids[j] == i)
							angidsReverse[i] = j;
					}
				}
				
				this->template getmat<dpa>() = csmat(0,0);
			}
		}
		
		int rqn() {
			return this->rbasis->radqN();
		}

		int aqn() {
			if (this->angOverride == 0)
				return this->thphbasis->angqN();
			else return this->angOverride;
		}
		
		int amax() {
			return this->thphbasis->angMax();
		}
		
		void clvcs() {
			rbasis->clearVecs();
		}
		csmat& angCouplings(bool bdp = false) { return thphbasis->angCouplings(bdp); }
		
		typename basistraits<rthphbasis<rbtype, thphbtype> >::radial_type& radial() const { return *rbasis; }
		typename basistraits<rthphbasis<rbtype, thphbtype> >::angular_type& angular() const { return *thphbasis; }
		
};




//It's called the evil product because it shouldn't be touched unless absolutely necessary
void evilprod(csmat& out,const csmat& a,const csmat& b) {
	out = csmat(a.rows()*b.rows(),a.cols()*b.cols());
	std::vector<T> triplets;
	triplets.reserve(a.nonZeros() * b.nonZeros());
	
	
	for(int i = 0; i < b.outerSize(); i++) {
		for(csmat::InnerIterator j(b,i); j; ++j) {
			for(int k = 0; k < a.outerSize(); k++) {
				for(csmat::InnerIterator l(a,k); l; ++l) {
					triplets.push_back(T(
						j.row() * a.rows() + l.row(),
						j.col() * a.cols() + l.col(),
						j.value() * l.value()
					));
				}
			}
		}
	}
	
	out.setFromTriplets(triplets.begin(),triplets.end());
	
	//cout << *out;
}



/*
def pzmat(self,lmax = 0):
        pzm = sprs.lil_matrix(((lmax+1)*self.N,(lmax+1)*self.N),dtype=complex)
        for l in range(0,lmax+1):
            lpot1=nltc.coloumb(-(l+1))
            lpot2=nltc.coloumb(-l)
            dd = sprs.lil_matrix(self.ddmat(1),dtype = complex)
            E1 = sprs.lil_matrix(self.Emat(lpot1),dtype = complex)
            E2 = sprs.lil_matrix(self.Emat(lpot2),dtype = complex)
            
            
            ddd = sprs.dia_matrix(dd.shape,dtype = complex)
            E1d = sprs.dia_matrix(E1.shape,dtype = complex)
            E2d = sprs.dia_matrix(E2.shape,dtype = complex)
            
            for i in range(0,self.k+1): 
                ddd.setdiag(dd.diagonal(i),i)
                ddd.setdiag(dd.diagonal(-i),-i)
                
                E1d.setdiag(E1.diagonal(i),i)
                E1d.setdiag(E1.diagonal(-i),-i)
                
                E2d.setdiag(E2.diagonal(i),i)
                E2d.setdiag(E2.diagonal(-i),-i)
                
            # plt.imshow(ddd.toarray().real)
            # plt.imshow(E1d.toarray().real)
            
            if l-1 >= 0 and l+1 < self.N:
                pzm[l*self.N:(l+1)*self.N,(l-1)*self.N:l*self.N] = ddd + E1d
                pzm[(l-1)*self.N:l*self.N,l*self.N:(l+1)*self.N] = ddd - E2d
        return sprs.dia_matrix(pzm)
*/

/*
cvec matvec_impl(std::integral_constant<matname,matname::bdpa> c, const cvec& v) {
	cout << "Matvec<dpa> called successfully\n";
	
	
	cvec out = cvec::Zero(v.size());
	
	int Nr = this->radqN();
	int Nth = this->angqN();
	int Nl = this->thphbasis->bdplmax();
	

			 

	std::vector<std::vector<cvec> > outparts(Nth);
	for(int i = 0; i < Nth; i++) {
		outparts[i] = std::vector<cvec>(Nl);
		for(int j = 0; j < Nl; j++) {
			outparts[i][j] = cvec::Zero(Nr);
		}
	}
	cout << "v size: " << v.size() << ", Nr: " << Nr << ", Nth: " << Nth <<"\n";
	std::vector<csmat*> bdpthU(Nl);
	std::vector<csmat*> bdpthL(Nl);
	std::vector<csmat*>  bdpth(Nl);
	this->clearVecs();
	
						  
						  
	
	for(int l = 0; l < Nl; l++) {
					  
					  
		
		this->rbasis->setState(0,0,1,l);
		bdpthU[l] = &(this->thphbasis->bdpam(1,l));
		
		for(int i = 0; i < Nth; i++) {
		//	cout << i << " " << Nr*i << std::endl;
			this->rbasis->template precacheMatfree<bdpa>(v.segment(Nr*i,Nr),i,l);
		}


						   
						   
		
		this->rbasis->setState(0,0,-1,l);
		bdpthL[l] = &(this->thphbasis->bdpam(-1,l));
		
		for(int i = 0; i < Nth; i++) {
			this->rbasis->template precacheMatfree<bdpa>(v.segment(Nr*i,Nr),i,l);
		}
		
		bdpth[l] = new csmat(Nth,Nth);
		*(bdpth[l]) = (*(bdpthU[l])) + (*(bdpthL[l]));
		
		cout << *(bdpth[l]) << std::endl;
	}


						   


	
												  
	
	
	//#pragma omp parallel for collapse(2)
	for(int l = 0; l < Nl; l++) {
		//#pragma omp parallel for
		
		for(int i = 0; i < bdpthU[l]->outerSize(); i++) {
		//csmat bdpth = bdpthU + bdpthL;
		//csmat::InnerIterator j = csmat::InnerIterator(bdpthU,i);
			
			//while (!j) {
			for(csmat::InnerIterator j(*(bdpthU[l]),i); j; ++j) {
				int k1 = -1;
				int k2 = -1;
				
				int row = -1;
				int col = -1;
				for(int m = 0; m < angids.size(); m++) {
					if(j.row() == angids[m]) 
					{
						k1 = angids[m];
						row = m;
					}
					if(j.col() == angids[m]) 
					{
						k2 = angids[m];
						col = m;
					}
				}
				if(k1 !=-1 && k2 != -1) {
							
					
					this->rbasis->template cacheMatfree<bdpa>(v.segment(Nr*col,Nr),outparts[row][l],col, j.value(),k1,k2,1,l);
					
				}
				
				//++j;
			}
		}
	}
	
	//#pragma omp parallel for collapse(2)
	for(int l = 0; l < Nl; l++) {
		for(int i = 0; i < bdpthL[l]->outerSize(); i++) {
			//csmat::InnerIterator j = csmat::InnerIterator(bdpthL,i);
			for(csmat::InnerIterator j(*(bdpthL[l]),i); j; ++j) {
			
											   
				
				int k1 = -1;
				int k2 = -1;
				
				int row = -1;
				int col = -1;
				for(int m = 0; m < angids.size(); m++) {
					if(j.row() == angids[m]) 
					{
						k1 = angids[m];
						row = m;
					}
					if(j.col() == angids[m]) 
					{
						k2 = angids[m];
						col = m;
					}
				}
				if(k1 !=-1 && k2 != -1) {
				
					this->rbasis->template cacheMatfree<bdpa>(v.segment(Nr*col,Nr),outparts[row][l],col, j.value(),k1,k2,-1,l);
					
					
				}
				//++j;
			}
		}
		
	}
	

	//#pragma omp parallel for
	for(int i = 0; i < Nth; i++) {
					
		for(int j = 0; j < Nl; j++) {
			out.segment(Nr*i,Nr).noalias() += outparts[i][j];
		}
					 
	}
	
	
	return out;
}*/

#endif

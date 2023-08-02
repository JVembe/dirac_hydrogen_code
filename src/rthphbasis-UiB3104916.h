#ifndef RTHPHBASIS_H
#define RTHPHBASIS_H

#include "defs.h"
#include "cWavefunc.h"
#include <omp.h>

long double lpot(long double r) {
	if(r == 0.0) return 0;
	else{
		return 1.0/r;
	}
}

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
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

	
	//Utility function for mapping indices in matvec function
	
	
	public:
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
		
		rthphbasis(basis<rbtype>& rbasis, basis<thphbtype>& thphbasis):basis<rthphbasis<rbtype,thphbtype> >() {
			this->rbasis = &rbasis;
			this->thphbasis = &thphbasis;
			angids = std::vector<int>(this->angqN());
			angidsReverse = std::vector<int>(this->angqN());
			for(int i = 0; i < this->angqN(); i++) {
				angids[i] = i;
				angidsReverse[i] = i;
			}
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
			
			return dpamatvecBlock(v);
			
			
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
			
			#pragma omp parallel
			{

				#pragma omp for
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
			
				#pragma omp for
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
			
			#pragma omp parallel for
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
			// cout << "Matvec<dpa> called successfully\n";
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
			
			/*for(int i = 0; i < Nth; i++) {
				cout << bdpthU[i] << std::endl;
				
				cout << bdpthL[i] << std::endl;
			
			}
			*/
			
			std::vector<cvec> vecparts(Nth);
			
			// cout << "Operating cache size: " << Nth << std::endl;
			
			#pragma omp parallel
				{
				this->rbasis->template precacheMatfree<bdpa>(v,Nth,Nl);
				
				
				#pragma omp for collapse(2)
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
							}
						}
					}
				}
				
				
				#pragma omp for
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
			
			#pragma omp parallel for collapse(2)
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
			
			#pragma omp parallel for
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
			#pragma omp parallel for collapse(2)
			
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
			
			#pragma omp parallel for
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
			return h0matvecBlock(v);
			cvec out(v.size());
			
			int Nr = this->radqN();
			int Nth = this->angqN();
			
			
			std::vector<cvec> outparts;
			for(int i = 0; i < Nth; i++) {
				outparts.push_back(cvec::Zero(Nr));
			}
			
			
			#pragma omp parallel 
			{
				#pragma omp for
				for(int i = 0; i < Nth; i++) {
					int k = angids[i];
					this->rbasis->template cacheMatfree<H0>(v.segment(Nr*i,Nr),outparts[i],i,1.0,k);
				}
				
				#pragma omp for
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
			
			for(int ul = -1; ul < 2; ul+=2) {
				csmat& dpam = thphbasis->dpam(ul);
					
				cmat outm = cmat::Zero(Nth,Nr);
				
				// cmat inm = dpam * v.reshaped(Nr,Nth)
				
				this->rbasis->template matmat<dpa>(inmat,outm,dpam,ilist,0,ul);
				// outmats[uli][l] = (bdpApxm * outmats[uli][l].transpose()).transpose();
				out.noalias() += outm;
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
		cvec matvec_impl(std::integral_constant<matname, M> c, const cvec& v) {
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
				
				cmat angcp = -cdouble(0,1) * cmat(angCouplings(bdp));
				
				// cout << angcp << "\n";
				
				int rn = this->radqN();
				int an = this->angqN();
				
				// cout << rn << "," << an << "\n";
				
				cmat eangc = expm(angcp,angcp.rows());
				
				// cout << eangc << "\n";
				
				cvec angslect = eangc * angStates;
				// cout << angStates << "\n";
				// cout << angslect << "\n";
				vector<int> slids(0);
				for(int i = 0; i < angslect.size(); i++) {
					if(angslect[i]!=cdouble(0.0,0.0)) slids.push_back(i);
				}
				
				
				int nslect = slids.size();
				
				/*for(int i = 0; i < nslect; i++) {
					cout << slids[i] << ",";
				}
				*/
				
				this->angovr(nslect);
				angids = slids;
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


						   


	
												  
	
	
	#pragma omp parallel for collapse(2)
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
	
	#pragma omp parallel for collapse(2)
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
	

	#pragma omp parallel for
	for(int i = 0; i < Nth; i++) {
					
		for(int j = 0; j < Nl; j++) {
			out.segment(Nr*i,Nr).noalias() += outparts[i][j];
		}
					 
	}
	
	
	return out;
}*/

#endif
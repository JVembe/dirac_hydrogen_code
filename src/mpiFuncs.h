/*
	This header contains template functions for dealing with Eigen matrices in MPI. Most are currently unused, and were implemented to learn MPI.
*/

#ifndef MPIFUNCS_H
#define MPIFUNCS_H

#include <mpi.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <iostream>
#include <vector>
#include <type_traits>

using namespace std;

//Found on github
template <typename T>
static inline MPI_Datatype mpi_get_type() noexcept {
  MPI_Datatype mpi_type = MPI_DATATYPE_NULL;
  if (std::is_same<T, char>::value) {
    mpi_type = MPI_CHAR;
  } else if (std::is_same<T, signed char>::value) {
    mpi_type = MPI_SIGNED_CHAR;
  } else if (std::is_same<T, unsigned char>::value) {
    mpi_type = MPI_UNSIGNED_CHAR;
  } else if (std::is_same<T, wchar_t>::value) {
    mpi_type = MPI_WCHAR;
  } else if (std::is_same<T, signed short>::value) {
    mpi_type = MPI_SHORT;
  } else if (std::is_same<T, unsigned short>::value) {
    mpi_type = MPI_UNSIGNED_SHORT;
  } else if (std::is_same<T, signed int>::value) {
    mpi_type = MPI_INT;
  } else if (std::is_same<T, unsigned int>::value) {
    mpi_type = MPI_UNSIGNED;
  } else if (std::is_same<T, signed long int>::value) {
     mpi_type = MPI_LONG;
  } else if (std::is_same<T, unsigned long int>::value) {
    mpi_type = MPI_UNSIGNED_LONG;
  } else if (std::is_same<T, signed long long int>::value) {
    mpi_type = MPI_LONG_LONG;
  } else if (std::is_same<T, unsigned long long int>::value) {
    mpi_type = MPI_UNSIGNED_LONG_LONG;
  } else if (std::is_same<T, float>::value) {
    mpi_type = MPI_FLOAT;
  } else if (std::is_same<T, double>::value) {
    mpi_type = MPI_DOUBLE;
  } else if (std::is_same<T, long double>::value) {
    mpi_type = MPI_LONG_DOUBLE;
  } else if (std::is_same<T, std::int8_t>::value) {
    mpi_type = MPI_INT8_T;
  } else if (std::is_same<T, std::int16_t>::value) {
    mpi_type = MPI_INT16_T;
  } else if (std::is_same<T, std::int32_t>::value) {
    mpi_type = MPI_INT32_T;
  } else if (std::is_same<T, std::int64_t>::value) {
    mpi_type = MPI_INT64_T;
  } else if (std::is_same<T, std::uint8_t>::value) {
    mpi_type = MPI_UINT8_T;
  } else if (std::is_same<T, std::uint16_t>::value) {
    mpi_type = MPI_UINT16_T;
  } else if (std::is_same<T, std::uint32_t>::value) {
    mpi_type = MPI_UINT32_T;
  } else if (std::is_same<T, std::uint64_t>::value) {
    mpi_type = MPI_UINT64_T;
  } else if (std::is_same<T, bool>::value) {
    mpi_type = MPI_C_BOOL;
  } else if (std::is_same<T, std::complex<float>>::value) {
    mpi_type = MPI_C_COMPLEX;
  } else if (std::is_same<T, std::complex<double>>::value) {
    mpi_type = MPI_C_DOUBLE_COMPLEX;
  } else if (std::is_same<T, std::complex<long double>>::value) {
    mpi_type = MPI_C_LONG_DOUBLE_COMPLEX;
  }
  return mpi_type;
}



template<typename Derived>
void sendMat(Eigen::MatrixBase<Derived>& mat,int target, int wsize){
	// cout << "Data to send: " << (void*)static_cast<Derived>(mat).data() << ", number of bytes: " << mat.size() * sizeof(typename Eigen::MatrixBase<Derived>::Scalar) << endl;
	
	// for(int i = 0; i < mat.size(); i++) {
		// cout << (static_cast<Derived>(mat).data())[i] << endl;
	// }
	int rows = mat.rows();
	int cols = mat.cols();
	MPI_Send(&rows,1,MPI_INT,target%wsize,0,MPI_COMM_WORLD);
	MPI_Send(&cols,1,MPI_INT,target%wsize,0,MPI_COMM_WORLD);
	MPI_Send((void*)(static_cast<Derived>(mat).data()),mat.size() * sizeof(typename Eigen::MatrixBase<Derived>::Scalar),MPI_BYTE,target%wsize,0,MPI_COMM_WORLD);
}

template<typename Derived>
void recMat(Eigen::MatrixBase<Derived>& mat, int wrank, int wsize, int inrank){
	int rows, cols;
	
	MPI_Recv(&rows,1,MPI_INT,inrank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	MPI_Recv(&cols,1,MPI_INT,inrank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	
	
	MPI_Status status;
	
	MPI_Probe(inrank,0,MPI_COMM_WORLD,&status);
	
	int matsize;
	MPI_Get_count(&status,MPI_BYTE,&matsize);
	int matN = matsize/sizeof(typename Eigen::MatrixBase<Derived>::Scalar);
	
	// cout << "Size of incoming matrix: " << matsize << ", number of matrix elements: " << matN << std::endl;
	
	
	
	typename Eigen::MatrixBase<Derived>::Scalar* indata = new typename Eigen::MatrixBase<Derived>::Scalar[matN];
	
	
	MPI_Recv((void*)indata,matsize, MPI_BYTE, inrank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	
	// for(int i = 0; i < 9; i++) {
		// cout << indata[i] << ",\n";
	// }
	
	mat = Eigen::Map<Derived>(indata,rows,cols);
	
	// cout << "Received matrix: " << mat << endl;
	// cout << "Mat dims: " << mat.rows() << ", " << mat.cols() << endl;
}

template <typename Derived>
void bcastMat(Eigen::MatrixBase<Derived>& mat) {
	using Scalar = typename Eigen::MatrixBase<Derived>::Scalar;
	int rows, cols;
	int matsize;
	typename Eigen::MatrixBase<Derived>::Scalar* mdata;
	int wrank;
	
	MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
	
	if(wrank == 0) {
		rows = mat.rows();
		cols = mat.cols();
		matsize = mat.size();
	}	

	MPI_Bcast(&rows,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&cols,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&matsize,1,MPI_INT,0,MPI_COMM_WORLD);
	
	if(wrank == 0) {
		mdata = &mat(0);
	} else {
		mdata = new Scalar[matsize];
	}
	
	MPI_Bcast((void*)mdata,matsize * sizeof(Scalar), MPI_BYTE, 0, MPI_COMM_WORLD);
	
	if(wrank != 0) {
		mat = Eigen::Map<Derived>(mdata,rows,cols);
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

template <typename Derived1, typename Derived2>
void scatterVec(const Eigen::MatrixBase<Derived1>& invec,Eigen::MatrixBase<Derived2>& outvecs, int blockSize) {
	//Preamble: Figure out vec segments
	using Scalar = typename Eigen::MatrixBase<Derived1>::Scalar;
	int wsize;
	
	int wrank;
	
	int nblocks;
	int blocksPer;
	
	const Scalar* vdata;
	Scalar* blkdata;
	
	MPI_Comm_size(MPI_COMM_WORLD, &wsize);
	MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
	
	
	if(wrank == 0)
		nblocks = invec.size()/blockSize;
	
	MPI_Bcast(&nblocks,1,MPI_INT,0,MPI_COMM_WORLD);
	
	blocksPer = nblocks/wsize;
	if(wrank == wsize-1) blocksPer += nblocks%wsize;
	
	
	
	if(wrank == 0) {
		vdata = &invec(0);
	}
	
	// MPI_Barrier(MPI_COMM_WORLD);
	
	blkdata = new Scalar[blocksPer*blockSize];
	
	
	MPI_Scatter((void*)vdata,blockSize*blocksPer*sizeof(Scalar),MPI_BYTE,(void*)blkdata,blockSize*blocksPer*sizeof(Scalar),MPI_BYTE,0,MPI_COMM_WORLD);
	
	// MPI_Barrier(MPI_COMM_WORLD);
	// if(isCached(outvecs)) {
		// ~outvecs();
	// }
	outvecs = Eigen::Map<Derived2>(blkdata,blockSize,blocksPer);
	
	// MPI_Barrier(MPI_COMM_WORLD);
	delete [] blkdata;
}

//If invec is known by all threads, this just copies the functionality of scatterVec without communication
template <typename Derived1, typename Derived2>
void pseudoscatterVec(const Eigen::MatrixBase<Derived1>& invec,Eigen::MatrixBase<Derived2>& outvecs, int blockSize) {
	//Preamble: Figure out vec segments
	using Scalar = typename Eigen::MatrixBase<Derived1>::Scalar;
	int wsize;
	
	int wrank;
	
	int nblocks;
	int blocksPer;
	
	MPI_Comm_size(MPI_COMM_WORLD, &wsize);
	MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
	
	
	nblocks = invec.size()/blockSize;
	
	blocksPer = nblocks/wsize;
	
	// cout << "invec size: " << invec.size() << endl;
	// cout << "blockSize: " << blockSize << endl;
	// cout << "blocksPer: " << blocksPer << endl;
	
	outvecs = invec.reshaped(blockSize,nblocks).middleCols(wrank*blocksPer,blocksPer);
}


template <typename Derived1, typename Derived2>
void gatherVec(Eigen::MatrixBase<Derived1>& invecs,Eigen::MatrixBase<Derived2>& outvec) {
	using Scalar = typename Eigen::MatrixBase<Derived1>::Scalar;
	
	int wsize;
	int wrank;
	
	Scalar* vdata;
	Scalar* blkdata;
	
	
	MPI_Comm_size(MPI_COMM_WORLD, &wsize);
	MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
	
	// cout << "invec at rank " << wrank << ":\n" << invecs << endl;
	
	blkdata = &invecs(0);
	// cout << "Memory address of invec data " << blkdata << endl;
	
	
	int blkRows = invecs.rows();
	int blkCols = invecs.cols();
	
	int totblks;
	
	
	// MPI_Barrier(MPI_COMM_WORLD);
	
	// cout << "BlkRows at rank " << wrank << ": " << blkRows << endl;
	// cout << "BlkCols at rank " << wrank << ": " << blkCols << endl;
	// MPI_Barrier(MPI_COMM_WORLD);
	MPI_Reduce(&blkCols,&totblks, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	// MPI_Barrier(MPI_COMM_WORLD);
	if(wrank == 0) {
		// cout << "Total number of blocks: " << totblks << std::endl;
		vdata = new Scalar[totblks * blkRows];
	}
	// MPI_Barrier(MPI_COMM_WORLD);
	
	//Check blocks
	// for(int i = 0; i < blkRows * blkCols; i++) {
		// // cout << blkdata[i] << "\n";
	// }
	
	// MPI_Barrier(MPI_COMM_WORLD);
	
	MPI_Gather((void*)blkdata,blkRows*blkCols*sizeof(Scalar),MPI_BYTE,(void*)vdata,blkRows*blkCols*sizeof(Scalar),MPI_BYTE,0,MPI_COMM_WORLD);
	// MPI_Barrier(MPI_COMM_WORLD);
	
	// delete [] blkdata;
	
	// if(wrank == 0) {
		// // cout << "vdata raw buffer" << endl;
		// for(int i = 0; i < totblks*blkRows; i++) {
			// cout << vdata[i] << "\n";
		// }
	// }
	// MPI_Barrier(MPI_COMM_WORLD);
	if(wrank == 0) {
		outvec = Eigen::Map<Derived2>(vdata,blkRows * totblks,1);
	}
}

template <typename Derived1, typename Derived2>
void allgatherVec(Eigen::MatrixBase<Derived1>& invecs,Eigen::MatrixBase<Derived2>& outvec) {
	using Scalar = typename Eigen::MatrixBase<Derived1>::Scalar;
	
	int wsize;
	int wrank;
	
	Scalar* vdata;
	Scalar* blkdata;
	
	
	MPI_Comm_size(MPI_COMM_WORLD, &wsize);
	MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
	
	// cout << "invec at rank " << wrank << ":\n" << invecs << endl;
	
	blkdata = &invecs(0);
	// cout << "Memory address of invec data " << blkdata << endl;
	
	
	int blkRows = invecs.rows();
	int blkCols = invecs.cols();
	
	
	int* recvcols = new int[wsize];
	int* recvn = new int[wsize];
	int* disps = new int[wsize];
	int totblks = 0;
	
	// cout << "BlkCols at rank " << wrank << ": " << blkCols << endl;
	// cout << "BlkRows at rank " << wrank << ": " << blkRows << endl;
	// MPI_Barrier(MPI_COMM_WORLD);
	// totblks = wsize * blkCols;
	MPI_Allgather(&blkCols,1, MPI_INT, recvcols, 1, MPI_INT, MPI_COMM_WORLD);
	// cout << "Receiving number of blocks: " << endl;
	
	for(int i = 0; i < wsize; i++) {
		// cout << recvcols[i] << std::endl;
		totblks += recvcols[i];
		recvn[i] = recvcols[i] * blkRows;
		disps[i] = 0;
		for(int j = 0; j < i; j++) {
			disps[i] += recvcols[j] * blkRows;
		}
		// cout << disps[i] << endl;
	};
	
	vdata = new Scalar[blkRows*totblks];
	
	// Check blocks
	// for(int i = 0; i < blkRows * blkCols; i++) {
		// cout << blkdata[i] << "\n";
	// }
	
	// MPI_Barrier(MPI_COMM_WORLD);
	
	MPI_Allgatherv(blkdata,blkRows*blkCols,mpi_get_type<Scalar>(),vdata,recvn,disps,mpi_get_type<Scalar>(),MPI_COMM_WORLD);
	// MPI_Barrier(MPI_COMM_WORLD);
	
	// delete [] blkdata;
	
	// cout << "vdata raw buffer" << endl;
	// for(int i = 0; i < totblks*blkRows; i++) {
		// cout << vdata[i] << "\n";
	// }
	// }
	// MPI_Barrier(MPI_COMM_WORLD);
	outvec = Eigen::Map<Derived2>(vdata,blkRows * totblks,1);
	delete[] vdata;
	delete[] recvcols;
	delete[] recvn;
}

template <typename Derived>
void reduceMat(Eigen::MatrixBase<Derived>& inmats, Eigen::MatrixBase<Derived>& outmat, int dest) {
	using Scalar = typename Eigen::MatrixBase<Derived>::Scalar;
	
	int wsize;
	int wrank;
	
	Scalar* indata;
	Scalar* outdata;
	
	
	MPI_Comm_size(MPI_COMM_WORLD, &wsize);
	MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
	
	indata = &inmats(0);
	// cout << "Data of local matrices" << endl;
	// for(int i = 0; i < inmats.size(); i++) {
		// cout << indata[i] << endl;
	// }
	
	if(wrank == dest) outdata = new Scalar[inmats.size()];
	
	MPI_Reduce(indata,outdata,inmats.size(),mpi_get_type<Scalar>(),MPI_SUM,dest,MPI_COMM_WORLD);
	
	// if(wrank == dest) {
		// cout << "Data of reduced matrix" << endl;
		// for(int i = 0; i < inmats.size(); i++) {
			// cout << outdata[i] << endl;
		// }
	// }
	if(wrank == dest) {
		outmat = Eigen::Map<Derived>(outdata,inmats.rows(),inmats.cols());
	}
}


template <typename Derived>
void allreduceMat(Eigen::MatrixBase<Derived>& inmats, Eigen::MatrixBase<Derived>& outmat) {
	using Scalar = typename Eigen::MatrixBase<Derived>::Scalar;
	
	Scalar* indata;
	Scalar* outdata;
	
	int Nrow,Ncol;
	
	indata = &inmats(0);
	
	// cout << "inmats:\n" << inmats;
	// cout << "Inmats size: " << inmats.size() << endl;
	outdata = new Scalar[inmats.size()];
	
	Nrow = inmats.rows();
	Ncol = inmats.cols();
	
	MPI_Allreduce(indata,outdata,inmats.size(),mpi_get_type<Scalar>(),MPI_SUM,MPI_COMM_WORLD);
	
	// cout << "Data of reduced matrix" << endl;
	// for(int i = 0; i < inmats.size(); i++) {
		// cout << outdata[i] << endl;
	// }
	//NOTE: Can't use inmats after this!!
	
	// if(isCached(outmat)) {
		// outmat = cmat::Zero(0,0);
	// }
	outmat = Eigen::Map<Derived>(outdata,Nrow,Ncol);
	delete [] outdata;
}


template <typename Derived1, typename Derived2>
void blockReduce(Eigen::MatrixBase<Derived1>& invecs,Eigen::MatrixBase<Derived2>& outvec) {
	using Scalar = typename Eigen::MatrixBase<Derived1>::Scalar;
	int wsize;
	
	int wrank;
	
	int nblocks;
	int blocksPer;
	
	Scalar* vdata;
	Scalar* blkdata;
	
	MPI_Comm_size(MPI_COMM_WORLD, &wsize);
	MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
	
	//Confirm all blocks are same size
	
	int blkRows,blkCols;
	
	blkRows = invecs.rows();
	blkCols = invecs.cols();
	
	// cout << "Rows at rank " <<wrank << ": " << blkRows << endl;
	
	int maxrows, minrows;
	
	MPI_Allreduce(&blkRows,&maxrows,1,MPI_INT,MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&blkRows,&minrows,1,MPI_INT,MPI_MIN, MPI_COMM_WORLD);
	
	if(wrank == 0) {
		// cout << "max number of rows: " << maxrows << endl;
		// cout << "min number of rows: " << minrows << endl;
		if(maxrows != minrows) cerr << "Attempt to reduce blocks of unequal rows, abort..." << endl;
	}
	
	
	// cout << "Cols at rank " <<wrank << ": " << blkCols << endl;
	
	int maxcols, mincols;
	
	MPI_Allreduce(&blkCols,&maxcols,1,MPI_INT,MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&blkCols,&mincols,1,MPI_INT,MPI_MIN, MPI_COMM_WORLD);
	
	if(wrank == 0) {
		// cout << "max number of cols: " << maxcols << endl;
		// cout << "min number of cols: " << mincols << endl;
		if(maxcols != mincols) cerr << "Attempt to reduce blocks of unequal cols, abort..." << endl;
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	// cout << "invec at wrank " << wrank << ": " << invecs << endl;
	
	int ncols;
	
	MPI_Reduce(&blkCols,&ncols,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
	// if(wrank == 0) cout << "Total number of cols: " << ncols << endl;
	
	blkdata = &invecs(0);
	// cout << "first element of blkdata: " << invecs(0);
	// cout << "blkdata pointer: " << blkdata << endl;
	// cout << "value at blkdata[0]: " <<blkdata[0] << endl;
	
	// cout << "blkdata at wrank"<<wrank<<":\n";
	
	for(int i = 0; i < maxcols * maxrows; i++) {
		//cout << blkdata[i] << std::endl;
	}
	MPI_Barrier(MPI_COMM_WORLD);
	
	if(wrank == 0) vdata = new Scalar[ncols * maxrows];
	
	MPI_Barrier(MPI_COMM_WORLD);
	
	MPI_Reduce((void*)blkdata,(void*)vdata,maxcols*maxrows,mpi_get_type<Scalar>(),MPI_SUM,0,MPI_COMM_WORLD);
	
	if(wrank == 0) {
		// cout << "Reduced vec: " << endl;
		for(int i = 0; i < maxcols * maxrows; i++) {
			// cout << vdata[i] << endl;
		}
		outvec = Eigen::Map<Derived2>(vdata,maxrows,maxcols).rowwise().sum();
		// cout << "Reduced vec:\n" << outvec << endl;
	}
}

template<typename Scalar, int Options, typename StorageIndex>
void sendSparseMat(Eigen::SparseMatrix<Scalar,Options,StorageIndex>& mat,int target, int wsize){
	// cout << "Data to send: " << (void*)static_cast<Derived>(mat).data() << ", number of bytes: " << mat.size() * sizeof(typename Eigen::MatrixBase<Derived>::Scalar) << endl;
	
	// for(int i = 0; i < mat.size(); i++) {
		// cout << (static_cast<Derived>(mat).data())[i] << endl;
	// }
	
	int wrank;
	
	MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
	
	int rows = mat.rows();
	int cols = mat.cols();
	
	Scalar* values = mat.valuePtr();
	StorageIndex* inners = mat.innerIndexPtr();
	StorageIndex* outers = mat.outerIndexPtr();
	
	// cout << "values" << endl;
	// for(int i = 0; i < mat.nonZeros(); i++) {
		// cout << values[i] << endl;
	// }
	// cout << "inner indices" <<endl;
	// for(int i = 0; i < mat.nonZeros(); i++) {
		// cout << inners[i] << endl;
	// }
	// cout << "outer indices" <<endl;
	// for(int i = 0; i < mat.outerSize() + 1; i++) {
		// cout << outers[i] << endl;
	// }

	
	MPI_Send(&rows,1,MPI_INT,target,0,MPI_COMM_WORLD);
	MPI_Send(&cols,1,MPI_INT,target,0,MPI_COMM_WORLD);
	MPI_Send(values,mat.nonZeros(), mpi_get_type<Scalar>(),target,0,MPI_COMM_WORLD);
	MPI_Send(inners,mat.nonZeros(), mpi_get_type<StorageIndex>(),target,0,MPI_COMM_WORLD);
	MPI_Send(outers,mat.outerSize()+1,mpi_get_type<StorageIndex>(),target,0,MPI_COMM_WORLD);
}

template<typename Scalar, int Options, typename StorageIndex>
void recvSparseMat(Eigen::SparseMatrix<Scalar,Options,StorageIndex>& mat,int origin, int wsize){
	using sparseMat = Eigen::SparseMatrix<Scalar,Options,StorageIndex>;
	int wrank;
	
	MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
	
	int rows;
	int cols;
	
	Scalar* values;
	StorageIndex* inners;
    StorageIndex* outers;
	
	int nonzeros;
	int outersize;
	
	MPI_Recv(&rows,1,MPI_INT,origin,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	MPI_Recv(&cols,1,MPI_INT,origin,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	
	// cout << "rows of incoming matrix: " << rows << endl;
	// cout << "cols of incoming matrix: " << cols << endl;
	
	MPI_Status status;
	
	MPI_Probe(origin,0,MPI_COMM_WORLD,&status);
	
	MPI_Get_count(&status,mpi_get_type<Scalar>(),&nonzeros);
	
	// cout << "Incoming nonzero elements: " << nonzeros << endl;
	
	values = new Scalar[nonzeros];
	inners = new StorageIndex[nonzeros];
	
	MPI_Recv(values,nonzeros,mpi_get_type<Scalar>(),origin,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	MPI_Recv(inners,nonzeros,mpi_get_type<StorageIndex>(),origin,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	
	// cout << "Received values and inner indices" << endl;
	
	// cout << "values" << endl;
	// for(int i = 0; i < nonzeros; i++) {
		// cout << values[i] << endl;
	// }
	// cout << "inner indices" <<endl;
	// for(int i = 0; i < nonzeros; i++) {
		// cout << inners[i] << endl;
	// }
	
	// cout << "Probing outer size" <<endl;
	
	MPI_Status status2;
	
	MPI_Probe(origin,0,MPI_COMM_WORLD,&status2);
	
	
	MPI_Get_count(&status2,mpi_get_type<StorageIndex>(),&outersize);
	
	// cout << "Incoming outer indices: "  << outersize << endl;
	
	outers = new StorageIndex[outersize];
	
	MPI_Recv(outers,outersize,mpi_get_type<Scalar>(),origin,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	
	// cout << "outer indices" <<endl;
	for(int i = 0; i < outersize; i++) {
		//cout << outers[i] << endl;
	}
	
	mat = Eigen::Map<sparseMat>(rows,cols,nonzeros,outers,inners,values);
	
	// cout << "nonzero entries in mapped matrix: " << mat.nonZeros() << endl;
	//cout << mat << endl;
}


template<typename Scalar, int Options, typename StorageIndex>
void bcastSparseMat(Eigen::SparseMatrix<Scalar,Options,StorageIndex>& mat,int origin, int wsize){
	using sparseMat = Eigen::SparseMatrix<Scalar,Options,StorageIndex>;
	int wrank;
	
	MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
	
	int rows;
	int cols;
	
	Scalar* values;
	StorageIndex* inners;
    StorageIndex* outers;
	
	int nonzeros;
	int outersize;
	
	if(wrank == origin) {		
		rows = mat.rows();
		cols = mat.cols();
	}
	
	MPI_Bcast(&rows,1,MPI_INT,origin,MPI_COMM_WORLD);
	MPI_Bcast(&cols,1,MPI_INT,origin,MPI_COMM_WORLD);
	
	// cout << "rows of incoming matrix: " << rows << endl;
	// cout << "cols of incoming matrix: " << cols << endl;
	
	if(wrank == origin) {
		nonzeros = mat.nonZeros();
		outersize = mat.outerSize();
	}
	
	MPI_Bcast(&nonzeros,1,MPI_INT,origin,MPI_COMM_WORLD);
	MPI_Bcast(&outersize,1,MPI_INT,origin,MPI_COMM_WORLD);
	
	// cout << "Incoming nonzero elements: " << nonzeros << endl;
	
	if(wrank == origin) {
		values = mat.valuePtr();
		inners = mat.innerIndexPtr();
	} else {
		// cout << "Attempting to allocate arrays of size " << nonzeros << endl;
		values = new Scalar[nonzeros];
		inners = new StorageIndex[nonzeros];
		// cout << "Allocation successful" << endl;
	}
	
	// cout << "Attempting to broadcast arrays" << endl;
	
	MPI_Bcast(values,nonzeros,mpi_get_type<Scalar>(),origin,MPI_COMM_WORLD);
	MPI_Bcast(inners,nonzeros,mpi_get_type<StorageIndex>(),origin,MPI_COMM_WORLD);
	
	// cout << "Successfully broadcast arrays" << endl;
	
	if(wrank == origin) {
		outers = mat.outerIndexPtr();
	} else {
		// cout << "Attempting to allocate array of size " << outersize << endl;	
		outers = new StorageIndex[outersize];
		// cout << "Allocation successful" << endl;
	}
	
	MPI_Bcast(outers,outersize + 1,mpi_get_type<Scalar>(),origin,MPI_COMM_WORLD);

	// cout << "Successfully broadcast array" << endl;
	
	// cout << "values" << endl;
	// for(int i = 0; i < nonzeros; i++) {
		// cout << values[i] << endl;
	// }
	
	// cout << "inner indices" <<endl;
	// for(int i = 0; i < nonzeros; i++) {
		// cout << inners[i] << endl;
	// }
	
	// cout << "outer indices" <<endl;
	// for(int i = 0; i < outersize+1; i++) {
		// cout << outers[i] << endl;
	// }
	
	if(wrank != origin) {
		// cout << "Constructing out mat" << endl;
		// cout << "out mat constructed" << endl;
		
		// cout << outmat << endl;
		mat = Eigen::Map<sparseMat>(rows,cols,nonzeros,outers,inners,values);
		// cout << "out mat copied" << endl;
	}
	
	// cout << "nonzero entries in mapped matrix: " << mat.nonZeros() << endl;
	// cout << mat << endl;
}

#endif


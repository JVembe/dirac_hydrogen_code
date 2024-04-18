OBJS=sorted_list.o \
	csr.o \
	gpu_sparse.o \
	solver.o \
	bicgstab.o \
	solver_common.o \
	gpu_bicgstab.o \
	utils.o \
	superlu.o

CUOBJS = cuda_solver.o

CFLAGS_CUDA = -DUSE_CUDA
CFLAGS_HIP = -march=native -I hipsparse -std=c99 -D__HIP_PLATFORM_AMD__ -DUSE_HIP -lhipsparse
CFLAGS = -DDEBUG0 -g -lm -DUSE_MPI -I/home/angainor/work/institution/usit/aus/dirac_gpu/install/include -I/cluster/home/marcink/software/superlu/include $(CFLAGS_CUDA) -fopenmp

LDFLAGS_CUDA = -lcublas -lcusparse -lcudart
LDFLAGS = $(LDFLAGS_CUDA) -L/home/angainor/work/institution/usit/aus/dirac_gpu/install/lib -L/cluster/home/marcink/software/superlu/lib64 -lsuperlu -lopenblas -fopenmp

CC?=gcc
CC_CUDA = nvcc
CC_HIP = hipcc

%.o : %.c
	$(CC) -O3 -march=native -c $(CFLAGS) $< -o $@

%.o : %.cu
	$(CC_CUDA) -Xptxas -O3,-v -arch=sm_80 -gencode=arch=compute_80,code=sm_80 -g -O3 -c $< -o $@

matrix_tool: $(OBJS) $(CUOBJS) matrix_tool.o
	$(CC) -O3 $(OBJS) $(CUOBJS) $@.o -o matrix_tool -lm $(LDFLAGS)

dirac_solver: $(OBJS) $(CUOBJS) dirac_solver.o
	$(CC) -O3 $(OBJS) $(CUOBJS) $@.o -o dirac_solver -lm $(LDFLAGS)

clean:
	rm *.o

cuda:
	nvcc -g -O3 $(OBJS) -o matrix_tool -lm -DUSE_CUDA -lcusparse


#matrix_tool_cuda.o: matrix_tool.c
#	$(CC_CUDA) $(CFLAGS_COMMON) $(CFLAGS_CUDA) matrix_tool.c -o matrix_tool_cuda

#matrix_tool_hip.o: matrix_tool.c
#	$(CC_HIP) $(CFLAGS_COMMON) $(CFLAGS_HIP) matrix_tool.c -o matrix_tool_hip

OBJS=sorted_list.o \
	csr.o \
	gpu_sparse.o \
	solver.o \
	bicgstab.o \
	solver_common.o \
	gpu_bicgstab.o \
	utils.o \
	superlu.o

CUOBJS0 = cuda_solver.o
CFLAGS_CUDA =

ROCM_HOME=/opt/rocm-5.4.2
CFLAGS_HIP = -march=native -I hipsparse -std=c99 -D__HIP_PLATFORM_AMD__ -DUSE_HIP -I$(ROCM_HOME)/include -DROCM_MATHLIBS_API_USE_HIP_COMPLEX

SLU_HOME = /itf-fi-ml/home/marcink/software/superlu
CFLAGS = -DDEBUG0 -g -lm -DUSE_MPI -I$(SLU_HOME)/include $(CFLAGS_HIP) -fopenmp

LDFLAGS_CUDA0 = -lcublas -lcusparse -lcudart
LDFLAGS_HIP = -L$(ROCM_HOME)/lib -lhipsparse -lhipblas -lamdhip64
LDFLAGS_ACCEL = $(LDFLAGS_HIP) $(LDFLAGS_CUDA)
LDFLAGS = $(LDFLAGS_ACCEL) -L$(SLU_HOME)/lib64 -lsuperlu -lopenblas -fopenmp

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

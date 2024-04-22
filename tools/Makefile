OBJS=sorted_list.o \
	csr.o \
	solver.o \
	bicgstab.o \
	solver_common.o \
	utils.o \
	superlu.o

SLU_HOME = $(HOME)/software/superlu

ifdef USE_CUDA
CUOBJS = gpu_solver.o gpu_sparse.o gpu_bicgstab.o
CFLAGS_CUDA = -DUSE_CUDA
LDFLAGS_CUDA = -lcublas -lcusparse -lcudart
CC_DEVICE = nvcc
CFLAGS_DEVICE = -Xptxas -O3,-v -arch=sm_80 -gencode=arch=compute_80,code=sm_80
endif

ifdef USE_HIP
ROCM_HOME=/opt/rocm-5.4.2
HIPOBJS = gpu_solver.o gpu_sparse.o gpu_bicgstab.o
CFLAGS_HIP = -march=native -I hipsparse -std=c99 -D__HIP_PLATFORM_AMD__ -DUSE_HIP -I$(ROCM_HOME)/include -DROCM_MATHLIBS_API_USE_HIP_COMPLEX
LDFLAGS_HIP = -L$(ROCM_HOME)/lib -lhipsparse -lhipblas -lamdhip64
CC_DEVICE = hipcc
endif

CFLAGS_ACCEL = $(CFLAGS_HIP) $(CFLAGS_CUDA)
CFLAGS = -DDEBUG0 -g -lm -DUSE_MPI -I$(SLU_HOME)/include $(CFLAGS_ACCEL) -fopenmp

LDFLAGS_ACCEL = $(LDFLAGS_HIP) $(LDFLAGS_CUDA)
LDFLAGS = $(LDFLAGS_ACCEL) -L$(SLU_HOME)/lib64 -lsuperlu -lopenblas -fopenmp

ACCELOBJS = $(CUOBJS) $(HIPOBJS)

CC?=gcc

%.o : %.c
	$(CC) -O3 -march=native -c $(CFLAGS) $< -o $@

%.o : %.cu
	$(CC_DEVICE) $(CFLAGS_DEVICE) -g -O3 -c $< -o $@

matrix_tool: $(OBJS) $(ACCELOBJS) matrix_tool.o
	$(CC) -O3 $(OBJS) $(ACCELOBJS) $@.o -o matrix_tool -lm $(LDFLAGS)

dirac_solver: $(OBJS) $(ACCELOBJS) dirac_solver.o
	$(CC) -O3 $(OBJS) $(ACCELOBJS) $@.o -o dirac_solver -lm $(LDFLAGS)

clean:
	rm *.o

cuda:
	nvcc -g -O3 $(OBJS) -o matrix_tool -lm -DUSE_CUDA -lcusparse


#matrix_tool_cuda.o: matrix_tool.c
#	$(CC_CUDA) $(CFLAGS_COMMON) $(CFLAGS_CUDA) matrix_tool.c -o matrix_tool_cuda

#matrix_tool_hip.o: matrix_tool.c
#	$(CC_HIP) $(CFLAGS_COMMON) $(CFLAGS_HIP) matrix_tool.c -o matrix_tool_hip

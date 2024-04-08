OBJS=sorted_list.o \
	csr.o \
	gpu_sparse.o \
	solver.o \
	bicgstab.o \
	gpu_bicgstab.o \
	utils.o \
	superlu.o

CFLAGS_CUDA = -DUSE_CUDA
CFLAGS_HIP = -march=native -I hipsparse -std=c99 -D__HIP_PLATFORM_AMD__ -DUSE_HIP -lhipsparse
CFLAGS = -DDEBUG -g -lm -DUSE_MPI -I/home/angainor/work/institution/usit/aus/dirac_gpu/install/include -I/cluster/home/marcink/software/superlu/include $(CFLAGS_CUDA)

LDFLAGS_CUDA = -lcublas -lcusparse -lcudart
LDFLAGS = $(LDFLAGS_CUDA) -L/home/angainor/work/institution/usit/aus/dirac_gpu/install/lib -L/cluster/home/marcink/software/superlu/lib64 -lsuperlu -lopenblas

CC?=gcc
CC_CUDA = nvcc
CC_HIP = hipcc

%.o : %.c
	$(CC) -O3 -march=native -c $(CFLAGS) $< -o $@

matrix_tool: $(OBJS) matrix_tool.o
	$(CC) -O3 $(OBJS) $@.o -o matrix_tool -lm $(LDFLAGS)

dirac_solver: $(OBJS) dirac_solver.o
	$(CC) -O3 $(OBJS) $@.o -o dirac_solver -lm $(LDFLAGS)

clean:
	rm *.o

cuda:
	nvcc -O3 $(OBJS) -o matrix_tool -lm -DUSE_CUDA -lcusparse


#matrix_tool_cuda.o: matrix_tool.c
#	$(CC_CUDA) $(CFLAGS_COMMON) $(CFLAGS_CUDA) matrix_tool.c -o matrix_tool_cuda

#matrix_tool_hip.o: matrix_tool.c
#	$(CC_HIP) $(CFLAGS_COMMON) $(CFLAGS_HIP) matrix_tool.c -o matrix_tool_hip

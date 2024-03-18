OBJS=sorted_list.o \
	csr.o \
	gpu_sparse.o \
	matrix_tool.o \
	bicgstab.o \
	superlu.o

CFLAGS_CUDA = #-DUSE_CUDA -lcusparse
CFLAGS_HIP = -march=native -I hipsparse -std=c99 -D__HIP_PLATFORM_AMD__ -DUSE_HIP -lhipsparse
CFLAGS = -DDEBUG -g -lm -DUSE_MPI -I/home/angainor/work/institution/usit/aus/dirac_gpu/install/include

LDFLAGS_CUDA = #-lcusparse -lcudart
LDFLAGS = $(LDFLAGS_CUDA) -L/home/angainor/work/institution/usit/aus/dirac_gpu/install/lib -lsuperlu -lblas

CC?=gcc
CC_CUDA = nvcc
CC_HIP = hipcc

%.o : %.c
	$(CC) -O3 -c $(CFLAGS) $< -o $@

all: $(OBJS)
	$(CC) -O3 $(OBJS) -o matrix_tool -lm $(LDFLAGS)

clean:
	rm *.o

cuda:
	nvcc -O3 $(OBJS) -o matrix_tool -lm -DUSE_CUDA -lcusparse


#matrix_tool_cuda.o: matrix_tool.c
#	$(CC_CUDA) $(CFLAGS_COMMON) $(CFLAGS_CUDA) matrix_tool.c -o matrix_tool_cuda

#matrix_tool_hip.o: matrix_tool.c
#	$(CC_HIP) $(CFLAGS_COMMON) $(CFLAGS_HIP) matrix_tool.c -o matrix_tool_hip

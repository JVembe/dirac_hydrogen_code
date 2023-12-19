OBJS=sorted_list.o \
	csr.o \
	matrix_tool.o

CFLAGS = -O3 -lm -DUSE_MPI
CFLAGS_CUDA = -DUSE_CUDA -lcusparse
CFLAGS_HIP = -march=native -I hipsparse -std=c99 -D__HIP_PLATFORM_AMD__ -DUSE_HIP -lhipsparse

CC=/home/angainor/work/institution/usit/prace_6ip/software/openmpi/bin/mpicc
CC_CUDA = nvcc
CC_HIP = hipcc

%.o : %.c
	$(CC) -O3 -c $(CFLAGS) $< -o $@

all: $(OBJS)
	$(CC) -O3 $(OBJS) -o matrix_tool -lm

clean:
	rm *.o

cuda:
	nvcc -O3 matrix_tool.c -o matrix_tool -lm -DUSE_CUDA -lcusparse


#matrix_tool_cuda.o: matrix_tool.c
#	$(CC_CUDA) $(CFLAGS_COMMON) $(CFLAGS_CUDA) matrix_tool.c -o matrix_tool_cuda

#matrix_tool_hip.o: matrix_tool.c
#	$(CC_HIP) $(CFLAGS_COMMON) $(CFLAGS_HIP) matrix_tool.c -o matrix_tool_hip

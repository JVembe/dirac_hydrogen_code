OBJS=sorted_list.o \
	csr.o \
	matrix_tool.o

CFLAGS=-DUSE_MPI

CC=/home/angainor/work/institution/usit/prace_6ip/software/openmpi/bin/mpicc

%.o : %.c
	$(CC) -O3 -c $(CFLAGS) $< -o $@

all: $(OBJS)
	$(CC) -O3 $(OBJS) -o matrix_tool -lm

clean:
	rm *.o

cuda:
	nvcc -O3 matrix_tool.c -o matrix_tool -lm -DUSE_CUDA -lcusparse

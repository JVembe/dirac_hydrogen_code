CC_CUDA = nvcc
CC_HIP = hipcc

CFLAGS_COMMON = -O3 -lm
CFLAGS_CUDA = -DUSE_CUDA -lcusparse
CFLAGS_HIP = -march=native -I hipsparse -std=c99 -D__HIP_PLATFORM_AMD__ -DUSE_HIP -lhipsparse

matrix_tool_cuda.o: matrix_tool.c
	$(CC_CUDA) $(CFLAGS_COMMON) $(CFLAGS_CUDA) matrix_tool.c -o matrix_tool_cuda

matrix_tool_hip.o: matrix_tool.c
	$(CC_HIP) $(CFLAGS_COMMON) $(CFLAGS_HIP) matrix_tool.c -o matrix_tool_hip

cuda: matrix_tool_cuda.o

hip: matrix_tool_hip.o

clean:
	rm -f matrix_tool_cuda matrix_tool_hip

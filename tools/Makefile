# GaDE - Gpu-accelerated solver for the time dependent Dirac Equation

# Copyright (C) 2025  Johanne Elise Vembe <johannevembe@gmail.com>
# Copyright (C) 2025  Marcin Krotkiewski <marcink@uio.no>
# Copyright (C) 2025  Hicham Agueny <hicham.agueny@uib.no>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


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
CFLAGS_DEVICE = -Xptxas -O3,-v -arch=sm_80 -gencode=arch=compute_80,code=sm_80 -DGPU_THREADS_PER_BLOCK=64 -allow-unsupported-compiler
endif

ifdef USE_HIP
ROCM_HOME=/opt/rocm-6.0.3
HIPOBJS = gpu_solver.o gpu_sparse.o gpu_bicgstab.o
CFLAGS_HIP = -march=native -I hipsparse -std=c99 -D__HIP_PLATFORM_AMD__ -DUSE_HIP -I$(ROCM_HOME)/include -DROCM_MATHLIBS_API_USE_HIP_COMPLEX
LDFLAGS_HIP = -L$(ROCM_HOME)/lib -lhipsparse -lhipblas -lamdhip64
CC_DEVICE = hipcc
CFLAGS_DEVICE = -DGPU_THREADS_PER_BLOCK=512 --offload-arch=gfx90a
endif

CFLAGS_ACCEL = $(CFLAGS_HIP) $(CFLAGS_CUDA)
ifndef SCALING_MODE
CFLAGS = -DDEBUG0 -g -lm -DUSE_MPI -I$(SLU_HOME)/include $(CFLAGS_ACCEL) -fopenmp
else
$(info Compiling in scaling analysis mod)
CFLAGS = -DDEBUG0 -g -lm -DUSE_MPI -I$(SLU_HOME)/include $(CFLAGS_ACCEL) -fopenmp -DSCALING_MODE
endif

ifdef SCALING_MODE
CFLAGS = $(CFLAGS) -DSCALING_MODE
endif

LDFLAGS_ACCEL = $(LDFLAGS_HIP) $(LDFLAGS_CUDA)
LDFLAGS = $(LDFLAGS_ACCEL) -L$(SLU_HOME)/lib64 -lsuperlu  -fopenmp -L$(HOME)/software/OpenBLAS/lib -lopenblas
#LDFLAGS = $(LDFLAGS_ACCEL) -L$(SLU_HOME)/lib64 -lsuperlu  -fopenmp /usr/lib64/libblas.so.3

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

# dirac_hydrogen_code
Parallelized simulation code to model the interaction between a hydrogen atom and a high intensity laser pulse at relativistic energies.

## Overview
The code is divided into two parts: A matrix construction/data processing module, and the main simulation code. The first is contained in the "src" folder, while the second is in the "tools" folder.

## Build instructions for the matrix construction module using CMake
### On SAGA
```
$ ml CMake/3.24.3-GCCcore-12.2.0
$ ml OpenMPI/4.1.4-GCC-12.2.0
$ ml Eigen/3.4.0-GCCcore-12.2.0
$ ml Boost/1.81.0-GCC-12.2.0
```
### Run CMake to Generate Build Files:
`cd dirac_hydrogen_code`

`cmake -S . -B build`

The `-S `flag specifies the source directory, and the `-B` flag specifies the directory where the build files will be generated.
To specify a particular build type (e.g., Debug or Release), you can also add the -DCMAKE_BUILD_TYPE flag:
`cmake -S path-to-src -B path-to-build -DCMAKE_BUILD_TYPE=Release`

### Compile the construction/processing module:
`cd build && make`
This generates several files: dumpCouplings, dumpEigvals, processOutput, spinProjection, nondipoleMPItest and dipoleMPItest. Alternatively,  run
`cd build && make [executable name here]`
to compile only one of the executables.

The executables nondipoleMPItest and dipoleMPItest are for small-scale runs of a less optimized version of the simulation code. dumpCouplings generates the matrices used by the full simulation, and must be run first. dumpEigvals, processOutput and spinProjection are for analysing the output once a simulation run has been completed.

### Run the small-scale simulation executables
The small-scale executables should be run with mpirun, or on the cluster using srun. Dipole and nondipole jobs need different parameters in the json input file, as the change in exclusion rules requires different basis functions, this is explained further below.

An example run with four tasks may be launched in mpirun as
`mpirun -np 4 xterm -e build/src/nondipoleMPItest input_filename.json`

### Running the construction executables
dumpCouplings is not particularly demanding while the simulation is carried out on a reasonably sized grid, and is not parallelized. As such, it may be run with
`./build/src/dumpCouplings input_filename.json´

dumpEigvals is significantly more demanding as it requires the solution of multiple generalized eigenvalue problems, but in return it is parallelized with no communication costs. It may be launched in mpirun as
`mpirun -np 4 xterm -e build/src/dumpEigvals input_filename.json`
NB/TODO: Note that the current implementation of parallelization for dumpEigvals does not account for uneven distribution of workloads between processes, and so the number of processes must be a factor of the parameter nKappa.

Output from dumpCouplings and dumpEigvals may be reused for multiple simulation runs as long as some parameters remain unchanged, detailed below.

### Choosing parameters

The parameters Intensity, Omega and Cycles determine the physical properties of the incident laser pulse, and so are determined by the physical problem of interest.

For testing purposes a single cycle at Omega=50 with Intensity=10 should be an easy problem to work with.

Z is the charge of the nucleus. Care should be taken when using higher values as the Coulomb potential may be inappropriate for heavy nuclei.

Similarly, rBox sets the physical size of the simulation domain. For this we use 30.

NSplines determines the radial resolution, this typically needs to be a minumum of 200 to produce results that make sense, but can be set lower for tests. 

Ntime determines the time resolution. For a 15-cycle pulse it has typically been necessary to do 8000-16000 time steps here, but as a single-cycle pulse is much shorter it's reasonable to do 500-1000 for tests, possibly less.

NKappa determines the maximum angular momentum quantum number, effectively acting as the angular resolution. 

It can be as low as 4, but for serious simulations it needs to be at least 8 and ideally 20 or more. Note that the wavefunction coefficient vector grows _quadratically_ with NKappa unless NMu is set, which impacts simulation time.

NMu limits the values of the magnetic quantum number, acting as the second angular resolution parameter. MUST be less than NKappa, if 0 then the magnetic quantum number will range from -|κ| to |κ|. Typically never needs to be larger than 10. This parameter is irrelevant in the dipole code, as the dipole approximation used here does not allow changes in magnetic quantum number.

Nl determines the number of expansion terms in the nondipole potential. If 0, the maximum permissible value for the given NKappa is used instead. With a simulation domain (rBox) of 30 atomic units, Nl never needs to be greater than 10. Note also that for the dipole approximation this must be set to 1 for the load balancing algorithm to operate properly.

## Compiling the main simulation module

The makefile for the full GPU simulations is located in the `tools` directory, and compilation of the main dirac_solver executable is done with

`make CC=compiler_name [options] dirac_solver`

The primary options of interest here are `USE_MPI=1`, `USE_CUDA=1` and `USE_HIP=1`, platform-dependent choices for the code to run on CPUs, NVIDIA GPUs or AMD GPUs, respectively. Dependencies are determined by these choices, but in all cases the following are required:
- `superLU`
- `openBLAS`

For compilation with cuda, the following standard CUDA libraries are required:
- `cusparse`
- `cublas`
- `cudart`

Meanwhile, for HIP the following are required. They are available in the ROCm software stack:
- `hipsparse`
- `hipblas`
- `amdhip64`

Care should be taken to ensure that compilation is done in a programming environment which implements GPU-aware MPI if available.

###Running the simulations

Move the files generated by `dumpCouplings` to the location of `dirac_solver`, then to prepare partitioning run
`python3 tools/main.py n`
where `n` is the number of GPUs you intend to run the simulation with. Sample job scripts for LUMI and Betzy are provided in the "job_scripts" folder.

###Simulation output and post-processing


### When to generate new matrices


### Cite this code
This software may be cited as
@software{GaDE,
  title = {GaDE: GPU-acceleration of time-dependent Dirac Equation for exascale},
  author = {....},
  url = {https://github.....},
  license = {GPL-v3},
  month = Jan,
  year = {2025}
} 

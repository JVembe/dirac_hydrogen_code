# dirac_hydrogen_code
Parallelized simulation code to model the interaction between a hydrogen atom and a high intensity laser pulse at relativistic energies.

## Build instructions using CMake
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

### Compile the Project:
`cd build && make`
This generates two files: dipoleMPItest and nondipoleMPItest. Alternatively, run
`cd build && make nondipoleMPItest`
or
`cd build && make dipoleMPItest`
To compile only one of the executables

### Run the Executable
The executables should be run with mpirun, or on the cluster using srun. Dipole and nondipole jobs need different parameters in the json input file, as the change in exclusion rules requires different basis functions, this is explained further below.

An example run with four tasks may be launched in mpirun as
`mpirun -np 4 xterm -e build/src/nondipoleMPItest input_filename.json`

### Choosing parameters

The parameters Intensity, Omega and Cycles determine the physical properties of the incident laser pulse, and so are determined by the physical problem at hand.
For testing purposes a single cycle at Omega=50 with Intensity=10 should be an easy problem to work with.
Similarly, rBox sets the physical size of the simulation domain. For this we use 30.
NSplines determines the radial resolution, this typically needs to be a minumum of 200 to produce results that make sense, but can be set lower for tests. 
Ntime determines the time resolution. For a 15-cycle pulse it has typically been necessary to do 8000-16000 time steps here, but as a single-cycle pulse is much shorter it's reasonable to do 500-1000 for tests, possibly less.
NKappa determines the maximum angular momentum quantum number, effectively acting as the angular resolution. 
It can be as low as 4, but for serious simulations it needs to be at least 8 and ideally 20 or more. Note that the wavefunction coefficient vector grows _quadratically_ with NKappa unless NMu is set, which impacts simulation time.
NMu limits the values of the magnetic quantum number, acting as the second angular resolution parameter. MUST be less than NKappa, if 0 then the magnetic quantum number will range from -|κ| to |κ|. Typically never needs to be larger than 10. This parameter is irrelevant in the dipole code, as the dipole approximation used here does not allow changes in magnetic quantum number.
Nl determines the number of expansion terms in the nondipole potential. If 0, the maximum permissible value for the given NKappa is used instead. With a simulation domain (rBox) of 30 atomic units, Nl never needs to be greater than 10. Note also that for the dipole approximation this must be set to 1 for the load balancing algorithm to operate properly.

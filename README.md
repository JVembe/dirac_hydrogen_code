# dirac_hydrogen_code
Parallelized simulation code to model the interaction between a hydrogen atom and a high intensity laser pulse at relativistic energies.

Currently, MPI implementation is only performed for the matrix-vector products S\*v H0\*v and the dipole version of HI\*v in the Dirac equation. I am working on the nondipole implementation.

Dependencies:

Eigen3

https://github.com/joeydumont/wignerSymbols


Compilation:

make mpitest

There are other options in the makefile, but currently none of them have been updated to use the MPI parallelized code

## Build instructions using CMake

### Run CMake to Generate Build Files:
`cmake -S path-to-src -B path-to-build`

The `-S `flag specifies the source directory, and the `-B` flag specifies the directory where the build files will be generated.
To specify a particular build type (e.g., Debug or Release), you can also add the -DCMAKE_BUILD_TYPE flag:
`cmake -S path-to-src -B path-to-build -DCMAKE_BUILD_TYPE=Release`

### Compile the Project:
cd path-to-build
make

### Run the Executable
cd src
./dipoleMPItest

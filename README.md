# dirac_hydrogen_code
Parallelized simulation code to model the interaction between a hydrogen atom and a high intensity laser pulse at relativistic energies.

## Build instructions using CMake

### Run CMake to Generate Build Files:
`cmake -S . -B path-to-build`

The `-S `flag specifies the source directory, and the `-B` flag specifies the directory where the build files will be generated.
To specify a particular build type (e.g., Debug or Release), you can also add the -DCMAKE_BUILD_TYPE flag:
`cmake -S path-to-src -B path-to-build -DCMAKE_BUILD_TYPE=Release`

### Compile the Project:
cd path-to-build
make

### Run the Executable
cd src
./dipoleMPItest

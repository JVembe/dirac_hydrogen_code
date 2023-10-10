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
`cd build && make dipoleMPItest`

### Run the Executable
`build/src/dipoleMPItest input_example.json`

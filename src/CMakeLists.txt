# Define the source files
set(SOURCES_DIPOLE
  dipoleMPItest.cpp
  dkbbasis.cpp
  splinehandler.cpp
  spnrbasis.cpp)

set(SOURCES_NONDIPOLE
  nondipoleMPItest.cpp
  dkbbasis.cpp
  splinehandler.cpp
  spnrbasis.cpp)

set(SOURCES_COUNTERPULSE
  counterpulseMPItest.cpp
  dkbbasis.cpp
  splinehandler.cpp
  spnrbasis.cpp)

set(SOURCES_DUMPLINGS
  dumpBdpCouplings.cpp
  dkbbasis.cpp
  splinehandler.cpp
  spnrbasis.cpp)


# Add the executable for dipoleMPItest
add_executable(dumpCouplings ${SOURCES_DUMPLINGS})

# Additional compile options based on build type
target_compile_options(dumpCouplings PRIVATE
  $<$<CONFIG:Debug>:-g>
  $<$<CONFIG:Release>:-O3>
)


# Include directories if wignerSymbols is not found
if(NOT wignerSymbols_FOUND)
  target_include_directories(dumpCouplings PRIVATE ${wignerSymbols_SOURCE_DIR}/include)
endif()

# Link libraries
# MPI::MPI_CXX, wignerSymbols, and OpenMP::OpenMP_CXX are linked publicly
# nlohmann_json::nlohmann_json is linked privately
target_link_libraries(dumpCouplings
  PUBLIC
    MPI::MPI_CXX
    wignerSymbols
    OpenMP::OpenMP_CXX
  PRIVATE
    nlohmann_json::nlohmann_json
)

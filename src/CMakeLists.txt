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
  
set(SOURCES_MOMENTUM
  momentum_expectation.cpp
  dkbbasis.cpp
  splinehandler.cpp
  spnrbasis.cpp)

set(SOURCES_REBUILD
  rebuild_psit.cpp
  dkbbasis.cpp
  splinehandler.cpp
  spnrbasis.cpp)

# Add the executable for dipoleMPItest
add_executable(dipoleMPItest ${SOURCES_DIPOLE})

# Add the executable for nondipoleMPItest
add_executable(nondipoleMPItest ${SOURCES_NONDIPOLE})

# Add the executable for nondipoleMPItest
add_executable(counterpulseMPItest ${SOURCES_COUNTERPULSE})

# Add the executable for momentum expectation value projection
add_executable(momentumExpectation ${SOURCES_MOMENTUM})

# Add the executable for momentum expectation value projection
add_executable(rebuildPsit ${SOURCES_REBUILD})

# Specify compile features
target_compile_features(dipoleMPItest PRIVATE cxx_std_11)

target_compile_features(momentumExpectation  PRIVATE cxx_std_11)
target_compile_features(rebuildPsit  PRIVATE cxx_std_11)

# Compiler options conditionally set based on compiler type
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(dipoleMPItest PRIVATE -ggdb3 -march=native)
  target_compile_options(momentumExpectation PRIVATE -ggdb3 -march=native)
  target_compile_options(rebuildPsit PRIVATE -ggdb3 -march=native)
endif()

# Additional compile options based on build type
target_compile_options(dipoleMPItest PRIVATE
  $<$<CONFIG:Debug>:-g>
  $<$<CONFIG:Release>:-O3>
)

target_compile_options(nondipoleMPItest PRIVATE
  $<$<CONFIG:Debug>:-g>
  $<$<CONFIG:Release>:-O3>
)


target_compile_options(counterpulseMPItest PRIVATE
  $<$<CONFIG:Debug>:-g>
  $<$<CONFIG:Release>:-O3>
)

target_compile_options(momentumExpectation PRIVATE
  $<$<CONFIG:Debug>:-g>
  $<$<CONFIG:Release>:-O3>
)

target_compile_options(rebuildPsit PRIVATE
  $<$<CONFIG:Debug>:-g>
  $<$<CONFIG:Release>:-O3>
)

# Include directories if wignerSymbols is not found
if(NOT wignerSymbols_FOUND)
  target_include_directories(dipoleMPItest PRIVATE ${wignerSymbols_SOURCE_DIR}/include)
  target_include_directories(nondipoleMPItest PRIVATE ${wignerSymbols_SOURCE_DIR}/include)
  target_include_directories(counterpulseMPItest PRIVATE ${wignerSymbols_SOURCE_DIR}/include)
  target_include_directories(momentumExpectation PRIVATE ${wignerSymbols_SOURCE_DIR}/include)  # Add this line
  target_include_directories(rebuildPsit PRIVATE ${wignerSymbols_SOURCE_DIR}/include)  # Add this line
endif()

# Link libraries
# MPI::MPI_CXX, wignerSymbols, and OpenMP::OpenMP_CXX are linked publicly
# nlohmann_json::nlohmann_json is linked privately
target_link_libraries(dipoleMPItest
  PUBLIC
    MPI::MPI_CXX
    wignerSymbols
    OpenMP::OpenMP_CXX
  PRIVATE
    nlohmann_json::nlohmann_json
)

target_link_libraries(nondipoleMPItest
  PUBLIC
    MPI::MPI_CXX
    wignerSymbols
    OpenMP::OpenMP_CXX
  PRIVATE
    nlohmann_json::nlohmann_json)
	
	
target_link_libraries(momentumExpectation
  PUBLIC
    MPI::MPI_CXX
    wignerSymbols
    OpenMP::OpenMP_CXX
  PRIVATE
    nlohmann_json::nlohmann_json)
	
target_link_libraries(rebuildPsit
  PUBLIC
    MPI::MPI_CXX
    wignerSymbols
    OpenMP::OpenMP_CXX
  PRIVATE
    nlohmann_json::nlohmann_json)
	
target_link_libraries(momentumExpectation
  PUBLIC
    MPI::MPI_CXX
    wignerSymbols
    OpenMP::OpenMP_CXX
  PRIVATE
    nlohmann_json::nlohmann_json)
	
target_link_libraries(momentumExpectation
  PUBLIC
    MPI::MPI_CXX
    wignerSymbols
    OpenMP::OpenMP_CXX
  PRIVATE
    nlohmann_json::nlohmann_json)
	
target_link_libraries(counterpulseMPItest
  PUBLIC
    MPI::MPI_CXX
    wignerSymbols
    OpenMP::OpenMP_CXX
  PRIVATE
    nlohmann_json::nlohmann_json)
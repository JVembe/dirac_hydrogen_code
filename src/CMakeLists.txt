# Define the source files
set(SOURCES
  dipoleMPItest.cpp
  dkbbasis.cpp
  splinehandler.cpp
  spnrbasis.cpp
)

# Add the executable for dipoleMPItest
add_executable(dipoleMPItest ${SOURCES})

# Specify compile features
target_compile_features(dipoleMPItest PRIVATE cxx_std_11)

# Compiler options conditionally set based on compiler type
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(dipoleMPItest PRIVATE -ggdb3 -march=native)
endif()

# Additional compile options based on build type
target_compile_options(dipoleMPItest PRIVATE
  $<$<CONFIG:Debug>:-g>
  $<$<CONFIG:Release>:-O3>
)

# Include directories if wignerSymbols is not found
if(NOT wignerSymbols_FOUND)
  target_include_directories(dipoleMPItest PRIVATE ${wignerSymbols_SOURCE_DIR}/include)
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

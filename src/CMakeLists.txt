# Define the source files
set(SOURCES_DKBTEST
  dkbbasistest.cpp
  dkbbasis.cpp
  splinehandler.cpp
  spnrbasis.cpp)	

# Add the executable for dkbbasistest
add_executable(dkbbasistest ${SOURCES_DKBTEST})

# Specify compile features
target_compile_features(dkbbasistest PRIVATE cxx_std_11)

# Compiler options conditionally set based on compiler type
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(dkbbasistest PRIVATE -ggdb3 -march=native)
endif()

target_compile_options(dkbbasistest PRIVATE
  $<$<CONFIG:Debug>:-g>
  -O3
)

# Include directories if wignerSymbols is not found
if(NOT wignerSymbols_FOUND)
  target_include_directories(dkbbasistest PRIVATE ${wignerSymbols_SOURCE_DIR}/include)
endif()

# Link libraries
# MPI::MPI_CXX, wignerSymbols, and OpenMP::OpenMP_CXX are linked publicly
# nlohmann_json::nlohmann_json is linked privately
target_link_libraries(dkbbasistest
  PUBLIC
    MPI::MPI_CXX
    wignerSymbols
    OpenMP::OpenMP_CXX
  PRIVATE
    nlohmann_json::nlohmann_json
)

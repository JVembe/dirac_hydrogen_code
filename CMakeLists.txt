# Minimum required version of CMake
cmake_minimum_required(VERSION 3.22)

# Project name and language
project(TD-Dirac CXX)

# Find required packages
# ------------------------------
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Fetch or find external libraries
# ------------------------------
include(FetchContent)

# Fetch wignerSymbols
find_package(wignerSymbols QUIET)
if(NOT wignerSymbols_FOUND)
  message(STATUS "wignerSymbols not found, downloading...")
  FetchContent_Declare(
    wignerSymbols
    GIT_REPOSITORY https://github.com/joeydumont/wignerSymbols.git
  )
  FetchContent_MakeAvailable(wignerSymbols)
endif()

# Fetch nlohmann_json
message(STATUS "Fetching nlohmann/json...")
FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.9.1  # adjust the version as needed
)

FetchContent_GetProperties(json)
if(NOT json_POPULATED)
  FetchContent_Populate(json)
  add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# Add subdirectories
# ------------------------------
add_subdirectory(src)

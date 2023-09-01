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
find_package(wignerSymbols QUIET)
if(NOT wignerSymbols_FOUND)
  message(STATUS "wignerSymbols not found, downloading...")
  FetchContent_Declare(
    wignerSymbols
    GIT_REPOSITORY https://github.com/joeydumont/wignerSymbols.git
  )
  FetchContent_MakeAvailable(wignerSymbols)
endif()

# Add subdirectories
# ------------------------------
add_subdirectory(src)

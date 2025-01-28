# GaDE - Gpu-accelerated solver for the time dependent Dirac Equation

# Copyright (C) 2025  Johanne Elise Vembe <johannevembe@gmail.com>
# Copyright (C) 2025  Marcin Krotkiewski <marcink@uio.no>
# Copyright (C) 2025  Hicham Agueny <hicham.agueny@uib.no>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


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

set(SOURCES_DUMPEVCS
  dumpEigvals.cpp
  dkbbasis.cpp
  splinehandler.cpp
  spnrbasis.cpp)

set(SOURCES_PROCESSOUTPUT 
  process_output.cpp
  dkbbasis.cpp
  splinehandler.cpp
  spnrbasis.cpp)

# Add the executable for dumpCouplings
add_executable(dumpCouplings ${SOURCES_DUMPLINGS})

# Add the executable for dumpEvecs
add_executable(dumpEvecs ${SOURCES_DUMPEVCS})

# Add the executable for processOutput
add_executable(processOutput ${SOURCES_PROCESSOUTPUT})

# Add the executable for nondipoleMPItest
add_executable(nondipoleMPItest ${SOURCES_NONDIPOLE})

# Add the executable for dipoleMPItest
add_executable(dipoleMPItest ${SOURCES_DIPOLE})

# Additional compile options based on build type
target_compile_options(dumpCouplings PRIVATE
  $<$<CONFIG:Debug>:-g>
  $<$<CONFIG:Release>:-O3>
)

target_compile_options(nondipoleMPItest PRIVATE
  $<$<CONFIG:Debug>:-g>
  $<$<CONFIG:Release>:-O3>
)

target_compile_options(dumpEvecs PRIVATE
  $<$<CONFIG:Debug>:-g>
  $<$<CONFIG:Release>:-O3>
)

target_compile_options(processOutput PRIVATE
  $<$<CONFIG:Debug>:-g>
  $<$<CONFIG:Release>:-O3>
)

target_compile_options(dipoleMPItest PRIVATE
  $<$<CONFIG:Debug>:-g -O3>
  $<$<CONFIG:Release>:-O3>
)

# Include directories if wignerSymbols is not found
if(NOT wignerSymbols_FOUND)
  target_include_directories(dumpCouplings PRIVATE ${wignerSymbols_SOURCE_DIR}/include)
  target_include_directories(nondipoleMPItest PRIVATE ${wignerSymbols_SOURCE_DIR}/include)
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

target_link_libraries(dumpEvecs
  PUBLIC
    MPI::MPI_CXX
    wignerSymbols
    OpenMP::OpenMP_CXX
  PRIVATE
    nlohmann_json::nlohmann_json
)

target_link_libraries(processOutput
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
	
target_link_libraries(dipoleMPItest
  PUBLIC
    MPI::MPI_CXX
    wignerSymbols
    OpenMP::OpenMP_CXX
  PRIVATE
    nlohmann_json::nlohmann_json)

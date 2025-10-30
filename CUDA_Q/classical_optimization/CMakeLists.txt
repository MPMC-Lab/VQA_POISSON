cmake_minimum_required(VERSION 3.10)
project(cudaq_project NONE)

set(CUDAQ_TARGET nvidia)
set(CUDAQ_EXEC nvq++)

add_custom_target(
    program ALL
    COMMAND ${CUDAQ_EXEC} --target=${CUDAQ_TARGET}
            ${CMAKE_SOURCE_DIR}/classical_functions.cpp
            ${CMAKE_SOURCE_DIR}/classical_optimization.cpp
            -I${NLOPT_INCLUDE_DIR}
            -L${NLOPT_LIBRARY_DIR} -lnlopt
            -o ${CMAKE_BINARY_DIR}/program.x
    DEPENDS ${CMAKE_SOURCE_DIR}/classical_functions.cpp ${CMAKE_SOURCE_DIR}/classical_optimization.cpp
    COMMENT "Building CUDA-Q program with nvq++"
)
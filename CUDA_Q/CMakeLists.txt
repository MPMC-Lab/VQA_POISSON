cmake_minimum_required(VERSION 3.10)
project(cudaq_project NONE)

set(CUDAQ_TARGET nvidia)
set(CUDAQ_EXEC nvq++)


set(INC_FLAGS "")
list(APPEND INC_FLAGS -I${CMAKE_SOURCE_DIR}/include)

set(COMMON_SOURCES
  ${CMAKE_SOURCE_DIR}/include/state_prep.cpp
  ${CMAKE_SOURCE_DIR}/include/classical_functions.cpp
  ${CMAKE_SOURCE_DIR}/include/IOmethods.cpp
)

if (DEFINED NLOPT_INCLUDE_DIR AND NLOPT_INCLUDE_DIR)
  list(APPEND INC_FLAGS -I${NLOPT_INCLUDE_DIR})
endif()

set(LINK_DIR_FLAGS "")
if (DEFINED NLOPT_LIBRARY_DIR AND NLOPT_LIBRARY_DIR)
  list(APPEND LINK_DIR_FLAGS -L${NLOPT_LIBRARY_DIR})
endif()

function(add_cudaq_program exe_name opt_src)
  set(out ${CMAKE_BINARY_DIR}/${exe_name}.x)
  set(workdir ${CMAKE_BINARY_DIR}/.nvqpp_${exe_name})

  add_custom_command(
    OUTPUT  ${out}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${workdir}
    COMMAND ${CMAKE_COMMAND} -E chdir ${workdir}
            ${CUDAQ_EXEC}
            --target=${CUDAQ_TARGET}
            ${COMMON_SOURCES}
            ${opt_src}
            ${INC_FLAGS}
            ${LINK_DIR_FLAGS}
            -lnlopt
            -o ${out}
    DEPENDS ${COMMON_SOURCES} ${opt_src}
    COMMENT "Building CUDA-Q program ${exe_name}.x with nvq++"
    VERBATIM
  )

  add_custom_target(${exe_name} ALL DEPENDS ${out})
endfunction()

add_cudaq_program(program_SD      ${CMAKE_SOURCE_DIR}/VQA_SD.cpp)
add_cudaq_program(program_FD_LNN  ${CMAKE_SOURCE_DIR}/VQA_FD_LNN.cpp)
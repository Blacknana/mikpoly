cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(mikpoly LANGUAGES CXX CUDA)

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  message(STATUS "Running submodule update to fetch cutlass")
  execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init 3rdparty/cutlass
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  RESULT_VARIABLE GIT_SUBMOD_RESULT)
  if(NOT GIT_SUBMOD_RESULT EQUAL "0")
    message(FATAL_ERROR "git submodule update --init 3rdparty/cutlass failed with ${GIT_SUBMOD_RESULT}, please checkout cutlass submodule")
  endif()
endif()

set(HEADER_DIRS
  ${PROJECT_SOURCE_DIR}
  ${CUDA_TOOLKIT_ROOT_DIR}/include
  ${PROJECT_SOURCE_DIR}/3rdparty/cutlass/include
  ${PROJECT_SOURCE_DIR}/3rdparty/cutlass/tools/util/include
)
include_directories(
  ${HEADER_DIRS}
)

set(LIB_DIRS
  ${PROJECT_SOURCE_DIR}
  ${CUDA_TOOLKIT_ROOT_DIR}/lib64
)
link_directories(
  ${LIB_DIRS}
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)

add_executable(test_mikpoly test_mikpoly.cu)
target_link_libraries(test_mikpoly PUBLIC mikpoly -lcublas)

cmake_minimum_required(VERSION 3.24.0)
project(mlinsights VERSION ${MLINSIGHTS_VERSION})

# CUDA is not used.
set(USE_CUDA 0)

#
# initialisation
#

message(STATUS "-------------------")
message(STATUS "MLINSIGHTS_VERSION=${MLINSIGHTS_VERSION}")
message(STATUS "CMAKE_VERSION=${CMAKE_VERSION}")
message(STATUS "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_C_COMPILER_VERSION=${CMAKE_C_COMPILER_VERSION}")
message(STATUS "CMAKE_CXX_COMPILER_VERSION=${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "USE_SETUP_PYTHON=${USE_SETUP_PYTHON}")
message(STATUS "USE_PYTHON_SETUP=${USE_PYTHON_SETUP}")
message(STATUS "PYTHON_VERSION=${PYTHON_VERSION}")
message(STATUS "PYTHON_VERSION_MM=${PYTHON_VERSION_MM}")
message(STATUS "PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}")
message(STATUS "PYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR}")
message(STATUS "PYTHON_LIBRARY=${PYTHON_LIBRARY}")
message(STATUS "PYTHON_LIBRARY_DIR=${PYTHON_LIBRARY_DIR}")
message(STATUS "PYTHON_NUMPY_INCLUDE_DIR=${PYTHON_NUMPY_INCLUDE_DIR}")
message(STATUS "PYTHON_MODULE_EXTENSION=${PYTHON_MODULE_EXTENSION}")
message(STATUS "PYTHON_NUMPY_VERSION=${PYTHON_NUMPY_VERSION}")
message(STATUS "USE_CUDA=${USE_CUDA}")
# message(STATUS "CUDA_BUILD=${CUDA_BUILD}")
# message(STATUS "CUDA_LINK=${CUDA_LINK}")
message(STATUS "USE_NVTX=${USE_NVTX}")
message(STATUS "ORT_VERSION=${ORT_VERSION}")

# message(STATUS "ENV-PATH=$ENV{PATH}")
# message(STATUS "ENV-PYTHONPATH=$ENV{PYTHONPATH}")
message(STATUS "--------------------------------------------")
message(STATUS "--------------------------------------------")
message(STATUS "--------------------------------------------")

# Don't let cmake set a default value for CMAKE_CUDA_ARCHITECTURES
# see https://cmake.org/cmake/help/latest/policy/CMP0104.html
# cmake_policy(SET CMP0104 OLD) # deprecated

list(APPEND CMAKE_MODULE_PATH
     "${CMAKE_CURRENT_SOURCE_DIR}"
     "${CMAKE_CURRENT_SOURCE_DIR}/externals")


#
# Packages and constants
#

include("constants.cmake")
include("load_externals.cmake")

#
# modules
#

message(STATUS "--------------------------------------------")
set(ROOT_PROJECT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(ROOT_UNITTEST_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../_unittests)
message(STATUS "ROOT_PROJECT_PATH=${ROOT_PROJECT_PATH}")
message(STATUS "ROOT_INCLUDE_PATH=${ROOT_INCLUDE_PATH}")
message(STATUS "ROOT_UNITTEST_PATH=${ROOT_UNITTEST_PATH}")
message(STATUS "--------------------------------------------")

#
# standalone modules
#

include("targets/_tree_digitize_cy.cmake")
include("targets/direct_blas_lapack_cy.cmake")
include("targets/piecewise_cy.cmake")

#
# write version
#

file(WRITE "../mlinsights/_config.py" "${config_content}")

#
# test
#

include(CTest)
enable_testing()

#
# Final
#

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

#
# Final check
#

get_property(targets_list GLOBAL PROPERTY PACKAGES_FOUND)
message(STATUS "-------------------")
message(STATUS "CMAKE_PROJECT_NAME = ${CMAKE_PROJECT_NAME}")
message(STATUS "list of found packages")
foreach(target ${targets_list})
  message(STATUS "  ${target}")
endforeach()
message(STATUS "-------------------")

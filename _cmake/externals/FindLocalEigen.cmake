#
# initialization
#
# function eigen_add_dependency
# output variables LOCAL_EIGEN_FOUND, LOCAL_EIGEN_TARGET

if(NOT LOCAL_EIGEN_VERSION)
  set(LOCAL_EIGEN_VERSION "3.4.0")
endif()
string(SUBSTRING "${LOCAL_EIGEN_VERSION}" 0 3 SHORT_EIGEN_VERSION)
set(LOCAL_EIGEN_ROOT https://gitlab.com/libeigen/eigen/-/archive/)
set(LOCAL_EIGEN_NAME "eigen-${LOCAL_EIGEN_VERSION}.zip")
set(LOCAL_EIGEN_URL "${LOCAL_EIGEN_ROOT}${LOCAL_EIGEN_VERSION}/${LOCAL_EIGEN_NAME}")
set(LOCAL_EIGEN_DEST "${CMAKE_CURRENT_BINARY_DIR}/eigen-download/${LOCAL_EIGEN_NAME}")
set(LOCAL_EIGEN_DEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/eigen-bin/")

FetchContent_Declare(eigen URL ${LOCAL_EIGEN_URL})

# This instruction add all the available targets in eigen
# including unit tests.
# FetchContent_makeAvailable(eigen)

FetchContent_Populate(eigen)

list(APPEND CMAKE_MODULE_PATH "${eigen_SOURCE_DIR}/cmake")
# find_package(Eigen3)

set(LOCAL_EIGEN_SOURCE "${eigen_SOURCE_DIR}")

# find_package(Eigen3 ${SHORT_EIGEN_VERSION} REQUIRED NO_MODULE)
set(LOCAL_EIGEN_TARGET Eigen3::Eigen)
set(LOCAL_EIGEN_VERSION ${Eigen3_VERSION})
set(EIGEN_INCLUDE_DIRS "${eigen_SOURCE_DIR}")

#
# !eigen_add_dependency: add a dependency to eigen.
#
#
# \arg:name target name
#
function(eigen_add_dependency name)
  target_include_directories(${name} PRIVATE ${EIGEN_INCLUDE_DIRS})
endfunction()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  LocalEigen
  VERSION_VAR LOCAL_EIGEN_VERSION
  REQUIRED_VARS LOCAL_EIGEN_TARGET LOCAL_EIGEN_URL LOCAL_EIGEN_SOURCE
                EIGEN_INCLUDE_DIRS)

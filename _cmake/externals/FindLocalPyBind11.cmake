#
# initialization
#
# defines LocalPyBind11 pybind11_SOURCE_DIR pybind11_BINARY_DIR
# and functions local_pybind11_add_module, cuda_pybind11_add_module

#
# pybind11
#

set(pybind11_TAG "v2.13.5")

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG ${pybind11_TAG})

FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
  FetchContent_Populate(pybind11)
  message(STATUS "pybind11_SOURCE_DIR=${pybind11_SOURCE_DIR}")
  message(STATUS "pybind11_BINARY_DIR=${pybind11_BINARY_DIR}")
  add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
else()
  message(FATAL_ERROR "Pybind11 was not found.")
endif()

set(pybind11_VERSION ${pybind11_TAG})
message(STATUS "PYBIND11_OPT_SIZE=${PYBIND11_OPT_SIZE}")
message(STATUS "pybind11_INCLUDE_DIR=${pybind11_INCLUDE_DIR}")
message(STATUS "pybind11_VERSION=${pybind11_VERSION}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  LocalPyBind11
  VERSION_VAR pybind11_VERSION
  REQUIRED_VARS pybind11_SOURCE_DIR pybind11_BINARY_DIR)

#
#! local_pybind11_add_module : compile a pybind11 extension
#
# \arg:name extension name
# \arg:omp_lib omp library to link with
# \argn: additional c++ files to compile
#
function(local_pybind11_add_module name omp_lib)
  message(STATUS "pybind11 module '${name}': ${pyx_file} ++ ${ARGN}")
  python3_add_library(${name} MODULE ${ARGN})
  target_include_directories(
    ${name} PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${PYTHON3_INCLUDE_DIR}
    ${Python3_NumPy_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIR}
    ${NUMPY_INCLUDE_DIR}
    ${OMP_INCLUDE_DIR})
  target_link_libraries(
    ${name} PRIVATE
    pybind11::headers
    ${Python3_LIBRARY_RELEASE}  # use ${Python3_LIBRARIES} if python debug
    ${Python3_NumPy_LIBRARIES}
    ${omp_lib})
  # if(MSVC) target_link_libraries(${target_name} PRIVATE
  # pybind11::windows_extras pybind11::lto) endif()
  set_target_properties(
    ${name} PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION ON
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}")
  message(STATUS "pybind11 added module '${name}'")
  get_target_property(prop ${name} BINARY_DIR)
  message(STATUS "pybind11 added into '${prop}'.")
endfunction()

#
#! cuda_pybind11_add_module : compile a pyx file into cpp
#
# \arg:name extension name
# \arg:pybindfile pybind11 extension
# \argn: additional c++ files to compile as the cuda extension
#
function(cuda_pybind11_add_module name pybindfile)
  local_pybind11_add_module(${name} OpenMP::OpenMP_CXX ${pybindfile} ${ARGN})
  target_compile_definitions(
    ${name}
    PRIVATE
    CUDA_VERSION=${CUDA_VERSION_INT}
    PYTHON_MANYLINUX=${PYTHON_MANYLINUX})
  target_include_directories(${name} PRIVATE ${CUDA_INCLUDE_DIRS})
  message(STATUS "    LINK ${name} <- stdc++ ${CUDA_LIBRARIES}")
  target_link_libraries(${name} PRIVATE stdc++ ${CUDA_LIBRARIES})
  if(USE_NVTX)
    message(STATUS "    LINK ${name} <- nvtx3-cpp")
    target_link_libraries(${name} PRIVATE nvtx3-cpp)
  endif()

  # add property --use_fast_math to cu files
  # set(NEW_LIST ${name}_src_files)
  # list(APPEND ${name}_cu_files ${ARGN})
  # list(FILTER ${name}_cu_files INCLUDE REGEX ".+[.]cu$")
  # set_source_files_properties(
  #   ${name}_cu_files PROPERTIES COMPILE_OPTIONS "--use_fast_math")
endfunction()

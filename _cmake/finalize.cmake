
if(CUDA_AVAILABLE)
  set(
    config_content_cuda
    "HAS_CUDA = 1\nCUDA_VERSION = '${CUDA_VERSION}'"
    "\nCUDA_VERSION_INT = ${CUDA_VERSION_INT}")
else()
  set(
    config_content_cuda "HAS_CUDA = 0")
endif()

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import cython;print(cython.__version__)"
  OUTPUT_VARIABLE CYTHON_VERSION
  ERROR_VARIABLE CYTHON_version_error
  RESULT_VARIABLE CYTHON_version_result
  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import sklearn;print(sklearn.__version__)"
  OUTPUT_VARIABLE SKLEARN_VERSION
  ERROR_VARIABLE SKLEARN_version_error
  RESULT_VARIABLE SKLEARN_version_result
  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE)
  
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import numpy;print(numpy.__version__)"
  OUTPUT_VARIABLE NUMPY_VERSION
  ERROR_VARIABLE NUMPY_version_error
  RESULT_VARIABLE NUMPY_version_result
  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE)
  
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import scipy;print(scipy.__version__)"
  OUTPUT_VARIABLE SCIPY_VERSION
  ERROR_VARIABLE SCIPY_version_error
  RESULT_VARIABLE SCIPY_version_result
  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE)
  
set(
  config_content_comma
  "${config_content_cuda}"
  "\nCXX_FLAGS = '${CMAKE_CXX_FLAGS}'"
  "\nCMAKE_CXX_STANDARD_REQUIRED = '${CMAKE_CXX_STANDARD_REQUIRED}'"
  "\nCMAKE_CXX_EXTENSIONS = '${CMAKE_CXX_EXTENSIONS}'"
  "\nCMAKE_CXX_STANDARD = ${CMAKE_CXX_STANDARD}"
  "\n\n# Was compiled with the following versions."
  "\nCYTHON_VERSION = '${CYTHON_VERSION}'"
  "\nSKLEARN_VERSION = '${SKLEARN_VERSION}'"
  "\nNUMPY_VERSION = '${NUMPY_VERSION}'"
  "\nSCIPY_VERSION = '${SCIPY_VERSION}'"
  "\n")

string(REPLACE ";" "" config_content "${config_content_comma}")

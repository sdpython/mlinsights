
if(CUDA_AVAILABLE)
  set(
    config_content_cuda
    "HAS_CUDA = 1\nCUDA_VERSION = '${CUDA_VERSION}'"
    "\nCUDA_VERSION_INT = ${CUDA_VERSION_INT}")
else()
  set(
    config_content_cuda "HAS_CUDA = 0")
endif()

set(
  config_content_comma
  "${config_content_cuda}"
  "\nCXX_FLAGS = '${CMAKE_CXX_FLAGS}'"
  "\nCMAKE_CXX_STANDARD_REQUIRED = '${CMAKE_CXX_STANDARD_REQUIRED}'"
  "\nCMAKE_CXX_EXTENSIONS = '${CMAKE_CXX_EXTENSIONS}'"
  "\nCMAKE_CXX_STANDARD = ${CMAKE_CXX_STANDARD}\n")

string(REPLACE ";" "" config_content "${config_content_comma}")

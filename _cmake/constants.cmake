
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
  cmake_policy(SET CMP0077 NEW)
endif()

#
# initialisation
#

message(STATUS "--------------------------------------------")
message(STATUS "--------------------------------------------")
message(STATUS "--------------------------------------------")
message(STATUS "ONNX_EXTENDED_VERSION=${ONNX_EXTENDED_VERSION}")
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
message(STATUS "CUDA_BUILD=${CUDA_BUILD}")
message(STATUS "CUDA_LINK=${CUDA_LINK}")
message(STATUS "USE_NVTX=${USE_NVTX}")
message(STATUS "ORT_VERSION=${ORT_VERSION}")
message(STATUS "PYTHON_MANYLINUX=${PYTHON_MANYLINUX}")
message(STATUS "SETUP_BUILD_PATH=${SETUP_BUILD_PATH}")
message(STATUS "SETUP_BUILD_LIB=${SETUP_BUILD_LIB}")
message(STATUS "--------------------------------------------")
message(STATUS "--------------------------------------------")
message(STATUS "--------------------------------------------")

#
# python extension
#
if(MSVC)
  set(DLLEXT "dll")
elseif(APPLE)
  set(DLLEXT "dylib")
else()
  set(DLLEXT "so")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")  # -DNDEBUG

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")  # -DNDEBUG

#
# C++ 14 or C++ 17 or...
#
if (PYTHON_MANYLINUX EQUAL "1")
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unknown-pragmas -Wextra")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wno-unknown-pragmas -Wextra")
  if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15")
  elseif(MSVC)
    # nothing
  else()
    execute_process(
      COMMAND ldd --version | grep "ldd (.*)"
      OUTPUT_VARIABLE ldd_version_output
      ERROR_VARIABLE ldd_version_error
      RESULT_VARIABLE ldd_version_result)
  endif()
else()
  if(MSVC)
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
    set(CMAKE_CXX_STANDARD 17)
  elseif(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unknown-pragmas -Wextra")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wno-unknown-pragmas -Wextra")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unknown-pragmas -Wextra")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wno-unknown-pragmas -Wextra")
    if(CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL "15")
      # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++23")
      set(CMAKE_CXX_STANDARD 23)
    elseif(CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL "11")
      # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
      set(CMAKE_CXX_STANDARD 20)
    elseif(CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL "9")
      # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
      set(CMAKE_CXX_STANDARD 17)
    elseif(CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL "6")
      # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
      set(CMAKE_CXX_STANDARD 14)
    else()
      message(FATAL_ERROR "gcc>=6.0 is needed but "
                          "${CMAKE_C_COMPILER_VERSION} was detected.")
    endif()
    # needed to build many linux build
    # set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lm")
    # set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -lm")

    execute_process(
      COMMAND ldd --version | grep "ldd (.*)"
      OUTPUT_VARIABLE ldd_version_output
      ERROR_VARIABLE ldd_version_error
      RESULT_VARIABLE ldd_version_result)
  endif()
endif()

# Disable fast-math for Intel oneAPI compiler
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "IntelLLVM")
  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC-like")
    # Using icx-cl compiler driver with MSVC-like arguments
    message(STATUS "IntelLLVM + MSVC-like")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:precise")
  else()
    # Using icpx compiler driver
    message(STATUS "IntelLLVM + no MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-fast-math")
  endif()
endif()

set(TEST_FOLDER "${CMAKE_CURRENT_SOURCE_DIR}/../_unittests")
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/test_constants.h.in
  ${TEST_FOLDER}/test_constants.h
)

#
# Compiling options
#

# AVX instructions
if(MSVC)
  # disable warning for #pragma unroll
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX")
  add_compile_options(/wd4068)
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

if(APPLE)
  message(STATUS "APPLE: set env var for open mp: CC, CCX, LDFLAGS, CPPFLAGS")
  set(ENV{CC} "/usr/local/opt/llvm/bin/clang")
  set(ENV{CXX} "/usr/local/opt/llvm/bin/clang++")
  set(ENV(LDFLAGS) "-L/usr/local/opt/llvm/lib")
  set(ENV(CPPFLAGS) "-I/usr/local/opt/llvm/include")
endif()

message(STATUS "**********************************")
message(STATUS "**********************************")
message(STATUS "**********************************")
message(STATUS "GLIBC_VERSION=${ldd_version_output}")
message(STATUS "CMAKE_C_COMPILER=${CMAKE_C_COMPILER}")
message(STATUS "CMAKE_C_FLAGS_INIT=${CMAKE_C_FLAGS_INIT}")
message(STATUS "CMAKE_C_FLAGS=${CMAKE_C_FLAGS}")
message(STATUS "CMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}")
message(STATUS "CMAKE_C_COMPILER_VERSION=${CMAKE_C_COMPILER_VERSION}")
message(STATUS "CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_CXX_FLAGS_INIT=${CMAKE_CXX_FLAGS_INIT}")
message(STATUS "CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "CMAKE_CXX_COMPILER_ID=${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}")
message(STATUS "CMAKE_CXX_STANDARD_REQUIRED=${CMAKE_CXX_STANDARD_REQUIRED}")
message(STATUS "CMAKE_CXX_EXTENSIONS=${CMAKE_CXX_EXTENSIONS}")
message(STATUS "CMAKE_EXE_LINKER_FLAGS=${CMAKE_EXE_LINKER_FLAGS}")
message(STATUS "CMAKE_LINKER=${CMAKE_LINKER}")
message(STATUS "CMAKE_SHARED_LINKER_FLAGS=${CMAKE_SHARED_LINKER_FLAGS}")
message(STATUS "LDFLAGS=${LDFLAGS}")
message(STATUS "CPPFLAGS=${CPPFLAGS}")
message(STATUS "DLL_EXT=${DLL_EXT}")
message(STATUS "TEST_FOLDER=${TEST_FOLDER}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "**********************************")
message(STATUS "**********************************")
message(STATUS "**********************************")

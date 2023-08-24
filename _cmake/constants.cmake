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

#
# C++ 14 or C++ 17
#
if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
else()
  if(CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL "11")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
  else()
    if(CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL "6")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    else()
      message(FATAL_ERROR "gcc>=6.0 is needed but "
                          "${CMAKE_C_COMPILER_VERSION} was detected.")
    endif()
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

message(STATUS "--------------------------------------------")
message(STATUS "CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")
message(STATUS "LDFLAGS=${LDFLAGS}")
message(STATUS "CPPFLAGS=${CPPFLAGS}")
message(STATUS "--------------------------------------------")


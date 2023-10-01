#
# initialization
#
# output variables Cython_FOUND, Cython_VERSION function cython_add_module

execute_process(
  COMMAND ${Python3_EXECUTABLE} -m cython --version
  OUTPUT_VARIABLE CYTHON_version_output
  ERROR_VARIABLE CYTHON_version_error
  RESULT_VARIABLE CYTHON_version_result
  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE)
message(STATUS "CYTHON_version_output=${CYTHON_version_output}")
message(STATUS "CYTHON_version_error=${CYTHON_version_error}")
message(STATUS "CYTHON_version_result=${CYTHON_version_result}")

if(NOT ${CYTHON_version_result} EQUAL 0)
  # installation of cython, numpy
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pip install cython numpy
    OUTPUT_VARIABLE install_version_output
    ERROR_VARIABLE install_version_error
    RESULT_VARIABLE install_version_result)
  message(STATUS "install_version_output=${install_version_output}")
  message(STATUS "install_version_error=${install_version_error}")
  message(STATUS "install_version_result=${install_version_result}")
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -m cython --version
    OUTPUT_VARIABLE CYTHON_version_output
    ERROR_VARIABLE CYTHON_version_error
    RESULT_VARIABLE CYTHON_version_result
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE)
  message(STATUS "CYTHON_version_output=${CYTHON_version_output}")
  message(STATUS "CYTHON_version_error=${CYTHON_version_error}")
  message(STATUS "CYTHON_version_result=${CYTHON_version_result}")
  if(NOT ${CYTHON_version_result} EQUAL 0)
    message(FATAL_ERROR("Unable to find cython for '${PYTHON_EXECUTABLE}'."))
  endif()
  set(Cython_VERSION ${CYTHON_version_error})
else()
  set(Cython_VERSION ${CYTHON_version_error})
endif()

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import numpy;print(numpy.get_include())"
  OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE NUMPY_NOT_FOUND)
if(NUMPY_NOT_FOUND)
  message(FATAL_ERROR
          "Numpy headers not found with "
          "Python3_EXECUTABLE='${Python3_EXECUTABLE}' and "
          "Cython_VERSION=${Cython_VERSION}.")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Cython
  VERSION_VAR Cython_VERSION
  REQUIRED_VARS NUMPY_INCLUDE_DIR)

#
#! compile_cython : compile a pyx file into cpp
#
# \arg:filename extension name
# \arg:pyx_file_cpp output pyx file name
#
function(compile_cython filename pyx_file_cpp)
  message(STATUS "cython cythonize '${filename}'")
  set(fullfilename "${CMAKE_CURRENT_SOURCE_DIR}/${filename}")
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/${pyx_file_cpp}
    COMMAND
      ${Python3_EXECUTABLE} -m cython -3 --cplus ${fullfilename} -X
      boundscheck=False -X cdivision=True -X wraparound=False -X
      cdivision_warnings=False -X embedsignature=True -X initializedcheck=False
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${filename})
  message(STATUS "cython cythonize '${filename}' - done")
endfunction()

#
#! cython_add_module : compile a pyx file into cpp
#
# \arg:name extension name
# \arg:pyx_file pyx file name
# \arg:omp_lib omp library to link with
# \argn: additional c++ files to compile
#
function(cython_add_module name pyx_file omp_lib)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs SOURCES DEPS)
  message(STATUS "cython module '${name}': ${pyx_file} ++ ${ARGN}")
  get_filename_component(pyx_dir ${pyx_file} DIRECTORY)

  # cythonize

  compile_cython(${pyx_file} ${pyx_dir}/${name}.cpp)
  list(APPEND ARGN ${pyx_dir}/${name}.cpp)

  # adding the library

  message(STATUS "cython all files: ${ARGN}")
  python3_add_library(${name} MODULE ${ARGN})

  target_include_directories(
    ${name} PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIR}
    ${Python3_NumPy_INCLUDE_DIRS}
    ${NUMPY_INCLUDE_DIR}
    ${OMP_INCLUDE_DIR})

  message(STATUS "    LINK ${name} <- ${Python3_LIBRARY_RELEASE} "
                 "${Python3_NumPy_LIBRARIES} ${omp_lib}")
  target_link_libraries(
    ${name} PRIVATE
    ${Python3_LIBRARY_RELEASE}  # use ${Python3_LIBRARIES} if python debug
    ${Python3_NumPy_LIBRARIES}
    ${omp_lib})

  target_compile_definitions(${name} PUBLIC NPY_NO_DEPRECATED_API)

  set_target_properties(
    ${name} PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}")

  # install(TARGETS ${name} LIBRARY DESTINATION ${pyx_dir})

  message(STATUS "cython added module '${name}'")
  get_target_property(prop ${name} BINARY_DIR)
  message(STATUS "cython added into '${prop}'.")
endfunction()

#
# initialization
#
# defines python3_add_library
# use FindPython.cmake or use the python defined in cmake variable
# if USE_SETUP_PYTHON is set.

#
# pybind11
#

if(USE_SETUP_PYTHON)
  message(STATUS "Use Python from setup.py")
  set(Python3_VERSION ${PYTHON_VERSION})
  set(Python3_Interpreter_FOUND 1)
  set(Python3_Development_FOUND 1)
  set(Python3_INCLUDE_DIRS ${PYTHON_INCLUDE_DIR})
  set(Python3_LIBRARY ${PYTHON_LIBRARY})
  set(Python3_LIBRARIES ${PYTHON_LIBRARY})
  set(Python3_LIBRARY_RELEASE ${PYTHON_LIBRARY})
  set(Python3_LIBRARY_DIRS ${PYTHON_LIBRARY_DIR})
  set(Python3_EXECUTABLE ${PYTHON_EXECUTABLE})
  set(Python3_MODULE_EXTENSION ${PYTHON_MODULE_EXTENSION})
  set(Python3_MODULE_PREFIX "")
  set(Python3_LINK_OPTIONS "")
  set(Python3_NumPy_INCLUDE_DIRS ${PYTHON_NUMPY_INCLUDE_DIR})
  set(Python3_NumPy_VERSION PYTHON_NUMPY_VERSION)

  #
  #! python3_add_library : add a python library
  #
  # The function fails because it is not adding Python3{version}.lib.
  # The code is here:
  # https://github.com/Kitware/CMake/blob/
  # master/Modules/FindPython/Support.cmake.
  #
  # \arg:name extension name
  # \arg:prefix MODULE,SHARED,STATIC
  #
  function(python3_add_library name prefix)
    cmake_parse_arguments(
      PARSE_ARGV 2 PYTHON_ADD_LIBRARY
      "STATIC;SHARED;MODULE;WITH_SOABI" "" "")

    message(STATUS "Build python3 '${name}' with type='${type}' and "
            "PYTHON_ADD_LIBRARY_UNPARSED_ARGUMENTS="
            "${PYTHON_ADD_LIBRARY_UNPARSED_ARGUMENTS}.")

    if(PYTHON_ADD_LIBRARY_STATIC)
      set(type STATIC)
    elseif(PYTHON_ADD_LIBRARY_SHARED)
      set(type SHARED)
    else()
      set(type MODULE)
    endif()

    add_library(${name} ${type} ${PYTHON_ADD_LIBRARY_UNPARSED_ARGUMENTS})
    target_compile_definitions(${name} PRIVATE PYTHON_MANYLINUX=${PYTHON_MANYLINUX})
    target_include_directories(
      ${name} PRIVATE
      ${Python3_INCLUDE_DIRS}
      ${PYTHON_INCLUDE_DIR}
      ${Python3_NumPy_INCLUDE_DIRS}
      ${NUMPY_INCLUDE_DIR})

    set_target_properties(
      ${name} PROPERTIES
      PREFIX "${PYTHON_MODULE_PREFIX}"
      SUFFIX "${PYTHON_MODULE_EXTENSION}")
  endfunction()
else()
  message(STATUS "Use find_package(Python3).")
  set(Python3_EXECUTABLE ${PYTHON_EXECUTABLE})
  if(APPLE)
    find_package(Python3 ${PYTHON_VERSION} COMPONENTS
                Interpreter Development.Module
                REQUIRED)
    set(Python_NumPy_INCLUDE_DIRS ${PYTHON_NUMPY_INCLUDE_DIR})
  else()
    find_package(Python3 ${PYTHON_VERSION} COMPONENTS
                Interpreter NumPy Development.Module
                REQUIRED)
  endif()

  if(Python3_Interpreter_FOUND)
    if(NOT Python3_LIBRARY)
      set(Python3_LIBRARY ${PYTHON_LIBRARY})
    endif()
    if(NOT Python3_LIBRARIES)
      set(Python3_LIBRARIES ${PYTHON_LIBRARY})
    endif()
    if(NOT Python3_LIBRARY_RELEASE)
      set(Python3_LIBRARY_RELEASE ${PYTHON_LIBRARY})
    endif()
    if(NOT Python3_MODULE_EXTENSION)
      set(Python3_MODULE_EXTENSION ${PYTHON_MODULE_EXTENSION})
    endif()
    if(NOT Python3_MODULE_PREFIX)
      set(Python3_MODULE_PREFIX "")
    endif()
    if(NOT Python3_NumPy_VERSION)
      set(Python3_NumPy_VERSION ${PYTHON_NUMPY_VERSION})
    endif()
    if(NOT Python3_NumPy_INCLUDE_DIRS)
      set(Python3_NumPy_INCLUDE_DIRS ${PYTHON_NUMPY_INCLUDE_DIR})
    endif()

    message(STATUS "Python3_Interpreter_FOUND=${Python3_Interpreter_FOUND}")
    message(STATUS "Python3_NumPy_VERSION=${Python3_NumPy_VERSION}")
    message(STATUS "PYTHON_VERSION=${PYTHON_VERSION}")
    message(STATUS "Python3_VERSION=${Python3_VERSION}")
    message(STATUS "Python3_EXECUTABLE=${Python3_EXECUTABLE}")
    message(STATUS "Python3_INCLUDE_DIRS=${Python3_INCLUDE_DIRS}")
    message(STATUS "Python3_LIBRARY_DIRS=${Python3_LIBRARY_DIRS}")
    message(STATUS "Python3_LIBRARIES=${Python3_LIBRARIES}")
    message(STATUS "Python3_LIBRARY=${Python3_LIBRARY}")
    message(STATUS "Python3_LIBRARY_RELEASE=${Python3_LIBRARY_RELEASE}")
    message(STATUS "Python3_LINK_OPTIONS=${Python3_LINK_OPTIONS}")
    message(STATUS "Python3_NumPy_FOUND=${Python3_NumPy_FOUND}")
    message(STATUS "Python3_NumPy_INCLUDE_DIRS=${Python3_NumPy_INCLUDE_DIRS}")
    message(STATUS "Python3_NumPy_VERSION=${Python3_NumPy_VERSION}")
    message(STATUS "Python3_Development_FOUND=${Python3_Development_FOUND}")
    message(STATUS "Python3_MODULE_EXTENSION=${Python3_MODULE_EXTENSION}")
    message(STATUS "Python3_MODULE_PREFIX=${Python3_MODULE_PREFIX}")
    message(STATUS "Python3_SOABI=${Python3_SOABI}")
    message(STATUS "Python3_SOSABI=${Python3_SOSABI}")
  else()
    message(STATUS "Python3_INCLUDE_DIRS=${Python3_INCLUDE_DIRS}")
    message(FATAL_ERROR "Python was not found.")
  endif()
endif()

set(MyPython_VERSION "0.1")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  MyPython
  VERSION_VAR MyPython_VERSION
  REQUIRED_VARS Python3_VERSION Python3_EXECUTABLE Python3_INCLUDE_DIRS
                Python3_MODULE_EXTENSION
                Python3_NumPy_INCLUDE_DIRS Python3_NumPy_VERSION)

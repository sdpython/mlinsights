#
# initialization
#
# downloads onnxruntime as a binary
# functions ort_add_dependency, ort_add_custom_op

file(WRITE "../_setup_ext.txt" "")

if(NOT ORT_VERSION)
  set(ORT_VERSION 1.15.1)
  set(ORT_VERSION_INT 1150)
endif()
string(LENGTH "${ORT_VERSION}" ORT_VERSION_LENGTH)

if(CUDAToolkit_FOUND)
  if(APPLE)
    message(WARNING "onnxruntime-gpu not available on MacOsx")
  endif()
  set(ORT_GPU "-gpu")
else()
  set(ORT_GPU "")
endif()

if(ORT_VERSION_LENGTH LESS_EQUAL 12)
  message(STATUS "ORT - retrieve release version ${ORT_VERSION}")
  if(MSVC)
    set(ORT_NAME "onnxruntime-win-x64${ORT_GPU}-${ORT_VERSION}.zip")
    set(ORT_FOLD "onnxruntime-win-x64${ORT_GPU}-${ORT_VERSION}")
  elseif(APPLE)
    set(ORT_NAME "onnxruntime-osx-universal2-${ORT_VERSION}.tgz")
    set(ORT_FOLD "onnxruntime-osx-universal2-${ORT_VERSION}")
  else()
    set(ORT_NAME "onnxruntime-linux-x64${ORT_GPU}-${ORT_VERSION}.tgz")
    set(ORT_FOLD "onnxruntime-linux-x64${ORT_GPU}-${ORT_VERSION}")
  endif()
  set(ORT_ROOT "https://github.com/microsoft/onnxruntime/releases/download/")
  set(ORT_URL "${ORT_ROOT}v${ORT_VERSION}/${ORT_NAME}")
  set(ORT_DEST "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime-download/${ORT_NAME}")
  set(ORT_DEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime-bin/")

  string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)" ORT_VERSION_MATCH ${ORT_VERSION})
  set(ORT_VERSION_MAJOR ${CMAKE_MATCH_1})
  set(ORT_VERSION_MINOR ${CMAKE_MATCH_2})
  math(
    EXPR
    ORT_VERSION_INT
    "${ORT_VERSION_MAJOR} * 1000 + ${ORT_VERSION_MINOR} * 10"
    OUTPUT_FORMAT DECIMAL)

  FetchContent_Declare(onnxruntime URL ${ORT_URL})
  FetchContent_makeAvailable(onnxruntime)
  set(ONNXRUNTIME_INCLUDE_DIR ${onnxruntime_SOURCE_DIR}/include)
  set(ONNXRUNTIME_LIB_DIR ${onnxruntime_SOURCE_DIR}/lib)
else()
  message(STATUS "ORT - retrieve development version from '${ORT_VERSION}'")
  set(ORT_VERSION_INT 99999)
  set(ONNXRUNTIME_LIB_DIR "${ORT_VERSION}")
  set(ONNXRUNTIME_INCLUDE_DIR
      "${ORT_VERSION}/../../../include/onnxruntime/core/session")
  set(ORT_URL ${ORT_VERSION})
endif()

find_library(ONNXRUNTIME onnxruntime HINTS "${ONNXRUNTIME_LIB_DIR}")
if(ONNXRUNTIME-NOTFOUND)
  message(FATAL_ERROR "onnxruntime cannot be found at '${ONNXRUNTIME_LIB_DIR}'")
endif()

file(GLOB ORT_LIB_FILES ${ONNXRUNTIME_LIB_DIR}/*.${DLLEXT}*)
file(GLOB ORT_LIB_HEADER ${ONNXRUNTIME_INCLUDE_DIR}/*.h)

list(LENGTH ORT_LIB_FILES ORT_LIB_FILES_LENGTH)
if (ORT_LIB_FILES_LENGTH LESS_EQUAL 1)
  message(FATAL_ERROR "No file found in '${ONNXRUNTIME_LIB_DIR}' "
                      "from url '${ORT_URL}', "
                      "found files [${ORT_LIB_FILES}].")
endif()

list(LENGTH ORT_LIB_HEADER ORT_LIB_HEADER_LENGTH)
if (ORT_LIB_HEADER_LENGTH LESS_EQUAL 1)
  message(FATAL_ERROR "No file found in '${ONNXRUNTIME_INCLUDE_DIR}' "
                      "from url '${ORT_URL}', "
                      "found files [${ORT_LIB_HEADER}]")
endif()

#
#! ort_add_dependency : copies necessary onnxruntime assembly
#                       to the location a target is build
#
# \arg:name target name
#
function(ort_add_dependency name folder_copy)
  get_target_property(target_output_directory ${name} BINARY_DIR)
  message(STATUS "ort: copy-1 ${ORT_LIB_FILES_LENGTH} files from '${ONNXRUNTIME_LIB_DIR}'")
  if(MSVC)
    set(destination_dir ${target_output_directory}/${CMAKE_BUILD_TYPE})
  else()
    set(destination_dir ${target_output_directory})
  endif()
  message(STATUS "ort: copy-2 to '${destination_dir}'")
  if(folder_copy)
    message(STATUS "ort: copy-3 to '${folder_copy}'")
  endif()
  foreach(file_i ${ORT_LIB_FILES})
    if(NOT EXISTS ${destination_dir}/${file_i})
      message(STATUS "ort: copy-4 '${file_i}' to '${destination_dir}'")
      add_custom_command(
        TARGET ${name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} ARGS -E copy ${file_i} ${destination_dir})
    endif()
    if(folder_copy)
      if(NOT EXISTS ${folder_copy}/${file_i})
        message(STATUS "ort: copy-5 '${file_i}' to '${folder_copy}'")
        # file(APPEND "../_setup_ext.txt" "copy,${file_i},${folder_copy}\n")
        add_custom_command(
          TARGET ${name} POST_BUILD
          COMMAND ${CMAKE_COMMAND} ARGS -E copy ${file_i} ${folder_copy})
      endif()
    endif()
  endforeach()
  # file(COPY ${ORT_LIB_FILES} DESTINATION ${target_output_directory})
endfunction()

#
#! ort_add_custom_op : compile a pyx file into cpp
#
# \arg:name project name
# \arg:folder where to copy the library
# \arg:provider CUDA if a cuda lib, CPU if CPU
# \argn: C++ file to compile
#
function(ort_add_custom_op name provider folder)
  if (WIN32)
    file(WRITE "${folder}/${name}.def" "LIBRARY "
               "\"${name}.dll\"\nEXPORTS\n  RegisterCustomOps @1")
    list(APPEND ARGN "${folder}/${name}.def")
  endif()
  if (provider STREQUAL "CUDA")
    message(STATUS "ort: custom op ${provider}: '${name}': ${ARGN}")
    add_library(${name} SHARED ${ARGN})

    # add property --use_fast_math to cu files
    # set(NEW_LIST ${name}_src_files)
    # list(APPEND ${name}_cu_files ${ARGN})
    # list(FILTER ${name}_cu_files INCLUDE REGEX ".+[.]cu$")
    # set_source_files_properties(
    #  ${name}_cu_files PROPERTIES COMPILE_OPTIONS "--use_fast_math")

    target_compile_definitions(
      ${name}
      PRIVATE
      CUDA_VERSION=${CUDA_VERSION_INT}
      ORT_VERSION=${ORT_VERSION_INT})
    if(USE_NVTX)
      message(STATUS "    LINK ${name} <- stdc++ nvtx3-cpp ${CUDA_LIBRARIES}")
      target_link_libraries(
        ${name}
        PRIVATE
        stdc++
        nvtx3-cpp
        ${CUDA_LIBRARIES})
    else()
      message(STATUS "    LINK ${name} <- stdc++ ${CUDA_LIBRARIES}")
      target_link_libraries(
        ${name}
        PRIVATE
        stdc++
        ${CUDA_LIBRARIES})
    endif()
    target_include_directories(
      ${name}
      PRIVATE
      ${ONNXRUNTIME_INCLUDE_DIR})
  else()
    message(STATUS "ort: custom op CPU: '${name}': ${ARGN}")
    add_library(${name} SHARED ${ARGN})
    target_include_directories(${name} PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    target_compile_definitions(${name} PRIVATE ORT_VERSION=${ORT_VERSION_INT})
  endif()
  set_property(TARGET ${name} PROPERTY POSITION_INDEPENDENT_CODE ON)
  get_target_property(target_file ${name} LIBRARY_OUTPUT_NAME)
  # add_custom_command(
  #   TARGET ${name} POST_BUILD
  #   COMMAND ${CMAKE_COMMAND} ARGS -E copy $<TARGET_FILE_NAME:${name}> ${folder})
  # $<TARGET_FILE_NAME:${name}> does not seem to work.
  # The following step adds a line in '_setup.txt' to tell setup.py
  # to copy an additional file.
  # if (provider STREQUAL "CUDA")
  #   file(APPEND "../_setup_ext.txt" "copy,${cuda_name},${folder}\n")
  # endif()
  file(APPEND "../_setup_ext.txt" "copy,${name},${folder}\n")
endfunction()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Ort
  VERSION_VAR ORT_VERSION
  REQUIRED_VARS ORT_URL ONNXRUNTIME_INCLUDE_DIR ONNXRUNTIME_LIB_DIR
                ORT_LIB_FILES ORT_LIB_HEADER ORT_VERSION_INT)

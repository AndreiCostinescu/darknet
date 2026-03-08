# findCUDNN.cmake
#
# Usage:
#   find_package(cudnn REQUIRED)
#   if (cudnn_FOUND)
#     target_include_directories(myTarget PRIVATE ${cudnn_INCLUDE_DIRS})
#     target_link_libraries(myTarget PRIVATE ${cudnn_LIBRARIES})
#     # version available in cudnn_VERSION (string) and cudnn_VERSION_MAJOR (int)
#   endif()
#
# This module tries to locate cuDNN for both:
#  - cuDNN < 8: single library (e.g., libcudnn.so / cudnn.lib)
#  - cuDNN >= 8: split libraries (e.g., libcudnn_adv_infer.so, libcudnn_ops_infer.so, libcudnn_ops_train.so, ...)
#
# It sets:
#  cudnn_FOUND        - TRUE if found
#  cudnn_INCLUDE_DIRS - include directory
#  cudnn_LIBRARIES    - list of libraries to link
#  cudnn_VERSION      - version string (if discoverable)
#  cudnn_VERSION_MAJOR - major version as integer (if discoverable)
#
# Notes:
#  - This module uses find_library/find_path and environment variables (CUDNN_ROOT, CUDNN_HOME).
#  - It prefers the user's CUDA installation if CUDA_TOOLKIT_ROOT_DIR is set.
#  - On Windows, it looks for cudnn.lib import libraries; on *nix, it looks for libcudnn.so (or .dylib).
#  - If exact version detection from headers is not possible, version variables may be empty.

cmake_minimum_required(VERSION 3.0)
include(CheckIncludeFileCXX)

# Helper: join list arguments into a semicolon-separated string and return via output var
function(cudnn_join out_var)
  set(_tmp "")
  foreach(_i IN LISTS ARGN)
    if(_tmp STREQUAL "")
      set(_tmp "${_i}")
    else()
      set(_tmp "${_tmp};${_i}")
    endif()
  endforeach()
  set(${out_var} "${_tmp}" PARENT_SCOPE)
endfunction()

if (NOT DEFINED CUDNN_ROOT)
  if (DEFINED ENV{CUDNN_ROOT})
    set(CUDNN_ROOT $ENV{CUDNN_ROOT})
  elseif (DEFINED ENV{CUDNN_HOME})
    set(CUDNN_ROOT $ENV{CUDNN_HOME})
  endif()
endif()

# Candidate include dirs
set(_candidates_include_list "")
if (CUDNN_ROOT)
  cudnn_join(_tmp
          "${CUDNN_ROOT}/include"
          "${CUDNN_ROOT}/cuda/include"
          "${CUDNN_ROOT}/targets/x86_64-linux/include"
          "${CUDNN_ROOT}/include/cudnn"
  )
  set(_candidates_include_list "${_tmp}")
endif()

if (DEFINED CUDA_TOOLKIT_ROOT_DIR)
  cudnn_join(_tmp
          "${CUDA_TOOLKIT_ROOT_DIR}/include"
          "${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include"
  )
  list(APPEND _candidates_include_list ${_tmp})
endif()

# Common system include locations
cudnn_join(_tmp
        "/usr/include"
        "/usr/local/include"
        "/usr/local/cuda/include"
        "/opt/cuda/include"
        "/opt/local/include"
)
list(APPEND _candidates_include_list ${_tmp})

# find include
set(cudnn_INCLUDE_DIR "")

foreach(dir IN LISTS _candidates_include_list)
  if (EXISTS "${dir}/cudnn.h" OR EXISTS "${dir}/cudnn_version.h")
    set(cudnn_INCLUDE_DIR "${dir}")
    break()
  endif()
endforeach()

if (NOT cudnn_INCLUDE_DIR)
  find_path(cudnn_INCLUDE_DIR
          NAMES cudnn.h cudnn_version.h
          PATHS
          ${CUDNN_ROOT}
          ${CUDA_TOOLKIT_ROOT_DIR}
          /usr/include /usr/local/include /opt/cuda/include /opt/local/include
          NO_DEFAULT_PATH
  )
endif()

# Candidate lib dirs
set(_candidates_lib_list "")
if (CUDNN_ROOT)
  cudnn_join(_tmp
          "${CUDNN_ROOT}/lib"
          "${CUDNN_ROOT}/lib64"
          "${CUDNN_ROOT}/lib/x64"
          "${CUDNN_ROOT}/cuda/lib64"
          "${CUDNN_ROOT}/targets/x86_64-linux/lib"
  )
  set(_candidates_lib_list "${_tmp}")
endif()

if (DEFINED CUDA_TOOLKIT_ROOT_DIR)
  cudnn_join(_tmp
          "${CUDA_TOOLKIT_ROOT_DIR}/lib64"
          "${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib"
  )
  list(APPEND _candidates_lib_list ${_tmp})
endif()

cudnn_join(_tmp
        "/usr/lib"
        "/usr/lib64"
        "/usr/local/lib"
        "/usr/local/lib64"
        "/usr/local/cuda/lib"
        "/usr/local/cuda/lib64"
        "/opt/cuda/lib"
        "/opt/cuda/lib64"
)
list(APPEND _candidates_lib_list ${_tmp})

# Platform-specific library names
if (WIN32)
  set(_single_lib_names cudnn)
  set(_split_lib_basenames
          cudnn_ops_infer
          cudnn_ops_train
          cudnn_cnn_infer
          cudnn_cnn_train
          cudnn_adv_infer
          cudnn_adv_train
          cudnn_static
  )
  set(_lib_prefix "")
  set(_lib_suffix ".lib")
else()
  set(_single_lib_names cudnn libcudnn)
  set(_split_lib_basenames
          cudnn_ops_infer
          cudnn_ops_train
          cudnn_cnn_infer
          cudnn_cnn_train
          cudnn_adv_infer
          cudnn_adv_train
          cudnn
  )
  set(_lib_prefix "lib")
  set(_lib_suffix ".so")
endif()

# find libraries
set(_found_single OFF)
set(_found_split OFF)
set(_found_libs "")

set(_split_found_list "")
unset(_cand_LIB CACHE)
foreach(basename IN LISTS _split_lib_basenames)
  # message(STATUS "Searching for ${basename} // ${_lib_prefix}${basename} // ${basename}${_lib_suffix} in ${_candidates_lib_list}")
  find_library(_cand_LIB
          NAMES ${basename} ${_lib_prefix}${basename} ${basename}${_lib_suffix}
          PATHS ${_candidates_lib_list}
          NO_DEFAULT_PATH
  )
  if (_cand_LIB)
    # message(STATUS "Found ${basename} in ${_cand_LIB}")
    list(APPEND _split_found_list "${_cand_LIB}")
  endif()
  unset(_cand_LIB CACHE)  # ← Add this line
endforeach()

if (_split_found_list)
  set(_found_split ON)
  set(_found_libs ${_split_found_list} ${_found_libs})
endif()

message(STATUS "Found split cuDNN library? ${_found_split}")
message(STATUS "cuDNN libraries ${_found_libs}")

unset(_cand_LIB CACHE)
foreach(_lib IN LISTS _single_lib_names)
  # message(STATUS "Searching for ${_lib} in ${_candidates_lib_list}")
  find_library(_cand_LIB
          NAMES ${_lib}
          PATHS ${_candidates_lib_list}
          NO_DEFAULT_PATH
  )
  if (_cand_LIB)
    set(_found_single ON)
    list(APPEND _found_libs "${_cand_LIB}")
    break()
  endif()
  unset(_cand_LIB CACHE)  # ← Add this line
endforeach()

message(STATUS "Found single cuDNN library? ${_found_single}")
message(STATUS "cuDNN libraries ${_found_libs}")

if (NOT _found_single AND NOT _found_split)
  unset(_cand_LIB CACHE)
  foreach(_lib IN LISTS _single_lib_names)
    find_library(_cand_LIB NAMES ${_lib})
    if (_cand_LIB)
      set(_found_single ON)
      list(APPEND _found_libs "${_cand_LIB}")
      break()
    endif()
    unset(_cand_LIB CACHE)  # ← Add this line
  endforeach()
endif()

if (_found_single OR _found_split)
  set(cudnn_FOUND TRUE)
  if (cudnn_INCLUDE_DIR)
    set(cudnn_INCLUDE_DIRS "${cudnn_INCLUDE_DIR}")
  endif()
  set(cudnn_LIBRARIES "${_found_libs}")

  # Try to detect version from header
  set(cudnn_VERSION "")
  set(cudnn_VERSION_MAJOR "")
  if (cudnn_INCLUDE_DIRS)
    if (EXISTS "${cudnn_INCLUDE_DIRS}/cudnn_version.h")
      file(READ "${cudnn_INCLUDE_DIRS}/cudnn_version.h" _cudnn_ver_text)
      string(REGEX MATCH "CUDNN_MAJOR[ \t]*([0-9]+)" _m_major "${_cudnn_ver_text}")
      if (_m_major)
        string(REGEX REPLACE "CUDNN_MAJOR[ \t]*([0-9]+).*" "\\1" cudnn_VERSION_MAJOR "${_m_major}")
      endif()
      string(REGEX MATCH "CUDNN_MINOR[ \t]*([0-9]+)" _m_minor "${_cudnn_ver_text}")
      if (_m_minor)
        string(REGEX REPLACE "CUDNN_MINOR[ \t]*([0-9]+).*" "\\1" cudnn_VERSION_MINOR "${_m_minor}")
      endif()
      string(REGEX MATCH "CUDNN_PATCHLEVEL[ \t]*([0-9]+)" _m_patch "${_cudnn_ver_text}")
      if (_m_patch)
        string(REGEX REPLACE "CUDNN_PATCHLEVEL[ \t]*([0-9]+).*" "\\1" cudnn_VERSION_PATCH "${_m_patch}")
      endif()

      if (cudnn_VERSION_MAJOR)
        if (DEFINED cudnn_VERSION_MINOR AND DEFINED cudnn_VERSION_PATCH)
          set(cudnn_VERSION "${cudnn_VERSION_MAJOR}.${cudnn_VERSION_MINOR}.${cudnn_VERSION_PATCH}")
        elseif (DEFINED cudnn_VERSION_MINOR)
          set(cudnn_VERSION "${cudnn_VERSION_MAJOR}.${cudnn_VERSION_MINOR}")
        else()
          set(cudnn_VERSION "${cudnn_VERSION_MAJOR}")
        endif()
      endif()
    elseif (EXISTS "${cudnn_INCLUDE_DIRS}/cudnn.h")
      file(READ "${cudnn_INCLUDE_DIRS}/cudnn.h" _cudnn_h_text)
      string(REGEX MATCH "CUDNN_MAJOR[ \t]*([0-9]+)" _m_major "${_cudnn_h_text}")
      if (_m_major)
        string(REGEX REPLACE "CUDNN_MAJOR[ \t]*([0-9]+).*" "\\1" cudnn_VERSION_MAJOR "${_m_major}")
      endif()
      string(REGEX MATCH "CUDNN_MINOR[ \t]*([0-9]+)" _m_minor "${_cudnn_h_text}")
      if (_m_minor)
        string(REGEX REPLACE "CUDNN_MINOR[ \t]*([0-9]+).*" "\\1" cudnn_VERSION_MINOR "${_m_minor}")
      endif()
      string(REGEX MATCH "CUDNN_PATCHLEVEL[ \t]*([0-9]+)" _m_patch "${_cudnn_h_text}")
      if (_m_patch)
        string(REGEX REPLACE "CUDNN_PATCHLEVEL[ \t]*([0-9]+).*" "\\1" cudnn_VERSION_PATCH "${_m_patch}")
      endif()

      if (cudnn_VERSION_MAJOR)
        if (DEFINED cudnn_VERSION_MINOR AND DEFINED cudnn_VERSION_PATCH)
          set(cudnn_VERSION "${cudnn_VERSION_MAJOR}.${cudnn_VERSION_MINOR}.${cudnn_VERSION_PATCH}")
        elseif (DEFINED cudnn_VERSION_MINOR)
          set(cudnn_VERSION "${cudnn_VERSION_MAJOR}.${cudnn_VERSION_MINOR}")
        else()
          set(cudnn_VERSION "${cudnn_VERSION_MAJOR}")
        endif()
      endif()
    endif()
  endif()

  if (cudnn_VERSION)
    set(cudnn_VERSION "${cudnn_VERSION}" CACHE STRING "cuDNN version found" FORCE)
  endif()
  if (cudnn_VERSION_MAJOR)
    string(REGEX REPLACE "^[ \t]*" "" _tmp "${cudnn_VERSION_MAJOR}")
    set(cudnn_VERSION_MAJOR "${_tmp}" CACHE STRING "cuDNN major version" FORCE)
  endif()

  if (NOT TARGET CuDNN::CuDNN)
    add_library(CuDNN::CuDNN INTERFACE IMPORTED)
    if (cudnn_INCLUDE_DIRS)
      target_include_directories(CuDNN::CuDNN INTERFACE ${cudnn_INCLUDE_DIRS})
    endif()
    if (cudnn_LIBRARIES)
      target_link_libraries(CuDNN::CuDNN INTERFACE ${cudnn_LIBRARIES})
    endif()
  endif()

else()
  set(cudnn_FOUND FALSE)
  set(cudnn_INCLUDE_DIRS "")
  set(cudnn_LIBRARIES "")
  set(cudnn_VERSION "")
  set(cudnn_VERSION_MAJOR "")
endif()

if (cudnn_FOUND)
  set(CUDNN_FOUND ${cudnn_FOUND} CACHE BOOL "cuDNN found" FORCE)
  set(CUDNN_INCLUDE_DIRS ${cudnn_INCLUDE_DIRS} CACHE PATH "cuDNN include dir" FORCE)
  set(CUDNN_LIBRARIES ${cudnn_LIBRARIES} CACHE STRING "cuDNN libraries" FORCE)
  if (cudnn_VERSION)
    set(CUDNN_VERSION ${cudnn_VERSION} CACHE STRING "cuDNN version" FORCE)
  endif()
endif()

if (cudnn_FOUND)
  message(STATUS "Found cuDNN: include=${cudnn_INCLUDE_DIRS} libs=${cudnn_LIBRARIES} version=${cudnn_VERSION}")
else()
  message(STATUS "cuDNN not found")
endif()

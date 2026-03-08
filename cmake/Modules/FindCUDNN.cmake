# Distributed under the OSI-approved BSD 3-Clause License.
# Copyright Stefano Sinigardi

#.rst:
# FindCUDNN
# --------
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module will set the following variables in your project::
#
#  ``CUDNN_FOUND``
#    True if CUDNN found on the local system
#
#  ``CUDNN_INCLUDE_DIRS``
#    Location of CUDNN header files.
#
#  ``CUDNN_LIBRARIES``
#    The CUDNN libraries.
#

include(FindPackageHandleStandardArgs)

if(NOT CUDNN_INCLUDE_DIR)
  find_path(CUDNN_INCLUDE_DIR cudnn.h
    HINTS ${CUDA_HOME} ${CUDA_TOOLKIT_ROOT_DIR} $ENV{cudnn} $ENV{CUDNN}
    PATH_SUFFIXES cuda/include include)
endif()

if(NOT CUDNN_LIBRARY)
  find_library(CUDNN_LIBRARY cudnn
    HINTS ${CUDA_HOME} ${CUDA_TOOLKIT_ROOT_DIR} $ENV{cudnn} $ENV{CUDNN}
    PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)
endif()

function(read_header_version HEADER_PATH PREFIX)
  # Read file
  file(READ "${HEADER_PATH}" _header_contents)

  # Try to extract major/minor/patch; return success if MAJOR found
  string(REGEX MATCH "define[ \t]+CUDNN_MAJOR[ \t]+([0-9]+)" _match_major "${_header_contents}")
  if(_match_major)
    string(REGEX REPLACE "define[ \t]+CUDNN_MAJOR[ \t]+([0-9]+)" "\\1" "${PREFIX}_VERSION_MAJOR" "${_match_major}")
  else()
    set("${PREFIX}_VERSION_MAJOR" "" PARENT_SCOPE)
    return()
  endif()

  string(REGEX MATCH "define[ \t]+CUDNN_MINOR[ \t]+([0-9]+)" _match_minor "${_header_contents}")
  if(_match_minor)
    string(REGEX REPLACE "define[ \t]+CUDNN_MINOR[ \t]+([0-9]+)" "\\1" "${PREFIX}_VERSION_MINOR" "${_match_minor}")
  else()
    set("${PREFIX}_VERSION_MINOR" "" PARENT_SCOPE)
  endif()

  string(REGEX MATCH "define[ \t]+CUDNN_PATCHLEVEL[ \t]+([0-9]+)" _match_patch "${_header_contents}")
  if(_match_patch)
    string(REGEX REPLACE "define[ \t]+CUDNN_PATCHLEVEL[ \t]+([0-9]+)" "\\1" "${PREFIX}_VERSION_PATCH" "${_match_patch}")
  else()
    set("${PREFIX}_VERSION_PATCH" "" PARENT_SCOPE)
  endif()

  # Export results to caller scope
  set("${PREFIX}_VERSION_MAJOR" "${${PREFIX}_VERSION_MAJOR}" PARENT_SCOPE)
  set("${PREFIX}_VERSION_MINOR" "${${PREFIX}_VERSION_MINOR}" PARENT_SCOPE)
  set("${PREFIX}_VERSION_PATCH" "${${PREFIX}_VERSION_PATCH}" PARENT_SCOPE)

  return()
endfunction()


if(EXISTS "${CUDNN_INCLUDE_DIR}/cudnn.h")
  read_header_version("${CUDNN_INCLUDE_DIR}/cudnn.h" "CUDNN")
  if(CUDNN_VERSION_MAJOR)
    set(CUDNN_VERSION "${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}")
  elseif(EXISTS "${CUDNN_INCLUDE_DIR}/cudnn_version.h")
    read_header_version("${CUDNN_INCLUDE_DIR}/cudnn_version.h" "CUDNN")
    if(CUDNN_VERSION_MAJOR)
      set(CUDNN_VERSION "${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}")
    else()
      set(CUDNN_VERSION "?")
    endif()
  else()
    set(CUDNN_VERSION "?")
  endif()
endif()

message(${CUDNN_INCLUDE_DIR})
message(${CUDNN_LIBRARY})
message(${CUDNN_LIBRARIES})

set(CUDNN_INCLUDE_DIRS ${CUDNN_INCLUDE_DIR})
set(CUDNN_LIBRARIES ${CUDNN_LIBRARY})
mark_as_advanced(CUDNN_LIBRARY CUDNN_INCLUDE_DIR)

find_package_handle_standard_args(CUDNN
      REQUIRED_VARS  CUDNN_INCLUDE_DIR CUDNN_LIBRARY
      VERSION_VAR    CUDNN_VERSION
)

if(WIN32)
  set(CUDNN_DLL_DIR ${CUDNN_INCLUDE_DIR})
  list(TRANSFORM CUDNN_DLL_DIR APPEND "/../bin")
  find_file(CUDNN_LIBRARY_DLL NAMES cudnn64_${CUDNN_VERSION_MAJOR}.dll PATHS ${CUDNN_DLL_DIR})
endif()

if( CUDNN_FOUND AND NOT TARGET CuDNN::CuDNN )
  if( EXISTS "${CUDNN_LIBRARY_DLL}" )
    add_library( CuDNN::CuDNN      SHARED IMPORTED )
    set_target_properties( CuDNN::CuDNN PROPERTIES
      IMPORTED_LOCATION                 "${CUDNN_LIBRARY_DLL}"
      IMPORTED_IMPLIB                   "${CUDNN_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES     "${CUDNN_INCLUDE_DIR}"
      IMPORTED_LINK_INTERFACE_LANGUAGES "C" )
  else()
    add_library( CuDNN::CuDNN      UNKNOWN IMPORTED )
    set_target_properties( CuDNN::CuDNN PROPERTIES
      IMPORTED_LOCATION                 "${CUDNN_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES     "${CUDNN_INCLUDE_DIR}"
      IMPORTED_LINK_INTERFACE_LANGUAGES "C" )
  endif()
endif()

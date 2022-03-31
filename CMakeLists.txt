cmake_minimum_required(VERSION 3.12)

set(Darknet_MAJOR_VERSION 0)
set(Darknet_MINOR_VERSION 2)
set(Darknet_PATCH_VERSION 5)
set(Darknet_TWEAK_VERSION 3)
set(Darknet_VERSION ${Darknet_MAJOR_VERSION}.${Darknet_MINOR_VERSION}.${Darknet_PATCH_VERSION}.${Darknet_TWEAK_VERSION})

project(Darknet VERSION ${Darknet_VERSION})

include(cmake/darknetOptions.cmake)

### Make relative paths absolute (needed later on)

foreach (p LIB BIN INCLUDE CMAKE)
    set(var INSTALL_${p}_DIR)
    if (NOT IS_ABSOLUTE "${${var}}")
        set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
    endif ()
endforeach ()

configure_file(
        "${CMAKE_CURRENT_LIST_DIR}/include/darknet/version.h.in"
        "${CMAKE_CURRENT_LIST_DIR}/include/darknet/version.h"
)

### Collect all source files

file(GLOB sources_src_c "${CMAKE_CURRENT_LIST_DIR}/src/*.c")
list(REMOVE_ITEM sources_src_c ${CMAKE_CURRENT_LIST_DIR}/src/darknet.c)  # only needed for executable and added explicitly there
file(GLOB sources_src_cpp "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")
list(REMOVE_ITEM sources_src_cpp ${CMAKE_CURRENT_LIST_DIR}/src/yolo_v2_class.cpp)  # not needed for executable but for library
list(REMOVE_ITEM sources_src_cpp ${CMAKE_CURRENT_LIST_DIR}/src/yolo_console_dll.cpp)  # needed only for uselib
SET(SOURCES_SRC ${sources_src_c} ${sources_src_cpp})

file(GLOB sources_src_applications_c "${CMAKE_CURRENT_LIST_DIR}/src/applications/*.c")
list(REMOVE_ITEM sources_src_applications_c ${CMAKE_CURRENT_LIST_DIR}/src/applications/demo_realsense.c)
file(GLOB sources_src_applications_cpp "${CMAKE_CURRENT_LIST_DIR}/src/applications/*.cpp")
SET(SOURCES_SRC_APPLICATIONS ${sources_src_applications_c} ${sources_src_applications_cpp})

file(GLOB sources_src_images_c "${CMAKE_CURRENT_LIST_DIR}/src/images/*.c")
file(GLOB sources_src_images_cpp "${CMAKE_CURRENT_LIST_DIR}/src/images/*.cpp")
list(REMOVE_ITEM sources_src_images_cpp ${CMAKE_CURRENT_LIST_DIR}/src/images/image_opencv_realsense.cpp)
SET(SOURCES_SRC_IMAGES ${sources_src_images_c} ${sources_src_images_cpp})

file(GLOB sources_src_layers_c "${CMAKE_CURRENT_LIST_DIR}/src/layers/*.c")
file(GLOB sources_src_layers_cpp "${CMAKE_CURRENT_LIST_DIR}/src/layers/*.cpp")
SET(SOURCES_SRC_LAYERS ${sources_src_layers_c} ${sources_src_layers_cpp})

file(GLOB sources_src_utils_c "${CMAKE_CURRENT_LIST_DIR}/src/utils/*.c")
if (NOT WIN32)
    list(REMOVE_ITEM sources_src_utils_c ${CMAKE_CURRENT_LIST_DIR}/src/utils/gettimeofday.c ${CMAKE_CURRENT_LIST_DIR}/src/utils/getopt.c)
endif ()
file(GLOB sources_src_utils_cpp "${CMAKE_CURRENT_LIST_DIR}/src/utils/*.cpp")
SET(SOURCES_SRC_UTILS ${sources_src_utils_c} ${sources_src_utils_cpp})

SET(SOURCES ${SOURCES_SRC} ${SOURCES_SRC_APPLICATIONS} ${SOURCES_SRC_IMAGES} ${SOURCES_SRC_LAYERS} ${SOURCES_SRC_UTILS})

SET(REALSENSE_SOURCES src/applications/demo_realsense.c src/images/image_opencv_realsense.cpp ${REALSENSE_SOURCES})

### Collect all cuda files

if (ENABLE_CUDA)
    file(GLOB cuda_sources_cuda_cu "${CMAKE_CURRENT_LIST_DIR}/cuda/*.cu")
    SET(CUDA_SOURCES_CUDA ${cuda_sources_cuda_cu})

    file(GLOB cuda_sources_cuda_layers_cu "${CMAKE_CURRENT_LIST_DIR}/cuda/layers/*.cu")
    SET(CUDA_SOURCES_CUDA_LAYERS ${cuda_sources_cuda_layers_cu})

    file(GLOB cuda_sources_cuda_utils_cu "${CMAKE_CURRENT_LIST_DIR}/cuda/utils/*.cu")
    SET(CUDA_SOURCES_CUDA_UTILS ${cuda_sources_cuda_utils_cu})

    SET(CUDA_SOURCES ${CUDA_SOURCES_CUDA} ${CUDA_SOURCES_CUDA_LAYERS} ${CUDA_SOURCES_CUDA_UTILS})
    message("WITH CUDA!!!")
else ()
    message("NO CUDA!!!")
endif ()

message("${SOURCES}")
message("${CUDA_SOURCES}")
message("${REALSENSE_SOURCES}")

### Set source files as c++/cpp files

if (BUILD_AS_CPP)
    set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)
    set_source_files_properties(${REALSENSE_SOURCES} PROPERTIES LANGUAGE CXX)
endif ()

### SET INCLUDE DIRECTORIES

include_directories(include)

### TARGET DARK ###

add_library(dark ${CMAKE_CURRENT_LIST_DIR}/src/yolo_v2_class.cpp ${SOURCES} ${CUDA_SOURCES})
set_target_properties(dark PROPERTIES POSITION_INDEPENDENT_CODE ON)
if (ENABLE_CUDA)
    set_target_properties(dark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif ()
if (BUILD_SHARED_LIBS)
    target_compile_definitions(dark PRIVATE LIB_EXPORTS=1)
endif ()
if (BUILD_AS_CPP)
    set_target_properties(dark PROPERTIES LINKER_LANGUAGE CXX)
endif ()
set_target_properties(dark PROPERTIES OUTPUT_NAME "darknet")
target_include_directories(dark PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/cuda>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${Stb_INCLUDE_DIR}>)

### TARGET DARK_REALSENSE ###

if (ENABLE_REALSENSE)
    add_library(dark_realsense ${REALSENSE_SOURCES})
    set_target_properties(dark_realsense PROPERTIES POSITION_INDEPENDENT_CODE ON)
    if (ENABLE_CUDA)
        set_target_properties(dark_realsense PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    endif ()
    if (BUILD_SHARED_LIBS)
        target_compile_definitions(dark_realsense PRIVATE LIB_EXPORTS=1)
    endif ()
    if (BUILD_AS_CPP)
        set_target_properties(dark_realsense PROPERTIES LINKER_LANGUAGE CXX)
    endif ()
    set_target_properties(dark_realsense PROPERTIES OUTPUT_NAME "darknet_realsense")
    target_include_directories(dark_realsense PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/cuda>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>
            $<INSTALL_INTERFACE:include>
            $<BUILD_INTERFACE:${Stb_INCLUDE_DIR}>)
endif ()

### TARGET uselib_track ###

if (OpenCV_FOUND AND OpenCV_VERSION VERSION_GREATER "3.0" AND BUILD_USELIB_TRACK)
    add_executable(uselib_track ${CMAKE_CURRENT_LIST_DIR}/src/yolo_console_dll.cpp)
endif ()

### TARGET uselib ###

add_executable(uselib ${CMAKE_CURRENT_LIST_DIR}/src/yolo_console_dll.cpp)
if (BUILD_AS_CPP)
    set_target_properties(uselib PROPERTIES LINKER_LANGUAGE CXX)
endif ()
target_include_directories(uselib PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/cuda>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${Stb_INCLUDE_DIR}>)

### TARGET DARKNET ###

add_executable(darknet ${CMAKE_CURRENT_LIST_DIR}/src/darknet.c ${SOURCES} ${CUDA_SOURCES})
if (BUILD_AS_CPP)
    set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/src/darknet.c PROPERTIES LANGUAGE CXX)
    set_target_properties(darknet PROPERTIES LINKER_LANGUAGE CXX)
endif ()
target_include_directories(darknet PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/cuda>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${Stb_INCLUDE_DIR}>)

### FINAL THINGS: link libraries

include(cmake/darknetLinkLibraries.cmake)

### FINAL THINGS: set install paths

include(cmake/darknetInstall.cmake)

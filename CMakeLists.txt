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

SET(SOURCES_SRC src/dark_cuda.c src/network.c)

SET(SOURCES_SRC_IMAGES src/images/image.c src/images/http_stream.cpp)
SET(SOURCES_SRC_IMAGES_OPENCV src/images/image_opencv.cpp)
SET(SOURCES_SRC_IMAGES_REALSENSE src/images/image_opencv_realsense.cpp)

SET(SOURCES_SRC_LAYERS
        src/layers/activation_layer.c
        src/layers/activations.c
        src/layers/avgpool_layer.c
        src/layers/batchnorm_layer.c
        src/layers/box.c
        src/layers/col2im.c
        src/layers/connected_layer.c
        src/layers/conv_lstm_layer.c
        src/layers/convolutional_layer.c
        src/layers/cost_layer.c
        src/layers/crnn_layer.c
        src/layers/crop_layer.c
        src/layers/deconvolutional_layer.c
        src/layers/detection_layer.c
        src/layers/dropout_layer.c
        src/layers/gaussian_yolo_layer.c
        src/layers/gru_layer.c
        src/layers/identity_layer.c
        src/layers/im2col.c
        src/layers/layer.c
        src/layers/linear_layer.c
        src/layers/local_layer.c
        src/layers/lstm_layer.c
        src/layers/maxpool_layer.c
        src/layers/normalization_layer.c
        src/layers/prelu_layer.c
        src/layers/region_layer.c
        src/layers/reorg_layer.c
        src/layers/reorg_old_layer.c
        src/layers/rnn_layer.c
        src/layers/route_layer.c
        src/layers/sam_layer.c
        src/layers/scale_channels_layer.c
        src/layers/shortcut_layer.c
        src/layers/softmax_layer.c
        src/layers/upsample_layer.c
        src/layers/yolo_layer.c
        )

SET(SOURCES_SRC_UTILS
        src/utils/blas.c
        src/utils/cpu_gemm.c
        src/utils/data.c
        src/utils/gemm.c
        src/utils/list.c
        src/utils/matrix.c
        src/utils/option_list.c
        src/utils/parser.c
        src/utils/tree.c
        src/utils/utils.c
        )
if (WIN32)
    set(SOURCES_SRC_UTILS src/utils/gettimeofday.c src/utils/getopt.c ${SOURCES_SRC_UTILS})
endif ()

SET(SOURCES ${SOURCES_SRC} ${SOURCES_SRC_IMAGES} ${SOURCES_SRC_LAYERS} ${SOURCES_SRC_UTILS})

SET(REALSENSE_SOURCES src/applications/demo_realsense.c src/images/image_opencv_realsense.cpp ${REALSENSE_SOURCES})

SET(OPENCV_SOURCES
        src/darknet_opencv.c
        src/network_opencv.c
        src/layers/convolutional_layer_opencv.c
        src/images/image_opencv.cpp src/images/http_stream_opencv.cpp
        src/utils/data_opencv.c
        ${OPENCV_SOURCES})

SET(APPLICATION_SOURCES
        src/applications/art.c
        src/applications/captcha.c
        src/applications/cifar.c
        src/applications/classifier.c
        src/applications/coco.c
        src/applications/compare.c
        src/applications/demo.c
        src/applications/detector.c
        src/applications/dice.c
        src/applications/go.c
        src/applications/nightmare.c
        src/applications/rnn.c
        src/applications/rnn_vid.c
        src/applications/super.c
        src/applications/swag.c
        src/applications/tag.c
        src/applications/voxel.c
        src/applications/writing.c
        src/applications/yolo.c
        ${APPLICATION_SOURCES})

### Collect all cuda files

if (ENABLE_CUDA)
    SET(CUDA_SOURCES_CUDA_LAYERS
            cuda/layers/activation_kernels.cu
            cuda/layers/avgpool_layer_kernels.cu
            cuda/layers/col2im_kernels.cu
            cuda/layers/convolutional_kernels.cu
            cuda/layers/crop_layer_kernels.cu
            cuda/layers/deconvolutional_kernels.cu
            cuda/layers/dropout_layer_kernels.cu
            cuda/layers/identity_layer_kernels.cu
            cuda/layers/im2col_kernels.cu
            cuda/layers/linear_layer_kernels.cu
            cuda/layers/maxpool_layer_kernels.cu
            cuda/layers/prelu_layer_kernels.cu
            )

    SET(CUDA_SOURCES cuda/network_kernels.cu cuda/utils/blas_kernels.cu ${CUDA_SOURCES_CUDA_LAYERS})

    SET(CUDA_OPENCV_SOURCES cuda/utils/cuda_opencv.cu cuda/network_kernels_opencv.cu cuda/layers/convolutional_kernels_opencv.cu)
    message("WITH CUDA!!!")
else ()
    message("NO CUDA!!!")
endif ()

message("${SOURCES}")
message("${CUDA_SOURCES}")
message("${REALSENSE_SOURCES}")
message("${OPENCV_SOURCES}")
message("${APPLICATION_SOURCES}")

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

### TARGET DARK_OPENCV ###

if (ENABLE_OPENCV)
    add_library(dark_opencv ${OPENCV_SOURCES} ${CUDA_OPENCV_SOURCES})
    set_target_properties(dark_opencv PROPERTIES POSITION_INDEPENDENT_CODE ON)
    if (ENABLE_CUDA)
        set_target_properties(dark_opencv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    endif ()
    if (BUILD_SHARED_LIBS)
        target_compile_definitions(dark_opencv PRIVATE LIB_EXPORTS=1)
    endif ()
    if (BUILD_AS_CPP)
        set_target_properties(dark_opencv PROPERTIES LINKER_LANGUAGE CXX)
    endif ()
    set_target_properties(dark_opencv PROPERTIES OUTPUT_NAME "darknet_opencv")
    target_include_directories(dark_opencv PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/cuda>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>
            $<INSTALL_INTERFACE:include>
            $<BUILD_INTERFACE:${Stb_INCLUDE_DIR}>)
endif ()

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

add_executable(darknet ${CMAKE_CURRENT_LIST_DIR}/src/darknet.c ${APPLICATION_SOURCES})
if (BUILD_AS_CPP)
    set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/src/darknet.c ${APPLICATION_SOURCES} PROPERTIES LANGUAGE CXX)
    set_target_properties(darknet PROPERTIES LINKER_LANGUAGE CXX)
endif ()
target_include_directories(darknet PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/cuda>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${Stb_INCLUDE_DIR}>)

### TARGET DARKNET_OPENCV ###

if (ENABLE_OPENCV)
    add_executable(darknet_opencv ${CMAKE_CURRENT_LIST_DIR}/src/darknet_opencv.c ${APPLICATION_SOURCES})
    if (BUILD_AS_CPP)
        set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/src/darknet_opencv.c ${APPLICATION_SOURCES} PROPERTIES LANGUAGE CXX)
        set_target_properties(darknet_opencv PROPERTIES LINKER_LANGUAGE CXX)
    endif ()
    target_include_directories(darknet_opencv PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/cuda>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>
            $<INSTALL_INTERFACE:include>
            $<BUILD_INTERFACE:${Stb_INCLUDE_DIR}>)
endif ()

### FINAL THINGS: link libraries

include(cmake/darknetLinkLibraries.cmake)

### FINAL THINGS: set install paths

include(cmake/darknetInstall.cmake)

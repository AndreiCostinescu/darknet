option(CMAKE_VERBOSE_MAKEFILE "Create verbose makefile" OFF)
option(CUDA_VERBOSE_BUILD "Create verbose CUDA build" ON)
option(BUILD_SHARED_LIBS "Create dark as a shared library" ON)
option(BUILD_AS_CPP "Build Darknet using C++ compiler also for C files" ON)
option(BUILD_USELIB_TRACK "Build uselib_track" ON)
option(MANUALLY_EXPORT_TRACK_OPTFLOW "Manually export the TRACK_OPTFLOW=1 define" OFF)
option(ENABLE_OPENCV "Enable OpenCV integration" ON)
option(ENABLE_REALSENSE "Enable Realsense integration" ON)
option(ENABLE_CUDA "Enable CUDA support" ON)
option(ENABLE_CUDNN "Enable CUDNN" ON)
option(ENABLE_CUDNN_HALF "Enable CUDNN Half precision" ON)
option(ENABLE_ZED_CAMERA "Enable ZED Camera support" OFF)
option(ENABLE_VCPKG_INTEGRATION "Enable VCPKG integration" OFF)
set(OPENCV_VERSION "" CACHE STRING "The opencv version to use in the project")

if (ENABLE_VCPKG_INTEGRATION AND DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
    message(STATUS "VCPKG found: $ENV{VCPKG_ROOT}")
    message(STATUS "Using VCPKG integration")
endif ()

message("${CMAKE_CURRENT_LIST_DIR}")
message("${CMAKE_MODULE_PATH}")

if (WIN32 AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(USE_INTEGRATED_LIBS "TRUE" CACHE BOOL "Use libs distributed with this repo")
else ()
    set(USE_INTEGRATED_LIBS "FALSE" CACHE BOOL "Use libs distributed with this repo")
endif ()

enable_language(C)
enable_language(CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/Modules/" ${CMAKE_MODULE_PATH})

set(default_build_type "Release")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/.." CACHE PATH "Install prefix" FORCE)
endif ()

set(INSTALL_BIN_DIR "${CMAKE_CURRENT_LIST_DIR}/.." CACHE PATH "Path where exe and dll will be installed")
set(INSTALL_LIB_DIR "${CMAKE_CURRENT_LIST_DIR}/.." CACHE PATH "Path where lib will be installed")
set(INSTALL_INCLUDE_DIR "include/darknet" CACHE PATH "Path where headers will be installed")
set(INSTALL_CMAKE_DIR "share/darknet" CACHE PATH "Path where cmake configs will be installed")

if (UNIX)
    include(CheckLanguage)
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER AND ENABLE_CUDA)
        message("TRYING TO FIND CUDA")
        set(CUDA_ARCHITECTURES "Auto" CACHE STRING "\"Auto\" detects local machine GPU compute arch at runtime, \"Common\" and \"All\" cover common and entire subsets of architectures, \"Names\" is a list of architectures to enable by name, \"Numbers\" is a list of compute capabilities (version number) to enable")
        set_property(CACHE CUDA_ARCHITECTURES PROPERTY STRINGS "Auto" "Common" "All" "Kepler Maxwell Kepler+Tegra Maxwell+Tegra Pascal" "5.0 7.5")
        enable_language(CUDA)
        if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "9.0")
            message(STATUS "Unsupported CUDA version, please upgrade to CUDA 9+. Disabling CUDA support")
            set(ENABLE_CUDA "FALSE" CACHE BOOL "Enable CUDA support" FORCE)
        else ()
            message("CALLING FIND PACKAGE...")
            find_package(CUDA REQUIRED)
            cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_ARCHITECTURES})
            message(STATUS "Building with CUDA flags: " "${CUDA_ARCH_FLAGS}")
            if (NOT "arch=compute_70,code=sm_70" IN_LIST CUDA_ARCH_FLAGS AND NOT "arch=compute_72,code=sm_72" IN_LIST CUDA_ARCH_FLAGS AND NOT "arch=compute_75,code=sm_75" IN_LIST CUDA_ARCH_FLAGS AND NOT "arch=compute_80,code=sm_80" IN_LIST CUDA_ARCH_FLAGS)
                set(ENABLE_CUDNN_HALF "FALSE" CACHE BOOL "Enable CUDNN Half precision" FORCE)
                message(STATUS "Your setup does not supports half precision (it requires CC >= 7.0)")
            else ()
                message(STATUS "Your setup supports half precision (it requires CC >= 7.0)")
            endif ()
        endif ()
        if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.17")
            if (BUILD_SHARED_LIBS)
                set(CMAKE_CUDA_RUNTIME_LIBRARY "Shared")
            else ()
                set(CMAKE_CUDA_RUNTIME_LIBRARY "Static")
            endif ()
        endif ()
    else ()
        message("CAN NOT FIND CUDA!!!")
        set(ENABLE_CUDA "FALSE" CACHE BOOL "Enable CUDA support" FORCE)
    endif ()
else ()
    message("CALLING FIND PACKAGE...")
    find_package(CUDA REQUIRED)
    cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_ARCHITECTURES})
    message(STATUS "Building with CUDA flags: " "${CUDA_ARCH_FLAGS}")
    if (NOT "arch=compute_70,code=sm_70" IN_LIST CUDA_ARCH_FLAGS AND NOT "arch=compute_72,code=sm_72" IN_LIST CUDA_ARCH_FLAGS AND NOT "arch=compute_75,code=sm_75" IN_LIST CUDA_ARCH_FLAGS AND NOT "arch=compute_80,code=sm_80" IN_LIST CUDA_ARCH_FLAGS)
        set(ENABLE_CUDNN_HALF "FALSE" CACHE BOOL "Enable CUDNN Half precision" FORCE)
        message(STATUS "Your setup does not supports half precision (it requires CC >= 7.0)")
    else ()
        message(STATUS "Your setup supports half precision (it requires CC >= 7.0)")
    endif ()
    set(ENABLE_CUDA "TRUE" CACHE BOOL "Enable CUDA support" FORCE)
endif ()

if (WIN32 AND ENABLE_CUDA AND CMAKE_MAKE_PROGRAM MATCHES "ninja")
    option(SELECT_OPENCV_MODULES "Use only few selected OpenCV modules to circumvent 8192 char limit when using Ninja on Windows" ON)
else ()
    option(SELECT_OPENCV_MODULES "Use only few selected OpenCV modules to circumvent 8192 char limit when using Ninja on Windows" OFF)
endif ()

if (USE_INTEGRATED_LIBS)
    set(PThreads_windows_DIR ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/pthreads CACHE PATH "Path where pthreads for windows can be located")
endif ()
set(Stb_DIR ${CMAKE_CURRENT_LIST_DIR}/../3rdparty/stb CACHE PATH "Path where Stb image library can be located")

set(CMAKE_DEBUG_POSTFIX d)
set(CMAKE_THREAD_PREFER_PTHREAD ON)
find_package(Threads REQUIRED)
if (MSVC)
    find_package(PThreads_windows REQUIRED)
endif ()
if (ENABLE_OPENCV)
    if ("${OPENCV_VERSION}" STREQUAL "")
        find_package(OpenCV)
    else ()
        find_package(OpenCV ${OPENCV_VERSION})
    endif ()

    if (OpenCV_FOUND)
        if (SELECT_OPENCV_MODULES)
            if (TARGET opencv_world)
                list(APPEND OpenCV_LINKED_COMPONENTS "opencv_world")
            else ()
                if (TARGET opencv_core)
                    list(APPEND OpenCV_LINKED_COMPONENTS "opencv_core")
                endif ()
                if (TARGET opencv_highgui)
                    list(APPEND OpenCV_LINKED_COMPONENTS "opencv_highgui")
                endif ()
                if (TARGET opencv_imgproc)
                    list(APPEND OpenCV_LINKED_COMPONENTS "opencv_imgproc")
                endif ()
                if (TARGET opencv_video)
                    list(APPEND OpenCV_LINKED_COMPONENTS "opencv_video")
                endif ()
                if (TARGET opencv_videoio)
                    list(APPEND OpenCV_LINKED_COMPONENTS "opencv_videoio")
                endif ()
                if (TARGET opencv_imgcodecs)
                    list(APPEND OpenCV_LINKED_COMPONENTS "opencv_imgcodecs")
                endif ()
                if (TARGET opencv_text)
                    list(APPEND OpenCV_LINKED_COMPONENTS "opencv_text")
                endif ()
            endif ()
        else ()
            list(APPEND OpenCV_LINKED_COMPONENTS ${OpenCV_LIBS})
        endif ()
    else ()
        message(STATUS "Opencv was requested but could not find it...")
    endif ()
endif ()
if (ENABLE_REALSENSE)
    find_package(realsense2 REQUIRED)
    list(APPEND realsense2_LINKED_COMPONENTS ${realsense2_LIBRARY})
endif ()
find_package(Stb REQUIRED)
find_package(OpenMP)

if (APPLE AND NOT OPENMP_FOUND)
    message(STATUS "  ->  To enable OpenMP on macOS, please install libomp from Homebrew")
endif ()

set(ADDITIONAL_CXX_FLAGS "-Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -Wno-deprecated-declarations -Wno-write-strings")
set(ADDITIONAL_C_FLAGS "-Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -Wno-deprecated-declarations -Wno-write-strings")

if (MSVC)
    set(ADDITIONAL_CXX_FLAGS "/wd4013 /wd4018 /wd4028 /wd4047 /wd4068 /wd4090 /wd4101 /wd4113 /wd4133 /wd4190 /wd4244 /wd4267 /wd4305 /wd4477 /wd4996 /wd4819 /fp:fast")
    set(ADDITIONAL_C_FLAGS "/wd4013 /wd4018 /wd4028 /wd4047 /wd4068 /wd4090 /wd4101 /wd4113 /wd4133 /wd4190 /wd4244 /wd4267 /wd4305 /wd4477 /wd4996 /wd4819 /fp:fast")
    set(CMAKE_CXX_FLAGS "${ADDITIONAL_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${ADDITIONAL_C_FLAGS} ${CMAKE_C_FLAGS}")
    string(REGEX REPLACE "/O2" "/Ox" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
    string(REGEX REPLACE "/O2" "/Ox" CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
endif ()

if (CMAKE_COMPILER_IS_GNUCC OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
        if (UNIX AND NOT APPLE)
            set(CMAKE_CXX_FLAGS "-pthread ${CMAKE_CXX_FLAGS}")  #force pthread to avoid bugs in some cmake setups
            set(CMAKE_C_FLAGS "-pthread ${CMAKE_C_FLAGS}")
        endif ()
    endif ()
    set(CMAKE_CXX_FLAGS "${ADDITIONAL_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${ADDITIONAL_C_FLAGS} ${CMAKE_C_FLAGS}")
    string(REGEX REPLACE "-O0" "-Og" CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
    string(REGEX REPLACE "-O3" "-Ofast" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
    string(REGEX REPLACE "-O0" "-Og" CMAKE_C_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
    string(REGEX REPLACE "-O3" "-Ofast" CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffp-contract=fast -mavx -mavx2 -msse3 -msse4.1 -msse4.2 -msse4a")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -ffp-contract=fast -mavx -mavx2 -msse3 -msse4.1 -msse4.2 -msse4a")
endif ()

if (OpenCV_FOUND)
    if (ENABLE_CUDA AND NOT OpenCV_CUDA_VERSION)
        set(BUILD_USELIB_TRACK "FALSE" CACHE BOOL "Build uselib_track" FORCE)
        message(STATUS "  ->  darknet is fine for now, but uselib_track has been disabled!")
        message(STATUS "  ->  Please rebuild OpenCV from sources with CUDA support to enable it")
    elseif (ENABLE_CUDA AND OpenCV_CUDA_VERSION)
        if (TARGET opencv_cudaoptflow)
            list(APPEND OpenCV_LINKED_COMPONENTS "opencv_cudaoptflow")
        endif ()
        if (TARGET opencv_cudaimgproc)
            list(APPEND OpenCV_LINKED_COMPONENTS "opencv_cudaimgproc")
        endif ()
    endif ()
endif ()

if (ENABLE_CUDA)
    find_package(CUDNN)
    if (NOT CUDNN_FOUND)
        message("CUDNN NOT FOUND!!!\nALERT\nALERT\nALERT\nALERT")
        set(ENABLE_CUDNN "FALSE" CACHE BOOL "Enable CUDNN" FORCE)
    endif ()
endif ()

if (ENABLE_CUDA)
    message("ENABLE CUDA!!!")
    if (MSVC)
        set(ADDITIONAL_CXX_FLAGS "${ADDITIONAL_CXX_FLAGS} /DGPU")
        if (CUDNN_FOUND)
            set(ADDITIONAL_CXX_FLAGS "${ADDITIONAL_CXX_FLAGS} /DCUDNN")
        endif ()
        if (OpenCV_FOUND)
            set(ADDITIONAL_CXX_FLAGS "${ADDITIONAL_CXX_FLAGS} /DOPENCV")
        endif ()
        string(REPLACE " " "," ADDITIONAL_CXX_FLAGS_COMMA_SEPARATED "${ADDITIONAL_CXX_FLAGS}")
        set(CUDA_HOST_COMPILER_FLAGS "-Wno-deprecated-declarations -Xcompiler=\"${ADDITIONAL_CXX_FLAGS_COMMA_SEPARATED}\"")
    else ()
        set(ADDITIONAL_CXX_FLAGS "${ADDITIONAL_CXX_FLAGS} -DGPU")
        if (CUDNN_FOUND)
            set(ADDITIONAL_CXX_FLAGS "${ADDITIONAL_CXX_FLAGS} -DCUDNN")
        endif ()
        if (OpenCV_FOUND)
            set(ADDITIONAL_CXX_FLAGS "${ADDITIONAL_CXX_FLAGS} -DOPENCV")
        endif ()
        if (APPLE)
            set(CUDA_HOST_COMPILER_FLAGS "--compiler-options \" ${ADDITIONAL_CXX_FLAGS} -fPIC -Xpreprocessor -fopenmp -Ofast \"")
        else ()
            set(CUDA_HOST_COMPILER_FLAGS "--compiler-options \" ${ADDITIONAL_CXX_FLAGS} -fPIC -fopenmp -Ofast \"")
        endif ()
    endif ()

    string(REPLACE ";" " " CUDA_ARCH_FLAGS_SPACE_SEPARATED "${CUDA_ARCH_FLAGS}")
    set(CMAKE_CUDA_FLAGS "${CUDA_ARCH_FLAGS_SPACE_SEPARATED} ${CUDA_HOST_COMPILER_FLAGS} ${CMAKE_CUDA_FLAGS}")
    message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
endif ()

if (ENABLE_CUDA)
    if (ENABLE_ZED_CAMERA)
        find_package(ZED 2 QUIET)
        if (ZED_FOUND)
            include_directories(${ZED_INCLUDE_DIRS})
            link_directories(${ZED_LIBRARY_DIR})
            message(STATUS "ZED SDK enabled")
        else ()
            message(STATUS "ZED SDK not found")
            set(ENABLE_ZED_CAMERA "FALSE" CACHE BOOL "Enable ZED Camera support" FORCE)
        endif ()
    endif ()
else ()
    message(STATUS "ZED SDK not enabled, since it requires CUDA")
    set(ENABLE_ZED_CAMERA "FALSE" CACHE BOOL "Enable ZED Camera support" FORCE)
endif ()

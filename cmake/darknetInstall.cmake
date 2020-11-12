#set_target_properties(dark PROPERTIES PUBLIC_HEADER "${exported_headers};${CMAKE_CURRENT_LIST_DIR}/include/yolo_v2_class.hpp")
set_target_properties(dark PROPERTIES PUBLIC_HEADER
        "${CMAKE_CURRENT_LIST_DIR}/../include/darknet/darknet.h;${CMAKE_CURRENT_LIST_DIR}/../include/darknet/yolo_v2_class.hpp")

# set_target_properties(dark PROPERTIES CXX_VISIBILITY_PRESET hidden)

install(TARGETS dark EXPORT DarknetTargets
        RUNTIME DESTINATION "${INSTALL_BIN_DIR}"
        LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
        ARCHIVE DESTINATION "${INSTALL_LIB_DIR}"
        PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}"
        COMPONENT dev
        )
install(TARGETS uselib darknet
        DESTINATION "${INSTALL_BIN_DIR}"
        )
if (OpenCV_FOUND AND OpenCV_VERSION VERSION_GREATER "3.0" AND BUILD_USELIB_TRACK)
    install(TARGETS uselib_track
            DESTINATION "${INSTALL_BIN_DIR}"
            )
endif ()

install(EXPORT DarknetTargets
        FILE DarknetTargets.cmake
        NAMESPACE Darknet::
        DESTINATION "${INSTALL_CMAKE_DIR}"
        )

message("HELLO WORLD!")
# Export the package for use from the build-tree (this registers the build-tree with a global CMake-registry)
export(PACKAGE Darknet)
message("HELLO, HELLO WORLD!!!")

# Create the DarknetConfig.cmake
# First of all we compute the relative path between the cmake config file and the include path
file(RELATIVE_PATH REL_INCLUDE_DIR "${INSTALL_CMAKE_DIR}" "${INSTALL_INCLUDE_DIR}")
set(CONF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")
configure_file(DarknetConfig.cmake.in "${PROJECT_BINARY_DIR}/DarknetConfig.cmake" @ONLY)
set(CONF_INCLUDE_DIRS "\${Darknet_CMAKE_DIR}/${REL_INCLUDE_DIR}")
configure_file(DarknetConfig.cmake.in "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/DarknetConfig.cmake" @ONLY)

# Create the DarknetConfigVersion.cmake
include(CMakePackageConfigHelpers)
write_basic_package_version_file("${PROJECT_BINARY_DIR}/DarknetConfigVersion.cmake"
        COMPATIBILITY SameMajorVersion
        )

install(FILES
        "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/DarknetConfig.cmake"
        "${PROJECT_BINARY_DIR}/DarknetConfigVersion.cmake"
        DESTINATION "${INSTALL_CMAKE_DIR}"
        )

include(GNUInstallDirs)

set(CMAKECONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

add_custom_target(uninstall "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake")

include(CMakePackageConfigHelpers)

#set_target_properties(dark PROPERTIES PUBLIC_HEADER "${exported_headers};${CMAKE_CURRENT_LIST_DIR}/include/yolo_v2_class.hpp")
#set_target_properties(dark PROPERTIES PUBLIC_HEADER
#        "${CMAKE_CURRENT_LIST_DIR}/../include/darknet/darknet.h;${CMAKE_CURRENT_LIST_DIR}/../include/darknet/yolo_v2_class.hpp")
#set_target_properties(dark_realsense PROPERTIES PUBLIC_HEADER
#        "${CMAKE_CURRENT_LIST_DIR}/../include/darknet/darknet.h;${CMAKE_CURRENT_LIST_DIR}/../include/darknet/yolo_v2_class.hpp")

# set_target_properties(dark PROPERTIES CXX_VISIBILITY_PRESET hidden)
set(PROJECT_INCLUDE_PREFIX darknet)

# install include files
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/${PROJECT_INCLUDE_PREFIX}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        )

install(TARGETS dark
        EXPORT DarknetTargets
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_PREFIX}/include/${PROJECT_INCLUDE_PREFIX}"
        COMPONENT dev
        )
install(TARGETS dark_realsense
        EXPORT DarknetTargets
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_PREFIX}/include/${PROJECT_INCLUDE_PREFIX}"
        COMPONENT dev
        )
install(TARGETS uselib darknet
        DESTINATION "${CMAKE_INSTALL_BINDIR}"
        )
if (OpenCV_FOUND AND OpenCV_VERSION VERSION_GREATER "3.0" AND BUILD_USELIB_TRACK)
    install(TARGETS uselib_track
            DESTINATION "${CMAKE_INSTALL_BINDIR}"
            )
endif ()

install(EXPORT DarknetTargets
        FILE DarknetTargets.cmake
        NAMESPACE Darknet::
        DESTINATION "${CMAKECONFIG_INSTALL_DIR}"
        )

message("HELLO WORLD!")
# Export the package for use from the build-tree (this registers the build-tree with a global CMake-registry)
export(PACKAGE Darknet)
message("HELLO, HELLO WORLD!!!")

configure_file("cmake/cmakeUninstall.cmake" "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake" IMMEDIATE @ONLY)


# Create the DarknetConfig.cmake
## First of all we compute the relative path between the cmake config file and the include path
#file(RELATIVE_PATH REL_INCLUDE_DIR "${INSTALL_CMAKE_DIR}" "${INSTALL_INCLUDE_DIR}")
#set(CONF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")
#configure_file(DarknetConfig.cmake.in "${PROJECT_BINARY_DIR}/DarknetConfig.cmake" @ONLY)
#set(CONF_INCLUDE_DIRS "\${Darknet_CMAKE_DIR}/${REL_INCLUDE_DIR}")
configure_file(cmake/DarknetConfig.cmake.in "${PROJECT_BINARY_DIR}/DarknetConfig.cmake" @ONLY)

# Create the DarknetConfigVersion.cmake
include(CMakePackageConfigHelpers)
write_basic_package_version_file("${PROJECT_BINARY_DIR}/DarknetConfigVersion.cmake"
        COMPATIBILITY SameMajorVersion
        )

install(FILES
        "${PROJECT_BINARY_DIR}/DarknetConfig.cmake"
        "${PROJECT_BINARY_DIR}/DarknetConfigVersion.cmake"
        DESTINATION "${CMAKECONFIG_INSTALL_DIR}"
        )

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA software released under the NVIDIA Community License is intended to be used to enable
# the further development of AI and robotics technologies. Such software has been designed, tested,
# and optimized for use with NVIDIA hardware, and this License grants permission to use the software
# solely with such hardware.
# Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
# modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
# outputs generated using the software or derivative works thereof. Any code contributions that you
# share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
# in future releases without notice or attribution.
# By using, reproducing, modifying, distributing, performing, or displaying any portion or element
# of the software or derivative works thereof, you agree to be bound by this License.

include(ExternalProject)

# Only build if not already built
set(LIBJPEG_VERSION "3.1.3")
set(LIBJPEG_INSTALL_DIR ${CMAKE_BINARY_DIR}/third_party/_install/libjpeg-turbo-${LIBJPEG_VERSION})

# Pre-create directories so CMake doesn't complain about non-existent paths
# (ExternalProject creates files at build-time, but CMake validates at configure-time)
file(MAKE_DIRECTORY ${LIBJPEG_INSTALL_DIR})
file(MAKE_DIRECTORY ${LIBJPEG_INSTALL_DIR}/include)
file(MAKE_DIRECTORY ${LIBJPEG_INSTALL_DIR}/lib)

if(NOT TARGET libjpeg_external)
    ExternalProject_Add(libjpeg_external
        URL https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/${LIBJPEG_VERSION}/libjpeg-turbo-${LIBJPEG_VERSION}.tar.gz
        URL_HASH SHA256=075920b826834ac4ddf97661cc73491047855859affd671d52079c6867c1c6c0
        PREFIX ${CMAKE_BINARY_DIR}/third_party/libjpeg-turbo-${LIBJPEG_VERSION}
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${LIBJPEG_INSTALL_DIR}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -DENABLE_SHARED=OFF
            -DWITH_SIMD=false
        BUILD_BYPRODUCTS
            ${LIBJPEG_INSTALL_DIR}/lib/libjpeg.a
            ${LIBJPEG_INSTALL_DIR}/lib/libturbojpeg.a
    )
endif()

# Create imported target for use in main project
# TODO: must link different libs depending on a target system
if(NOT TARGET jpeg::jpeg)
    add_library(jpeg::jpeg STATIC IMPORTED GLOBAL)
    set_target_properties(jpeg::jpeg PROPERTIES
        IMPORTED_LOCATION ${LIBJPEG_INSTALL_DIR}/lib/libjpeg.a
        INTERFACE_INCLUDE_DIRECTORIES ${LIBJPEG_INSTALL_DIR}/include
    )
    add_dependencies(jpeg::jpeg libjpeg_external)
endif()

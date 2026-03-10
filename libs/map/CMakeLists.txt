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

set(MODULE_NAME "map")

set(HEADERS
    keyframe.h
    map.h
    service.h
    )

set(SOURCES
    landmark.cpp
    keyframe.cpp
    map.cpp
    service.cpp
    )

setup_staticlib_project(
        MODULE_NAME map
        HEADERS ${HEADERS}
        SOURCES ${SOURCES}
        LIBRARIES
            camera
            common
            dense_hash_map::dense_hash_map
            imu
            profiler
)

add_subdirectory(test)

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

set(MODULE_NAME "pnp")

set(HEADERS
    mono_pnp.h
    multicam_pnp.h
    )

set(SOURCES
    mono_pnp.cpp
    multicam_pnp.cpp
    )


set(LIBS
    camera
    common
    cuvslam_math
    epipolar
    profiler
)

if(USE_CUDA)
    set(HEADERS ${HEADERS} visual_icp.h)
    set(SOURCES ${SOURCES} visual_icp.cpp lift_tracks.cpp)
    set(LIBS ${LIBS} cuda_modules)
endif()

if(USE_RERUN)
    set(HEADERS ${HEADERS} visualizer.h)
    set(SOURCES ${SOURCES} visualizer.cpp)
    set(LIBS ${LIBS} visualizer)
endif()

setup_staticlib_project(
        MODULE_NAME ${MODULE_NAME}
        HEADERS ${HEADERS}
        SOURCES ${SOURCES}
        LIBRARIES ${LIBS}
)

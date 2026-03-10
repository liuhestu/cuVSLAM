
/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA software released under the NVIDIA Community License is intended to be used to enable
 * the further development of AI and robotics technologies. Such software has been designed, tested,
 * and optimized for use with NVIDIA hardware, and this License grants permission to use the software
 * solely with such hardware.
 * Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
 * modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
 * outputs generated using the software or derivative works thereof. Any code contributions that you
 * share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
 * in future releases without notice or attribution.
 * By using, reproducing, modifying, distributing, performing, or displaying any portion or element
 * of the software or derivative works thereof, you agree to be bound by this License.
 */

#pragma once

#include "common/isometry.h"
#include "common/types.h"

namespace cuvslam {

enum class CoordinateSystem { CUVSLAM = 0, ROS = 1, OPENCV = 2 };
// matrix to change basis
Matrix3T CoordinateSystemTocuVSLAM(CoordinateSystem cs);

const Isometry3T kCuvslamFromRos{CoordinateSystemTocuVSLAM(CoordinateSystem::ROS)};

const Isometry3T kRosFromCuvslam{kCuvslamFromRos.inverse()};

const Isometry3T kCuvslamFromOpencv{CoordinateSystemTocuVSLAM(CoordinateSystem::OPENCV)};

const Isometry3T kOpencvFromCuvslam{kCuvslamFromOpencv.inverse()};

inline Isometry3T CuvslamFromOpencv(const Isometry3T& isometry) {
  return kCuvslamFromOpencv * isometry * kOpencvFromCuvslam;
}

inline Isometry3T OpencvFromCuvslam(const Isometry3T& isometry) {
  return kOpencvFromCuvslam * isometry * kCuvslamFromOpencv;
}

}  // namespace cuvslam

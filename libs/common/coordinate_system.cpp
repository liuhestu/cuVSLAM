
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

#include "common/coordinate_system.h"

namespace cuvslam {

// matrix to change basis
Matrix3T CoordinateSystemTocuVSLAM(CoordinateSystem cs) {
  Matrix3T change_basis;
  switch (cs) {
    case CoordinateSystem::ROS:
      change_basis << 0.f, -1.f, 0.f, 0.f, 0.f, 1.f, -1.f, 0.f, 0.f;
      return change_basis;
    case CoordinateSystem::OPENCV:
      change_basis = Matrix3T::Identity();
      change_basis(1, 1) = -1.f;
      change_basis(2, 2) = -1.f;
      return change_basis;
    default:
      return Matrix3T::Identity();
  }
}

}  // namespace cuvslam

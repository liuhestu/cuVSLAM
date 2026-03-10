
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

#include "math/twist_to_angle.h"

#include "common/isometry.h"
#include "common/types.h"
#include "common/vector_3t.h"

#include "math/twist.h"

namespace cuvslam::math {

/*
  jacobian implies that Q = [qx, qy, qz, qw]
 */
Eigen::Matrix<float, 3, 4> roll_pitch_yaw_from_quat_jacobian(const QuaternionT& quat_) {
  Eigen::Matrix<float, 3, 4> J = Eigen::Matrix<float, 3, 4>::Zero();
  QuaternionT quat = quat_;
  if (quat.w() < 0) {
    quat = QuaternionT(-quat_.w(), -quat_.x(), -quat_.y(), -quat_.z());
  }

  float determinant = quat.w() * quat.y() - quat.x() * quat.z();

  if (std::abs(std::abs(determinant) - 0.5) > 1e-5) {
    float t1 = 1 - 2 * (quat.x() * quat.x() + quat.y() * quat.y());
    float t2 = 2 * (quat.w() * quat.x() + quat.y() * quat.z());
    float t3 = 1 / (t1 * t1 + t2 * t2);

    float t4 = 1.f / (float)sqrt(1 - 4 * pow(determinant, 2));

    float t5 = 1.f - 2.f * (quat.y() * quat.y() + quat.z() * quat.z());
    float t6 = 2.f * (quat.w() * quat.z() + quat.x() * quat.y());
    float t7 = 1 / (t5 * t5 + t6 * t6);

    J(0, 0) = (2 * quat.w() * t1 + 4 * quat.x() * t2) * t3;  // d roll / d q.x
    J(0, 1) = (2 * quat.z() * t1 + 4 * quat.y() * t2) * t3;  // d roll / d q.y
    J(0, 2) = 2 * quat.y() * t1 * t3;                        // d roll / d q.z
    J(0, 3) = 2 * quat.x() * t1 * t3;                        // d roll / d q.w

    J(1, 0) = -2 * quat.z() * t4;  // d pitch / d q.x
    J(1, 1) = 2 * quat.w() * t4;   // d pitch / d q.y
    J(1, 2) = -2 * quat.x() * t4;  // d pitch / d q.z
    J(1, 3) = 2 * quat.y() * t4;   // d pitch / d q.w

    J(2, 0) = 2 * quat.y() * t5 * t7;                        // d yaw / d q.x
    J(2, 1) = (2 * quat.x() * t5 + 4 * quat.y() * t6) * t7;  // d yaw / d q.y
    J(2, 2) = (2 * quat.w() * t5 + 4 * quat.z() * t6) * t7;  // d yaw / d q.z
    J(2, 3) = 2 * quat.z() * t5 * t7;                        // d yaw / d q.w
  } else {
    float t1 = quat.w() * quat.w();
    float t2 = quat.x() * quat.x();
    float t4 = 1.f / (t1 + t2);

    if (determinant < 0) {
      // roll  = 0
      // pitch = -M_PI/2.0
      // yaw   = 2.0 * atan2(q.x(), q.w())

      J(2, 0) = 2 * quat.w() * t4;   // d yaw / d q.x
      J(2, 3) = -2 * quat.x() * t4;  // d yaw / d q.w
    } else {
      // roll  = 0;
      // pitch = M_PI/2.0;
      // yaw   = -2.0 * atan2(q.x(), q.w());
      J(2, 0) = -2 * quat.w() * t4;  // d yaw / d q.x
      J(2, 3) = 2 * quat.x() * t4;   // d yaw / d q.w
    }
  }

  return J;
}

/*
  jacobian implies that Q = [qx, qy, qz, qw]
 */
Eigen::Matrix<float, 4, 3> quat_from_twist_jacobian(const Vector3T& twist) {
  float phi_norm = twist.norm();
  if (phi_norm < 1e-2) {
    Eigen::Matrix<float, 4, 3> out;
    out.block<3, 3>(0, 0) = 0.5 * Matrix3T::Identity() - twist * twist.transpose() / 24.f;
    out.block<1, 3>(3, 0) = -twist / 4.f;
    return out;
  }

  float s = sinf(phi_norm * 0.5);
  float c = cosf(phi_norm * 0.5);
  float m = (0.5 * c / phi_norm - s / powf(phi_norm, 2));

  Matrix4T J1 = Matrix4T::Zero();
  J1.block<3, 3>(0, 0) = Matrix3T::Identity() * s / phi_norm;
  J1.block<3, 1>(0, 3) = twist * m;
  J1(3, 3) = -s * 0.5;

  Eigen::Matrix<float, 4, 3> J2;
  J2.block<3, 3>(0, 0) = Matrix3T::Identity();
  J2.block<1, 3>(3, 0) = twist / phi_norm;
  return J1 * J2;
}

// TODO: get rid of it, when IMU code is merged
Matrix3T translation_from_twist_jacobian(const Vector6T& twist) {
  float phi_norm = twist.head(3).norm();
  Vector3T a = twist.head(3) / phi_norm;

  if (phi_norm < std::numeric_limits<float>::epsilon()) {
    return Matrix3T::Identity();
  } else {
    float s = (float)sin(phi_norm) / phi_norm;
    float c = (1 - (float)cos(phi_norm)) / phi_norm;
    return s * Matrix3T ::Identity() + (1 - s) * a * a.transpose() + c * SkewSymmetric(a);
  }
}

Matrix6T PoseCovToRollPitchYawCov(const Matrix6T& pose_cov, const Isometry3T& pose) {
  QuaternionT quat(pose.linear());
  quat.normalize();

  Vector6T twist;
  Log(twist, pose);

  Eigen::Matrix<float, 4, 3> J_qt = quat_from_twist_jacobian(twist.head(3));

  Matrix6T J = Matrix6T::Zero();

  J.block<3, 3>(0, 0) = roll_pitch_yaw_from_quat_jacobian(quat) * J_qt;

  J.block<3, 3>(3, 3) = translation_from_twist_jacobian(twist);

  return J * pose_cov * J.transpose();
}

}  // namespace cuvslam::math

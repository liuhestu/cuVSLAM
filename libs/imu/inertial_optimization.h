
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

#include "common/unaligned_types.h"
#include "profiler/profiler.h"
#include "profiler/profiler_enable.h"

#include "imu/imu_sba_problem.h"

namespace cuvslam::sba_imu {

class InertialOptimizer {
public:
  explicit InertialOptimizer(float gyro_prior = 1e2, float acc_prior = 1e10)
      : gyro_prior_info(Matrix3T::Identity() * gyro_prior), acc_prior_info(Matrix3T::Identity() * acc_prior) {
    JG << 0, -GRAVITY_VALUE, 0, 0, GRAVITY_VALUE, 0;
  }
  bool optimize_inertial(std::vector<Pose>& poses, Matrix3T& Rgravity, float robustifier = 1e1);

  [[nodiscard]] const Vector3T& get_default_gravity() const { return default_gravity; }

private:
  float calc_cost_with_update(const std::vector<Pose>& poses, const Matrix3T& Rgravity, const Vector3T& gyro_bias,
                              const Vector3T& acc_bias, const Eigen::VectorXf& updates, float robustifier) const;

  void build_hessian(const std::vector<Pose>& poses, const Matrix3T& Rgravity, const Vector3T& gyro_bias,
                     const Vector3T& acc_bias, float robustifier, Eigen::MatrixXf& hessian, Eigen::VectorXf& rhs) const;

  const Vector3T default_gravity = {0, -GRAVITY_VALUE, 0};
  Eigen::Matrix<float, 3, 2> JG;
  const Matrix3T gyro_prior_info;
  const Matrix3T acc_prior_info;

  profiler::GravityProfiler::DomainHelper profiler_domain_ = profiler::GravityProfiler::DomainHelper("Gravity");
};

}  // namespace cuvslam::sba_imu

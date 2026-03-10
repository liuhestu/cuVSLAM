
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

#include "math/robust_cost_function.h"

#include "imu/inertial_optimization.h"

namespace {

using Mat93 = Eigen::Matrix<float, 9, 3>;
using Mat92 = Eigen::Matrix<float, 9, 2>;

using Mat39 = Eigen::Matrix<float, 3, 9>;
using Mat29 = Eigen::Matrix<float, 2, 9>;

struct InertialJacobians {
  Mat92 J_gravity = Mat92::Zero();
  Mat93 J_ba = Mat93::Zero();
  Mat93 J_bg = Mat93::Zero();

  Mat93 J_v1 = Mat93::Zero();
  Mat93 J_v2 = Mat93::Zero();
};

}  // namespace

namespace cuvslam::sba_imu {

float InertialOptimizer::calc_cost_with_update(const std::vector<Pose>& poses, const Matrix3T& Rgravity,
                                               const Vector3T& gyro_bias, const Vector3T& acc_bias,
                                               const Eigen::VectorXf& updates, float robustifier) const {
  TRACE_EVENT ev = profiler_domain_.trace_event("calc_cost_with_update");

  Matrix3T Rguess;
  Vector3T twist = {updates[0], 0, updates[1]};
  math::Exp(Rguess, twist);

  Vector3T gravity = Rgravity * Rguess * default_gravity;

  Vector3T gyro_guess = gyro_bias + updates.segment<3>(2);
  Vector3T acc_guess = acc_bias + updates.segment<3>(5);

  Vector3T v1, v2;

  float cost = 0;

  Vector9T err;
  Matrix9T info;
  for (size_t i = 0; i < poses.size() - 1; i++) {
    const Pose& pose_left = poses[i];
    const Pose& pose_right = poses[i + 1];

    const auto& preint = pose_left.preintegration;
    const Isometry3T& w_from_imu1 = pose_left.w_from_imu;
    const Isometry3T& w_from_imu2 = pose_right.w_from_imu;

    const Matrix3T dR = preint.GetDeltaRotation(gyro_guess);
    const Vector3T dV = preint.GetDeltaVelocity(gyro_guess, acc_guess);
    const Vector3T dP = preint.GetDeltaPosition(gyro_guess, acc_guess);
    const float dT = preint.GetDeltaT_s();

    v1 = pose_left.velocity + updates.segment<3>(8 + 3 * i);
    v2 = pose_right.velocity + updates.segment<3>(11 + 3 * i);

    const Matrix3T R1T = w_from_imu1.linear().transpose();
    const Matrix3T& R2 = w_from_imu2.linear();

    Vector3T rot_error;
    math::Log(rot_error, dR.transpose() * R1T * R2);

    err.segment<3>(0) = rot_error;
    err.segment<3>(3) = R1T * (v2 - v1 - gravity * dT) - dV;
    err.segment<3>(6) =
        R1T * (w_from_imu2.translation() - w_from_imu1.translation() - v1 * dT - 0.5 * gravity * dT * dT) - dP;

    preint.InfoMatrix(info);

    float squared_err = err.dot(info * err);
    // TODO add robust cost
    cost += math::ComputeHuberLoss(squared_err, robustifier);
  }

  // priors
  cost += gyro_guess.dot(gyro_prior_info * gyro_guess);
  cost += acc_guess.dot(acc_prior_info * acc_guess);

  return cost;
}

void InertialOptimizer::build_hessian(const std::vector<Pose>& poses, const Matrix3T& Rgravity,
                                      const Vector3T& gyro_bias, const Vector3T& acc_bias, float robustifier,
                                      Eigen::MatrixXf& hessian, Eigen::VectorXf& rhs) const {
  TRACE_EVENT ev = profiler_domain_.trace_event("build_hessian");
  int num_poses = static_cast<int>(poses.size());
  int num_inertials = num_poses - 1;
  // 3 - for each velocity in each pose, 2 - for gravity, 3 - for gyro bias, 3 - for acc bias
  hessian.setZero();
  rhs.setZero();

  Vector3T rot_error, velocity_error, trans_error;
  Vector3T gravity = Rgravity * default_gravity;

  Vector9T inertial_residual = Vector9T::Zero();
  InertialJacobians inertial_jacobians;

  hessian.block<3, 3>(2, 2) += gyro_prior_info;
  hessian.block<3, 3>(5, 5) += acc_prior_info;
  rhs.segment<3>(2) += gyro_prior_info * gyro_bias;
  rhs.segment<3>(5) += acc_prior_info * acc_bias;

  Vector3T gyro_bias_diff, acc_bias_diff;

  Matrix2T h_rr;
  Eigen::Matrix<float, 2, 3> h_rbg, h_rba, h_rv1, h_rv2;

  Matrix3T h_bgbg, h_bgba, h_bgv1, h_bgv2, h_baba, h_bav1, h_bav2, h_v1v1, h_v1v2, h_v2v2;

  Matrix9T info;
  for (int i = 0; i < num_inertials; i++) {
    const Pose& pose_left = poses[i];
    const Pose& pose_right = poses[i + 1];
    const auto& preint = pose_left.preintegration;
    const Isometry3T& w_from_imu1 = pose_left.w_from_imu;
    const Isometry3T& w_from_imu2 = pose_right.w_from_imu;

    const Matrix3T dR = preint.GetDeltaRotation(gyro_bias);
    const Vector3T dV = preint.GetDeltaVelocity(gyro_bias, acc_bias);
    const Vector3T dP = preint.GetDeltaPosition(gyro_bias, acc_bias);
    const float dT = preint.GetDeltaT_s();
    preint.InfoMatrix(info);

    const Matrix3T R1T = w_from_imu1.linear().transpose();
    const Matrix3T& R2 = w_from_imu2.linear();

    math::Log(rot_error, dR.transpose() * R1T * R2);

    inertial_residual.segment<3>(0) = rot_error;

    inertial_residual.segment<3>(3) = R1T * (pose_right.velocity - pose_left.velocity - gravity * dT) - dV;
    inertial_residual.segment<3>(6) = R1T * (w_from_imu2.translation() - w_from_imu1.translation() -
                                             pose_left.velocity * dT - 0.5 * gravity * dT * dT) -
                                      dP;

    {
      inertial_jacobians.J_gravity.setZero();
      // vel
      inertial_jacobians.J_gravity.block<3, 2>(3, 0) = R1T * Rgravity * JG * dT;
      // tr
      inertial_jacobians.J_gravity.block<3, 2>(6, 0) = 0.5 * R1T * Rgravity * JG * dT * dT;
    }

    {
      gyro_bias_diff = gyro_bias - preint.GetOriginalGyroBias();
      // rot
      inertial_jacobians.J_bg.block<3, 3>(0, 0) = -math::twist_left_inverse_jacobian(rot_error) *
                                                  math::twist_right_jacobian(preint.JRg * gyro_bias_diff) * preint.JRg;
      // vel
      inertial_jacobians.J_bg.block<3, 3>(3, 0) = -preint.JVg;
      // tr
      inertial_jacobians.J_bg.block<3, 3>(6, 0) = -preint.JPg;
    }

    {
      inertial_jacobians.J_ba.setZero();
      // vel
      inertial_jacobians.J_ba.block<3, 3>(3, 0) = -preint.JVa;
      // tr
      inertial_jacobians.J_ba.block<3, 3>(6, 0) = -preint.JPa;
    }
    {
      inertial_jacobians.J_v1.setZero();
      // vel
      inertial_jacobians.J_v1.block<3, 3>(3, 0) = -R1T;
      // tr
      inertial_jacobians.J_v1.block<3, 3>(6, 0) = -R1T * dT;
    }

    {
      inertial_jacobians.J_v2.setZero();
      // vel
      inertial_jacobians.J_v2.block<3, 3>(3, 0) = R1T;
    }

    float squared_err = inertial_residual.dot(info * inertial_residual);
    float w = math::ComputeDHuberLoss(squared_err, robustifier);
    info *= w;

    {
      Mat29 temp = inertial_jacobians.J_gravity.transpose() * info;
      h_rr = temp * inertial_jacobians.J_gravity;
      h_rbg = temp * inertial_jacobians.J_bg;
      h_rba = temp * inertial_jacobians.J_ba;
      h_rv1 = temp * inertial_jacobians.J_v1;
      h_rv2 = temp * inertial_jacobians.J_v2;

      rhs.segment<2>(0) += temp * inertial_residual;
    }

    {
      Mat39 temp = inertial_jacobians.J_bg.transpose() * info;
      h_bgbg = temp * inertial_jacobians.J_bg;
      h_bgba = temp * inertial_jacobians.J_ba;
      h_bgv1 = temp * inertial_jacobians.J_v1;
      h_bgv2 = temp * inertial_jacobians.J_v2;

      rhs.segment<3>(2) += temp * inertial_residual;
    }

    {
      Mat39 temp = inertial_jacobians.J_ba.transpose() * info;
      h_baba = temp * inertial_jacobians.J_ba;
      h_bav1 = temp * inertial_jacobians.J_v1;
      h_bav2 = temp * inertial_jacobians.J_v2;

      rhs.segment<3>(5) += temp * inertial_residual;
    }

    {
      Mat39 temp = inertial_jacobians.J_v1.transpose() * info;
      h_v1v1 = temp * inertial_jacobians.J_v1;
      h_v1v2 = temp * inertial_jacobians.J_v2;

      rhs.segment<3>(8 + 3 * i) += temp * inertial_residual;
    }

    h_v2v2 = inertial_jacobians.J_v2.transpose() * info * inertial_jacobians.J_v2;
    rhs.segment<3>(11 + 3 * i) += inertial_jacobians.J_v2.transpose() * info * inertial_residual;

    hessian.block<2, 2>(0, 0) += h_rr;

    hessian.block<3, 2>(2, 0) += h_rbg.transpose();
    hessian.block<3, 3>(2, 2) += h_bgbg;

    hessian.block<3, 2>(5, 0) += h_rba.transpose();
    hessian.block<3, 3>(5, 2) += h_bgba.transpose();
    hessian.block<3, 3>(5, 5) += h_baba;

    hessian.block<3, 2>(8 + 3 * i, 0) += h_rv1.transpose();
    hessian.block<3, 3>(8 + 3 * i, 2) += h_bgv1.transpose();
    hessian.block<3, 3>(8 + 3 * i, 5) += h_bav1.transpose();
    hessian.block<3, 3>(8 + 3 * i, 8 + 3 * i) += h_v1v1;

    hessian.block<3, 2>(11 + 3 * i, 0) += h_rv2.transpose();
    hessian.block<3, 3>(11 + 3 * i, 2) += h_bgv2.transpose();
    hessian.block<3, 3>(11 + 3 * i, 5) += h_bav2.transpose();
    hessian.block<3, 3>(11 + 3 * i, 8 + 3 * i) += h_v1v2.transpose();
    hessian.block<3, 3>(11 + 3 * i, 11 + 3 * i) += h_v2v2;
  }
}

bool InertialOptimizer::optimize_inertial(std::vector<Pose>& poses, Matrix3T& Rgravity, float robustifier) {
  TRACE_EVENT ev = profiler_domain_.trace_event("optimize_inertial");

  if (poses.size() < 2) {
    return false;
  }
  int num_poses = static_cast<int>(poses.size());

  Rgravity.setIdentity();
  Vector3T gyro_bias = Vector3T::Zero();
  Vector3T acc_bias = Vector3T::Zero();

  Eigen::MatrixXf hessian;
  // 3 - for each velocity in each pose, 2 - for gravity, 3 - for gyro bias, 3 - for acc bias
  hessian.setZero(3 * num_poses + 8, 3 * num_poses + 8);

  Eigen::VectorXf negative_gradient;
  negative_gradient.setZero(3 * num_poses + 8);

  auto initial_cost = calc_cost_with_update(poses, Rgravity, gyro_bias, acc_bias, negative_gradient, robustifier);
  if (initial_cost < 1e-5) {
    return true;
  }
  auto current_cost = initial_cost;

  build_hessian(poses, Rgravity, gyro_bias, acc_bias, robustifier, hessian, negative_gradient);

  Eigen::VectorXf scaling = hessian.diagonal();

  float lambda = 1.f;

  int32_t num_iterations = 0;

  const int max_iterations = 8;
  do {
    ++num_iterations;
    Eigen::MatrixXf augmented_system = hessian + (lambda * scaling).asDiagonal().toDenseMatrix();

    TRACE_EVENT ev2 = profiler_domain_.trace_event("solve");
    Eigen::LDLT<Eigen::MatrixXf, Eigen::Lower> decomposition(augmented_system);
    Eigen::VectorXf step = decomposition.solve(-negative_gradient);
    ev2.Pop();

    auto cost = calc_cost_with_update(poses, Rgravity, gyro_bias, acc_bias, step, robustifier);
    auto predicted_relative_reduction =
        step.dot(hessian * step) / current_cost + 2.f * lambda * step.dot(scaling.asDiagonal() * step) / current_cost;

    if ((predicted_relative_reduction < sqrt_epsilon()) && (step.template lpNorm<1>() < sqrt_epsilon())) {
      current_cost = cost;

      for (Pose& p : poses) {
        p.gyro_bias = gyro_bias;
        p.acc_bias = acc_bias;
      }
      break;
    }

    auto rho = (1.f - cost / current_cost) / predicted_relative_reduction;

    // we have achieved sufficient decrease
    if (rho > 0.25f) {
      // accept step

      {
        TRACE_EVENT ev1 = profiler_domain_.trace_event("update");
        Matrix3T R;
        Vector3T twist = {step[0], 0, step[1]};
        math::Exp(R, twist);
        Rgravity = Rgravity * R;

        gyro_bias += step.segment<3>(2);
        acc_bias += step.segment<3>(5);

        {
          TRACE_EVENT ev2 = profiler_domain_.trace_event("update velocity");
          for (int id = 0; id < num_poses; id++) {
            Pose& p = poses[id];
            p.velocity += step.segment<3>(8 + 3 * id);
          }
        }

        TRACE_EVENT ev2 = profiler_domain_.trace_event("set new bias");
        for (int id = 0; id < num_poses - 1; id++) {
          Pose& p = poses[id];
          p.preintegration.SetNewBias(gyro_bias, acc_bias);
        }
      }

      // our model is good
      if (rho > 0.75f) {
        if (lambda * 0.125f > 0.f) {
          lambda *= 0.5f;
        }
      }

      current_cost = cost;

      build_hessian(poses, Rgravity, gyro_bias, acc_bias, robustifier, hessian, negative_gradient);

      scaling = scaling.cwiseMax(hessian.diagonal());
    } else {
      lambda *= 5.f;
    }
  } while (num_iterations < max_iterations);

  if (current_cost < initial_cost) {
    for (Pose& p : poses) {
      p.gyro_bias = gyro_bias;
      p.acc_bias = acc_bias;
    }
  }

  return current_cost < initial_cost;
}

}  // namespace cuvslam::sba_imu

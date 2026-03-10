
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

#include "pipelines/track_online_inertial.h"

#include "camera/observation.h"
#include "camera/rig.h"
#include "common/frame_id.h"
#include "common/include_eigen.h"
#include "common/isometry.h"
#include "common/vector_2t.h"
#include "common/vector_3t.h"
#include "imu/imu_sba_problem.h"
#include "imu/inertial_optimization.h"
#include "sof/sof_create.h"

#include "pipelines/inertial_pnp.h"
#include "pipelines/service_sba.h"
#include "pipelines/track.h"
#ifdef USE_CUDA
#include "pipelines/service_sba_gpu.h"
#endif

namespace cuvslam::pipelines {

using Vector15T = Eigen::Matrix<float, 15, 1>;

SolverSfMInertial::SolverSfMInertial(map::UnifiedMap& map, const camera::Rig& rig, const sba::Settings& sba_settings,
                                     const imu::ImuCalibration& calib, bool debug_imu_mode,
                                     bool disable_fusion_except_gravity)
    : imu_storage_(1e5),
      calib_(calib),
      rig_(rig),
      debug_imu_mode_(debug_imu_mode),
      disable_fusion_except_gravity_(disable_fusion_except_gravity),
      map_(map),
      optimizer_(1e2, 1e6),
      stereo_pnp_(rig, pnp::PNPSettings::InertialSettings()),
      triangulator(rig) {
  sba::Mode sba_mode;
  if (disable_fusion_except_gravity_ && rig_.num_cameras > 2) {
    sba_mode = sba::OriginalGPU;
  } else {
    sba_mode = sba_settings.mode;  // InertialCPU
  }
  if (sba_mode != sba::InertialCPU && sba_mode != sba::InertialGPU && sba_mode != sba::Disabled) {  // Add GPU version
    TraceDebug("Cant launch fusion with regular SBA");
  }

  switch (sba_mode) {
    case sba::InertialCPU:
      sba_service_ = std::make_unique<ImuSbaCPUService>(sba_settings, rig, calib, map_);
      break;
    case sba::OriginalCPU:
      sba_service_ = std::make_unique<CpuSbaService>(sba_settings, rig, map_);
      break;
#ifdef USE_CUDA
    case sba::InertialGPU:
      sba_service_ = std::make_unique<ImuSbaGPUService>(sba_settings, rig, calib, map_);
      break;
    case sba::OriginalGPU:
      sba_service_ = std::make_unique<GpuSbaService>(sba_settings, rig, map_);
      break;
#endif
    default:
      sba_service_ = nullptr;
      break;
  }

  const Isometry3T& rig_from_imu = calib_.rig_from_imu();
  prev_pose.w_from_imu = rig_from_imu;

  auto gravity_estimation_callback = [this](size_t num_kfs) {
    TRACE_EVENT ev = profiler_domain_.trace_event("gravity_estimation_callback");
    auto recent_map = map_.get_recent_submap(num_kfs);

    if (recent_map.consecutive_keyframes.size() < num_kfs) {
      TraceDebug("Not enough keyframes in map, (%d, %d)", map_.size(), num_kfs);
      return false;
    }
    const Isometry3T& rig_from_imu = calib_.rig_from_imu();
    Matrix3T Rgravity = Matrix3T::Identity();
    std::vector<sba_imu::Pose> gravity_cache;

    for (const auto& kf : recent_map.consecutive_keyframes) {
      State s = kf.keyframe->get_state();
      gravity_cache.push_back({s.rig_from_w.inverse() * rig_from_imu, s.velocity, s.gyro_bias, s.acc_bias,
                               kf.preintegration ? *kf.preintegration : IMUPreintegration()});
    }
    if (optimizer_.optimize_inertial(gravity_cache, Rgravity, 1e-3)) {
      TRACE_EVENT ev1 = profiler_domain_.trace_event("update map");
      const Vector3T gravity = Rgravity * optimizer_.get_default_gravity();

      {
        // update biases
        const Vector3T& gyro_bias = gravity_cache[0].gyro_bias;
        const Vector3T& acc_bias = gravity_cache[0].acc_bias;

        int id = 0;
        for (const auto& kf : recent_map.consecutive_keyframes) {
          const sba_imu::Pose& pose = gravity_cache[id];

          kf.keyframe->set_gyro_bias(gyro_bias);
          kf.keyframe->set_acc_bias(acc_bias);
          kf.keyframe->set_velocity(pose.velocity);

          if (kf.preintegration) {
            kf.preintegration->SetNewBias(gyro_bias, acc_bias);
            kf.preintegration->Reintegrate(calib_);
          }

          id++;
        }

        curr_pose.gyro_bias = gyro_bias;
        curr_pose.acc_bias = acc_bias;
        curr_pose.preintegration.SetNewBias(gyro_bias, acc_bias);
        curr_pose.preintegration.Reintegrate(calib_);

        // TODO: check update biases here
        prev_pose.gyro_bias = gyro_bias;
        prev_pose.acc_bias = acc_bias;
        prev_pose.preintegration.SetNewBias(gyro_bias, acc_bias);
        prev_pose.preintegration.Reintegrate(calib_);

        integ_kf.gyro_bias = gyro_bias;
        integ_kf.acc_bias = acc_bias;
        integ_kf.preintegration.SetNewBias(gyro_bias, acc_bias);
        integ_kf.preintegration.Reintegrate(calib_);
      }

      map_.set_gravity(gravity);
      TraceDebug("Gravity set to {%f, %f, %f}", gravity.x(), gravity.y(), gravity.z());

      // TODO: update biases here
      return true;
    }
    return false;
  };
  imu_sm_.register_gravity_estimation_callback(gravity_estimation_callback);
}

// OUT: pose, prev_pose - updated poses if success, otherwise it's guaranteed to be unchanged
bool runInertialPnP(const imu::ImuCalibration& calib, const InertialPnP& solver,
                    const std::unordered_map<TrackId, Track>& tracks3d,
                    const std::vector<camera::Observation>& observations, const camera::Rig& rig,
                    const Vector3T& gravity,
                    sba_imu::Pose& prev_pose,  // non-const because of velocities updates
                    sba_imu::Pose& curr_pose) {
  return solver.Solve(calib, tracks3d, observations, rig, gravity, prev_pose, curr_pose);
}

// OUT: curr_pose - current pose if success otherwise curr_pose is unchanged
bool runStereoPnP(pnp::PNPSolver& solver, const std::unordered_map<TrackId, Track>& tracks3d,
                  const std::vector<camera::Observation>& observations, const Isometry3T& imu_from_rig,
                  sba_imu::Pose& prev_pose,  // non-const because of velocities updates
                  sba_imu::Pose& curr_pose) {
  /* logic here:
   * world_from_rig = prev_pose.w_from_imu * imu_from_rig;
   * world_from_rig = prev_rig_from_world_.inverse();
   * prev_pose.w_from_imu * imu_from_rig = prev_rig_from_world_.inverse()
   * prev_rig_from_world_ = (prev_pose.w_from_imu * imu_from_rig).inverse()  */

  std::unordered_map<TrackId, Vector3T> landmarks;
  for (const auto& [track_id, track] : tracks3d) {
    if (track.hasLocation()) {
      landmarks.insert({track_id, track.getLocation3D()});
    }
  }

  Isometry3T rig_from_world = (prev_pose.w_from_imu * imu_from_rig).inverse();
  Matrix6T info;
  const bool result = solver.solve(rig_from_world, info, observations, landmarks);

  if (result) {
    /* logic here:
     * prev_pose.w_from_imu * imu_from_rig = rig_from_world.inverse()
     * prev_pose.w_from_imu = rig_from_world.inverse() * imu_from_rig.inverse() */
    curr_pose.w_from_imu = rig_from_world.inverse() * imu_from_rig.inverse();
    curr_pose.info.setZero();
    curr_pose.info.block<6, 6>(0, 0) = info;
  }

  return result;
}

const camera::Rig& SolverSfMInertial::getRig() const { return rig_; }

void SolverSfMInertial::reset() {
  imu_storage_.clear();
  triangulator.reset();
  if (sba_service_) {
    sba_service_->restart();
  }
  imu_sm_.reset();
  is_first_run = true;

  prev_pose_ts_ns = -1;

  curr_pose.preintegration.Initialize(Vector3T::Zero(), Vector3T::Zero());
  prev_pose.preintegration.Initialize(Vector3T::Zero(), Vector3T::Zero());

  last_valid_pose.preintegration.Initialize(Vector3T::Zero(), Vector3T::Zero());
  integ_kf.preintegration.Initialize(Vector3T::Zero(), Vector3T::Zero());

  last_kf_preint.Initialize(Vector3T::Zero(), Vector3T::Zero());
  last_frame_preint_.Initialize(Vector3T::Zero(), Vector3T::Zero());
}

bool SolverSfMInertial::predict_pose(Isometry3T& pose) const {
  (void)pose;
  return false;
}

void SolverSfMInertial::set_verbose(bool verbose) { verbose_ = verbose; }

bool check_imu_drops(sba_imu::IMUPreintegration& preint, const imu::ImuCalibration& calib, int64_t start_time_ns,
                     int64_t end_time_ns) {
  int preint_size = static_cast<int>(preint.size());
  float dT_s = static_cast<float>((end_time_ns - start_time_ns) * 1e-9f);

  int correct_imu_size = static_cast<int>(calib.frequency() * 0.95 * dT_s);

  size_t delta = std::max(correct_imu_size - preint_size, 0);

  float drop_ratio = static_cast<float>(delta) / static_cast<float>(correct_imu_size);

  if (drop_ratio > 0.1) {
    TraceWarning("Lost IMU msgs: %d, Frame time delta = %f [s], drop ratio = %f [%]", delta, dT_s, drop_ratio * 100);
    return false;
  }
  return true;
}

bool SolverSfMInertial::solveNextFrame(int64_t time_ns, const sof::FrameState& frameState,
                                       const MulticamObservations& observations, Isometry3T& world_from_rig,
                                       Matrix6T& static_info_exp, std::vector<Track2D>* tracks2d,
                                       Tracks3DMap* tracks3d) {
  TRACE_EVENT ev = profiler_domain_.trace_event("solveNextFrame()");

  const Isometry3T& rig_from_imu = calib_.rig_from_imu();
  if (is_first_run) {
    Vector15T diag = Vector15T::Zero();
    diag.segment<6>(0).setConstant(1e6);  // prev_pose is fixed
    // diag.segment<6>(9).setConstant(1e6); // biases are fixed
    prev_pose.info = diag.asDiagonal();  // first pose is fixed, but velocities and biases are free

    last_valid_pose = prev_pose;
    integ_kf = prev_pose;

    last_valid_pose.preintegration = last_frame_preint_;
    integ_kf.preintegration = last_frame_preint_;
    last_kf_preint = last_frame_preint_;
  }

  prev_pose.preintegration = last_frame_preint_;

  curr_pose = prev_pose;
  // TODO: use last_valid_kf_pose to integrate pose, not last
  // find 2 last tracking_state with optimized_by_sba = true and timedelta > 0.5 s,
  // then compare it with last_kf_valid_pose.frameid and update it

  const Isometry3T imu_from_rig = rig_from_imu.inverse();

  bool pnp_result = false;
  StateMachine::State imu_state;
  if (disable_fusion_except_gravity_ && rig_.num_cameras > 2) {
    imu_state = StateMachine::State::Uninitialized;
  } else {
    imu_state = imu_sm_.get_state();
  }

  world_from_rig = prev_pose.w_from_imu * imu_from_rig;
  Isometry3T rig_from_w = world_from_rig.inverse();

  static_info_exp.setZero();
  std::optional<Vector3T> maybe_gravity = map_.get_gravity();

  if (debug_imu_mode_) {
    last_valid_pose.predict_pose(*maybe_gravity, last_valid_pose.preintegration, curr_pose);

    curr_pose.w_from_imu.translation().setZero();
    curr_pose.velocity.setZero();

    if (is_first_run) {
      is_first_run = false;
    }
    std::swap(prev_pose, curr_pose);
    return true;
  }

  // TODO: refactor the code to use std::vector<std::reference_wrapper>
  std::vector<camera::Observation> obs_vector;
  for (const auto& [cam_id, obs] : observations) {
    std::copy(obs.begin(), obs.end(), std::back_inserter(obs_vector));
  }

  bool no_drops = check_imu_drops(prev_pose.preintegration, calib_, prev_pose_ts_ns, time_ns);

  if (!map_.empty()) {
    std::unordered_map<TrackId, Track> landmarks;
    {
      auto map_landmarks = map_.get_recent_landmarks();
      for (const auto& [track_id, point_3d] : map_landmarks) {
        Track track;
        track.setLocation3D(point_3d, TrackState::kTriangulated);
        landmarks.insert({track_id, track});
      }
    }

    if (imu_state == StateMachine::State::Ok) {
      assert(maybe_gravity != std::nullopt);
      TRACE_EVENT ev2 = profiler_domain_.trace_event("runInertialPnP");
      pnp_result = runInertialPnP(calib_, inertial_pnp_, landmarks, obs_vector, rig_, *maybe_gravity, prev_pose,
                                  curr_pose);  // prev_pose is also updated here (if success)
                                               //            TraceMessage("runInertialPnP result: %d", pnp_result);
    } else {
      {
        TRACE_EVENT ev2 = profiler_domain_.trace_event("runStereoPnP");
        pnp_result = runStereoPnP(stereo_pnp_, landmarks, obs_vector, imu_from_rig, prev_pose, curr_pose);
      }
      if (pnp_result) {
        curr_pose.gyro_bias = prev_pose.gyro_bias;
        curr_pose.acc_bias = prev_pose.acc_bias;
        if (!is_first_run) {
          float dt = static_cast<float>((time_ns - prev_pose_ts_ns) * 1e-9);
          Vector3T dp = (prev_pose.w_from_imu * curr_pose.w_from_imu.inverse()).translation();
          curr_pose.velocity = dp / dt;
        } else {
          curr_pose.velocity.setZero();
        }
      }
      //            TraceMessage("runStereoPnP result: %d", pnp_result);
    }
  }

  prev_pose_ts_ns = time_ns;

  {
    TRACE_EVENT ev2 = profiler_domain_.trace_event("update_frame_state");
    bool is_keyframe = frameState == sof::FrameState::Key;
    imu_sm_.update_frame_state(is_keyframe, pnp_result && no_drops,
                               time_ns);  // estimates gravity and biases through callback
  }

  if (pnp_result && imu_state == StateMachine::State::Ok) {
    // inertial pnp succeeded
    last_valid_pose = curr_pose;
    last_valid_pose.preintegration = sba_imu::IMUPreintegration(curr_pose.gyro_bias, curr_pose.acc_bias);
  }

  integrated = !pnp_result && imu_state == StateMachine::State::Ok;
  if (!map_.empty() && !integrated) {
    auto recent_map = map_.get_recent_submap(1);

    auto it = recent_map.consecutive_keyframes.rbegin();

    State s = it->keyframe->get_state();

    sba_imu::IMUPreintegration preint(calib_, imu_storage_, s.gyro_bias, s.acc_bias, it->keyframe->time_ns(), time_ns);

    integ_kf = {s.rig_from_w.inverse() * rig_from_imu, s.velocity, s.gyro_bias, s.acc_bias, preint};
  }

  if (integrated) {
    integ_kf.predict_pose(*maybe_gravity, integ_kf.preintegration, curr_pose);

    TraceDebug("Pose was integrated!");

    // investigate integration pose update here
    integ_kf = curr_pose;

    // TODO: investigate integration further.
    // need to be sure, that all biases are calculated correctly
    // maybe better integrate imu based of keyframe, not last frame
  }
  if (pnp_result || imu_state == StateMachine::State::Ok) {
    // either we successfully converged, or successfully integrated the pose
    world_from_rig = curr_pose.w_from_imu * imu_from_rig;
    rig_from_w = world_from_rig.inverse();
    static_info_exp = curr_pose.info.block<6, 6>(0, 0);
    std::swap(prev_pose, curr_pose);
  }
  last_frame_preint_ = sba_imu::IMUPreintegration(prev_pose.gyro_bias, prev_pose.acc_bias);

  bool lost = !is_first_run && !pnp_result && !integrated;
  if (lost) {
    return false;
  }

  if (is_first_run) {
    is_first_run = false;
  }

  if (frameState == sof::FrameState::Key) {
    auto tr_landmarks = triangulator.triangulate(world_from_rig, obs_vector);

    State state = {rig_from_w, !lost ? prev_pose.velocity : last_valid_pose.velocity,
                   !lost ? prev_pose.acc_bias : last_valid_pose.acc_bias,
                   !lost ? prev_pose.gyro_bias : last_valid_pose.gyro_bias};
    map_.add_keyframe(time_ns, state,
                      last_kf_preint,  // preintegration
                      obs_vector, tr_landmarks);
    if (sba_service_) {
      sba_service_->notify();
    }
    if (!lost) {
      last_kf_preint = sba_imu::IMUPreintegration(prev_pose.gyro_bias, prev_pose.acc_bias);
    } else {
      last_kf_preint = sba_imu::IMUPreintegration(last_valid_pose.gyro_bias, last_valid_pose.acc_bias);
    }
  }

  if (tracks2d && tracks3d) {
    exportTracks(obs_vector, *tracks2d, *tracks3d, rig_from_w);
  }

  return !lost;
}

// Exports observations in left camera along with corresponding 3d points
// out_tracks2d - output 2d track coordinates in pixels
// out_tracks3d - in rig space
void SolverSfMInertial::exportTracks(const std::vector<camera::Observation>& observations,
                                     std::vector<Track2D>& out_tracks2d, Tracks3DMap& out_tracks3d,
                                     const Isometry3T& rig_from_world) const {
  out_tracks2d.clear();
  out_tracks3d.clear();

  // export 2d tracks
  for (const camera::Observation& obs : observations) {
    const ICameraModel& camera = *rig_.intrinsics[obs.cam_id];
    Vector2T uv;  // in pixels
    if (camera.denormalizePoint(obs.xy, uv)) {
      out_tracks2d.push_back({obs.cam_id, obs.id, uv});
    }
  }

  // export 3d tracks
  auto map_landmarks = map_.get_recent_landmarks();
  for (const camera::Observation& obs : observations) {
    if (map_landmarks.find(obs.id) != map_landmarks.end()) {
      const Vector3T& point_3d = map_landmarks.at(obs.id);
      out_tracks3d[obs.id] = rig_from_world * point_3d;
    }
  }
}

void SolverSfMInertial::add_imu_measurement(const imu::ImuMeasurement& m) {
  imu_storage_.push_back(m);

  last_frame_preint_.IntegrateNewMeasurement(calib_, m);
  last_valid_pose.preintegration.IntegrateNewMeasurement(calib_, m);
  last_kf_preint.IntegrateNewMeasurement(calib_, m);
  integ_kf.preintegration.IntegrateNewMeasurement(calib_, m);
}

std::optional<Vector3T> SolverSfMInertial::get_gravity() const {
  auto gravity = map_.get_gravity();
  if (!gravity) {
    return std::nullopt;
  }
  Isometry3T rig_from_w = calib_.rig_from_imu() * prev_pose.w_from_imu.inverse();  // compare with curr_pose.w_from_imu

  Vector3T gravity_rig = rig_from_w.linear() * (*gravity);

  return gravity_rig;
}

}  // namespace cuvslam::pipelines

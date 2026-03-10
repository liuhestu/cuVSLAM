
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

#include <optional>

#include "camera/rig.h"
#include "common/log.h"
#include "sba/sba_config.h"
#include "sba/schur_complement_bundler_cpu.h"

#ifdef USE_CUDA
#include "sba/schur_complement_bundler_gpu.h"
#endif

#include "imu/imu_sba.h"
#include "map/service.h"

#include "pipelines/winsorizer.h"

namespace cuvslam::pipelines {
using namespace cuvslam::map;

int CalcNumFixedKeyframes(size_t map_size, size_t numFixedKeyFrames);

namespace {
template <class Bundler>
void run_sba(const UnifiedMap::SubMap& recent_map, const camera::Rig& rig, const sba::Settings& sba_settings,
             Bundler& bundler) {
  sba::BundleAdjustmentProblem problem;
  problem.rig = rig;

  // relax all points
  problem.num_fixed_points = 0;

  // we want this many fixed key frames
  problem.num_fixed_key_frames =
      CalcNumFixedKeyframes(recent_map.consecutive_keyframes.size(), sba_settings.num_fixed_sba_frames);

  if (problem.num_fixed_key_frames < 1) {
    return;
  }

  // We will count observations for each point and will skip
  // points with insufficient number of observations.
  std::unordered_map<LandmarkPtr, int> pointObservationCount;
  size_t total_observarions = 0;

  problem.points.reserve(1e4);

  // count observations for each 3D point
  {
    for (const auto& landmarks : recent_map.landmark_and_obs) {
      for (const auto& [landmark, obs] : landmarks) {
        auto it = pointObservationCount.find(landmark);
        if (it == pointObservationCount.end()) {
          int point_id = static_cast<int>(problem.points.size());
          pointObservationCount.insert(it, {landmark, point_id});
          Vector3T point_w = *landmark->get_pose();
          problem.points.push_back(point_w);
        }
        total_observarions += obs.size();
      }
    }
  }

  // remaps track ids into consecutive indices to give to bundler
  std::unordered_map<KeyframePtr, int32_t> keyframe_to_pose_id;

  for (size_t i = problem.num_fixed_key_frames; i < recent_map.consecutive_keyframes.size(); i++) {
    const auto& kf = recent_map.consecutive_keyframes[i].keyframe;

    int pose_id = keyframe_to_pose_id.size();
    keyframe_to_pose_id.insert({kf, pose_id});

    problem.rig_from_world.push_back(kf->get_pose());
  }

  for (size_t i = 0; i < (size_t)problem.num_fixed_key_frames; i++) {
    const auto& kf = recent_map.consecutive_keyframes[i].keyframe;

    int pose_id = keyframe_to_pose_id.size();
    keyframe_to_pose_id.insert({kf, pose_id});

    problem.rig_from_world.push_back(kf->get_pose());
  }

  {
    problem.observation_xys.reserve(total_observarions);
    problem.observation_infos.reserve(total_observarions);
    problem.point_ids.reserve(total_observarions);
    problem.pose_ids.reserve(total_observarions);
    problem.camera_ids.reserve(total_observarions);
  }

  for (size_t i = 0; i < recent_map.consecutive_keyframes.size(); i++) {
    const auto& kf = recent_map.consecutive_keyframes[i].keyframe;
    const auto& landmarks = recent_map.landmark_and_obs[i];
    int pose_id = keyframe_to_pose_id.at(kf);
    for (const auto& [landmark, obs] : landmarks) {
      if (!landmark->get_pose()) {
        continue;
      }

      int point_id = pointObservationCount.at(landmark);

      for (const auto& o : obs) {
        problem.observation_xys.push_back(o.xy);
        problem.observation_infos.push_back(o.xy_info);
        problem.point_ids.push_back(point_id);
        problem.pose_ids.push_back(pose_id);
        problem.camera_ids.push_back(o.cam_id);
      }
    }
  }

  problem.max_iterations = sba_settings.num_sba_iterations;
  problem.robustifier_scale = sba_settings.robustifier_scale;
  bool res = bundler.solve(problem);

  // copy data back
  if (res) {
    for (auto& [kf, pose_id] : keyframe_to_pose_id) {
      kf->set_pose(problem.rig_from_world[pose_id]);
    }

    for (auto& [l, point_id] : pointObservationCount) {
      l->set_pose(problem.points[point_id]);
    }
  }
}

template <typename bundler_type>
void run_imu_sba(const UnifiedMap::SubMap& recent_map, const Vector3T& gravity, const camera::Rig& rig,
                 const imu::ImuCalibration& calib, const sba::Settings& sba_settings, bundler_type& bundler) {
  sba_imu::ImuBAProblem problem;
  problem.rig = rig;
  problem.gravity = gravity;
  problem.robustifier_scale = 0.6f;
  problem.prior_acc = 1e6f;
  problem.prior_gyro = 1e3f;
  problem.imu_penalty = 1e-3f;
  problem.reintegration_thresh = 1e-3f;

  // we want this many fixed key frames
  problem.num_fixed_key_frames =
      CalcNumFixedKeyframes(recent_map.consecutive_keyframes.size(), sba_settings.num_fixed_sba_frames);

  if (problem.num_fixed_key_frames < 1) {
    return;
  }

  // We will count observations for each point and will skip
  // points with insufficient number of observations.
  std::unordered_map<LandmarkPtr, int> pointObservationCount;
  size_t total_observarions = 0;

  // count observations for each 3D point
  {
    for (const auto& landmarks : recent_map.landmark_and_obs) {
      for (const auto& [landmark, obs] : landmarks) {
        if (!landmark->get_pose()) {
          continue;
        }
        auto it = pointObservationCount.find(landmark);
        if (it == pointObservationCount.end()) {
          int point_id = static_cast<int>(problem.points.size());
          pointObservationCount.insert(it, {landmark, point_id});
          Vector3T point_w = *landmark->get_pose();
          problem.points.push_back(point_w);
        }
        total_observarions += obs.size();
      }
    }
  }

  // remaps track ids into consecutive indices to give to bundler
  std::unordered_map<KeyframePtr, int32_t> keyframe_to_pose_id;

  {
    problem.observation_xys.reserve(total_observarions);
    problem.observation_infos.reserve(total_observarions);
    problem.point_ids.reserve(total_observarions);
    problem.pose_ids.reserve(total_observarions);
    problem.camera_ids.reserve(total_observarions);
  }

  for (const auto& [kf, preint] : recent_map.consecutive_keyframes) {
    keyframe_to_pose_id[kf] = problem.rig_poses.size();

    State s = kf->get_state();
    problem.rig_poses.push_back({s.rig_from_w.inverse() * calib.rig_from_imu(), s.velocity, s.gyro_bias, s.acc_bias,
                                 preint ? *preint : IMUPreintegration()});
  }

  for (size_t i = 0; i < recent_map.consecutive_keyframes.size(); i++) {
    const auto& kf = recent_map.consecutive_keyframes[i].keyframe;
    const auto& landmarks = recent_map.landmark_and_obs[i];
    int pose_id = keyframe_to_pose_id.at(kf);
    for (const auto& [landmark, obs] : landmarks) {
      if (!landmark->get_pose()) {
        continue;
      }

      int point_id = pointObservationCount.at(landmark);

      for (const auto& o : obs) {
        problem.observation_xys.push_back(o.xy);
        problem.observation_infos.push_back(o.xy_info);
        problem.point_ids.push_back(point_id);
        problem.pose_ids.push_back(pose_id);
        problem.camera_ids.push_back(o.cam_id);
      }
    }
  }

  problem.max_iterations = 10;
  bundler.solve(problem);

  // copy data back
  {
    int pose_id = 0;
    for (const auto& [kf, preint] : recent_map.consecutive_keyframes) {
      const auto& pose = problem.rig_poses[pose_id];
      kf->set_state({calib.rig_from_imu() * pose.w_from_imu.inverse(),  // rig_from_w
                     pose.velocity, pose.acc_bias, pose.gyro_bias});

      if (preint) {
        *preint = pose.preintegration;
      }
      pose_id++;
    }

    for (auto& [l, point_id] : pointObservationCount) {
      l->set_pose(problem.points[point_id]);
    }
  }
}

}  // namespace

using namespace cuvslam::map;

template <class Bundler>
class SbaService : public ServiceBase {
public:
  SbaService(const sba::Settings& sba_settings, const camera::Rig& rig, UnifiedMap& map)
      : ServiceBase(map, sba_settings.async), rig_(rig), sba_settings_(sba_settings) {}

  virtual void service_task() final {
    TRACE_EVENT ev = profiler_domain_.trace_event("service_task");

    TRACE_EVENT id1 = profiler_domain_.trace_event_start("get_recent_submap");
    auto recent_map = map_.get_recent_submap(sba_settings_.num_sba_frames, true);
    profiler_domain_.trace_event_end(id1);

    // first run
    {
      TRACE_EVENT ev1 = profiler_domain_.trace_event("run_sba");
      run_sba(recent_map, rig_, sba_settings_, bundler_);
    }
    if (sba_settings_.use_sba_winsorizer) {
      KeyframePtr last_kf = recent_map.consecutive_keyframes.back().keyframe;
      auto& last_landmarks = recent_map.landmark_and_obs.back();
      std::vector<KeyframeLandmarkObs> observations;
      observations.reserve(1e3);

      for (const auto& [landmark, obs] : last_landmarks) {
        for (const auto& o : obs) {
          observations.push_back({last_kf, landmark, o});
        }
      }

      // may mark landmarks as Winsorized, thus excluding them from SBA calculations!
      winsorize(rig_, observations);

      // second run
      run_sba(recent_map, rig_, sba_settings_, bundler_);
    }
  }

  ~SbaService() { stop(); }

private:
  camera::Rig rig_;
  sba::Settings sba_settings_;
  Bundler bundler_;
  profiler::SBAProfiler::DomainHelper profiler_domain_ = profiler::SBAProfiler::DomainHelper("SBA Service");
};

using CpuSbaService = SbaService<sba::SchurComplementBundlerCpu>;

template <class RegularBundler, class ImuBundler>
class ImuSbaService : public ServiceBase {
public:
  ImuSbaService(const sba::Settings& sba_settings, const camera::Rig& rig, const imu::ImuCalibration& calib,
                UnifiedMap& map)
      : ServiceBase(map, sba_settings.async),
        rig_(rig),
        calib_(calib),
        sba_settings_(sba_settings),
        imu_bundler_(calib){};

  virtual void service_task() final {
    TRACE_EVENT ev = profiler_domain_.trace_event("service_task");

    auto recent_map = map_.get_recent_submap(sba_settings_.num_inertial_sba_frames);
    const std::optional<Vector3T> gravity = map_.get_gravity();

    // first run
    if (gravity) {
      TRACE_EVENT ev1 = profiler_domain_.trace_event("run_imu_sba");
      run_imu_sba(recent_map, *gravity, rig_, calib_, sba_settings_, imu_bundler_);
      TraceDebug("IMU SBA has finished");
    } else {
      TRACE_EVENT ev1 = profiler_domain_.trace_event("run_sba");
      run_sba(recent_map, rig_, sba_settings_, bundler_);
      TraceDebug("Regular SBA has finished");
    }
  }

  ~ImuSbaService() { stop(); }

private:
  camera::Rig rig_;
  imu::ImuCalibration calib_;
  sba::Settings sba_settings_;
  RegularBundler bundler_;
  ImuBundler imu_bundler_;
  profiler::SBAProfiler::DomainHelper profiler_domain_ = profiler::SBAProfiler::DomainHelper("Inertial SBA Service");
};

#ifdef USE_CUDA
using ImuSbaCPUService = ImuSbaService<sba::SchurComplementBundlerGpu, sba_imu::IMUBundlerCpuFixedVel>;
#else
using ImuSbaCPUService = ImuSbaService<sba::SchurComplementBundlerCpu, sba_imu::IMUBundlerCpuFixedVel>;
#endif

}  // namespace cuvslam::pipelines

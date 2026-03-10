
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

#include "pipelines/inertial_pnp.h"

#include <unordered_map>
#include <vector>

#include "camera/observation.h"
#include "camera/rig.h"
#include "common/vector_3t.h"
#include "imu/soft_inertial_pnp.h"

#include "pipelines/track.h"

namespace cuvslam::pipelines {

bool InertialPnP::Solve(const imu::ImuCalibration& calib, const std::unordered_map<TrackId, Track>& tracks3d,
                        const std::vector<camera::Observation>& observations, const camera::Rig& rig,
                        const Vector3T& gravity_w,
                        sba_imu::Pose& prev_pose,  // non-const because of velocities updates
                        sba_imu::Pose& curr_pose) const {
  TRACE_EVENT ev = profiler_domain_.trace_event("Solve");

  sba_imu::StereoPnPInput input;
  input.gravity = gravity_w;
  input.prior_acc = 1e5f;
  input.prior_gyro = 1e3f;
  input.rig = rig;
  input.robustifier_scale = 0.4f;
  input.robustifier_scale_pose = 2.f;
  input.max_iterations = 20;
  input.outlier_thresh = {10.f, 7.f};
  input.translation_constraint = 1e5f;
  input.robustifier_scale_tr = 0.02f;

  std::unordered_map<int, int> track_to_point;
  for (const auto& track : observations) {
    const TrackId track_id = track.id;
    const auto it = tracks3d.find(track_id);
    if (it == tracks3d.end()) {
      continue;
    }
    const Track& track3d = it->second;

    if (track3d.hasLocation()) {
      const Vector3T p_w = track3d.getLocation3D();
      int point_id = static_cast<int>(input.points.size());
      auto it = track_to_point.find(track_id);
      if (it == track_to_point.end()) {
        track_to_point[track_id] = point_id;
        input.points.push_back(p_w);
      } else {
        point_id = it->second;
      }
      input.point_ids.push_back(point_id);
      input.camera_ids.push_back(track.cam_id);
      input.observation_xys.push_back(track.xy);
      input.observation_infos.push_back(track.xy_info);
    }
  }

  const int kMinObservations = 25;
  if (input.observation_xys.size() < kMinObservations) {
    return false;
  }

  TRACE_EVENT ev1 = profiler_domain_.trace_event("SoftInertialPnP");
  bool status = SoftInertialPnP(calib, input, prev_pose, curr_pose, 1e-3f);

  return status;
}

}  // namespace cuvslam::pipelines


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

#include <iostream>

#include "common/log.h"
#include "epipolar/camera_selection.h"

#include "pnp/visual_icp.h"

namespace cuvslam::pnp {

void VisualICP::lift_mono_tracks(const IcpInfo& inputs, const Isometry3T& world_from_rig,
                                 const std::vector<camera::Observation>& observations,
                                 std::vector<cuvslam::pipelines::Landmark>& landmarks) const {
  landmarks.clear();

  std::vector<cuvslam::cuda::GPUICPTools::ObsLmPair> tracks;
  tracks.reserve(observations.size());

  cuvslam::cuda::GPUICPTools::ObsLmPair pair;

  for (const auto& obs : observations) {
    pair.obs_xy = {-obs.xy.x(), obs.xy.y()};

    tracks.push_back(pair);
  }

  const auto& intrinsics = *rig_.intrinsics[inputs.depth_id];

  const Vector2T& focal = intrinsics.getFocal();
  const Vector2T& principal = intrinsics.getPrincipal();

  std::vector<Vector3T> lms;

  icp_tools_.lift_points(inputs.curr_depth[0], focal, principal, tracks, lms);

  Isometry3T world_from_depth = world_from_rig * rig_.camera_from_rig[inputs.depth_id].inverse();

  landmarks.reserve(tracks.size());

  for (size_t i = 0; i < tracks.size(); i++) {
    const auto& obs = observations[i];
    const Vector3T& lm = lms[i];
    if (lm.z() < -1e-1) {
      landmarks.push_back({obs.id, world_from_depth * lm});
    }
  }
}

}  // namespace cuvslam::pnp


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

#include "merge_dbs.h"

#include "slam/slam/slam.h"

namespace cuvslam::slam {

bool MergeDatabases(const camera::Rig& rig, const std::vector<std::string>& dbs, const std::string& slam_db_out) {
  const auto slam = std::make_shared<LocalizerAndMapper>(rig, FeatureDescriptorType::kNone, false);
  // slam->SetKeyframesLimit(300);

  for (auto& db : dbs) {
    const auto const_slam = std::make_shared<LocalizerAndMapper>(rig, FeatureDescriptorType::kNone, false);
    const_slam->SetLandmarksSpatialIndex({});
    PoseGraphOptimizerOptions pg_options;
    pg_options.type = Simple;
    const_slam->SetPoseGraphOptimizerOptions(pg_options);

    if (!const_slam->AttachToExistingReadOnlyDatabase(db)) {
      return false;
    }

    auto root = slam->GetMap().GetRootKeyframe();
    auto const_root = const_slam->GetMap().GetRootKeyframe();

    Isometry3T pose_of_frame_id_in_const_slam = root.second;

    const Matrix6T covariance = Map::GetHardEdgeDefaultCovariance();

    UnionWithOptions options;
    options.optimize_after_union = false;
    slam->UnionWith(const_slam->GetMap(), root.first, const_root.first, pose_of_frame_id_in_const_slam, covariance,
                    nullptr, options);
  }

  slam->RebuildSpatialIndex();
  slam->ReduceKeyframes();

  if (!slam_db_out.empty()) {
    return slam->AttachToNewDatabaseSaveMapAndDetach(slam_db_out);
  }
  return true;
}

}  // namespace cuvslam::slam

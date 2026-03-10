
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

#include "slam/map/map.h"

#include "slam/map/descriptor/st_descriptor_gpu_ops.h"
#include "slam/map/descriptor/st_descriptor_ops.h"

namespace cuvslam::slam {
Map::Map(const camera::Rig& rig, FeatureDescriptorType descriptor_type, bool use_gpu) {
  uint32_t n_shift_only_iterations;
  uint32_t n_full_mapping_iterations;
  switch (descriptor_type) {
    case FeatureDescriptorType::kNone:  // same as kShiTomasi2
    case FeatureDescriptorType::kShiTomasi2:
      n_shift_only_iterations = 20;
      n_full_mapping_iterations = 0;
      break;
    case FeatureDescriptorType::kShiTomasi6:
      n_shift_only_iterations = 10;
      n_full_mapping_iterations = 20;
      break;
    default:
      throw std::invalid_argument("Unknown descriptor type.");
  }
  if (use_gpu) {
    feature_descriptor_ops_ = std::make_unique<STDescriptorGpuOps>(n_shift_only_iterations, n_full_mapping_iterations);
  } else {
    feature_descriptor_ops_ = std::make_unique<STDescriptorOps>(n_shift_only_iterations, n_full_mapping_iterations);
  }
  landmarks_spatial_index_ = std::make_shared<LSIGrid>(*feature_descriptor_ops_, rig, 0.25f, 0, "probes_composed");
}

bool Map::AttachDatabase(std::shared_ptr<ISlamDatabase> database, bool load_data) {
  // set database
  database_ = database;
  landmarks_spatial_index_->SetDatabase(database_);
  if (!load_data) {
    return true;  // skip loading
  }
  database_->GetSingleton(SlamDatabaseSingleton::SpatialIndex, [&](const BlobReader& blob_reader) {
    return landmarks_spatial_index_->FromBlob(blob_reader);
  });
  pose_graph_.GetFromDatabase(database_.get());
  pose_graph_hypothesis_.GetFromDatabase(database_.get());
  return true;
}

void Map::DetachDatabase(bool copy_all_from_db) {
  if (!database_) {
    return;
  }
  constexpr bool read_only = false;
  if (!read_only) {
    // TODO: copy all data to database and copy all data from database to LSI
    landmarks_spatial_index_->SyncDatabase(pose_graph_hypothesis_);
    pose_graph_.PutToDatabase(database_.get());
    pose_graph_hypothesis_.PutToDatabase(database_.get());
  }

  database_->Flush();

  if (copy_all_from_db) {
    landmarks_spatial_index_->SyncDatabaseReverse();
  }
  database_ = nullptr;
  landmarks_spatial_index_->SetDatabase(nullptr);
}

float Map::GetCellSize() const { return landmarks_spatial_index_->GetCellSize(); }

bool Map::HasKeyframes() const { return pose_graph_.GetKeyframeCount() != 0; }

const PoseGraph& Map::GetPoseGraph() const { return pose_graph_; }
const PoseGraphHypothesis& Map::GetPoseGraphHypothesis() const { return pose_graph_hypothesis_; }
std::shared_ptr<const LSIGrid> Map::GetLandmarksSpatialIndex() const { return landmarks_spatial_index_; }

Matrix6T Map::GetHardEdgeDefaultCovariance() {
  Matrix6T covariance = Matrix6T::Zero();
  covariance.diagonal() = (Vector6T() << 1e-10f, 1e-10f, 1e-10f, 1e-9f, 1e-9f, 1e-9f).finished();
  return covariance;
}

std::pair<KeyFrameId, Isometry3T> Map::GetRootKeyframe() const {
  std::pair error(InvalidKeyFrameId, Isometry3T::Identity());

  KeyFrameId min_keyframe_id = InvalidKeyFrameId;
  pose_graph_.QueryKeyframes([&](KeyFrameId keyframe_id) {
    if (min_keyframe_id == InvalidKeyFrameId || keyframe_id < min_keyframe_id) {
      min_keyframe_id = keyframe_id;
    }
  });
  if (min_keyframe_id == InvalidKeyFrameId) {
    return error;
  }
  const Isometry3T* pose = pose_graph_hypothesis_.GetKeyframePose(min_keyframe_id);
  if (!pose) {
    return error;
  }
  return std::pair(min_keyframe_id, *pose);
}

}  // namespace cuvslam::slam

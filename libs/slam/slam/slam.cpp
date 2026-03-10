
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

#include "slam/slam/slam.h"

#include "camera/observation.h"
#include "common/log_types.h"
#include "math/twist.h"

#include "slam/map/database/lmdb_slam_database.h"
#include "slam/map/pose_graph/slam_optimize_options.h"
#include "slam/map/spatial_index/lsi_grid.h"
#include "slam/slam/slam_check_hypothesis.h"

namespace cuvslam::slam {

LocalizerAndMapper::LocalizerAndMapper(const camera::Rig& rig, FeatureDescriptorType descriptor_type, bool use_gpu)
    : rig_(rig),
      map_(rig, descriptor_type, use_gpu),
      pnp_(rig, pnp::PNPSettings::LCSettings()),
      random_generator_(random_device_()) {}

LocalizerAndMapper::~LocalizerAndMapper() { SlamStdout("Destroyed LocalizerAndMapper instance. "); }

void LocalizerAndMapper::SetReproduceMode(bool reproduce_mode) {
  if (reproduce_mode) {
    random_generator_.seed(0);
  }
}

void LocalizerAndMapper::SetLandmarksSpatialIndex(const SpatialIndexOptions& options) {
  float size = options.cell_size;
  std::string weight_func = "probes_composed";

  // default size
  if (size <= 0) {
    // default size if not set
    size = 0.25f;

    // default size based on "pupillary distance"
    // TODO: cover multi-cam setup
    if (rig_.num_cameras == 2) {
      auto left_right = rig_.camera_from_rig[0] * rig_.camera_from_rig[1].inverse();
      auto pd = left_right.translation().norm();
      // constant here
      size = pd * 4;
    }
  }

  map_.landmarks_spatial_index_ =
      std::make_shared<LSIGrid>(*map_.feature_descriptor_ops_, rig_, size, options.max_landmarks_in_cell, weight_func);
}

void LocalizerAndMapper::SetKeyframesLimit(int max_keyframes_count) { max_keyframes_count_ = max_keyframes_count; }

bool LocalizerAndMapper::SetPoseGraphOptimizerOptions(const PoseGraphOptimizerOptions& options) {
  if (options.type == Simple) {
    pose_graph_optimizer_ = "simple";
  } else if (options.type == Dummy) {
    pose_graph_optimizer_ = "";
  } else {
    return false;
  }
  return true;
}

void LocalizerAndMapper::SetKeepTrackPoses(bool keep_track_poses) {
  // if true - CalcFramePose() will working
  keep_track_poses_ = keep_track_poses;

  if (keep_track_poses_) {
    PoseGraph::RemoveNodeCB remove_node_cb = [&](KeyFrameId keyframe_id, KeyFrameId instead_keyframe_id,
                                                 const Isometry3T& to_instead) -> void {
      ChangeKeyframeInfo change_keyframe_info;
      change_keyframe_info.new_node = instead_keyframe_id;
      change_keyframe_info.old_node_to_new = to_instead;
      keyframe_removed_[keyframe_id] = change_keyframe_info;
      // SlamStdout("\nkeyframe_removed_[%zd]={%zd, %zd}", keyframe_id, change_keyframe_info.new_node,
      // change_keyframe_info.old_node_to_new);
    };
    map_.pose_graph_.RegisterRemoveNodeCB(remove_node_cb);
  }
}

void LocalizerAndMapper::SetActiveCameras(const std::vector<CameraId>& cameras) {
  if (active_cameras_) {
    throw std::runtime_error("Set active camera should be called once before map update.");
  }
  active_cameras_ = cameras;
}

bool LocalizerAndMapper::UnionWith(const Map& const_map,
                                   KeyFrameId to_keyframe_id,          // of this
                                   KeyFrameId const_slam_keyframe_id,  // of const_slam
                                   const Isometry3T& pose_of_frame_id_in_const_slam, const Matrix6T& covariance,
                                   Isometry3T* slam_to_head, const UnionWithOptions& options) {
  // Transform for const_slam poses
  Isometry3T from_to = Isometry3T::Identity();
  const Isometry3T* const_slam_keyframe_pose =
      const_map.GetPoseGraphHypothesis().GetKeyframePose(const_slam_keyframe_id);
  if (const_slam_keyframe_pose) {
    from_to = const_slam_keyframe_pose->inverse() * pose_of_frame_id_in_const_slam;
  }

  // Reindex exists data
  std::map<KeyFrameId, KeyFrameId> keyframe_id_remap;
  if (!map_.pose_graph_.CreateKeyframeIdRemap(const_map.pose_graph_, keyframe_id_remap)) {
    SlamStderr("Failed to create keyframe remapping in pose graph.\n");
    return false;
  }

  std::function<void(LandmarkId, KeyFrameId)> empty_func = [&](LandmarkId, KeyFrameId) {};
  map_.landmarks_spatial_index_->RemoveDeadLandmarks(empty_func);

  std::map<LandmarkId, LandmarkId> landmark_id_remap;
  if (!map_.landmarks_spatial_index_->CreateLandmarkIdRemap(const_map.landmarks_spatial_index_.get(),
                                                            landmark_id_remap)) {
    SlamStderr("Failed to create landmark remapping in LSI grid.\n");
    return false;
  }

  if (to_keyframe_id != InvalidKeyFrameId) {
    // change to_keyframe_id
    auto it = keyframe_id_remap.find(to_keyframe_id);
    if (it == keyframe_id_remap.end()) {
      SlamStderr("to_keyframe_id is not found in remapped keyframes.\n");
      return false;
    }
    to_keyframe_id = it->second;
  }

  if (!map_.pose_graph_.Reindex(keyframe_id_remap, landmark_id_remap)) {
    SlamStderr("Failed to reindex pose graph corresponding to remapped keyframes and landmarks.\n");
    return false;
  }
  if (!map_.pose_graph_hypothesis_.Reindex(keyframe_id_remap)) {
    SlamStderr("Failed to reindex pose graph hypothesis corresponding to remapped keyframes.\n");
    return false;
  }
  if (!map_.landmarks_spatial_index_->Reindex(keyframe_id_remap, landmark_id_remap)) {
    SlamStderr("Failed to reindex LSI grid corresponding to remapped keyframes and landmarks.\n");
    return false;
  }

  bool reassign_head_node = (to_keyframe_id == InvalidKeyFrameId);
  if (!map_.pose_graph_.Union(const_map.pose_graph_, reassign_head_node)) {
    SlamStderr("Failed to union pose graph with const_slam pose graph.\n");
    return false;
  }
  if (!map_.pose_graph_hypothesis_.Union(const_map.pose_graph_hypothesis_)) {
    SlamStderr("Failed to union pose graph hypothesis with const_slam pose graph hypothesis.\n");
    return false;
  }

  // Add edge
  if (to_keyframe_id != InvalidKeyFrameId) {
    map_.pose_graph_.AddEdge(map_.pose_graph_hypothesis_, const_slam_keyframe_id, to_keyframe_id, from_to, covariance,
                             nullptr);
  }

  if (options.optimize_after_union) {
    // Optimize Pose Graph
    Isometry3T vo_to_head;
    OptimizeOptions optimize_options;
    optimize_options.condition = SuccessfullLC;
    optimize_options.constraint_first_node = true;
    optimize_options.max_iterations = 10;
    if (!map_.pose_graph_.Optimize(map_.pose_graph_hypothesis_, map_.pose_graph_hypothesis_for_swap_, vo_to_head,
                                   optimize_options)) {
      SlamStderr("Failed to optimize pose graph.\n");
      return false;
    }
    Isometry3T pose_estimate = this->pose_estimate_;
    pose_estimate = pose_estimate * vo_to_head;
    RemoveScaleFromTransform(pose_estimate);
    this->pose_estimate_ = pose_estimate;

    map_.pose_graph_hypothesis_.swap(map_.pose_graph_hypothesis_for_swap_);

    if (slam_to_head) {
      *slam_to_head = vo_to_head;
    }
  }

  // reset database
  map_.landmarks_spatial_index_->SetDatabase(nullptr);

  if (!map_.landmarks_spatial_index_->Union(const_map.landmarks_spatial_index_.get(), map_.pose_graph_hypothesis_)) {
    SlamStderr("Failed to union LSI grid with const_slam LSI grid.\n");
    return false;
  }

  // required for CalcFramePose()
  {
    auto reindex_keyframe = [&](KeyFrameId src, KeyFrameId& dst) {
      auto it_remap = keyframe_id_remap.find(src);
      if (it_remap == keyframe_id_remap.end()) {
        SlamStderr("Failed to reindex keyframe, keyframe not found in remapped keyframes.\n");
        return false;
      }
      dst = it_remap->second;
      return true;
    };

    std::map<KeyFrameId, FrameId> keyframe_sources;
    std::map<KeyFrameId, ChangeKeyframeInfo> keyframe_removed;

    for (auto& it : keyframe_sources_) {
      KeyFrameId keyframe_id;
      if (!reindex_keyframe(it.first, keyframe_id)) {
        return false;
      }
      keyframe_sources[keyframe_id] = it.second;
    }

    for (auto& it : keyframe_removed_) {
      KeyFrameId keyframe_id;
      if (!reindex_keyframe(it.first, keyframe_id)) {
        return false;
      }
      auto& value = keyframe_removed[keyframe_id];
      value = it.second;
      if (!reindex_keyframe(value.new_node, value.new_node)) {
        return false;
      }
    }

    keyframe_removed_ = keyframe_removed;
    keyframe_sources_ = keyframe_sources;
  }

  // Copy from slam
  map_.database_ = const_map.database_;
  return true;
}

bool LocalizerAndMapper::CalcFramePose(FrameId frame_id, Isometry3T& pose) const {
  if (!keep_track_poses_) {
    return false;
  }

  KeyFrameId keyframe_id = InvalidKeyFrameId;
  Isometry3T from_keyframe_to_frame = Isometry3T::Identity();
  FindKeyframeByFrame(frame_id, keyframe_id, from_keyframe_to_frame);

  const Isometry3T* keyframe_pose = map_.pose_graph_hypothesis_.GetKeyframePose(keyframe_id);

  if (!keyframe_pose) {
    SlamStderr("Failed to calculate frame pose for keyframe_id %zd.\n", static_cast<uint64_t>(frame_id));
    return false;
  }

  pose = (*keyframe_pose) * from_keyframe_to_frame;
  return true;
}

bool LocalizerAndMapper::FindKeyframeByFrame(FrameId frame_id, KeyFrameId& keyframe_id,
                                             Isometry3T& from_keyframe_to_frame) const {
  keyframe_id = InvalidKeyFrameId;
  int diff = INT32_MAX;

  // sourced keyframes
  for (auto it : keyframe_sources_) {
    const auto& td = it.second;

    if (frame_id < td) {
      continue;
    }

    if (diff < static_cast<int>(frame_id) - static_cast<int>(td)) {
      continue;
    }

    diff = frame_id - td;
    keyframe_id = it.first;

    if (diff == 0) {
      break;
    }
  }

  // removed keyframes
  from_keyframe_to_frame = Isometry3T::Identity();
  for (;;) {
    auto it = keyframe_removed_.find(keyframe_id);
    if (it == keyframe_removed_.end()) {
      // key exists
      break;
    }
    auto& cki = it->second;
    keyframe_id = cki.new_node;
    from_keyframe_to_frame = cki.old_node_to_new * from_keyframe_to_frame;
  }

  return keyframe_id != InvalidKeyFrameId;
}

bool LocalizerAndMapper::FindFrameByKeyframe(KeyFrameId keyframe_id, FrameId& frame_id) const {
  const auto it = keyframe_sources_.find(keyframe_id);
  if (it != keyframe_sources_.end()) {
    frame_id = it->second;
    return true;
  }
  return false;
}

// current estimated pose
Isometry3T LocalizerAndMapper::GetCurrentPose() const { return pose_estimate_; }

bool LocalizerAndMapper::FlushActiveDatabase() const {
  if (!map_.database_) {
    return false;
  }
  TRACE_EVENT ev = profiler_domain_.trace_event("FlushActiveDatabase()", profiler_color_);
  map_.database_->SetSingleton(SlamDatabaseSingleton::SpatialIndex, 0, [&](BlobWriter& blob_writer) {
    return map_.landmarks_spatial_index_->ToBlob(blob_writer);
  });

  map_.database_->Flush();
  return true;
}

bool LocalizerAndMapper::AttachToExistingReadOnlyDatabase(const std::string& path) {
#ifdef USE_LMDB
  const auto lmdb = std::make_shared<LmdbSlamDatabase>();

  const char* url = path.c_str();
  if (!lmdb->Open(url, LmdbSlamDatabase::OpenMode::READ_ONLY_EXISTS)) {
    SlamStderr("Failed to open Slam Database %s.\n", url);
    return false;
  }
  SlamStdout("Successfully opened Slam Database %s.\n", url);
  if (!map_.AttachDatabase(lmdb, true)) {
    SlamStderr("Failed to attach empty database \"%s\"", url);
    return false;
  }
  return true;
#else
  SlamStderr("AttachToExistingReadOnlyDatabase is not implemented for databases other than LMDB.");
  return false;
#endif
}

bool LocalizerAndMapper::AttachToNewDatabase(const std::string& path) {
#ifdef USE_LMDB
  auto lmdb = std::make_shared<LmdbSlamDatabase>();
  const char* url = path.c_str();
  if (!lmdb->Open(url, LmdbSlamDatabase::OpenMode::READ_WRITE_HARD_RESET)) {
    SlamStderr("Failed to open Slam Database %s.\n", url);
    return false;
  }
  if (!map_.AttachDatabase(lmdb, false)) {
    SlamStderr("Failed to attach database \"%s\".\n", url);
    return false;
  }
  SlamStdout("Successfully opened Slam Database %s for read-write.\n", url);
  return true;
#else
  SlamStderr("AttachToNewDatabase is not implemented for databases other than LMDB.");
  return false;
#endif
}

bool LocalizerAndMapper::AttachToNewDatabaseSaveMapAndDetach(const std::string& path) {
#ifdef USE_LMDB
  if (!AttachToNewDatabase(path)) {
    return false;
  }
  map_.DetachDatabase(true);
  SlamStdout("Successfully copied Slam Database %s.\n", path.c_str());
  return true;
#else
  SlamStderr("AttachToNewDatabaseSaveMapAndDetach is not implemented for databases other than LMDB.");
  return false;
#endif
}

void LocalizerAndMapper::DetachDatabase() { map_.DetachDatabase(false); }

void LocalizerAndMapper::DetectLoopClosure(const ILoopClosureSolver& loop_closure_solver, const Images& images,
                                           const Isometry3T& world_from_rig_guess, LoopClosureStatus& status) const {
  TRACE_EVENT ev = profiler_domain_.trace_event("LC()", profiler_color_);

  status = LoopClosureStatus();
  status.success = false;

  SlamCheckTask task;
  task.loop_closure_task.current_images = images;
  task.loop_closure_task.pose_graph_hypothesis = map_.pose_graph_hypothesis_.MakeCopy();
  task.loop_closure_task.guess_world_from_rig = world_from_rig_guess;
  map_.pose_graph_.GetHeadKeyframe(task.loop_closure_task.pose_graph_head);
  const LSIGrid& lsi = *map_.landmarks_spatial_index_;
  SlamCheckHypothesis(lsi, &loop_closure_solver, rig_, map_.feature_descriptor_ops_.get(), task);

  status.keyframes_in_sight.assign(task.keyframes_in_sight.begin(), task.keyframes_in_sight.end());
  status.success = task.succesed;
  status.result_pose = task.result_pose;
  status.result_pose_covariance = task.result_pose_covariance;
  status.reprojection_error = task.reprojection_error;
  status.good_landmarks_count = task.landmarks.size();
  status.pnp_landmarks_count = task.landmarks.size() + task.probes_types[LP_PNP_FAILED];
  status.tracked_landmarks_count =
      task.landmarks.size() + task.probes_types[LP_PNP_FAILED] + task.probes_types[LP_RANSAC_FAILED];
  status.selected_landmarks_count = task.landmarks.size() + task.probes_types[LP_PNP_FAILED] +
                                    task.probes_types[LP_RANSAC_FAILED] + task.probes_types[LP_TRACKING_FAILED];
  status.landmarks = task.landmarks;
  status.discarded_landmarks = task.discarded_landmarks;
}

// Update Landmark Statistic in spatial index
void LocalizerAndMapper::UpdateLandmarkProbeStatistics(
    const std::vector<std::pair<LandmarkId, LandmarkProbe> >& discarded_landmarks) {
  for (auto& discarded_landmark : discarded_landmarks) {
    map_.landmarks_spatial_index_->AddLandmarkProbeStatistic(discarded_landmark.first, discarded_landmark.second);
  }
}

bool LocalizerAndMapper::ApplyLoopClosureResult(const Isometry3T& world_from_lc, const Matrix6T& lc_pose_covariance,
                                                const std::vector<LandmarkInSolver>& lc_landmarks) {
  KeyFrameId headkf;
  if (!map_.pose_graph_.GetHeadKeyframe(headkf)) {
    return false;
  }
  PoseGraph::EdgeStat edge_stat;
  const KeyFrameId lckf = FindKeyframeWithMostLandmarks(lc_landmarks, &edge_stat);
  if (lckf == InvalidKeyFrameId) {
    return false;
  }

  const Isometry3T* world_from_headkf = map_.pose_graph_hypothesis_.GetKeyframePose(headkf);
  if (!world_from_headkf) {
    return false;
  }
  const Isometry3T* world_from_lckf = map_.pose_graph_hypothesis_.GetKeyframePose(lckf);
  if (!world_from_lckf) {
    return false;
  }

  const Isometry3T headkf_from_world = world_from_headkf->inverse();
  const Isometry3T& world_from_estimate = pose_estimate_;
  const Isometry3T headkf_from_estimate = headkf_from_world * world_from_estimate;
  const Isometry3T estimate_from_head = headkf_from_estimate.inverse();
  const Isometry3T world_from_correctedheadkf = world_from_lc * estimate_from_head;  // lc correction lc ~= estimate
  const Isometry3T lckf_from_world = world_from_lckf->inverse();
  const Isometry3T lckf_from_correctedheadkf = lckf_from_world * world_from_correctedheadkf;

  if (!AddEdgeToPoseGraph(lckf, headkf, lckf_from_correctedheadkf, lc_pose_covariance, edge_stat)) {
    return false;
  }
  // Add Landmark Relation to spatial index and pose graph
  for (auto& landmark_in_solver : lc_landmarks) {
    const bool valid = map_.pose_graph_.AddLandmarkRelation(landmark_in_solver.id, headkf);
    if (valid) {
      map_.landmarks_spatial_index_->AddLandmarkRelation(landmark_in_solver.id, headkf, nullptr,
                                                         landmark_in_solver.uv_norm, map_.pose_graph_hypothesis_);
    }
  }
  return true;
}

// Optimize
bool LocalizerAndMapper::OptimizePoseGraph(bool planar_constraints, int max_iterations) {
  TRACE_EVENT ev1 = profiler_domain_.trace_event("Optimize()", profiler_color_);

  if (pose_graph_optimizer_.empty()) {
    return false;
  }

  OptimizeOptions optimize_options;
  optimize_options.planar_constraints = planar_constraints;
  optimize_options.max_iterations = max_iterations;
  optimize_options.condition = SuccessfullLC;
  Isometry3T vo_to_head;
  if (!map_.pose_graph_.Optimize(map_.pose_graph_hypothesis_, map_.pose_graph_hypothesis_for_swap_, vo_to_head,
                                 optimize_options)) {
    return false;
  }

  // Update cells in pose graph
  {
    TRACE_EVENT ev2 = profiler_domain_.trace_event("update cells in pose graph", profiler_color_);
    map_.pose_graph_.QueryKeyframePoses([&](KeyFrameId keyframe_id, const Isometry3T& pose) {
      const auto* pose_src = map_.pose_graph_hypothesis_.GetKeyframePose(keyframe_id);
      const auto* pose_dst = map_.pose_graph_hypothesis_for_swap_.GetKeyframePose(keyframe_id);
      if (pose_src && pose_dst) {
        map_.landmarks_spatial_index_->OnUpdateKeyframePose(keyframe_id, (*pose_src) * pose, (*pose_dst) * pose);
      }
    });
    map_.landmarks_spatial_index_->OnUpdateKeyframePoseFinished();
  }

  // ok, so use new poses
  map_.pose_graph_hypothesis_.swap(map_.pose_graph_hypothesis_for_swap_);
  Isometry3T pose_estimate = this->pose_estimate_;
  pose_estimate = pose_estimate * vo_to_head;
  RemoveScaleFromTransform(pose_estimate);
  pose_estimate_ = pose_estimate;

  if (map_.database_) {
    map_.pose_graph_hypothesis_.PutToDatabase(map_.database_.get());
  }
  return true;
}

void LocalizerAndMapper::MergeLandmarks(
    KeyFrameId keyframe_id, float uv_norm_min_distance,
    std::function<bool(LandmarkId landmark0, LandmarkId landmark1, float ncc)> func) {
  TRACE_EVENT ev = profiler_domain_.trace_event("MergeLandmarks()", profiler_color_);

  int uv_grid_size = static_cast<int>(2 / uv_norm_min_distance);
  uv_grid_size = std::max(uv_grid_size, 1);
  uv_grid_size = std::min(uv_grid_size, 32);

  auto xy_grid_key = [&](int x, int y) -> size_t {
    x = std::max(x, 0);
    x = std::min(x, uv_grid_size - 1);
    y = std::max(y, 0);
    y = std::min(y, uv_grid_size - 1);
    return x + y * uv_grid_size;
  };

  // grid key from uv_norm
  auto uv_grid_key = [&](Vector2T uv_norm) -> size_t {
    uv_norm = uv_norm * 0.5 + Vector2T(0.5, 0.5);
    int x = floor(uv_norm.x() * uv_grid_size);
    int y = floor(uv_norm.y() * uv_grid_size);
    return xy_grid_key(x, y);
  };

  // fetch all landmarks from keyframe
  struct LandmarkInfo {
    LandmarkId id;
    Vector2T uv_norm;
    size_t key;
  };
  std::vector<LandmarkInfo> landmarks_in_kf;
  size_t landmark_count = map_.pose_graph_.QueryKeyframeLandmarks(keyframe_id, [](LandmarkId) { return false; });
  landmarks_in_kf.reserve(landmark_count);

  map_.pose_graph_.QueryKeyframeLandmarks(keyframe_id, [&](LandmarkId landmark_id) {
    LandmarkInfo landmark_info;
    landmark_info.id = landmark_id;

    if (!map_.landmarks_spatial_index_->GetLandmarkRelation(landmark_id, keyframe_id, &landmark_info.uv_norm)) {
      return true;
    }

    landmark_info.key = uv_grid_key(landmark_info.uv_norm);

    landmarks_in_kf.emplace_back(landmark_info);

    return true;
  });
  landmark_count = landmarks_in_kf.size();

  // grid list: sorted by keys
  std::vector<int> uv_grid(landmark_count);

  for (size_t i = 0; i < landmark_count; i++) {
    uv_grid[i] = i;
  }

  std::sort(uv_grid.begin(), uv_grid.end(),
            [&](const int a, const int b) { return landmarks_in_kf[a].key < landmarks_in_kf[b].key; });

  auto feed_cell = [&](int begin) {
    size_t key = landmarks_in_kf[uv_grid[begin]].key;
    size_t end = begin + 1;

    for (; end < uv_grid.size(); end++) {
      size_t key_end = landmarks_in_kf[uv_grid[end]].key;

      if (key != key_end) {
        break;
      }
    }

    return end;
  };
  auto cell_by_key = [&](size_t key, size_t& begin, size_t& end) {
    auto it = std::lower_bound(uv_grid.begin(), uv_grid.end(), key,
                               [landmarks_in_kf](const int& p_lmi, const size_t key) -> bool {
                                 const LandmarkInfo& lmi = landmarks_in_kf[p_lmi];
                                 return lmi.key < key;
                               });

    if (it == uv_grid.end()) {
      begin = 0, end = 0;
      return;
    }

    begin = std::distance(uv_grid.begin(), it);
    const LandmarkInfo& lmi = landmarks_in_kf[uv_grid[begin]];

    if (lmi.key != key) {
      begin = 0, end = 0;
      return;
    }

    end = feed_cell(begin);
  };

  int match_count = 0;
  auto match = [&](int i0, int i1) {
    // test uv distance
    float dist = (landmarks_in_kf[i0].uv_norm - landmarks_in_kf[i1].uv_norm).squaredNorm();

    if (dist > uv_norm_min_distance * uv_norm_min_distance) {
      return;
    }

    // match feature_descriptors
    LandmarkId l0 = landmarks_in_kf[i0].id;
    LandmarkId l1 = landmarks_in_kf[i1].id;
    const auto fd0 = map_.landmarks_spatial_index_->GetLandmarkFeatureDescriptor(l0);
    const auto fd1 = map_.landmarks_spatial_index_->GetLandmarkFeatureDescriptor(l1);

    if (!fd0 || !fd1) {
      return;
    }

    auto ncc = map_.feature_descriptor_ops_->Match(fd0, fd1);
    match_count++;

    if (ncc > 0) {
      func(l0, l1, ncc);
    }
  };

  // for each cell in grid:
  for (size_t begin = 0, end = 0; begin < landmark_count; begin = end) {
    end = feed_cell(begin);

    // match landmarks in this cell
    for (size_t i0 = begin; i0 < end; i0++) {
      for (size_t i1 = i0 + 1; i1 < end; i1++) {
        match(uv_grid[i0], uv_grid[i1]);
      }
    }

    size_t key1 = landmarks_in_kf[uv_grid[begin]].key;
    int y = key1 / uv_grid_size;
    int x = key1 - uv_grid_size * y;

    // match landmarks in 3 neighbours cells
    static const std::pair<int, int> dxdys[] = {{1, 0}, {1, 1}, {0, 1}};

    for (auto dxdy : dxdys) {
      if (x + dxdy.first >= uv_grid_size || y + dxdy.second >= uv_grid_size) {
        continue;
      }

      size_t key2 = xy_grid_key(x + dxdy.first, y + dxdy.second);
      size_t begin2, end2;
      cell_by_key(key2, begin2, end2);

      if (begin2 == end2) {
        continue;
      }

      // all permutation this cell to neighbours
      for (size_t i0 = begin; i0 < end; i0++) {
        for (size_t i1 = begin2; i1 < end2; i1++) {
          match(uv_grid[i0], uv_grid[i1]);
        }
      }
    }
  }
}

// callback for landmarks_spatial_index_->RemoveDeadLandmarks()
void LocalizerAndMapper::RemoveLandmarkRelation(LandmarkId landmark_id, KeyFrameId keyframe_id) {
  // Remove pose graph links
  map_.pose_graph_.RemoveLandmarkRelation(landmark_id, keyframe_id);
}

void LocalizerAndMapper::RebuildSpatialIndex() {
  TRACE_EVENT ev = profiler_domain_.trace_event("RebuildSpatialIndex()", profiler_color_);

  map_.landmarks_spatial_index_->RebuildAllGridCells(map_.pose_graph_hypothesis_);

  std::function<void(LandmarkId, KeyFrameId)> func_remove_from_keyframe =
      [&](LandmarkId landmark_id, KeyFrameId keyframe_id) { this->RemoveLandmarkRelation(landmark_id, keyframe_id); };
  map_.landmarks_spatial_index_->ReduceLandmarks("probes_composed");

  map_.landmarks_spatial_index_->RemoveDeadLandmarks(func_remove_from_keyframe);
}

KeyFrameId LocalizerAndMapper::FindKeyframeWithMostLandmarks(const std::vector<LandmarkInSolver>& landmarks,
                                                             PoseGraph::EdgeStat* edge_stat) const {
  KeyFrameId pose_graph_head = InvalidKeyFrameId;
  map_.pose_graph_.GetHeadKeyframe(pose_graph_head);  // can set to InvalidKeyFrameId

  std::map<KeyFrameId, PoseGraph::EdgeStat> keyframes_of_landmarks;

  for (const auto& landmark_in_solver : landmarks) {
    map_.landmarks_spatial_index_->QueryLandmarkRelations(landmark_in_solver.id,
                                                          [&](KeyFrameId kf_id, const Vector2T&) {
                                                            if (kf_id == pose_graph_head) {
                                                              return true;
                                                            }

                                                            auto& counter = keyframes_of_landmarks[kf_id];
                                                            counter.tracks3d_number++;
                                                            return true;
                                                          });
  }

  if (keyframes_of_landmarks.empty()) {
    return InvalidKeyFrameId;
  }

  using keyframesOfLandmarksKeyValue = std::pair<KeyFrameId, PoseGraph::EdgeStat>;
  auto max_it = std::max_element(keyframes_of_landmarks.begin(), keyframes_of_landmarks.end(),
                                 [&](const keyframesOfLandmarksKeyValue& a1, const keyframesOfLandmarksKeyValue& a2) {
                                   float w1 = a1.second.tracks3d_number;
                                   float w2 = a2.second.tracks3d_number;

                                   // TODO: remove hack
                                   auto exist_stat1 = map_.pose_graph_.GetEdgeStatistic(a1.first, pose_graph_head);

                                   if (exist_stat1) {
                                     w1 = w1 * 0.5f;

                                     if (exist_stat1->Weight() < a1.second.Weight()) {
                                       w1 = w1 * 0.5f;
                                     }
                                   }

                                   auto exist_stat2 = map_.pose_graph_.GetEdgeStatistic(a2.first, pose_graph_head);

                                   if (exist_stat2) {
                                     w2 = w2 * 0.5f;

                                     if (exist_stat2->Weight() < a2.second.Weight()) {
                                       w2 = w2 * 0.5f;
                                     }
                                   }

                                   return w1 < w2;
                                 });
  const KeyFrameId max_landmarks_keyframe_id = max_it->first;

  if (edge_stat) {
    *edge_stat = max_it->second;
  }

  return max_landmarks_keyframe_id;
}

bool LocalizerAndMapper::AddEdgeToPoseGraph(const KeyFrameId start, KeyFrameId end, const Isometry3T& start_from_end,
                                            const Matrix6T& start_from_end_covariance, PoseGraph::EdgeStat& stat) {
  TRACE_EVENT ev = profiler_domain_.trace_event("AddEdgeToPoseGraph()", profiler_color_);

  // discard if existed edges has more weight
  const auto statistic_exists_edge = map_.pose_graph_.GetEdgeStatistic(start, end);

  if (statistic_exists_edge) {
    if (statistic_exists_edge->Weight() > stat.Weight()) {
      return false;
    }
  }

  // Add Edge to pose graph
  map_.pose_graph_.AddEdge(map_.pose_graph_hypothesis_, start, end, start_from_end, start_from_end_covariance, &stat);
  return true;
}

// Reduce landmarks number
bool LocalizerAndMapper::ReduceLandmarks(const char* weight_func) {
  TRACE_EVENT ev = profiler_domain_.trace_event("ReduceLandmarks()", profiler_color_);

  std::function<void(LandmarkId, KeyFrameId)> func_remove_from_keyframe =
      [&](LandmarkId landmark_id, KeyFrameId keyframe_id) { this->RemoveLandmarkRelation(landmark_id, keyframe_id); };
  map_.landmarks_spatial_index_->ReduceLandmarks(weight_func);
  map_.landmarks_spatial_index_->RemoveDeadLandmarks(func_remove_from_keyframe);

  FlushActiveDatabase();
  return true;
}

// Reduce keyframes count
void LocalizerAndMapper::ReduceKeyframes() {
  if (max_keyframes_count_ == 0) {
    return;  // special value - unlimited pose graph
  }

  std::function update_landmark_keyframe = [&](KeyFrameId old, KeyFrameId neo, const std::vector<LandmarkId>& landmarks,
                                               const Isometry3T& transform_old_to_new) {
    // change keyframe of landmarks
    for (const auto landmark_id : landmarks) {
      map_.landmarks_spatial_index_->UpdateLandmarkRelation(landmark_id, old, neo, transform_old_to_new);
    }
  };

  while (map_.pose_graph_.GetKeyframeCount() > max_keyframes_count_) {
    // select not standalone most confident edge
    const EdgeId edge_id = map_.pose_graph_.GetSmallestVarianceEdgeId();
    if (edge_id != InvalidEdgeId) {
      map_.pose_graph_.ReduceSingleEdge(edge_id, map_.pose_graph_hypothesis_, update_landmark_keyframe);
    }
  }
}

const Map& LocalizerAndMapper::GetMap() const { return map_; }

bool LocalizerAndMapper::GetLastKeyframePoseAndTimestamp(Isometry3T& last_keyframe_pose,
                                                         int64_t& last_keyframe_ts) const {
  const PoseGraph& pg = map_.GetPoseGraph();

  KeyFrameId head_keyframe_id;
  if (!pg.GetHeadKeyframe(head_keyframe_id)) {
    return false;
  }

  const Isometry3T* p_head_keyframe_pose = map_.GetPoseGraphHypothesis().GetKeyframePose(head_keyframe_id);
  if (!p_head_keyframe_pose) {
    return false;
  }
  const PoseGraph::KeyFrame& head_keyframe = pg.GetKeyframe(head_keyframe_id);
  last_keyframe_pose = *p_head_keyframe_pose;
  last_keyframe_ts = head_keyframe.keyframe_info.timestamp_ns;
  return true;
}

// calc covariation from prev VO keyframe
bool LocalizerAndMapper::CalcBetweenPose(
    KeyFrameId from, KeyFrameId to,
    const std::vector<VOFrameData::Track2DXY>& tracks2d_norm,  // normalized coordinates
    const std::map<TrackId, Vector3T>& tracks3d_rel,           // xyz in camera space
    Isometry3T& pose, Matrix6T& covariance) const {
  // xyz in "from" keyframe space
  // uv in "to" keyframe space
  // if keyframe==InvalidKeyframeId - use tracks2d_norm and tracks3d_rel

  std::unordered_map<TrackId, Vector3T> pnp_landmarks;
  std::vector<camera::Observation> pnp_observations;
  pnp_observations.reserve(tracks2d_norm.size());

  // extract tracks exists in staging3d_
  for (auto& track_xy : tracks2d_norm) {
    auto& id = track_xy.track_id;
    auto& last_uv_norm = track_xy.xy;
    auto it_3d = tracks3d_rel.find(id);

    if (it_3d == tracks3d_rel.end()) {
      continue;
    }

    auto& last_xyz = it_3d->second;

    auto it_staging = staging3d_.find(id);

    if (it_staging == staging3d_.end()) {
      continue;  // discard new tracks
    }

    auto& staging = it_staging->second;

    Vector3T xyz;

    if (from == InvalidKeyFrameId) {
      xyz = last_xyz;
    } else {
      auto kfd = staging.FindKeyframe(from);

      if (!kfd) {
        continue;
      }

      xyz = kfd->xyz_rel;
    }

    Vector2T uv_norm;

    if (to == InvalidKeyFrameId) {
      uv_norm = last_uv_norm;
    } else {
      auto kfd = staging.FindKeyframe(to);

      if (!kfd) {
        continue;
      }

      uv_norm = kfd->uv_norm;
    }

    TrackId track_id = pnp_landmarks.size();
    pnp_landmarks.insert({track_id, xyz});

    const camera::ICameraModel& intrinsics = *this->rig_.intrinsics[track_xy.cam_id];
    pnp_observations.emplace_back(track_xy.cam_id, track_id, uv_norm,
                                  camera::ObservationInfoUVToNormUV(intrinsics, camera::GetDefaultObservationInfoUV()));
  }

  Isometry3T rig_from_world_estimate = Isometry3T::Identity();
  Matrix6T precision;
  bool res = pnp_.solve(rig_from_world_estimate, precision, pnp_observations, pnp_landmarks);

  if (!res) {
    return false;
  }

  pose = rig_from_world_estimate.inverse();
  // Eigen::CompleteOrthogonalDecomposition<Matrix6T> psInv(precision);
  covariance = precision.ldlt().solve(Matrix6T::Identity());

  return true;
}

const LocalizerAndMapper::TrackOnKeyframe* LocalizerAndMapper::StagingTrack3d::FindKeyframe(
    KeyFrameId keyframe_id) const {
  for (auto& kf : keyframes)
    if (kf.keyframe_id == keyframe_id) {
      return &kf;
    }
  return nullptr;
}

}  // namespace cuvslam::slam

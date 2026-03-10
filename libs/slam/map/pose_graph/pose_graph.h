
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

#include <functional>
#include <list>
#include <map>

#include "common/isometry.h"
#include "common/unaligned_types.h"
#include "common/vector_3t.h"
#include "math/pgo.h"
#include "profiler/profiler.h"
#include "profiler/profiler_enable.h"

#include "slam/map/database/slam_database.h"
#include "slam/map/pose_graph/pose_graph_hypothesis.h"
#include "slam/map/pose_graph/slam_optimize_options.h"
#include "slam/map/pose_graph/slam_posegraph_keyframe_info.h"

namespace cuvslam::slam {

using EdgeId = size_t;
const EdgeId InvalidEdgeId = ~EdgeId(0);

class PoseGraph {
  const std::string format_and_version_ = "PoseGraph v0.00";
  const uint32_t keyframe_version_ = 2;
  const uint32_t edge_version_ = 0;

public:
  struct KeyFrame {
    PoseGraphKeyframeInfo keyframe_info;
    std::vector<LandmarkId> landmarks;
    std::string frame_information = "";  // use for test dumps
    // to keep spatial volume for landmarks
    std::vector<storage::Pose<float>> merged_keyframe_transforms;
    // hierarchy
    KeyFrameId upper_id = InvalidKeyFrameId;
    std::vector<KeyFrameId>
        lower_keyframes_ids;  // group which represented by this node. First node is "key node" of the group
    storage::Pose<float> group_node_to_this;  // transformation from "key node"

    void AddLandmark(LandmarkId landmark_id);
    void RemoveLandmark(LandmarkId landmark_id);
    bool HasLandmark(LandmarkId landmark_id) const;

    // to keep transforms of merged keyframes
    void AddKeyframeTransform(Isometry3T me, Isometry3T other);
  };
  PoseGraph();
  ~PoseGraph();

  PoseGraph(const PoseGraph&) = delete;
  PoseGraph& operator=(const PoseGraph&) = delete;

  bool PutToDatabase(ISlamDatabase* database) const;
  bool GetFromDatabase(ISlamDatabase* database);

  void Clear();

  KeyFrame& GetKeyframe(KeyFrameId id) { return keyframes_.at(id); }
  const KeyFrame& GetKeyframe(KeyFrameId id) const { return keyframes_.at(id); }

  // Asc the class PoseGraphHypothesis for pose of the keyframe
  // const Isometry3T* PoseGraphHypothesis::GetKeyframePose(KeyFrameId keyframe)

  // head_keyframe will contain last added keyframe or InvalidKeyFrameId
  bool GetHeadKeyframe(KeyFrameId& head_keyframe) const;

  bool GetHeadCovariance(Matrix6T& covariance) const;

  size_t GetKeyframeCount() const;

  // gravity
  PoseGraphKeyframeInfo GetKeyframeInfo(KeyFrameId keyframe) const;

  struct EdgeStat {
    size_t tracks3d_number = 0;
    float square_reprojection_errors = 0;

  public:
    float Weight() const;
  };

  // Add keyframe
  KeyFrameId AddKeyframe(const PoseGraphHypothesis& pgh, const Isometry3T* pose_rel,
                         const Matrix6T* head_pose_covariance, const std::string& frame_information,
                         const PoseGraphKeyframeInfo& extra_keyframe_info, EdgeStat* stat = nullptr);

  void RemoveKeyframe(KeyFrameId keyframe_id);
  // add landmark-keyframe relation
  bool AddLandmarkRelation(LandmarkId landmark_id, KeyFrameId keyframe_id);
  // remove landmark-keyframe relation
  void RemoveLandmarkRelation(LandmarkId landmark_id, KeyFrameId keyframe_id);

  bool OptimizeInternal(const PoseGraphHypothesis& pose_graph_hypothesis_src,
                        PoseGraphHypothesis& pose_graph_hypothesis_dst, Isometry3T& vo_to_head,
                        const OptimizeOptions& options, const std::vector<KeyFrameId>& keyframes_to_optimize,
                        const std::vector<EdgeId>& edges_to_optimize,
                        const std::vector<KeyFrameId>& constraint_keyframes = {}) const;

  // Optimize pose graph
  bool Optimize(const PoseGraphHypothesis& pose_graph_hypothesis_src, PoseGraphHypothesis& pose_graph_hypothesis_dst,
                Isometry3T& vo_to_head,  // transform for last keyframe
                const OptimizeOptions& options) const;

  // Add Edge
  EdgeId AddEdge(const PoseGraphHypothesis& pgh, KeyFrameId from_keyframe, KeyFrameId to_keyframe,
                 const Isometry3T& from_to, const Matrix6T& from_to_covariance, EdgeStat* stat = nullptr);

  // Update edge
  bool UpdateEdge(const PoseGraphHypothesis& pgh, KeyFrameId from_keyframe, KeyFrameId to_keyframe,
                  const Isometry3T& from_to);

  // Set Edge covariance and from_to
  bool SetEdgeCovarianceAndFromTo(const PoseGraphHypothesis& pgh, KeyFrameId to_keyframe, const Matrix6T& covariance,
                                  const Isometry3T& new_to_keyframe_pose);

  // Remove edge
  void RemoveEdge(KeyFrameId from_keyframe, KeyFrameId to_keyframe);

  // methods for merge pose graphs:
  // 1. Create remap index
  bool CreateKeyframeIdRemap(const PoseGraph& pose_graph, std::map<KeyFrameId, KeyFrameId>& keyframe_id_remap) const;
  // 2. Reindex
  bool Reindex(const std::map<KeyFrameId, KeyFrameId>& keyframe_id_remap,
               const std::map<LandmarkId, LandmarkId>& landmark_id_remap);
  // 3. Union
  bool Union(const PoseGraph& pose_graph, bool reassign_head_node);

  //
  const EdgeStat* GetEdgeStatistic(KeyFrameId from, KeyFrameId to) const;

  std::string GetKeyframeInformation(KeyFrameId from) const;

  // Find most reliable pose-graph edge.
  // Return InvalidEdgeId if not found
  EdgeId GetSmallestVarianceEdgeId() const;

  typedef void OnUpdateLandmarkRelation(KeyFrameId old, KeyFrameId neo, const std::vector<LandmarkId>& landmarks,
                                        const Isometry3T& transform_old_to_new);

  // merge keyframes of this edge
  bool ReduceSingleEdge(EdgeId edge_id, const PoseGraphHypothesis& pose_graph_hypothesis,
                        std::function<OnUpdateLandmarkRelation>& func_update_landmark_relation);

  using RemoveNodeCB =
      std::function<void(KeyFrameId keyframe_id, KeyFrameId instead_keyframe_id, const Isometry3T& to_instead)>;
  void RegisterRemoveNodeCB(RemoveNodeCB cb);

protected:
  // Keyframes
  KeyFrameId next_keyframe_auto_id_ = 0;

  std::map<KeyFrameId, KeyFrame> keyframes_;

  RemoveNodeCB remove_node_cb_;

protected:
  bool ToBlob(BlobWriter& blob) const;
  bool FromBlob(const BlobReader& blob);

protected:
  // edge struct
  class Edge {
  public:
    // keyframes
    KeyFrameId from_keyframe, to_keyframe;
    // transform from to to
    Isometry3T from_to;
    // covariance
    Matrix6T from_to_covariance;

    // edge statistic
    EdgeStat statistic;
  };

  EdgeId next_edge_auto_id_ = 0;
  std::map<EdgeId, Edge> edges_;

  // nodes "before" key
  std::map<KeyFrameId, std::list<KeyFrameId>> edges_from_;
  // nodes "after" key
  std::map<KeyFrameId, std::list<KeyFrameId>> edges_to_;
  // from-to
  std::map<std::pair<KeyFrameId, KeyFrameId>, EdgeId> edges_from_to_;

  // tail
  KeyFrameId head_keyframe_id_ = InvalidKeyFrameId;
  // Matrix6T head_pose_covariance_ = Matrix6T::Zero();

  // latest keyframes to guard in FindHardestEdge()
  size_t max_latest_keyframes_ = 20;
  std::list<KeyFrameId> latest_keyframes_;

protected:
  struct OptimizeEdgeInfo {
    KeyFrameId from_id;
    KeyFrameId to_id;
    Isometry3T from_to;
    Matrix6T from_to_covariance;
  };

public:
  // QueryKeyframes([&](KeyFrameId keyframe_id){});
  void QueryKeyframes(const std::function<void(KeyFrameId)>& lambda) const;

  using QueryKeyframeEdgesLambda = std::function<void(KeyFrameId, KeyFrameId, const Isometry3T&, const Matrix6T&)>;
  // Usage:
  // QueryKeyframeEdges(keyframe_id, [&](KeyFrameId from_keyframe, KeyFrameId to_keyframe, const Isometry3T& from_to,
  // const Matrix6T& from_to_covariance){});
  void QueryKeyframeEdges(KeyFrameId keyframe_id, const QueryKeyframeEdgesLambda& func) const;

  // Usage:
  // QueryKeyframeEdges([&](KeyFrameId from_keyframe, KeyFrameId to_keyframe){});
  void QueryEdges(const std::function<bool(KeyFrameId, KeyFrameId, const Isometry3T&, const Matrix6T&)>& func) const;

  // Usage:
  // QueryKeyframeLandmarks(keyframe_id, [&](const LandmarkId& landmark_id){});
  size_t QueryKeyframeLandmarks(KeyFrameId keyframe_id, const std::function<bool(const LandmarkId&)>& func);

  // Usage:
  // QueryKeyframePoses(keyframe_id, [&](const KeyFrameId& keyframe_id, const Isometry3T& pose){});
  void QueryKeyframePoses(const std::function<void(const KeyFrameId&, const Isometry3T&)>& func);

  void QueryEdgeId(KeyFrameId from_keyframe, KeyFrameId to_keyframe, const std::function<void(EdgeId)>& func);

private:
  profiler::SLAMProfiler::DomainHelper profiler_domain_ = profiler::SLAMProfiler::DomainHelper("SLAM");
  uint32_t profiler_color_ = 0x0000FF;

  math::PGO pgo;
  struct Gravity {
    bool gravity_valid = false;
    Vector3T gravity_cf = Vector3T(0.f, -9.81f, 0.f);  // in camera frame (rig frame)
  };

  Gravity dummy_gravity;
};

}  // namespace cuvslam::slam

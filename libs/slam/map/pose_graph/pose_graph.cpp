
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

#include "slam/map/pose_graph/pose_graph.h"

#include "common/log_types.h"
#include "math/twist.h"

namespace cuvslam::slam {

void PoseGraph::KeyFrame::AddLandmark(LandmarkId landmark_id) {
  auto it = std::lower_bound(landmarks.begin(), landmarks.end(), landmark_id);
  if (it != this->landmarks.end() && *it == landmark_id) {
    return;  // already in list
  }
  this->landmarks.insert(it, landmark_id);
}
void PoseGraph::KeyFrame::RemoveLandmark(LandmarkId landmark_id) {
  auto it = std::lower_bound(landmarks.begin(), landmarks.end(), landmark_id);
  if (it == this->landmarks.end() || *it != landmark_id) {
    return;  // not in list
  }
  this->landmarks.erase(it);
}
bool PoseGraph::KeyFrame::HasLandmark(LandmarkId landmark_id) const {
  auto it = std::lower_bound(landmarks.begin(), landmarks.end(), landmark_id);
  if (it == this->landmarks.end() || *it != landmark_id) {
    return false;  // not in list
  }
  return true;
}

void PoseGraph::KeyFrame::AddKeyframeTransform(Isometry3T me, Isometry3T other) {
  Isometry3T from_to = me.inverse() * other;
  merged_keyframe_transforms.push_back(from_to);
}

float PoseGraph::EdgeStat::Weight() const {
  // TODO: weight
  return static_cast<float>(tracks3d_number);
}

PoseGraph::PoseGraph() {}

PoseGraph::~PoseGraph() { SlamStdout("Destroyed PoseGraph instance. "); }

bool PoseGraph::PutToDatabase(ISlamDatabase* database) const {
  // copy all to DB
  return database->SetSingleton(SlamDatabaseSingleton::PoseGraph, 0,
                                [&](BlobWriter& blob_writer) { return this->ToBlob(blob_writer); });
}
bool PoseGraph::GetFromDatabase(ISlamDatabase* database) {
  Clear();
  return database->GetSingleton(SlamDatabaseSingleton::PoseGraph,
                                [&](const BlobReader& blob_reader) { return this->FromBlob(blob_reader); });
}

void PoseGraph::Clear() {
  keyframes_.clear();
  edges_.clear();
  edges_from_to_.clear();
  edges_from_.clear();
  edges_to_.clear();
  head_keyframe_id_ = InvalidKeyFrameId;
  // head_pose_covariance_ = Matrix6T::Zero();

  next_edge_auto_id_ = 0;
}

bool PoseGraph::GetHeadKeyframe(KeyFrameId& head_keyframe) const {
  head_keyframe = head_keyframe_id_;
  return head_keyframe_id_ != InvalidKeyFrameId;
}

bool PoseGraph::GetHeadCovariance(Matrix6T& covariance) const {
  KeyFrameId head_keyframe;
  if (!GetHeadKeyframe(head_keyframe)) {
    return false;
  }

  covariance = Matrix6T::Zero();
  for (auto to_keyframe = head_keyframe;;) {
    auto it = edges_from_.find(to_keyframe);
    if (it == edges_from_.end()) {
      return true;
    }

    auto& list = it->second;
    if (list.size() != 1) {
      return true;
    }

    auto from_keyframe = list.front();

    std::pair<KeyFrameId, KeyFrameId> key(from_keyframe, to_keyframe);
    auto it_edges_from_to = edges_from_to_.find(key);
    if (it_edges_from_to == edges_from_to_.end()) {
      return true;
    }
    auto edge_id = it_edges_from_to->second;

    auto it_e = edges_.find(edge_id);
    if (it_e == edges_.end()) {
      return true;
    }

    covariance += it_e->second.from_to_covariance;

    to_keyframe = from_keyframe;
  }
  return true;
}

size_t PoseGraph::GetKeyframeCount() const { return keyframes_.size(); }

// gravity
PoseGraphKeyframeInfo PoseGraph::GetKeyframeInfo(KeyFrameId keyframe_id) const {
  auto it = keyframes_.find(keyframe_id);
  if (it == keyframes_.end()) {
    return PoseGraphKeyframeInfo();
  }
  auto& keyframe = it->second;
  return keyframe.keyframe_info;
}

KeyFrameId PoseGraph::AddKeyframe(const PoseGraphHypothesis& pgh, const Isometry3T* pose_rel,
                                  const Matrix6T* head_pose_covariance, const std::string& frame_information,
                                  const PoseGraphKeyframeInfo& extra_keyframe_info, EdgeStat* stat) {
  TRACE_EVENT ev = profiler_domain_.trace_event("AddKeyframe", profiler_color_);
  KeyFrameId from_keyframe;
  bool has_head = GetHeadKeyframe(from_keyframe);
  auto keyframe_id = next_keyframe_auto_id_++;

  if (keyframes_.find(keyframe_id) != this->keyframes_.end()) {
    SlamStderr("Failed to add keyframe to pose graph with auto_id.\n");
    return InvalidKeyFrameId;
  }

  auto& keyFrame = keyframes_[keyframe_id];
  keyFrame.frame_information = frame_information;
  keyFrame.keyframe_info = extra_keyframe_info;
  head_keyframe_id_ = keyframe_id;

  // add edge
  if (has_head && pose_rel && head_pose_covariance) {
    AddEdge(pgh, from_keyframe, keyframe_id, *pose_rel, *head_pose_covariance, stat);
  }

  // latest_keyframes_. For proper GetSmallestVarianceEdgeId()
  latest_keyframes_.push_back(keyframe_id);
  while (latest_keyframes_.size() > max_latest_keyframes_) {
    latest_keyframes_.pop_front();
  }

  log::Message<LogFrames>(log::kInfo, "AddKeyframe() = %zd", keyframe_id);
  return keyframe_id;
}

void PoseGraph::RemoveKeyframe(KeyFrameId keyframe_id) {
  TRACE_EVENT ev = profiler_domain_.trace_event("RemoveKeyframe", profiler_color_);
  // remove all edges with keyframe_id
  std::vector<std::pair<KeyFrameId, KeyFrameId>> edge_pairs;
  this->QueryKeyframeEdges(keyframe_id,
                           [&](KeyFrameId from_keyframe, KeyFrameId to_keyframe, const Isometry3T&, const Matrix6T&) {
                             edge_pairs.push_back({from_keyframe, to_keyframe});
                           });

  for (auto& edge_pair : edge_pairs) {
    KeyFrameId from_keyframe = edge_pair.first;
    KeyFrameId to_keyframe = edge_pair.second;
    // RemoveEdge
    this->RemoveEdge(from_keyframe, to_keyframe);
  }

  keyframes_.erase(keyframe_id);
}

// add landmark relation
bool PoseGraph::AddLandmarkRelation(LandmarkId landmark_id, KeyFrameId keyframe_id) {
  auto it = keyframes_.find(keyframe_id);
  if (it == keyframes_.end()) {
    return false;
  }
  it->second.AddLandmark(landmark_id);
  return true;
}

void PoseGraph::RemoveLandmarkRelation(LandmarkId landmark_id, KeyFrameId keyframe_id) {
  auto it = keyframes_.find(keyframe_id);
  if (it == keyframes_.end()) {
    return;
  }
  auto& keyframe = it->second;
  keyframe.RemoveLandmark(landmark_id);
}

EdgeId PoseGraph::AddEdge(const PoseGraphHypothesis& pgh, KeyFrameId from_keyframe, KeyFrameId to_keyframe,
                          const Isometry3T& from_to, const Matrix6T& from_to_covariance, EdgeStat* stat) {
  TRACE_EVENT ev = profiler_domain_.trace_event("AddEdge", profiler_color_);
  (void)pgh;
  if (from_keyframe == to_keyframe) {
    SlamStderr("Failed to add edge to pose graph (%zd->%zd).", from_keyframe, to_keyframe);
    return InvalidEdgeId;
  }
  if (keyframes_.find(from_keyframe) == keyframes_.end()) {
    SlamStderr("Failed to add edge to pose graph (%zd->%zd): %zd does not exist.", from_keyframe, to_keyframe,
               from_keyframe);
    return InvalidEdgeId;
  }
  if (keyframes_.find(to_keyframe) == keyframes_.end()) {
    SlamStderr("Failed to add edge to pose graph (%zd->%zd): %zd does not exist.", from_keyframe, to_keyframe,
               to_keyframe);
    return InvalidEdgeId;
  }

  EdgeId id;
  auto it_edges_from_to = edges_from_to_.find(std::pair<KeyFrameId, KeyFrameId>(from_keyframe, to_keyframe));
  if (it_edges_from_to == edges_from_to_.end()) {
    // Add new edge
    id = next_edge_auto_id_++;

    edges_from_to_[std::pair<KeyFrameId, KeyFrameId>(from_keyframe, to_keyframe)] = id;

    auto& from_list = edges_from_[to_keyframe];
    from_list.push_back(from_keyframe);

    auto& to_list = edges_to_[from_keyframe];
    to_list.push_back(to_keyframe);
  } else {
    // update exists edge
    id = it_edges_from_to->second;
  }

  auto& edge = edges_[id];
  edge.from_keyframe = from_keyframe;
  edge.to_keyframe = to_keyframe;
  edge.from_to = from_to;
  edge.from_to_covariance = from_to_covariance;
  if (stat) {
    edge.statistic = *stat;
  }
  return id;
}

// Update edge
bool PoseGraph::UpdateEdge(const PoseGraphHypothesis& pgh, KeyFrameId from_keyframe, KeyFrameId to_keyframe,
                           const Isometry3T& from_to) {
  TRACE_EVENT ev = profiler_domain_.trace_event("UpdateEdge", profiler_color_);
  (void)pgh;
  auto it_edges = edges_from_to_.find(std::pair<KeyFrameId, KeyFrameId>(from_keyframe, to_keyframe));
  if (it_edges == edges_from_to_.end()) {
    SlamStderr("Failed to update edge in pose graph ( %zd->%zd).", from_keyframe, to_keyframe);
    return false;
  }
  EdgeId edge_id = it_edges->second;
  auto it = edges_.find(edge_id);
  if (it == edges_.end()) {
    SlamStderr("Failed to update edge in pose graph ( %zd->%zd).", from_keyframe, to_keyframe);
    return false;
  }

  Edge& edge = it->second;
  edge.from_to = from_to;
  return true;
}

// Set Edge covariance and from_to
bool PoseGraph::SetEdgeCovarianceAndFromTo(const PoseGraphHypothesis& pgh, KeyFrameId to_keyframe,
                                           const Matrix6T& covariance, const Isometry3T& new_to_keyframe_pose) {
  const auto it_edge_to = edges_from_.find(to_keyframe);
  if (it_edge_to == edges_from_.end()) {
    return false;
  }
  const KeyFrameId from_keyframe = it_edge_to->second.front();
  const auto it_edges = edges_from_to_.find(std::pair<KeyFrameId, KeyFrameId>(from_keyframe, to_keyframe));
  if (it_edges == edges_from_to_.end()) {
    return false;
  }
  const EdgeId edge_id = it_edges->second;
  const auto it = edges_.find(edge_id);
  if (it == edges_.end()) {
    return false;
  }

  Edge& edge = it->second;
  edge.from_to_covariance = covariance;

  if (pgh.GetKeyframePose(from_keyframe) == nullptr) {
    return false;
  }
  const Isometry3T from_pose = *pgh.GetKeyframePose(from_keyframe);
  const Isometry3T from_to = from_pose.inverse() * new_to_keyframe_pose;
  edge.from_to = from_to;
  return true;
}

// Remove edge
void PoseGraph::RemoveEdge(KeyFrameId from_keyframe, KeyFrameId to_keyframe) {
  TRACE_EVENT ev = profiler_domain_.trace_event("RemoveEdge", profiler_color_);
  auto it_edges = edges_from_to_.find(std::pair<KeyFrameId, KeyFrameId>(from_keyframe, to_keyframe));
  if (it_edges != edges_from_to_.end()) {
    EdgeId edge_id = it_edges->second;
    edges_.erase(edge_id);

    edges_from_to_.erase(it_edges);
  }

  auto it_from = edges_from_.find(to_keyframe);
  if (it_from != edges_from_.end()) {
    std::list<KeyFrameId>& list = it_from->second;
    list.remove(from_keyframe);
  }

  auto it_to = edges_to_.find(from_keyframe);
  if (it_to != edges_to_.end()) {
    std::list<KeyFrameId>& list = it_to->second;
    list.remove(to_keyframe);
  }
}

bool PoseGraph::CreateKeyframeIdRemap(const PoseGraph& pose_graph,
                                      std::map<KeyFrameId, KeyFrameId>& keyframe_id_remap) const {
  // find next_keyframe_auto_id
  KeyFrameId next_keyframe_auto_id = 0;
  for (const auto& [const_id, _] : pose_graph.keyframes_) {
    next_keyframe_auto_id = std::max(next_keyframe_auto_id, const_id + 1);
  }

  // fill keyframe_id_remap
  for (auto& it : this->keyframes_) {
    const KeyFrameId src_id = it.first;
    auto keyframe_id = next_keyframe_auto_id++;
    keyframe_id_remap[src_id] = keyframe_id;
  }
  return true;
}

bool PoseGraph::Reindex(const std::map<KeyFrameId, KeyFrameId>& keyframe_id_remap,
                        const std::map<LandmarkId, LandmarkId>& landmark_id_remap) {
  auto reindex_keyframe = [&](KeyFrameId src, KeyFrameId& dst) {
    auto it_remap = keyframe_id_remap.find(src);
    if (it_remap == keyframe_id_remap.end()) {
      return false;
    }
    dst = it_remap->second;
    return true;
  };

  auto reindex_landmarks = [&](LandmarkId src, LandmarkId& dst) {
    auto it_remap = landmark_id_remap.find(src);
    if (it_remap == landmark_id_remap.end()) {
      return false;
    }
    dst = it_remap->second;
    return true;
  };

  // keyframes
  std::map<KeyFrameId, KeyFrame> keyframes;
  for (auto& it : this->keyframes_) {
    KeyFrameId keyframe_id;
    if (!reindex_keyframe(it.first, keyframe_id)) {
      return false;
    }
    keyframes[keyframe_id] = it.second;

    auto& keyframe = keyframes[keyframe_id];
    // reindex landmarks
    for (auto& landmark_id : keyframe.landmarks) {
      if (!reindex_landmarks(landmark_id, landmark_id)) {
        landmark_id = InvalidLandmarkId;
      }
    }
    std::sort(keyframe.landmarks.begin(), keyframe.landmarks.end());

    // todo: hierarchy
    // keyframe.upper_id = InvalidKeyFrameId;
    // keyframe.lower_keyframes_ids;    // group which represented by this node. First node is "key node" of the group
    // keyframe.group_node_to_this;     // Transformation from "key node"
  }

  // nodes "before" key
  std::map<KeyFrameId, std::list<KeyFrameId>> edges_from;
  for (auto& it : edges_from_) {
    KeyFrameId keyframe_id;
    if (!reindex_keyframe(it.first, keyframe_id)) {
      return false;
    }
    edges_from[keyframe_id] = it.second;
    auto& list = edges_from[keyframe_id];
    for (auto& id_keyframe : list) {
      if (!reindex_keyframe(id_keyframe, id_keyframe)) {
        return false;
      }
    }
  }
  // nodes "after" key
  std::map<KeyFrameId, std::list<KeyFrameId>> edges_to;
  for (auto& it : edges_to_) {
    KeyFrameId keyframe_id;
    if (!reindex_keyframe(it.first, keyframe_id)) {
      return false;
    }
    edges_to[keyframe_id] = it.second;
    auto& list = edges_to[keyframe_id];
    for (auto& id_keyframe : list) {
      if (!reindex_keyframe(id_keyframe, id_keyframe)) {
        return false;
      }
    }
  }

  // from-to
  std::map<std::pair<KeyFrameId, KeyFrameId>, EdgeId> edges_from_to;
  for (auto& it : edges_from_to_) {
    std::pair<KeyFrameId, KeyFrameId> id;
    if (!reindex_keyframe(it.first.first, id.first)) {
      return false;
    }
    if (!reindex_keyframe(it.first.second, id.second)) {
      return false;
    }
    edges_from_to[id] = it.second;
  }

  std::map<EdgeId, Edge> edges;
  for (const auto& it : edges_) {
    Edge& edge = edges[it.first];
    edge = it.second;
    if (!reindex_keyframe(edge.from_keyframe, edge.from_keyframe)) {
      return false;
    }
    if (!reindex_keyframe(edge.to_keyframe, edge.to_keyframe)) {
      return false;
    }
  }

  // latest_keyframes
  std::list<KeyFrameId> latest_keyframes = latest_keyframes_;
  for (auto& it : latest_keyframes) {
    if (!reindex_keyframe(it, it)) {
      return false;
    }
  }

  // tail
  if (head_keyframe_id_ != InvalidKeyFrameId) {
    if (!reindex_keyframe(head_keyframe_id_, head_keyframe_id_)) {
      return false;
    }
  }

  // next_keyframe_auto_id
  KeyFrameId next_keyframe_auto_id = 0;
  for (const auto& [id, _] : keyframes) {
    next_keyframe_auto_id = std::max(next_keyframe_auto_id, id + 1);
  }

  // copy
  this->keyframes_ = keyframes;
  this->edges_ = edges;
  this->edges_from_ = edges_from;
  this->edges_to_ = edges_to;
  this->edges_from_to_ = edges_from_to;
  this->latest_keyframes_ = latest_keyframes;
  this->next_keyframe_auto_id_ = next_keyframe_auto_id;
  return true;
}

bool PoseGraph::Union(const PoseGraph& pose_graph, bool reassign_head_node) {
  {
    // Reindex EdgeId
    EdgeId next_edge_auto_id = 0;
    for (auto& it : pose_graph.edges_) {
      EdgeId id = it.first;
      next_edge_auto_id = std::max(next_edge_auto_id, id + 1);
    }
    this->next_edge_auto_id_ = next_edge_auto_id;
    std::map<EdgeId, EdgeId> edge_id_remap;
    std::map<EdgeId, Edge> edges;
    for (auto& it : this->edges_) {
      EdgeId src_id = it.first;
      EdgeId edge_id = this->next_edge_auto_id_++;
      auto& edge = edges[edge_id];
      edge = it.second;
      edge_id_remap[src_id] = edge_id;
    }

    auto reindex_edge = [&](EdgeId src, EdgeId& dst) {
      auto it_remap = edge_id_remap.find(src);
      if (it_remap == edge_id_remap.end()) {
        return false;
      }
      dst = it_remap->second;
      return true;
    };

    // edges_from_to_
    for (auto& it : edges_from_to_) {
      if (!reindex_edge(it.second, it.second)) {
        return false;
      }
    }

    this->edges_ = edges;
  }

  // copy all from const_pose_graph to this
  for (auto& it : pose_graph.keyframes_) {
    if (this->keyframes_.find(it.first) != this->keyframes_.end()) {
      SlamStderr("Failed to union keyframes %zd.", it.first);
    }
    this->keyframes_[it.first] = it.second;
  }
  for (auto& it : pose_graph.edges_from_) {
    if (this->edges_from_.find(it.first) != this->edges_from_.end()) {
      SlamStderr("Failed to union edges from %zd.", it.first);
    }
    this->edges_from_[it.first] = it.second;
  }
  for (auto& it : pose_graph.edges_to_) {
    if (this->edges_to_.find(it.first) != this->edges_to_.end()) {
      SlamStderr("Failed to union edges to %zd.", it.first);
    }
    this->edges_to_[it.first] = it.second;
  }
  for (auto& it : pose_graph.edges_from_to_) {
    if (this->edges_from_to_.find(it.first) != this->edges_from_to_.end()) {
      SlamStderr("Failed to union edges [%zd->%zd].", it.first.first, it.first.second);
    }
    this->edges_from_to_[it.first] = it.second;
  }
  for (auto& it : pose_graph.edges_) {
    if (this->edges_.find(it.first) != this->edges_.end()) {
      SlamStderr("Failed to union edges %zd.", it.first);
    }
    this->edges_[it.first] = it.second;
  }

  if (reassign_head_node) {
    // Importantly?
    pose_graph.GetHeadKeyframe(this->head_keyframe_id_);
  }

  next_keyframe_auto_id_ = std::max(next_keyframe_auto_id_, pose_graph.next_keyframe_auto_id_);

  // don't copy latest_keyframes
  // this->latest_keyframes_ = const_pose_graph->latest_keyframes;
  return true;
}

//
const PoseGraph::EdgeStat* PoseGraph::GetEdgeStatistic(KeyFrameId from_keyframe, KeyFrameId to_keyframe) const {
  std::pair<KeyFrameId, KeyFrameId> key(from_keyframe, to_keyframe);
  auto it = this->edges_from_to_.find(key);
  if (it == this->edges_from_to_.end()) {
    return nullptr;
  }
  auto& edge = edges_.at(it->second);
  return &edge.statistic;
}

std::string PoseGraph::GetKeyframeInformation(KeyFrameId keyframe_id) const {
  auto it = this->keyframes_.find(keyframe_id);
  if (it == this->keyframes_.end()) {
    return "";
  }
  return it->second.frame_information;
}

void PoseGraph::RegisterRemoveNodeCB(RemoveNodeCB cb) { remove_node_cb_ = cb; }

bool PoseGraph::ToBlob(BlobWriter& blob) const {
  auto keyframe_to_blob = [&](KeyFrameId keyframe_id, const KeyFrame& keyframe) {
    blob.write(this->keyframe_version_);
    blob.write(keyframe_id);
    blob.write_std(keyframe.landmarks);
    blob.write_std(keyframe.merged_keyframe_transforms);
    blob.write_str(keyframe.frame_information);
    {
      // keyframe_info
      blob.write(keyframe.keyframe_info.current_version);
      blob.write(keyframe.keyframe_info.timestamp_ns);

      // This is not mandatory, but we maintain this to save compatibility with
      // previously created maps and avoid wrong keyframe version in map
      blob.write(dummy_gravity);
    }
  };
  auto edge_to_blob = [&](EdgeId edge_id, const Edge& edge) {
    blob.write(this->edge_version_);
    blob.write(edge_id);
    blob.write(edge.from_keyframe);
    blob.write(edge.to_keyframe);
    blob.write_eigen(edge.from_to);
    blob.write_eigen(edge.from_to_covariance);

    // edge statistic
    blob.write(edge.statistic.tracks3d_number);
    blob.write(edge.statistic.square_reprojection_errors);
  };

  blob.write_str(this->format_and_version_);

  blob.write(next_keyframe_auto_id_);
  blob.write(next_edge_auto_id_);
  blob.write(head_keyframe_id_);

  blob.write(keyframes_.size());
  for (auto& it : keyframes_) {
    keyframe_to_blob(it.first, it.second);
  }
  blob.write(edges_.size());
  for (auto& it : edges_) {
    edge_to_blob(it.first, it.second);
  }
  return true;
}
bool PoseGraph::FromBlob(const BlobReader& blob) {
  auto keyframe_from_blob = [&](KeyFrameId& keyframe_id, KeyFrame& keyframe) {
    uint32_t keyframe_version;
    if (!blob.read(keyframe_version)) {
      SlamStderr("Can't read keyframe version in pose graph.\n");
      return false;
    }
    if (keyframe_version != this->keyframe_version_) {
      SlamStderr("Wrong keyframe version in pose graph: %d!=%d.\n", keyframe_version, this->keyframe_version_);
      return false;
    }
    bool res = blob.read(keyframe_id) && blob.read_std(keyframe.landmarks) &&
               blob.read_std(keyframe.merged_keyframe_transforms) && blob.read_str(keyframe.frame_information);
    // keyframe_info
    uint32_t ki_version;
    res &= blob.read(ki_version);
    if (!res) {
      SlamStderr("Can't read keyframe_info version in pose graph.\n");
      return false;
    }
    if (ki_version != keyframe.keyframe_info.current_version) {
      SlamStderr("Wrong keyframe_info version in pose graph: %d!=%d.\n", ki_version,
                 keyframe.keyframe_info.current_version);
      return false;
    }
    res &= blob.read(keyframe.keyframe_info.timestamp_ns);

    // This is not mandatory, but we maintain this to save compatibility with
    // previously created maps and avoid wrong keyframe version in map
    res &= blob.read(dummy_gravity);
    return res;
  };
  auto edge_from_blob = [&](EdgeId& edge_id, Edge& edge) {
    uint32_t edge_version;

    if (!blob.read(edge_version)) {
      SlamStderr("Can't read edge version in pose graph.\n");
      return false;
    }
    if (edge_version != this->edge_version_) {
      SlamStderr("Wrong edge version in pose graph: %d!=%d.\n", edge_version, this->edge_version_);
      return false;
    }

    return blob.read(edge_id) && blob.read(edge.from_keyframe) && blob.read(edge.to_keyframe) &&
           blob.read_eigen(edge.from_to) && blob.read_eigen(edge.from_to_covariance) &&
           // edge statistic
           blob.read(edge.statistic.tracks3d_number) && blob.read(edge.statistic.square_reprojection_errors);
  };

  std::string format_and_version;
  if (!blob.read_str(format_and_version) || format_and_version != this->format_and_version_) {
    SlamStderr("Failed to read LSIGrid: wrong version %s.\n", format_and_version.c_str());
    SlamStderr("Current is %s.\n", this->format_and_version_.c_str());
    return false;
  }

  blob.read(next_keyframe_auto_id_);
  blob.read(next_edge_auto_id_);
  blob.read(head_keyframe_id_);

  size_t sz = 0;
  blob.read(sz);
  for (size_t i = 0; i < sz; i++) {
    KeyFrameId keyframe_id;
    KeyFrame keyframe;
    if (!keyframe_from_blob(keyframe_id, keyframe)) {
      return false;
    }
    keyframes_[keyframe_id] = keyframe;
  }

  blob.read(sz);
  for (size_t i = 0; i < sz; i++) {
    EdgeId edge_id;
    Edge edge;
    if (!edge_from_blob(edge_id, edge)) {
      return false;
    }
    edges_[edge_id] = edge;

    edges_to_[edge.from_keyframe].push_back(edge.to_keyframe);
    edges_from_[edge.to_keyframe].push_back(edge.from_keyframe);
    std::pair<KeyFrameId, KeyFrameId> key(edge.from_keyframe, edge.to_keyframe);
    edges_from_to_[key] = edge_id;
  }
  return true;
}

void PoseGraph::QueryKeyframes(const std::function<void(KeyFrameId)>& lambda) const {
  for (auto& it : keyframes_) {
    lambda(it.first);
  }
}

void PoseGraph::QueryKeyframeEdges(KeyFrameId keyframe_id, const QueryKeyframeEdgesLambda& func) const {
  // edges_from_
  {
    auto it = edges_from_.find(keyframe_id);
    if (it != edges_from_.end()) {
      auto& list = it->second;
      for (KeyFrameId from_id : list) {
        auto it_edges_from_to = edges_from_to_.find(std::pair<KeyFrameId, KeyFrameId>(from_id, keyframe_id));
        if (it_edges_from_to == edges_from_to_.end()) {
          continue;
        }
        EdgeId edge_id = it_edges_from_to->second;
        auto it_edge = edges_.find(edge_id);
        if (it_edge == edges_.end()) {
          continue;
        }
        auto& edge = it_edge->second;
        func(edge.from_keyframe, edge.to_keyframe, edge.from_to, edge.from_to_covariance);
      }
    }
  }
  // edges_to_
  {
    auto it = edges_to_.find(keyframe_id);
    if (it != edges_to_.end()) {
      auto& list = it->second;
      for (KeyFrameId to_id : list) {
        auto it_edges_from_to = edges_from_to_.find(std::pair<KeyFrameId, KeyFrameId>(keyframe_id, to_id));
        if (it_edges_from_to == edges_from_to_.end()) {
          continue;
        }
        EdgeId edge_id = it_edges_from_to->second;
        auto it_edge = edges_.find(edge_id);
        if (it_edge == edges_.end()) {
          continue;
        }
        auto& edge = it_edge->second;
        func(edge.from_keyframe, edge.to_keyframe, edge.from_to, edge.from_to_covariance);
      }
    }
  }
}

void PoseGraph::QueryEdges(
    const std::function<bool(KeyFrameId, KeyFrameId, const Isometry3T&, const Matrix6T&)>& func) const {
  for (const auto& x : edges_) {
    const Edge& edge = x.second;
    if (!func(edge.from_keyframe, edge.to_keyframe, edge.from_to, edge.from_to_covariance)) {
      break;
    }
  }
}

size_t PoseGraph::QueryKeyframeLandmarks(KeyFrameId keyframe_id, const std::function<bool(const LandmarkId&)>& func) {
  auto it = keyframes_.find(keyframe_id);
  if (it == keyframes_.end()) {
    return 0;
  }
  const auto& kf = it->second;
  for (auto& it1 : kf.landmarks) {
    if (!func(it1)) {
      return kf.landmarks.size();
    }
  }
  return kf.landmarks.size();
}

void PoseGraph::QueryKeyframePoses(const std::function<void(const KeyFrameId&, const Isometry3T&)>& func) {
  for (auto& it : keyframes_) {
    const KeyFrameId& keyframe_id = it.first;
    auto& keyframe = it.second;

    func(keyframe_id, Isometry3T::Identity());
    for (auto& store_pose : keyframe.merged_keyframe_transforms) {
      Isometry3T pose(store_pose);
      func(keyframe_id, pose);
    }
  }
}

void PoseGraph::QueryEdgeId(KeyFrameId from_keyframe, KeyFrameId to_keyframe, const std::function<void(EdgeId)>& func) {
  auto it = edges_from_to_.find({from_keyframe, to_keyframe});
  if (it == edges_from_to_.end()) {
    SlamStderr("Invalid graph edge (%zd-%zd) in pose graph.\n", from_keyframe, to_keyframe);
  } else {
    func(edges_from_to_[{from_keyframe, to_keyframe}]);
  }
}

}  // namespace cuvslam::slam

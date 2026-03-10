
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

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "camera/rig.h"
#include "common/camera_id.h"

namespace cuvslam::camera {

enum class MulticameraMode {
  // each secondary camera must be connected to only one primary camera
  Performance,
  // all cameras are primary cameras
  Precision,
  // primary camera(s) are cameras with highest degree, intersecting cameras are secondary
  Moderate,
  // primary/secondary cameras are set manually; no checks are done
  Manual
};

using MulticamManualSetup = std::vector<std::vector<CameraId>>;

// Building a camera graph without the need for manual assumption hard-coding of the camera configuration.
// It leverages camera intrinsic, distortion and extrinsic calibrations in order to check overlapping between
// the camera frustums and define the stereo pairs.
class FrustumIntersectionGraph {
public:
  // description of a camera in camera configuration in the context of a graph representation
  struct CameraGraphNode {
    struct ConnectedCam {
      ConnectedCam(CameraId id_, float fir_) : id(id_), frustrim_intersection_ratio(fir_){};
      CameraId id;
      float frustrim_intersection_ratio;  // 0 < FIR < 1;
    };
    // the vector of camera_id_js that forms a stereo pair with camera_id_i
    std::vector<ConnectedCam> stereo_camera_pairs;
    // the number of camera_id_js that forms a stereo_pair with camera_id_i
    size_t degree_of_vertex = 0;
  };

  FrustumIntersectionGraph() = default;

  FrustumIntersectionGraph(const camera::Rig& rig, MulticameraMode mode = MulticameraMode::Performance,
                           const std::vector<CameraId>& depth_ids = {}, bool allow_stereo_track_for_depth = false,
                           const MulticamManualSetup& manual_setup = {});

  FrustumIntersectionGraph(const std::vector<CameraGraphNode>& graph,
                           MulticameraMode mode = MulticameraMode::Performance,
                           const std::vector<CameraId>& depth_ids = {}, bool allow_stereo_track_for_depth = false,
                           const MulticamManualSetup& manual_setup = {});

  // primary camera is tracked on every frame
  const std::vector<CameraId>& primary_cameras() const;

  // we track primary -> secondary only on keyframes
  const std::vector<CameraId>& secondary_cameras(CameraId primary_camera) const;

  bool is_valid() const;

private:
  void set_precision_mode(const std::vector<CameraGraphNode>& graph);
  void set_performance_mode(const std::vector<CameraGraphNode>& graph);
  void set_moderate_mode(const std::vector<CameraGraphNode>& graph);
  void set_manual_mode(const std::vector<CameraGraphNode>& graph, const MulticamManualSetup& manual_setup);

  std::vector<CameraId> primary_cameras_;
  std::vector<CameraId> depth_ids_;
  std::unordered_map<CameraId, std::vector<CameraId>> secondary_from_primary_;
};

}  // namespace cuvslam::camera

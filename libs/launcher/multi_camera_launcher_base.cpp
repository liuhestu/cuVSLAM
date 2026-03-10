
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

#include "launcher/multi_camera_launcher_base.h"

#include "gflags/gflags.h"

DEFINE_bool(stereo_track_for_depth, false, "Use stereo tracking for the RGB camera");
DEFINE_int32(slam_camera, -1, "Select single camera for slam. If -1, all cameras will be used");

namespace cuvslam::launcher {
MultiCameraBaseLauncher::MultiCameraBaseLauncher(ICameraRig& cameraRig, const odom::Settings& svo_settings)
    : BaseLauncher(cameraRig, svo_settings) {
  const std::vector<CameraId> depth_ids = cameraRig_.getCamerasWithDepth();

  fig = camera::FrustumIntersectionGraph(rig, svo_settings_.sof_settings.multicam_mode, depth_ids,
                                         FLAGS_stereo_track_for_depth, svo_settings_.sof_settings.multicam_setup);
  const std::vector<CameraId>& prim_cams = fig.primary_cameras();
  {
    std::stringstream s;
    s << "Fig setup: ";
    for (const CameraId& prim_id : prim_cams) {
      s << static_cast<int>(prim_id) << ": [";
      for (const CameraId& sec_id : fig.secondary_cameras(prim_id)) {
        s << static_cast<int>(sec_id) << ", ";
      }
      s << "] | ";
    }
    TraceMessage(s.str().c_str());
  }

  if (!fig.is_valid()) {
    throw std::runtime_error(
        "Bad calibration. cuVSLAM is supposed to work with at least"
        " one stereo pair available.");
  }

  if (FLAGS_slam_camera != -1) {
    auto it = std::find(prim_cams.begin(), prim_cams.end(), FLAGS_slam_camera);
    if (it == prim_cams.end()) {
      std::stringstream ss;
      ss << "FLAGS_slam_camera must be one of: {";
      for (CameraId cam_id : prim_cams) {
        ss << static_cast<int>(cam_id) << " ";
      }
      ss << "}";
      throw std::runtime_error("Bad calibration. cuVSLAM is supposed to work with at least one stereo pair available.");
    }
    assert(*it == FLAGS_slam_camera);
    const CameraId manual_selected_slam_camera = *it;
    slam_cameras_ = {manual_selected_slam_camera};
  } else {
    slam_cameras_ = prim_cams;
  }
}
}  // namespace cuvslam::launcher

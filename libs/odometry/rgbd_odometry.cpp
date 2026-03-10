
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

#include "odometry/rgbd_odometry.h"

#include "math/twist.h"
#include "pipelines/feature_predictor.h"
#include "sof/sof_create.h"

namespace cuvslam::odom {

RGBDOdometry::RGBDOdometry(const camera::Rig& rig, const camera::FrustumIntersectionGraph& fig,
                           const Settings& settings, bool use_gpu)

    : fig_(fig),
      settings_(settings),
      map_(20),
      feature_predictor_(std::make_shared<pipelines::FeaturePredictor>(map_, rig)),
      solver_(map_, rig, settings.sba_settings) {
  sof::Implementation implementation = sof::Implementation::kCPU;
  if (use_gpu) {
#ifdef USE_CUDA
    implementation = sof::Implementation::kGPU;
#else
    TraceError("To use GPU SOF one must use USE_CUDA=true cmake option");
#endif
  }
  feature_tracker_ =
      sof::CreateMultiSOF(implementation, rig, fig_, feature_predictor_, settings.sof_settings, settings.kf_settings);
}

void RGBDOdometry::reset() {
  prediction_model_.reset();  // don't do any prediction until two frames tracked successfully
  feature_tracker_->reset();
  solver_.reset();
  map_.clear();
}

bool RGBDOdometry::track(const Sources& curr_sources, const DepthSources& depth_sources, sof::Images& curr_images,
                         const sof::Images& prev_images, const Sources& masks_sources, Isometry3T& delta,
                         Matrix6T& static_info_exp) {
  if (curr_images.empty()) {
    reset();
    delta = Isometry3T::Identity();
    static_info_exp.setZero();
    TraceError("Failed to track, images are not available");
    return false;
  }
  TRACE_EVENT ev = profiler_domain_.trace_event("RGBDOdometry::track()", profiler_color_);
  const int64_t timestamp = curr_images.begin()->second->get_image_meta().timestamp;  // current frame timestamp
  Isometry3T predicted_world_from_rig = prev_world_from_rig_;

  if (settings_.use_prediction) {
    do_predict(&prediction_model_, timestamp, predicted_world_from_rig);
  }

  sof::FrameState frame_type;
  std::unordered_map<CameraId, std::vector<camera::Observation>> observations;

  const bool track_result = feature_tracker_->trackNextFrame(curr_sources, curr_images, prev_images, masks_sources,
                                                             predicted_world_from_rig, observations, frame_type);
  if (!track_result) {
    reset();
    delta = Isometry3T::Identity();
    static_info_exp.setZero();
    TraceError("Failed to track on the 2D tracking stage");
    return false;
  }

  IVisualOdometry::VOFrameStat* stat = last_frame_stat_.get();
  std::vector<Track2D>* tracks2d = stat ? &(stat->tracks2d) : nullptr;
  Tracks3DMap* tracks3d = stat ? &(stat->tracks3d) : nullptr;
  Isometry3T world_from_rig;

  bool depth_icp = false;
  CameraId camera_id_icp;

  for (const auto& [cam_id, image] : curr_images) {
    if (image->support_depth()) {
      camera_id_icp = cam_id;

      const auto& depthptr = depth_sources.at(cam_id);

      const ImageSource* mask_source = nullptr;
      auto it = masks_sources.find(cam_id);
      if (it != masks_sources.end()) {
        if (it->second.data != nullptr) {
          mask_source = &it->second;
        }
      }

      auto depth_pyramids = image->build_gpu_depth_pyramid(depthptr, stream.get_stream(), mask_source);

      if (depth_pyramids) {
        depth_icp = true;
      }
    }
  }

  cudaStreamSynchronize(stream.get_stream());

  pipelines::SFMInputs inputs{observations, nullptr};

  bool have_pose;
  if (depth_icp) {
    pnp::IcpInfo depth_info{
        camera_id_icp,
        curr_images.at(camera_id_icp)->gpu_image_pyramid(),
        curr_images.at(camera_id_icp)->gpu_gradient_pyramid(),
        curr_images.at(camera_id_icp)->gpu_depth_pyramid()->get(),
    };

    inputs.depth_info = &depth_info;

    have_pose =
        solver_.solveNextFrame(timestamp, frame_type, inputs, world_from_rig, static_info_exp, tracks2d, tracks3d);
  } else {
    have_pose =
        solver_.solveNextFrame(timestamp, frame_type, inputs, world_from_rig, static_info_exp, tracks2d, tracks3d);
  }

  if (stat) {
    stat->keyframe = frame_type == sof::FrameState::Key;
    stat->heating = false;
  }

  if (!have_pose) {
    reset();
    delta = Isometry3T::Identity();
    static_info_exp.setZero();
    TraceError("Failed to track on the PnP stage");
    return false;
  }

  prediction_model_.add_known_pose(world_from_rig, timestamp);
  delta = prev_world_from_rig_.inverse() * world_from_rig;
  prev_world_from_rig_ = world_from_rig;

  return true;
}

void RGBDOdometry::enable_stat(bool enable) {
  const bool current_state_is_enable = last_frame_stat_ != nullptr;
  if (current_state_is_enable == enable) {
    return;  // if nothing is changed do nothing
  }
  last_frame_stat_ = enable ? std::make_unique<IVisualOdometry::VOFrameStat>() : nullptr;
}

const std::unique_ptr<IVisualOdometry::VOFrameStat>& RGBDOdometry::get_last_stat() const { return last_frame_stat_; }

bool RGBDOdometry::do_predict(PredictorRef predictor, int64_t timestamp, Isometry3T& sof_prediction) {
  Isometry3T update;
  const int64_t prev_abs_timestamp = prediction_model_.last_timestamp_ns();
  if (predictor->predict(prev_abs_timestamp, timestamp, update)) {
    sof_prediction = update * sof_prediction;
    return true;
  }
  return false;
}

}  // namespace cuvslam::odom

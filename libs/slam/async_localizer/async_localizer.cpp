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

#include "slam/async_localizer/async_localizer.h"

#include <chrono>
#include <thread>

#include "common/log_types.h"
#include "common/types.h"
#include "common/unaligned_types.h"
#include "log/log_eigen.h"
#include "profiler/profiler.h"

#include "slam/map/database/lmdb_slam_database.h"
#include "slam/slam/loop_closure_solver/iloop_closure_solver.h"
#include "slam/view/map_to_view.h"

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

namespace cuvslam::slam {

void AsyncLocalizer::LoopClosureStatus::SetLC(KeyFrameId _from_keyframe_id,
                                              const LocalizerAndMapper::LoopClosureStatus& _slam_loop_closure_status) {
  from_keyframe_id = _from_keyframe_id;
  succesed = _slam_loop_closure_status.success;
  result_pose = _slam_loop_closure_status.result_pose;
  result_pose_covariance = _slam_loop_closure_status.result_pose_covariance;
  good_landmarks_count = _slam_loop_closure_status.good_landmarks_count;
  tracked_landmarks_count = _slam_loop_closure_status.tracked_landmarks_count;
  selected_landmarks_count = _slam_loop_closure_status.selected_landmarks_count;
  reprojection_error = _slam_loop_closure_status.reprojection_error;
}

void AsyncLocalizer::LoopClosureStatus::SetFrame(uint64_t _trial_id, float _dist_from_hint,
                                                 const AsyncSlam::VOTrackData& _track_data,
                                                 const Isometry3T& _pose_in_frame, const Isometry3T& _guess_pose) {
  trial_id = _trial_id;
  dist_from_hint = _dist_from_hint;
  track_data = _track_data;
  pose_in_frame = _pose_in_frame;
  guess_pose = _guess_pose;
}

float AsyncLocalizer::LoopClosureStatus::Weight() const {
  if (tracked_landmarks_count == 0 || !succesed) {
    return 0;
  }
  float w = good_landmarks_count / static_cast<float>(tracked_landmarks_count);

  float f = 2 * atan(good_landmarks_count / 500.f) / cuvslam::PI;
  w *= f;
  return w;
}

AsyncLocalizer::~AsyncLocalizer() {
  StopInternal();
  SlamStdout("Destroyed AsyncLocalizer instance. ");
}

void AsyncLocalizer::Init(const camera::Rig& rig, const AsyncLocalizerOptions& options) {
  options_ = options;
  PrepareProbes();

  if (options_.horizontal_search_radius <= 0 || options_.vertical_search_radius <= 0) {
    TraceWarning("Localization search radius must be greater than zero.");
  }

  if (options_.horizontal_step <= 0 || options_.vertical_step <= 0 || options_.angle_step_rads <= 0) {
    TraceWarning("Localization step must be greater than zero.");
  }

  if (options_.angle_step_rads >= 2 * cuvslam::PI) {
    TraceWarning("Angular step must less then 2 Pi");
  }

  rig_ = rig;
  slam_ = std::make_shared<LocalizerAndMapper>(rig_, FeatureDescriptorType::kShiTomasi2, options.use_gpu);
  slam_->SetReproduceMode(options.reproduce_mode);

  first_lcs_.reset(CreateLoopClosureSolver(LoopClosureSolverType::kSimplePoint, RansacType::kNone, true, rig));
  second_lcs_.reset(CreateLoopClosureSolver(LoopClosureSolverType::kTwoStepsEasy, RansacType::kPnP, true, rig));
}

bool AsyncLocalizer::OpenDatabase(const std::string& path) {
  if (!slam_->AttachToExistingReadOnlyDatabase(path)) {
    return false;
  }
  // reset probe view
  uint64_t timestamp_ns = trial_;
  PublishProbesToView(timestamp_ns);
  PublishObservationToView(timestamp_ns);

  // view landmarks
  std::shared_ptr<ViewLandmarks> localizer_landmarks_view =
      localizer_landmarks_view_ ? localizer_landmarks_view_->acquire_earliest() : nullptr;
  if (localizer_landmarks_view) {
    PublishAllLandmarksToView(slam_->GetMap(), timestamp_ns, *localizer_landmarks_view);
    localizer_landmarks_view.reset();
  }

  return true;
}
// Async API
void AsyncLocalizer::StartLocalizationThread(const Isometry3T& guess_pose, const Isometry3T& current_pose) {
  guess_pose_ = guess_pose;
  pose_on_start_ = current_pose;

  thread_ = std::thread{&AsyncLocalizer::BackgroundLoop, this};
}

// Sync API
// frame_id = track_data.end_frame_id & dP = track_data.from_keyframe
bool AsyncLocalizer::LocalizeSync(const Isometry3T& guess_pose, const Isometry3T& current_pose,
                                  const AsyncSlam::VOTrackData& track_data, const sof::Images& images,
                                  AsyncSlam::LocalizationResult& localization_result) {
  guess_pose_ = guess_pose;
  pose_on_start_ = current_pose;

  VOFrameInfo fi;
  fi.images_ = images;
  fi.valid_ = true;
  fi.finished_ = false;
  fi.frame_information_ = FrameInformationString(images);  // vo camera on this frame
  fi.track_data_ = track_data;
  fi.pose_in_frame_ = current_pose;

  // Cycle of localizations
  process_frame_info_ = fi;
  for (;;) {
    const bool to_be_continued = Step();
    if (!to_be_continued) {
      break;
    }
  }
  // Finish
  if (max_exact_loop_closure_status_.Weight() > 0) {
    // slam_localization_info
    SlamStdout("Localization successful (%s). ", sw_step_.Verbose().c_str());

    const auto& lc = max_exact_loop_closure_status_;

    // result
    localization_result.slam_from = slam_;
    localization_result.track_data = lc.track_data;
    localization_result.pose_in_frame = lc.pose_in_frame;
    localization_result.from_keyframe_id = lc.from_keyframe_id;
    localization_result.pose_in_slam = lc.result_pose;
    localization_result.pose_in_slam_covariance = lc.result_pose_covariance;
    return true;
  } else {
    // slam_localization_info
    SlamStdout("Localization failed (%s). ", sw_step_.Verbose().c_str());
  }

  return false;
}

bool AsyncLocalizer::ReceiveResult(AsyncSlam::LocalizationResult& localization_result) {
  {
    // extract current results
    std::shared_ptr<LoopClosureStatus> slam_localization_info;
    if (output_queue_.TryPop(slam_localization_info)) {
      // slam_localization_info
      localization_result.slam_from = slam_;
      localization_result.track_data = slam_localization_info->track_data;
      localization_result.pose_in_frame = slam_localization_info->pose_in_frame;
      localization_result.from_keyframe_id = slam_localization_info->from_keyframe_id;
      localization_result.pose_in_slam = slam_localization_info->result_pose;
      localization_result.pose_in_slam_covariance = slam_localization_info->result_pose_covariance;
      // abort
      {
        SlamStdout("Stopping Localizer because result was fetched. ");
        std::unique_lock<std::mutex> locker(mutex_);
        abort_ = true;
        event_.notify_all();
      }
      return true;  // stop
    }
  }

  {
    std::unique_lock<std::mutex> locker(mutex_);
    if (stopped_) {
      SlamStdout("Forced to stop localizer.\n");
      localization_result.slam_from.reset();
      return true;  // stop
    }
  }
  return false;
}

void AsyncLocalizer::AddNewRequest(const AsyncSlam::VOTrackData& track_data, const sof::Images& images,
                                   const Isometry3T& pose_in_frame) {
  std::unique_lock<std::mutex> locker(mutex_);

  current_frame_info_.frame_information_ = FrameInformationString(images);
  current_frame_info_.pose_in_frame_ = pose_in_frame;
  current_frame_info_.track_data_ = track_data;
  current_frame_info_.images_ = images;
  current_frame_info_.valid_ = true;

  event_.notify_all();
}

// stop thread
void AsyncLocalizer::StopLocalizationThread() { StopInternal(); }

// Set Localizer Probes view
void AsyncLocalizer::SetLocalizerProbesView(std::shared_ptr<ViewManager<ViewLocalizerProbes>> view) {
  this->localizer_probes_view_ = view;
}

// Set DB landmarks view
void AsyncLocalizer::SetLocalizerLandmarksView(std::shared_ptr<ViewManager<ViewLandmarks>> view) {
  this->localizer_landmarks_view_ = view;
}
// Set Localizer observations view
void AsyncLocalizer::SetLocalizerObservationView(std::shared_ptr<ViewManager<ViewLandmarks>> view) {
  this->localizer_observation_view_ = view;
}
// Set Localizer loop closure view
void AsyncLocalizer::SetLocalizerLCLandmarksView(std::shared_ptr<ViewManager<ViewLandmarks>> view) {
  this->localizer_lc_landmarks_view_ = view;
}

void AsyncLocalizer::BackgroundLoop() {
#ifdef USE_CUDA
  cudaSetDevice(0);
#endif
  {
    std::unique_lock locker(mutex_);
    stopped_ = false;
  }

  while (true) {
    {
      std::unique_lock locker(mutex_);
      // check finishing
      if (abort_) {
        SlamStdout("Localizer aborted. ");
        break;
      }

      // check current_frame_info_
      if (current_frame_info_.is_valid()) {
        // copy current_frame_info_ to process_frame_info_?
        if (!process_frame_info_.is_valid()) {
          process_frame_info_ = current_frame_info_;
          SlamStdout(" (~");  // Processing new frame
        } else {
          if (process_frame_info_.frame_id() != current_frame_info_.frame_id()) {
            if (!options_.static_frame_calculation) {
              // update working frame
              process_frame_info_ = current_frame_info_;
              SlamStdout("~");  // Switching to frame
            }
          }
        }
      }
    }

    if (!process_frame_info_.is_valid()) {
      // waiting for valid data
      std::unique_lock<std::mutex> locker(mutex_);

      auto const timeout = std::chrono::seconds(1);
      if (event_.wait_for(locker, timeout) == std::cv_status::timeout) {
        break;  // no new data in coming, abort
      }
      continue;
    }

    const bool to_be_continued = Step();

    if (!to_be_continued) {
      // Finish
      if (max_exact_loop_closure_status_.Weight() > 0) {
        SlamStdout("Localization successful (%s). ", sw_step_.Verbose().c_str());
        auto message = std::make_shared<LoopClosureStatus>(max_exact_loop_closure_status_);
        output_queue_.Push(message);
      } else {
        // ToDo: put negative result to the output_queue_
        SlamStdout("Localization failed (%s). ", sw_step_.Verbose().c_str());
      }
      break;
    }
  }

  {
    std::unique_lock locker(mutex_);
    stopped_ = true;
  }
}

void AsyncLocalizer::PublishProbesToView(uint64_t timestamp_ns) const {
  if (!localizer_probes_view_) {
    return;
  }
  std::shared_ptr<ViewLocalizerProbes> localizer_probes_view = localizer_probes_view_->acquire_earliest();
  if (localizer_probes_view) {
    PublishLocalizerProbesToView(slam_->GetMap(), timestamp_ns, probes_cache_, *localizer_probes_view);
    localizer_probes_view.reset();
  }
}

void AsyncLocalizer::PublishObservationToView(uint64_t timestamp_ns) const {
  if (!localizer_observation_view_) {
    return;
  }
  // view landmarks
  std::shared_ptr<ViewLandmarks> localizer_landmarks_view =
      localizer_observation_view_ ? localizer_observation_view_->acquire_earliest() : nullptr;
  if (localizer_landmarks_view) {
    PublishAllLandmarksToView(slam_->GetMap(), timestamp_ns, *localizer_landmarks_view);
    localizer_landmarks_view.reset();
  }
}

bool AsyncLocalizer::GenerateTrial(uint64_t trial, Isometry3T& isometry_current) const {
  if (trial >= shifts.size()) {
    return false;
  }

  const Shift& shift = shifts[trial];

  Vector3T t(shift.x, shift.y, shift.z);

  Eigen::Matrix3f r = Eigen::AngleAxis<float>(shift.angle, Vector3T::UnitY()).toRotationMatrix();

  Isometry3T isometry = Isometry3T::Identity();
  isometry.translate(t);
  isometry.rotate(r);

  isometry_current = guess_pose_ * isometry;

  return true;
}

void AsyncLocalizer::TryAccurateLoopClosure(const LoopClosureStatus& simple_loop_closure_status) {
  LoopClosureStatus exact_loop_closure_status;
  std::vector<LandmarkInSolver> landmarks_for_view;

  if (!AccurateLoopClosure(simple_loop_closure_status, exact_loop_closure_status,
                           localizer_lc_landmarks_view_ ? &landmarks_for_view : nullptr)) {
    return;
  }

  if (localizer_probes_view_) {
    // exact_result_pose & exact_result_weight
    auto& view_probe = probes_cache_.back();
    view_probe.exact_result_pose.set(exact_loop_closure_status.result_pose);
    view_probe.exact_result_weight = exact_loop_closure_status.Weight();
    view_probe.solved = true;
  }

  // exact solution is found so "simple_step" is good
  max_simple_loop_closure_status_ = simple_loop_closure_status;
  if (exact_loop_closure_status.Weight() > max_exact_loop_closure_status_.Weight()) {
    // the best is found
    SlamStdout("q");
    max_exact_loop_closure_status_ = exact_loop_closure_status;

    // view loop closure landmarks
    std::shared_ptr<ViewLandmarks> loop_close_view =
        localizer_lc_landmarks_view_ ? localizer_lc_landmarks_view_->acquire_earliest() : nullptr;
    if (loop_close_view) {
      PublishLoopClosureToView(slam_->GetMap(), landmarks_for_view, *loop_close_view);
      loop_close_view.reset();
    }
  }
}

float AsyncLocalizer::Shift::dist() const { return sqrtf(x * x + y * y + z * z); }

bool AsyncLocalizer::Step() {
  SlamStdout(".");

  TRACE_EVENT ev = profiler_domain_.trace_event("AsyncLocalizer::Step()", profiler_color_);

  Isometry3T isometry_current;
  const uint64_t trial = trial_;
  if (!GenerateTrial(trial_, isometry_current)) {
    return false;
  }

  const float dist_from_hint = shifts[trial].dist();

  if (dist_from_hint > max_exact_loop_closure_status_.dist_from_hint) {
    return false;
  }

  trial_++;

  const auto& frame_images = process_frame_info_.images_;

  LoopClosureStatus simple_loop_closure_status;
  simple_loop_closure_status.SetFrame(trial, dist_from_hint, process_frame_info_.track_data_,
                                      process_frame_info_.pose_in_frame_, isometry_current);
  FastLoopClosure(isometry_current, frame_images, simple_loop_closure_status);
  if (localizer_probes_view_) {
    const auto& lc = simple_loop_closure_status;
    ViewLocalizerProbe view_probe;
    view_probe.id = lc.trial_id;
    view_probe.guess_pose.set(lc.guess_pose);
    view_probe.weight = lc.Weight();
    probes_cache_.push_back(view_probe);
  }

  if (simple_loop_closure_status.succesed &&
      simple_loop_closure_status.Weight() > max_simple_loop_closure_status_.Weight()) {
    TryAccurateLoopClosure(simple_loop_closure_status);
  }

  if (localizer_probes_view_) {
    if ((trial % 100) == 0) {
      PublishProbesToView(trial);
    }
  }
  if (localizer_observation_view_) {
    if ((trial % 100) == 0) {
      PublishObservationToView(trial);
    }
  }

  return true;
}

bool AsyncLocalizer::FastLoopClosure(const Isometry3T& isometry_current, const Images& frame_images,
                                     LoopClosureStatus& simple_loop_closure_status) const {
  LocalizerAndMapper::LoopClosureStatus slam_loop_closure_status;
  const Isometry3T guess = isometry_current;

  bool simple_lc_success = false;
  if (first_lcs_) {
    slam_->DetectLoopClosure(*first_lcs_, frame_images, guess, slam_loop_closure_status);
    simple_lc_success = slam_loop_closure_status.success;
  }

  const KeyFrameId best_kf = slam_->FindKeyframeWithMostLandmarks(slam_loop_closure_status.landmarks);
  simple_loop_closure_status.SetLC(best_kf, slam_loop_closure_status);

  if (localizer_observation_view_) {
    IncTrackedLandmarks(slam_loop_closure_status);
  }

  return simple_lc_success;
}

bool AsyncLocalizer::AccurateLoopClosure(const LoopClosureStatus& source, LoopClosureStatus& loop_closure_status,
                                         std::vector<LandmarkInSolver>* landmarks_for_view) const {
  const auto& frame_images = process_frame_info_.images_;
  const Isometry3T& guess = source.result_pose;
  LocalizerAndMapper::LoopClosureStatus slam_loop_closure_status;

  if (!second_lcs_) {
    return false;
  }
  slam_->DetectLoopClosure(*second_lcs_, frame_images, guess, slam_loop_closure_status);
  if (!slam_loop_closure_status.success) {
    return false;
  }

  const KeyFrameId best_kf = slam_->FindKeyframeWithMostLandmarks(slam_loop_closure_status.landmarks);

  loop_closure_status.SetLC(best_kf, slam_loop_closure_status);
  loop_closure_status.SetFrame(source.trial_id, source.dist_from_hint, source.track_data, source.pose_in_frame,
                               source.guess_pose);

  if (landmarks_for_view) {
    *landmarks_for_view = slam_loop_closure_status.landmarks;
  }

  return true;
}

void AsyncLocalizer::StopInternal() {
  {
    SlamStdout("Stopping localization thread. ");
    std::unique_lock<std::mutex> locker(mutex_);
    abort_ = true;
    event_.notify_all();
  }

  output_queue_.Abort();
  telemetry_queue_.Abort();

  if (thread_.joinable()) {
    thread_.join();
  }
}

std::string AsyncLocalizer::FrameInformationString(const sof::Images& images) {
#ifdef CUVSLAM_LOG_ENABLE
  Json::Value json;

  if (images.size() >= 1) {
    json["frame_id"] = static_cast<Json::UInt64>(images.begin()->second->get_image_meta().frame_id);
    json["timestamp"] = static_cast<Json::UInt64>(images.begin()->second->get_image_meta().timestamp);
    json["frame_number"] = images.begin()->second->get_image_meta().frame_number;
    for (size_t i = 0; i < images.size(); i++) {
      json["image_file" + std::to_string(i)] = images.begin()->second->get_image_meta().filename;
    }
  }

  std::string jsonStr = writeString(Json::StreamWriterBuilder(), json);
  return jsonStr;
#else
  return "";
#endif
}

void AsyncLocalizer::LogFrameInformationString(const std::string& frame_information) {
#ifdef CUVSLAM_LOG_ENABLE
  Json::Value root;
  std::stringstream json(frame_information);
  Json::CharReaderBuilder builder;
  std::string errs;
  parseFromStream(builder, json, &root, &errs);

  log::Json<LogFrames>(root);
#endif
}

void AsyncLocalizer::PrepareProbes() {
  const float& x_radius = options_.horizontal_search_radius;
  const float& y_radius = options_.vertical_search_radius;  // meters, y is vertical
  const float& z_radius = options_.horizontal_search_radius;

  const float& h_step = options_.horizontal_step;
  const float& v_step = options_.vertical_step;
  const float& a_step = options_.angle_step_rads;

  SlamStdout("horizontal_search_radius %f \n", options_.horizontal_search_radius);
  SlamStdout("vertical_search_radius %f \n", options_.vertical_search_radius);
  SlamStdout("horizontal_step %f \n", options_.horizontal_step);
  SlamStdout("vertical_step %f \n", options_.vertical_step);
  SlamStdout("angle_step_rads %f \n", options_.angle_step_rads);

  shifts.clear();
  for (float x = -x_radius; x <= x_radius; x += h_step) {
    for (float y = -y_radius; y <= y_radius; y += v_step) {
      for (float z = -z_radius; z <= z_radius; z += h_step) {
        for (float angle = 0; angle < 2 * cuvslam::PI; angle += a_step) {
          shifts.push_back({x, y, z, angle});
        }
      }
    }
  }

  std::sort(shifts.begin(), shifts.end(), [](const Shift& lhs, const Shift& rhs) { return lhs.dist() < rhs.dist(); });

  SlamStdout("Prepared %zu probes for localization", shifts.size());
}

void AsyncLocalizer::IncTrackedLandmarks(const LocalizerAndMapper::LoopClosureStatus& slam_loop_closure_status) const {
  auto inc_tracked_landmarks = [&](LandmarkId id) {
    const auto it = tracked_landmarks_.find(id);
    if (it == tracked_landmarks_.end()) {
      tracked_landmarks_[id] = 0;
    }
    auto& counter = tracked_landmarks_[id];
    counter++;
    max_tracked_landmarks_ = std::max(max_tracked_landmarks_, counter);
  };
  for (auto& landmark : slam_loop_closure_status.landmarks) {
    inc_tracked_landmarks(landmark.id);
  }
  for (auto& landmark : slam_loop_closure_status.discarded_landmarks) {
    if (landmark.second != LP_TRACKING_FAILED) {
      inc_tracked_landmarks(landmark.first);
    }
  }
}

}  // namespace cuvslam::slam


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

#include <string>
#include <thread>

#include "camera/rig.h"
#include "common/image.h"
#include "common/stopwatch.h"
#include "common/thread_safe_queue.h"
#include "common/unaligned_types.h"
#include "log/log_eigen.h"
#include "profiler/profiler.h"

#include "slam/async_slam/async_slam.h"
#include "slam/slam/slam.h"
#include "slam/view/view_landmarks.h"
#include "slam/view/view_localizer_probes.h"
#include "slam/view/view_manager.h"
#include "sof/gradient_pyramid.h"

namespace cuvslam::slam {

using namespace cuvslam;

// options see ISlamLocalizer::Init()
struct AsyncLocalizerOptions {
  bool reproduce_mode = false;  // allow to repeat results: ransac.seed(0), sync=true
  bool use_gpu = false;
  bool static_frame_calculation = true;  // operate only on first read frame

  float horizontal_search_radius = 1.5f;  // meters
  float vertical_search_radius = 0.5;

  float horizontal_step = 0.5f;
  float vertical_step = 0.25f;
  float angle_step_rads = 2 * cuvslam::PI / 36;
};

struct AsyncLocalizerTelemetry {
  uint64_t timestamp_ns = 0;  // timestamp of these measurements (in nanoseconds)
};

class AsyncLocalizer {
public:
  AsyncLocalizer() = default;
  ~AsyncLocalizer();

  void Init(const camera::Rig& rig, const AsyncLocalizerOptions& options);

  bool OpenDatabase(const std::string& path);

  // Async API (non-blocking, localization in parallel thread)
  void StartLocalizationThread(const Isometry3T& guess_pose, const Isometry3T& current_pose);
  void StopLocalizationThread();
  bool ReceiveResult(AsyncSlam::LocalizationResult& localization_result);
  void AddNewRequest(const AsyncSlam::VOTrackData& track_data, const sof::Images& images,
                     const Isometry3T& pose_in_frame);

  // Sync API (blocking) for reproducible results
  //  frame_id = track_data.end_frame_id & dP = track_data.from_keyframe
  bool LocalizeSync(const Isometry3T& guess_pose, const Isometry3T& current_pose,
                    const AsyncSlam::VOTrackData& track_data, const sof::Images& images,
                    AsyncSlam::LocalizationResult& localization_result);
  // view
public:
  // Set Localizer Probes view
  void SetLocalizerProbesView(std::shared_ptr<ViewManager<ViewLocalizerProbes>> view);
  // Set DB landmarks view
  void SetLocalizerLandmarksView(std::shared_ptr<ViewManager<ViewLandmarks>> view);
  // Set Localizer observations view
  void SetLocalizerObservationView(std::shared_ptr<ViewManager<ViewLandmarks>> view);
  // Set Localizer loop closure view
  void SetLocalizerLCLandmarksView(std::shared_ptr<ViewManager<ViewLandmarks>> view);

private:
  camera::Rig rig_;
  AsyncLocalizerOptions options_;
  std::shared_ptr<LocalizerAndMapper> slam_;
  std::unique_ptr<ILoopClosureSolver> first_lcs_, second_lcs_;
  Isometry3T guess_pose_;
  Isometry3T pose_on_start_;

  struct LoopClosureStatus {
    uint64_t trial_id = 0;
    float dist_from_hint = std::numeric_limits<float>::infinity();
    AsyncSlam::VOTrackData track_data;
    KeyFrameId from_keyframe_id = InvalidKeyFrameId;
    Isometry3T pose_in_frame;
    Isometry3T guess_pose;
    bool succesed = false;
    Isometry3T result_pose;
    Matrix6T result_pose_covariance;
    uint32_t selected_landmarks_count = 0;  // Count of Selected Landmarks in LC
    uint32_t tracked_landmarks_count = 0;   // Count of Tracked Landmarks in LC
    uint32_t good_landmarks_count = 0;      // Count of good Landmarks in LC
    double reprojection_error = 0;

    void SetLC(KeyFrameId from_keyframe_id, const LocalizerAndMapper::LoopClosureStatus& slam_loop_closure_status);
    void SetFrame(uint64_t trial_id, float _dist_from_hint, const AsyncSlam::VOTrackData& track_data,
                  const Isometry3T& pose_in_frame, const Isometry3T& guess_pose);
    float Weight() const;
  };

  struct VOFrameInfo {
    bool valid_ = false;
    bool finished_ = false;
    sof::Images images_;

    std::string frame_information_;
    // pose on this frame in current slam
    Isometry3T pose_in_frame_;
    // frame_id and from_keyframe
    AsyncSlam::VOTrackData track_data_;

    FrameId frame_id() const { return images_.begin()->second->get_image_meta().frame_id; }
    bool is_valid() const { return valid_; }
  };

  VOFrameInfo current_frame_info_;
  VOFrameInfo process_frame_info_;
  LoopClosureStatus max_simple_loop_closure_status_;
  LoopClosureStatus max_exact_loop_closure_status_;

  std::thread thread_;

  bool abort_ = false;
  bool stopped_ = true;
  std::mutex mutex_;
  std::condition_variable event_;  // ? is needed

  // trial
  uint64_t trial_ = 0;

  // profiler
  profiler::SLAMProfiler::DomainHelper profiler_domain_ = profiler::SLAMProfiler::DomainHelper("SLAM");
  uint32_t profiler_color_ = 0x00FF00;

  ThreadSafeQueue<std::shared_ptr<LoopClosureStatus>> output_queue_;
  ThreadSafeQueue<std::shared_ptr<AsyncLocalizerTelemetry>> telemetry_queue_;
  // telemetry of last Step()
  AsyncLocalizerTelemetry last_telemetry_;

  Stopwatch sw_step_;

  std::shared_ptr<ViewManager<ViewLocalizerProbes>> localizer_probes_view_;
  std::shared_ptr<ViewManager<ViewLandmarks>> localizer_landmarks_view_;
  std::shared_ptr<ViewManager<ViewLandmarks>> localizer_observation_view_;
  std::shared_ptr<ViewManager<ViewLandmarks>> localizer_lc_landmarks_view_;

  // cache
  std::vector<ViewLocalizerProbe> probes_cache_;
  // landmarks statistic. If localizer_observation_view_ is not null
  mutable uint32_t max_tracked_landmarks_ = 1;
  mutable std::unordered_map<LandmarkId, uint32_t> tracked_landmarks_;

  void BackgroundLoop();
  bool Step();
  void StopInternal();

  void PublishProbesToView(uint64_t timestamp_ns) const;
  void PublishObservationToView(uint64_t timestamp_ns) const;

  bool GenerateTrial(uint64_t trial, Isometry3T& isometry_current) const;

  // Attempts accurate refinement of a fast loop closure result and updates best result if improved
  void TryAccurateLoopClosure(const LoopClosureStatus& simple_loop_closure_status);

  // Fast loop closure using SimplePoint solver without RANSAC for quick candidate screening
  bool FastLoopClosure(const Isometry3T& isometry_current, const Images& frame_image,
                       LoopClosureStatus& loop_closure_status) const;

  // Accurate loop closure using TwoStepsEasy solver with PnP RANSAC for precise pose refinement
  bool AccurateLoopClosure(const LoopClosureStatus& source, LoopClosureStatus& loop_closure_status,
                           std::vector<LandmarkInSolver>* landmarks_for_view) const;

  void IncTrackedLandmarks(const LocalizerAndMapper::LoopClosureStatus& slam_loop_closure_status) const;

protected:
  static void LogFrameInformationString(const std::string& frame_information);
  static std::string FrameInformationString(const sof::Images& images);

  struct Shift {
    float x;
    float y;
    float z;
    float angle;  // 0-2Pi

    float dist() const;
  };

  std::vector<Shift> shifts;

  void PrepareProbes();
};

}  // namespace cuvslam::slam

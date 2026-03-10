
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

#include <experimental/iterator>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "gflags/gflags.h"
#include "json/json.h"

#include "camera_rig_edex/camera_rig_edex.h"
#include "camera_rig_edex/repeated_camera_rig_edex.h"
#include "camera_rig_edex/shuttle_camera_rig_edex.h"
#include "common/log.h"
#include "common/stream.h"
#include "common/time.h"
#include "common/types.h"
#include "common/utils.h"
#include "cuvslam/cuvslam2.h"
#include "cuvslam/internal.h"

using namespace cuvslam;

// Get default configurations to use for flag defaults
namespace {
const Odometry::Config kDefaultOdomCfg = Odometry::GetDefaultConfig();
const Slam::Config kDefaultSlamCfg = Slam::GetDefaultConfig();
}  // namespace

DEFINE_int32(verbosity, 2, "Verbosity level");
// replay settings
DEFINE_string(dataset, ".", "Path to edex dataset to run on");
DEFINE_string(cameras, "", "Comma-separated list of cameras");
DEFINE_int32(start_frame, 0, "Frame to start with");
DEFINE_int32(repeat, 1, "Number of repetitions of the dataset");
DEFINE_bool(shuttle, false, "Shuttle replay forward and backward");
DEFINE_double(max_fps, 0., "Throttle main loop to not exceed max_fps. 0 (default) to disable");
DEFINE_bool(ignore_tracking_errors, false, "Don't stop on tracking errors");
// output settings
DEFINE_string(print_odom_poses, "", "Path to save odometry poses");
DEFINE_string(print_slam_poses, "", "Path to save SLAM poses");
DEFINE_string(print_loc_poses, "", "Path to save localization poses");
DEFINE_string(print_format, "matrix", "Format of output poses (matrix, tum)");
DEFINE_bool(print_nan_on_failure, false, "Print a NaN pose when the localization failed");
DEFINE_bool(ros_frame_conversion, false, "Convert input/output poses from/to ROS frame");
DEFINE_string(output_map, "", "Path to output map (cuVSLAM will try to save it at the end)");
DEFINE_string(print_map_keyframes, "", "Path to save map keyframes");
// hinted localization settings
DEFINE_string(loc_input_map, "", "Path to input map (cuVSLAM will try to localize in it)");
DEFINE_string(loc_input_hints, "", "Path to hint data for localization (format: `timestamp x y z`)");
DEFINE_string(loc_hint_ts_format, "detect", "Localization hint timestamp format (detect, int, float)");
DEFINE_int32(loc_start_frame, 0, "Start frame for localization");
DEFINE_int32(loc_skip_frames, 0, "Number of frames to skip after previous localization");
DEFINE_int32(loc_retries, 0, "Number of retries if localization fails");
DEFINE_double(loc_hint_noise, 0., "Introduce noise to hints (uniform, max deviation in meters)");
DEFINE_bool(loc_random_rot, false, "Randomize hint rotation");
DEFINE_bool(localize_wait, false, "Wait for localization to finish (not the same as reproduce_mode in slam)");
DEFINE_bool(localize_forever, false, "Run localization continuously (each time previous call finished)");
// cuvslam configuration
DEFINE_string(debug_dump, "", "Path to debug dump");
DEFINE_int32(cfg_multicam_mode, static_cast<int>(kDefaultOdomCfg.multicam_mode),
             "Multicamera mode: performance (0), precision (1), or moderate (2)");
DEFINE_bool(cfg_denoising, kDefaultOdomCfg.use_denoising, "Enable image denoising");
DEFINE_bool(cfg_horizontal, kDefaultOdomCfg.rectified_stereo_camera,
            "Enable tracking for rectified cameras with principal points on the horizontal line");
DEFINE_bool(cfg_planar, kDefaultSlamCfg.planar_constraints,
            "Slam poses are so that the camera moves on a horizontal plane");
DEFINE_int32(cfg_odom_mode, static_cast<int>(kDefaultOdomCfg.odometry_mode),
             "Odometry mode: Multicamera (0), Inertial (1), RGBD (2), Mono (3)");
DEFINE_bool(cfg_enable_slam, false, "Enable localization and mapping");
DEFINE_bool(cfg_sync_slam, kDefaultSlamCfg.sync_mode,
            "Run localization and mapping in the same thread with visual odometry");
DEFINE_int32(cfg_slam_max_map_size, kDefaultSlamCfg.max_map_size,
             "Maximum numbers of poses in SLAM pose graph, 0 means unlimited pose-graph");
DEFINE_double(cfg_max_frame_delta_s, kDefaultOdomCfg.max_frame_delta_s,
              "Set maximum camera frame time in seconds to warn users");
DEFINE_bool(cfg_enable_export, false, "Enable export of observations & landmarks");
// image crop settings
DEFINE_int32(border_top, 0, "top border to ignore in pixels (0 to use full frame)");
DEFINE_int32(border_bottom, 0, "bottom border to ignore in pixels (0 to use full frame)");
DEFINE_int32(border_left, 0, "left border to ignore in pixels (0 to use full frame)");
DEFINE_int32(border_right, 0, "right border to ignore in pixels (0 to use full frame)");
// rgbd settings
// set default to 0, so users don't have to specify it explicitly for the most common case,
// while library defaults to -1 to avoid errors
DEFINE_int32(cfg_depth_camera, 0, "Depth camera index");
DEFINE_double(cfg_depth_scale_factor, kDefaultOdomCfg.rgbd_settings.depth_scale_factor, "Depth scale factor");
DEFINE_bool(cfg_enable_depth_stereo_tracking, kDefaultOdomCfg.rgbd_settings.enable_depth_stereo_tracking,
            "Enable depth stereo tracking");

#define VERIFY_TRACE(condition, ...) \
  if (!(condition)) {                \
    TraceError(__VA_ARGS__);         \
    return false;                    \
  }

struct Hint {
  int64_t timestamp;
  std::array<float, 3> hint;
};
namespace {

void setStreamFormat(std::ofstream& out_poses) {
  out_poses << std::fixed << std::setprecision(9)
            << PoseIOManip(FLAGS_print_format == "tum" ? PoseFormat::TUM : PoseFormat::MATRIX,
                           FLAGS_ros_frame_conversion);
}

template <typename T>
bool loadValues(const std::string& file_name, std::vector<std::vector<T>>& values) {
  std::ifstream file(file_name);
  VERIFY_TRACE(file.is_open(), "Unable to open file %s", file_name.c_str());

  std::string line;
  while (std::getline(file, line)) {
    std::istringstream ss(line);
    std::vector<T> vec;
    T val;
    while (ss >> val) {
      vec.push_back(val);
    }
    values.emplace_back(std::move(vec));
  }

  return true;
}

double getTimestampMultiplier(const std::vector<std::vector<double>>& values) {
  if (FLAGS_loc_hint_ts_format == "detect") {
    // if all timestamps are integer
    if (std::all_of(values.begin(), values.end(),
                    [](const std::vector<double>& vec) { return vec[0] == static_cast<int64_t>(vec[0]); })) {
      return 1;
    } else {
      return 1e9;
    }
  } else if (FLAGS_loc_hint_ts_format == "float") {
    return 1e9;
  } else if (FLAGS_loc_hint_ts_format == "int") {
    return 1;
  }
  TraceError("Unknown hint timestamp format %s", FLAGS_loc_hint_ts_format.c_str());
  return 1;
}

bool loadHints(const std::string& file_name, std::vector<Hint>& hints) {
  std::vector<std::vector<double>> values;
  if (!loadValues(file_name, values)) {
    return false;
  }
  const double mul = getTimestampMultiplier(values);
  for (auto&& vec : values) {
    VERIFY_TRACE(vec.size() == 4 || vec.size() == 8, "4/8 values per line expected (ts, x, y, z [, quaternion])");
    auto ts = static_cast<int64_t>(vec[0] * mul);
    VERIFY_TRACE(hints.empty() || ts > hints.back().timestamp, "Hints must be sorted by timestamp (%zu > %zu)", ts,
                 hints.back().timestamp);
    hints.push_back({ts, {(float)vec[1], (float)vec[2], (float)vec[3]}});
  }
  return !hints.empty();
}

Pose getHintPose(std::array<float, 3> hint, float noise, bool random_rot) {
  std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<> dis(-noise, noise);
  for (float& v : hint) {
    v += dis(gen);
  }
  Isometry3T transform(Translation3T{hint[0], hint[1], hint[2]});
  if (random_rot) {
    transform.rotate(Matrix3T::Random());
  }
  if (FLAGS_ros_frame_conversion) {
    transform = kCuvslamFromRos * transform * kRosFromCuvslam;
  }
  Pose pose;
  vec<3>(pose.translation) = transform.translation();
  Eigen::Quaternionf quat(transform.linear());
  pose.rotation = {quat.x(), quat.y(), quat.z(), quat.w()};
  return pose;
}

// get the latest hint, but no later than a given frame
bool getLatestHint(const std::vector<Hint>& hints, int64_t timestamp, Hint& out) {
  auto it = std::upper_bound(hints.begin(), hints.end(), timestamp,
                             [](int64_t ts, const Hint& hint) { return ts < hint.timestamp; });
  if (it != hints.begin()) {
    out = *(--it);
    return true;
  }
  return false;
}

std::ostream& operator<<(std::ostream& stream, const Pose& pose) {
  Isometry3T iso_pose;
  iso_pose.setIdentity();
  Eigen::Quaternionf quat{pose.rotation.data()};
  iso_pose.linear() = quat.toRotationMatrix();
  iso_pose.translation() = vec<3>(pose.translation);
  return ::cuvslam::operator<<(stream, iso_pose);
}

void printTsPose(std::ostream& stream, bool is_valid, int64_t ts, const Pose& pose) {
  constexpr const float kNaN = std::numeric_limits<float>::quiet_NaN();
  constexpr const Pose kNanPose = {{kNaN, kNaN, kNaN, kNaN}, {kNaN, kNaN, kNaN}};
  if (stream) {
    const auto timestamp = Timestamp(ts);
    if (is_valid) {
      stream << timestamp << " " << pose << std::endl;
    } else if (FLAGS_print_nan_on_failure) {
      stream << timestamp << " " << kNanPose << std::endl;
    }
  }
}

// Asynchronous response for CUVSLAM_LocalizeInExistDb()
enum class LocalizeInMapStatus { NOT_LOCALIZED, IN_PROGRESS, LOCALIZED };

struct LocalizeInMapContext {
  LocalizeInMapStatus status;
  int64_t timestamp;
  std::ofstream out_poses;
};

void localize_in_exist_db_response(const Result<Pose>& result, LocalizeInMapContext* ctx) {
  TraceDebug("LocalizeInMap sent status: %s", result.data.has_value() ? "SUCCESS" : result.error_message.data());
  printTsPose(ctx->out_poses, result.data.has_value(), ctx->timestamp,
              result.data.has_value() ? result.data.value() : Pose{});
  ctx->status = (result.data.has_value() ? LocalizeInMapStatus::LOCALIZED : LocalizeInMapStatus::NOT_LOCALIZED);
}

// Asynchronous response for SaveMap()
enum class SaveMapStatus { NOT_SAVED, IN_PROGRESS, SAVED };

void save_to_slam_db_response(bool success, SaveMapStatus* context) {
  TraceDebug("SaveMap sent status: %s", success ? "SUCCESS" : "FAILED");
  *context = (success ? SaveMapStatus::SAVED : SaveMapStatus::NOT_SAVED);
}

Rig createRig(const edex::EdexFile& edex_file, const camera_rig_edex::ICameraRigReplay* edex_rig) {
  Rig rig;
  rig.cameras.reserve(edex_rig->getCamerasNum());
  for (size_t i = 0; i < edex_rig->getCamerasNum(); i++) {
    const edex::Camera& cam = edex_file.cameras_[edex_rig->getCameraIds()[i]];

    Camera camera{};
    camera.size[0] = cam.intrinsics.resolution.x();
    camera.size[1] = cam.intrinsics.resolution.y();
    camera.principal = {cam.intrinsics.principal.x(), cam.intrinsics.principal.y()};
    camera.focal = {cam.intrinsics.focal.x(), cam.intrinsics.focal.y()};

    vec<3>(camera.rig_from_camera.translation) = cam.transform.translation();
    Eigen::Quaternionf quat(cam.transform.linear());
    camera.rig_from_camera.rotation = {quat.x(), quat.y(), quat.z(), quat.w()};

    camera.distortion.model = StringToDistortionModel(cam.intrinsics.distortion_model);
    camera.distortion.parameters = cam.intrinsics.distortion_params;
    camera.border_top = FLAGS_border_top;
    camera.border_bottom = FLAGS_border_bottom;
    camera.border_left = FLAGS_border_left;
    camera.border_right = FLAGS_border_right;

    rig.cameras.push_back(std::move(camera));
  }

  return rig;
}

bool trackEdexDataSet(const std::string& data_folder, const Odometry::Config& odom_cfg, const Slam::Config& slam_cfg,
                      const std::string& input_map_name, const std::string& output_map_name) {
  std::vector<CameraId> camera_ids{StringToIntVector<CameraId>(FLAGS_cameras, ',')};
  std::string edex_name{std::filesystem::path{data_folder} / "stereo.edex"};
  std::unique_ptr<camera_rig_edex::ICameraRigReplay> edex_rig;
  if (FLAGS_shuttle) {
    edex_rig = std::make_unique<camera_rig_edex::ShuttleCameraRigEdex>(
        std::make_unique<camera_rig_edex::CameraRigEdex>(edex_name, data_folder, camera_ids), FLAGS_repeat);
  } else {
    edex_rig = std::make_unique<camera_rig_edex::RepeatedCameraRigEdex>(
        std::make_unique<camera_rig_edex::CameraRigEdex>(edex_name, data_folder, camera_ids), FLAGS_repeat);
  }

  edex::EdexFile edex_file;
  VERIFY_TRACE(edex_file.read(edex_name), "Failed to read %s\n", edex_name.c_str());

  const ErrorCode ret = edex_rig->start();
  VERIFY_TRACE(ret == ErrorCode::S_True, "Failed to start camera rig %s\n", ret.str());

  WarmUpGPU();
  Rig rig = createRig(edex_file, edex_rig.get());
  std::unique_ptr<Odometry> odom = std::make_unique<Odometry>(rig, odom_cfg);
  TraceMessage("Odometry tracker created");

  std::unique_ptr<Slam> slam;
  if (FLAGS_cfg_enable_slam) {
    slam = std::make_unique<Slam>(rig, odom->GetPrimaryCameras(), slam_cfg);
    TraceMessage("SLAM created");
  }

  edex_rig->registerIMUCallback([&](const imu::ImuMeasurement& measurement) {
    if (odom_cfg.odometry_mode != Odometry::OdometryMode::Inertial) {
      return;
    }
    ImuMeasurement imu_measurement;
    imu_measurement.timestamp_ns = measurement.time_ns;
    vec<3>(imu_measurement.linear_accelerations) = measurement.linear_acceleration;
    vec<3>(imu_measurement.angular_velocities) = measurement.angular_velocity;
    odom->RegisterImuMeasurement(0, imu_measurement);
  });

  VERIFY_TRACE(FLAGS_loc_input_map.empty() == FLAGS_loc_input_hints.empty(),
               "Both `loc_input_map` and `loc_input_hints` must be provided for localization");
  std::vector<Hint> hints;
  if (!FLAGS_loc_input_hints.empty()) {
    VERIFY_TRACE(loadHints(FLAGS_loc_input_hints, hints), "Failed to load localization hints from %s",
                 FLAGS_loc_input_hints.c_str());
  }

  std::ofstream out_odom_poses(FLAGS_print_odom_poses);
  std::ofstream out_slam_poses(FLAGS_print_slam_poses);

  LocalizeInMapContext loc_context{LocalizeInMapStatus::NOT_LOCALIZED, 0, std::ofstream(FLAGS_print_loc_poses)};

  setStreamFormat(out_odom_poses);
  setStreamFormat(out_slam_poses);
  setStreamFormat(loc_context.out_poses);

  if (FLAGS_localize_forever) {
    FLAGS_loc_retries = std::numeric_limits<int32_t>::max();
  }

  FrameId frame = static_cast<FrameId>(FLAGS_start_frame);
  edex_rig->setCurrentFrame(frame);
  FrameId next_loc_frame = FLAGS_loc_start_frame;
  while (true) {
    ScopedThrottler throttle(FLAGS_max_fps);
    Sources cur_sources;
    Sources masks_sources;
    DepthSources depth_sources;
    Metas cur_meta;
    const ErrorCode ret = edex_rig->getFrame(cur_sources, cur_meta, masks_sources, depth_sources);
    if (ret != ErrorCode::S_True) {
      VERIFY_TRACE(ret == ErrorCode::E_Bounds, "getFrame fails with error %s at frame %zu", ErrorCode::GetString(ret),
                   frame);
      // if not another error, we successfully reached last frame
      break;
    }

    std::vector<Image> images;
    std::vector<Image> masks;
    for (size_t i = 0; i < cur_sources.size(); i++) {
      auto&& src = cur_sources[i];
      auto&& mask_src = masks_sources[i];
      auto&& meta = cur_meta[i];
      images.emplace_back(Image{{src.data, meta.shape.width, meta.shape.height, src.pitch,
                                 static_cast<Image::Encoding>(src.image_encoding), ImageData::DataType::UINT8, false},
                                meta.timestamp,
                                static_cast<uint32_t>(i)});
      if (mask_src.data != nullptr) {
        masks.emplace_back(
            Image{{mask_src.data, meta.mask_shape.width, meta.mask_shape.height, mask_src.pitch,
                   static_cast<Image::Encoding>(mask_src.image_encoding), ImageData::DataType::UINT8, false},
                  meta.timestamp,
                  static_cast<uint32_t>(i)});
      }
    }

    PoseEstimate pose_estimate = odom->Track(images, masks);
    if (!pose_estimate.world_from_rig.has_value()) {
      TraceWarning("Track(): Tracking lost at frame %zu.", frame);
      if (FLAGS_ignore_tracking_errors) {
        continue;
      } else {
        return false;
      }
    }
    auto odom_pose = pose_estimate.world_from_rig.value().pose;
    printTsPose(out_odom_poses, true, pose_estimate.timestamp_ns, odom_pose);

    if (slam) {
      Odometry::State state;
      odom->GetState(state);
      Pose slam_pose = slam->Track(state);
      printTsPose(out_slam_poses, true, pose_estimate.timestamp_ns, slam_pose);
    }

    if (odom_cfg.enable_observations_export) {
      // Export observations for camera 0
      std::vector<Observation> observations = odom->GetLastObservations(0);
      TraceDebug("Exported %zu observations at frame %zu", observations.size(), frame);
    }

    if (odom_cfg.enable_landmarks_export) {
      std::vector<Landmark> landmarks = odom->GetLastLandmarks();
      TraceDebug("Exported %zu landmarks at frame %zu", landmarks.size(), frame);
    }

    if (!input_map_name.empty() && !hints.empty() && slam) {
      if (loc_context.status == LocalizeInMapStatus::NOT_LOCALIZED && frame >= next_loc_frame) {
        Hint hint;
        if (getLatestHint(hints, images[0].timestamp_ns, hint) && FLAGS_loc_retries-- >= 0) {
          Pose guess_pose = getHintPose(hint.hint, FLAGS_loc_hint_noise, FLAGS_loc_random_rot);
          loc_context.status = LocalizeInMapStatus::IN_PROGRESS;
          loc_context.timestamp = images[0].timestamp_ns;

          // Default localization settings from SlamLocalizerOptions
          Slam::LocalizationSettings settings;
          settings.horizontal_search_radius = 1.5f;
          settings.vertical_search_radius = 0.5f;
          settings.horizontal_step = 0.5f;
          settings.vertical_step = 0.25f;
          settings.angular_step_rads = 2 * PI / 36;
          settings.enable_reading_internals = false;

          slam->LocalizeInMap(input_map_name, guess_pose, images, settings, [&loc_context](const Result<Pose>& result) {
            localize_in_exist_db_response(result, &loc_context);
          });

          while (FLAGS_localize_wait && loc_context.status == LocalizeInMapStatus::IN_PROGRESS) {
            // Wait for localization to complete
            // Make dummy tracking calls to allow async localization to progress
            std::vector<Image> dummy_images = images;
            for (auto&& im : dummy_images) {
              im.timestamp_ns += 1000;
            }
            odom->Track(dummy_images);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
          }
          next_loc_frame = frame + FLAGS_loc_skip_frames + 1;
        }
      }

      if (FLAGS_localize_forever && loc_context.status == LocalizeInMapStatus::LOCALIZED) {
        // Slam can end up in a bad state, so we need to recreate tracker and slam for next localization attempt
        odom = std::make_unique<Odometry>(rig, odom_cfg);
        slam = std::make_unique<Slam>(rig, odom->GetPrimaryCameras(), slam_cfg);
        loc_context.status = LocalizeInMapStatus::NOT_LOCALIZED;
        next_loc_frame = frame + FLAGS_loc_skip_frames + 1;
      }
    }

    frame++;
  }
  while (loc_context.status == LocalizeInMapStatus::IN_PROGRESS) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  if (!output_map_name.empty() && slam) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    auto save_status{SaveMapStatus::IN_PROGRESS};
    TraceDebug("Calling SaveMap with %s", output_map_name.c_str());
    slam->SaveMap(output_map_name, [&save_status](bool success) { save_to_slam_db_response(success, &save_status); });
    while (save_status == SaveMapStatus::IN_PROGRESS) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    std::vector<PoseStamped> kf_poses;
    slam->GetAllSlamPoses(kf_poses, slam_cfg.max_map_size);
    std::ofstream out_kfs(FLAGS_print_map_keyframes);
    setStreamFormat(out_kfs);
    for (const auto& kf : kf_poses) {
      printTsPose(out_kfs, true, kf.timestamp_ns, kf.pose);
    }
  }

  return true;
}

}  // end of anonymous namespace

int main(int arg_c, char** arg_v) {
  gflags::ParseCommandLineFlags(&arg_c, &arg_v, /*remove flags = */ true);
  if (arg_c != 1) {
    std::cout << "This tools doesn't expect command line arguments, only flags listed with --help" << std::endl
              << "If you see this message, please check your command line." << std::endl;
    return EXIT_FAILURE;
  }

  Trace::SetVerbosity(Trace::ToVerbosity(FLAGS_verbosity));
  SetVerbosity(FLAGS_verbosity);

  // Create Odometry configuration
  Odometry::Config odom_cfg;
  odom_cfg.debug_dump_directory = FLAGS_debug_dump;
  odom_cfg.use_denoising = FLAGS_cfg_denoising;
  odom_cfg.rectified_stereo_camera = FLAGS_cfg_horizontal;
  odom_cfg.max_frame_delta_s = static_cast<float>(FLAGS_cfg_max_frame_delta_s);
  odom_cfg.enable_observations_export = FLAGS_cfg_enable_export;
  odom_cfg.enable_landmarks_export = FLAGS_cfg_enable_export;

  // Set multicamera mode
  if (FLAGS_cfg_multicam_mode == 0) {
    odom_cfg.multicam_mode = Odometry::MulticameraMode::Performance;
  } else if (FLAGS_cfg_multicam_mode == 1) {
    odom_cfg.multicam_mode = Odometry::MulticameraMode::Precision;
  } else if (FLAGS_cfg_multicam_mode == 2) {
    odom_cfg.multicam_mode = Odometry::MulticameraMode::Moderate;
  } else {
    TraceError("Invalid multicamera mode");
    return EXIT_FAILURE;
  }

  // Set odometry mode
  if (FLAGS_cfg_odom_mode == 0) {
    odom_cfg.odometry_mode = Odometry::OdometryMode::Multicamera;
  } else if (FLAGS_cfg_odom_mode == 1) {
    odom_cfg.odometry_mode = Odometry::OdometryMode::Inertial;
  } else if (FLAGS_cfg_odom_mode == 2) {
    odom_cfg.odometry_mode = Odometry::OdometryMode::RGBD;
    odom_cfg.rgbd_settings.depth_camera_id = FLAGS_cfg_depth_camera;
    odom_cfg.rgbd_settings.depth_scale_factor = static_cast<float>(FLAGS_cfg_depth_scale_factor);
    odom_cfg.rgbd_settings.enable_depth_stereo_tracking = FLAGS_cfg_enable_depth_stereo_tracking;
  } else if (FLAGS_cfg_odom_mode == 3) {
    odom_cfg.odometry_mode = Odometry::OdometryMode::Mono;
  } else {
    TraceError("Unsupported odometry mode");
    return EXIT_FAILURE;
  }

  // Create SLAM configuration
  Slam::Config slam_cfg;
  slam_cfg.sync_mode = FLAGS_cfg_sync_slam;
  slam_cfg.max_map_size = FLAGS_cfg_slam_max_map_size;
  slam_cfg.planar_constraints = FLAGS_cfg_planar;

  return trackEdexDataSet(FLAGS_dataset, odom_cfg, slam_cfg, FLAGS_loc_input_map, FLAGS_output_map) ? 0 : 1;
}


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

#include <cstdint>
#include <string>

#include "common/imu_calibration.h"
#include "sba/sba_config.h"
#include "sof/sof_config.h"

namespace cuvslam::odom {

struct KeyFrameSettings {
  // The current frame becomes keyframe if the percent of survivor tracks
  // from the last keyframe is less than this value
  float survivor_from_last = 41.f;

  // maximum timedelta between consecutive keyframes in seconds
  int64_t max_timedelta_between_kfs_s = 60;
};

struct Settings {
  sba::Settings sba_settings;
  sof::Settings sof_settings;
  cuvslam::imu::ImuCalibration imu_calibration;
  KeyFrameSettings kf_settings;
  bool verbose = false;
  bool use_prediction = true;
};

}  // namespace cuvslam::odom

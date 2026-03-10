
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
#include <deque>
#include "odometry/ipredictor.h"

namespace cuvslam::odom {

/*

This class predicts motion trajectory given a few sample poses
from the past.

*/
class PosePredictionModel : public IPredictor {
public:
  // Predict relative motion from the last pose:
  // pose(timestamp) = update * latest_pose
  //
  // The method can optionally return latest pose.
  bool predict_left_update(Isometry3T& update, int64_t timestamp_ns, Isometry3T* latest_pose = nullptr) const;

  // Timestamp must be in nanoseconds
  void add_known_pose(const Isometry3T& pose, int64_t timestamp_ns);

  // Should be called whenever we reset coordinate system for the poses.
  void reset();

  int64_t last_timestamp_ns() const;

  bool predict(int64_t prev_timestamp, int64_t current_timestamp, Isometry3T& delta) const override final;

private:
  std::deque<Isometry3T> poses_;
  std::deque<int64_t> timestamps_ns_;
};

}  // namespace cuvslam::odom

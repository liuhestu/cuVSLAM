
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

#include <list>
#include <mutex>
#include <utility>

#include "common/isometry.h"

#include "slam/slam/slam.h"

namespace cuvslam::slam {

// Nodes that were sent to the slam stream but from which notifications
// about addition to the pos graph were not still received.
// Current pose schema: PG_1->...->PG_N->tail_1->...->tail_tip->+odometry_delta
//                                   |                   |             |
//                               slam tread          main thread   GetSlamPose
// All posses in world space
class Tail {
public:
  Tail(const LocalizerAndMapper& slam);
  bool IsEmpty() const { return tail_.empty(); }

  // called from main thread
  Isometry3T GetTipPose() const;
  void Grow(int64_t timestamp_ns, const Isometry3T& pose);
  void Clear();

  // called from slam thread
  void MakeShortAndFollowBody();

private:
  const LocalizerAndMapper& slam_;  // keep reference
  // first - timestamp us
  // second - pose in world space
  std::list<std::pair<int64_t, Isometry3T>> tail_;  // updates from main and slam thread see @tail_guard_
  mutable std::mutex tail_guard_;
};

}  // namespace cuvslam::slam

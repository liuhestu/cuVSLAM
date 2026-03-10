
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

#include "slam/async_slam/tail.h"

namespace cuvslam::slam {

Tail::Tail(const LocalizerAndMapper& slam) : slam_{slam} {}

Isometry3T Tail::GetTipPose() const {
  std::lock_guard locker(tail_guard_);

  if (tail_.empty()) {
    return Isometry3T::Identity();
  }
  return tail_.back().second;
}

void Tail::Grow(int64_t timestamp_ns, const Isometry3T& pose) {
  std::lock_guard locker(tail_guard_);
  tail_.push_back({timestamp_ns, pose});
}

void Tail::Clear() {
  std::lock_guard locker(tail_guard_);
  tail_.clear();
}

// called from slam thread
void Tail::MakeShortAndFollowBody() {
  Isometry3T last_keyframe_pose;
  int64_t last_keyframe_ts;
  if (!slam_.GetLastKeyframePoseAndTimestamp(last_keyframe_pose, last_keyframe_ts)) {
    return;
  }

  std::lock_guard locker(tail_guard_);
  while (!tail_.empty() && tail_.front().first < last_keyframe_ts) {
    tail_.pop_front();
  }
  if (tail_.empty() || tail_.front().first != last_keyframe_ts) {
    return;
  }
  const Isometry3T& old_pose = tail_.front().second;
  const Isometry3T& new_pose = last_keyframe_pose;

  if (old_pose.isApprox(new_pose, 0.01f)) {  // neither pose should have scale
    return;
  }

  const Isometry3T delta = old_pose.inverse() * new_pose;
  for (auto& p : tail_) {
    Isometry3T updated = p.second * delta;
    RemoveScaleFromTransform(updated);  // remove scale caused by floating-point errors
    p.second = updated;
  }
  tail_.front().second = new_pose;
}

}  // namespace cuvslam::slam


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

#include "pipelines/tracker_state_machine.h"

#include "common/log.h"

namespace cuvslam::pipelines {

StateMachine::StateMachine(const StateMachineSettings& s) : s_(s) {}

void StateMachine::register_gravity_estimation_callback(const std::function<bool(size_t num_kfs)>& cb) {
  gravity_calculate_callback_ = cb;
}

void StateMachine::reset() {
  frames_timeline_.clear();
  keyframe_timeline_.clear();
  last_gravity_estimation_time_ns = -1;
  last_successfull_frame_time_ns = -1;
  state_ = Uninitialized;
}

StateMachine::State StateMachine::update_frame_state(bool is_keyframe, bool track_result, int64_t ts_ns) {
  if (is_keyframe) {
    keyframe_timeline_.push_back(ts_ns);
    while (keyframe_timeline_.back() - keyframe_timeline_.front() > s_.max_time_period_ns) {
      keyframe_timeline_.pop_front();
    }
  }

  if (track_result) {
    last_successfull_frame_time_ns = ts_ns;
  } else {
    keyframe_timeline_.clear();
  }

  if (ts_ns - last_successfull_frame_time_ns > s_.max_integration_time_ns) {
    reset();
    return state_;
  }

  if (keyframe_timeline_.size() >= s_.min_num_kf_for_gravity &&    // we have enough good keyframes in the map
      ts_ns - *keyframe_timeline_.begin() > s_.min_time_period_ns  // the success timeline is long enough
  ) {
    if (state_ == Ok) {
      // check enough time has passed since last gravity optimization
      if (ts_ns - last_gravity_estimation_time_ns > s_.gravity_update_period_ns) {
        // optimize
        bool res = gravity_calculate_callback_(s_.min_num_kf_for_gravity);
        if (!res) {
          reset();
          return Uninitialized;
        }
        last_gravity_estimation_time_ns = ts_ns;
        state_ = Ok;
        return state_;
      }

    } else {
      // optimize
      bool res = gravity_calculate_callback_(s_.min_num_kf_for_gravity);
      if (!res) {
        reset();
        return Uninitialized;
      }
      last_gravity_estimation_time_ns = ts_ns;
      state_ = Ok;
      return state_;
    }
  }

  return state_;
}

const StateMachine::State& StateMachine::get_state() const { return state_; }

}  // namespace cuvslam::pipelines


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

#include <functional>
#include <list>

#include "common/types.h"

namespace cuvslam::pipelines {

struct StateMachineSettings {
  int64_t gravity_update_period_ns = 5e8;  // 1 sec = 1e9 ns

  int64_t max_integration_time_ns = 1e9;

  size_t min_num_kf_for_gravity = 20;  // cant be larger then map size!

  int64_t min_time_period_ns = 2e9;
  int64_t max_time_period_ns = 8e9;
};

class StateMachine {
public:
  enum State { Ok, Uninitialized };

public:
  StateMachine(const StateMachineSettings& s = StateMachineSettings());

  State update_frame_state(bool is_keyframe, bool track_result, int64_t ts_ns);

  void register_gravity_estimation_callback(const std::function<bool(size_t num_kfs)>& cb);
  const State& get_state() const;

  void reset();

private:
  struct TrackResult {
    bool status;
    int64_t time_ns;
  };

  std::list<TrackResult> frames_timeline_;
  std::list<int64_t> keyframe_timeline_;

  int64_t last_gravity_estimation_time_ns = -1;
  int64_t last_successfull_frame_time_ns = -1;

  std::function<bool(size_t num_kfs)> gravity_calculate_callback_ = nullptr;

  State state_ = Uninitialized;
  StateMachineSettings s_;
};

}  // namespace cuvslam::pipelines

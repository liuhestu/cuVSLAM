
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

#include "sof/selector_interface.h"
#include "sof/sof.h"

namespace cuvslam::sof {

struct SelectorMonoSettings {
  // The current frame becomes keyframe if the percent of survivor tracks
  // from the last keyframe is less than this value
  float survivor_from_last = 41.f;

  // The current frame becomes keyframe if the percent of survivor tracks
  // from penultimate keyframe is less than this value
  float survivor_from_penultimate = 10.f;

  // The current frame becomes keyframe if the average motion
  // in normalized (/ width) values is higher than this value
  float av_motion_from_last = 0.194f;

  // Current frame is not keyframe until motion > min_av_motion_to_keyframe
  float min_av_motion_to_keyframe = 0.1f;
};

class SelectorMono : public ISelector {
public:
  SelectorMono(const SelectorMonoSettings& settings, bool mono);

  void set_image_width(uint32_t width) override;
  void reset_selector() override;
  bool select(const TracksVector& cur_frame_tracks) override;
  void set_tracks(const TracksVector& cur_frame_tracks) override;

private:
  SelectorMonoSettings settings_;
  bool mono_ = false;
  uint32_t width_ = 0;
  bool first_kf_selected_ = false;
  TracksVector last_but_one_kf_tracks_;  // expected to be ordered by TrackeId
  TracksVector last_kf_tracks_;          // expected to be ordered by TrackeId
};

}  // namespace cuvslam::sof

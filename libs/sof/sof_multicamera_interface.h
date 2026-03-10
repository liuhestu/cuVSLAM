
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

#include <memory>
#include <unordered_map>
#include <vector>

#include "camera/observation.h"
#include "common/image.h"
#include "common/isometry.h"

#include "sof/feature_prediction_interface.h"
#include "sof/image_manager.h"
#include "sof/kf_selector.h"
#include "sof/sof_mono_interface.h"

namespace cuvslam::sof {

class IMultiSOF {
public:
  virtual bool trackNextFrame(const Sources& curr_sources, Images& curr_images, const Images& prev_images,
                              const Sources& masks_sources, const Isometry3T& predicted_world_from_rig,
                              std::unordered_map<CameraId, std::vector<camera::Observation>>& observations,
                              FrameState& state) = 0;

  virtual void reset() = 0;

  virtual void reset_keyframe_selector() = 0;
  virtual ~IMultiSOF() = default;
};

}  // namespace cuvslam::sof


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

#include <vector>

#include "camera/observation.h"
#include "camera/rig.h"
#include "common/isometry.h"
#include "common/vector_2t.h"
#include "common/vector_3t.h"
#include "sof/sof_multicamera_interface.h"

namespace cuvslam::pipelines {

using MulticamObservations = std::unordered_map<CameraId, std::vector<camera::Observation>>;

class ISFMSolver {
public:
  virtual ~ISFMSolver() = default;

  virtual const camera::Rig& getRig() const = 0;

  // IN:  time_ns, frameState,
  //      observations - observations from all cameras
  // OUT: world_from_rig - if return code is true - estimated world_from_rig otherwise - previous pose.
  //                       Start pose is Identity.
  //      static_info_exp - information matrix for static frame in exponential mapping form
  //      tracks2d        - optional output 2d track coordinates in pixels
  //      tracks3d        - in camera space
  // return true if accurate solution was found
  virtual bool solveNextFrame(int64_t time_ns, const sof::FrameState& frameState,
                              const MulticamObservations& observations, Isometry3T& world_from_rig,
                              Matrix6T& static_info_exp, std::vector<Track2D>* tracks2d = nullptr,
                              Tracks3DMap* tracks3d = nullptr) = 0;

  virtual void reset() = 0;
};

}  // namespace cuvslam::pipelines

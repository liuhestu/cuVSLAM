
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

#include "camera_rig_edex/blackout_oscillator_filter.h"

#include "common/imu_measurement.h"
#include "common/log.h"

namespace cuvslam::camera_rig_edex {

BlackoutOscillatorFilter::BlackoutOscillatorFilter(std::unique_ptr<ICameraRig>&& base, int period, int duration)
    : base_(std::move(base)), period_(period), duration_(duration) {
  assert(period > duration);
}

ErrorCode BlackoutOscillatorFilter::start() { return base_->start(); }

ErrorCode BlackoutOscillatorFilter::stop() { return base_->stop(); }

const camera::ICameraModel& BlackoutOscillatorFilter::getIntrinsic(uint32_t index) const {
  return base_->getIntrinsic(index);
}

const Isometry3T& BlackoutOscillatorFilter::getExtrinsic(uint32_t index) const { return base_->getExtrinsic(index); }

ErrorCode BlackoutOscillatorFilter::getFrame(Sources& sources, Metas& metas, Sources& masks_sources,
                                             DepthSources& depth_sources) {
  ErrorCode status = base_->getFrame(sources, metas, masks_sources, depth_sources);

  assert(sources.size() == metas.size());

  if (status == ErrorCode::S_True && !metas.empty()) {
    int frame_id = static_cast<int>(metas[0].frame_id);
    int index = frame_id % period_;
    if (index < duration_ && frame_id >= period_) {
      // do blackout
      for (size_t i = 0; i < metas.size(); ++i) {
        auto& source = sources[i];
        auto& meta = metas[i];

        void* data = source.data;
        int size = meta.shape.width * meta.shape.height;
        if (source.type == ImageSource::U8) {
          memset(data, 0, size);
        }
        if (source.type == ImageSource::F32) {
          memset(data, 0, size * sizeof(float));
        }
      }
      for (auto& [cam_id, source] : depth_sources) {
        const auto& meta = metas.at(cam_id);
        int size = meta.shape.width * meta.shape.height;

        void* depth_data = source.data;
        memset(depth_data, 0, size * sizeof(float));
      }
      log::Value<LogRoot>("blackout_oscilator", true);
    } else {
      log::Value<LogRoot>("blackout_oscilator", false);
    }
  }

  return status;
}

void BlackoutOscillatorFilter::registerIMUCallback(
    const std::function<void(const imu::ImuMeasurement& integrator)>& func) {
  base_->registerIMUCallback(func);
}

}  // namespace cuvslam::camera_rig_edex

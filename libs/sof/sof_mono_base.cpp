
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

#include "sof/internal/sof_mono_base.h"

namespace cuvslam::sof {

void MonoSOFBase::PredictFeatureLocations(const Isometry3T& predicted_world_from_rig) {
  const auto numTracks = tracks_.size();
  predictionTrackIds_.resize(numTracks);
  predictedUVs_.resize(numTracks);

  for (unsigned i = 0; i < numTracks; ++i) {
    predictionTrackIds_[i] = tracks_[i].id();
  }

  if (!feature_predictor_) {
    return;
  }

  feature_predictor_->predictObservations(predicted_world_from_rig, cam_id_, predictionTrackIds_, predictedUVs_);
}

void MonoSOFBase::filterByPredictionError() {
  if (!feature_predictor_) {
    return;
  }

  const size_t n = tracks_.size();

  // discrepancy between the prediction and the estimated position
  std::vector<float> deltas;
  deltas.reserve(n);

  for (unsigned k = 0; k < n; ++k) {
    if (tracks_[k].dead()) {
      continue;
    }

    if (!predictedUVs_[k]) {
      // point was not triangulated
      continue;
    }
    const Vector2T& predicted_uv = *predictedUVs_[k];
    const Vector2T delta = predicted_uv - tracks_[k].position();
    deltas.push_back(delta.norm());
  }

  if (deltas.size() >= 3U) {
    // we are going to reorder deltas
    auto originalDeltas = deltas;
    std::nth_element(deltas.begin(), deltas.begin() + deltas.size() / 2, deltas.end());

    // This threshold is somewhat arbitrary, but it gives
    // good results on KITTI.
    // More rigorous bound should probably include an estimate
    // of standard deviation.
    const auto threshold = std::max(3.f, 2.f * deltas[deltas.size() / 2]);

    auto d = originalDeltas.begin();
    for (unsigned k = 0; k < n; ++k) {
      if (tracks_[k].dead()) {
        continue;
      }
      if (!predictedUVs_[k]) {
        // point was not triangulated
        continue;
      }
      if (*d > threshold) {
        tracks_.kill(k);
      }
      ++d;
    }
  }
}

void MonoSOFBase::IncrementTracksAge() {
  const size_t n = tracks_.size();
  for (size_t i = 0; i < n; ++i) {
    if (!tracks_[i].dead()) {
      tracks_.IncrementAge(i);
    }
  }
}

void MonoSOFBase::PrepareInputMask(const ImageShape& shape) {
  if (!input_mask_present_) {
    // Prepare the mask even if not present because it's required for the first frame
    const size_t n_rows = shape.height;
    const size_t n_cols = shape.width;
    input_mask_.resize(n_rows, n_cols);
  }
  input_mask_.setZero();
}

void MonoSOFBase::KillTracksWithinMask() {
  if (input_mask_present_) {
    for (size_t i = 0; i < tracks_.size(); ++i) {
      const Track& track = tracks_[i];
      if (track.dead()) {
        continue;
      }
      const Vector2T& p = track.position();
      if (input_mask_(static_cast<size_t>(p.y()), static_cast<size_t>(p.x())) != 0) {
        tracks_.kill(i);
      }
    }
  }
}

}  // namespace cuvslam::sof

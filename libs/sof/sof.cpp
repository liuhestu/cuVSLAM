
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

#include "sof/sof.h"

#include <string>

#include "sof/klt_tracker.h"
#include "sof/lk_tracker.h"
#include "sof/st_tracker.h"

namespace cuvslam::sof {

std::unique_ptr<IFeatureTracker> CreateTracker(const char* name) {
  if (std::string("klt") == name) {
    return std::make_unique<KLTTracker>();
  } else if (std::string("lk") == name) {
    return std::make_unique<LKFeatureTracker>();
  }
  /*else if (std::string("st2") == name)
  {
      return std::make_unique<STTracker>(20, 0);
  }
  else if (std::string("st6") == name)
  {
      return std::make_unique<STTracker>(0, 20);
  }*/
  else if (std::string("lk_horizontal") == name) {
    return std::make_unique<LKTrackerHorizontal>();
  } else if (std::string("klt_horizontal") == name) {
    return std::make_unique<KLTTrackerHorizontal>();
  }

  TraceError("Unknown CPU tracker name=%s", name);
  return nullptr;
}

}  // namespace cuvslam::sof

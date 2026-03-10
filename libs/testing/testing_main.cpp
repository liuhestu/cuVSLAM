
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

#include <chrono>

#include "gflags/gflags.h"

#include "common/include_gtest.h"
#include "common/log.h"

#define CUVSLAM_TEST_RAND_SEED \
  static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count())

DEFINE_VARIABLE(gflags::uint64, U64, cuvslam_test_seed, CUVSLAM_TEST_RAND_SEED,
                "Set initial seed for random generator");

DEFINE_string(logger_filename, "", "Log filename");

namespace test::utils {

class cuVSLAMTestRandSeedClass {
  mutable unsigned int init_;
  mutable unsigned int seed_;

public:
  cuVSLAMTestRandSeedClass(unsigned int s) : init_(s), seed_(0) {}

  void init(unsigned int s) const { init_ = s; }

  unsigned int get() const { return seed_; }

  void seed() const {
    seed_ = init_;
    std::srand(seed_);
    init_ += static_cast<unsigned int>(std::rand());
  }
};

extern const cuVSLAMTestRandSeedClass CUVSLAM_TEST_SEED(0);

class GTestEvenListener : public ::testing::EmptyTestEventListener {
  void OnTestIterationStart(const ::testing::UnitTest&, int) override {
    CUVSLAM_TEST_SEED.seed();
    printf("\n[CUVSLAM_TEST_RAND_SEED]\t%u\n\n", CUVSLAM_TEST_SEED.get());
  }
};

}  // namespace test::utils

using namespace test::utils;

int main(int argc, char** argv) {
  // This allows the user to override the flag on the command line.
  ::testing::InitGoogleTest(&argc, argv);

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  cuvslam::Trace::SetVerbosity(cuvslam::Trace::Verbosity::Error);
  // log
  if (!FLAGS_logger_filename.empty()) {
#ifdef CUVSLAM_LOG_ENABLE
    std::cout << "Create spd logger: " << FLAGS_logger_filename << std::endl;
    auto logger = cuvslam::log::CreateSpdlogLogger(FLAGS_logger_filename.c_str());
    cuvslam::log::SetLogger(logger);
#else  // !CUVSLAM_LOG_ENABLE
    std::cout << "No CUVSLAM_LOG_ENABLE definition. Flag -logger_filename will be ignored " << std::endl;
#endif
  }

  CUVSLAM_TEST_SEED.init(static_cast<unsigned int>(FLAGS_cuvslam_test_seed));

  // Gets hold of the event listener list.
  ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();

  // Adds a listener to the end of the list.
  listeners.Append(new GTestEvenListener);

  return RUN_ALL_TESTS();
}

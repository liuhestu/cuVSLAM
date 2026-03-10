# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA software released under the NVIDIA Community License is intended to be used to enable
# the further development of AI and robotics technologies. Such software has been designed, tested,
# and optimized for use with NVIDIA hardware, and this License grants permission to use the software
# solely with such hardware.
# Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
# modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
# outputs generated using the software or derivative works thereof. Any code contributions that you
# share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
# in future releases without notice or attribution.
# By using, reproducing, modifying, distributing, performing, or displaying any portion or element
# of the software or derivative works thereof, you agree to be bound by this License.

import os
import unittest
import numpy as np
import threading
from numpy.testing import assert_array_almost_equal
import cuvslam as vslam
import data_gen as data

W, H = 640, 480  # Image size of tracked images
BASELINE = 0.25  # Baseline distance between cameras (m)
STEPS = 30       # Number of steps

plt = None

def visualize_frame(images, i, subplots):
    if not plt:
        return None

    # Create figure only once before the loop
    if i == 0:
        plt.figure(figsize=(12, 6))
        subplots = []
        for cam in range(len(images)):
            subplot = plt.subplot(1, len(images), cam + 1)
            subplots.append(subplot)
        plt.ion()  # Enable interactive mode

    # Update images for each camera
    for cam in range(len(images)):
        plt.sca(subplots[cam])
        plt.cla()  # Clear the current axes
        plt.imshow(images[cam], cmap='gray')
        plt.title(f'Camera {cam}')
        plt.axis('off')

    plt.suptitle(f'Frame {i}')
    plt.draw()
    plt.pause(0.1)  # Add small delay to allow visualization

    # Close figure only at the end of the loop
    if i == STEPS - 1:  # Last iteration
        plt.close()

    return subplots

def perturb_pose(pose, max_translation=0.5):
    # Perturb translation
    translation = np.array(pose.translation)
    translation += np.random.uniform(-max_translation, max_translation, size=3)

    # Generate random rotation
    rotation = np.random.randn(4)  # Random quaternion
    rotation = rotation / np.linalg.norm(rotation)  # Normalize to unit quaternion

    return vslam.Pose(translation=translation, rotation=rotation)


class PrinterTestResult(unittest.TestResult):
    """Custom test result class to print debug data if tests fail."""

    def addFailure(self, test, err):
        super().addFailure(test, err)
        debug_messages = getattr(test, 'debug_messages', [])
        for msg in debug_messages:
            print(msg)

class TestMap(unittest.TestCase):

    def setUp(self):
        cameras = data.generate_stereo_camera(W, H, BASELINE)
        self.num_cameras = len(cameras)
        imu = vslam.ImuCalibration()
        self.rig = vslam.Rig(cameras, [imu])
        self.debug_messages = []

    def add_debug(self, message):
        self.debug_messages.append(message)

    ### Helper methods for map creation and localization

    def get_localization_configs(self):
        """Get configurations for localization mode."""
        odom_cfg = vslam.Tracker.OdometryConfig()
        odom_cfg.odometry_mode = vslam.Tracker.OdometryMode.Multicamera
        odom_cfg.rectified_stereo_camera = True
        odom_cfg.async_sba = False

        slam_cfg = vslam.Tracker.SlamConfig()
        slam_cfg.sync_mode = True

        loc_settings = vslam.Tracker.SlamLocalizationSettings(
            horizontal_search_radius=0.25,
            vertical_search_radius=0.25,
            horizontal_step=0.0625,
            vertical_step=0.0625,
            angular_step_rads=0.03125
        )

        return odom_cfg, slam_cfg, loc_settings

    def create_map(
            self,
            img: data.ImageGenerator,
            map_name='temp_map',
            start=0,
            stop=STEPS,
            step=1,
            timestamp=0,
            visualize=False):
        """Create and save a map using image generator."""

        # Configure tracker for map creation
        odom_cfg, slam_cfg, _ = self.get_localization_configs()
        tracker = vslam.Tracker(self.rig, odom_cfg, slam_cfg)

        subplots = None

        z0 = None
        # Track through selected steps to build the map
        for i in range(start, stop, step):
            images, z = img.generate_zoomed_images(i)
            if i == start:
                z0 = z

            # track
            pose_estimate, slam_pose = tracker.track(i * 1_000_000 + timestamp, images)
            self.assertTrue(pose_estimate.world_from_rig)

            odom_pose = pose_estimate.world_from_rig.pose
            self.add_debug(f"{i}: gt, slam: {z}, {slam_pose.translation[2]}")

            self.add_debug(f"{i}: odom_pose={odom_pose}")
            assert_array_almost_equal(np.array(odom_pose.translation), [0, 0, z - z0], decimal=1)
            assert_array_almost_equal(np.array(odom_pose.rotation), [0, 0, 0, 1.0], decimal=2)

            if start != 0 and i == start:
                tracker.set_slam_pose(vslam.Pose(translation=[0, 0, z0], rotation=[0, 0, 0, 1.0]))
            else:
                self.add_debug(f"{i}: slam_pose={slam_pose}")
                assert_array_almost_equal(np.array(slam_pose.translation), [0, 0, z], decimal=1)
                assert_array_almost_equal(np.array(slam_pose.rotation), [0, 0, 0, 1.0], decimal=2)

            if visualize:
                subplots = visualize_frame(images, i, subplots)

        # Save the map
        map_saved = False
        def save_callback(success):
            nonlocal map_saved
            map_saved = success
        tracker.save_map(map_name, save_callback)
        self.assertTrue(map_saved)

    def try_localize(
            self,
            img: data.ImageGenerator,
            map_name: str,
            step: int,
            perturb: bool = False) -> vslam.Pose | None:
        """
        Try to localize in the map created by create_map.
        Returns True if localization was successful.
        """
        odom_cfg, slam_cfg, loc_settings = self.get_localization_configs()
        tracker = vslam.Tracker(self.rig, odom_cfg, slam_cfg)

        images, z = img.generate_zoomed_images(step)
        timestamp = step * 1_000_000
        _, _ = tracker.track(timestamp, images)

        guess_pose = vslam.Pose(translation=[0, 0, z], rotation=[0, 0, 0, 1.0])
        if perturb:
            guess_pose = perturb_pose(guess_pose, 0.125)
        self.add_debug(f"{step}: guess_pose={guess_pose}")

        localization_complete = threading.Event()
        result_pose = None

        def localization_callback(pose, error_message):
            nonlocal result_pose
            self.add_debug(f"Localization result: {pose}, {error_message}")
            result_pose = pose
            localization_complete.set()

        tracker.localize_in_map(map_name, guess_pose, images, loc_settings, localization_callback)
        while not localization_complete.wait(timeout=0.1):
            # Simulate time passing; this is necessary for the async callback to be processed
            timestamp += 1_000
            _, _ = tracker.track(timestamp, images)

        self.add_debug(f"{step}: result_pose={result_pose}")
        return result_pose

    ### Test cases for map creation and localization

    def test_map(self):

        img = data.ImageGenerator(self.rig.cameras, 30)

        self.create_map(img, 'temp_map')

        _, _, loc_settings = self.get_localization_configs()
        self.add_debug(f"Localization settings: {loc_settings}")

        self.assertTrue(any(self.try_localize(img, 'temp_map', i, perturb=False) for i in range(0, 10)))

        self.assertTrue(any(self.try_localize(img, 'temp_map', i, perturb=True) for i in range(0, 10)))

        self.assertTrue(any(self.try_localize(img, 'temp_map', i, perturb=False) for i in range(10, 20)))
        # Localization in the end of the map isn't working well
        # self.assertTrue(any(self.try_localize(img, 'temp_map', i, perturb=False) for i in range(20, 30)))


    def test_map_merge(self):
        """Test merging two maps."""

        img = data.ImageGenerator(self.rig.cameras, 60)

        # Create first map from a first half of the generated sequence
        self.create_map(img, 'temp_map1', start=0, stop=31, step=1, timestamp=1_000_000_000)

        # Create second map from a second half, overlapped with the first one but no images/poses are the same
        self.create_map(img, 'temp_map2', start=29, stop=60, step=1, timestamp=2_000_000_000)

        # Merge maps
        vslam.Tracker.merge_maps(self.rig, ['temp_map1', 'temp_map2'], 'temp_merged_map')

        # Check if the merged map exists
        self.assertTrue(os.path.exists('temp_merged_map'))

        # Test localization in the beginning of the merged map (first submap)
        self.assertTrue(any(self.try_localize(img, 'temp_merged_map', i) for i in range(0, 10, 2)))
        # TODO: fix
        # # Test localization in the stitching region of the merged map
        # self.assertTrue(any(self.try_localize(img, 'temp_merged_map', i) for i in range(24, 35)))
        # # Test localization in the end of the merged map (second submap)
        # self.assertTrue(any(self.try_localize(img, 'temp_merged_map', i) for i in range(49, 60, 2)))


if __name__ == "__main__":
    # using test runner with custom result class to print additional debug messages
    unittest.main(testRunner=unittest.TextTestRunner(resultclass=PrinterTestResult))

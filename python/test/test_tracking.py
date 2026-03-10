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

import unittest
import numpy as np
import cuvslam as vslam
import data_gen as data

class TestTracking(unittest.TestCase):
    def setUp(self):
        cameras = data.generate_stereo_camera(640, 480, baseline=0.25)
        self.num_cameras = len(cameras)
        imu = vslam.ImuCalibration()
        self.rig = vslam.Rig(cameras, [imu])

    def test_init_arguments(self):
        # accept partial, positional & keyword arguments
        _ = vslam.Tracker(self.rig, vslam.Tracker.OdometryConfig())
        _ = vslam.Tracker(self.rig)
        _ = vslam.Tracker(rig=self.rig, odom_config=vslam.Tracker.OdometryConfig())
        _ = vslam.Tracker(rig=self.rig)
        _ = vslam.Tracker(rig=self.rig, slam_config=vslam.Tracker.SlamConfig())

        with self.assertRaises(TypeError):
            vslam.Tracker()  # type: ignore # missing required argument "rig"

    @unittest.skip("TODO: add a check that cameras don't have the same pose")
    def test_init_same_cameras(self):
        # don't accept rig with duplicate cameras
        bad_rig = vslam.Rig(cameras=[self.rig.cameras[0], self.rig.cameras[0]])
        with self.assertRaises(ValueError):
            vslam.Tracker(bad_rig)

    def test_init_no_cameras(self):
        # don't accept empty rig with no cameras
        with self.assertRaises(ValueError):
            vslam.Tracker(vslam.Rig([]))

    def test_init_different_sizes(self):
        # don't accept cameras with different sizes
        self.rig.cameras[1].size[0] = 480
        with self.assertRaises(ValueError):
            vslam.Tracker(self.rig)

    def test_init_inertial_no_imus(self):
        # don't accept rig with no IMUs in inertial mode
        self.rig.imus = []
        cfg = vslam.Tracker.OdometryConfig()
        cfg.odometry_mode = vslam.Tracker.OdometryMode.Inertial
        with self.assertRaises(ValueError):
            vslam.Tracker(self.rig, cfg)

    def test_init_multiple_imus(self):
        # don't accept rig with multiple IMUs
        self.rig.imus = [vslam.ImuCalibration(), vslam.ImuCalibration()]
        cfg = vslam.Tracker.OdometryConfig()
        cfg.odometry_mode = vslam.Tracker.OdometryMode.Inertial
        with self.assertRaises(ValueError):
            vslam.Tracker(self.rig)

    def test_init_negative_image_size(self):
        # image sizes should be positive
        self.rig.cameras[0].size[0] = -1
        self.rig.cameras[1].size[0] = -1
        with self.assertRaises(ValueError):
            vslam.Tracker(self.rig)

    def test_tracking(self):
        modes = [vslam.Tracker.OdometryMode.Multicamera, vslam.Tracker.OdometryMode.Inertial,
                 vslam.Tracker.OdometryMode.Mono, vslam.Tracker.OdometryMode.RGBD]
        synthetic_images = [np.zeros((480, 640), dtype=np.uint8) for _ in range(self.num_cameras)]
        synthetic_masks = [np.ones((480, 640), dtype=np.uint8) for _ in range(self.num_cameras)]
        synthetic_depths = [np.random.randint(0, 1024, size=(480, 640), dtype=np.uint16)]
        print("")  # to insert a newline after python unittest output
        for mode in modes:
            for with_mask in [False, True]:
                with self.subTest(mode=mode, with_mask=with_mask):
                    print(f"Testing mode={mode}, with_mask={with_mask}")
                    cfg = vslam.Tracker.OdometryConfig()
                    cfg.odometry_mode = mode
                    cfg.rgbd_settings.depth_scale_factor = 1000.0
                    cfg.rgbd_settings.depth_camera_id = 0
                    cfg.rgbd_settings.enable_depth_stereo_tracking = False
                    tracker = vslam.Tracker(self.rig, cfg)
                    for i in range(60):
                        # Add 10 IMU measurements between frames
                        if mode == vslam.Tracker.OdometryMode.Inertial:
                            for j in range(10):
                                imu = vslam.ImuMeasurement()
                                imu.timestamp_ns = i * 1000 + j * 100
                                # Gravity points down (+Y in OpenCV)
                                imu.linear_accelerations = [0.0, 9.81, 0.0]
                                # No rotation
                                imu.angular_velocities = [0.0, 0.0, 0.0]
                                tracker.register_imu_measurement(0, imu)
                        odom_pose, slam_pose = tracker.track(
                            (i + 1) * 1000, synthetic_images, synthetic_masks if with_mask else None,
                            synthetic_depths if mode == vslam.Tracker.OdometryMode.RGBD else None)
                        self.assertIs(slam_pose, None)
                        if odom_pose.world_from_rig:
                            np.testing.assert_array_almost_equal(
                                odom_pose.world_from_rig.pose.rotation, [0.0, 0.0, 0.0, 1.0],
                                err_msg=f"iteration {i}")
                            np.testing.assert_array_almost_equal(
                                odom_pose.world_from_rig.pose.translation, [0.0, 0.0, 0.0],
                                err_msg=f"iteration {i}")
                            # TODO: fix gravity
                            # if mode == vslam.Tracker.OdometryMode.Inertial:
                            #     gravity = tracker.get_last_gravity()
                            #     if gravity is not None:
                            #         self.assertAlmostEqual(gravity[1], 9.81, msg=f"iteration {i}")


    def test_slam_pose(self):
        img = data.ImageGenerator(self.rig.cameras, 10)
        synthetic_images, _ = img.generate_zoomed_images(0)

        cfg = vslam.Tracker.OdometryConfig()
        cfg.odometry_mode = vslam.Tracker.OdometryMode.Multicamera

        s_cfg = vslam.Tracker.SlamConfig()
        s_cfg.sync_mode = True

        tracker = vslam.Tracker(self.rig, cfg, s_cfg)

        identity = vslam.Pose(translation = [0.0, 0.0, 0.0], rotation = [0.0, 0.0, 0.0, 1.0])
        odom_pose, slam_pose = tracker.track(1000, synthetic_images)

        np.testing.assert_array_almost_equal(slam_pose.translation, identity.translation)
        np.testing.assert_array_almost_equal(slam_pose.rotation, identity.rotation)
        np.testing.assert_array_almost_equal(odom_pose.world_from_rig.pose.translation, identity.translation)
        np.testing.assert_array_almost_equal(odom_pose.world_from_rig.pose.rotation, identity.rotation)

        new_pose = vslam.Pose(translation = [1.0, 2.0, 3.0], rotation = [0.5, 0.5, 0.5, -0.5])
        tracker.set_slam_pose(new_pose)
        odom_pose, slam_pose = tracker.track(2000, synthetic_images)

        np.testing.assert_array_almost_equal(slam_pose.translation, new_pose.translation)
        np.testing.assert_array_almost_equal(slam_pose.rotation, new_pose.rotation)
        np.testing.assert_array_almost_equal(odom_pose.world_from_rig.pose.translation, identity.translation)
        np.testing.assert_array_almost_equal(odom_pose.world_from_rig.pose.rotation, identity.rotation)

if __name__ == "__main__":
    unittest.main()

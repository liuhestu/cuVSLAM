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


class TestBindings(unittest.TestCase):
    def test_version(self):
        ver, major, minor, patch = vslam.get_version()
        self.assertIsInstance(ver, str)
        self.assertIsInstance(major, int)
        self.assertIsInstance(minor, int)
        self.assertIsInstance(patch, int)
        expected_prefix = f"{major}.{minor}.{patch}+"
        self.assertTrue(ver.startswith(expected_prefix),
                        f"Expected version to start with '{expected_prefix}', got '{ver}'")

    def test_distortion_default(self):
        d = vslam.Distortion()
        self.assertEqual(d.model, vslam.Distortion.Model.Pinhole)
        self.assertEqual(d.parameters, [])

    def test_distortion_constructor(self):
        d1 = vslam.Distortion(vslam.Distortion.Model.Polynomial,
                              [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(d1.model, vslam.Distortion.Model.Polynomial)
        self.assertEqual(d1.parameters, [1, 2, 3, 4, 5, 6, 7, 8])

        d2 = vslam.Distortion(parameters=[1, 2, 3, 4, 5],
                              model=vslam.Distortion.Model.Brown)
        self.assertEqual(d2.model, vslam.Distortion.Model.Brown)
        self.assertEqual(d2.parameters, [1, 2, 3, 4, 5])

    def test_distortion_assignment(self):
        d = vslam.Distortion()
        d.model = vslam.Distortion.Model.Fisheye
        self.assertEqual(d.model, vslam.Distortion.Model.Fisheye)
        self.assertEqual(d.parameters, [])
        d.parameters = [4, 3, 2, 1]
        self.assertEqual(d.parameters, [4, 3, 2, 1])

        # Distortion parameters are not individually mutable, they are supposed to be obtained
        # as a whole from calibration but this leads to an unexpected behavior in python
        # (would be better if exception was raised by nanobind)
        d.parameters[0] = 1000000
        self.assertEqual(d.parameters, [4, 3, 2, 1])

    def test_pose_default(self):
        p = vslam.Pose()
        np.testing.assert_array_equal(p.translation, [0, 0, 0])
        np.testing.assert_array_equal(p.rotation, [0, 0, 0, 1])

    def test_pose_assignment(self):
        p = vslam.Pose()
        p.translation = [1, 2, 3]
        p.rotation = [0, 0, 0.7071, 0.7071]
        np.testing.assert_array_equal(p.translation, [1, 2, 3])
        np.testing.assert_array_almost_equal(p.rotation, [0, 0, 0.7071, 0.7071])

        with self.assertRaisesRegex(ValueError, 'Sequence must have exactly 3 elements'):
            p.translation = []
        with self.assertRaisesRegex(ValueError, 'Sequence must have exactly 3 elements'):
            p.translation = [0, 0]
        with self.assertRaisesRegex(ValueError, 'Sequence must have exactly 3 elements'):
            p.translation = [0, 0, 0, 0]

        with self.assertRaises(TypeError):
            p.translation = None  # type: ignore
        with self.assertRaises(TypeError):
            p.translation = {"x": 1, "y": 2, "z": 3}  # type: ignore

    def test_pose_constructor(self):
        p1 = vslam.Pose([0, 0, 0.7071, 0.7071], [1, 2, 3])
        np.testing.assert_array_equal(p1.translation, [1, 2, 3])
        np.testing.assert_array_almost_equal(p1.rotation, [0, 0, 0.7071, 0.7071])

        p2 = vslam.Pose(translation=[1, 2, 3], rotation=[0, 0, 0, 1])
        np.testing.assert_array_equal(p2.translation, [1, 2, 3])
        np.testing.assert_array_equal(p2.rotation, [0, 0, 0, 1])

    def test_pose_element_access(self):
        p = vslam.Pose()
        p.translation = [1, 2, 3]
        p.rotation = [0, 0, 0, 1]

        p.translation[0] = 9
        p.translation[1] *= 3
        p.rotation[2] = 0.5
        np.testing.assert_array_equal(p.translation, [9, 6, 3])
        np.testing.assert_array_equal(p.rotation, [0, 0, 0.5, 1])

        with self.assertRaisesRegex(IndexError, 'index 3 is out of bounds'):
            p.translation[3] = 1

        with self.assertRaisesRegex(IndexError, 'index 4 is out of bounds'):
            p.rotation[4] = 1

        # Cannot prevent assignment of some non-numeric types convertible to float
        p.translation[0] = None  # type: ignore # NaN
        p.translation[0] = "42"  # type: ignore # float("42")==42.0

        with self.assertRaises(ValueError):
            p.translation[0] = "nonsence"  # type: ignore

    def test_camera_default(self):
        c = vslam.Camera()
        np.testing.assert_array_equal(c.size, (0, 0))
        np.testing.assert_array_equal(c.focal, (0.0, 0.0))
        np.testing.assert_array_equal(c.principal, (0.0, 0.0))
        # TODO: add __eq__ to distortion?
        # self.assertEqual(c.distortion, vslam.Distortion())
        self.assertEqual(c.distortion.model, vslam.Distortion.Model.Pinhole)
        self.assertEqual(c.distortion.parameters, [])

    def test_camera_constructor(self):
        c1 = vslam.Camera(
            size=[640, 480], principal=[320, 240], focal=[320.0, 320.0],
            distortion=vslam.Distortion(vslam.Distortion.Model.Polynomial, [1, 2, 3, 4, 5, 6, 7, 8]))
        np.testing.assert_array_equal(c1.size, (640, 480))
        np.testing.assert_array_equal(c1.focal, (320.0, 320.0))
        np.testing.assert_array_equal(c1.principal, (320.0, 240.0))
        self.assertEqual(c1.distortion.model, vslam.Distortion.Model.Polynomial)
        self.assertEqual(c1.distortion.parameters, [1, 2, 3, 4, 5, 6, 7, 8])

        # unnamed parameters are not supported
        with self.assertRaises(TypeError):
            vslam.Camera((640, 480), (320, 240), (320.0, 320.0))  # type: ignore

    def test_rig_constructor(self):
        r = vslam.Rig()
        self.assertEqual(len(r.cameras), 0)
        self.assertEqual(len(r.imus), 0)

        # accept partial, positional & keyword arguments
        r1 = vslam.Rig([], [])
        self.assertEqual(len(r1.cameras), 0)
        self.assertEqual(len(r1.imus), 0)

        r2 = vslam.Rig(cameras=[vslam.Camera()])
        self.assertEqual(len(r2.cameras), 1)
        self.assertEqual(len(r2.imus), 0)

        r3 = vslam.Rig([vslam.Camera()])
        self.assertEqual(len(r3.cameras), 1)
        self.assertEqual(len(r3.imus), 0)

        r4 = vslam.Rig(cameras=[vslam.Camera()], imus=[vslam.ImuCalibration()])
        self.assertEqual(len(r4.cameras), 1)
        self.assertEqual(len(r4.imus), 1)

        with self.assertRaises(TypeError):
            vslam.Rig([vslam.ImuCalibration()], [vslam.Camera()])  # type: ignore

    def test_rig_modifiers(self):
        r = vslam.Rig()
        r.cameras = [vslam.Camera()]
        self.assertEqual(len(r.cameras), 1)
        self.assertEqual(len(r.imus), 0)
        # TODO: support appending
        # rig.cameras.append(vslam.Camera())
        # self.assertEqual(len(rig.cameras), 2)

    def test_imu_measurement_assignment(self):
        imu_measurement = vslam.ImuMeasurement()
        imu_measurement.timestamp_ns = 1234567890
        imu_measurement.linear_accelerations = np.array([1, 2, 3])  # type: ignore
        imu_measurement.angular_velocities = (4, 5, 6)
        self.assertEqual(imu_measurement.timestamp_ns, 1234567890)
        np.testing.assert_array_equal(imu_measurement.linear_accelerations, [1, 2, 3])
        np.testing.assert_array_equal(imu_measurement.angular_velocities, np.array([4, 5, 6]))

    def test_tracker_config_constructor(self):
        # Test default constructor
        cfg = vslam.Tracker.OdometryConfig()
        self.assertEqual(cfg.multicam_mode, vslam.Tracker.MulticameraMode.Precision)
        self.assertEqual(cfg.odometry_mode, vslam.Tracker.OdometryMode.Multicamera)
        self.assertTrue(cfg.use_gpu)
        self.assertTrue(cfg.async_sba)
        self.assertTrue(cfg.use_motion_model)
        self.assertFalse(cfg.use_denoising)
        self.assertFalse(cfg.rectified_stereo_camera)
        self.assertTrue(cfg.enable_observations_export)
        self.assertTrue(cfg.enable_landmarks_export)
        self.assertFalse(cfg.enable_final_landmarks_export)
        self.assertEqual(cfg.max_frame_delta_s, 1.0)
        self.assertEqual(cfg.debug_dump_directory, "")
        self.assertFalse(cfg.debug_imu_mode)

        # Test constructor with keyword arguments
        cfg = vslam.Tracker.OdometryConfig(
            multicam_mode=vslam.Tracker.MulticameraMode.Performance,
            odometry_mode=vslam.Tracker.OdometryMode.Inertial,
            use_gpu=False,
            async_sba=False,
            use_motion_model=False,
            use_denoising=True,
            rectified_stereo_camera=True,
            enable_observations_export=False,
            enable_landmarks_export=False,
            enable_final_landmarks_export=True,
            max_frame_delta_s=0.5,
            debug_dump_directory="/tmp/debug"
        )
        self.assertEqual(cfg.multicam_mode, vslam.Tracker.MulticameraMode.Performance)
        self.assertEqual(cfg.odometry_mode, vslam.Tracker.OdometryMode.Inertial)
        self.assertFalse(cfg.use_gpu)
        self.assertFalse(cfg.async_sba)
        self.assertFalse(cfg.use_motion_model)
        self.assertTrue(cfg.use_denoising)
        self.assertTrue(cfg.rectified_stereo_camera)
        self.assertFalse(cfg.enable_observations_export)
        self.assertFalse(cfg.enable_landmarks_export)
        self.assertTrue(cfg.enable_final_landmarks_export)
        self.assertEqual(cfg.max_frame_delta_s, 0.5)
        self.assertEqual(cfg.debug_dump_directory, "/tmp/debug")

    def test_tracker_config_modifiers(self):
        cfg = vslam.Tracker.OdometryConfig()
        cfg.multicam_mode = vslam.Tracker.MulticameraMode.Moderate
        cfg.odometry_mode = vslam.Tracker.OdometryMode.Mono
        cfg.use_gpu = False
        cfg.async_sba = False
        cfg.use_motion_model = False
        cfg.use_denoising = True
        cfg.rectified_stereo_camera = True
        cfg.enable_observations_export = True
        cfg.enable_landmarks_export = True
        cfg.enable_final_landmarks_export = True
        cfg.max_frame_delta_s = 2.0
        cfg.debug_dump_directory = "/tmp/test"

        self.assertEqual(cfg.multicam_mode, vslam.Tracker.MulticameraMode.Moderate)
        self.assertEqual(cfg.odometry_mode, vslam.Tracker.OdometryMode.Mono)
        self.assertFalse(cfg.use_gpu)
        self.assertFalse(cfg.async_sba)
        self.assertFalse(cfg.use_motion_model)
        self.assertTrue(cfg.use_denoising)
        self.assertTrue(cfg.rectified_stereo_camera)
        self.assertTrue(cfg.enable_observations_export)
        self.assertTrue(cfg.enable_landmarks_export)
        self.assertTrue(cfg.enable_final_landmarks_export)
        self.assertEqual(cfg.max_frame_delta_s, 2.0)
        self.assertEqual(cfg.debug_dump_directory, "/tmp/test")


if __name__ == "__main__":
    unittest.main()

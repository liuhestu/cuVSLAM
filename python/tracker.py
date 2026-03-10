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

"""
cuVSLAM Tracker - A wrapper combining Odometry and SLAM functionality.
"""

from typing import Any, Optional, List, Callable, Tuple
from .pycuvslam import Odometry, Slam, Rig, PoseEstimate, Pose, ImuMeasurement, Observation, Landmark, PoseStamped

ImageArray = Any  # np.ndarray, torch.Tensor, etc. (CPU or CUDA)


class Tracker:
    """
    A wrapper that combines cuVSLAM Odometry and SLAM functionality.

    This class automatically manages both the Odometry and SLAM instances,
    providing a simplified interface for common use cases.
    """

    # Expose inner classes of Odometry & Slam
    OdometryMode = Odometry.OdometryMode
    MulticameraMode = Odometry.MulticameraMode
    OdometryConfig = Odometry.Config
    OdometryRGBDSettings = Odometry.RGBDSettings

    SlamConfig = Slam.Config
    SlamMetrics = Slam.Metrics
    SlamLocalizationSettings = Slam.LocalizationSettings
    SlamDataLayer = Slam.DataLayer
    PoseGraphNode = Slam.PoseGraphNode
    PoseGraphEdge = Slam.PoseGraphEdge
    PoseGraph = Slam.PoseGraph
    SlamLandmark = Slam.Landmark
    SlamLandmarks = Slam.Landmarks
    LocalizerProbe = Slam.LocalizerProbe
    LocalizerProbes = Slam.LocalizerProbes

    def __init__(self,
                 rig: Rig,
                 odom_config: Optional[Odometry.Config] = None,
                 slam_config: Optional[Slam.Config] = None) -> None:
        """
        Initialize the cuVSLAM system. Each `Odometry.OdometryMode` has specific `rig` requirements.

        Parameters:
            rig: Camera rig configuration
            odom_config: Optional odometry configuration (uses defaults if omitted)
            slam_config: Optional SLAM configuration (disables SLAM if None)
        """
        if odom_config is None:
            odom_config = self.OdometryConfig()
        # need to export observations/landmarks for Slam
        if slam_config is not None:
            odom_config.enable_observations_export = True
            odom_config.enable_landmarks_export = True
        # Create odometry
        self.odom = Odometry(rig, odom_config)
        # Create SLAM
        primary_cams = self.odom.get_primary_cameras()
        self.slam = Slam(rig, primary_cams, slam_config) if slam_config is not None else None

    def track(self,
              timestamp: int,
              images: List[ImageArray],
              masks: Optional[List[ImageArray]] = None,
              depths: Optional[List[ImageArray]] = None,
              gt_pose: Optional[Pose] = None) -> Tuple[PoseEstimate,
                                                       Optional[Pose]]:
        """
        Track a rig pose using current image frame.

        This method combines odometry and SLAM processing in a single call.

        In inertial mode, if visual odometry tracker fails to compute a pose, the function returns the position calculated from a user-provided IMU data.
        If after several calls of Track() visual odometry is not able to recover, then invalid pose will be returned.
        Odometry will output poses in the same coordinate frame until a loss of tracking.

        To get SLAM poses, SLAM must be enabled in the constructor by providing a non-null `slam_config`.
        SLAM poses may have loop closure (LC) jumps when LC is detected and pose graph is optimized.
        SLAM poses cannot be adjusted retroactively, so use `get_all_slam_poses` method to get smooth trajectory up to the latest frame.
        Also, in asynchronous mode, LC is done in a separate work thread to keep `track` call fast, so SLAM poses are not updated immediately.

        All cameras must be synchronized. If a camera rig provides "almost synchronized" frames, the timestamps should be within 1 millisecond.

        Images (masks, depth images, etc.) can be numpy arrays or tensors, both GPU (CUDA) and CPU.
        All data must be of the same type (either GPU or CPU).
        This is not the same as `odom_config.use_gpu` - if odometry uses GPU for computations,
        images etc. can still be either CPU or GPU arrays/tensors.

        The images etc. must be in the same order as cameras in the rig.
        If data for a camera is not available, pass empty array or tensor for that camera image.

        Parameters:
            timestamp: Images timestamp in nanoseconds
            images: List of numpy arrays or tensors containing the camera images
            masks: Optional list of numpy arrays or tensors containing masks for the images
            depths: Optional list of numpy arrays or tensors containing depth images
            gt_pose: Optional ground truth pose. Should be provided if `gt_align_mode` is enabled, otherwise should be None.

        Returns:
            PoseEstimate: The computed pose estimate from Odometry. On failure `world_from_rig` will be `None`.
            Pose: If SLAM is enabled, the computed pose estimate from SLAM, otherwise `None`.

        Raises:
            ValueError: If data checks fail (e.g. timestamps are out of order, images sizes are inconsistent, etc.).
        """
        pose_estimate = self.odom.track(timestamp, images, masks, depths)

        slam_pose = None
        if self.slam and pose_estimate.world_from_rig:
            state = self.odom.get_state()
            slam_pose = self.slam.track(state, gt_pose)

        return pose_estimate, slam_pose

    def register_imu_measurement(self, sensor_index: int, imu_measurement: ImuMeasurement) -> None:
        """
        Register an IMU measurement with the tracker.

        Requires Inertial odometry mode.

        If visual odometry loses camera position, it briefly continues execution
        using user-provided IMU measurements while trying to recover the position.
        :meth:`register_imu_measurement` should be called in between image acquisitions
        however many IMU measurements there are:

        - tracker.track
        - tracker.register_imu_measurement
        - ...
        - tracker.register_imu_measurement
        - tracker.track
        - tracker.register_imu_measurement
        - ...

        Higher frequency IMU measurements are recommended.

        IMU sensors and cameras clocks must be synchronized. :meth:`track` and :meth:`register_imu_measurement`
        must be called in strict ascending order of timestamps.

        Parameters:
            sensor_index: Sensor index; must be 0, as only one sensor is supported now
            imu_measurement: IMU measurement to register

        Raises:
            ValueError: If IMU fusion is disabled or if called out of the order of timestamps.
        """
        self.odom.register_imu_measurement(sensor_index, imu_measurement)

    def get_last_observations(self, camera_index: int) -> list[Observation]:
        """
        Get observations from the last frame for specified camera. See :class:`Observation`.
        Requires `enable_observations_export=True` in :class:`OdometryConfig`.
        """
        return self.odom.get_last_observations(camera_index)

    def get_last_landmarks(self) -> list[Landmark]:
        """
        Get landmarks from the last frame. See :class:`Landmark`.
        Requires `enable_landmarks_export=True` in :class:`OdometryConfig`.
        """
        return self.odom.get_last_landmarks()

    def get_last_gravity(self) -> Optional[list[float]]:
        """
        Get gravity vector from the last frame.
        Returns `None` if gravity is not yet available.
        Requires Inertial mode (`odometry_mode=OdometryMode.Inertial` in :class:`OdometryConfig`)
        """
        return self.odom.get_last_gravity()

    def get_final_landmarks(self) -> dict[int, list[float]]:
        """
        Get all final landmarks from all frames.
        Landmarks are 3D points in the odometry start frame.
        Requires `enable_final_landmarks_export=True` in :class:`OdometryConfig`.
        """
        return self.odom.get_final_landmarks()

    def get_all_slam_poses(self, max_poses_count: int = 0) -> List[PoseStamped]:
        """
        Get all SLAM poses for each frame.

        Returns all SLAM poses after optimization with the latest pose graph.
        SLAM poses from track() method will have jumps after loop closures.
        With this method, on the other hand, you will have smooth output because it recalculates past poses using the current pose graph.

        Parameters:
            max_poses_count: Maximum number of poses to return (0 for all)
        Returns:
            List of poses with timestamps
        """
        return self.slam.get_all_slam_poses(max_poses_count) if self.slam else []

    def set_slam_pose(self, pose: Pose) -> None:
        """
        Set the rig SLAM pose to a value provided by a user.
        This is useful for initializing SLAM (or resetting in the process) with a pose known from another source.
        Subsequent calls to `track` will use this pose as the starting point for SLAM.
        """
        if self.slam:
            self.slam.set_slam_pose(pose)

    def save_map(self, folder_name: str, callback: Callable[[bool], None]) -> None:
        """
        Save SLAM database (map) to a folder asynchronously.
        This folder will be created, if it does not exist.
        **WARNING**: *Contents of the folder will be overwritten.*
        This method can work asynchronously depending on the `sync_mode` parameter in `Slam.Config`.
        In both cases, the callback will be called with a flag indicating if the map was saved successfully.

        Parameters:
            folder_name: Folder name where SLAM database will be saved
            callback: Function to be called when save is complete (takes bool success parameter)
        """
        if self.slam:
            self.slam.save_map(folder_name, callback)
        else:
            callback(False)  # No SLAM instance to save the map

    def localize_in_map(self,
                        folder_name: str,
                        guess_pose: Pose,
                        images: List[ImageArray],
                        settings: Slam.LocalizationSettings,
                        callback: Callable[[Optional[Pose], str], None]) -> None:
        """
        Localize in the existing database (map) asynchronously.

        Finds the position of the camera in existing SLAM database.
        If successful, sets the SLAM pose to the found position.
        This method works asynchronously depending on the `sync_mode` parameter in `Slam.Config`.
        In both cases, the callback will be called with localization result or error message.

        Parameters:
            folder_name: Folder name which stores saved SLAM database
            guess_pose: Proposed pose where the robot might be
            images: List of numpy arrays or tensors containing the camera images
            settings: Localization settings
            callback: Function to be called when localization is complete (takes <Pose | None> result and error message parameters)
        """
        if self.slam:
            self.slam.localize_in_map(folder_name, guess_pose, images, settings, callback)
        else:
            callback(None, "SLAM is not enabled; cannot localize in map.")

    def get_slam_landmarks(self, layer: SlamDataLayer) -> Optional[SlamLandmarks]:
        """Get landmarks for a given data layer of SLAM. See :class:`SlamLandmarks`, :class:`SlamDataLayer`."""
        return self.slam.get_landmarks(layer) if self.slam else None

    def get_pose_graph(self) -> Optional[PoseGraph]:
        """Get pose graph consisting of all keyframes and their connections including loop closures. See :class:`PoseGraph`."""
        return self.slam.get_pose_graph() if self.slam else None

    def get_localizer_probes(self) -> Optional[LocalizerProbes]:
        """Get localizer probes from the most recent localization attempt. See :class:`LocalizerProbes`."""
        return self.slam.get_localizer_probes() if self.slam else None

    def get_slam_metrics(self) -> Optional[SlamMetrics]:
        """Get SLAM metrics. See :class:`SlamMetrics`"""
        return self.slam.get_slam_metrics() if self.slam else None

    def get_loop_closure_poses(self) -> Optional[List[PoseStamped]]:
        """Get list of last 10 loop closure poses with timestamps."""
        return self.slam.get_loop_closure_poses() if self.slam else None

    @staticmethod
    def merge_maps(rig: Rig, databases: List[str], output_folder: str) -> None:
        """
        Merge existing maps into one map.

        Don't use any of the input folders as `output_folder` to avoid overwriting.

        This method merges multiple maps into a single map. Maps must have the same world coordinate frame,
        e.g. using `set_slam_pose` from a ground truth source.

        This method cannot be used to merge maps from different locations,
        because pose graphs from input maps must be combined into a single graph.

        Parameters:
            rig: Camera rig configuration
            databases: Input array of folders with existing databases
            output_folder: folder to save output database
        """
        Slam.merge_maps(rig, databases, output_folder)

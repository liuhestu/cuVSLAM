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
import sys
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import depthai as dai
from scipy.spatial.transform import Rotation

import cuvslam as vslam

# Add the realsense folder to the system path to import visualizer
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../realsense"))
)

from visualizer import RerunVisualizer

# Constants
FPS = 30
RESOLUTION = (1280, 720)
#RESOLUTION = (640, 480)
WARMUP_FRAMES = 60
IMAGE_JITTER_THRESHOLD_NS = 35 * 1e6  # 35ms in nanoseconds
IMU_JITTER_THRESHOLD_NS = 6 * 1e6  # 6ms in nanoseconds
IMU_FREQUENCY = 200

# Camera border margins to exclude features near image edges
# This helps avoid using features from highly distorted regions in unrectified OAK-D images
# Features detected within these margins will not be processed
BORDER_TOP = 10
BORDER_BOTTOM = 10
BORDER_LEFT = 10
BORDER_RIGHT = 10

# Conversion factor from cm to meters
CM_TO_METERS = 100

# TODO: Replace placeholders with calibrated IMU parameters.
IMU_GYROSCOPE_NOISE_DENSITY = 6.0673370376614875e-03
IMU_GYROSCOPE_RANDOM_WALK = 3.6211951458325785e-05
IMU_ACCELEROMETER_NOISE_DENSITY = 3.3621979208052800e-02
IMU_ACCELEROMETER_RANDOM_WALK = 9.8256589971851467e-04


def oak_transform_to_pose(oak_extrinsics: List[List[float]]) -> vslam.Pose:
    """Convert 4x4 transformation matrix to cuVSLAM pose.

    Args:
        oak_extrinsics: 4x4 transformation matrix from OAK calibration

    Returns:
        vslam.Pose object
    """
    extrinsics_array = np.array(oak_extrinsics)
    rotation_matrix = extrinsics_array[:3, :3]
    translation_vector = extrinsics_array[:3, 3] / CM_TO_METERS  # Convert to meters

    rotation_quat = Rotation.from_matrix(rotation_matrix).as_quat()
    return vslam.Pose(rotation=rotation_quat, translation=translation_vector)


def set_cuvslam_camera(oak_params: Dict[str, Any]) -> vslam.Camera:
    """Create a Camera object from OAK camera parameters.

    Args:
        oak_params: Dictionary containing camera parameters

    Returns:
        vslam.Camera object
    """
    cam = vslam.Camera()
    cam.distortion = vslam.Distortion(
        vslam.Distortion.Model.Polynomial, oak_params["distortion"]
    )

    cam.focal = (
        oak_params["intrinsics"][0][0],
        oak_params["intrinsics"][1][1]
    )
    cam.principal = (
        oak_params["intrinsics"][0][2],
        oak_params["intrinsics"][1][2]
    )
    cam.size = oak_params["resolution"]
    cam.rig_from_camera = oak_transform_to_pose(oak_params["extrinsics"])

    # Features within these outer frames will be ignored by cuVSLAM
    cam.border_top = BORDER_TOP
    cam.border_bottom = BORDER_BOTTOM
    cam.border_left = BORDER_LEFT
    cam.border_right = BORDER_RIGHT

    return cam


def get_stereo_calibration(
    calib_data: dai.CalibrationHandler, resolution: Tuple[int, int]
) -> Dict[str, Dict[str, Any]]:
    """Get calibration data from the OAK-D calibration handler.

    Args:
        calib_data: Calibration handler from the OAK-D device
        resolution: Camera resolution as (width, height)

    Returns:
        Dictionary containing stereo calibration parameters
    """
    stereo_camera = {"left": {}, "right": {}}

    # Set image size
    stereo_camera["left"]["resolution"] = resolution
    stereo_camera["right"]["resolution"] = resolution

    # Get intrinsics for left and right cameras (scaled to the requested resolution)
    stereo_camera["left"]["intrinsics"] = calib_data.getCameraIntrinsics(
        dai.CameraBoardSocket.CAM_B, resolution[0], resolution[1]
    )
    stereo_camera["right"]["intrinsics"] = calib_data.getCameraIntrinsics(
        dai.CameraBoardSocket.CAM_C, resolution[0], resolution[1]
    )

    # Get extrinsics (transformation of left and right cameras relative to center RGB camera)
    stereo_camera["left"]["extrinsics"] = calib_data.getCameraExtrinsics(
        dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_A
    )
    stereo_camera["right"]["extrinsics"] = calib_data.getCameraExtrinsics(
        dai.CameraBoardSocket.CAM_C, dai.CameraBoardSocket.CAM_A
    )

    # Get distortion coefficients for left and right cameras (first 8 coefficients)
    stereo_camera["left"]["distortion"] = calib_data.getDistortionCoefficients(
        dai.CameraBoardSocket.CAM_B
    )[:8]
    stereo_camera["right"]["distortion"] = calib_data.getDistortionCoefficients(
        dai.CameraBoardSocket.CAM_C
    )[:8]

    return stereo_camera


def get_imu_calibration(imu_extrinsics: List[List[float]]) -> vslam.ImuCalibration:
    """Create an IMU calibration object with calibrated parameters.

    Args:
        imu_extrinsics: 4x4 transformation matrix from IMU to center RGB camera (the Rig origin).

    Returns:
        vslam.ImuCalibration object
    """
    imu = vslam.ImuCalibration()
    
    # DepthAI firmware already rotates the IMU readings (ACCELEROMETER_RAW) into the 
    # camera's coordinate frame (X-Right, Y-Down, Z-Forward). However, the extrinsics 
    # matrix read from EEPROM contains the physical chip rotation (often swapped X/Y).
    # Applying it twice would ruin the mapping. So we force Identity rotation, but 
    # keep the translation offset (converted from cm to meters).
    extrinsics_array = np.array(imu_extrinsics)
    translation_vector = extrinsics_array[:3, 3] / CM_TO_METERS
    
    imu.rig_from_imu = vslam.Pose(
        rotation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64), # Identity Quat
        translation=translation_vector
    )
    
    imu.gyroscope_noise_density = IMU_GYROSCOPE_NOISE_DENSITY
    imu.gyroscope_random_walk = IMU_GYROSCOPE_RANDOM_WALK
    imu.accelerometer_noise_density = IMU_ACCELEROMETER_NOISE_DENSITY
    imu.accelerometer_random_walk = IMU_ACCELEROMETER_RANDOM_WALK
    imu.frequency = IMU_FREQUENCY
    return imu


def _timestamp_to_ns(timestamp: Any) -> Optional[int]:
    """Best-effort conversion for DepthAI timestamps to nanoseconds."""
    if timestamp is None:
        return None

    if hasattr(timestamp, "total_seconds"):
        return int(timestamp.total_seconds() * 1e9)

    if hasattr(timestamp, "get"):
        ts = timestamp.get()
        if hasattr(ts, "total_seconds"):
            return int(ts.total_seconds() * 1e9)

    return None


def _imu_packet_timestamp_ns(packet: Any) -> Optional[int]:
    """Extract an IMU packet timestamp across DepthAI API variants."""
    # Prefer exact method calls for device timestamps
    for getter in ("getTimestampDevice", "getTimestamp", "getTimestampHost"):
        if hasattr(packet, getter):
            ts = _timestamp_to_ns(getattr(packet, getter)())
            if ts is not None:
                return ts

    # Fallback to direct field access
    for field_name in ("acceleroMeter", "gyroscope"):
        if hasattr(packet, field_name):
            field = getattr(packet, field_name)
            for attr in ("tsDevice", "timestamp", "timestampDevice"):
                if hasattr(field, attr):
                    ts = _timestamp_to_ns(getattr(field, attr))
                    if ts is not None:
                        return ts

    return None


def _message_group_timestamp_ns(message_group: Any) -> Optional[int]:
    """Extract message group timestamp with preference for device time."""
    for getter in ("getTimestampDevice", "getTimestamp", "getTimestampHost"):
        if hasattr(message_group, getter):
            ts = _timestamp_to_ns(getattr(message_group, getter)())
            if ts is not None:
                return ts
    return None


def _get_imu_messages(imu_queue: Any) -> List[Any]:
    """Collect IMU messages without blocking the main loop."""
    try:
        return imu_queue.tryGetAll()
    except AttributeError:
        message = imu_queue.tryGet()
        if message is None:
            return []
        return [message]


def main() -> None:
    """Main function for OAK-D VIO tracking."""
    # Create device and read calibration before pipeline creation
    device = dai.Device()
    calib_data = device.readCalibration()
    stereo_camera = get_stereo_calibration(calib_data, RESOLUTION)

    cameras = [
        set_cuvslam_camera(stereo_camera["left"]),
        set_cuvslam_camera(stereo_camera["right"])
    ]

    rig = vslam.Rig(cameras)
    imu_extrinsics = calib_data.getImuToCameraExtrinsics(dai.CameraBoardSocket.CAM_A)
    rig.imus = [get_imu_calibration(imu_extrinsics)]

    # Create rig and tracker
    cfg = vslam.Tracker.OdometryConfig(
        async_sba=False,
        enable_final_landmarks_export=True,
        enable_observations_export=True,
        rectified_stereo_camera=False,
        odometry_mode=vslam.Tracker.OdometryMode.Inertial
    )
    
    # Configure SLAM backend to enable Loop Closure
    slam_cfg = vslam.Tracker.SlamConfig()
    # You can tweak cell size, max_map_size depending on your needs.
    
    tracker = vslam.Tracker(rig, cfg, slam_config=slam_cfg)

    # Create pipeline with the device
    pipeline = dai.Pipeline(device)

    # Create stereo pair using new Camera node API
    left_camera = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_B, sensorFps=FPS
    )
    right_camera = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_C, sensorFps=FPS
    )

    # Use Sync node to synchronize stereo frames
    sync = pipeline.create(dai.node.Sync)
    sync.setSyncThreshold(timedelta(seconds=0.5 / FPS))

    # Request grayscale outputs at specified resolution
    left_output = left_camera.requestOutput(RESOLUTION, type=dai.ImgFrame.Type.GRAY8)
    right_output = right_camera.requestOutput(RESOLUTION, type=dai.ImgFrame.Type.GRAY8)

    # Link camera outputs to sync node
    left_output.link(sync.inputs["left"])
    right_output.link(sync.inputs["right"])

    # Create output queue from sync node
    sync_queue = sync.out.createOutputQueue()

    # Create IMU node
    imu_node = pipeline.create(dai.node.IMU)
    imu_node.enableIMUSensor(
        [
            dai.IMUSensor.ACCELEROMETER_RAW,
            dai.IMUSensor.GYROSCOPE_RAW
        ],
        IMU_FREQUENCY
    )
    imu_node.setBatchReportThreshold(1)
    imu_node.setMaxBatchReports(10)
    imu_queue = imu_node.out.createOutputQueue()

    # Initialize visualization and tracking variables
    enable_rerun = os.environ.get("CUVSLAM_NO_RERUN") is None
    visualizer = RerunVisualizer() if enable_rerun else None
    frame_id = 0
    prev_camera_timestamp: Optional[int] = None
    trajectory: List[np.ndarray] = []
    last_imu_timestamp: Optional[int] = None
    imu_count_since_last_camera = 0
    last_sent_timestamp: Optional[int] = None
    imu_buffer: List[Tuple[int, Any, Any]] = []

    # Open TUM trajectory file to append poses
    tum_file = open("oak_vio_tum.txt", "w")

    # Start the pipeline
    pipeline.start()

    # Capture and process stereo frames
    while pipeline.isRunning():
        message_group: dai.MessageGroup = sync_queue.get()
        left_frame = message_group["left"]
        right_frame = message_group["right"]

        # Get synchronized timestamp from message group (convert timedelta to ns)
        timestamp_ns = _message_group_timestamp_ns(message_group)
        if timestamp_ns is None:
            continue

        # Drain IMU queue after getting camera timestamp; buffer for ordering
        imu_messages = _get_imu_messages(imu_queue)
        for imu_message in imu_messages:
            for packet in imu_message.packets:
                accel = packet.acceleroMeter
                gyro = packet.gyroscope
                imu_ts = _imu_packet_timestamp_ns(packet)
                if imu_ts is None:
                    continue
                imu_buffer.append((imu_ts, accel, gyro))

        if imu_buffer:
            imu_buffer.sort(key=lambda item: item[0])

        # Register IMU samples up to current camera timestamp
        while imu_buffer and imu_buffer[0][0] <= timestamp_ns:
            imu_ts, accel, gyro = imu_buffer.pop(0)
            if last_sent_timestamp is not None and imu_ts <= last_sent_timestamp:
                continue

            if last_imu_timestamp is not None:
                imu_diff = imu_ts - last_imu_timestamp
                if imu_diff <= 0:
                    continue
                if imu_diff > IMU_JITTER_THRESHOLD_NS:
                    '''
                    print(
                        "Warning: IMU message gap "
                        f"({imu_diff/1e6:.2f} ms) exceeds threshold "
                        f"{IMU_JITTER_THRESHOLD_NS/1e6:.2f} ms"
                    )'''

            last_imu_timestamp = imu_ts

            imu_measurement = vslam.ImuMeasurement()
            imu_measurement.timestamp_ns = imu_ts
            imu_measurement.linear_accelerations = np.array(
                [accel.x, accel.y, accel.z], dtype=np.float32
            )
            imu_measurement.angular_velocities = np.array(
                [gyro.x, gyro.y, gyro.z], dtype=np.float32
            )
            tracker.register_imu_measurement(0, imu_measurement)
            imu_count_since_last_camera += 1
            last_sent_timestamp = imu_ts

        # Check timestamp difference with previous frame
        if prev_camera_timestamp is not None:
            timestamp_diff = timestamp_ns - prev_camera_timestamp
            if timestamp_diff > IMAGE_JITTER_THRESHOLD_NS:
                print(
                    "Warning: Camera stream message drop: timestamp gap "
                    f"({timestamp_diff/1e6:.2f} ms) exceeds threshold "
                    f"{IMAGE_JITTER_THRESHOLD_NS/1e6:.2f} ms"
                )

        if prev_camera_timestamp is not None and imu_count_since_last_camera == 0:
            print(
                "Warning: No IMU measurements between timestamps "
                f"{prev_camera_timestamp} and {timestamp_ns}"
            )

        prev_camera_timestamp = timestamp_ns
        imu_count_since_last_camera = 0

        frame_id += 1

        # Warmup for specified number of frames
        if frame_id > WARMUP_FRAMES:
            left_img = left_frame.getCvFrame()
            right_img = right_frame.getCvFrame()

            # Track frame
            odom_pose_estimate, slam_pose_estimate = tracker.track(timestamp_ns, (left_img, right_img))
            
            odom_pose = odom_pose_estimate.world_from_rig
            
            # Prefer SLAM pose (which includes loop closures) if available
            active_pose = slam_pose_estimate if slam_pose_estimate is not None else odom_pose
            
            if active_pose is None:
                print(f"Tracking failed at frame {frame_id}")
                continue

            # Save to TUM trajectory format (timestamp tx ty tz qx qy qz qw)
            t_s = timestamp_ns / 1e9
            t = active_pose.translation
            q = active_pose.rotation
            tum_file.write(f"{t_s:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")
            tum_file.flush()

            trajectory.append(active_pose.translation)

            gravity = tracker.get_last_gravity()
            last_sent_timestamp = timestamp_ns
            
            #if left_img is not None:
            #    print(f"Frame {frame_id}, image shape: {left_img.shape}, timestamp: {timestamp_ns}")

            # Visualize results
            if visualizer is not None:
                # Scale down gravity vector visualization so it's not a 9.8m long giant arrow
                #vis_gravity = list(np.array(gravity) * 0.1) if gravity is not None else None
                visualizer.visualize_frame(
                    frame_id=frame_id,
                    images=[left_img],
                    pose=active_pose,
                    observations_main_cam=[tracker.get_last_observations(0)],
                    trajectory=trajectory,
                    timestamp=timestamp_ns,
                    #gravity=vis_gravity
                )


if __name__ == "__main__":
    main()

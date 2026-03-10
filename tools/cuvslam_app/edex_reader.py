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

import io
import json
import os
import re
import numpy as np
from PIL import Image
import tarfile
from typing import List, Optional
import cuvslam as vslam

import conversions as conv
from dataset_reader import DatasetReader, Processing


class EdexReader(DatasetReader):
    def __init__(self, edex_dir: str, stereo_edex: Optional[str] = None, num_loops: int = 0, rgbd_mode: bool = False):
        super().__init__(edex_dir, stereo_edex, num_loops)
        self.rgbd_mode = rgbd_mode
        self.rgbd_settings = None
        self.depth_sequence = None

        with open(os.path.join(edex_dir, 'stereo.edex'), 'r') as f:
            # Load configuration JSON
            config_data = json.load(f)
            rig = self.parse_config(config_data)

            # Parse RGBD settings if in RGBD mode
            if self.rgbd_mode:
                try:
                    self.rgbd_settings = self._parse_rgbd_settings(config_data)
                except ValueError as e:
                    raise ValueError(f"Failed to initialize RGBD mode: {str(e)}") from e

            # Load IMU data if available
            imu_data_list = []
            imu_file = None

            # If IMUs are configured in the rig, measurements file MUST be present
            if rig.imus:
                # Validate config structure
                if len(config_data) == 0 or not isinstance(config_data[0], dict):
                    raise ValueError("Invalid stereo.edex format: rig configuration section is missing or invalid")

                # Check for IMU configuration
                if 'imu' not in config_data[0]:
                    raise ValueError(
                        "IMU is configured in the rig but 'imu' section is missing in stereo.edex. "
                        "Please add an 'imu' section with 'measurements' field."
                    )

                if not isinstance(config_data[0]['imu'], dict):
                    raise ValueError("Invalid stereo.edex format: 'imu' section must be a dictionary")

                if 'measurements' not in config_data[0]['imu']:
                    raise ValueError(
                        "IMU is configured in the rig but 'measurements' file is not specified in stereo.edex. "
                        "Please add 'measurements' field in the 'imu' section pointing to the IMU data file."
                    )

                imu_file = config_data[0]['imu']['measurements']

                # Validate the measurements file exists
                imu_file_path = os.path.join(edex_dir, imu_file)
                if not os.path.exists(imu_file_path):
                    raise ValueError(
                        f"IMU measurements file not found: {imu_file_path}. "
                        f"Please ensure the file exists or update the path in stereo.edex."
                    )

                imu_data_list = self.load_imu_data(imu_file_path)

            # Load frame metadata or use sequence
            metadata = config_data[1]
            if not isinstance(metadata, dict):
                raise ValueError("Invalid stereo.edex format: metadata section must be a dictionary")

            self.fps = metadata.get('fps', 30)
            self.interval_ns = int(1e9 / self.fps)

            # Load depth sequence if in RGBD mode
            if self.rgbd_mode and 'depth_sequence' in metadata:
                self.depth_sequence = metadata['depth_sequence']

            self.frames = {}
            if 'frame_metadata' in metadata:
                frame_metadata_file = metadata['frame_metadata']
                self.frames = self.load_frame_metadata_edex(
                    os.path.join(edex_dir, frame_metadata_file))
            elif 'sequence' in metadata:
                sequence = metadata['sequence']
                self.frame_id_end, self.frame_id_start = config_data[0]['frame_end'], config_data[0]['frame_start']
                frame_id = 0
                for i in range(self.frame_id_start, self.frame_id_end + 1):
                    self.frames[frame_id] = {"cams": [], "cams_max_ts": 0}
                    for cam_id, image_paths in enumerate(sequence):
                        ts = self.interval_ns * frame_id
                        self.frames[frame_id]["cams"].append({
                            'id': cam_id,
                            'filename': self.replace_last_digits(image_paths[0], i),
                            'timestamp': ts
                        })
                        self.frames[frame_id]["cams_max_ts"] = ts
                        self.frames[frame_id]["cams_min_ts"] = ts

                    # Add depth data if in RGBD mode
                    if self.rgbd_mode and self.depth_sequence:
                        self.frames[frame_id]["depth"] = []
                        for depth_id, depth_paths in enumerate(self.depth_sequence):
                            self.frames[frame_id]["depth"].append({
                                'id': depth_id,
                                'filename': self.replace_last_digits(depth_paths[0], i),
                                'timestamp': ts
                            })

                    frame_id += 1
            else:
                raise ValueError(
                    "Missing 'frame_metadata' or 'sequence' in stereo.edex")

            self.total_frames = len(self.frames)
            self.current_frame = min(self.frames.keys())
            self.first_ts = min(self.frames.values(), key=lambda x: x["cams_max_ts"])["cams_max_ts"]
            self.last_ts = max(self.frames.values(), key=lambda x: x["cams_max_ts"])["cams_max_ts"]
            self.max_ts = max(self.last_ts, self.first_ts)
            self.sequence_duration = self.last_ts - self.first_ts

            self.rig = rig
            self.merge_and_sort_frames_and_imu(imu_data_list)
            self.edex_dir = edex_dir

            # Check for tar archives
            self.tar_archives = {}
            self._detect_tar_archives()

    def _detect_tar_archives(self):
        """Detect and cache tar archives for image folders.

        Maps folder names from stereo.edex (e.g., "00", "01") to tar archives (e.g., "00.tar", "01.tar").
        """
        # Get unique folder names from image paths
        folder_names = set()
        for frame_data in self.frames.values():
            for cam_data in frame_data["cams"]:
                folder_name = os.path.dirname(cam_data['filename'])
                if folder_name:
                    folder_names.add(folder_name)

            # Also check depth folders if in RGBD mode
            if self.rgbd_mode and "depth" in frame_data:
                for depth_data in frame_data["depth"]:
                    folder_name = os.path.dirname(depth_data['filename'])
                    if folder_name:
                        folder_names.add(folder_name)

        # Check if tar archives exist for each folder
        for folder_name in folder_names:
            tar_path = os.path.join(self.edex_dir, f"{folder_name}.tar")
            if os.path.exists(tar_path):
                try:
                    tar_file = tarfile.open(tar_path, 'r')
                    self.tar_archives[folder_name] = tar_file
                    print(f"Found tar archive for folder '{folder_name}': {folder_name}.tar")
                except Exception as e:
                    print(f"Warning: Could not open tar archive {tar_path}: {e}")

    def merge_and_sort_frames_and_imu(self, imu_data_list: List[dict]):
        """Merge and sort IMU and image data by timestamp."""
        # Sort IMU data by timestamp
        imu_data_list.sort(key=lambda x: x['timestamp'])

        imu_idx = 0
        total_imu = len(imu_data_list)

        for frame_id in sorted(self.frames.keys()):
            current_frame = self.frames[frame_id]
            next_frame = self.frames.get(frame_id + 1)
            current_frame['imu_data'] = []

            # Get time window for this frame
            frame_start_ts = current_frame['cams_min_ts']
            frame_end_ts = next_frame['cams_min_ts'] if next_frame else float('inf')

            # Process IMU data until we reach next frame's timestamp
            while imu_idx < total_imu:
                imu_data = imu_data_list[imu_idx]
                imu_ts = imu_data['timestamp']

                if imu_ts < frame_start_ts:
                    imu_idx += 1
                    continue

                if imu_ts >= frame_end_ts:
                    break

                current_frame['imu_data'].append(imu_data)
                imu_idx += 1

    @staticmethod
    def normalize_timestamp_to_ns(timestamp: float, is_start_zero: bool = False) -> int:
        """
        Normalize timestamp to integer nanoseconds.

        Detects if timestamp is in seconds or nanoseconds
        and converts to nanoseconds as an integer.

        Args:
            timestamp: Timestamp in seconds or nanoseconds (can be float)
            is_start_zero: If True, timestamps start from zero. For int - return as-is, for float - multiply by 10^9

        Returns:
            Timestamp in nanoseconds as integer
        """
        # If timestamps start from zero (relative timestamps)
        if is_start_zero:
            if isinstance(timestamp, int):
                return timestamp
            else:
                return int(timestamp * 1_000_000_000)

        # Handle the case where timestamp is already an int
        if isinstance(timestamp, int):
            # If already in nanoseconds range, return as-is
            if timestamp >= 1e13:
                return timestamp
            # If in seconds range, convert to nanoseconds
            else:
                return timestamp * 1_000_000_000

        # Handle float timestamps
        # Heuristic:
        # - timestamp < 1e10: likely seconds (covers dates up to year ~2286)
        # - timestamp >= 1e13: likely nanoseconds

        if timestamp < 1e10:
            # Likely in seconds
            return int(timestamp * 1_000_000_000)
        else:
            # Likely in nanoseconds
            return int(timestamp)

    @staticmethod
    def load_image(image_path: str) -> np.ndarray:

        # Try to replace .png with .tga for faster loading if exists
        if image_path.endswith('.png'):
            tga_path = image_path + '.tga'
            if os.path.exists(tga_path):
                image_path = tga_path

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = Image.open(image_path)

        image_tensor = np.array(image)

        if image.mode == 'L':
            # mono
            if len(image_tensor.shape) != 2:
                raise ValueError(
                    "Expected mono image to have 2 dimensions [H W].")
        elif image.mode == 'RGB':
            # rgb
            if len(image_tensor.shape) != 3 or image_tensor.shape[2] != 3:
                raise ValueError(
                    "Expected rgb image to have 3 dimensions with 3 channels [H W C].")
        else:
            raise ValueError(f"Unsupported image mode: {image.mode}")

        return image_tensor

    def load_image_from_tar(self, tar_file: tarfile.TarFile, image_path: str) -> np.ndarray:
        """Load image from tar archive."""

        # Extract filename from path for tar lookup
        filename = os.path.basename(image_path)

        # print(f"Looking for image '{filename}' in tar archive")

        # Try to replace .png with .tga for faster loading if exists in tar
        if filename.endswith('.png'):
            tga_filename = filename + '.tga'
            try:
                member = tar_file.getmember(tga_filename)
                file_obj = tar_file.extractfile(member)
                if file_obj is not None:
                    image = Image.open(file_obj)
                    image_tensor = np.array(image)
                    file_obj.close()
                    print(f"Found TGA version: {tga_filename}")
                    return image_tensor
            except KeyError:
                pass  # tga file not found, try original filename

        try:
            member = tar_file.getmember(filename)
            file_obj = tar_file.extractfile(member)
            if file_obj is None:
                raise FileNotFoundError(f"Could not extract {filename} from tar archive")

            image = Image.open(file_obj)
            image_tensor = np.array(image)
            file_obj.close()
            #print(f"Found image: {filename}")

            if image.mode == 'L':
                # mono8
                if len(image_tensor.shape) != 2:
                    raise ValueError(
                        "Expected mono8 image to have 2 dimensions [H W].")
            elif image.mode == 'RGB':
                # rgb8
                if len(image_tensor.shape) != 3 or image_tensor.shape[2] != 3:
                    raise ValueError(
                        "Expected rgb8 image to have 3 dimensions with 3 channels [H W C].")
            else:
                raise ValueError(f"Unsupported image mode: {image.mode}")

            return image_tensor
        except KeyError:
            raise FileNotFoundError(f"Image file {filename} not found in tar archive")

        # except KeyError:
        #     # List available files in tar for debugging
        #     available_files = tar_file.getnames()
        #     print(f"Available files in tar: {available_files[:10]}...")  # Show first 10 files
        #     raise FileNotFoundError(f"Image file {filename} not found in tar archive")

    def _parse_rgbd_settings(self, config_data):
        """Parse RGBD settings from stereo.edex config.

        Args:
            config_data: Configuration data loaded from stereo.edex

        Returns:
            OdometryRGBDSettings object

        Raises:
            ValueError: If config_data structure is invalid or required fields are missing
        """
        try:
            # Validate config_data structure
            if not isinstance(config_data, list):
                raise ValueError("config_data must be a list")

            if len(config_data) < 2:
                raise ValueError("config_data must have at least 2 elements (rig config and metadata)")

            if not isinstance(config_data[0], dict):
                raise ValueError("config_data[0] (rig configuration) must be a dictionary")

            if not isinstance(config_data[1], dict):
                raise ValueError("config_data[1] (metadata) must be a dictionary")

            # Get depth_id from camera config
            depth_camera_id = None
            depth_scale_factor = 1.0

            # Validate and parse camera configuration
            if 'cameras' not in config_data[0]:
                raise ValueError("'cameras' key not found in rig configuration (config_data[0])")

            cameras = config_data[0]['cameras']
            if not isinstance(cameras, list):
                raise ValueError("'cameras' must be a list")

            if len(cameras) == 0:
                raise ValueError("'cameras' list is empty")

            # Search for depth_id in camera configs
            for cam in cameras:
                if not isinstance(cam, dict):
                    continue  # Skip invalid camera entries

                if 'depth_scale_factor' in cam:
                    try:
                        depth_scale_factor = float(cam['depth_scale_factor'])
                    except (ValueError, TypeError):
                        print(f"Warning: Invalid depth_scale_factor value, using default 1.0")
                        depth_scale_factor = 1.0

                if 'depth_id' in cam:
                    try:
                        depth_camera_id = int(cam['depth_id'])
                        break
                    except (ValueError, TypeError):
                        print(f"Warning: Invalid depth_id value: {cam['depth_id']}")
                        continue

            # Check for depth_sequence and update scale factor if needed
            if 'depth_sequence' in config_data[1]:
                depth_sequence = config_data[1]['depth_sequence']

                if isinstance(depth_sequence, list):
                    for depth_id, depth_paths in enumerate(depth_sequence):
                        # Validate depth_paths is a list with at least one element
                        if isinstance(depth_paths, list) and len(depth_paths) > 0:
                            first_path = depth_paths[0]
                            if isinstance(first_path, str) and first_path.endswith('.npy'):
                                depth_scale_factor = 1000.0
                                break

            # Validate that depth_camera_id was found
            if depth_camera_id is None:
                raise ValueError(
                    "RGBD mode is enabled but 'depth_id' not found in camera config in stereo.edex. "
                    "Please add 'depth_id' field to one of the cameras in the configuration."
                )

            # Get enable_depth_stereo_tracking if provided
            enable_depth_stereo_tracking = config_data[0].get('enable_depth_stereo_tracking', False)

            # Create RGBDSettings
            rgbd_settings = vslam.Tracker.OdometryRGBDSettings()
            rgbd_settings.depth_camera_id = depth_camera_id
            rgbd_settings.depth_scale_factor = depth_scale_factor
            rgbd_settings.enable_depth_stereo_tracking = enable_depth_stereo_tracking

            print(f"RGBD settings parsed: depth_camera_id={depth_camera_id}, "
                  f"depth_scale_factor={depth_scale_factor}, "
                  f"enable_depth_stereo_tracking={enable_depth_stereo_tracking}")

            return rgbd_settings

        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(
                f"Failed to parse RGBD settings from stereo.edex: {str(e)}. "
                "Please ensure the configuration file has the correct structure with 'cameras' "
                "and 'depth_id' fields."
            ) from e

    @staticmethod
    def _convert_depth_to_uint16(depth: np.ndarray) -> np.ndarray:
        """Convert depth array to uint16 format.

        Args:
            depth: Input depth array (float32/float64/uint16)

        Returns:
            uint16 depth array
        """
        if depth.dtype == np.float32 or depth.dtype == np.float64:
            # Convert float (meters) to uint16 using scale factor
            # uint16_value = float_value * 1000
            return np.round(depth * 1000, 2).astype(np.uint16)
        elif depth.dtype == np.uint16:
            # Already uint16, use as is
            return depth.astype(np.uint16)
        else:
            raise ValueError(f"Unsupported dtype for depth: {depth.dtype}. Expected float32, float64, or uint16")

    @staticmethod
    def _load_and_process_depth(file_obj, filename: str) -> np.ndarray:
        """Load depth data from file object and process to uint16.

        Args:
            file_obj: File-like object or path string
            filename: Filename to determine format

        Returns:
            Processed uint16 depth image
        """
        if filename.endswith('.npy'):
            # Load numpy array (expected to be float32 in meters or uint16)
            depth = np.load(file_obj)
            depth = EdexReader._convert_depth_to_uint16(depth)
        elif filename.endswith('.png'):
            # Load PNG image (already uint16)
            depth_img = Image.open(file_obj)
            depth = np.array(depth_img, dtype=np.uint16)
        else:
            raise ValueError(f"Unsupported depth file format: {filename}")

        # Ensure depth is 2D
        if len(depth.shape) != 2:
            raise ValueError(f"Expected depth image to have 2 dimensions [H W], got {depth.shape}")

        return depth

    @staticmethod
    def load_depth_image(depth_path: str) -> np.ndarray:
        """Load depth image from file (.npy or .png).

        Returns uint16 depth image. If input is float32 .npy, it will be converted to uint16.
        """
        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"Depth file not found: {depth_path}")

        return EdexReader._load_and_process_depth(depth_path, depth_path)

    def load_depth_image_from_tar(self, tar_file: tarfile.TarFile, depth_path: str) -> np.ndarray:
        """Load depth image from tar archive.

        Returns uint16 depth image. If input is float32 .npy, it will be converted to uint16.
        """
        filename = os.path.basename(depth_path)

        try:
            member = tar_file.getmember(filename)
            file_obj = tar_file.extractfile(member)
            if file_obj is None:
                raise FileNotFoundError(f"Could not extract {filename} from tar archive")

            # Wrap in BytesIO for .npy files to allow seeking
            if filename.endswith('.npy'):
                file_obj = io.BytesIO(file_obj.read())

            depth = self._load_and_process_depth(file_obj, filename)
            file_obj.close()

            return depth
        except KeyError:
            raise FileNotFoundError(f"Depth file {filename} not found in tar archive")

    @staticmethod
    def load_imu_data(imu_file: str) -> List[dict]:
        imu_data = []
        if os.path.exists(imu_file):
            is_start_zero = False
            first_timestamp = None

            # First pass: read first timestamp to determine if it starts from zero
            with open(imu_file, 'r') as f:
                first_line = f.readline()
                if first_line:
                    first_data = json.loads(first_line)
                    if 'timestamp' in first_data:
                        first_timestamp = first_data['timestamp']
                        # Check if timestamp starts from zero (small value)
                        # If timestamp < 1000, it's likely starting from zero
                        if isinstance(first_timestamp, (int, float)) and abs(first_timestamp) < 1000:
                            is_start_zero = True

            # Second pass: load all data with the determined flag
            with open(imu_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    # Normalize timestamp to integer nanoseconds
                    if 'timestamp' in data:
                        data['timestamp'] = EdexReader.normalize_timestamp_to_ns(data['timestamp'], is_start_zero)
                    imu_data.append(data)
        return imu_data

    @staticmethod
    def load_frame_metadata_edex(frame_metadata_file: str) -> dict:
        frames = {}
        if os.path.exists(frame_metadata_file):
            with open(frame_metadata_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    max_ts = max(data["cams"], key=lambda x: x['timestamp'])[
                        'timestamp']
                    min_ts = min(data["cams"], key=lambda x: x['timestamp'])[
                        'timestamp']
                    if max_ts - min_ts > 1e6:
                        raise ValueError(
                            f'Frame timestamps are not synchronized within 1ms. ({min_ts} : {max_ts})')
                    frame_info = {"cams": data["cams"], "cams_max_ts": max_ts, "cams_min_ts": min_ts}

                    # Include depth data if present
                    if "depth" in data:
                        frame_info["depth"] = data["depth"]

                    frames[data["frame_id"]] = frame_info
        return frames

    @staticmethod
    def replace_last_digits(input_str: str, new_value: int) -> str:
        """
        Replace the last sequence of digits in the string with a padded integer.

        Args:
            input_str (str): The input string containing digits.
            new_value (int): The new integer value to replace the digits.

        Returns:
            str: The string with the last sequence of digits replaced.
        """
        match = re.search(r"(\d+)(?=\D*$)", input_str)
        if not match:
            return input_str

        old_digits = match.group(1)
        padding_length = len(old_digits)
        new_digits = f"{new_value:0{padding_length}d}"
        return re.sub(r"(\d+)(?=\D*$)", new_digits, input_str)

    def replay(self, processor: Processing):
        assert self.rig is not None
        # Iterate over events sorted by timestamp
        images = [np.array([])] * len(self.rig.cameras)
        masks = [np.array([])] * len(self.rig.cameras)
        depths = [np.array([])] * len(self.rig.cameras) if self.rgbd_mode else None
        timestamps = [0] * len(self.rig.cameras)
        frame_id = self.current_frame
        while self.check_end_of_sequence():
            frame_data = self.frames[self.current_frame]
            for cam_data in frame_data["cams"]:
                cam_id = cam_data['id']
                assert cam_id < len(self.rig.cameras)
                # Check image size, for ndarrays it's (height, width)
                expected_size = (
                    self.rig.cameras[cam_id].size[1], self.rig.cameras[cam_id].size[0])

                image_path = os.path.join(self.edex_dir, cam_data['filename'])

                # Check if image should be loaded from tar archive
                folder_name = os.path.dirname(cam_data['filename'])
                if folder_name and folder_name in self.tar_archives:
                    # Load from tar archive
                    # print(f"Loading from tar: {cam_data['filename']}")
                    image = self.load_image_from_tar(self.tar_archives[folder_name], cam_data['filename'])
                else:
                    # Load from regular file system
                    # print(f"Loading from filesystem: {image_path}")
                    image = self.load_image(image_path)

                image_size = image.shape[:2]
                if image_size != expected_size:
                    raise ValueError(
                        f"Image size mismatch for {image_path}. Expected {expected_size}, got {image_size}.")
                images[cam_id] = image

                basename, ext = os.path.splitext(cam_data['filename'])
                mask_file_name = f"{basename}_mask{ext}"

                # Check if mask should be loaded from tar archive
                if folder_name and folder_name in self.tar_archives:
                    # Load mask from tar archive
                    try:
                        mask = self.load_image_from_tar(self.tar_archives[folder_name], mask_file_name)
                        masks[cam_id] = mask
                    except FileNotFoundError:
                        # Mask not found in tar, which is fine
                        pass
                else:
                    # Load mask from regular file system
                    mask_path = os.path.join(self.edex_dir, mask_file_name)
                    if os.path.exists(mask_path):
                        mask = self.load_image(mask_path)
                        masks[cam_id] = mask

                timestamps[cam_id] = self.adjust_timestamp(cam_data['timestamp'])

            # Load depth images if in RGBD mode
            if self.rgbd_mode and "depth" in frame_data:
                depth_scale_factor = self.rgbd_settings.depth_scale_factor if self.rgbd_settings else 1.0
                for depth_data in frame_data["depth"]:
                    depth_id = depth_data['id']
                    depth_path = os.path.join(self.edex_dir, depth_data['filename'])

                    # Check if depth should be loaded from tar archive
                    folder_name = os.path.dirname(depth_data['filename'])
                    if folder_name and folder_name in self.tar_archives:
                        # Load from tar archive
                        depth = self.load_depth_image_from_tar(self.tar_archives[folder_name],
                                                               depth_data['filename'])
                    else:
                        # Load from regular file system
                        depth = self.load_depth_image(depth_path)

                    # Store depth at the camera index corresponding to depth_id
                    if self.rgbd_settings and depth_id == self.rgbd_settings.depth_camera_id:
                        depths[self.rgbd_settings.depth_camera_id] = depth

            processor.process_images(frame_id, timestamps, images, masks, depths)

            if frame_data["imu_data"]:
                imu_data = frame_data["imu_data"]
                for imu_measurement in imu_data:
                    # IMU data is not supported for shuttle mode, no need to adjust timestamp
                    # Normalize timestamp to integer nanoseconds (handles float/int and various scales)
                    timestamp = self.normalize_timestamp_to_ns(imu_measurement['timestamp'])
                    # IMU data is stored in CUVSLAM coordinate system, convert it to Opencv coordinate system
                    linear_accelerations = [
                        imu_measurement['LinearAccelerationX'],
                        -imu_measurement['LinearAccelerationY'],
                        -imu_measurement['LinearAccelerationZ']
                    ]
                    angular_velocities = [
                        imu_measurement['AngularVelocityX'],
                        -imu_measurement['AngularVelocityY'],
                        -imu_measurement['AngularVelocityZ']
                    ]
                    processor.process_imu(timestamp, linear_accelerations, angular_velocities)

            # Store pose from first forward run as ground truth
            if self.gt_from_shuttle and self.replay_forward and self.current_loop == 0:
                if hasattr(processor, 'get_camera_pose'):
                    pose = processor.get_camera_pose(frame_id)
                    if pose is not None:
                        self.gt_transforms.append(conv.pose_to_transform(pose))
                    else:
                        raise ValueError(f"Camera pose not found for frame {frame_id}")

            # Store metadata about when this frame was processed
            if hasattr(processor, 'set_frame_metadata'):
                processor.set_frame_metadata(frame_id, {
                    'loop': self.current_loop,
                    'forward': self.replay_forward
                })

            # Update frame counter based on direction
            self.current_frame = self.current_frame + (1 if self.replay_forward else -1)
            frame_id += 1

    def __del__(self):
        """Clean up tar archives when object is destroyed."""
        if hasattr(self, 'tar_archives'):
            for tar_file in self.tar_archives.values():
                try:
                    tar_file.close()
                except:
                    pass  # Ignore errors during cleanup

#!/usr/bin/env python3

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
Example usage:
    pip install rerun-sdk
    python3 cuvslam.py \
        --odometry_mode=mono \
        --dataset=/path/to/video \
        --config_path=/path/to/config \
        --visualize_rerun

Note: If you don't set config_path, it will try to find stereo.edex in the same directory as the video.
"""

import argparse
import copy
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Dict, List, Sequence, Any, Optional
import numpy as np
import cuvslam as vslam
import conversions as conv
from edex_reader import EdexReader
from video_reader import VideoReader
from visualizer import RerunVisualizer, plot_trajectory
from metrics import calculate_sequence_errors
from generate_report import generate_report, get_fps, save_stats_to_json


class Stat:
    sequence_title: str = ""
    n_frames: int = 0
    tracking_time: float = 0
    average_fps: float = -1
    bird_view_with_errors_path: str = ""
    gt_av_translation_error: float = 0
    gt_av_rotation_error: float = 0
    gt_n_error_segments: int = 0
    gt_simple_error: float = 0
    num_tracking_losts: int = -1
    odometry_mode: str = ""


class TrackerResults:
    """ Store results of tracking """
    rig: vslam.Rig
    frame_metadata: Dict[int, Dict[str, Any]]
    world_from_rig: Dict[int, vslam.Pose]
    final_landmarks: Dict[int, vslam.Landmark]
    tracks2D: Dict[int, List[vslam.Observation]]
    loop_closures: Dict[int, vslam.Pose]
    stat: Stat = Stat()


class Tracker:
    def __init__(self, rig: vslam.Rig, args: argparse.Namespace):
        self.visualizer = None

        # Configure tracker with args if they're in dict format
        self.odom_cfg = vslam.Tracker.OdometryConfig()
        self.configure_tracker(args)

        # Configure RGBD settings if in RGBD mode
        rgbd_settings = self._initialize_rgbd_settings(args)
        if rgbd_settings:
            self.odom_cfg.rgbd_settings = rgbd_settings

        # Configure SLAM if needed
        self.slam_cfg = None
        if args.use_slam:
            self.odom_cfg.enable_observations_export = True
            self.odom_cfg.enable_landmarks_export = True
            self.slam_cfg = vslam.Tracker.SlamConfig()
            self.slam_cfg.use_gpu = self.odom_cfg.use_gpu
            self.slam_cfg.sync_mode = args.sync_slam
            if args.visualize_rerun:
                self.slam_cfg.enable_reading_internals = True

        print(
            f"cuVSLAM version: {vslam.get_version()}\nOdometry config:\n{conv.to_str(self.odom_cfg)}")
        if self.slam_cfg:
            print(f"SLAM config:\n{conv.to_str(self.slam_cfg)}")

        self.stat = Stat()
        self.stat.odometry_mode = str(self.odom_cfg.odometry_mode)
        self.tracker = vslam.Tracker(rig, self.odom_cfg, self.slam_cfg)

        self.frame_id_from_ts = {}
        self.world_from_rig = {}
        self.final_landmarks = {}
        self.tracks2D = {}
        self.landmarks = {}
        self.frame_metadata = {}  # Store loop and direction info for each frame
        self.loop_closures = {}
        self.num_loops = args.num_loops

        if args.visualize_rerun:
            self.visualizer = RerunVisualizer()
            if (not self.odom_cfg.enable_observations_export and
                not self.odom_cfg.enable_landmarks_export and
                not self.odom_cfg.enable_final_landmarks_export ):
                print("Exporting landmarks or observations is disabled, skipping visualization in Rerun")

    def configure_tracker(self, args: argparse.Namespace) -> None:
        """Configure tracker with default and user settings"""

        # Configure tracker with safe argument checking
        config_attrs = [
            'odometry_mode', 'multicam_mode', 'use_gpu', 'async_sba',
            'use_motion_model', 'use_denoising', 'rectified_stereo_camera',
            'enable_observations_export', 'enable_landmarks_export',
            'enable_final_landmarks_export', 'max_frame_delta_s',
            'debug_dump_directory', 'debug_imu_mode'
        ]

        for attr in config_attrs:
            if hasattr(args, attr):
                setattr(self.odom_cfg, attr, getattr(args, attr))

    def _initialize_rgbd_settings(self, args: argparse.Namespace) -> Optional[vslam.Tracker.OdometryRGBDSettings]:
        """Initialize RGBD settings based on args and default values.

        Args:
            args: Command-line arguments containing RGBD configuration

        Returns:
            OdometryRGBDSettings if in RGBD mode, None otherwise

        Raises:
            ValueError: If depth_camera_id is not provided when in RGBD mode
        """
        # Only initialize RGBD settings if in RGBD mode
        if not hasattr(args, 'odometry_mode') or args.odometry_mode != vslam.Tracker.OdometryMode.RGBD:
            return None

        # Create RGBD settings
        rgbd_settings = vslam.Tracker.OdometryRGBDSettings()

        # depth_camera_id is REQUIRED - must be provided in stereo.edex file
        depth_camera_id = getattr(args, 'depth_camera_id', None)
        if depth_camera_id is None:
            raise ValueError(
                "RGBD mode is enabled but 'depth_camera_id' is not provided. "
                "This parameter must be specified in the stereo.edex configuration file."
            )
        rgbd_settings.depth_camera_id = depth_camera_id

        # Set depth_scale_factor (default: 1.0)
        rgbd_settings.depth_scale_factor = getattr(args, 'depth_scale_factor', 1.0)

        # Set enable_depth_stereo_tracking (default: False)
        rgbd_settings.enable_depth_stereo_tracking = getattr(args, 'enable_depth_stereo_tracking', False)

        print(f"RGBD settings initialized: depth_camera_id={rgbd_settings.depth_camera_id}, "
              f"depth_scale_factor={rgbd_settings.depth_scale_factor}, "
              f"enable_depth_stereo_tracking={rgbd_settings.enable_depth_stereo_tracking}")

        return rgbd_settings

    def process_images(self, frame_id: int, timestamps: List[int], images: List, masks: List, depths: List = None):
        timestamp = max(timestamps)
        self.start_time = time.perf_counter()
        odom_pose, slam_pose = self.tracker.track(timestamp, images, masks, depths)

        # Use SLAM pose if available, otherwise use odometry pose
        if slam_pose:
            self.world_from_rig[frame_id] = slam_pose
        else:
            self.world_from_rig[frame_id] = odom_pose.world_from_rig.pose if odom_pose.world_from_rig else None

        self.end_time = time.perf_counter()
        self.stat.tracking_time += self.end_time - self.start_time

        self.frame_id_from_ts[timestamp] = frame_id
        observations_0 = []
        if self.odom_cfg.enable_observations_export:
            # Get last observations for the main camera
            observations_0 = self.tracker.get_last_observations(0)
            self.tracks2D[frame_id] = observations_0
        if self.odom_cfg.enable_landmarks_export:
            # Get last landmarks for the main camera
            landmarks = self.tracker.get_last_landmarks()
            self.landmarks[frame_id] = landmarks

        # Get loop closures if SLAM is enabled
        if self.slam_cfg:
            last_loop_closures = self.tracker.get_loop_closure_poses()
            if last_loop_closures:
                for lc in last_loop_closures:
                    self.loop_closures[lc.timestamp_ns] = lc.pose

        if self.visualizer and odom_pose.world_from_rig:
            gravity = None
            if self.odom_cfg.odometry_mode == vslam.Tracker.OdometryMode.Inertial:
                # Gravity estimation requires collecting sufficient number of keyframes
                # with motion diversity
                gravity = self.tracker.get_last_gravity()
            if self.odom_cfg.enable_final_landmarks_export:
                self.final_landmarks = self.tracker.get_final_landmarks()
            # SLAM data
            pose_graph = self.tracker.get_pose_graph()
            map_landmarks = self.tracker.get_slam_landmarks(vslam.Tracker.SlamDataLayer.Map)
            lc_landmarks = self.tracker.get_slam_landmarks(vslam.Tracker.SlamDataLayer.LoopClosure)
            self.visualizer.visualize_frame(
                frame_id=frame_id,
                images=images,
                odom_pose=odom_pose.world_from_rig.pose,
                slam_pose=slam_pose,
                observations_0=observations_0,
                last_landmarks=landmarks,
                loop_closures=self.loop_closures,
                final_landmarks=self.final_landmarks,
                pose_graph=pose_graph,
                map_landmarks=map_landmarks,
                lc_landmarks=lc_landmarks,
                timestamp=timestamp,
                gravity=gravity,
            )

    def process_imu(self, timestamp: int, linear_accelerations: Sequence[float],
                    angular_velocities: Sequence[float]):
        if self.odom_cfg.odometry_mode == vslam.Tracker.OdometryMode.Inertial:
            imu_measurement = vslam.ImuMeasurement()
            imu_measurement.timestamp_ns = timestamp
            imu_measurement.linear_accelerations = linear_accelerations
            imu_measurement.angular_velocities = angular_velocities
            self.tracker.register_imu_measurement(0, imu_measurement)

    def get_camera_pose(self, frame_id: int):
        return self.world_from_rig.get(frame_id, None)

    def set_frame_metadata(self, frame_id: int, metadata: Dict[str, Any]):
        self.frame_metadata[frame_id] = metadata

    def run_tracking_and_measure_performance(self, dataset, tracker_results: TrackerResults):
        dataset.replay(self)

        # if slam is enabled, overwrite all slam poses in the end after LCs and PGOs
        if self.slam_cfg:
            slam_poses = self.tracker.get_all_slam_poses()
            if slam_poses:
                for pose in slam_poses:
                    self.world_from_rig[self.frame_id_from_ts[pose.timestamp_ns]] = pose.pose

        self.stat.n_frames = len(self.world_from_rig)
        self.stat.average_fps = get_fps(self.stat.tracking_time, self.stat.n_frames)

        if self.odom_cfg.enable_final_landmarks_export:
            self.final_landmarks = self.tracker.get_final_landmarks()
        tracker_results.frame_metadata = self.frame_metadata
        tracker_results.world_from_rig = self.world_from_rig
        tracker_results.loop_closures = self.loop_closures
        tracker_results.final_landmarks = self.final_landmarks
        tracker_results.tracks2D = self.tracks2D
        tracker_results.stat = self.stat


def save_result_to_edex(world_from_rig: Dict[int, vslam.Pose],
                        final_landmarks: Dict[int, vslam.Landmark],
                        tracks2D: Dict[int, List[vslam.Observation]],
                        output_data_file: str):

    # TODO: add get_internal_rig in python API to get camera intrinsics and save yaml output config
    output_data = {
        'camera_poses': {},
        'landmarks_3d': {},
        'tracks_2d': {}
    }

    # Store camera poses
    for frame_id, pose in world_from_rig.items():
        output_data['camera_poses'][frame_id] = {
            'rotation': pose.rotation,
            'translation': pose.translation
        }

    # Store 3D landmarks
    for landmark_id, landmark in final_landmarks.items():
        output_data['landmarks_3d'][landmark_id] = landmark

    # Store 2D tracks
    for frame_id, observations in tracks2D.items():
        output_data['tracks_2d'][frame_id] = [
            {
                'id': obs.id,
                'u': obs.u,
                'v': obs.v,
                'camera_index': obs.camera_index
            }
            for obs in observations
        ]

    # Save all data to a single .npy file
    np.save(output_data_file, output_data)


def track(args: argparse.Namespace,
          refined_focal: Optional[tuple[float, float]] = None,
          refined_principal: Optional[tuple[float, float]] = None) -> TrackerResults:

    if args.num_loops > 0 and args.odometry_mode == vslam.Tracker.OdometryMode.Inertial:
        raise ValueError("Inertial mode is not supported for shuttle mode")

    if args.dataset.endswith('.mp4'):
        dataset = VideoReader(args.dataset, stereo_edex=args.config_path, num_loops=args.num_loops)
    else:
        rgbd_mode = args.odometry_mode == vslam.Tracker.OdometryMode.RGBD
        dataset = EdexReader(args.dataset, stereo_edex=args.config_path, num_loops=args.num_loops, rgbd_mode=rgbd_mode)

    tracker_results = TrackerResults()
    if args.sequence_title:
        tracker_results.stat.sequence_title = args.sequence_title
    if not dataset.validate_rig():
        print("Rig parameters are invalid")
        return tracker_results

    assert dataset.rig is not None
    if refined_focal is not None and refined_principal is not None:
        dataset.rig.cameras[0].focal = refined_focal
        dataset.rig.cameras[0].principal = refined_principal

    # If dataset has RGBD settings, transfer them to args
    # depth_camera_id MUST come from stereo.edex
    if hasattr(dataset, 'rgbd_settings') and dataset.rgbd_settings:
        # depth_camera_id always comes from stereo.edex
        args.depth_camera_id = dataset.rgbd_settings.depth_camera_id

        # Use dataset settings if args don't have custom values (command-line can override these)
        if not hasattr(args, 'depth_scale_factor') or args.depth_scale_factor == 1.0:
            args.depth_scale_factor = dataset.rgbd_settings.depth_scale_factor
        if not hasattr(args, 'enable_depth_stereo_tracking') or not args.enable_depth_stereo_tracking:
            args.enable_depth_stereo_tracking = dataset.rgbd_settings.enable_depth_stereo_tracking

    tracker = Tracker(dataset.rig, args)
    tracker_results.rig = dataset.rig
    if args.sequence_title:
        tracker.stat.sequence_title = args.sequence_title
    tracker.run_tracking_and_measure_performance(dataset, tracker_results)

    if dataset.gt_transforms:
        calculate_sequence_errors(tracker.world_from_rig, dataset.gt_transforms, tracker.stat,
                                  tracker.frame_metadata, args.use_segments, args.segment_lengths, args.num_loops)

    suffix = "_refined" if refined_focal is not None and refined_principal is not None else ""
    plot_trajectory_path = None
    if args.output_dir:
        os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)
        plot_trajectory_path = os.path.join(args.output_dir, "plots", f"{args.sequence_title}{suffix}.png")
        tracker.stat.bird_view_with_errors_path = os.path.abspath(plot_trajectory_path)

    plot_trajectory(
        tracker.world_from_rig,
        tracker.loop_closures,
        dataset.gt_transforms,
        args.visualize_plot,
        plot_trajectory_path,
        dataset.gt_from_shuttle)

    if args.save_output_tracker_data:
        if not args.output_dir:
            print("No output directory provided, skipping output tracker data saving")
            return tracker_results
        if tracker.odom_cfg.enable_final_landmarks_export:
            os.makedirs(os.path.join(args.output_dir, "output_tracker_data"), exist_ok=True)
            output_data_path = os.path.join(args.output_dir, "output_tracker_data",
                                            f"{args.sequence_title}{suffix}.npy")
            save_result_to_edex(
                tracker.world_from_rig,
                tracker.final_landmarks,
                tracker.tracks2D,
                output_data_path)
        else:
            print("Final landmarks export is disabled, skipping output tracker data saving")

    return tracker_results


def process_sequence(sequence, args, CUVSLAM_DATASETS, dataset_folder):
    """Process a single sequence."""

    args_copy = copy.deepcopy(args)
    args_copy.sequence_title = sequence['sequence_title']
    args_copy.dataset = os.path.join(CUVSLAM_DATASETS, dataset_folder, sequence['sequence_folder'])
    args_copy.use_slam = 'use_slam' in sequence and sequence['use_slam']

    tracker_results = track(args_copy)
    return tracker_results.stat


def run_parallel_tracking(json_data, args, CUVSLAM_DATASETS, max_workers=None):
    """Run tracking on multiple sequences in parallel.

    Note: For GPU workloads, use a conservative number of workers (12) to avoid
    out of memory errors, as each worker loads a CUDA tracker into GPU memory.
    """
    if max_workers is None:
        # Default to conservative worker count for GPU workloads to avoid OOM
        # Can be overridden with --max_workers argument
        max_workers = 12
        print(f"Using default max_workers={max_workers} for GPU workloads (use --max_workers to override)")
    else:
        max_workers = max(1, min(max_workers, multiprocessing.cpu_count()))
        print(f"Using max_workers={max_workers} (user specified)")

    stats = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for sequence in json_data['sequence_cfgs']:
            if sequence["enable"] is False:
                continue
            future = executor.submit(
                process_sequence,
                sequence,
                args,
                CUVSLAM_DATASETS,
                json_data['dataset_folder']
            )
            futures.append(future)

        for future in futures:
            stat = future.result()
            if stat is not None:
                stats.append(stat)

    return stats


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        help='Path to the dataset', default='')
    parser.add_argument('--test_config', type=str,
                        help='Path to the test data edex config file', default='')
    parser.add_argument('--sequence_title', type=str,
                        help='Title of the sequence', default='sequence')
    parser.add_argument('--output_dir', type=str,
                        help='Path to the output directory for visualization plots, output tracker data and report', default='')
    parser.add_argument('--save_output_tracker_data', action='store_true',
                        help='Save output tracker data to edex file')
    parser.add_argument('--visualize_rerun', action='store_true',
                        help='Enable real-time visualization of tracking results using Rerun viewer')
    parser.add_argument('--visualize_plot', action='store_true',
                        help='Enable plot visualization of tracking results')
    parser.add_argument('--num_loops', type=int,
                        help='Number of loops for shuttle mode', default=0)
    parser.add_argument('--config_path', type=str,
                        help='Path to the stereo.edex file', default='')
    parser.add_argument('--use_segments', action='store_true',
                        help='Use segments for error calculation', default=False)
    parser.add_argument('--segment_lengths', type=list,
                        help='Segment lengths for error calculation', default=[])

    # Add tracker configuration arguments
    default_cfg = vslam.Tracker.OdometryConfig()
    parser.add_argument('--multicam_mode', type=conv.str2multicam_mode,
                        default=default_cfg.multicam_mode,
                        help='Multicamera mode: performance, precision, or moderate')
    parser.add_argument('--odometry_mode', type=conv.str2odometry_mode,
                        default=default_cfg.odometry_mode,
                        help='Odometry mode: mono, multicamera, inertial, rgbd')
    parser.add_argument('--use_gpu', type=conv.str2bool, default=default_cfg.use_gpu,
                        help='Enable GPU acceleration')
    parser.add_argument('--async_sba', type=conv.str2bool, default=False,  # different from library default
                        help='Enable asynchronous Sparse Bundle Adjustment')
    parser.add_argument('--use_motion_model', type=conv.str2bool, default=default_cfg.use_motion_model,
                        help='Enable motion model for prediction')
    parser.add_argument('--use_denoising', type=conv.str2bool, default=default_cfg.use_denoising,
                        help='Enable denoising of input images')
    parser.add_argument('--rectified_stereo_camera', type=conv.str2bool, default=default_cfg.rectified_stereo_camera,
                        help='Enable rectified stereo camera tracking mode')
    parser.add_argument('--enable_observations_export', type=conv.str2bool, default=default_cfg.enable_observations_export,
                        help='Enable exporting landmark observations during tracking')
    parser.add_argument('--enable_landmarks_export', type=conv.str2bool, default=default_cfg.enable_landmarks_export,
                        help='Enable exporting landmarks during tracking')
    parser.add_argument('--enable_final_landmarks_export', type=conv.str2bool, default=default_cfg.enable_final_landmarks_export,
                        help='Enable exporting final landmarks')
    parser.add_argument('--max_frame_delta_s', type=float, default=default_cfg.max_frame_delta_s,
                        help='Maximum time difference between frames in seconds')
    parser.add_argument('--debug_dump_directory', type=str, default=default_cfg.debug_dump_directory,
                        help='Directory for debug data dumps')
    parser.add_argument('--debug_imu_mode', type=conv.str2bool, default=default_cfg.debug_imu_mode,
                        help='Enable IMU debug mode')
    parser.add_argument('--sync_slam', type=conv.str2bool, default=True,  # different from library default
                        help='Use synchronous SLAM')
    parser.add_argument('--max_workers', type=int, default=None,
                        help='Maximum number of workers for parallel tracking')
    parser.add_argument('--pdf', action='store_true',
                        help='Generate PDF report in addition to HTML report')

    # RGBD-specific arguments
    # Note: depth_camera_id must be specified in stereo.edex file, not via command-line
    parser.add_argument('--depth_scale_factor', type=float, default=1.0,
                        help='Depth scale factor for RGBD mode (e.g., 1000.0 for depth in mm)')
    parser.add_argument('--enable_depth_stereo_tracking', type=conv.str2bool, default=False,
                        help='Enable depth-stereo tracking in RGBD mode')

    args = parser.parse_args()
    stats = []
    config_name = None
    if args.test_config:

        try:
            CUVSLAM_DATASETS = os.environ['CUVSLAM_DATASETS']
        except KeyError:
            raise ValueError("CUVSLAM_DATASETS environment variable is not set")
        json_data = json.load(open(os.path.join(CUVSLAM_DATASETS, args.test_config)))

        # Extract config name (basename without extension) for report naming
        config_name = os.path.splitext(os.path.basename(args.test_config))[0]

        args.segment_lengths = json_data['segment_lengths']

        try:
            cuvslam_output = os.environ['CUVSLAM_OUTPUT']
        except KeyError:
            raise ValueError("CUVSLAM_OUTPUT environment variable is not set")
        args.output_dir = os.path.join(cuvslam_output, os.path.basename(args.test_config).split('.')[0],
                                       datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        stats = run_parallel_tracking(json_data, args, CUVSLAM_DATASETS, max_workers=args.max_workers)
    else:
        tracker_results = track(args)
        stats.append(tracker_results.stat)

    if args.output_dir:
        # Save all stats to a single JSON file
        save_stats_to_json(stats, args.output_dir)
        generate_report(args.output_dir, sys.argv[1:], stats, generate_pdf=args.pdf, config_name=config_name)
    else:
        print("No output directory provided, skipping report generation")

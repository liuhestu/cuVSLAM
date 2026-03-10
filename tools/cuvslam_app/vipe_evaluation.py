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
Example of running evaluation and analysis.

Usage:
    # Run evaluation
    python3 vipe_evaluation.py --video_dir /path/to/videos --output_dir /path/to/output --config_path /path/to/config

    # This will:
    # 1. Process all MP4 videos in video_dir
    # 2. Save results to output_dir/evaluation_results.pkl
    # 3. Save trajectory plots to output_dir/plots/
    # 4. Generate analysis plots
"""

import os
import pickle
from argparse import Namespace
from cuvslam_app import track
from conversions import str2odometry_mode
from refinement import run_refinement
from dataclasses import dataclass
from visualizer import plot_trajectory

@dataclass
class VideoResults:
    video_path: str
    output_dir: str
    initial_t_avg: float
    initial_r_avg: float
    initial_focal: tuple[float, float]
    initial_principal: tuple[float, float]
    refined_focal: tuple[float, float]
    refined_principal: tuple[float, float]
    refined_t_avg: float = None
    refined_r_avg: float = None
    refined_focal: tuple[float, float] = None
    refined_principal: tuple[float, float] = None

def evaluate_videos(video_dir: str, output_dir: str, config_path: str = '',
                    save_output_tracker_data: bool = False, global_optimizer: bool = False):
    """Run evaluation on all MP4 videos in directory."""

    # Create output directory if doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    pickle_path = os.path.join(output_dir, 'evaluation_results.pkl')

    # Load existing results if available
    existing_results = []
    processed_videos = set()
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            existing_results = pickle.load(f)
            processed_videos = {result.video_path for result in existing_results}
            print(f"Found {len(existing_results)} existing results")

    results = existing_results

    # Process each video
    for root, _, files in os.walk(video_dir):
        for filename in files:
            if filename.endswith('.mp4'):
                video_path = os.path.join(root, filename)

                # Skip if already processed
                if video_path in processed_videos:
                   print(f"\nSkipping {video_path} (already processed)")
                   #continue

                print(f"\nProcessing {video_path}")

                # Create args for tracking
                args = Namespace(
                    dataset=video_path,
                    sequence_title=os.path.splitext(os.path.basename(video_path))[0],
                    odometry_mode=str2odometry_mode('mono'),
                    enable_final_landmarks_export=True,
                    enable_observations_export=True,
                    enable_landmarks_export=True,
                    enable_iteration_state_export=False,
                    output_dir=output_dir,
                    visualize_rerun=False,
                    visualize_plot=False,
                    num_loops=1,
                    use_segments=False,
                    segment_lengths=[],
                    save_output_tracker_data=save_output_tracker_data,
                    config_path=config_path
                )

                # Run tracking and get results
                tracker_results = track(args)

                if global_optimizer:

                    refinement_problem, summary = run_refinement(tracker_results, export_iteration_states=args.enable_iteration_state_export)
                    refined_tracker_results = track(
                        args, refinement_problem.rig.cameras[0].focal, refinement_problem.rig.cameras[0].principal)

                    plot_trajectory_path = None
                    if args.output_dir:
                        os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)
                        plot_trajectory_path = os.path.join(args.output_dir, "plots", f"{args.sequence_title}_refinement_output.png")
                    plot_trajectory(refinement_problem.world_from_rigs, None, None, args.visualize_plot, plot_trajectory_path,
                                    False, "Refinement output", invert_pose=False)

                    # Store results
                    results.append(VideoResults(
                        video_path=video_path,
                        initial_t_avg=tracker_results.stat.gt_av_translation_error,
                        initial_r_avg=tracker_results.stat.gt_av_rotation_error,
                        initial_focal=tuple(tracker_results.rig.cameras[0].focal),
                        initial_principal=tuple(tracker_results.rig.cameras[0].principal),
                        output_dir=output_dir,
                        refined_t_avg=refined_tracker_results.stat.gt_av_translation_error,
                        refined_r_avg=refined_tracker_results.stat.gt_av_rotation_error,
                        refined_focal=tuple(refined_tracker_results.rig.cameras[0].focal),
                        refined_principal=tuple(refined_tracker_results.rig.cameras[0].principal)
                        # initial_trans_drift_percentage=tracker_results['initial_trans_drift_percentage'],
                        # initial_rot_error=tracker_results['initial_rot_error'],
                        # refined_trans_drift_percentage=refined_tracker_results['refined_trans_drift_percentage'],
                        # refined_rot_error=refined_tracker_results['refined_rot_error'],
                    ))
                else:
                    results.append(VideoResults(
                        video_path=video_path,
                        initial_t_avg=tracker_results.stat.gt_av_translation_error,
                        initial_r_avg=tracker_results.stat.gt_av_rotation_error,
                        initial_focal=tuple(tracker_results.rig.cameras[0].focal),
                        initial_principal=tuple(tracker_results.rig.cameras[0].principal),
                        output_dir=output_dir
                    ))

                print(f"Results saved for {filename}")
                # Save all results to pickle file
                with open(pickle_path, 'wb') as f:
                    pickle.dump(results, f)
    print(f"\nAll results saved to {pickle_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, required=True,
                        help='Directory containing MP4 videos')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save evaluation results')
    parser.add_argument('--config_path', type=str, default='',
                        help='Path to the config file')
    parser.add_argument('--save_output_tracker_data', action='store_true',
                        help='Save output tracker data')
    parser.add_argument('--global_optimizer', action='store_true',
                        help='Run refinement on the tracking results')
    args = parser.parse_args()
    evaluate_videos(args.video_dir, args.output_dir, args.config_path,
                    args.save_output_tracker_data, args.global_optimizer)

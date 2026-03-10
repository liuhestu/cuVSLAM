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

import numpy as np
import cuvslam as vslam
from visualizer import plot_trajectory

def remove_unseen_points(problem: vslam.refinement.BundleAdjustmentProblem,
                         points_3d: dict[int, np.ndarray]):
    """Removes points not seen by any camera."""
    seen_points = set()
    points_in_world = problem.points_in_world

    for frame_id, observations in problem.observations.items():
        for obs in observations:
            seen_points.add(obs.id)

    # Remove points not seen by any camera
    for landmark_id, _ in points_3d.items():
        if landmark_id not in seen_points:
            if landmark_id in points_in_world:
                del points_in_world[landmark_id]

    problem.points_in_world = points_in_world

def remove_observations_without_3d_points(problem: vslam.refinement.BundleAdjustmentProblem,
                                          points_3d: dict[int, np.ndarray]):
    """Removes observations that don't have corresponding 3D points."""
    valid_landmark_ids = set(points_3d.keys())
    filtered_observations = {}
    total_valid_obs = 0

    for frame_id, observations in problem.observations.items():
        valid_obs = [obs for obs in observations if obs.id in valid_landmark_ids]
        if valid_obs:
            filtered_observations[frame_id] = valid_obs
            total_valid_obs += len(valid_obs)

    problem.observations = filtered_observations
    if problem.fixed_points:
        problem.fixed_points = list(set(problem.fixed_points) & valid_landmark_ids)
        print(f"Updated fixed points: {len(problem.fixed_points)}")

    print(f"Total valid observations: {total_valid_obs} across {len(filtered_observations)} frames")

def run_refinement(tracker_results, export_iteration_states=False) -> vslam.Rig:
    """Run bundle adjustment refinement on tracking results."""
    problem = vslam.refinement.BundleAdjustmentProblem()
    problem.rig = tracker_results.rig
    problem.fixed_frames = [0]

    # Convert camera poses and landmarks to OpenCV coordinate system
    landmarks = {}
    for landmark_id, landmark_coords in tracker_results.finalTracks3D.items():
        landmark = vslam.Landmark()
        landmark.id = landmark_id
        landmark.coords = landmark_coords
        landmarks[landmark_id] = landmark
    problem.points_in_world = landmarks

    # Filter poses and observations from first forward run only
    world_from_rigs = {}
    observations = {}
    for frame_id, pose in tracker_results.world_from_rig.items():
        if frame_id in tracker_results.frame_metadata:
            frame_meta = tracker_results.frame_metadata[frame_id]
            if frame_meta["loop"] == 0 and frame_meta["forward"]:
                # Transform pose to OpenCV coordinates
                world_from_rigs[frame_id] = pose

                # Add observations for this frame if available
                if frame_id in tracker_results.tracks2D:
                    observations[frame_id] = tracker_results.tracks2D[frame_id]

    problem.world_from_rigs = world_from_rigs

    problem.observations = observations
    problem.fixed_points = []

    remove_unseen_points(problem, tracker_results.finalTracks3D)
    remove_observations_without_3d_points(problem, tracker_results.finalTracks3D)

    print(f"Loaded from tracker:")
    print(f"- {len(problem.world_from_rigs)} camera poses")
    print(f"- {len(problem.observations)} frames with observations")
    print(f"- {len(problem.points_in_world)} landmarks")
    print(f"- {len(problem.fixed_points)} fixed points")

    options = vslam.refinement.BundleAdjustmentProblemOptions()
    options.verbose = True
    options.estimate_intrinsics = True
    options.symmetric_focal_length = True
    options.export_iteration_state = export_iteration_states
    refined_problem, summary = vslam.refinement.refine(problem, options)

    print(summary.brief_report())

    print("problem.rig.cameras[0].focal = ", problem.rig.cameras[0].focal)
    print("problem.rig.cameras[0].principal = ", problem.rig.cameras[0].principal)
    if problem.rig.cameras[0].distortion.model == vslam.Distortion.Model.Polynomial:
        print("problem.rig.cameras[0].distortion.parameters = ", problem.rig.cameras[0].distortion.parameters)
    print("refined_problem.rig.cameras[0].focal = ", refined_problem.rig.cameras[0].focal)
    print("refined_problem.rig.cameras[0].principal = ", refined_problem.rig.cameras[0].principal)
    if refined_problem.rig.cameras[0].distortion.model == vslam.Distortion.Model.Polynomial:
        print("refined_problem.rig.cameras[0].distortion.parameters = ", refined_problem.rig.cameras[0].distortion.parameters)

    return refined_problem, summary

def visualize_reprojection_errors(problem, summary, output_dir):
    """
    Visualize reprojection errors for each frame across iterations.

    Args:
        problem: BundleAdjustmentProblem containing the original problem data
        summary: BundleAdjustmentProblemSummary containing iteration data
        output_dir: Directory to save visualization images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get number of iterations
    num_iterations = len(summary.iteration_rigs_from_world)
    if num_iterations == 0:
        print("No iterations to visualize")
        return

    # Process each frame
    for frame_id, observations in problem.observations.items():
        if frame_id % 10 != 0:
            continue

        # Get the camera parameters
        rig = problem.rig

        # Create a base image (black background)
        # Use the first camera's dimensions
        img_width = int(rig.cameras[0].size[0])
        img_height = int(rig.cameras[0].size[1])
        base_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

        # Draw observations (red dots)
        obs_img = base_img.copy()
        for obs in observations:
            # Draw observation point in red
            u, v = int(obs.u), int(obs.v)
            if 0 <= u < img_width and 0 <= v < img_height:
                cv2.circle(obs_img, (u, v), 3, (0, 0, 255), -1)  # Red dot

        # For each iteration, project 3D points and visualize
        for iter_idx in range(0, len(summary.iteration_rigs_from_world), 5):
            # Get the camera parameters for this iteration
            iter_rig = summary.iteration_cameras[iter_idx]

            # Get the frame pose for this iteration
            frame_pose = summary.iteration_rigs_from_world[iter_idx][frame_id]

            # Create a copy of the observation image
            iter_img = obs_img.copy()

            # For each observation, find the corresponding 3D point and project it
            for obs in observations:
                landmark_id = obs.id
                camera_idx = obs.camera_index

                # Get the 3D point for this iteration
                landmark = summary.iteration_points_in_world[iter_idx][landmark_id]
                # Extract coordinates from the landmark
                point_3d = landmark.coords

                # Get camera parameters from the iteration-specific rig
                camera = iter_rig.cameras[camera_idx]
                focal = camera.focal
                principal = camera.principal

                # Transform point from world to camera frame
                # First transform from world to rig
                rig_from_world = frame_pose
                world_point = np.array(point_3d)

                # Apply rotation and translation
                rotation_matrix = np.array(rig_from_world.rotation).reshape(3, 3).transpose()
                translation_vector = np.array(rig_from_world.translation)
                rig_point = np.dot(rotation_matrix, world_point - translation_vector)

                # Then transform from rig to camera using the iteration-specific camera extrinsics
                # camera_from_rig = camera.rig_from_camera
                # camera_rotation = np.array(camera_from_rig.rotation).reshape(3, 3).T
                # camera_translation = np.array(camera_from_rig.translation)
                # camera_point = np.dot(camera_rotation, rig_point - camera_translation)
                camera_point = rig_point

                # Project to image plane (simple pinhole projection)
                if camera_point[2] <= 0:
                    continue  # Point is behind the camera

                # Normalize coordinates
                x_n = camera_point[0] / camera_point[2]
                y_n = camera_point[1] / camera_point[2]

                # Apply distortion if it's a polynomial model
                if camera.distortion.model == vslam.Distortion.Model.Polynomial:
                    # Get distortion parameters
                    params = camera.distortion.parameters
                    # Extract distortion coefficients (OpenCV order: k1, k2, p1, p2, k3, k4, k5, k6)
                    k1 = params[0]
                    k2 = params[1]
                    p1 = params[2]
                    p2 = params[3]
                    k3 = params[4]
                    k4 = params[5]
                    k5 = params[6]
                    k6 = params[7]

                    # Calculate r^2, r^4, r^6
                    r2 = x_n*x_n + y_n*y_n
                    r4 = r2 * r2
                    r6 = r4 * r2

                    # Calculate radial distortion factor
                    numerator = 1.0 + k1*r2 + k2*r4 + k3*r6
                    denominator = 1.0 + k4*r2 + k5*r4 + k6*r6

                    radial_factor = numerator / denominator

                    # Calculate tangential distortion terms
                    xy = x_n * y_n  # x'y'
                    x_n_2 = x_n * x_n  # x'^2
                    y_n_2 = y_n * y_n  # y'^2

                    # Apply distortion (OpenCV's exact formulation)
                    # x'' = x'*(radial_factor) + 2*p1*x'*y' + p2*(r^2 + 2*x'^2)
                    # y'' = y'*(radial_factor) + p1*(r^2 + 2*y'^2) + 2*p2*x'*y'
                    x_d = x_n * radial_factor + 2.0 * p1 * xy + p2 * (r2 + 2.0 * x_n_2)
                    y_d = y_n * radial_factor + p1 * (r2 + 2.0 * y_n_2) + 2.0 * p2 * xy
                else:
                    # No distortion
                    x_d, y_d = x_n, y_n

                # Project to pixel coordinates
                proj_u = int(principal[0] + focal[0] * x_d)
                proj_v = int(principal[1] + focal[1] * y_d)

                # Draw projected point (blue) and line to observation
                obs_u, obs_v = int(obs.u), int(obs.v)
                if (0 <= proj_u < img_width and 0 <= proj_v < img_height and
                    0 <= obs_u < img_width and 0 <= obs_v < img_height):
                    cv2.circle(iter_img, (proj_u, proj_v), 3, (255, 0, 0), -1)  # Blue dot
                    cv2.line(iter_img, (obs_u, obs_v), (proj_u, proj_v), (0, 255, 0), 1)  # Green line

            # Save the image
            output_path = os.path.join(output_dir, f"frame_{frame_id}_iter_{iter_idx}.png")
            cv2.imwrite(output_path, iter_img)


    print(f"Saved reprojection error visualizations to {output_dir}")

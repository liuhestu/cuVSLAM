
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

#pragma once

#ifdef USE_RERUN
#include <unordered_map>
#include <vector>

#include "camera/observation.h"
#include "camera/rig.h"
#include "common/isometry.h"
#include "common/log.h"
#include "common/rerun.h"
#include "common/track_id.h"
#include "common/vector_2t.h"
#include "common/vector_3t.h"
#include "pipelines/track.h"

#include "visualizer/visualizer.hpp"

namespace cuvslam::pipelines {
void logObservations(const std::vector<camera::Observation>& observations, const camera::Rig& rig,
                     const std::string& viewport_name, const Color& color);

void logLandmarks(const std::unordered_map<TrackId, Vector3T>& landmarks, const Isometry3T& camera_from_world,
                  const camera::ICameraModel& camera_model, const std::string& viewport_name, const Color& color);

void logLandmarks(const std::vector<pipelines::Landmark>& landmarks, const Isometry3T& camera_from_world,
                  const camera::ICameraModel& camera_model, const std::string& viewport_name, const Color& color);

/**
 * Log landmarks as 3D points in world frame
 * @param landmarks Map of track IDs to 3D positions in world frame
 * @param viewport_name Rerun viewport name for the 3D landmarks
 * @param color Color for the landmark points
 * @param point_radius Radius of the landmark points
 */
void logLandmarks3D(const std::unordered_map<TrackId, Vector3T>& landmarks, const std::string& viewport_name,
                    const Color& color, float point_radius = 0.01f);

/**
 * Log landmarks as 3D points in world frame (vector version)
 * @param landmarks Vector of landmarks with 3D positions in world frame
 * @param viewport_name Rerun viewport name for the 3D landmarks
 * @param color Color for the landmark points
 * @param point_radius Radius of the landmark points
 */
void logLandmarks3D(const std::vector<pipelines::Landmark>& landmarks, const std::string& viewport_name,
                    const Color& color, float point_radius = 0.01f);

/**
 * Clear 3D landmarks visualization
 * @param viewport_name Rerun viewport name to clear
 */
void clearLandmarks3D(const std::string& viewport_name);

/**
 * Log camera trajectory in world frame.
 * Limitation: Currently this function can only be called in a single thread (main thread for a pipeline and base
 * launcher).
 * @param rig_from_world The current rig pose (world to rig transform)
 * @param viewport_name Rerun viewport name for the trajectory
 * @param color Color for the trajectory points/line
 * @param trajectory_type Type of trajectory (VO, SBA, or GT)
 * @param show_axes If true, show camera orientation axes
 * @param axis_length Length of the camera axes arrows
 */
void logTrajectory(const Isometry3T& rig_from_world, const std::string& viewport_name, const Color& color,
                   TrajectoryType trajectory_type = TrajectoryType::VO, bool show_axes = true,
                   float axis_length = 0.1f);

/**
 * Clear the accumulated trajectory
 * Limitation: Currently this function can only be called in a single thread (main thread for a pipeline and base
 * launcher).
 * @param viewport_name Rerun viewport name to clear
 * @param trajectory_type Type of trajectory to clear (VO, SBA, or GT)
 */
void clearTrajectory(const std::string& viewport_name, TrajectoryType trajectory_type);

}  // namespace cuvslam::pipelines
#endif

// SPDX-License-Identifier: MIT
/**
 * @file      triangulation.h
 * @brief     Triangulation of a point from unit rays observed by several camera poses.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2026-09-28
 * @copyright Copyright (c) 2026 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 *
 * The linear solution is the point closest to every observation ray in the least-squares sense (Hartley & Zisserman,
 * "Multiple View Geometry", 2nd ed., Ch. 12). The refinement minimizes the reprojection error in anchored inverse depth
 * (Mourikis & Roumeliotis, ICRA 2007, Appendix) with Levenberg-Marquardt damping (Marquardt, SIAM J. Appl. Math. 1963).
 * Both work with unit rays, so observations anywhere on the sphere are supported: the anchor frame and every
 * observation frame are rotated so that their ray is the +z axis before the inverse-depth parameterization is applied.
 */

#pragma once

#include <Eigen/Dense>

#include <vector>

namespace vio {

// Triangulation and refinement parameters
struct TriangulationOptions {
  int max_iterations = 5;                 // Maximum accepted refinement steps
  double initial_damping = 1e-3;          // Initial damping of the refinement
  double max_damping = 1e10;              // Damping at which the refinement gives up
  double min_step = 1e-6;                 // Step norm considered converged
  double min_relative_decrease = 1e-6;    // Relative cost decrease considered converged
  double damping_factor = 10.0;           // Damping multiplier after a rejected step (divisor after an accepted one)
  double min_depth = 0.10;                // Minimum depth along the anchor ray [m]
  double max_depth = 60.0;                // Maximum depth along the anchor ray [m]
  double max_depth_baseline_ratio = 40.0; // Maximum ratio of the point distance to the largest baseline
  double max_condition_number = 10000.0;  // Maximum condition number of the linear system

  bool Check() const;
};

// Outcome of a triangulation
enum class TriangulationStatus {
  kSuccess,
  kIllConditioned,  // Linear system condition number too large
  kLinearDepth,     // Linear solution outside the depth limits
  kRefinedDepth,    // Refined solution outside the depth limits
  kBaseline,        // Distance / baseline ratio too large
  kNotFinite        // NaN in the solution
};

// Pose of a camera in the world frame
struct CameraPose {
  Eigen::Matrix3d R_GtoC;  // Rotation world -> camera
  Eigen::Vector3d p_CinG;  // Camera position in the world
};

/**
 * @brief Linear triangulation followed by the refinement; the last observation is the anchor
 * @param options Triangulation parameters
 * @param rays Unit rays in each camera frame
 * @param poses Camera pose of each observation
 * @param p_FinG Output point in the world frame
 * @return kSuccess, or the check that rejected the point
 */
TriangulationStatus TriangulatePoint(const TriangulationOptions& options,
                                     const std::vector<Eigen::Vector3d>& rays,
                                     const std::vector<CameraPose>& poses,
                                     Eigen::Vector3d* p_FinG);

/**
 * @brief Minimal rotation that maps a unit vector onto +z
 * @param b Unit vector
 * @return R with R * b = (0, 0, 1)
 */
Eigen::Matrix3d RotationToPlusZ(const Eigen::Vector3d& b);

}  // namespace vio

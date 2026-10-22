// SPDX-License-Identifier: MIT
/**
 * @file      keyframe.h
 * @brief     A keyframe: an image whose pose estimate is final because it left the estimator's window.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2026-10-05
 * @copyright Copyright (c) 2026 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 *
 * Reported through the keyframe callback of the odometry pipeline (the hook for a later loop-closure component).
 */

#pragma once

#include "vio/util/types.h"

#include <Eigen/Core>

#include <vector>

namespace vio {

struct Keyframe {
  Timestamp timestamp = 0.0;                                        // Image time stamp [s]
  Timestamp capture_time = 0.0;                                     // Pose time (sensor 0 exposure, IMU clock) [s]
  Eigen::Vector4d q_GtoI = Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);     // Orientation (JPL quaternion)
  Eigen::Vector3d p_IinG = Eigen::Vector3d::Zero();                 // Position [m]
  Eigen::Matrix<double, 6, 6> covariance = Eigen::Matrix<double, 6, 6>::Zero();  // [theta, p]
  std::vector<PixelObservation> observations;                       // Features seen in the image
  std::vector<Eigen::Vector3d> rays;                                // Their unit rays in the camera frame
};

}  // namespace vio

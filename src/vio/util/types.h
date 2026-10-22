// SPDX-License-Identifier: MIT
/**
 * @file      types.h
 * @brief     Basic types shared by all modules: time, IMU samples and noise, pixel observations.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2026-10-05
 * @copyright Copyright (c) 2026 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include <Eigen/Core>

namespace vio {

// Time [s]
using Timestamp = double;

// One IMU measurement (IMU clock)
struct ImuSample {
  Timestamp timestamp = 0.0;
  Eigen::Vector3d gyro = Eigen::Vector3d::Zero();   // Angular velocity [rad/s]
  Eigen::Vector3d accel = Eigen::Vector3d::Zero();  // Specific force [m/s^2]
};

// Continuous-time IMU noise densities
struct ImuNoise {
  double gyro = 0.0;        // Gyroscope white noise [rad/s/sqrt(Hz)]
  double gyro_walk = 0.0;   // Gyroscope random walk [rad/s^2/sqrt(Hz)]
  double accel = 0.0;       // Accelerometer white noise [m/s^2/sqrt(Hz)]
  double accel_walk = 0.0;  // Accelerometer random walk [m/s^3/sqrt(Hz)]
};

// Observation of a feature track in an image
struct PixelObservation {
  int id = -1;                                      // Track id
  Eigen::Vector2d pixel = Eigen::Vector2d::Zero();  // Pixel coordinates [u, v]
  int camera = 0;                                   // Camera of the pixel: 0, or 1 (second camera of a stereo rig)
};

}  // namespace vio

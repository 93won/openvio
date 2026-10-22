// SPDX-License-Identifier: MIT
/**
 * @file      rig.h
 * @brief     Camera-IMU rig: extrinsics, clocks and the timing of the camera's sensors.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2026-10-05
 * @copyright Copyright (c) 2026 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 *
 * One camera image can be made by several sensors (a two-lens 360-degree camera stitches the exposures of its front
 * and back lens). Sensor 0 looks along +z of the camera frame, sensor 1 along -z. The time offsets and the
 * extrinsics are estimated online by the estimator, which keeps its current Rig.
 *
 * A stereo rig adds a second camera (camera 1), exposed with camera 0 and fixed to it by its calibration: a point
 * moves with the camera-0 extrinsics into camera 0, then by (R_C0toC1, p_C0inC1) into camera 1.
 */

#pragma once

#include <Eigen/Core>

namespace vio {

struct Rig {
  Eigen::Matrix3d R_ItoC = Eigen::Matrix3d::Identity();  // Rotation IMU -> camera
  Eigen::Vector3d p_IinC = Eigen::Vector3d::Zero();      // IMU position in the camera frame [m]
  double clock_offset = 0.0;          // IMU clock minus camera clock: t_imu = t_cam + clock_offset [s] (fixed)
  double time_offset = 0.0;           // Exposure of sensor 0 minus the image time stamp, on the IMU clock [s]
  double sensor_time_offset = 0.0;    // Exposure of sensor 1 minus the exposure of sensor 0 [s]
  Eigen::Matrix3d R_C0toC1 = Eigen::Matrix3d::Identity();  // Stereo: rotation camera 0 -> camera 1 (fixed)
  Eigen::Vector3d p_C0inC1 = Eigen::Vector3d::Zero();      // Stereo: camera-0 center in the camera-1 frame [m] (fixed)
};

}  // namespace vio

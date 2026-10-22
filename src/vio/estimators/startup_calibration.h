// SPDX-License-Identifier: MIT
/**
 * @file      startup_calibration.h
 * @brief     Calibration from the data before the initialization: sensor time offsets and the accel bias at rest.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2026-09-28
 * @copyright Copyright (c) 2026 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 *
 * Sensor time offsets: the camera rotation of every image pair is measured per sensor (lens) from its tracks and
 * compared with the gyro rotation over the same interval shifted by dt (with a constant gyro bias); dt minimizes the
 * squared difference. Accel bias at rest: starting at rest, the mean specific force over all samples up to the
 * initialization is gravity plus the accel bias plus the mean acceleration, which is the velocity at the
 * initialization over the elapsed time; its component along gravity gives the bias component |f| - g - v_up / T.
 */

#pragma once

#include "vio/scene/camera.h"
#include "vio/sensor/rig.h"
#include "vio/util/types.h"

#include <Eigen/Core>

#include <map>
#include <utility>
#include <vector>

namespace vio {

class StartupCalibration {
 public:
  /**
   * @param camera Camera model of the observations
   * @param rig Camera-IMU extrinsics
   */
  StartupCalibration(const Camera& camera, const Rig& rig);

  // Add one IMU sample (camera clock)
  void AddImu(const ImuSample& sample);

  // Add the tracked features of one image (camera clock): per-sensor rotation from the previous image
  void AddImage(Timestamp timestamp, const std::vector<PixelObservation>& observations);

  /**
   * @brief Time offset of each sensor of the camera from the rotations recorded so far
   * @param offset0 Output sensor-0 offset [s]
   * @param offset1 Output sensor-1 offset [s] (zero for a camera with one sensor)
   * @return false if a sensor lacks rotation or data (offsets are then zero)
   */
  bool EstimateSensorTimeOffsets(double* offset0, double* offset1) const;

  /**
   * @brief Accel-bias component along gravity, if the data start at rest (median track motion of the first second
   *        below half a pixel)
   * @param gravity_magnitude [m/s^2]
   * @param velocity_up Velocity along gravity at the initialization [m/s]
   * @param direction Output unit direction of gravity (mean specific force) in the IMU frame
   * @param bias Output bias component [m/s^2]
   * @param elapsed Output time from the first sample to the initialization [s]
   * @return false if the start is not at rest or there is no data
   */
  bool EstimateRestAccelBias(double gravity_magnitude, double velocity_up, Eigen::Vector3d* direction, double* bias,
                             double* elapsed) const;

 private:
  // Camera rotation between two consecutive images per sensor
  struct SensorRotation {
    double t0;                 // Previous image time [s]
    double t1;                 // Image time [s]
    Eigen::Vector3d theta[2];  // Rotation vector from t0 to t1, IMU frame
    bool valid[2];             // Enough tracks in the sensor
  };

  Camera camera_;
  Eigen::Matrix3d R_ItoC_;
  std::vector<std::pair<double, Eigen::Vector3d>> gyro_;   // Gyro samples (camera clock)
  std::vector<std::pair<double, Eigen::Vector3d>> accel_;  // Accel samples (camera clock)
  std::map<int, Eigen::Vector3d> prev_rays_;               // Rays of the previous image by track id
  double prev_time_;                                       // Time of the previous image [s]
  std::vector<SensorRotation> rotations_;                  // Per-sensor rotations of the image pairs
  std::vector<std::pair<double, double>> motion_;          // Mean track displacement per image [rad]
};

}  // namespace vio

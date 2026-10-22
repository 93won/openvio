// SPDX-License-Identifier: MIT
/**
 * @file      equirectangular_camera_model.h
 * @brief     Equirectangular (360-degree) image of a two-lens camera.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 *
 * Camera frame: x right, y down, z forward (axis of the front lens, sensor 0; the back lens, sensor 1, sees z <= 0).
 *   u = W (0.5 + atan2(x, z) / 2pi),   v = H (0.5 + asin(y / |p|) / pi)
 * Parameter k: lens-angle error of the stitching. A ray at angle theta from its lens axis is imaged at
 * theta + k theta (1 - (2 theta / pi)^2), same azimuth (zero at the axis and at the seam). k = 0 is the plain
 * projection above, which the model then evaluates directly. k is small and refined online; the derivative w.r.t. the
 * point, the image scale and the tracking front end use the plain projection (k = 0).
 */

#pragma once

#include "vio/sensor/camera_model.h"

#include <Eigen/Core>
#include <opencv2/core/types.hpp>

namespace vio {

struct EquirectangularCameraModel : public CameraModelBase<EquirectangularCameraModel> {
  static constexpr CameraModelId kId = CameraModelId::kEquirectangular;
  static constexpr const char* kName = "EQUIRECTANGULAR";
  static constexpr int kNumParameters = 1;             // Lens-angle error k
  static constexpr int kNumOnlineParams = 1;           // k
  static constexpr int kNumSensors = 2;                // Front lens (z > 0), back lens
  static constexpr bool kWrapsHorizontally = true;     // 360 degrees of longitude

  // Plain projection: k = 0
  static void PlainParams(double* params) { params[0] = 0.0; }

  // ============ Estimation (double precision) ============

  /**
   * @brief Pixel of a point in the camera frame
   * @param d_pixel_d_params Output d pixel / d k [2] (null: not computed)
   */
  static Eigen::Vector2d Project(int width, int height, const double* params, const Eigen::Vector3d& point,
                                 double* d_pixel_d_params);

  /**
   * @brief Ray of a pixel (unit length for k = 0), with the lens-angle error removed by Newton steps
   */
  static Eigen::Vector3d Unproject(int width, int height, const double* params, const Eigen::Vector2d& pixel);

  /**
   * @brief d pixel / d point of the plain projection
   * @return false near the poles where the derivative is undefined
   */
  static bool ProjectionJacobian(int width, int height, const double* params, const Eigen::Vector3d& point,
                                 Eigen::Matrix<double, 2, 3>* J);

  /**
   * @brief Measured minus predicted pixel, the horizontal part wrapped into [-W/2, W/2] across the seam
   */
  static Eigen::Vector2d Residual(int width, int height, const Eigen::Vector2d& measured,
                                  const Eigen::Vector2d& predicted);

  static int SensorOf(const Eigen::Vector3d& ray) { return (ray.z() > 0.0) ? 0 : 1; }

  static double PixelsPerRadian(int width, int height, const double* params);

  // ============ Tracking front end (single precision, k = 0) ============

  static Eigen::Vector3f PixelToRay(int width, int height, const double* params, const cv::Point2f& pixel);
  static cv::Point2f RayToPixel(int width, int height, const double* params, const Eigen::Vector3f& ray);
};

}  // namespace vio

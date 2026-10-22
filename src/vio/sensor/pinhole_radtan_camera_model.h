// SPDX-License-Identifier: MIT
/**
 * @file      pinhole_radtan_camera_model.h
 * @brief     Pinhole camera with radial and tangential lens distortion.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2026-10-06
 * @copyright Copyright (c) 2026 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 *
 * Camera frame: x right, y down, z forward. Parameters [fx, fy, cx, cy, k1, k2, p1, p2], all calibrated (none is
 * refined online). With x = X / Z, y = Y / Z and r^2 = x^2 + y^2 the distorted point is
 *   x_d = x (1 + k1 r^2 + k2 r^4) + 2 p1 x y + p2 (r^2 + 2 x^2)
 *   y_d = y (1 + k1 r^2 + k2 r^4) + p1 (r^2 + 2 y^2) + 2 p2 x y
 * and the pixel u = fx x_d + cx, v = fy y_d + cy (radial and decentering distortion, Brown 1966 and 1971; see
 * doc/references.md). Points with Z <= 0 cannot be imaged; the ray of a pixel inverts the distortion by Gauss-Newton.
 */

#pragma once

#include "vio/sensor/camera_model.h"

#include <Eigen/Core>
#include <opencv2/core/types.hpp>

namespace vio {

struct PinholeRadtanCameraModel : public CameraModelBase<PinholeRadtanCameraModel> {
  static constexpr CameraModelId kId = CameraModelId::kPinholeRadtan;
  static constexpr const char* kName = "PINHOLE_RADTAN";
  static constexpr int kNumParameters = 8;           // fx, fy, cx, cy, k1, k2, p1, p2
  static constexpr int kNumOnlineParams = 0;         // All calibrated
  static constexpr int kNumSensors = 1;
  static constexpr bool kWrapsHorizontally = false;

  // No online corrections: the calibrated parameters
  static void PlainParams(double* params) { (void)params; }

  // ============ Estimation (double precision) ============

  /**
   * @brief Pixel of a point in the camera frame; (-1, -1), outside the image, for Z <= 0
   * @param d_pixel_d_params Output d pixel / d params [2 x 8, column-major] (null: not computed)
   */
  static Eigen::Vector2d Project(int width, int height, const double* params, const Eigen::Vector3d& point,
                                 double* d_pixel_d_params);

  // Unit ray of a pixel
  static Eigen::Vector3d Unproject(int width, int height, const double* params, const Eigen::Vector2d& pixel);

  /**
   * @brief d pixel / d point
   * @return false for points that cannot be imaged (Z <= 0)
   */
  static bool ProjectionJacobian(int width, int height, const double* params, const Eigen::Vector3d& point,
                                 Eigen::Matrix<double, 2, 3>* J);

  static Eigen::Vector2d Residual(int width, int height, const Eigen::Vector2d& measured,
                                  const Eigen::Vector2d& predicted) {
    (void)width;
    (void)height;
    return measured - predicted;
  }

  static int SensorOf(const Eigen::Vector3d& ray) {
    (void)ray;
    return 0;
  }

  // Geometric mean of the focal lengths
  static double PixelsPerRadian(int width, int height, const double* params);

  // ============ Tracking front end (single precision) ============

  static Eigen::Vector3f PixelToRay(int width, int height, const double* params, const cv::Point2f& pixel);
  static cv::Point2f RayToPixel(int width, int height, const double* params, const Eigen::Vector3f& ray);
};

}  // namespace vio

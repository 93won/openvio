// SPDX-License-Identifier: MIT
/**
 * @file      fisheye_equidistant_camera_model.h
 * @brief     Fisheye camera with the equidistant projection and a polynomial in the incidence angle.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2026-10-06
 * @copyright Copyright (c) 2026 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 *
 * Camera frame: x right, y down, z forward. Parameters [fx, fy, cx, cy, k1, k2, k3, k4], all calibrated (none is
 * refined online). A point at angle theta from the optical axis (theta = atan2(sqrt(X^2 + Y^2), Z)) is imaged at
 * radius theta_d = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8) in its azimuth:
 *   x_d = theta_d X / r,  y_d = theta_d Y / r  (r = sqrt(X^2 + Y^2)),  u = fx x_d + cx,  v = fy y_d + cy
 * (Kannala and Brandt 2006; see doc/references.md). Points beyond 90 degrees (Z < 0) are imaged as long as the lens
 * sees them; the ray of a pixel inverts the polynomial by Newton steps.
 */

#pragma once

#include "vio/sensor/camera_model.h"

#include <Eigen/Core>
#include <opencv2/core/types.hpp>

namespace vio {

struct FisheyeEquidistantCameraModel : public CameraModelBase<FisheyeEquidistantCameraModel> {
  static constexpr CameraModelId kId = CameraModelId::kFisheyeEquidistant;
  static constexpr const char* kName = "FISHEYE_EQUIDISTANT";
  static constexpr int kNumParameters = 8;           // fx, fy, cx, cy, k1, k2, k3, k4
  static constexpr int kNumOnlineParams = 0;         // All calibrated
  static constexpr int kNumSensors = 1;
  static constexpr bool kWrapsHorizontally = false;

  // No online corrections: the calibrated parameters
  static void PlainParams(double* params) { (void)params; }

  // ============ Estimation (double precision) ============

  /**
   * @brief Pixel of a point in the camera frame; (-1, -1), outside the image, for a point straight behind
   * @param d_pixel_d_params Output d pixel / d params [2 x 8, column-major] (null: not computed)
   */
  static Eigen::Vector2d Project(int width, int height, const double* params, const Eigen::Vector3d& point,
                                 double* d_pixel_d_params);

  // Unit ray of a pixel
  static Eigen::Vector3d Unproject(int width, int height, const double* params, const Eigen::Vector2d& pixel);

  /**
   * @brief d pixel / d point
   * @return false for a point straight behind the camera
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

  // Geometric mean of the focal lengths (the projection is equidistant at the center)
  static double PixelsPerRadian(int width, int height, const double* params);

  // ============ Tracking front end (single precision) ============

  static Eigen::Vector3f PixelToRay(int width, int height, const double* params, const cv::Point2f& pixel);
  static cv::Point2f RayToPixel(int width, int height, const double* params, const Eigen::Vector3f& ray);
};

}  // namespace vio

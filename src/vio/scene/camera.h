// SPDX-License-Identifier: MIT
/**
 * @file      camera.h
 * @brief     Camera: model, image size, parameters and the image regions that are never tracked.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include "vio/sensor/camera_model.h"

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <vector>

namespace vio {

struct Camera {
  CameraModelId model_id = CameraModelId::kInvalid;
  int width = 0;                           // Image width [px]
  int height = 0;                          // Image height [px]
  std::vector<double> params;              // Model parameters (see the model, e.g. fx fy cx cy k1 k2 p1 p2)

  // Image regions that are never tracked
  float polar_exclusion_ratio = 0.0f;      // Image-height fraction excluded at the top and at the bottom
  int border_margin = 0;                   // Margin at the image edges [px]
  std::vector<cv::Vec4f> mask_rectangles;  // Excluded rectangles [x0, x1, y0, y1] as image fractions

  bool Check() const;

  // ============ Model properties ============

  int NumOnlineParams() const { return CameraModelNumOnlineParams(model_id); }
  int NumSensors() const { return CameraModelNumSensors(model_id); }
  bool WrapsHorizontally() const { return CameraModelWrapsHorizontally(model_id); }

  // The camera without the online corrections of its model
  Camera Plain() const {
    Camera plain = *this;
    CameraModelPlainParams(model_id, &plain.params);
    return plain;
  }

  // ============ Estimation (double precision) ============

  /**
   * @brief Pixel of a point in the camera frame
   * @param d_pixel_d_params Output 2 x params.size() derivative, column-major (null: not computed)
   */
  Eigen::Vector2d Project(const Eigen::Vector3d& point, double* d_pixel_d_params = nullptr) const {
    return CameraModelProject(model_id, width, height, params, point, d_pixel_d_params);
  }
  Eigen::Vector3d Unproject(const Eigen::Vector2d& pixel) const {
    return CameraModelUnproject(model_id, width, height, params, pixel);
  }
  bool ProjectionJacobian(const Eigen::Vector3d& point, Eigen::Matrix<double, 2, 3>* J) const {
    return CameraModelProjectionJacobian(model_id, width, height, params, point, J);
  }
  Eigen::Vector2d Residual(const Eigen::Vector2d& measured, const Eigen::Vector2d& predicted) const {
    return CameraModelResidual(model_id, width, height, measured, predicted);
  }
  int SensorOf(const Eigen::Vector3d& ray) const { return CameraModelSensorOf(model_id, ray); }
  double PixelsPerRadian() const { return CameraModelPixelsPerRadian(model_id, width, height, params); }

  // ============ Tracking front end (single precision) ============

  Eigen::Vector3f PixelToRay(const cv::Point2f& pixel) const {
    return CameraModelPixelToRay(model_id, width, height, params, pixel);
  }
  cv::Point2f RayToPixel(const Eigen::Vector3f& ray) const {
    return CameraModelRayToPixel(model_id, width, height, params, ray);
  }

  // In a polar band or a mask rectangle
  bool IsMasked(const cv::Point2f& pixel) const;
  // Within the border margin of an image edge
  bool IsNearBorder(const cv::Point2f& pixel) const;
  bool IsTrackable(const cv::Point2f& pixel) const { return !IsMasked(pixel) && !IsNearBorder(pixel); }
};

}  // namespace vio

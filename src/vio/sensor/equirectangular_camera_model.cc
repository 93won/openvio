// SPDX-License-Identifier: MIT
/**
 * @file      equirectangular_camera_model.cc
 * @brief     Implements the equirectangular camera model.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "vio/sensor/equirectangular_camera_model.h"

#include <Eigen/Dense>

#include <cmath>

namespace vio {
namespace {

// Plain projection (k = 0)
Eigen::Vector2d PlainProject(double width, double height, const Eigen::Vector3d& p_C) {
  const double lon = std::atan2(p_C.x(), p_C.z());
  const double lat = std::asin(p_C.y() / p_C.norm());
  return Eigen::Vector2d(width * (0.5 + lon / (2.0 * M_PI)), height * (0.5 + lat / M_PI));
}

// Plain unit ray (k = 0)
Eigen::Vector3d PlainUnproject(double width, double height, const Eigen::Vector2d& uv) {
  const double lon = (uv.x() / width - 0.5) * 2.0 * M_PI;
  const double lat = -(uv.y() / height - 0.5) * M_PI;
  return Eigen::Vector3d(std::cos(lat) * std::sin(lon), -std::sin(lat), std::cos(lat) * std::cos(lon));
}

// Horizontal pixel difference wrapped into [-W/2, W/2]
double WrapU(double width, double du) {
  if (du > 0.5 * width) {
    return du - width;
  }
  if (du < -0.5 * width) {
    return du + width;
  }
  return du;
}

}  // namespace

Eigen::Vector2d EquirectangularCameraModel::Project(int width, int height, const double* params,
                                                    const Eigen::Vector3d& point, double* d_pixel_d_params) {
  const double k = params[0];
  const double radial = std::hypot(point.x(), point.y());
  if ((k == 0.0 && d_pixel_d_params == nullptr) || radial < 1e-12 * point.norm()) {
    if (d_pixel_d_params != nullptr) {
      d_pixel_d_params[0] = 0.0;
      d_pixel_d_params[1] = 0.0;
    }
    return PlainProject(width, height, point);
  }
  // Angle theta from the lens axis (+z front, -z back), imaged at t = theta + k g(theta), same azimuth
  const double theta = std::atan2(radial, std::abs(point.z()));
  const double s = 2.0 * theta / M_PI;
  const double g = theta * (1.0 - s * s);
  const double t = theta + k * g;
  const double axis = (point.z() >= 0.0) ? 1.0 : -1.0;
  const double cx = point.x() / radial;
  const double cy = point.y() / radial;
  const Eigen::Vector3d b(std::sin(t) * cx, std::sin(t) * cy, axis * std::cos(t));
  if (d_pixel_d_params != nullptr) {
    Eigen::Vector2d dz_dk;
    Eigen::Matrix<double, 2, 3> J;
    dz_dk.setZero();
    if (ProjectionJacobian(width, height, params, b, &J)) {
      dz_dk = J * Eigen::Vector3d(std::cos(t) * cx, std::cos(t) * cy, -axis * std::sin(t)) * g;
    }
    d_pixel_d_params[0] = dz_dk.x();
    d_pixel_d_params[1] = dz_dk.y();
  }
  return PlainProject(width, height, b);
}

Eigen::Vector3d EquirectangularCameraModel::Unproject(int width, int height, const double* params,
                                                      const Eigen::Vector2d& pixel) {
  const double k = params[0];
  const Eigen::Vector3d b = PlainUnproject(width, height, pixel);
  const double radial = std::hypot(b.x(), b.y());
  if (k == 0.0 || radial < 1e-12) {
    return b;
  }
  // Newton steps on theta + k g(theta) = measured angle
  const double measured = std::atan2(radial, std::abs(b.z()));
  double theta = measured;
  for (int iteration = 0; iteration < 3; ++iteration) {
    const double s = 2.0 * theta / M_PI;
    theta -= (theta + k * theta * (1.0 - s * s) - measured) / (1.0 + k * (1.0 - 3.0 * s * s));
  }
  const double scale = std::sin(theta) / radial;
  return Eigen::Vector3d(b.x() * scale, b.y() * scale, std::copysign(std::cos(theta), b.z()));
}

bool EquirectangularCameraModel::ProjectionJacobian(int width, int height, const double* params,
                                                    const Eigen::Vector3d& point, Eigen::Matrix<double, 2, 3>* J) {
  (void)params;  // Plain projection (k = 0)
  const double x = point.x();
  const double y = point.y();
  const double z = point.z();
  const double rho2 = x * x + z * z;
  const double L2 = rho2 + y * y;
  if (rho2 < 1e-10 || L2 < 1e-10) {
    return false;
  }

  // d atan2(x, z) = (z dx - x dz) / rho^2
  // d asin(y / L) = (rho^2 dy - y (x dx + z dz)) / (rho L^2)
  const double rho = std::sqrt(rho2);
  const double ku = width / (2.0 * M_PI);
  const double kv = height / M_PI;
  *J << ku * z / rho2, 0.0, -ku * x / rho2,
        -kv * x * y / (rho * L2), kv * rho / L2, -kv * y * z / (rho * L2);
  return true;
}

Eigen::Vector2d EquirectangularCameraModel::Residual(int width, int height, const Eigen::Vector2d& measured,
                                                     const Eigen::Vector2d& predicted) {
  (void)height;
  return Eigen::Vector2d(WrapU(width, measured.x() - predicted.x()), measured.y() - predicted.y());
}

double EquirectangularCameraModel::PixelsPerRadian(int width, int height, const double* params) {
  (void)height;
  (void)params;
  return width / (2.0 * M_PI);
}

Eigen::Vector3f EquirectangularCameraModel::PixelToRay(int width, int height, const double* params,
                                                       const cv::Point2f& pixel) {
  (void)params;  // Plain projection (k = 0)
  // Normalize pixel coordinates
  float u_norm = pixel.x / static_cast<float>(width);   // [0, 1]
  float v_norm = pixel.y / static_cast<float>(height);  // [0, 1]

  // Longitude: angle from z towards x; latitude: angle from the xz plane, positive up (y points down)
  float lon = (u_norm - 0.5f) * 2.0f * M_PI;  // [-pi, pi]
  float lat = -(v_norm - 0.5f) * M_PI;        // [pi/2, -pi/2]

  float cos_lat = std::cos(lat);
  Eigen::Vector3f ray;
  ray.x() = cos_lat * std::sin(lon);
  ray.y() = -std::sin(lat);
  ray.z() = cos_lat * std::cos(lon);
  ray.normalize();
  return ray;
}

cv::Point2f EquirectangularCameraModel::RayToPixel(int width, int height, const double* params,
                                                   const Eigen::Vector3f& ray) {
  (void)params;  // Plain projection (k = 0)
  float theta = std::atan2(ray.x(), ray.z());     // [-pi, pi]
  float phi = -std::asin(ray.y() / ray.norm());   // [-pi/2, pi/2]
  cv::Point2f pixel;
  pixel.x = static_cast<float>(width) * (0.5f + theta / (2.0f * M_PI));
  pixel.y = static_cast<float>(height) * (0.5f - phi / M_PI);
  return pixel;
}

}  // namespace vio

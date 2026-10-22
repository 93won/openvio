// SPDX-License-Identifier: MIT
/**
 * @file      fisheye_equidistant_camera_model.cc
 * @brief     Implements the fisheye camera with the equidistant projection.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2026-10-06
 * @copyright Copyright (c) 2026 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "vio/sensor/fisheye_equidistant_camera_model.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>

namespace vio {
namespace {

// theta_d(theta) and its derivative
double Distort(const double* params, double theta, double* derivative) {
  const double t2 = theta * theta;
  const double poly = 1.0 + t2 * (params[4] + t2 * (params[5] + t2 * (params[6] + t2 * params[7])));
  if (derivative != nullptr) {
    *derivative = 1.0 + t2 * (3.0 * params[4] + t2 * (5.0 * params[5] + t2 * (7.0 * params[6] + t2 * 9.0 * params[7])));
  }
  return theta * poly;
}

// A point this close to the optical axis (relative to its distance) uses the limit of the projection on the axis
constexpr double kOnAxis = 1e-9;

}  // namespace

Eigen::Vector2d FisheyeEquidistantCameraModel::Project(int width, int height, const double* params,
                                                       const Eigen::Vector3d& point, double* d_pixel_d_params) {
  (void)width;
  (void)height;
  const double r = std::hypot(point.x(), point.y());
  const double norm = point.norm();
  if (!(norm > 0.0) || (r <= kOnAxis * norm && point.z() <= 0.0)) {
    if (d_pixel_d_params != nullptr) {
      std::fill(d_pixel_d_params, d_pixel_d_params + 2 * kNumParameters, 0.0);
    }
    return Eigen::Vector2d(-1.0, -1.0);
  }
  double x_d = point.x() / point.z();
  double y_d = point.y() / point.z();
  double theta = 0.0;
  if (r > kOnAxis * norm) {
    theta = std::atan2(r, point.z());
    const double scale = Distort(params, theta, nullptr) / r;
    x_d = scale * point.x();
    y_d = scale * point.y();
  }
  if (d_pixel_d_params != nullptr) {
    Eigen::Map<Eigen::Matrix<double, 2, kNumParameters>> D(d_pixel_d_params);
    D.setZero();
    D(0, 0) = x_d;  // fx
    D(1, 1) = y_d;  // fy
    D(0, 2) = 1.0;  // cx
    D(1, 3) = 1.0;  // cy
    if (r > kOnAxis * norm) {
      // d theta_d / d k_i = theta^(2i + 1), in the azimuth (X, Y) / r
      double power = theta * theta * theta;
      for (int i = 0; i < 4; ++i) {
        D(0, 4 + i) = params[0] * power * point.x() / r;
        D(1, 4 + i) = params[1] * power * point.y() / r;
        power *= theta * theta;
      }
    }
  }
  return Eigen::Vector2d(params[0] * x_d + params[2], params[1] * y_d + params[3]);
}

Eigen::Vector3d FisheyeEquidistantCameraModel::Unproject(int width, int height, const double* params,
                                                         const Eigen::Vector2d& pixel) {
  (void)width;
  (void)height;
  const double x_d = (pixel.x() - params[2]) / params[0];
  const double y_d = (pixel.y() - params[3]) / params[1];
  const double theta_d = std::hypot(x_d, y_d);
  if (theta_d < 1e-12) {
    return Eigen::Vector3d(0.0, 0.0, 1.0);
  }
  double theta = theta_d;
  for (int iteration = 0; iteration < 20; ++iteration) {
    double derivative = 1.0;
    const double step = (Distort(params, theta, &derivative) - theta_d) / derivative;
    theta -= step;
    if (std::abs(step) < 1e-14) {
      break;
    }
  }
  const double s = std::sin(theta) / theta_d;
  return Eigen::Vector3d(s * x_d, s * y_d, std::cos(theta));
}

bool FisheyeEquidistantCameraModel::ProjectionJacobian(int width, int height, const double* params,
                                                       const Eigen::Vector3d& point, Eigen::Matrix<double, 2, 3>* J) {
  (void)width;
  (void)height;
  const double X = point.x();
  const double Y = point.y();
  const double Z = point.z();
  const double r = std::hypot(X, Y);
  const double norm = point.norm();
  if (!(norm > 0.0)) {
    return false;
  }
  Eigen::Matrix<double, 2, 3> J_d;  // d (x_d, y_d) / d point
  if (r <= kOnAxis * norm) {
    if (Z <= 0.0) {
      return false;
    }
    // On the axis the projection is pinhole to first order
    J_d << 1.0 / Z, 0.0, -X / (Z * Z),
           0.0, 1.0 / Z, -Y / (Z * Z);
  } else {
    // x_d = a X, y_d = a Y with a = theta_d(theta) / r and theta = atan2(r, Z)
    const double rho2 = r * r + Z * Z;
    const double theta = std::atan2(r, Z);
    double derivative = 1.0;
    const double theta_d = Distort(params, theta, &derivative);
    const double a = theta_d / r;
    const double c = (derivative * Z / rho2 - a) / (r * r);  // d a / dX = X c, d a / dY = Y c
    const double dz = -derivative / rho2;                    // d a / dZ
    J_d << a + X * X * c, X * Y * c, X * dz,
           X * Y * c, a + Y * Y * c, Y * dz;
  }
  *J = Eigen::Vector2d(params[0], params[1]).asDiagonal() * J_d;
  return true;
}

double FisheyeEquidistantCameraModel::PixelsPerRadian(int width, int height, const double* params) {
  (void)width;
  (void)height;
  return std::sqrt(params[0] * params[1]);
}

Eigen::Vector3f FisheyeEquidistantCameraModel::PixelToRay(int width, int height, const double* params,
                                                          const cv::Point2f& pixel) {
  return Unproject(width, height, params, Eigen::Vector2d(pixel.x, pixel.y)).cast<float>();
}

cv::Point2f FisheyeEquidistantCameraModel::RayToPixel(int width, int height, const double* params,
                                                      const Eigen::Vector3f& ray) {
  const Eigen::Vector2d uv = Project(width, height, params, ray.cast<double>(), nullptr);
  return cv::Point2f(static_cast<float>(uv.x()), static_cast<float>(uv.y()));
}

}  // namespace vio

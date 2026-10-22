// SPDX-License-Identifier: MIT
/**
 * @file      pinhole_radtan_camera_model.cc
 * @brief     Implements the pinhole camera with radial and tangential lens distortion.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2026-10-06
 * @copyright Copyright (c) 2026 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "vio/sensor/pinhole_radtan_camera_model.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>

namespace vio {
namespace {

// Distorted normalized point of (x, y) and its derivative d (x_d, y_d) / d (x, y)
Eigen::Vector2d Distort(const double* params, double x, double y, Eigen::Matrix2d* J) {
  const double k1 = params[4];
  const double k2 = params[5];
  const double p1 = params[6];
  const double p2 = params[7];
  const double x2 = x * x;
  const double y2 = y * y;
  const double xy = x * y;
  const double r2 = x2 + y2;
  const double radial = 1.0 + r2 * (k1 + k2 * r2);
  if (J != nullptr) {
    // d radial / dx = x * g, d radial / dy = y * g
    const double g = 2.0 * (k1 + 2.0 * k2 * r2);
    (*J)(0, 0) = radial + x2 * g + 2.0 * p1 * y + 6.0 * p2 * x;
    (*J)(0, 1) = xy * g + 2.0 * p1 * x + 2.0 * p2 * y;
    (*J)(1, 0) = (*J)(0, 1);
    (*J)(1, 1) = radial + y2 * g + 6.0 * p1 * y + 2.0 * p2 * x;
  }
  return Eigen::Vector2d(x * radial + 2.0 * p1 * xy + p2 * (r2 + 2.0 * x2),
                         y * radial + p1 * (r2 + 2.0 * y2) + 2.0 * p2 * xy);
}

}  // namespace

Eigen::Vector2d PinholeRadtanCameraModel::Project(int width, int height, const double* params,
                                                  const Eigen::Vector3d& point, double* d_pixel_d_params) {
  (void)width;
  (void)height;
  if (!(point.z() > 0.0)) {
    if (d_pixel_d_params != nullptr) {
      std::fill(d_pixel_d_params, d_pixel_d_params + 2 * kNumParameters, 0.0);
    }
    return Eigen::Vector2d(-1.0, -1.0);
  }
  const double fx = params[0];
  const double fy = params[1];
  const double x = point.x() / point.z();
  const double y = point.y() / point.z();
  const Eigen::Vector2d d = Distort(params, x, y, nullptr);
  if (d_pixel_d_params != nullptr) {
    const double r2 = x * x + y * y;
    Eigen::Map<Eigen::Matrix<double, 2, kNumParameters>> D(d_pixel_d_params);
    D.setZero();
    D(0, 0) = d.x();                      // fx
    D(1, 1) = d.y();                      // fy
    D(0, 2) = 1.0;                        // cx
    D(1, 3) = 1.0;                        // cy
    D(0, 4) = fx * x * r2;                // k1
    D(1, 4) = fy * y * r2;
    D(0, 5) = fx * x * r2 * r2;           // k2
    D(1, 5) = fy * y * r2 * r2;
    D(0, 6) = fx * 2.0 * x * y;           // p1
    D(1, 6) = fy * (r2 + 2.0 * y * y);
    D(0, 7) = fx * (r2 + 2.0 * x * x);    // p2
    D(1, 7) = fy * 2.0 * x * y;
  }
  return Eigen::Vector2d(fx * d.x() + params[2], fy * d.y() + params[3]);
}

Eigen::Vector3d PinholeRadtanCameraModel::Unproject(int width, int height, const double* params,
                                                    const Eigen::Vector2d& pixel) {
  (void)width;
  (void)height;
  const Eigen::Vector2d target((pixel.x() - params[2]) / params[0], (pixel.y() - params[3]) / params[1]);
  Eigen::Vector2d u = target;
  for (int iteration = 0; iteration < 20; ++iteration) {
    Eigen::Matrix2d J;
    const Eigen::Vector2d e = Distort(params, u.x(), u.y(), &J) - target;
    const Eigen::Vector2d step = J.partialPivLu().solve(e);
    u -= step;
    if (step.squaredNorm() < 1e-24) {
      break;
    }
  }
  return Eigen::Vector3d(u.x(), u.y(), 1.0).normalized();
}

bool PinholeRadtanCameraModel::ProjectionJacobian(int width, int height, const double* params,
                                                  const Eigen::Vector3d& point, Eigen::Matrix<double, 2, 3>* J) {
  (void)width;
  (void)height;
  if (!(point.z() > 1e-9 * point.norm())) {
    return false;
  }
  const double iz = 1.0 / point.z();
  const double x = point.x() * iz;
  const double y = point.y() * iz;
  Eigen::Matrix2d J_distort;
  Distort(params, x, y, &J_distort);
  Eigen::Matrix<double, 2, 3> J_normalized;  // d (x, y) / d point
  J_normalized << iz, 0.0, -x * iz,
                  0.0, iz, -y * iz;
  *J = Eigen::Vector2d(params[0], params[1]).asDiagonal() * J_distort * J_normalized;
  return true;
}

double PinholeRadtanCameraModel::PixelsPerRadian(int width, int height, const double* params) {
  (void)width;
  (void)height;
  return std::sqrt(params[0] * params[1]);
}

Eigen::Vector3f PinholeRadtanCameraModel::PixelToRay(int width, int height, const double* params,
                                                     const cv::Point2f& pixel) {
  return Unproject(width, height, params, Eigen::Vector2d(pixel.x, pixel.y)).cast<float>();
}

cv::Point2f PinholeRadtanCameraModel::RayToPixel(int width, int height, const double* params,
                                                 const Eigen::Vector3f& ray) {
  const Eigen::Vector2d uv = Project(width, height, params, ray.cast<double>(), nullptr);
  return cv::Point2f(static_cast<float>(uv.x()), static_cast<float>(uv.y()));
}

}  // namespace vio

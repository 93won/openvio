// SPDX-License-Identifier: MIT
/**
 * @file      startup_calibration.cc
 * @brief     Implements the startup calibration.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2026-09-28
 * @copyright Copyright (c) 2026 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "vio/estimators/startup_calibration.h"

#include "vio/geometry/lie_group.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>

namespace vio {
namespace {

// Rotation R minimizing sum w |b - R a|^2 over unit rays, Huber-reweighted at ~1 px
Eigen::Matrix3d RobustRotation(const std::vector<Eigen::Vector3d>& a, const std::vector<Eigen::Vector3d>& b) {
  constexpr double kHuber = 0.005;  // [rad]
  std::vector<double> w(a.size(), 1.0);
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  for (int iter = 0; iter < 6; ++iter) {
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    for (size_t k = 0; k < a.size(); ++k) {
      H += w[k] * a[k] * b[k].transpose();
    }
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
    D(2, 2) = (svd.matrixV() * svd.matrixU().transpose()).determinant() < 0.0 ? -1.0 : 1.0;
    R = svd.matrixV() * D * svd.matrixU().transpose();
    for (size_t k = 0; k < a.size(); ++k) {
      const double r = (b[k] - R * a[k]).norm();
      w[k] = (r < kHuber) ? 1.0 : kHuber / r;
    }
  }
  return R;
}

}  // namespace

StartupCalibration::StartupCalibration(const Camera& camera, const Rig& rig)
    : camera_(camera), R_ItoC_(rig.R_ItoC), prev_time_(0.0) {}

void StartupCalibration::AddImu(const ImuSample& sample) {
  gyro_.emplace_back(sample.timestamp, sample.gyro);
  accel_.emplace_back(sample.timestamp, sample.accel);
}

void StartupCalibration::AddImage(Timestamp timestamp, const std::vector<PixelObservation>& observations) {
  std::map<int, Eigen::Vector3d> rays;
  for (const auto& observation : observations) {
    rays[observation.id] = camera_.Unproject(observation.pixel);
  }
  if (!prev_rays_.empty()) {
    constexpr size_t kMinTracks = 15;
    std::vector<Eigen::Vector3d> now[2], prev[2];
    double angle_sum = 0.0;
    for (const auto& entry : rays) {
      const auto it = prev_rays_.find(entry.first);
      if (it != prev_rays_.end()) {
        const int sensor = camera_.SensorOf(entry.second);
        now[sensor].push_back(entry.second);
        prev[sensor].push_back(it->second);
        angle_sum += std::acos(std::min(1.0, entry.second.dot(it->second)));
      }
    }

    // Mean track displacement of the image [rad] (rest check of the accel-bias estimate)
    const size_t matched = now[0].size() + now[1].size();
    if (matched >= kMinTracks) {
      motion_.emplace_back(timestamp, angle_sum / static_cast<double>(matched));
    }
    SensorRotation rotation;
    rotation.t0 = prev_time_;
    rotation.t1 = timestamp;
    for (int sensor = 0; sensor < 2; ++sensor) {
      rotation.valid[sensor] = sensor < camera_.NumSensors() && now[sensor].size() >= kMinTracks;
      rotation.theta[sensor].setZero();
      if (rotation.valid[sensor]) {
        // Camera rotation from t0 to t1 (prev = R now), expressed in the IMU frame
        const Eigen::Matrix3d R = RobustRotation(now[sensor], prev[sensor]);
        rotation.theta[sensor] = R_ItoC_.transpose() * LogSO3(R);
      }
    }
    rotations_.push_back(rotation);
  }
  prev_rays_.swap(rays);
  prev_time_ = timestamp;
}

bool StartupCalibration::EstimateSensorTimeOffsets(double* offset0, double* offset1) const {
  constexpr size_t kMinPairs = 60;       // Two seconds of image pairs
  constexpr double kMaxOffset = 0.060;   // Search range [s]
  constexpr double kStep = 0.0005;       // Grid step [s]
  constexpr double kMaxSigma = 0.003;    // Accepted standard error [s]
  *offset0 = 0.0;
  *offset1 = 0.0;
  if (gyro_.size() < 2) {
    return false;
  }

  // Cumulative gyro rotation (sample i held over [t_i, t_i+1)) and the body rotation between two times
  std::vector<Eigen::Matrix3d> cumulative(gyro_.size(), Eigen::Matrix3d::Identity());
  for (size_t i = 1; i < gyro_.size(); ++i) {
    cumulative[i] = cumulative[i - 1] * ExpSO3(gyro_[i - 1].second * (gyro_[i].first - gyro_[i - 1].first));
  }
  auto rotation_at = [&](double t) -> Eigen::Matrix3d {
    size_t i = std::upper_bound(gyro_.begin(), gyro_.end(), t,
                                [](double x, const std::pair<double, Eigen::Vector3d>& s) { return x < s.first; }) -
               gyro_.begin();
    i = (i == 0) ? 0 : i - 1;
    return cumulative[i] * ExpSO3(gyro_[i].second * (t - gyro_[i].first));
  };
  auto gyro_theta = [&](double t0, double t1) { return LogSO3(rotation_at(t0).transpose() * rotation_at(t1)); };

  double offsets[2] = {0.0, 0.0};
  for (int sensor = 0; sensor < camera_.NumSensors(); ++sensor) {
    std::vector<const SensorRotation*> pairs;
    for (const auto& rotation : rotations_) {
      if (rotation.valid[sensor] && rotation.t0 > gyro_.front().first + kMaxOffset &&
          rotation.t1 < gyro_.back().first - kMaxOffset) {
        pairs.push_back(&rotation);
      }
    }
    if (pairs.size() < kMinPairs) {
      return false;
    }

    // Pairs with a residual above 5x the median at dt = 0 are outliers
    std::vector<double> r0;
    for (const SensorRotation* p : pairs) {
      r0.push_back((gyro_theta(p->t0, p->t1) - p->theta[sensor]).norm());
    }
    std::vector<double> sorted = r0;
    std::nth_element(sorted.begin(), sorted.begin() + static_cast<long>(sorted.size() / 2), sorted.end());
    const double gate = 5.0 * sorted[sorted.size() / 2];

    // Squared residual with a constant gyro bias solved at every dt
    std::vector<double> costs;
    int used = 0;
    for (double dt = -kMaxOffset; dt <= kMaxOffset + 1e-9; dt += kStep) {
      std::vector<Eigen::Vector3d> e;
      std::vector<double> d;
      Eigen::Vector3d sum = Eigen::Vector3d::Zero();
      double sum_t = 0.0;
      for (size_t k = 0; k < pairs.size(); ++k) {
        if (r0[k] > gate) {
          continue;
        }
        e.push_back(gyro_theta(pairs[k]->t0 + dt, pairs[k]->t1 + dt) - pairs[k]->theta[sensor]);
        d.push_back(pairs[k]->t1 - pairs[k]->t0);
        sum += e.back();
        sum_t += d.back();
      }
      const Eigen::Vector3d bias = sum / sum_t;
      double cost = 0.0;
      for (size_t k = 0; k < e.size(); ++k) {
        cost += (e[k] - bias * d[k]).squaredNorm();
      }
      costs.push_back(cost);
      used = static_cast<int>(e.size());
    }

    // Interior minimum, parabolic refinement, standard error from the curvature
    const size_t j = static_cast<size_t>(std::min_element(costs.begin(), costs.end()) - costs.begin());
    if (j == 0 || j + 1 >= costs.size()) {
      return false;
    }
    const double curvature = (costs[j + 1] - 2.0 * costs[j] + costs[j - 1]) / (kStep * kStep);
    if (curvature <= 0.0) {
      return false;
    }
    const double variance = costs[j] / (3.0 * used);
    const double sigma = std::sqrt(2.0 * variance / curvature);
    offsets[sensor] = -kMaxOffset + kStep * static_cast<double>(j) +
                      0.5 * kStep * (costs[j - 1] - costs[j + 1]) / (costs[j + 1] - 2.0 * costs[j] + costs[j - 1]);
    if (!(sigma < kMaxSigma)) {
      return false;
    }
  }
  *offset0 = offsets[0];
  *offset1 = offsets[1];
  return true;
}

bool StartupCalibration::EstimateRestAccelBias(double gravity_magnitude, double velocity_up,
                                               Eigen::Vector3d* direction, double* bias, double* elapsed) const {
  if (accel_.empty() || motion_.empty()) {
    return false;
  }
  constexpr double kRestCheck = 1.0;  // Duration of the rest check [s]
  const double t0 = accel_.front().first;
  std::vector<double> start;
  for (const auto& motion : motion_) {
    if (motion.first <= t0 + kRestCheck) {
      start.push_back(motion.second);
    }
  }
  const double half_pixel = 0.5 / camera_.PixelsPerRadian();
  std::nth_element(start.begin(), start.begin() + static_cast<long>(start.size() / 2), start.end());
  if (!(start.size() >= 10 && start[start.size() / 2] < half_pixel)) {
    return false;
  }
  Eigen::Vector3d f = Eigen::Vector3d::Zero();
  for (const auto& sample : accel_) {
    f += sample.second;
  }
  f /= static_cast<double>(accel_.size());
  *elapsed = accel_.back().first - t0;
  *bias = f.norm() - gravity_magnitude - velocity_up / *elapsed;
  *direction = f.normalized();
  return true;
}

}  // namespace vio

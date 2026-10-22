// SPDX-License-Identifier: MIT
/**
 * @file      imu_propagation.cc
 * @brief     Estimator: IMU propagation (RK4 mean, closed-form transition and noise), cloning and sensor timing.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2026-10-06
 * @copyright Copyright (c) 2026 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 *
 * References:
 *   [1] N. Trawny, S. I. Roumeliotis, "Indirect Kalman Filter for 3D Attitude Estimation", Tech. Rep. 2005-002,
 *       University of Minnesota, 2005 (JPL quaternion kinematics dq/dt = 1/2 Omega(w) q).
 *   [2] E. Hairer, S. P. Norsett, G. Wanner, "Solving Ordinary Differential Equations I", 2nd ed., 1993, Ch. II
 *       (classical fourth-order Runge-Kutta method).
 *   [3] Y. Yang, B. P. W. Babu, C. Chen, G. Huang, L. Ren, "Analytic Combined IMU Integration (ACI^2) for Visual
 *       Inertial Navigation", ICRA 2020 (closed-form integrals Xi_1..Xi_4 over an interval with constant rate).
 *   [4] M. Li, A. I. Mourikis, "High-precision, consistent EKF-based visual-inertial odometry", IJRR 2013
 *       (error-state transition evaluated at first estimates).
 *   [5] A. I. Mourikis, S. I. Roumeliotis, "A Multi-State Constraint Kalman Filter for Vision-aided Inertial
 *       Navigation", ICRA 2007, Sec. III-B, III-C (covariance propagation, state augmentation with a clone).
 *   [6] M. Li, A. I. Mourikis, "Online temporal calibration for camera-IMU systems: Theory and algorithms", IJRR 2014
 *       (clone Jacobian w.r.t. the camera-IMU time offset).
 */

#include "vio/estimators/estimator.h"

#include "vio/geometry/lie_group.h"
#include "vio/sensor/imu_interval.h"

#include <algorithm>
#include <cmath>

namespace vio {
namespace {

using Matrix15d = Eigen::Matrix<double, 15, 15>;

struct RotationIntegrals {
  Eigen::Matrix3d Xi1;
  Eigen::Matrix3d Xi2;
  Eigen::Matrix3d Xi3;
  Eigen::Matrix3d Xi4;
};

/**
 * Closed-form integrals of one interval dt with constant body rate w and specific force a ([3]). With
 * theta = |w| dt, k = w / |w|, K = [k]x, A = [a]x, c = cos(theta), s = sin(theta):
 *   Xi1 = int_0^dt Exp(w t) dt = dt I + (1 - c) / |w| K + (dt - s / |w|) K^2,
 *   Xi2 = int_0^dt int_0^t Exp(w u) du dt = dt^2 / 2 I + (theta - s) / |w|^2 K + (dt^2 / 2 - (1 - c) / |w|^2) K^2,
 *   Xi3, Xi4: the single and double integrals of Exp(w t) [a]x Jr(w t) t, which carry the gyro-bias error through
 *   the rotated specific force.
 * Below 0.5 deg/s the inverse powers of |w| lose precision and the interval is short against the rotation:
 * Xi1 = dt Exp(w dt), Xi2 = dt Xi1 / 2, Xi4 = dt Xi3 / 3, and Xi3 from its leading terms.
 */
RotationIntegrals IntegrateRotation(double dt, const Eigen::Vector3d& w, const Eigen::Vector3d& a) {
  constexpr double kSmallRate = M_PI / 360.0;  // 0.5 deg/s

  const double rate = w.norm();
  const Eigen::Vector3d k = (rate > 1e-12) ? Eigen::Vector3d(w / rate) : Eigen::Vector3d::Zero();
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d K = Skew(k);
  const Eigen::Matrix3d K2 = K * K;
  const Eigen::Matrix3d A = Skew(a);
  const double ka = k.dot(a);
  const double theta = rate * dt;
  const double c = std::cos(theta);
  const double s = std::sin(theta);
  const double dt2 = dt * dt;
  // Products shared by Xi3 and Xi4
  const Eigen::Matrix3d AK = A * K;
  const Eigen::Matrix3d KA = K * A;
  const Eigen::Matrix3d AK2 = A * K2;
  const Eigen::Matrix3d K2A_kaK = K2 * A + ka * K;
  const Eigen::Matrix3d kaK2 = ka * K2;

  RotationIntegrals xi;
  if (rate < kSmallRate) {
    xi.Xi1 = dt * (I + s * K + (1.0 - c) * K2);
    xi.Xi2 = 0.5 * dt * xi.Xi1;
    xi.Xi3 = 0.5 * dt2 * (A + s * (-AK + KA + kaK2) + (1.0 - c) * (AK2 + K2A_kaK));
    xi.Xi4 = (dt / 3.0) * xi.Xi3;
    return xi;
  }
  const double rate2 = rate * rate;
  const double rate3 = rate2 * rate;
  const double theta2 = theta * theta;
  xi.Xi1 = dt * I + ((1.0 - c) / rate) * K + (dt - s / rate) * K2;
  xi.Xi2 = 0.5 * dt2 * I + ((theta - s) / rate2) * K + (0.5 * dt2 - (1.0 - c) / rate2) * K2;
  xi.Xi3 = 0.5 * dt2 * A + ((s - theta) / rate2) * AK + ((s - theta * c) / rate2) * KA +
           (0.5 * dt2 - (1.0 - c) / rate2) * AK2 + (0.5 * dt2 + (1.0 - c - theta * s) / rate2) * K2A_kaK -
           ((3.0 * s - 2.0 * theta - theta * c) / rate2) * kaK2;
  xi.Xi4 = (dt2 * dt / 6.0) * A + ((2.0 * (1.0 - c) - theta2) / (2.0 * rate3)) * AK +
           ((2.0 * (1.0 - c) - theta * s) / rate3) * KA + ((s - theta) / rate3 + dt2 * dt / 6.0) * AK2 +
           ((theta - 2.0 * s + theta2 * theta / 6.0 + theta * c) / rate3) * K2A_kaK +
           ((4.0 * c - 4.0 + theta2 + theta * s) / rate3) * kaK2;
  return xi;
}

struct ImuMean {
  Eigen::Vector4d q;  // q_GtoI
  Eigen::Vector3d p;  // p_IinG
  Eigen::Vector3d v;  // v_IinG
};

/**
 * Mean of the IMU state over one interval ([1], [2]). Rate w and specific force a are linear between the two
 * bias-corrected samples. With q(t) = dq(t) (x) q(t0) the rotation is integrated as dq from the identity:
 *   d(dq)/dt = 1/2 Omega(w) dq,  dv/dt = R(dq (x) q(t0))^T a - g,  dp/dt = v,
 * by the classical Runge-Kutta method, with dq renormalized at every stage.
 */
ImuMean IntegrateMean(const ImuMean& start, double gravity_magnitude, double dt, const Eigen::Vector3d& w0,
                      const Eigen::Vector3d& a0, const Eigen::Vector3d& w1, const Eigen::Vector3d& a1) {
  const Eigen::Vector3d gravity(0.0, 0.0, gravity_magnitude);
  const Eigen::Vector4d identity(0.0, 0.0, 0.0, 1.0);
  const Eigen::Vector3d w_mid = w0 + 0.5 * dt * ((w1 - w0) / dt);
  const Eigen::Vector3d a_mid = a0 + 0.5 * dt * ((a1 - a0) / dt);

  // Increments of dq and v over the interval at one stage
  auto stage = [&](const Eigen::Vector4d& dq, const Eigen::Vector3d& w, const Eigen::Vector3d& a,
                   Eigen::Vector4d* d_dq, Eigen::Vector3d* d_v) {
    *d_dq = 0.5 * jpl::Omega(w) * dq * dt;
    *d_v = (jpl::ToRotation(jpl::Multiply(dq, start.q)).transpose() * a - gravity) * dt;
  };
  Eigen::Vector4d k1_q, k2_q, k3_q, k4_q;
  Eigen::Vector3d k1_v, k2_v, k3_v, k4_v;
  stage(identity, w0, a0, &k1_q, &k1_v);
  const Eigen::Vector4d dq1 = jpl::NormalizePositive(identity + 0.5 * k1_q);
  const Eigen::Vector3d v1 = start.v + 0.5 * k1_v;
  stage(dq1, w_mid, a_mid, &k2_q, &k2_v);
  const Eigen::Vector4d dq2 = jpl::NormalizePositive(identity + 0.5 * k2_q);
  const Eigen::Vector3d v2 = start.v + 0.5 * k2_v;
  stage(dq2, w_mid, a_mid, &k3_q, &k3_v);
  const Eigen::Vector4d dq3 = jpl::NormalizePositive(identity + k3_q);
  const Eigen::Vector3d v3 = start.v + k3_v;
  stage(dq3, w1, a1, &k4_q, &k4_v);

  ImuMean end;
  end.q = jpl::Multiply(jpl::NormalizePositive(identity + (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q) / 6.0), start.q);
  end.p = start.p + (start.v + 2.0 * v1 + 2.0 * v2 + v3) * (dt / 6.0);
  end.v = start.v + (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) / 6.0;
  return end;
}

}  // namespace

void Estimator::PropagateInterval(const ImuSample& start, const ImuSample& end, Eigen::Matrix<double, 15, 15>* F,
                                  Eigen::Matrix<double, 15, 15>* Qd) {
  const double dt = end.timestamp - start.timestamp;
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  const Eigen::Vector3d gravity(0.0, 0.0, gravity_magnitude_);

  // Bias-corrected samples; the transition uses their average over the interval
  const Eigen::Vector3d w0 = start.gyro - bg_;
  const Eigen::Vector3d w1 = end.gyro - bg_;
  const Eigen::Vector3d a0 = start.accel - ba_;
  const Eigen::Vector3d a1 = end.accel - ba_;
  const Eigen::Vector3d w_mean = 0.5 * (w0 + w1);
  const RotationIntegrals xi = IntegrateRotation(dt, w_mean, 0.5 * (a0 + a1));
  const Eigen::Matrix3d Jr = RightJacobianSO3(-w_mean * dt);
  const ImuMean next = IntegrateMean({q_, p_, v_}, gravity_magnitude_, dt, w0, a0, w1, a1);

  // Error-state transition [theta, p, v, bg, ba] with the first estimates of the interval start ([4]) and the
  // propagated mean at its end
  const Eigen::Matrix3d Rt = jpl::ToRotation(q_fej_).transpose();
  const Eigen::Matrix3d dR = jpl::ToRotation(next.q) * Rt;
  F->setZero();
  F->block<3, 3>(0, 0) = dR;
  F->block<3, 3>(0, 9) = -dR * Jr * dt;
  F->block<3, 3>(3, 0) = -Skew(next.p - p_fej_ - v_fej_ * dt + 0.5 * gravity * dt * dt) * Rt;
  F->block<3, 3>(3, 3) = I;
  F->block<3, 3>(3, 6) = dt * I;
  F->block<3, 3>(3, 9) = Rt * xi.Xi4;
  F->block<3, 3>(3, 12) = -Rt * xi.Xi2;
  F->block<3, 3>(6, 0) = -Skew(next.v - v_fej_ + gravity * dt) * Rt;
  F->block<3, 3>(6, 6) = I;
  F->block<3, 3>(6, 9) = Rt * xi.Xi3;
  F->block<3, 3>(6, 12) = -Rt * xi.Xi1;
  F->block<3, 3>(9, 9) = I;
  F->block<3, 3>(12, 12) = I;

  // Noise [gyro, accel, gyro walk, accel walk] enters like the biases; the continuous densities become sigma^2 / dt
  // over the interval
  Eigen::Matrix<double, 15, 12> G = Eigen::Matrix<double, 15, 12>::Zero();
  G.block<3, 3>(0, 0) = F->block<3, 3>(0, 9);
  G.block<3, 3>(3, 0) = F->block<3, 3>(3, 9);
  G.block<3, 3>(6, 0) = F->block<3, 3>(6, 9);
  G.block<3, 3>(3, 3) = F->block<3, 3>(3, 12);
  G.block<3, 3>(6, 3) = F->block<3, 3>(6, 12);
  G.block<3, 3>(9, 6) = dt * I;
  G.block<3, 3>(12, 9) = dt * I;
  Eigen::Matrix<double, 12, 1> density;
  density << Eigen::Vector3d::Constant(imu_noise_.gyro * imu_noise_.gyro),
      Eigen::Vector3d::Constant(imu_noise_.accel * imu_noise_.accel),
      Eigen::Vector3d::Constant(imu_noise_.gyro_walk * imu_noise_.gyro_walk),
      Eigen::Vector3d::Constant(imu_noise_.accel_walk * imu_noise_.accel_walk);
  *Qd = G * (density / dt).asDiagonal() * G.transpose();
  *Qd = 0.5 * (*Qd + Qd->transpose()).eval();

  // The propagated mean is also the new first estimate
  q_ = next.q;
  p_ = next.p;
  v_ = next.v;
  q_fej_ = next.q;
  p_fej_ = next.p;
  v_fej_ = next.v;
}

bool Estimator::PropagateToImage(double t) {
  // The image was exposed at t + t_d on the IMU clock
  const double t_capture = t + rig_.time_offset;
  if (t_capture <= time_) {
    failure_ = "propagation not forward in time";
    return false;
  }

  // Transition and noise of the whole step, interval by interval
  const std::vector<ImuSample> samples = ImuSamplesOverInterval(imu_, time_, t_capture, true);
  Matrix15d Phi = Matrix15d::Identity();
  Matrix15d Q = Matrix15d::Zero();
  double covered = 0.0;
  for (size_t k = 0; k + 1 < samples.size(); ++k) {
    Matrix15d F, Q_interval;
    PropagateInterval(samples[k], samples[k + 1], &F, &Q_interval);
    Phi = F * Phi;
    Q = F * Q * F.transpose() + Q_interval;
    Q = 0.5 * (Q + Q.transpose()).eval();
    covered += samples[k + 1].timestamp - samples[k].timestamp;
  }
  if (samples.empty() || std::abs((t_capture - time_) - covered) >= 1e-4) {
    failure_ = "IMU does not cover the propagation interval";
    return false;
  }

  // Covariance ([5]): the IMU rows and columns move with Phi, the rest stays; the rig time offsets are random walks
  const int n = static_cast<int>(P_.rows());
  const Eigen::MatrixXd imu_rows = P_.topRows(15);
  P_.topRows(15) = Phi * imu_rows;
  P_.leftCols(15) = P_.topRows(15).transpose().eval();
  P_.topLeftCorner(15, 15) = Phi * imu_rows.leftCols(15) * Phi.transpose() + Q;
  const double elapsed = t_capture - time_;
  const double offset_walk = options_.time_offset_walk * options_.time_offset_walk * elapsed;
  P_(kTimeOffset, kTimeOffset) += offset_walk;
  P_(kSensorTimeOffset, kSensorTimeOffset) += offset_walk;
  if ((P_.diagonal().array() < 0.0).any()) {
    failure_ = "negative covariance diagonal after propagation";
    return false;
  }
  time_ = t_capture;

  // Clone of the IMU pose [theta, p] at the exposure ([5], [6]). An error dt_d of the time offset moves the exposed
  // pose by [w dt_d, v dt_d] (body rate w; R_GtoI = Exp(-dtheta) R_GtoI_hat), so the clone error is J x with
  // J = [I_6, 0, ..., (w; v) at the time offset, 0, ...]
  Eigen::Matrix<double, 6, 1> J_offset;
  J_offset << samples.back().gyro - bg_, v_;
  const Eigen::MatrixXd JP = P_.topRows(6) + J_offset * P_.row(kTimeOffset);
  const Eigen::Matrix<double, 6, 6> JPJt = JP.leftCols(6) + JP.col(kTimeOffset) * J_offset.transpose();
  Eigen::MatrixXd augmented(n + 6, n + 6);  // every block is written below
  augmented.topLeftCorner(n, n) = P_;
  augmented.block(n, 0, 6, n) = JP;
  augmented.block(0, n, n, 6) = JP.transpose();
  augmented.block(n, n, 6, 6) = 0.5 * (JPJt + JPJt.transpose());
  P_.swap(augmented);
  clones_.push_back({t, t_capture, q_, p_, q_fej_, p_fej_, v_, n});
  sensor_motion_.emplace_back();
  return true;
}

Eigen::Matrix3d Estimator::GyroRotation(double t0, double t1, Eigen::Vector3d& w_end) const {
  w_end.setZero();
  if (imu_.empty()) {
    return Eigen::Matrix3d::Identity();
  }

  // Rate at t1: linear interpolation of the buffer, held beyond its ends
  const auto it = std::lower_bound(imu_.begin(), imu_.end(), t1,
                                   [](const ImuSample& s, double t) { return s.timestamp < t; });
  if (it == imu_.begin() || it == imu_.end()) {
    w_end = (it == imu_.end() ? imu_.back().gyro : it->gyro) - bg_;
  } else {
    const double l = (t1 - (it - 1)->timestamp) / (it->timestamp - (it - 1)->timestamp);
    w_end = (1.0 - l) * (it - 1)->gyro + l * it->gyro - bg_;
  }

  // R_{I(lo) <- I(hi)} = product of Exp(w dt) in time order (midpoint rates)
  const double lo = std::min(t0, t1);
  const double hi = std::max(t0, t1);
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  if (hi - lo > 1e-9) {
    const std::vector<ImuSample> samples = ImuSamplesOverInterval(imu_, lo, hi, true);
    for (size_t i = 0; i + 1 < samples.size(); ++i) {
      const Eigen::Vector3d w = 0.5 * (samples[i].gyro + samples[i + 1].gyro) - bg_;
      R = R * ExpSO3(w * (samples[i + 1].timestamp - samples[i].timestamp));
    }
  }
  return (t1 >= t0) ? R : Eigen::Matrix3d(R.transpose());
}

const Estimator::SensorMotion& Estimator::SensorOneMotion(size_t c) const {
  // Recomputed when the sensor time offset moved by more than 10 us (the rotation then changes by microradians)
  SensorMotion& motion = sensor_motion_[c];
  if (!(std::abs(motion.dt - rig_.sensor_time_offset) < 1e-5)) {
    const double t_front = clones_[c].t_capture;
    motion.R_back_to_front = GyroRotation(t_front, t_front + rig_.sensor_time_offset, motion.w_back);
    motion.dt = rig_.sensor_time_offset;
  }
  return motion;
}

const Estimator::SensorMotion* Estimator::SensorPose(size_t c, const Eigen::Vector3d& ray, bool first_estimate,
                                                     Eigen::Matrix3d& R_GtoI, Eigen::Vector3d& p_IinG) const {
  const Clone& clone = clones_[c];
  R_GtoI = jpl::ToRotation(first_estimate ? clone.q_fej : clone.q);
  p_IinG = first_estimate ? clone.p_fej : clone.p;
  if (camera_.SensorOf(ray) == 0) {
    return nullptr;
  }

  // Sensor 1: R_GtoI(1) = R_{I(1) <- I(0)} R_GtoI(0), p(1) = p(0) + v dt_1
  const SensorMotion& motion = SensorOneMotion(c);
  R_GtoI = motion.R_back_to_front.transpose() * R_GtoI;
  p_IinG += clone.v * rig_.sensor_time_offset;
  return &motion;
}

}  // namespace vio

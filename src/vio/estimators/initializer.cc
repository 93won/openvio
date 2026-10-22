// SPDX-License-Identifier: MIT
/**
 * @file      initializer.cc
 * @brief     Implements the initializer with its IMU preintegration and cost functions.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2026-10-06
 * @copyright Copyright (c) 2026 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 *
 * Pose blocks of the refinement are [q_GtoI (JPL quaternion, 4), p_IinG (3)] with the left orientation error
 * R_GtoI = Exp(-dtheta) * R_GtoI_hat.
 *
 * References:
 *   [1] K. Eckenhoff, P. Geneva, G. Huang, "Closed-form preintegration methods for graph-based visual-inertial
 *       navigation", IJRR 38(5), 2019 (piecewise-constant measurement model: closed-form means, bias Jacobians,
 *       covariance dynamics, and the preintegrated residual).
 *   [2] P. Geneva, G. Huang, "Revisiting Visual-Inertial Structure-from-Motion for Odometry and SLAM
 *       Initialization", IEEE RA-L 8(4), 2023 (moving start: linear system in the first IMU frame with the
 *       gravity-magnitude constraint, then maximum-likelihood refinement).
 *   [3] T.-C. Dong-Si, A. I. Mourikis, "Estimator initialization in vision-aided inertial navigation with unknown
 *       camera-IMU calibration", IROS 2012 (gravity magnitude by a Lagrange multiplier, polynomial in it).
 *   [4] A. Edelman, H. Murakami, "Polynomial roots from companion matrix eigenvalues", Math. Comp. 64(210), 1995.
 *   [5] N. Trawny, S. I. Roumeliotis, "Indirect Kalman Filter for 3D Attitude Estimation", Tech. Rep. 2005-002,
 *       University of Minnesota, 2005 (JPL quaternion algebra).
 *   [6] P. D. Groves, "Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems", 2nd ed., 2013,
 *       Ch. 5 (initialization at rest: gravity from the mean specific force, gyro bias from the mean rate).
 *   [7] E. Hairer, S. P. Norsett, G. Wanner, "Solving Ordinary Differential Equations I", 2nd ed., 1993, Ch. II
 *       (classical fourth-order Runge-Kutta method).
 */

#include "vio/estimators/initializer.h"

#include "vio/geometry/lie_group.h"
#include "vio/sensor/imu_interval.h"
#include "vio/util/logging.h"

#include <ceres/ceres.h>
#include <ceres/local_parameterization.h>
#include <ceres/sized_cost_function.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <memory>
#include <set>
#include <sstream>

namespace vio {
namespace {

using Matrix15 = Eigen::Matrix<double, 15, 15>;

// ============================================================================
// IMU preintegration ([1], piecewise-constant measurements)
// ============================================================================

/**
 * Preintegrated IMU from a start frame k ([1]): alpha and beta, the double and single integrals of the rotated specific
 * force R_{s->k} a, the rotation R_{k->tau}, their first-order Jacobians w.r.t. the bias linearization points, and the
 * covariance of [theta, b_g, beta, b_a, alpha]. Within a step, rate and specific force are the means of the step's two
 * samples minus the bias linearization points.
 */
class Preintegration {
 public:
  Preintegration(const ImuNoise& noise, const Eigen::Vector3d& gyro_bias_lin, const Eigen::Vector3d& accel_bias_lin)
      : bg_lin_(gyro_bias_lin), ba_lin_(accel_bias_lin) {
    // Continuous noise densities in the order [gyro, gyro walk, accel, accel walk]
    const double sigma[4] = {noise.gyro, noise.gyro_walk, noise.accel, noise.accel_walk};
    for (int k = 0; k < 4; ++k) {
      Qc_.block<3, 3>(3 * k, 3 * k) = sigma[k] * sigma[k] * Eigen::Matrix3d::Identity();
    }
  }

  // Add the step between two consecutive samples
  void AddStep(const ImuSample& start, const ImuSample& end);

  double delta_time() const { return dt_; }
  const Eigen::Vector3d& alpha() const { return alpha_; }
  const Eigen::Vector3d& beta() const { return beta_; }
  const Eigen::Matrix3d& rotation_start_to_end() const { return R_k_to_tau_; }
  const Eigen::Vector4d& quaternion_start_to_end() const { return q_k_to_tau_; }
  const Eigen::Matrix3d& rotation_gyro_bias_jacobian() const { return dR_dbg_; }
  const Eigen::Matrix3d& alpha_gyro_bias_jacobian() const { return dalpha_dbg_; }
  const Eigen::Matrix3d& beta_gyro_bias_jacobian() const { return dbeta_dbg_; }
  const Eigen::Matrix3d& alpha_accel_bias_jacobian() const { return dalpha_dba_; }
  const Eigen::Matrix3d& beta_accel_bias_jacobian() const { return dbeta_dba_; }
  const Matrix15& covariance() const { return P_; }
  const Eigen::Vector3d& gyro_bias_lin() const { return bg_lin_; }
  const Eigen::Vector3d& accel_bias_lin() const { return ba_lin_; }

 private:
  /**
   * Integrals of a step of length dt with constant rate w, W = [w]x ([1]):
   *   int_0^dt Exp(-w u) du   = dt I + f3 W + f4 W^2,  f3 = -(1 - cos)/|w|^2,  f4 = (theta - sin)/|w|^3,
   *   int_0^dt u Exp(-w u) du = dt^2/2 I + f1 W + f2 W^2,  f1 = (theta cos - sin)/|w|^3,
   *                                                        f2 = (theta^2 - 2 cos - 2 theta sin + 2)/(2 |w|^4),
   * theta = |w| dt. With w = gyro - b_g, d f_i / d b_g,j = w_j g_i where g_i = -(d f_i / d|w|) / |w|. Below the
   * small-rate threshold the leading terms of the Taylor series are used.
   */
  struct StepCoefficients {
    double f1, f2, f3, f4;
    double g1, g2, g3, g4;
  };
  static StepCoefficients Coefficients(double rate, double dt, bool small_rate);

  // Exp(-w h) (second-order series below the small-rate threshold)
  static Eigen::Matrix3d StepRotation(const Eigen::Matrix3d& W, double rate, double h, bool small_rate);

  // Covariance dynamics dP/dt = F P + P F^T + G Qc G^T at the rotation R_{k->t} ([1]):
  //   theta' = -[w]x theta - b_g - n_g,  b_g' = n_wg,  beta' = -R_{t->k} ([a]x theta + b_a + n_a),  b_a' = n_wa,
  //   alpha' = beta
  Matrix15 CovarianceRate(const Matrix15& P, const Eigen::Matrix3d& R_k_to_t, const Eigen::Matrix3d& W,
                          const Eigen::Matrix3d& A) const;

  Eigen::Matrix<double, 12, 12> Qc_ = Eigen::Matrix<double, 12, 12>::Zero();
  Eigen::Vector3d bg_lin_;                                         // Gyro bias linearization point
  Eigen::Vector3d ba_lin_;                                         // Accel bias linearization point
  double dt_ = 0.0;                                                // Integrated time [s]
  Eigen::Vector3d alpha_ = Eigen::Vector3d::Zero();                // Double integral
  Eigen::Vector3d beta_ = Eigen::Vector3d::Zero();                 // Single integral
  Eigen::Matrix3d R_k_to_tau_ = Eigen::Matrix3d::Identity();       // Rotation from the start to the latest time
  Eigen::Vector4d q_k_to_tau_ = Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);  // Same as a JPL quaternion
  Eigen::Matrix3d dR_dbg_ = Eigen::Matrix3d::Zero();               // Orientation w.r.t. gyro bias
  Eigen::Matrix3d dalpha_dbg_ = Eigen::Matrix3d::Zero();           // alpha w.r.t. gyro bias
  Eigen::Matrix3d dbeta_dbg_ = Eigen::Matrix3d::Zero();            // beta w.r.t. gyro bias
  Eigen::Matrix3d dalpha_dba_ = Eigen::Matrix3d::Zero();           // alpha w.r.t. accel bias
  Eigen::Matrix3d dbeta_dba_ = Eigen::Matrix3d::Zero();            // beta w.r.t. accel bias
  Matrix15 P_ = Matrix15::Zero();                                  // Covariance [theta, b_g, beta, b_a, alpha]
};

Preintegration::StepCoefficients Preintegration::Coefficients(double rate, double dt, bool small_rate) {
  StepCoefficients c;
  if (small_rate) {
    c.f1 = -std::pow(dt, 3) / 3.0;
    c.f2 = std::pow(dt, 4) / 8.0;
    c.f3 = -std::pow(dt, 2) / 2.0;
    c.f4 = std::pow(dt, 3) / 6.0;
    c.g1 = -std::pow(dt, 5) / 15.0;
    c.g2 = std::pow(dt, 6) / 72.0;
    c.g3 = -std::pow(dt, 4) / 12.0;
    c.g4 = std::pow(dt, 5) / 60.0;
    return c;
  }
  const double theta = rate * dt;
  const double cs = std::cos(theta);
  const double sn = std::sin(theta);
  c.f1 = (theta * cs - sn) / std::pow(rate, 3);
  c.f2 = (theta * theta - 2.0 * cs - 2.0 * theta * sn + 2.0) / (2.0 * std::pow(rate, 4));
  c.f3 = -(1.0 - cs) / std::pow(rate, 2);
  c.f4 = (theta - sn) / std::pow(rate, 3);
  c.g1 = (theta * theta * sn - 3.0 * sn + 3.0 * theta * cs) / std::pow(rate, 5);
  c.g2 = (theta * theta - 4.0 * cs - 4.0 * theta * sn + theta * theta * cs + 4.0) / std::pow(rate, 6);
  c.g3 = (2.0 * (cs - 1.0) + theta * sn) / std::pow(rate, 4);
  c.g4 = (2.0 * theta + theta * cs - 3.0 * sn) / std::pow(rate, 5);
  return c;
}

Eigen::Matrix3d Preintegration::StepRotation(const Eigen::Matrix3d& W, double rate, double h, bool small_rate) {
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  if (small_rate) {
    return I - h * W + 0.5 * h * h * W * W;
  }
  return I - (std::sin(rate * h) / rate) * W + ((1.0 - std::cos(rate * h)) / (rate * rate)) * W * W;
}

Matrix15 Preintegration::CovarianceRate(const Matrix15& P, const Eigen::Matrix3d& R_k_to_t, const Eigen::Matrix3d& W,
                                        const Eigen::Matrix3d& A) const {
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d R_t_to_k = R_k_to_t.transpose();
  Matrix15 F = Matrix15::Zero();
  F.block<3, 3>(0, 0) = -W;
  F.block<3, 3>(0, 3) = -I;
  F.block<3, 3>(6, 0) = -R_t_to_k * A;
  F.block<3, 3>(6, 9) = -R_t_to_k;
  F.block<3, 3>(12, 6) = I;
  Eigen::Matrix<double, 15, 12> G = Eigen::Matrix<double, 15, 12>::Zero();
  G.block<3, 3>(0, 0) = -I;
  G.block<3, 3>(3, 3) = I;
  G.block<3, 3>(6, 6) = -R_t_to_k;
  G.block<3, 3>(9, 9) = I;
  return F * P + P * F.transpose() + G * Qc_ * G.transpose();
}

void Preintegration::AddStep(const ImuSample& start, const ImuSample& end) {
  constexpr double kSmallRate = M_PI / 360.0;  // 0.5 deg/s

  const double dt = end.timestamp - start.timestamp;
  dt_ += dt;
  if (dt == 0.0) {
    return;
  }
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  const Eigen::Vector3d w = 0.5 * (start.gyro + end.gyro) - bg_lin_;
  const Eigen::Vector3d a = 0.5 * (start.accel + end.accel) - ba_lin_;
  const double rate = w.norm();
  const bool small_rate = rate < kSmallRate;
  const Eigen::Matrix3d W = Skew(w);
  const Eigen::Matrix3d W2 = W * W;
  const Eigen::Matrix3d A = Skew(a);

  // Rotation to the end of the step, R_{k->tau'} = Exp(-w dt) R_{k->tau}, and back
  const Eigen::Matrix3d R_step = StepRotation(W, rate, dt, small_rate);
  const Eigen::Matrix3d R_end = R_step * R_k_to_tau_;
  const Eigen::Matrix3d R_back = R_end.transpose();

  // Means: alpha moves with the beta of the start of the step
  const StepCoefficients f = Coefficients(rate, dt, small_rate);
  const Eigen::Matrix3d single = dt * I + f.f3 * W + f.f4 * W2;
  const Eigen::Matrix3d weighted = 0.5 * dt * dt * I + f.f1 * W + f.f2 * W2;
  const Eigen::Matrix3d G_alpha = R_back * weighted;
  const Eigen::Matrix3d G_beta = R_back * single;
  alpha_ += beta_ * dt + G_alpha * a;
  beta_ += G_beta * a;

  // Orientation w.r.t. the gyro bias: J <- Exp(-w dt) J + Jr(w dt) dt
  const Eigen::Matrix3d Wdt = Skew(w * dt);
  Eigen::Matrix3d Jr;
  if (small_rate) {
    Jr = I - 0.5 * Wdt + (1.0 / 6.0) * Wdt * Wdt;
  } else {
    const double theta = rate * dt;
    Jr = I - ((1.0 - std::cos(theta)) / (theta * theta)) * Wdt +
         ((theta - std::sin(theta)) / (theta * theta * theta)) * Wdt * Wdt;
  }
  dR_dbg_ = R_step * dR_dbg_ + Jr * dt;

  // Accelerometer bias (alpha with the start-of-step beta Jacobian)
  dalpha_dba_ += dt * dbeta_dba_ - G_alpha;
  dbeta_dba_ -= G_beta;

  // Gyro bias, one axis at a time: the change of the rotation to frame k and of the step integrals
  Eigen::Matrix3d dalpha_step;
  Eigen::Matrix3d dbeta_step;
  for (int j = 0; j < 3; ++j) {
    const Eigen::Vector3d e = Eigen::Vector3d::Unit(j);
    const Eigen::Matrix3d E = Skew(e);
    const Eigen::Matrix3d dR_back = -R_back * Skew(dR_dbg_ * e);
    const Eigen::Matrix3d EW = E * W + W * E;
    const Eigen::Matrix3d d_weighted = w(j) * f.g1 * W - f.f1 * E + w(j) * f.g2 * W2 - f.f2 * EW;
    const Eigen::Matrix3d d_single = w(j) * f.g3 * W - f.f3 * E + w(j) * f.g4 * W2 - f.f4 * EW;
    dalpha_step.col(j) = (dR_back * weighted + R_back * d_weighted) * a;
    dbeta_step.col(j) = (dR_back * single + R_back * d_single) * a;
  }
  dalpha_dbg_ += dt * dbeta_dbg_ + dalpha_step;
  dbeta_dbg_ += dbeta_step;

  // Covariance: classical Runge-Kutta ([7]) with the rotation at the start, the middle and the end of the step
  const Eigen::Matrix3d R_mid = StepRotation(W, rate, 0.5 * dt, small_rate) * R_k_to_tau_;
  const Matrix15 k1 = CovarianceRate(P_, R_k_to_tau_, W, A);
  const Matrix15 k2 = CovarianceRate(P_ + 0.5 * dt * k1, R_mid, W, A);
  const Matrix15 k3 = CovarianceRate(P_ + 0.5 * dt * k2, R_mid, W, A);
  const Matrix15 k4 = CovarianceRate(P_ + dt * k3, R_end, W, A);
  P_ += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
  P_ = 0.5 * (P_ + P_.transpose()).eval();

  R_k_to_tau_ = R_end;
  q_k_to_tau_ = jpl::FromRotation(R_end);
}

// IMU preintegration between two times; nullptr when the IMU does not cover the interval
std::shared_ptr<Preintegration> Preintegrate(const std::vector<ImuSample>& imu, double t0, double t1,
                                             const ImuNoise& noise, const Eigen::Vector3d& gyro_bias_lin,
                                             const Eigen::Vector3d& accel_bias_lin) {
  constexpr double kMaxSpanError = 0.01;  // Accepted IMU span against the interval [s]
  const std::vector<ImuSample> samples = ImuSamplesOverInterval(imu, t0, t1, false);
  if (samples.size() < 2 ||
      std::abs((samples.back().timestamp - samples.front().timestamp) - (t1 - t0)) > kMaxSpanError) {
    return nullptr;
  }
  auto preintegration = std::make_shared<Preintegration>(noise, gyro_bias_lin, accel_bias_lin);
  for (size_t k = 0; k + 1 < samples.size(); ++k) {
    preintegration->AddStep(samples[k], samples[k + 1]);
  }
  return preintegration;
}

// ============================================================================
// Cost functions of the refinement
// ============================================================================

// Quaternion update q <- [sin(|d|/2) d/|d|; cos(|d|/2)] (x) q ([5]), tangent Jacobian [I; 0]
class JplQuaternionParameterization : public ceres::LocalParameterization {
 public:
  bool Plus(const double* x, const double* delta, double* x_plus_delta) const override {
    Eigen::Map<const Eigen::Vector4d> q(x);
    Eigen::Map<const Eigen::Vector3d> d(delta);
    const double t = d.norm();
    Eigen::Vector4d dq;
    if (t < 1e-8) {
      dq << 0.5 * d, 1.0;
    } else {
      dq << std::sin(0.5 * t) * d / t, std::cos(0.5 * t);
    }
    Eigen::Map<Eigen::Vector4d> q_plus(x_plus_delta);
    q_plus = jpl::Multiply(dq.normalized(), q);
    return true;
  }
  bool ComputeJacobian(const double* x, double* jacobian) const override {
    (void)x;
    Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> J(jacobian);
    J.setZero();
    J.topRows<3>().setIdentity();
    return true;
  }
  int GlobalSize() const override { return 4; }
  int LocalSize() const override { return 3; }
};

/**
 * Preintegrated IMU between two poses ([1]) with an analytic Jacobian. Residual [theta, b_g, v, b_a, p] (the order of
 * the preintegration covariance), whitened with it. The measurement is corrected to first order for the bias change
 * from its linearization point. Parameters: q1 [4], p1, v1, bg1, ba1 [3 each], q2 [4], p2, v2, bg2, ba2 [3 each].
 */
class ImuPreintegrationCost : public ceres::CostFunction {
 public:
  ImuPreintegrationCost(const Preintegration& pre, const Eigen::Vector3d& gravity)
      : dt_(pre.delta_time()),
        gravity_(gravity),
        alpha_(pre.alpha()),
        beta_(pre.beta()),
        q_meas_(pre.quaternion_start_to_end()),
        bg_lin_(pre.gyro_bias_lin()),
        ba_lin_(pre.accel_bias_lin()),
        dR_dbg_(pre.rotation_gyro_bias_jacobian()),
        dalpha_dbg_(pre.alpha_gyro_bias_jacobian()),
        dbeta_dbg_(pre.beta_gyro_bias_jacobian()),
        dalpha_dba_(pre.alpha_accel_bias_jacobian()),
        dbeta_dba_(pre.beta_accel_bias_jacobian()) {
    // Whitening S with S^T S = P^-1: S = L^T for the Cholesky factor L L^T = P^-1
    const Matrix15 information = pre.covariance().llt().solve(Matrix15::Identity());
    whitening_ = information.llt().matrixL().transpose();
    valid_ = whitening_.allFinite();
    set_num_residuals(15);
    for (int size : {4, 3, 3, 3, 3, 4, 3, 3, 3, 3}) {
      mutable_parameter_block_sizes()->push_back(size);
    }
  }

  // The preintegration covariance could be inverted
  bool IsValid() const { return valid_; }

  bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override {
    Eigen::Map<const Eigen::Vector4d> q1(parameters[0]);
    Eigen::Map<const Eigen::Vector3d> p1(parameters[1]);
    Eigen::Map<const Eigen::Vector3d> v1(parameters[2]);
    Eigen::Map<const Eigen::Vector3d> bg1(parameters[3]);
    Eigen::Map<const Eigen::Vector3d> ba1(parameters[4]);
    Eigen::Map<const Eigen::Vector4d> q2(parameters[5]);
    Eigen::Map<const Eigen::Vector3d> p2(parameters[6]);
    Eigen::Map<const Eigen::Vector3d> v2(parameters[7]);
    Eigen::Map<const Eigen::Vector3d> bg2(parameters[8]);
    Eigen::Map<const Eigen::Vector3d> ba2(parameters[9]);

    const Eigen::Matrix3d R1 = jpl::ToRotation(q1);
    const Eigen::Vector3d dbg = bg1 - bg_lin_;
    const Eigen::Vector3d dba = ba1 - ba_lin_;

    // Rotation: the relative rotation against the measurement corrected for the gyro bias,
    // e = (q2 (x) q1^-1) (x) q_meas^-1 (x) q_bias, residual 2 vec(e)
    Eigen::Vector4d q_bias;
    q_bias << 0.5 * dR_dbg_ * dbg, 1.0;
    q_bias.normalize();
    const Eigen::Vector4d q_rel = jpl::Multiply(q2, jpl::Inverse(q1));
    const Eigen::Vector4d q_off = jpl::Multiply(q_rel, jpl::Inverse(q_meas_));
    const Eigen::Vector4d q_err = jpl::Multiply(q_off, q_bias);

    // Velocity and position in frame 1 against the bias-corrected beta and alpha
    const Eigen::Vector3d dv = v2 - v1 + gravity_ * dt_;
    const Eigen::Vector3d dp = p2 - p1 - v1 * dt_ + 0.5 * gravity_ * dt_ * dt_;
    Eigen::Matrix<double, 15, 1> r;
    r << 2.0 * q_err.head<3>(), bg2 - bg1, R1 * dv - dbeta_dbg_ * dbg - dbeta_dba_ * dba - beta_, ba2 - ba1,
        R1 * dp - dalpha_dbg_ * dbg - dalpha_dba_ * dba - alpha_;
    Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);
    residual = whitening_ * r;
    if (!jacobians) {
      return true;
    }

    // Tangent-space Jacobian, columns [q1, p1, v1, bg1, ba1, q2, p2, v2, bg2, ba2] x 3 ([5]: products of JPL
    // quaternions as left and right product matrices)
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Eigen::Matrix<double, 15, 30> H = Eigen::Matrix<double, 15, 30>::Zero();
    const Eigen::Vector4d q_meas_bias = jpl::Multiply(jpl::Inverse(q_meas_), q_bias);
    // q1 <- dq (x) q1 gives q_err = L(q_rel) R(q_meas_bias) dq^-1
    H.block<3, 3>(0, 0) = -(jpl::LeftProductBlock(q_rel) * jpl::RightProductBlock(q_meas_bias) -
                            q_rel.head<3>() * q_meas_bias.head<3>().transpose());
    H.block<3, 3>(0, 9) = jpl::LeftProductBlock(q_off) * dR_dbg_;
    // q2 <- dq (x) q2 gives q_err = R(q_err) dq
    H.block<3, 3>(0, 15) = jpl::RightProductBlock(q_err);
    H.block<3, 3>(3, 9) = -I;
    H.block<3, 3>(3, 24) = I;
    H.block<3, 3>(6, 0) = Skew(R1 * dv);
    H.block<3, 3>(6, 6) = -R1;
    H.block<3, 3>(6, 9) = -dbeta_dbg_;
    H.block<3, 3>(6, 12) = -dbeta_dba_;
    H.block<3, 3>(6, 21) = R1;
    H.block<3, 3>(9, 12) = -I;
    H.block<3, 3>(9, 27) = I;
    H.block<3, 3>(12, 0) = Skew(R1 * dp);
    H.block<3, 3>(12, 3) = -R1;
    H.block<3, 3>(12, 6) = -R1 * dt_;
    H.block<3, 3>(12, 9) = -dalpha_dbg_;
    H.block<3, 3>(12, 12) = -dalpha_dba_;
    H.block<3, 3>(12, 18) = R1;
    H = whitening_ * H;

    // Quaternion blocks carry a zero fourth column (tangent Jacobian [I; 0])
    for (int b = 0; b < 10; ++b) {
      if (!jacobians[b]) {
        continue;
      }
      if (b == 0 || b == 5) {
        Eigen::Map<Eigen::Matrix<double, 15, 4, Eigen::RowMajor>> J(jacobians[b]);
        J.leftCols<3>() = H.block<15, 3>(0, 3 * b);
        J.col(3).setZero();
      } else {
        Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J(jacobians[b]);
        J = H.block<15, 3>(0, 3 * b);
      }
    }
    return true;
  }

 private:
  double dt_;                    // Preintegrated time [s]
  Eigen::Vector3d gravity_;      // Gravity (up) in the world frame [m/s^2]
  Eigen::Vector3d alpha_;        // Double integral
  Eigen::Vector3d beta_;         // Single integral
  Eigen::Vector4d q_meas_;       // Measured rotation start->end (JPL)
  Eigen::Vector3d bg_lin_;       // Gyro bias linearization point
  Eigen::Vector3d ba_lin_;       // Accel bias linearization point
  Eigen::Matrix3d dR_dbg_;       // Rotation w.r.t. gyro bias
  Eigen::Matrix3d dalpha_dbg_;   // alpha w.r.t. gyro bias
  Eigen::Matrix3d dbeta_dbg_;    // beta w.r.t. gyro bias
  Eigen::Matrix3d dalpha_dba_;   // alpha w.r.t. accel bias
  Eigen::Matrix3d dbeta_dba_;    // beta w.r.t. accel bias
  Matrix15 whitening_;           // S with S^T S = P^-1
  bool valid_ = false;           // Covariance could be inverted
};

/**
 * Reprojection of a point with an analytic Jacobian; the extrinsics are fixed. Residual: (projected - observed) /
 * sigma, measured with the camera model's residual. Parameters: q_GtoI [4], p_IinG [3], p_FinG [3].
 */
class ReprojectionCost : public ceres::SizedCostFunction<2, 4, 3, 3> {
 public:
  ReprojectionCost(const Camera* camera, const Eigen::Matrix3d& R_ItoC, const Eigen::Vector3d& p_IinC,
                   const Eigen::Vector2d& uv_measured, double pixel_sigma)
      : camera_(camera), R_ItoC_(R_ItoC), p_IinC_(p_IinC), uv_(uv_measured), inv_sigma_(1.0 / pixel_sigma) {}

  bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override {
    Eigen::Map<const Eigen::Vector4d> q_GtoI(parameters[0]);
    Eigen::Map<const Eigen::Vector3d> p_IinG(parameters[1]);
    Eigen::Map<const Eigen::Vector3d> p_FinG(parameters[2]);

    const Eigen::Matrix3d R_GtoI = jpl::ToRotation(q_GtoI);
    const Eigen::Vector3d p_FinI = R_GtoI * (p_FinG - p_IinG);
    const Eigen::Vector3d p_FinC = R_ItoC_ * p_FinI + p_IinC_;

    Eigen::Matrix<double, 2, 3> J_proj;
    if (!camera_->ProjectionJacobian(p_FinC, &J_proj)) {
      return false;
    }

    const Eigen::Vector2d uv = camera_->Project(p_FinC);
    const Eigen::Vector2d d = camera_->Residual(uv, uv_);
    residuals[0] = inv_sigma_ * d.x();
    residuals[1] = inv_sigma_ * d.y();

    if (!jacobians) {
      return true;
    }

    const Eigen::Matrix<double, 2, 3> Hc = inv_sigma_ * J_proj * R_ItoC_;
    if (jacobians[0]) {
      // q <- Exp(-d) q gives p_FinI <- p_FinI + [p_FinI]x d
      Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> J(jacobians[0]);
      J.leftCols<3>() = Hc * Skew(p_FinI);
      J.col(3).setZero();
    }
    if (jacobians[1]) {
      Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[1]);
      J = -Hc * R_GtoI;
    }
    if (jacobians[2]) {
      Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[2]);
      J = Hc * R_GtoI;
    }
    return true;
  }

 private:
  const Camera* camera_;    // Camera model (outlives the problem)
  Eigen::Matrix3d R_ItoC_;  // Rotation from IMU to camera
  Eigen::Vector3d p_IinC_;  // IMU position in the camera frame
  Eigen::Vector2d uv_;      // Observed pixel coordinates
  double inv_sigma_;        // 1 / pixel noise
};

/**
 * Prior on the first pose fixing the yaw and position gauge, with bias priors. Residual: [yaw, p (3), bg (3),
 * ba (3)] divided by their sigmas. Parameters: q_GtoI [4], p_IinG [3], bg [3], ba [3].
 */
class FirstPosePrior : public ceres::SizedCostFunction<10, 4, 3, 3, 3> {
 public:
  FirstPosePrior(const Eigen::Vector4d& q_lin, const Eigen::Vector3d& p_lin, const Eigen::Vector3d& bg_lin,
                 const Eigen::Vector3d& ba_lin, double sigma_yaw, double sigma_p, double sigma_bg, double sigma_ba)
      : R_lin_(jpl::ToRotation(q_lin)),
        p_lin_(p_lin),
        bg_lin_(bg_lin),
        ba_lin_(ba_lin),
        inv_sigma_yaw_(1.0 / sigma_yaw),
        inv_sigma_p_(1.0 / sigma_p),
        inv_sigma_bg_(1.0 / sigma_bg),
        inv_sigma_ba_(1.0 / sigma_ba) {}

  bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override {
    Eigen::Map<const Eigen::Vector4d> q(parameters[0]);
    Eigen::Map<const Eigen::Vector3d> p(parameters[1]);
    Eigen::Map<const Eigen::Vector3d> bg(parameters[2]);
    Eigen::Map<const Eigen::Vector3d> ba(parameters[3]);

    // Rotation error in the gravity-aligned frame; only its z (yaw) component is constrained
    const Eigen::Vector3d err = LogSO3(jpl::ToRotation(q).transpose() * R_lin_);

    Eigen::Map<Eigen::Matrix<double, 10, 1>> res(residuals);
    res(0) = -inv_sigma_yaw_ * err.z();
    res.segment<3>(1) = inv_sigma_p_ * (p - p_lin_);
    res.segment<3>(4) = inv_sigma_bg_ * (bg - bg_lin_);
    res.segment<3>(7) = inv_sigma_ba_ * (ba - ba_lin_);

    if (!jacobians) {
      return true;
    }
    if (jacobians[0]) {
      // q <- Exp(-d) q gives err <- err + Jr^-1(err) R_lin^T d
      Eigen::Map<Eigen::Matrix<double, 10, 4, Eigen::RowMajor>> J(jacobians[0]);
      J.setZero();
      J.block<1, 3>(0, 0) = -inv_sigma_yaw_ * (RightJacobianInverseSO3(err) * R_lin_.transpose()).row(2);
    }
    if (jacobians[1]) {
      Eigen::Map<Eigen::Matrix<double, 10, 3, Eigen::RowMajor>> J(jacobians[1]);
      J.setZero();
      J.block<3, 3>(1, 0) = inv_sigma_p_ * Eigen::Matrix3d::Identity();
    }
    if (jacobians[2]) {
      Eigen::Map<Eigen::Matrix<double, 10, 3, Eigen::RowMajor>> J(jacobians[2]);
      J.setZero();
      J.block<3, 3>(4, 0) = inv_sigma_bg_ * Eigen::Matrix3d::Identity();
    }
    if (jacobians[3]) {
      Eigen::Map<Eigen::Matrix<double, 10, 3, Eigen::RowMajor>> J(jacobians[3]);
      J.setZero();
      J.block<3, 3>(7, 0) = inv_sigma_ba_ * Eigen::Matrix3d::Identity();
    }
    return true;
  }

 private:
  Eigen::Matrix3d R_lin_;  // Prior rotation R_GtoI
  Eigen::Vector3d p_lin_;  // Prior position
  Eigen::Vector3d bg_lin_; // Prior gyro bias
  Eigen::Vector3d ba_lin_; // Prior accel bias
  double inv_sigma_yaw_;   // 1 / yaw sigma
  double inv_sigma_p_;     // 1 / position sigma
  double inv_sigma_bg_;    // 1 / gyro bias sigma
  double inv_sigma_ba_;    // 1 / accel bias sigma
};

// ============================================================================
// Polynomial, geometry and IMU helpers
// ============================================================================

using Poly = std::vector<double>;  // Polynomial coefficients, ascending degree

Poly PolyAdd(const Poly& a, const Poly& b) {
  Poly r(std::max(a.size(), b.size()), 0.0);
  for (size_t i = 0; i < a.size(); ++i) {
    r[i] += a[i];
  }
  for (size_t i = 0; i < b.size(); ++i) {
    r[i] += b[i];
  }
  return r;
}

Poly PolyScale(const Poly& a, double s) {
  Poly r = a;
  for (double& c : r) {
    c *= s;
  }
  return r;
}

Poly PolyMul(const Poly& a, const Poly& b) {
  Poly r(a.size() + b.size() - 1, 0.0);
  for (size_t i = 0; i < a.size(); ++i) {
    for (size_t j = 0; j < b.size(); ++j) {
      r[i + j] += a[i] * b[j];
    }
  }
  return r;
}

// Two unit rows spanning the plane orthogonal to a unit ray: T * ray = 0
Eigen::Matrix<double, 2, 3> OrthogonalRows(const Eigen::Vector3d& bearing) {
  int axis = 0;
  bearing.cwiseAbs().minCoeff(&axis);
  const Eigen::Vector3d t1 = bearing.cross(Eigen::Vector3d::Unit(axis)).normalized();
  const Eigen::Vector3d t2 = bearing.cross(t1).normalized();
  Eigen::Matrix<double, 2, 3> T;
  T.row(0) = t1.transpose();
  T.row(1) = t2.transpose();
  return T;
}

// sqrt(sum |s - mean|^2 / (n - 1))
double SampleStd(const std::vector<Eigen::Vector3d>& samples, const Eigen::Vector3d& mean) {
  double sum = 0.0;
  for (const auto& s : samples) {
    sum += (s - mean).squaredNorm();
  }
  return std::sqrt(sum / static_cast<double>(samples.size() - 1));
}

/**
 * Polynomial whose roots are the Lagrange multipliers of |g| = G ([3]): for min |A1 x + A2 g - b|^2 s.t. |g| = G with
 * reduced normal equations D g = d, the stationary points satisfy (D - l I) g = d and |g| = G, i.e.
 * G^2 det(D - l I)^2 - |adj(D - l I) d|^2 = 0. Returns the seven monic coefficients, highest degree first.
 */
Eigen::Matrix<double, 7, 1> GravityConstraintPolynomial(const Eigen::Matrix3d& D, const Eigen::Vector3d& d,
                                                        double gravity_magnitude) {
  // M(l) = D - l I with polynomial entries
  Poly M[3][3];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (i == j) {
        M[i][j] = Poly{D(i, j), -1.0};
      } else {
        M[i][j] = Poly{D(i, j)};
      }
    }
  }

  // Cofactors C_ij = (-1)^(i+j) * minor_ij
  Poly C[3][3];
  for (int i = 0; i < 3; ++i) {
    const int r0 = (i == 0) ? 1 : 0;
    const int r1 = (i == 2) ? 1 : 2;
    for (int j = 0; j < 3; ++j) {
      const int c0 = (j == 0) ? 1 : 0;
      const int c1 = (j == 2) ? 1 : 2;
      const Poly minor =
          PolyAdd(PolyMul(M[r0][c0], M[r1][c1]), PolyScale(PolyMul(M[r0][c1], M[r1][c0]), -1.0));
      C[i][j] = ((i + j) % 2 == 0) ? minor : PolyScale(minor, -1.0);
    }
  }

  // det(M) along the first row, and |adj(M) d|^2 with adj(M)_ij = C_ji
  Poly det{0.0};
  for (int j = 0; j < 3; ++j) {
    det = PolyAdd(det, PolyMul(M[0][j], C[0][j]));
  }
  Poly adj_norm2{0.0};
  for (int i = 0; i < 3; ++i) {
    Poly u{0.0};
    for (int j = 0; j < 3; ++j) {
      u = PolyAdd(u, PolyScale(C[j][i], d(j)));
    }
    adj_norm2 = PolyAdd(adj_norm2, PolyMul(u, u));
  }

  const double g2 = gravity_magnitude * gravity_magnitude;
  Poly p = PolyAdd(PolyScale(PolyMul(det, det), g2), PolyScale(adj_norm2, -1.0));
  p.resize(7, 0.0);

  Eigen::Matrix<double, 7, 1> coeff;
  for (int k = 0; k <= 6; ++k) {
    coeff(k) = p[6 - k] / p[6];
  }
  coeff(0) = 1.0;
  return coeff;
}

/**
 * min |A x - b|^2 subject to |g| = G for g = x.tail(3) ([3]). Eliminating the other unknowns x1 leaves the reduced
 * normal equations D g = d; the multipliers l of the constraint are the roots of GravityConstraintPolynomial, found as
 * the eigenvalues of its companion matrix ([4]). The smallest real root is the minimum on the sphere (l below the
 * smallest eigenvalue of D); the other roots are saddle points or maxima. False if no real root exists or |g| misses G
 * by more than 1e-3.
 */
bool SolveGravityConstrainedLeastSquares(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, double gravity_magnitude,
                                         Eigen::VectorXd& x, double& lambda, std::string& reason) {
  constexpr double kMaxGravityError = 1e-3;  // Accepted | |g| - G | [m/s^2]

  const Eigen::Index n = A.cols();
  const Eigen::MatrixXd A1 = A.leftCols(n - 3);
  const Eigen::MatrixXd A2 = A.rightCols(3);
  const Eigen::MatrixXd N11_inv = (A1.transpose() * A1).llt().solve(Eigen::MatrixXd::Identity(n - 3, n - 3));
  const Eigen::MatrixXd N21 = A2.transpose() * A1;
  const Eigen::Matrix3d D = A2.transpose() * A2 - N21 * N11_inv * N21.transpose();
  const Eigen::Vector3d d = A2.transpose() * b - N21 * N11_inv * (A1.transpose() * b);

  // Companion matrix of the monic degree-6 polynomial: ones below the diagonal, the negated coefficients in the last
  // column
  const Eigen::Matrix<double, 7, 1> coeff = GravityConstraintPolynomial(D, d, gravity_magnitude);
  Eigen::Matrix<double, 6, 6> companion = Eigen::Matrix<double, 6, 6>::Zero();
  companion.diagonal(-1).setOnes();
  for (int row = 0; row < 6; ++row) {
    companion(row, 5) = -coeff(6 - row);
  }
  if (Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6>>(companion).rank() != 6) {
    reason = "gravity polynomial companion matrix is rank deficient";
    return false;
  }
  const Eigen::EigenSolver<Eigen::Matrix<double, 6, 6>> eigen(companion, false);
  if (eigen.info() != Eigen::Success) {
    reason = "gravity polynomial eigen decomposition failed";
    return false;
  }
  bool found = false;
  for (int k = 0; k < 6; ++k) {
    const std::complex<double> root = eigen.eigenvalues()(k);
    if (root.imag() == 0.0 && (!found || root.real() < lambda)) {
      found = true;
      lambda = root.real();
    }
  }
  if (!found) {
    reason = "gravity polynomial has no real root";
    return false;
  }

  const Eigen::Vector3d g = (D - lambda * Eigen::Matrix3d::Identity()).fullPivLu().solve(d);
  x.resize(n);
  x.head(n - 3) = N11_inv * (A1.transpose() * b - A1.transpose() * (A2 * g));
  x.tail(3) = g;
  if (std::abs(g.norm() - gravity_magnitude) > kMaxGravityError) {
    std::ostringstream text;
    text << "gravity magnitude error " << std::abs(g.norm() - gravity_magnitude) << " > " << kMaxGravityError;
    reason = text.str();
    return false;
  }
  return true;
}

// ============================================================================
// Moving start ([2]), step by step. Tracks are passed as a map from track id to {times, pixels, bearings}.
// ============================================================================

// Image times that carry a pose of the moving start, and the tracks that observe them
struct PoseWindow {
  std::set<double> times;                                    // Ascending; the newest image time is one of them
  std::vector<int> track_ids;                                // Tracks used by the linear system
  int num_rows = 0;                                          // Two rows per kept observation
  double first = std::numeric_limits<double>::infinity();    // Earliest pose time
};

/**
 * A track keeps an observation that is at least W / (num_poses + 1) away from every pose time chosen so far and from
 * its own kept observations, or that lies exactly on a pose time. A track with fewer kept observations than the window
 * length in whole seconds is not used; the kept observations of the others become pose times.
 */
template <typename Tracks>
PoseWindow ChoosePoseTimes(const Tracks& tracks, double newest, double window_time, int num_poses) {
  const int min_kept = static_cast<int>(window_time);
  const double spacing = window_time / static_cast<double>(num_poses + 1);
  PoseWindow window;
  window.times.insert(newest);
  for (const auto& entry : tracks) {
    std::vector<double> kept;
    for (double t : entry.second.times) {
      double gap = std::numeric_limits<double>::infinity();
      for (double s : window.times) {
        gap = std::min(gap, std::abs(t - s));
      }
      for (double s : kept) {
        gap = std::min(gap, std::abs(t - s));
      }
      if (gap >= spacing || gap == 0.0) {
        kept.push_back(t);
      }
    }
    if (static_cast<int>(kept.size()) < min_kept) {
      continue;
    }
    window.times.insert(kept.begin(), kept.end());
    if (!kept.empty()) {
      window.first = std::min(window.first, *std::min_element(kept.begin(), kept.end()));
    }
    window.num_rows += 2 * static_cast<int>(kept.size());
    window.track_ids.push_back(entry.first);
  }
  return window;
}

// Preintegration to every pose time from the first pose (for the linear system) and from the previous pose (for the
// refinement); null at the first pose
struct PoseChain {
  std::map<double, std::shared_ptr<Preintegration>> from_first;
  std::map<double, std::shared_ptr<Preintegration>> from_previous;
};

// False at the first pose interval the IMU does not cover; its end time goes to *uncovered
bool PreintegratePoses(const std::vector<ImuSample>& imu, const PoseWindow& window, const ImuNoise& noise,
                       const Eigen::Vector3d& gyro_bias_lin, const Eigen::Vector3d& accel_bias_lin, PoseChain* chain,
                       double* uncovered) {
  double previous = window.first;
  for (double t : window.times) {
    if (t == window.first) {
      chain->from_first[t] = nullptr;
      chain->from_previous[t] = nullptr;
      continue;
    }
    chain->from_first[t] = Preintegrate(imu, window.first, t, noise, gyro_bias_lin, accel_bias_lin);
    chain->from_previous[t] = Preintegrate(imu, previous, t, noise, gyro_bias_lin, accel_bias_lin);
    if (!chain->from_first[t] || !chain->from_previous[t]) {
      *uncovered = t;
      return false;
    }
    previous = t;
  }
  return true;
}

// Motion from the first pose: (DT, R_I0toIk, alpha, beta), the identity at the first pose
struct FirstPoseMotion {
  double DT = 0.0;
  Eigen::Matrix3d R_I0toIk = Eigen::Matrix3d::Identity();
  Eigen::Vector3d alpha = Eigen::Vector3d::Zero();
  Eigen::Vector3d beta = Eigen::Vector3d::Zero();
};

FirstPoseMotion MotionFromFirstPose(const PoseChain& chain, double t) {
  FirstPoseMotion motion;
  if (const auto& pre = chain.from_first.at(t)) {
    motion.DT = pre->delta_time();
    motion.R_I0toIk = pre->rotation_start_to_end();
    motion.alpha = pre->alpha();
    motion.beta = pre->beta();
  }
  return motion;
}

/**
 * Linear moving start ([2]): unknowns x = [p_F of every used track, v_I0, g_I0] in the first IMU frame I0. The IMU at
 * pose k is at p_k = v DT - 1/2 g DT^2 + alpha_k, and a measured ray b of a point gives two rows
 * T(b) (R_ItoC R_I0toIk (p_F - p_k) + p_IinC) = 0 with the rows of T(b) orthogonal to b. Solved with |g| = G ([3]).
 * False (with the reason) when the rows do not add up or the constrained solution fails.
 */
template <typename Tracks>
bool SolveLinearStart(const Tracks& tracks, const PoseWindow& window, const PoseChain& chain,
                      const Eigen::Matrix3d& R_ItoC, const Eigen::Vector3d& p_IinC, double gravity_magnitude,
                      Eigen::VectorXd* x, InitializerResult& result) {
  const int num_points = static_cast<int>(window.track_ids.size());
  const int v_col = 3 * num_points;
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(window.num_rows, v_col + 6);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(window.num_rows);
  int row = 0;
  for (int f = 0; f < num_points; ++f) {
    const auto& track = tracks.at(window.track_ids[f]);
    for (size_t i = 0; i < track.times.size(); ++i) {
      if (window.times.count(track.times[i]) == 0) {
        continue;
      }
      if (row + 2 > window.num_rows) {
        result.reason = "linear system row count mismatch";
        return false;
      }
      const FirstPoseMotion m = MotionFromFirstPose(chain, track.times[i]);
      const Eigen::Matrix<double, 2, 3> T = OrthogonalRows(track.bearings[i]);
      const Eigen::Matrix<double, 2, 3> Y = T * R_ItoC * m.R_I0toIk;
      A.block<2, 3>(row, 3 * f) = Y;
      A.block<2, 3>(row, v_col) = -m.DT * Y;
      A.block<2, 3>(row, v_col + 3) = 0.5 * m.DT * m.DT * Y;
      b.segment<2>(row) = Y * m.alpha - T * p_IinC;
      row += 2;
    }
  }
  if (row != window.num_rows) {
    result.reason = "linear system row count mismatch";
    return false;
  }
  return SolveGravityConstrainedLeastSquares(A, b, gravity_magnitude, *x, result.gravity_lambda, result.reason);
}

// IMU pose and velocity at a pose time in I0, from the linear solution
struct FramePose {
  Eigen::Matrix3d R_I0toIk;
  Eigen::Vector3d p;
  Eigen::Vector3d v;
};

std::map<double, FramePose> PosesInFirstFrame(const PoseWindow& window, const PoseChain& chain,
                                              const Eigen::Vector3d& v_I0, const Eigen::Vector3d& g_I0) {
  std::map<double, FramePose> poses;
  for (double t : window.times) {
    const FirstPoseMotion m = MotionFromFirstPose(chain, t);
    poses[t] = FramePose{m.R_I0toIk, v_I0 * m.DT - 0.5 * g_I0 * m.DT * m.DT + m.alpha, v_I0 - g_I0 * m.DT + m.beta};
  }
  return poses;
}

// Points of the linear solution that lie in front of every ray that observed them at a pose time
template <typename Tracks>
std::map<int, Eigen::Vector3d> PointsInFront(const Tracks& tracks, const PoseWindow& window, const Eigen::VectorXd& x,
                                             const std::map<double, FramePose>& poses, const Eigen::Matrix3d& R_ItoC,
                                             const Eigen::Vector3d& p_IinC) {
  std::map<int, Eigen::Vector3d> points;
  for (size_t f = 0; f < window.track_ids.size(); ++f) {
    const auto& track = tracks.at(window.track_ids[f]);
    const Eigen::Vector3d p_F = x.segment<3>(3 * static_cast<Eigen::Index>(f));
    bool in_front = true;
    for (size_t i = 0; i < track.times.size() && in_front; ++i) {
      const auto pose = poses.find(track.times[i]);
      if (pose != poses.end()) {
        const Eigen::Vector3d p_C = R_ItoC * pose->second.R_I0toIk * (p_F - pose->second.p) + p_IinC;
        in_front = track.bearings[i].dot(p_C) > 0.0;
      }
    }
    if (in_front) {
      points[window.track_ids[f]] = p_F;
    }
  }
  return points;
}

/**
 * Maximum-likelihood refinement of the moving start ([2]) in the gravity-aligned frame G (origin at I0): every pose
 * [q, p, v, b_g, b_a], the points seen at pose times, a prior on the first pose (yaw, position, biases), preintegrated
 * IMU between consecutive poses ([1]) and reprojections with a Cauchy loss. The newest pose with its marginal
 * covariance (scaled per block) becomes the initial state. False (with the reason) when a factor is invalid, the
 * refinement does not converge, or the covariance cannot be recovered.
 */
template <typename Tracks>
bool RefineStart(const Tracks& tracks, const PoseWindow& window, const PoseChain& chain,
                 const std::map<double, FramePose>& poses, const std::map<int, Eigen::Vector3d>& points_I0,
                 double newest, double gravity_magnitude, const Camera& camera, const Eigen::Matrix3d& R_ItoC,
                 const Eigen::Vector3d& p_IinC, const Eigen::Matrix3d& R_GtoI0, const InitializerOptions& opt,
                 InitializerResult& result) {
  constexpr double kPriorSigmaYaw = 1e-5;        // First-pose gauge [rad]
  constexpr double kPriorSigmaPosition = 1e-5;   // First-pose gauge [m]
  constexpr double kPriorSigmaGyroBias = 0.05;   // [rad/s]
  constexpr double kPriorSigmaAccelBias = 0.10;  // [m/s^2]
  const Eigen::Vector3d gravity_G(0.0, 0.0, gravity_magnitude);

  // All parameter blocks live in one array in a fixed order: pose by pose [q, p, v, b_g, b_a], then the points.
  // Ceres orders the blocks of the covariance by memory address, so a fixed layout keeps the result independent of
  // where the memory lands.
  constexpr size_t kPoseSize = 16;
  size_t num_points = 0;
  for (int id : window.track_ids) {
    const auto& times = tracks.at(id).times;
    if (points_I0.count(id) > 0 &&
        std::any_of(times.begin(), times.end(), [&](double t) { return window.times.count(t) > 0; })) {
      ++num_points;
    }
  }
  std::vector<double> parameters(kPoseSize * window.times.size() + 3 * num_points);
  struct PoseBlocks {
    double* q;
    double* p;
    double* v;
    double* bg;
    double* ba;
  };
  auto pose_blocks = [&parameters](size_t index) {
    double* base = parameters.data() + kPoseSize * index;
    return PoseBlocks{base, base + 4, base + 7, base + 10, base + 13};
  };

  ceres::Problem problem;
  std::map<double, size_t> pose_of_time;
  for (double t : window.times) {
    const size_t k = pose_of_time.size();
    pose_of_time[t] = k;
    const FramePose& pose = poses.at(t);
    const PoseBlocks blocks = pose_blocks(k);
    Eigen::Map<Eigen::Vector4d>(blocks.q) = jpl::FromRotation(pose.R_I0toIk * R_GtoI0);
    Eigen::Map<Eigen::Vector3d>(blocks.p) = R_GtoI0.transpose() * pose.p;
    Eigen::Map<Eigen::Vector3d>(blocks.v) = R_GtoI0.transpose() * pose.v;
    Eigen::Map<Eigen::Vector3d>(blocks.bg) = opt.gyro_bias_prior;
    Eigen::Map<Eigen::Vector3d>(blocks.ba) = opt.accel_bias_prior;
    problem.AddParameterBlock(blocks.q, 4, new JplQuaternionParameterization());
    for (double* block : {blocks.p, blocks.v, blocks.bg, blocks.ba}) {
      problem.AddParameterBlock(block, 3);
    }
    if (k == 0) {
      auto* prior = new FirstPosePrior(Eigen::Map<const Eigen::Vector4d>(blocks.q),
                                       Eigen::Map<const Eigen::Vector3d>(blocks.p), opt.gyro_bias_prior,
                                       opt.accel_bias_prior, kPriorSigmaYaw, kPriorSigmaPosition, kPriorSigmaGyroBias,
                                       kPriorSigmaAccelBias);
      problem.AddResidualBlock(prior, nullptr, std::vector<double*>{blocks.q, blocks.p, blocks.bg, blocks.ba});
      continue;
    }
    auto* imu = new ImuPreintegrationCost(*chain.from_previous.at(t), gravity_G);
    if (!imu->IsValid()) {
      delete imu;
      result.reason = "preintegration covariance is not positive definite";
      return false;
    }
    const PoseBlocks previous = pose_blocks(k - 1);
    problem.AddResidualBlock(imu, nullptr,
                             std::vector<double*>{previous.q, previous.p, previous.v, previous.bg, previous.ba,
                                                  blocks.q, blocks.p, blocks.v, blocks.bg, blocks.ba});
  }

  // Points: a block at the first observation at a pose time, then one reprojection per such observation
  double* next_point = parameters.data() + kPoseSize * window.times.size();
  std::map<int, double*> point_blocks;
  for (int id : window.track_ids) {
    const auto point_I0 = points_I0.find(id);
    if (point_I0 == points_I0.end()) {
      continue;
    }
    const auto& track = tracks.at(id);
    for (size_t i = 0; i < track.times.size(); ++i) {
      const auto pose = pose_of_time.find(track.times[i]);
      if (pose == pose_of_time.end()) {
        continue;
      }
      auto block = point_blocks.find(id);
      if (block == point_blocks.end()) {
        Eigen::Map<Eigen::Vector3d> p_FinG(next_point);
        p_FinG = R_GtoI0.transpose() * point_I0->second;
        block = point_blocks.emplace(id, next_point).first;
        next_point += 3;
        problem.AddParameterBlock(block->second, 3);
      }
      const PoseBlocks at = pose_blocks(pose->second);
      problem.AddResidualBlock(new ReprojectionCost(&camera, R_ItoC, p_IinC, track.pixels[i], opt.pixel_sigma),
                               new ceres::CauchyLoss(1.0), std::vector<double*>{at.q, at.p, block->second});
    }
  }
  result.num_refined_features = static_cast<int>(point_blocks.size());

  ceres::Solver::Options solver_options;
  solver_options.linear_solver_type = ceres::DENSE_SCHUR;
  solver_options.trust_region_strategy_type = ceres::DOGLEG;
  solver_options.num_threads = 1;  // Single thread and no time limit keep replays deterministic
  solver_options.max_num_iterations = opt.refine_max_iterations;
  solver_options.function_tolerance = 1e-5;
  solver_options.gradient_tolerance = 1e-4 * solver_options.function_tolerance;
  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);
  result.refine_iterations = static_cast<int>(summary.iterations.size());
  result.refine_initial_cost = summary.initial_cost;
  result.refine_final_cost = summary.final_cost;
  if (opt.refine_max_iterations != 0 && summary.termination_type != ceres::CONVERGENCE) {
    result.reason = "refinement did not converge: " + summary.message;
    return false;
  }

  // Marginal covariance of the newest pose in the tangent space [theta, p, v, b_g, b_a]
  const PoseBlocks last = pose_blocks(window.times.size() - 1);
  const std::array<const double*, 5> blocks{last.q, last.p, last.v, last.bg, last.ba};
  std::vector<std::pair<const double*, const double*>> pairs;
  for (size_t i = 0; i < blocks.size(); ++i) {
    for (size_t j = i; j < blocks.size(); ++j) {
      pairs.emplace_back(blocks[i], blocks[j]);
    }
  }
  ceres::Covariance::Options covariance_options;
  covariance_options.min_reciprocal_condition_number = opt.min_covariance_rcond;
  covariance_options.apply_loss_function = true;
  covariance_options.num_threads = 1;
  ceres::Covariance covariance(covariance_options);
  if (!covariance.Compute(pairs, &problem)) {
    result.reason = "covariance recovery failed";
    return false;
  }
  Matrix15 P = Matrix15::Zero();
  for (size_t i = 0; i < blocks.size(); ++i) {
    for (size_t j = i; j < blocks.size(); ++j) {
      Eigen::Matrix<double, 3, 3, Eigen::RowMajor> block;
      covariance.GetCovarianceBlockInTangentSpace(blocks[i], blocks[j], block.data());
      P.block<3, 3>(3 * i, 3 * j) = block;
      P.block<3, 3>(3 * j, 3 * i) = block.transpose();
    }
  }
  P.block<3, 3>(0, 0) *= opt.covariance_scale_orientation;
  P.block<3, 3>(6, 6) *= opt.covariance_scale_velocity;
  P.block<3, 3>(9, 9) *= opt.covariance_scale_gyro_bias;
  P.block<3, 3>(12, 12) *= opt.covariance_scale_accel_bias;
  result.covariance = 0.5 * (P + P.transpose());

  // The newest pose is the initial state; the position gauge is reset to the origin
  result.timestamp = newest;
  result.q_GtoI = Eigen::Map<const Eigen::Vector4d>(last.q);
  result.p_IinG.setZero();
  result.v_IinG = Eigen::Map<const Eigen::Vector3d>(last.v);
  result.gyro_bias = Eigen::Map<const Eigen::Vector3d>(last.bg);
  result.accel_bias = Eigen::Map<const Eigen::Vector3d>(last.ba);
  return true;
}

// Body rotation R_{I(t0) <- I(t1)} of the gyro from t0 to t1 (the rates interpolated at both ends; identity where the
// buffer does not cover the interval)
Eigen::Matrix3d GyroRotationBetween(const std::vector<ImuSample>& imu, double t0, double t1,
                                    const Eigen::Vector3d& gyro_bias) {
  const std::vector<ImuSample> knots = ImuSamplesOverInterval(imu, t0, t1, false);
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  for (size_t k = 0; k + 1 < knots.size(); ++k) {
    const Eigen::Vector3d w = 0.5 * (knots[k].gyro + knots[k + 1].gyro) - gyro_bias;
    R = R * ExpSO3(w * (knots[k + 1].timestamp - knots[k].timestamp));
  }
  return R;
}

}  // namespace

// ============================================================================
// Initializer
// ============================================================================

bool InitializerOptions::Check() const {
  REQUIRE_OPTION(window_time > 0.0);
  REQUIRE_OPTION(excitation_accel_std >= 0.0);
  REQUIRE_OPTION(max_still_disparity >= 0.0);
  REQUIRE_OPTION(max_features > 0);
  REQUIRE_OPTION(refine_max_iterations >= 0);
  REQUIRE_OPTION(num_poses >= 2);
  REQUIRE_OPTION(min_rotation_deg >= 0.0);
  REQUIRE_OPTION(covariance_scale_orientation > 0.0 && covariance_scale_velocity > 0.0);
  REQUIRE_OPTION(covariance_scale_gyro_bias > 0.0 && covariance_scale_accel_bias > 0.0);
  REQUIRE_OPTION(min_covariance_rcond > 0.0);
  REQUIRE_OPTION(pixel_sigma > 0.0);
  REQUIRE_OPTION(rest_accel_bias_sigma >= 0.0);
  return true;
}

Initializer::Initializer(const InitializerOptions& options, const Camera& camera, const Rig& rig,
                         const ImuNoise& imu_noise, double gravity_magnitude)
    : options_(options),
      camera_(camera),
      R_ItoC_(rig.R_ItoC),
      p_IinC_(rig.p_IinC),
      imu_noise_(imu_noise),
      gravity_magnitude_(gravity_magnitude) {}

void Initializer::AddImu(const ImuSample& sample) {
  imu_.push_back(sample);

  // Keep the samples the initializer can still use
  const double oldest = sample.timestamp - options_.window_time - 0.10;
  size_t drop = 0;
  while (drop < imu_.size() && imu_[drop].timestamp < oldest) {
    ++drop;
  }
  imu_.erase(imu_.begin(), imu_.begin() + static_cast<long>(drop));
}

void Initializer::AddImage(Timestamp timestamp, const std::vector<PixelObservation>& observations) {
  for (const auto& observation : observations) {
    Track& track = tracks_[observation.id];
    track.times.push_back(timestamp);
    track.pixels.push_back(observation.pixel);
    track.bearings.push_back(camera_.Unproject(observation.pixel));
  }
}

void Initializer::DropMeasurementsUpTo(double t) {
  for (auto it = tracks_.begin(); it != tracks_.end();) {
    Track& track = it->second;
    size_t keep = 0;
    for (size_t i = 0; i < track.times.size(); ++i) {
      if (track.times[i] > t) {
        track.times[keep] = track.times[i];
        track.pixels[keep] = track.pixels[i];
        track.bearings[keep] = track.bearings[i];
        ++keep;
      }
    }
    track.times.resize(keep);
    track.pixels.resize(keep);
    track.bearings.resize(keep);

    if (keep == 0) {
      it = tracks_.erase(it);
    } else {
      ++it;
    }
  }
}

double Initializer::NewestImageTime() const {
  double newest = -1.0;
  for (const auto& entry : tracks_) {
    for (double t : entry.second.times) {
      newest = std::max(newest, t);
    }
  }
  return newest;
}

void Initializer::MeanTrackMotion(std::optional<double> before, std::optional<double> after, double* mean,
                                  int* count) const {
  // Per track: pixel distance from its first observation after `after` to its last later observation before `before`.
  // Rotation compensated: the first ray is turned by the gyro into the camera of the last observation first, so that
  // the distance measures translation only.
  std::vector<double> disparities;
  std::map<std::pair<double, double>, Eigen::Matrix3d> rotations;  // R_{I(first) <- I(last)} by observation times
  for (const auto& entry : tracks_) {
    const Track& track = entry.second;
    const size_t num = track.times.size();
    if (num < 2) {
      continue;
    }
    size_t first = num;
    for (size_t i = 0; i < num; ++i) {
      if (!after || track.times[i] > *after) {
        first = i;
        break;
      }
    }
    size_t last = num;
    for (size_t i = first + 1; i < num; ++i) {
      if (!before || track.times[i] < *before) {
        last = i;
      }
    }
    if (first == num || last == num) {
      continue;
    }
    Eigen::Vector2d from = track.pixels[first];
    if (options_.rotation_compensated_disparity) {
      const auto key = std::make_pair(track.times[first], track.times[last]);
      auto it = rotations.find(key);
      if (it == rotations.end()) {
        it = rotations.emplace(key, GyroRotationBetween(imu_, key.first, key.second, options_.gyro_bias_prior)).first;
      }
      const Eigen::Vector3d ray = R_ItoC_ * it->second.transpose() * R_ItoC_.transpose() * track.bearings[first];
      from = camera_.Project(ray);
      if (!(from.x() >= 0.0 && from.y() >= 0.0 && from.x() < camera_.width && from.y() < camera_.height)) {
        continue;  // The turned ray leaves the image: no disparity to compare
      }
    }
    const Eigen::Vector2d d = camera_.Residual(track.pixels[last], from);
    disparities.push_back(std::hypot(d.x(), d.y()));
  }

  *count = static_cast<int>(disparities.size());
  *mean = -1.0;
  if (!disparities.empty()) {
    double sum = 0.0;
    for (double value : disparities) {
      sum += value;
    }
    *mean = sum / static_cast<double>(disparities.size());
  }
}

InitializerResult Initializer::TryInitialize() {
  constexpr int kMinDisparityTracks = 15;  // Tracks needed by each disparity statistic

  InitializerResult result;
  const double newest = NewestImageTime();
  const double oldest = newest - options_.window_time - 0.10;
  if (newest < 0.0 || oldest < 0.0) {
    result.reason = "no image window yet";
    return result;
  }

  // Only the window is kept
  DropMeasurementsUpTo(oldest);
  const auto first_kept =
      std::find_if(imu_.begin(), imu_.end(), [&](const ImuSample& s) { return s.timestamp >= oldest; });
  imu_.erase(imu_.begin(), first_kept);
  result.num_tracks = static_cast<int>(tracks_.size());

  // Visual motion of the older and the newer half of the window
  bool moving_old = false;
  bool moving_new = false;
  if (options_.max_still_disparity > 0.0) {
    const double mid = newest - 0.5 * options_.window_time;
    MeanTrackMotion(mid, std::nullopt, &result.disparity_old, &result.disparity_tracks_old);
    MeanTrackMotion(newest, mid, &result.disparity_new, &result.disparity_tracks_new);
    if (result.disparity_tracks_old < kMinDisparityTracks || result.disparity_tracks_new < kMinDisparityTracks) {
      result.reason = "not enough tracks for disparity";
      return result;
    }
    moving_old = result.disparity_old > options_.max_still_disparity;
    moving_new = result.disparity_new > options_.max_still_disparity;
  }

  // Still, then moving (a jerk): start at rest. Moving: the moving start. Still all along: wait.
  const bool jerk = !moving_old && moving_new;
  const bool still = !moving_old && !moving_new;
  if (jerk && options_.excitation_accel_std > 0.0) {
    result.method = "at rest";
    result.success = StartAtRest(result);
    return result;
  }
  if (!still) {
    result.method = "moving";
    result.success = StartMoving(result);
    return result;
  }
  result.reason = "platform still (no visual jerk)";
  return result;
}

bool Initializer::StartAtRest(InitializerResult& result) const {
  if (imu_.size() < 2) {
    result.reason = "not enough IMU samples";
    return false;
  }
  const double t_end = imu_.back().timestamp;
  const double W = options_.window_time;
  if (t_end - imu_.front().timestamp < W) {
    result.reason = "IMU buffer shorter than the window";
    return false;
  }

  // Start at rest ([6]): the older half of the window (t_end - W, t_end - W/2] must be still and the newer half
  // (t_end - W/2, t_end] excited (the platform started moving)
  std::vector<Eigen::Vector3d> accel_newer;
  std::vector<Eigen::Vector3d> accel_older;
  std::vector<Eigen::Vector3d> gyro_older;
  double older_end = -1.0;
  for (const ImuSample& s : imu_) {
    if (s.timestamp > t_end - 0.5 * W && s.timestamp <= t_end) {
      accel_newer.push_back(s.accel);
    }
    if (s.timestamp > t_end - W && s.timestamp <= t_end - 0.5 * W) {
      accel_older.push_back(s.accel);
      gyro_older.push_back(s.gyro);
      older_end = s.timestamp;
    }
  }
  if (accel_newer.size() < 2 || accel_older.size() < 2) {
    result.reason = "not enough IMU samples in a half window";
    return false;
  }
  auto mean_of = [](const std::vector<Eigen::Vector3d>& values) {
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    for (const auto& v : values) {
      sum += v;
    }
    return Eigen::Vector3d(sum / static_cast<double>(values.size()));
  };
  const Eigen::Vector3d a_newer = mean_of(accel_newer);
  const Eigen::Vector3d a_older = mean_of(accel_older);
  const Eigen::Vector3d w_older = mean_of(gyro_older);
  result.accel_std_new = SampleStd(accel_newer, a_newer);
  result.accel_std_old = SampleStd(accel_older, a_older);
  if (result.accel_std_new < options_.excitation_accel_std) {
    result.reason = "no IMU excitation in the newer half";
    return false;
  }
  if (result.accel_std_old > options_.excitation_accel_std) {
    result.reason = "IMU excitation in the older half (was not stationary)";
    return false;
  }

  // At rest the mean specific force is gravity plus the accelerometer bias, the mean rate the gyro bias
  const Eigen::Matrix3d R_GtoI = GravityAlignedRotation(a_older);
  result.timestamp = older_end;
  result.q_GtoI = jpl::FromRotation(R_GtoI);
  result.p_IinG.setZero();
  result.v_IinG.setZero();
  result.gyro_bias = w_older;
  result.accel_bias = a_older - R_GtoI * Eigen::Vector3d(0.0, 0.0, gravity_magnitude_);
  // Sigmas [theta, p, v, b_g, b_a]
  const double sigma[5] = {0.02, 0.05, 0.01, 0.02, 0.02};
  result.covariance.setZero();
  for (int k = 0; k < 5; ++k) {
    result.covariance.block<3, 3>(3 * k, 3 * k) = sigma[k] * sigma[k] * Eigen::Matrix3d::Identity();
  }
  return true;
}

bool Initializer::StartMoving(InitializerResult& result) {
  constexpr int kMinTracks = 8;  // Tracks the moving start must keep

  const InitializerOptions& opt = options_;
  const double newest = NewestImageTime();
  const double oldest = newest - opt.window_time;
  if (newest < 0.0 || oldest < 0.0) {
    result.reason = "no image window yet";
    return false;
  }

  // Only the window is kept; the IMU must reach back before it
  DropMeasurementsUpTo(oldest);
  const auto first_kept =
      std::find_if(imu_.begin(), imu_.end(), [&](const ImuSample& s) { return s.timestamp >= oldest; });
  const bool imu_before_window = first_kept != imu_.begin();
  imu_.erase(imu_.begin(), first_kept);
  result.num_tracks = static_cast<int>(tracks_.size());
  if (tracks_.size() < 0.75 * opt.max_features) {
    result.reason = "only " + std::to_string(tracks_.size()) + " tracks in the window";
    return false;
  }
  if (imu_.size() < 2 || !imu_before_window) {
    result.reason = "IMU window not yet full";
    return false;
  }
  const std::map<int, Track> tracks = tracks_;

  // ---- Pose times and the tracks that observe them
  const PoseWindow window = ChoosePoseTimes(tracks, newest, opt.window_time, opt.num_poses);
  result.num_valid_tracks = static_cast<int>(window.track_ids.size());
  result.num_poses = static_cast<int>(window.times.size());
  if (result.num_poses < opt.num_poses) {
    result.reason = "only " + std::to_string(window.times.size()) + " pose times";
    return false;
  }
  if (result.num_valid_tracks < kMinTracks) {
    result.reason = "only " + std::to_string(window.track_ids.size()) + " valid tracks";
    return false;
  }

  // ---- Rotation over the pose window: the gravity direction needs excitation
  const std::vector<ImuSample> samples = ImuSamplesOverInterval(imu_, window.first, newest, false);
  if (samples.size() <= 2) {
    result.reason = "not enough IMU samples over the pose window";
    return false;
  }
  double rotation = 0.0;
  for (size_t k = 0; k + 1 < samples.size(); ++k) {
    const Eigen::Vector3d w = 0.5 * (samples[k].gyro + samples[k + 1].gyro) - opt.gyro_bias_prior;
    rotation += w.norm() * (samples[k + 1].timestamp - samples[k].timestamp);
  }
  result.rotation_deg = rotation * 180.0 / M_PI;
  if (result.rotation_deg < opt.min_rotation_deg) {
    result.reason = "rotation " + std::to_string(result.rotation_deg) + " deg below threshold";
    return false;
  }
  const int num_points = static_cast<int>(window.track_ids.size());
  if (window.num_rows < 3 * num_points + 6) {
    result.reason = "fewer constraints than unknowns";
    return false;
  }

  // ---- Preintegration along the pose times
  PoseChain chain;
  double uncovered = 0.0;
  if (!PreintegratePoses(imu_, window, imu_noise_, opt.gyro_bias_prior, opt.accel_bias_prior, &chain, &uncovered)) {
    result.reason = "IMU does not cover the pose interval ending at " + std::to_string(uncovered);
    return false;
  }

  // ---- Linear solution in the first IMU frame I0, the poses it gives and its points in front of their rays
  Eigen::VectorXd x;
  if (!SolveLinearStart(tracks, window, chain, R_ItoC_, p_IinC_, gravity_magnitude_, &x, result)) {
    return false;
  }
  const Eigen::Vector3d v_I0 = x.segment<3>(3 * num_points);
  const Eigen::Vector3d g_I0 = x.tail<3>();
  result.gravity_norm_error = std::abs(g_I0.norm() - gravity_magnitude_);
  result.linear_velocity_I0 = v_I0;
  const std::map<double, FramePose> poses = PosesInFirstFrame(window, chain, v_I0, g_I0);
  const std::map<int, Eigen::Vector3d> points = PointsInFront(tracks, window, x, poses, R_ItoC_, p_IinC_);
  if (static_cast<int>(points.size()) < kMinTracks) {
    result.reason = "only " + std::to_string(points.size()) + " features in front of their bearings";
    return false;
  }

  // ---- Refinement in the gravity-aligned frame that shares the origin of I0
  return RefineStart(tracks, window, chain, poses, points, newest, gravity_magnitude_, camera_, R_ItoC_, p_IinC_,
                     GravityAlignedRotation(g_I0), opt, result);
}

}  // namespace vio

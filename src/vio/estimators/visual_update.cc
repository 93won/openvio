// SPDX-License-Identifier: MIT
/**
 * @file      visual_update.cc
 * @brief     Estimator: updates with the tracks leaving the window and with the landmarks.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2026-10-06
 * @copyright Copyright (c) 2026 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 *
 * References:
 *   [1] A. I. Mourikis, S. I. Roumeliotis, "A Multi-State Constraint Kalman Filter for Vision-aided Inertial
 *       Navigation", ICRA 2007, Sec. III-D (measurement model of a track), Sec. III-E (projection onto the left
 *       nullspace of the feature Jacobian; QR compression of the stacked measurements).
 *   [2] M. Li, A. I. Mourikis, "Optimization-Based Estimator Design for Vision-Aided Inertial Navigation", RSS 2012
 *       (features kept in the state next to the window; their initialization from the QR-separated measurements).
 *   [3] M. Li, A. I. Mourikis, "High-precision, consistent EKF-based visual-inertial odometry", IJRR 2013
 *       (Jacobians at first estimates).
 *   [4] G. H. Golub, C. F. Van Loan, "Matrix Computations", 4th ed., 2013, §5.2 (Householder QR).
 *   [5] J. Civera, A. J. Davison, J. M. M. Montiel, "Inverse Depth Parametrization for Monocular SLAM", T-RO 2008
 *       (inverse depth, linearity index).
 */

#include "vio/estimators/estimator.h"

#include "vio/geometry/inverse_depth.h"
#include "vio/geometry/lie_group.h"

#include <algorithm>
#include <cmath>

namespace vio {
namespace {

// Rows of [H_f | H_x | r] turned by Q^T, with Q from the Householder QR of H_f ([4]). Afterwards H_f is its triangular
// factor: the first H_f.cols() rows carry the point, the remaining rows do not depend on it (they span the left
// nullspace of the original H_f).
void RotateByQrOf(Eigen::MatrixXd& H_f, Eigen::MatrixXd& H_x, Eigen::VectorXd& r) {
  const Eigen::HouseholderQR<Eigen::MatrixXd> qr(H_f);
  H_x = (qr.householderQ().adjoint() * H_x).eval();
  r = (qr.householderQ().adjoint() * r).eval();
  H_f = qr.matrixQR().triangularView<Eigen::Upper>();
}

// Measurement of a track without the point error ([1], Sec. III-E): the rows orthogonal to the columns of H_f
void ProjectOutPoint(Eigen::MatrixXd& H_f, Eigen::MatrixXd& H_x, Eigen::VectorXd& r) {
  RotateByQrOf(H_f, H_x, r);
  const Eigen::Index kept = H_x.rows() - H_f.cols();
  H_x = H_x.bottomRows(kept).eval();
  r = r.tail(kept).eval();
}

// More rows than states: with the thin QR H = Q1 R, (R, Q1^T r) gives the same update as (H, r) for isotropic noise
// ([1], Sec. III-E)
void CompressRows(Eigen::MatrixXd& H, Eigen::VectorXd& r) {
  const Eigen::Index n = H.cols();
  if (H.rows() <= n) {
    return;
  }
  const Eigen::HouseholderQR<Eigen::MatrixXd> qr(H);
  const Eigen::VectorXd rotated = qr.householderQ().adjoint() * r;
  r = rotated.head(n);
  H = qr.matrixQR().topRows(n).triangularView<Eigen::Upper>();
}

}  // namespace

Eigen::Vector3d Estimator::LandmarkPosition(const Landmark& landmark, const Eigen::Vector3d& theta) {
  return InverseDepthPoint(landmark.c, landmark.R_BtoG, theta);
}

Eigen::Matrix3d Estimator::LandmarkJacobian(const Landmark& landmark, const Eigen::Vector3d& theta) {
  return InverseDepthJacobian(landmark.R_BtoG, theta);
}

std::vector<int> Estimator::CloneStates(int index) {
  return {index, index + 1, index + 2, index + 3, index + 4, index + 5};
}

std::vector<int> Estimator::TailStates() const {
  std::vector<int> states{kSensorTimeOffset};
  for (int k = 0; k < CalibrationSize(); ++k) {
    states.push_back(kExtrinsic + k);
  }
  return states;
}

bool Estimator::LinearizeMeasurement(size_t c, const Eigen::Vector3d& bearing, const Eigen::Vector2d& pixel,
                                     const Eigen::Vector3d& p_FinG, const Eigen::Vector3d& p_FinG_fej,
                                     MeasurementJacobian* m, int camera) const {
  // Residual at the current estimate ([1], Sec. III-D), from the pose of the sensor that saw the ray
  Eigen::Matrix3d R_GtoI;
  Eigen::Vector3d p_IinG;
  SensorPose(c, bearing, false, R_GtoI, p_IinG);
  const Eigen::Vector3d p_FinI = R_GtoI * (p_FinG - p_IinG);
  if (camera == 0) {
    m->r = camera_.Residual(pixel, camera_.Project(rig_.R_ItoC * p_FinI + rig_.p_IinC));
  } else {
    m->r = camera1_.Residual(pixel, camera1_.Project(rig_.R_C0toC1 * (rig_.R_ItoC * p_FinI + rig_.p_IinC) +
                                                     rig_.p_C0inC1));
  }

  // Jacobians at the first estimates ([3])
  Eigen::Matrix3d R_GtoI_fej;
  Eigen::Vector3d p_IinG_fej;
  const SensorMotion* motion = SensorPose(c, bearing, true, R_GtoI_fej, p_IinG_fej);
  const Eigen::Vector3d p_FinI_fej = R_GtoI_fej * (p_FinG_fej - p_IinG_fej);
  const Eigen::Vector3d p_FinC_fej = rig_.R_ItoC * p_FinI_fej + rig_.p_IinC;
  Eigen::Matrix<double, 2, 3> J;  // d pixel / d p_FinC (camera-0 frame)
  if (camera == 0) {
    if (!camera_.ProjectionJacobian(p_FinC_fej, &J)) {
      return false;
    }
  } else {
    // Camera 1: p_C1 = R_C0toC1 p_C0 + p_C0inC1, so d pixel / d p_C0 = (d pixel / d p_C1) R_C0toC1, and every
    // Jacobian below follows from the camera-0 point as for camera 0
    Eigen::Matrix<double, 2, 3> J1;
    if (!camera1_.ProjectionJacobian(rig_.R_C0toC1 * p_FinC_fej + rig_.p_C0inC1, &J1)) {
      return false;
    }
    J = J1 * rig_.R_C0toC1;
  }
  // R_GtoI = Exp(-dtheta) R_GtoI_hat turns p_FinI by [p_FinI]x dtheta; the IMU position enters with the opposite sign
  // of the point
  const Eigen::Matrix<double, 2, 3> d_theta = J * rig_.R_ItoC * Skew(p_FinI_fej);
  m->H_point = J * (rig_.R_ItoC * R_GtoI_fej);
  const Eigen::Matrix<double, 2, 3> d_position = -m->H_point;
  if (motion == nullptr) {
    m->H_clone << d_theta, d_position;
    m->H_sensor.setZero();
  } else {
    // Sensor 1: theta(1) = R_{I(1) <- I(0)} theta(clone) + w_1 ddt, p(1) = p(clone) + v ddt
    m->H_clone << d_theta * motion->R_back_to_front.transpose(), d_position;
    m->H_sensor = d_theta * motion->w_back + d_position * clones_[c].v;
  }

  // Calibration: p_C = Exp(-dphi) R_ItoC p_I + p_IinC + dp, so d p_C / d dphi = [R_ItoC p_I]x and d p_C / d dp = I;
  // then the online parameters of the camera model
  m->H_calibration = Eigen::MatrixXd::Zero(2, CalibrationSize());
  if (ExtrinsicSize() > 0) {
    m->H_calibration.leftCols<3>() = J * Skew(rig_.R_ItoC * p_FinI_fej);
    m->H_calibration.middleCols<3>(3) = J;
  }
  if (CameraParamSize() > 0 && camera == 0) {
    // d pixel / d params of every model parameter; the online ones lead (the camera-1 model has none online)
    Eigen::Matrix<double, 2, Eigen::Dynamic> d_params(2, static_cast<Eigen::Index>(camera_.params.size()));
    camera_.Project(p_FinC_fej, d_params.data());
    m->H_calibration.middleCols(ExtrinsicSize(), CameraParamSize()) = d_params.leftCols(CameraParamSize());
  }
  return true;
}

CameraPose Estimator::CameraPoseOf(int camera, const Eigen::Matrix3d& R_GtoI, const Eigen::Vector3d& p_IinG) const {
  const Eigen::Matrix3d R_GtoC = rig_.R_ItoC * R_GtoI;
  if (camera == 0) {
    return {R_GtoC, p_IinG - R_GtoC.transpose() * rig_.p_IinC};
  }
  // Camera 1: p_C1 = R_C0toC1 (R_GtoC (p - p_IinG) + p_IinC) + p_C0inC1 is zero at its center
  const Eigen::Matrix3d R_GtoC1 = rig_.R_C0toC1 * R_GtoC;
  return {R_GtoC1, p_IinG - R_GtoC1.transpose() * (rig_.R_C0toC1 * rig_.p_IinC + rig_.p_C0inC1)};
}

bool Estimator::LinearizeTrack(const Track& track, const Eigen::Vector3d& p_FinG, TrackSystem* system) const {
  const int num = static_cast<int>(track.times.size());

  // Clone of every measurement; the clones get their columns in order of first appearance
  std::vector<size_t> clone_of(num);
  std::map<size_t, int> column_of;
  system->states.clear();
  for (int k = 0; k < num; ++k) {
    const auto it = std::find_if(clones_.begin(), clones_.end(),
                                 [&](const Clone& clone) { return clone.t == track.times[k]; });
    if (it == clones_.end()) {
      return false;
    }
    clone_of[k] = static_cast<size_t>(it - clones_.begin());
    if (column_of.emplace(clone_of[k], static_cast<int>(system->states.size())).second) {
      const std::vector<int> block = CloneStates(it->index);
      system->states.insert(system->states.end(), block.begin(), block.end());
    }
  }
  const int tail_column = static_cast<int>(system->states.size());
  const std::vector<int> tail = TailStates();
  system->states.insert(system->states.end(), tail.begin(), tail.end());

  system->H_f = Eigen::MatrixXd::Zero(2 * num, 3);
  system->H_x = Eigen::MatrixXd::Zero(2 * num, static_cast<Eigen::Index>(system->states.size()));
  system->r = Eigen::VectorXd::Zero(2 * num);
  MeasurementJacobian m;
  for (int k = 0; k < num; ++k) {
    if (!LinearizeMeasurement(clone_of[k], track.bearings[k], track.pixels[k], p_FinG, p_FinG, &m,
                              track.cameras[k])) {
      return false;
    }
    system->r.segment<2>(2 * k) = m.r;
    system->H_f.block<2, 3>(2 * k, 0) = m.H_point;
    system->H_x.block<2, 6>(2 * k, column_of[clone_of[k]]) = m.H_clone;
    system->H_x.block<2, 1>(2 * k, tail_column) = m.H_sensor;
    system->H_x.block(2 * k, tail_column + 1, 2, CalibrationSize()) = m.H_calibration;
  }
  return true;
}

bool Estimator::TriangulateTrack(Track& track, TrackReport& report, Eigen::Vector3d& p_FinG) {
  // Only measurements at clone times constrain the window
  size_t keep = 0;
  for (size_t i = 0; i < track.times.size(); ++i) {
    const bool at_clone =
        std::any_of(clones_.begin(), clones_.end(), [&](const Clone& clone) { return clone.t == track.times[i]; });
    if (at_clone) {
      track.times[keep] = track.times[i];
      track.pixels[keep] = track.pixels[i];
      track.bearings[keep] = track.bearings[i];
      track.cameras[keep] = track.cameras[i];
      ++keep;
    }
  }
  track.times.resize(keep);
  track.pixels.resize(keep);
  track.bearings.resize(keep);
  track.cameras.resize(keep);
  // Measurements in at least two images: the two cameras of one stereo image only see the point, not the motion
  if (keep < 2 || track.times.front() == track.times.back()) {
    statistics_.too_few_measurements++;
    report.outcome = TrackOutcome::kTooFew;
    return false;
  }

  // Camera poses at the exposure of the measuring sensor, from the current clone estimates
  std::vector<CameraPose> poses;
  for (size_t i = 0; i < track.times.size(); ++i) {
    for (size_t c = 0; c < clones_.size(); ++c) {
      if (clones_[c].t == track.times[i]) {
        Eigen::Matrix3d R_GtoI;
        Eigen::Vector3d p_IinG;
        SensorPose(c, track.bearings[i], false, R_GtoI, p_IinG);
        poses.push_back(CameraPoseOf(track.cameras[i], R_GtoI, p_IinG));
        break;
      }
    }
  }
  const TriangulationStatus status = TriangulatePoint(options_.triangulation, track.bearings, poses, &p_FinG);
  report.triangulation = status;
  if (status != TriangulationStatus::kSuccess) {
    report.outcome = TrackOutcome::kTriangulation;
    statistics_.triangulation_failed++;
    switch (status) {
      case TriangulationStatus::kIllConditioned:
        statistics_.triangulation_ill_conditioned++;
        break;
      case TriangulationStatus::kLinearDepth:
        statistics_.triangulation_linear_depth++;
        break;
      case TriangulationStatus::kRefinedDepth:
        statistics_.triangulation_refined_depth++;
        break;
      case TriangulationStatus::kBaseline:
        statistics_.triangulation_baseline++;
        break;
      default:
        statistics_.triangulation_not_finite++;
        break;
    }
    return false;
  }
  report.has_point = true;
  report.point = p_FinG;
  return true;
}

bool Estimator::PassesGate(const Eigen::MatrixXd& H, const Eigen::VectorXd& r, const std::vector<int>& states,
                           TrackReport& report) {
  // Mahalanobis distance of the residual with the marginal covariance of the involved states
  Eigen::MatrixXd S = H * P_(states, states) * H.transpose();
  S.diagonal().array() += options_.pixel_sigma * options_.pixel_sigma;
  const double chi2 = r.dot(S.llt().solve(r));
  const int dof = static_cast<int>(r.size());
  const double threshold = (dof < static_cast<int>(chi2_table_.size())) ? chi2_table_[dof]
                                                                         : ChiSquaredQuantile(dof, 0.95);
  statistics_.gate_tested++;
  statistics_.nis_sum += chi2 / dof;
  report.nis = chi2 / dof;
  if (chi2 > threshold) {
    statistics_.gate_rejected++;
    report.outcome = TrackOutcome::kGate;
    return false;
  }
  return true;
}

void Estimator::UpdateWithTracks(const std::vector<int>& ids) {
  // Accepted tracks, each projected onto the nullspace of its point ([1], Sec. III-E)
  std::vector<TrackSystem> accepted;
  int total_rows = 0;
  for (int id : ids) {
    Track& track = tracks_.at(id);
    track_reports_.push_back(TrackReport());
    TrackReport& report = track_reports_.back();
    report.id = id;

    Eigen::Vector3d p_FinG;
    if (!TriangulateTrack(track, report, p_FinG)) {
      continue;
    }
    TrackSystem system;
    if (!LinearizeTrack(track, p_FinG, &system)) {
      statistics_.projection_failed++;
      report.outcome = TrackOutcome::kProjection;
      continue;
    }
    ProjectOutPoint(system.H_f, system.H_x, system.r);
    if (!PassesGate(system.H_x, system.r, system.states, report)) {
      continue;
    }
    statistics_.nis_accepted_sum += report.nis;
    report.outcome = TrackOutcome::kAccepted;
    total_rows += static_cast<int>(system.r.size());
    accepted.push_back(std::move(system));
    statistics_.accepted++;
  }
  if (total_rows < 1) {
    return;
  }

  // One system over the union of the involved states: clone blocks in order of first appearance, then the tail
  std::vector<int> states;
  std::map<int, int> column_of;  // Covariance index -> column
  const std::vector<int> tail = TailStates();
  for (const TrackSystem& system : accepted) {
    for (size_t k = 0; k + tail.size() < system.states.size(); ++k) {
      if (column_of.emplace(system.states[k], static_cast<int>(states.size())).second) {
        states.push_back(system.states[k]);
      }
    }
  }
  for (int state : tail) {
    column_of.emplace(state, static_cast<int>(states.size()));
    states.push_back(state);
  }
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(total_rows, static_cast<Eigen::Index>(states.size()));
  Eigen::VectorXd r(total_rows);
  int row = 0;
  for (const TrackSystem& system : accepted) {
    const int rows = static_cast<int>(system.r.size());
    for (size_t k = 0; k < system.states.size(); ++k) {
      H.block(row, column_of.at(system.states[k]), rows, 1) = system.H_x.col(static_cast<Eigen::Index>(k));
    }
    r.segment(row, rows) = system.r;
    row += rows;
  }

  CompressRows(H, r);
  if (H.rows() < 1) {
    return;
  }
  statistics_.update_rows = static_cast<int>(H.rows());
  SparseJacobian jacobian;
  jacobian.shared = std::move(states);
  jacobian.H_shared = std::move(H);
  KalmanUpdate(jacobian, r);
}

bool Estimator::UpdateLandmarks() {
  // Every landmark is observed at most once per image, at the newest clone
  const size_t newest = clones_.size() - 1;
  const Clone& clone = clones_[newest];
  const double sigma2 = options_.pixel_sigma * options_.pixel_sigma;

  // Mean squared inverse depth of the landmarks (the information of a ray on translation scales with rho^2)
  double mean_rho2 = 0.0;
  for (const auto& entry : landmarks_) {
    mean_rho2 += entry.second.theta(2) * entry.second.theta(2);
  }
  if (!landmarks_.empty()) {
    mean_rho2 /= static_cast<double>(landmarks_.size());
  }

  // Accepted measurements: Jacobian over the shared states [newest clone, sensor-1 time offset, calibration] and over
  // the landmark's own states
  const std::vector<int> tail = TailStates();
  std::vector<int> shared = CloneStates(clone.index);
  shared.insert(shared.end(), tail.begin(), tail.end());
  const Eigen::Index num_shared = static_cast<Eigen::Index>(shared.size());
  std::vector<int> remove;
  std::vector<Eigen::Matrix<double, 2, Eigen::Dynamic>> H_shared_rows;
  std::vector<Eigen::Matrix<double, 2, 3>> H_own_rows;
  std::vector<int> own_index;
  std::vector<Eigen::Vector2d> r_rows;
  for (auto& entry : landmarks_) {
    const int id = entry.first;
    Landmark& landmark = entry.second;
    const auto it = tracks_.find(id);
    const bool observed = it != tracks_.end() && !it->second.times.empty() && it->second.times.back() == clone.t;
    const bool behind = landmark.theta(2) <= 0.0;
    if (!observed || behind) {
      // A lost landmark stays in the state for a while (dormant), so that it can be found again by prediction
      if (!observed && !behind && options_.landmark_dormant_time > 0.0 &&
          clone.t - landmark.last_seen <= options_.landmark_dormant_time) {
        landmark.dormant = true;
        statistics_.landmarks_dormant++;
        continue;
      }
      statistics_.landmarks_lost++;
      remove.push_back(id);
      continue;
    }

    // A first estimate far from the current estimate scales the translation part of every Jacobian of the
    // landmark by the same wrong factor (it is proportional to rho). Weighted by the landmark's share of the
    // translation information (rho^2 relative to the mean), a large error biases the metric scale:
    // remove the landmark, its track can start a new one
    if (options_.landmark_relinearize_ratio > 0.0 && mean_rho2 > 0.0 &&
        std::abs(landmark.theta(2) / landmark.theta_fej(2) - 1.0) * landmark.theta(2) * landmark.theta(2) >
            options_.landmark_relinearize_ratio * mean_rho2) {
      statistics_.landmarks_relinearized++;
      remove.push_back(id);
      continue;
    }
    track_reports_.push_back(TrackReport());
    TrackReport& report = track_reports_.back();
    report.id = id;
    report.has_point = true;
    const Eigen::Vector3d p_FinG = LandmarkPosition(landmark, landmark.theta);
    report.point = p_FinG;

    // Measurements of this image: camera 0, then camera 1 of a stereo rig; each has its own chi-square gate
    const Track& track = it->second;
    size_t first = track.times.size() - 1;
    while (first > 0 && track.times[first - 1] == clone.t) {
      --first;
    }
    const Eigen::Vector3d p_FinG_fej = LandmarkPosition(landmark, landmark.theta_fej);
    // Jacobian over [clone, landmark, sensor-1 time offset, calibration]
    std::vector<int> states = CloneStates(clone.index);
    for (int k = 0; k < 3; ++k) {
      states.push_back(landmark.index + k);
    }
    states.insert(states.end(), tail.begin(), tail.end());
    bool projection_failed = false;
    int accepted = 0;
    for (size_t i = first; i < track.times.size(); ++i) {
      MeasurementJacobian m;
      if (!LinearizeMeasurement(newest, track.bearings[i], track.pixels[i], p_FinG, p_FinG_fej, &m,
                                track.cameras[i])) {
        // Camera 0 decides about the landmark; a camera-1 measurement is only skipped
        projection_failed = projection_failed || track.cameras[i] == 0;
        if (projection_failed) {
          break;
        }
        continue;
      }
      Eigen::MatrixXd H(2, static_cast<Eigen::Index>(states.size()));
      H.leftCols<6>() = m.H_clone;
      H.middleCols<3>(6) = m.H_point * LandmarkJacobian(landmark, landmark.theta_fej);
      H.col(9) = m.H_sensor;
      H.rightCols(CalibrationSize()) = m.H_calibration;
      Eigen::Matrix2d S = H * P_(states, states) * H.transpose();
      S.diagonal().array() += sigma2;
      const double chi2 = m.r.dot(S.llt().solve(m.r));
      report.nis = chi2 / 2.0;
      if (chi2 > chi2_table_[2]) {
        continue;
      }
      Eigen::Matrix<double, 2, Eigen::Dynamic> H_shared_row(2, num_shared);
      H_shared_row << H.leftCols<6>(), H.rightCols(num_shared - 6);
      H_shared_rows.push_back(std::move(H_shared_row));
      H_own_rows.push_back(H.middleCols<3>(6));
      own_index.push_back(landmark.index);
      r_rows.push_back(m.r);
      ++accepted;
    }
    if (projection_failed) {
      statistics_.projection_failed++;
      report.outcome = TrackOutcome::kProjection;
      remove.push_back(id);
      continue;
    }
    if (accepted == 0) {
      // The measurements are rejected; the landmark only after max_gate_rejections consecutive rejections
      statistics_.landmarks_gate_rejected++;
      report.outcome = TrackOutcome::kGate;
      if (++landmark.rejections >= std::max(1, options_.landmark_max_gate_rejections)) {
        remove.push_back(id);
      }
      continue;
    }
    landmark.rejections = 0;
    report.outcome = TrackOutcome::kLandmark;
    statistics_.landmarks_updated++;
    if (landmark.dormant) {
      statistics_.landmarks_found_again++;
      landmark.dormant = false;
    }
    landmark.last_seen = clone.t;
    landmark.updates++;
  }

  if (!r_rows.empty()) {
    const Eigen::Index rows = 2 * static_cast<Eigen::Index>(r_rows.size());
    SparseJacobian jacobian;
    jacobian.shared = shared;
    jacobian.H_shared.resize(rows, num_shared);
    jacobian.block_rows = 2;
    jacobian.block_cols = 3;
    jacobian.block_index = own_index;
    jacobian.H_blocks.resize(rows, 3);
    Eigen::VectorXd r(rows);
    for (size_t k = 0; k < r_rows.size(); ++k) {
      jacobian.H_shared.middleRows(2 * k, 2) = H_shared_rows[k];
      jacobian.H_blocks.middleRows(2 * k, 2) = H_own_rows[k];
      r.segment<2>(2 * k) = r_rows[k];
    }
    if (!KalmanUpdate(jacobian, r)) {
      return false;
    }
  }

  // The new measurements are consumed; lost and rejected landmarks leave the state
  for (const auto& entry : landmarks_) {
    tracks_.erase(entry.first);
  }
  RemoveLandmarks(remove, clone.t);
  return true;
}

bool Estimator::InitializeLandmarks(const std::vector<int>& ids) {
  const double sigma2 = options_.pixel_sigma * options_.pixel_sigma;
  for (int id : ids) {
    Track& track = tracks_.at(id);
    track.used = true;
    track_reports_.push_back(TrackReport());
    TrackReport& report = track_reports_.back();
    report.id = id;

    Eigen::Vector3d p_FinG;
    if (!TriangulateTrack(track, report, p_FinG)) {
      continue;
    }
    TrackSystem system;
    if (!LinearizeTrack(track, p_FinG, &system)) {
      statistics_.projection_failed++;
      report.outcome = TrackOutcome::kProjection;
      continue;
    }

    // Separate the measurements ([2]): after the QR rotation the first three rows constrain the point and the state,
    // r1 = H_x1 dx + H_f1 dp + n1, the remaining rows only the state
    RotateByQrOf(system.H_f, system.H_x, system.r);
    const Eigen::Index rest = system.r.size() - 3;
    const Eigen::MatrixXd H_x2 = system.H_x.bottomRows(rest);
    const Eigen::VectorXd r2 = system.r.tail(rest);
    if (!PassesGate(H_x2, r2, system.states, report)) {
      continue;
    }
    const Eigen::Matrix3d H_f1 = system.H_f.topRows<3>();
    if (std::abs(H_f1.diagonal().prod()) < 1e-12 * std::pow(H_f1.norm(), 3)) {
      statistics_.triangulation_failed++;
      report.outcome = TrackOutcome::kTriangulation;
      continue;
    }

    // dp = H_f1^-1 (r1 - H_x1 dx - n1) gives the point's covariance and its cross covariance with the state (H_x1 is
    // nonzero only at the track's states)
    const Eigen::MatrixXd H_x1 = system.H_x.topRows(3);
    const Eigen::Matrix3d H_f1_inv = H_f1.triangularView<Eigen::Upper>().solve(Eigen::Matrix3d::Identity());
    const Eigen::MatrixXd H_x1_P = H_x1 * P_(system.states, Eigen::all);
    const Eigen::MatrixXd P_fx = -H_f1_inv * H_x1_P;
    const Eigen::Matrix3d P_ff =
        H_f1_inv * (H_x1_P(Eigen::all, system.states) * H_x1.transpose() + sigma2 * Eigen::Matrix3d::Identity()) *
        H_f1_inv.transpose();
    SparseJacobian jacobian;
    jacobian.shared = system.states;
    jacobian.H_shared = H_x2;

    // Inverse depth from the current camera center along the ray of the point ([5]). The triangulated point is the
    // estimate given the current clones, and it is also the linearization point of the Jacobians above: using it as
    // the first estimate keeps the unobservable directions (yaw, global position) of this initialization in later
    // updates.
    const Clone& newest = clones_.back();
    Landmark landmark;
    landmark.c = newest.p - jpl::ToRotation(newest.q).transpose() * rig_.R_ItoC.transpose() * rig_.p_IinC;
    const Eigen::Vector3d ray = p_FinG - landmark.c;
    const double depth = ray.norm();
    const Eigen::Matrix3d R = RotationToPlusZ(ray / depth);
    landmark.R_BtoG = ((R * ray).z() > 0.0) ? Eigen::Matrix3d(R.transpose()) : R;

    // How well the observation geometry determines the depth: linearity index L = 4 sigma_d / d ([5]), with sigma_d
    // from the pixel noise alone (the camera poses of the window are known relative to each other far better than
    // globally)
    const Eigen::Matrix3d P_geometry = sigma2 * H_f1_inv * H_f1_inv.transpose();
    const double sigma_depth = std::sqrt(std::max(0.0, ray.dot(P_geometry * ray))) / depth;
    const double linearity = 4.0 * sigma_depth / depth;

    // A landmark must constrain the translation. A point whose depth the window does not determine only repeats the
    // rotation the gyro gives, and over the long life of a landmark it follows any slow motion of what it sees
    // (clouds drifting in the wind turn the estimate). Such a track updates the window like a leaving track instead.
    if (options_.landmark_max_linearity > 0.0 && linearity > options_.landmark_max_linearity) {
      statistics_.landmarks_undetermined++;
      statistics_.nis_accepted_sum += report.nis;
      statistics_.accepted++;
      report.outcome = TrackOutcome::kAccepted;
      if (!KalmanUpdate(jacobian, r2)) {
        return false;
      }
      continue;
    }
    landmark.theta = Eigen::Vector3d(0.0, 0.0, 1.0 / depth);
    landmark.theta_fej = landmark.theta;

    // Augmented covariance: the point covariance turned to (alpha, beta, rho) to first order
    const Eigen::Matrix3d J_inv = LandmarkJacobian(landmark, landmark.theta).inverse();
    const Eigen::MatrixXd P_tx = J_inv * P_fx;
    const Eigen::Matrix3d P_tt = J_inv * P_ff * J_inv.transpose();
    const int n = static_cast<int>(P_.rows());
    Eigen::MatrixXd augmented(n + 3, n + 3);
    augmented.topLeftCorner(n, n) = P_;
    augmented.bottomLeftCorner(3, n) = P_tx;
    augmented.topRightCorner(n, 3) = P_tx.transpose();
    augmented.bottomRightCorner<3, 3>() = 0.5 * (P_tt + P_tt.transpose());
    P_.swap(augmented);

    landmark.index = n;
    landmark.last_seen = newest.t;
    landmark.t_init = newest.t;
    landmarks_[id] = landmark;
    statistics_.landmarks_initialized++;
    report.outcome = TrackOutcome::kLandmark;
    report.point = p_FinG;

    // The remaining rows update the state (their landmark columns are zero)
    if (!KalmanUpdate(jacobian, r2)) {
      return false;
    }
  }
  return true;
}

void Estimator::RemoveLandmarks(const std::vector<int>& ids, double t) {
  if (ids.empty()) {
    return;
  }
  std::vector<std::pair<int, int>> blocks;
  for (int id : ids) {
    const Landmark& landmark = landmarks_.at(id);
    const double life = t - landmark.t_init;
    statistics_.landmarks_removed++;
    statistics_.landmark_life_sum += life;
    statistics_.landmark_updates_sum += landmark.updates;
    constexpr double kEdges[5] = {1.0, 2.0, 5.0, 10.0, 30.0};
    statistics_.landmark_life_histogram[std::upper_bound(kEdges, kEdges + 5, life) - kEdges]++;
    blocks.emplace_back(landmark.index, 3);
    landmarks_.erase(id);
  }
  RemoveFromCovariance(std::move(blocks));
}

}  // namespace vio

// SPDX-License-Identifier: MIT
/**
 * @file      estimator.cc
 * @brief     Estimator: initialization, the per-image order, the EKF update and the window.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2026-09-28
 * @copyright Copyright (c) 2026 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "vio/estimators/estimator.h"

#include "vio/geometry/lie_group.h"
#include "vio/util/logging.h"

#include <algorithm>
#include <cmath>

namespace vio {
namespace {

/**
 * Upper tail Q(x | k) = P(chi2 with k degrees of freedom > x) for an integer k, from its closed forms
 * (M. Abramowitz, I. A. Stegun, Handbook of Mathematical Functions, 1964, §26.4), with h = x / 2:
 *   k even: Q = exp(-h) sum_{j=0}^{k/2-1} h^j / j!
 *   k odd:  Q = erfc(sqrt(h)) + exp(-h) sum_{j=1}^{(k-1)/2} h^(j-1/2) / Gamma(j+1/2)
 * Consecutive terms differ by the factor h / j (even) or h / (j - 1/2) (odd). The terms are carried as logarithms
 * when exp(-h) would underflow.
 */
double ChiSquaredUpperTail(int k, double x) {
  if (x <= 0.0) {
    return 1.0;
  }
  const double h = 0.5 * x;
  const bool even = (k % 2 == 0);
  const int num_terms = even ? k / 2 : (k - 1) / 2;
  double tail = even ? 0.0 : std::erfc(std::sqrt(h));
  if (num_terms == 0) {
    return tail;
  }
  // First term: exp(-h) for even k, exp(-h) h^(1/2) / Gamma(3/2) for odd k (Gamma(3/2) = sqrt(pi) / 2); the n-th
  // following term is the previous one times h / (n + shift)
  const double first_log = even ? -h : -h + 0.5 * std::log(h) - std::log(0.5 * std::sqrt(M_PI));
  const double shift = even ? 0.0 : 0.5;
  if (h < 700.0) {
    double term = std::exp(first_log);
    tail += term;
    for (int n = 1; n < num_terms; ++n) {
      term *= h / (n + shift);
      tail += term;
    }
    return tail;
  }
  double log_term = first_log;
  tail += std::exp(log_term);
  for (int n = 1; n < num_terms; ++n) {
    log_term += std::log(h / (n + shift));
    tail += std::exp(log_term);
  }
  return tail;
}

}  // namespace

bool EstimatorOptions::Check() const {
  REQUIRE_OPTION(max_clones >= 2);
  REQUIRE_OPTION(max_landmarks >= 0);
  REQUIRE_OPTION(landmark_max_linearity >= 0.0);
  REQUIRE_OPTION(landmark_relinearize_ratio >= 0.0);
  REQUIRE_OPTION(landmark_max_gate_rejections >= 0);
  REQUIRE_OPTION(landmark_dormant_time >= 0.0);
  REQUIRE_OPTION(pixel_sigma > 0.0);
  REQUIRE_OPTION(time_offset_sigma >= 0.0);
  REQUIRE_OPTION(time_offset_walk >= 0.0);
  REQUIRE_OPTION(extrinsic_rotation_sigma >= 0.0);
  REQUIRE_OPTION(extrinsic_translation_sigma >= 0.0);
  REQUIRE_OPTION(camera_param_sigma >= 0.0);
  return triangulation.Check();
}

double Estimator::ChiSquaredQuantile(int dof, double probability) {
  // The tail falls monotonically with x: bracket the point where it reaches 1 - probability, then halve the bracket
  // until it cannot shrink in double precision
  const double tail = 1.0 - probability;
  double below = 0.0;
  double above = std::max(1.0, 2.0 * dof);
  while (ChiSquaredUpperTail(dof, above) > tail) {
    above *= 2.0;
  }
  for (;;) {
    const double middle = 0.5 * (below + above);
    if (!(middle > below && middle < above)) {
      return above;
    }
    if (ChiSquaredUpperTail(dof, middle) > tail) {
      below = middle;
    } else {
      above = middle;
    }
  }
}

Estimator::Estimator(const EstimatorOptions& options, const Camera& camera, const Rig& rig, const ImuNoise& imu_noise,
                     double gravity_magnitude, const Camera& camera1)
    : options_(options),
      imu_noise_(imu_noise),
      gravity_magnitude_(gravity_magnitude),
      initial_camera_(camera),
      initial_rig_(rig),
      initialized_(false),
      time_(0.0),
      q_(0.0, 0.0, 0.0, 1.0),
      p_(Eigen::Vector3d::Zero()),
      v_(Eigen::Vector3d::Zero()),
      bg_(Eigen::Vector3d::Zero()),
      ba_(Eigen::Vector3d::Zero()),
      rig_(rig),
      camera_(camera),
      camera1_(camera1),
      q_fej_(0.0, 0.0, 0.0, 1.0),
      p_fej_(Eigen::Vector3d::Zero()),
      v_fej_(Eigen::Vector3d::Zero()) {
  rig_.time_offset = 0.0;
  rig_.sensor_time_offset = 0.0;

  // 95% quantiles for the gate, precomputed for common sizes
  chi2_table_.assign(500, 0.0);
  for (int dof = 1; dof < 500; ++dof) {
    chi2_table_[dof] = ChiSquaredQuantile(dof, 0.95);
  }
}

void Estimator::Initialize(Timestamp timestamp, const Eigen::Vector4d& q_GtoI, const Eigen::Vector3d& p_IinG,
                           const Eigen::Vector3d& v_IinG, const Eigen::Vector3d& gyro_bias,
                           const Eigen::Vector3d& accel_bias, const Eigen::Matrix<double, 15, 15>& covariance) {
  time_ = timestamp;
  q_ = q_GtoI;
  p_ = p_IinG;
  v_ = v_IinG;
  bg_ = gyro_bias;
  ba_ = accel_bias;
  rig_ = initial_rig_;
  rig_.time_offset = 0.0;
  rig_.sensor_time_offset = 0.0;
  camera_ = initial_camera_;
  q_fej_ = q_;
  p_fej_ = p_;
  v_fej_ = v_;
  const int size = kSensorTimeOffset + 1 + CalibrationSize();
  P_ = Eigen::MatrixXd::Zero(size, size);
  P_.topLeftCorner<15, 15>() = covariance;
  P_(kTimeOffset, kTimeOffset) = options_.time_offset_sigma * options_.time_offset_sigma;
  P_(kSensorTimeOffset, kSensorTimeOffset) = options_.time_offset_sigma * options_.time_offset_sigma;
  if (ExtrinsicSize() > 0) {
    const double s_rot = options_.extrinsic_rotation_sigma;
    const double s_pos = options_.extrinsic_translation_sigma;
    P_.block<3, 3>(kExtrinsic, kExtrinsic) = s_rot * s_rot * Eigen::Matrix3d::Identity();
    P_.block<3, 3>(kExtrinsic + 3, kExtrinsic + 3) = s_pos * s_pos * Eigen::Matrix3d::Identity();
  }
  for (int k = 0; k < CameraParamSize(); ++k) {
    const int param = kExtrinsic + ExtrinsicSize() + k;
    P_(param, param) = options_.camera_param_sigma * options_.camera_param_sigma;
  }
  clones_.clear();
  sensor_motion_.clear();
  landmarks_.clear();
  tracks_.clear();
  failure_.clear();
  initialized_ = true;
}

void Estimator::SetTimeOffsets(double time_offset, double sensor_time_offset) {
  time_ += time_offset - rig_.time_offset;
  rig_.time_offset = time_offset;
  rig_.sensor_time_offset = sensor_time_offset;
}

bool Estimator::UpdateAccelBiasComponent(const Eigen::Vector3d& direction, double value, double sigma) {
  // KalmanUpdate uses the pixel noise: scale the row so that the noise becomes sigma
  const double scale = options_.pixel_sigma / sigma;
  SparseJacobian H;
  H.shared = {12, 13, 14};
  H.H_shared = scale * direction.transpose();
  Eigen::VectorXd r(1);
  r(0) = scale * (value - direction.dot(ba_));
  return KalmanUpdate(H, r);
}

void Estimator::AddImu(const ImuSample& sample) { imu_.push_back(sample); }

bool Estimator::KalmanUpdate(const SparseJacobian& H, const Eigen::VectorXd& r) {
  // Kalman update with the pixel noise on every row (P. S. Maybeck, "Stochastic Models, Estimation, and Control",
  // Vol. 1, 1979, Ch. 5): S = H P H^T + sigma^2 I = L L^T, K = P H^T S^-1. With W = P H^T L^-T the update is
  // P <- P - W W^T and dx = K r = W L^-1 r. P H^T and H P H^T use only the nonzero columns of H; the upper triangle
  // of P is updated and mirrored.
  const Eigen::Index n = P_.rows();
  const Eigen::Index m = r.size();
  const int block_rows = H.block_rows;
  const int block_cols = H.block_cols;
  Eigen::MatrixXd PHt(n, m);
  Eigen::MatrixXd S(m, m);
  if (!H.shared.empty()) {
    PHt.noalias() = P_(Eigen::all, H.shared) * H.H_shared.transpose();
  } else {
    PHt.setZero();
  }
  for (size_t k = 0; k < H.block_index.size(); ++k) {
    const Eigen::Index row = static_cast<Eigen::Index>(k) * block_rows;
    PHt.middleCols(row, block_rows).noalias() +=
        P_.middleCols(H.block_index[k], block_cols) * H.H_blocks.middleRows(row, block_rows).transpose();
  }
  if (!H.shared.empty()) {
    S.noalias() = H.H_shared * PHt(H.shared, Eigen::all);
  } else {
    S.setZero();
  }
  for (size_t k = 0; k < H.block_index.size(); ++k) {
    const Eigen::Index row = static_cast<Eigen::Index>(k) * block_rows;
    S.middleRows(row, block_rows).noalias() +=
        H.H_blocks.middleRows(row, block_rows) * PHt.middleRows(H.block_index[k], block_cols);
  }
  S.diagonal().array() += options_.pixel_sigma * options_.pixel_sigma;
  S = 0.5 * (S + S.transpose()).eval();
  const Eigen::LLT<Eigen::MatrixXd> llt(S);
  if (llt.info() != Eigen::Success) {
    failure_ = "innovation covariance not positive definite";
    return false;
  }
  Eigen::MatrixXd& W = PHt;
  llt.matrixU().solveInPlace<Eigen::OnTheRight>(W);
  P_.selfadjointView<Eigen::Upper>().rankUpdate(W, -1.0);
  P_.triangularView<Eigen::StrictlyLower>() = P_.transpose();
  if ((P_.diagonal().array() < 0.0).any()) {
    failure_ = "negative covariance diagonal after update";
    return false;
  }
  ApplyCorrection(W * llt.matrixL().solve(r));
  return true;
}

void Estimator::ApplyCorrection(const Eigen::VectorXd& dx) {
  // Orientations: q <- dq (x) q with the small-angle quaternion dq = normalize([dtheta / 2; 1]) (Trawny & Roumeliotis
  // 2005); the camera-IMU rotation R_ItoC <- Exp(-dphi) R_ItoC; all other states are additive
  auto turn = [](const Eigen::Vector3d& dtheta, const Eigen::Vector4d& q) {
    Eigen::Vector4d dq;
    dq << 0.5 * dtheta, 1.0;
    return jpl::Multiply(jpl::NormalizePositive(dq), q);
  };
  q_ = turn(dx.segment<3>(0), q_);
  p_ += dx.segment<3>(3);
  v_ += dx.segment<3>(6);
  bg_ += dx.segment<3>(9);
  ba_ += dx.segment<3>(12);
  rig_.time_offset += dx(kTimeOffset);
  rig_.sensor_time_offset += dx(kSensorTimeOffset);
  if (ExtrinsicSize() > 0) {
    rig_.R_ItoC = ExpSO3(-dx.segment<3>(kExtrinsic)) * rig_.R_ItoC;
    rig_.p_IinC += dx.segment<3>(kExtrinsic + 3);
  }
  for (int k = 0; k < CameraParamSize(); ++k) {
    camera_.params[k] += dx(kExtrinsic + ExtrinsicSize() + k);
  }
  for (auto& clone : clones_) {
    clone.q = turn(dx.segment<3>(clone.index), clone.q);
    clone.p += dx.segment<3>(clone.index + 3);
  }
  for (auto& entry : landmarks_) {
    entry.second.theta += dx.segment<3>(entry.second.index);
  }
}

void Estimator::RemoveFromCovariance(std::vector<std::pair<int, int>> blocks) {
  // The remaining states keep their order: the kept index ranges between the removed blocks are copied pairwise in
  // one pass
  std::sort(blocks.begin(), blocks.end());
  const int n = static_cast<int>(P_.rows());
  std::vector<std::pair<int, int>> kept;  // (start, length) in P_
  int start = 0;
  int removed = 0;
  for (const auto& block : blocks) {
    if (block.first > start) {
      kept.emplace_back(start, block.first - start);
    }
    start = block.first + block.second;
    removed += block.second;
  }
  if (n > start) {
    kept.emplace_back(start, n - start);
  }
  Eigen::MatrixXd reduced(n - removed, n - removed);
  int col = 0;
  for (const auto& kc : kept) {
    int row = 0;
    for (const auto& kr : kept) {
      reduced.block(row, col, kr.second, kc.second) = P_.block(kr.first, kc.first, kr.second, kc.second);
      row += kr.second;
    }
    col += kc.second;
  }
  P_.swap(reduced);

  // Every index moves down by the sizes of the removed blocks before it
  auto new_index = [&blocks](int index) {
    int shift = 0;
    for (const auto& block : blocks) {
      if (block.first < index) {
        shift += block.second;
      }
    }
    return index - shift;
  };
  for (auto& clone : clones_) {
    clone.index = new_index(clone.index);
  }
  for (auto& entry : landmarks_) {
    entry.second.index = new_index(entry.second.index);
  }
}

void Estimator::RemoveClone(size_t c) {
  const int index = clones_[c].index;
  clones_.erase(clones_.begin() + static_cast<long>(c));
  sensor_motion_.erase(sensor_motion_.begin() + static_cast<long>(c));
  RemoveFromCovariance({{index, 6}});
}

Estimator::TrackSelection Estimator::SelectTracks(Timestamp timestamp) {
  // Tracks that reach back to the oldest clone are used once the window holds more than max_clones or more than five
  // clones (the oldest clone is about to leave)
  const int num_clones = static_cast<int>(clones_.size());
  const double oldest = clones_.front().t;
  const bool use_oldest = num_clones > options_.max_clones || num_clones > 5;

  // Unused tracks that are not landmarks, by what the window update can do with them
  std::vector<int> ended;        // Not observed in this image
  std::vector<int> leaving;      // Reach back to the oldest clone
  std::vector<int> long_tracks;  // Reach back to the oldest clone, observed in more images than the window holds:
                                 // can become landmarks
  for (const auto& entry : tracks_) {
    const int id = entry.first;
    const Track& track = entry.second;
    if (track.used || track.times.empty() || landmarks_.count(id) > 0) {
      continue;
    }
    const bool reaches_oldest =
        use_oldest && std::find(track.times.begin(), track.times.end(), oldest) != track.times.end();
    const bool observed = track.times.back() == timestamp;
    if (observed && track.frames > options_.max_clones && reaches_oldest) {
      long_tracks.push_back(id);
    } else if (reaches_oldest) {
      leaving.push_back(id);
    } else if (!observed) {
      ended.push_back(id);
    }
  }

  // Long tracks become landmarks while the state has room. With dormant landmarks, those not seen in this image make
  // room, the longest unseen first.
  int room = options_.max_landmarks - static_cast<int>(landmarks_.size());
  if (options_.landmark_dormant_time > 0.0 && room < static_cast<int>(long_tracks.size())) {
    std::vector<std::pair<double, int>> unseen;  // (time of the last update, id)
    for (const auto& entry : landmarks_) {
      const auto it = tracks_.find(entry.first);
      const bool seen = it != tracks_.end() && !it->second.times.empty() && it->second.times.back() == timestamp;
      if (!seen) {
        unseen.emplace_back(entry.second.last_seen, entry.first);
      }
    }
    std::sort(unseen.begin(), unseen.end());
    std::vector<int> make_room;
    for (size_t k = 0; k < unseen.size() && room < static_cast<int>(long_tracks.size()); ++k) {
      make_room.push_back(unseen[k].second);
      statistics_.landmarks_lost++;
      ++room;
    }
    RemoveLandmarks(make_room, timestamp);
  }
  TrackSelection selection;
  const auto first_rest = long_tracks.begin() + std::clamp(room, 0, static_cast<int>(long_tracks.size()));
  selection.landmarks.assign(long_tracks.begin(), first_rest);

  // Window update: the ended and leaving tracks, and the long tracks without room; shortest first
  selection.window = ended;
  selection.window.insert(selection.window.end(), leaving.begin(), leaving.end());
  selection.window.insert(selection.window.end(), first_rest, long_tracks.end());
  std::stable_sort(selection.window.begin(), selection.window.end(),
                   [this](int a, int b) { return tracks_.at(a).times.size() < tracks_.at(b).times.size(); });
  return selection;
}

bool Estimator::ProcessImage(Timestamp timestamp, const std::vector<PixelObservation>& observations) {
  statistics_ = EstimatorStatistics();
  track_reports_.clear();
  marginalized_.reset();
  if (!initialized_) {
    failure_ = "estimator not initialized";
    return false;
  }

  // New measurements enter the track database before the update (stereo: a camera-1 measurement follows the camera-0
  // one of its track; the image count and the keyframe report follow camera 0)
  std::vector<PixelObservation> camera0_observations;
  std::vector<Eigen::Vector3d> rays;
  camera0_observations.reserve(observations.size());
  rays.reserve(observations.size());
  for (const auto& observation : observations) {
    Track& track = tracks_[observation.id];
    track.times.push_back(timestamp);
    track.pixels.push_back(observation.pixel);
    track.cameras.push_back(observation.camera);
    if (observation.camera == 0) {
      track.bearings.push_back(camera_.Unproject(observation.pixel));
      track.frames++;
      camera0_observations.push_back(observation);
      rays.push_back(track.bearings.back());
    } else {
      track.bearings.push_back(camera1_.Unproject(observation.pixel));
    }
  }

  if (!PropagateToImage(timestamp)) {
    return false;
  }
  clones_.back().observations = std::move(camera0_observations);
  clones_.back().rays = std::move(rays);

  // The first updates wait for min(max_clones, 5) clones, so that a track has a baseline to be triangulated from
  if (static_cast<int>(clones_.size()) < std::min(options_.max_clones, 5)) {
    return true;
  }
  const TrackSelection selection = SelectTracks(timestamp);
  const std::vector<int>& candidates = selection.window;
  const std::vector<int>& new_landmarks = selection.landmarks;
  statistics_.candidates = static_cast<int>(candidates.size());

  UpdateWithTracks(candidates);
  if (!failure_.empty()) {
    return false;
  }
  if (!UpdateLandmarks()) {
    return false;
  }
  if (!InitializeLandmarks(new_landmarks)) {
    return false;
  }
  statistics_.landmarks_in_state = static_cast<int>(landmarks_.size());

  // Consumed tracks are never reused
  for (int id : candidates) {
    tracks_.at(id).used = true;
  }
  for (auto it = tracks_.begin(); it != tracks_.end();) {
    if (it->second.used) {
      it = tracks_.erase(it);
    } else {
      ++it;
    }
  }

  // Slide the window: the oldest clone leaves, with the measurements at or before it
  if (static_cast<int>(clones_.size()) > options_.max_clones) {
    const double t_remove = clones_.front().t;
    for (auto it = tracks_.begin(); it != tracks_.end();) {
      Track& track = it->second;
      size_t keep = 0;
      for (size_t i = 0; i < track.times.size(); ++i) {
        if (track.times[i] > t_remove) {
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
      if (keep == 0) {
        it = tracks_.erase(it);
      } else {
        ++it;
      }
    }
    // The oldest clone leaves with its final estimate: a keyframe
    Clone& clone = clones_.front();
    Keyframe keyframe;
    keyframe.timestamp = clone.t;
    keyframe.capture_time = clone.t_capture;
    keyframe.q_GtoI = clone.q;
    keyframe.p_IinG = clone.p;
    keyframe.covariance = P_.block<6, 6>(clone.index, clone.index);
    keyframe.observations = std::move(clone.observations);
    keyframe.rays = std::move(clone.rays);
    marginalized_ = std::move(keyframe);
    RemoveClone(0);
  }

  // Keep one second of IMU before the state time, and back to the oldest clone (its sensor-1 motion is recomputed
  // from the gyro when the sensor time offset changes)
  const double t_keep = std::min(time_ - 1.0, clones_.front().t_capture - 0.1);
  size_t drop = 0;
  while (drop + 2 < imu_.size() && imu_[drop + 1].timestamp < t_keep) {
    ++drop;
  }
  imu_.erase(imu_.begin(), imu_.begin() + static_cast<long>(drop));
  return true;
}

std::vector<std::pair<Eigen::Vector4d, Eigen::Vector3d>> Estimator::ClonePoses() const {
  std::vector<std::pair<Eigen::Vector4d, Eigen::Vector3d>> poses;
  poses.reserve(clones_.size());
  for (const auto& clone : clones_) {
    poses.emplace_back(clone.q, clone.p);
  }
  return poses;
}

std::vector<std::pair<int, Eigen::Vector3d>> Estimator::Landmarks() const {
  std::vector<std::pair<int, Eigen::Vector3d>> landmarks;
  landmarks.reserve(landmarks_.size());
  for (const auto& entry : landmarks_) {
    landmarks.emplace_back(entry.first, LandmarkPosition(entry.second, entry.second.theta));
  }
  return landmarks;
}

std::vector<PixelObservation> Estimator::PredictLandmarks(Timestamp timestamp, Eigen::Matrix3d* R_GtoC) const {
  std::vector<PixelObservation> predicted;
  *R_GtoC = rig_.R_ItoC * jpl::ToRotation(q_);
  if (!initialized_) {
    return predicted;
  }
  // IMU pose at the exposure: R_GtoI(t) = R_{I(state) <- I(t)}^T R_GtoI(state), p(t) = p + v (t - t_state)
  const double t_capture = timestamp + rig_.time_offset;
  Eigen::Vector3d w_end;
  const Eigen::Matrix3d R_GtoI = GyroRotation(time_, t_capture, w_end).transpose() * jpl::ToRotation(q_);
  const Eigen::Vector3d p_IinG = p_ + v_ * (t_capture - time_);
  *R_GtoC = rig_.R_ItoC * R_GtoI;
  for (const auto& entry : landmarks_) {
    const Eigen::Vector3d p_FinI = R_GtoI * (LandmarkPosition(entry.second, entry.second.theta) - p_IinG);
    const Eigen::Vector3d p_FinC = rig_.R_ItoC * p_FinI + rig_.p_IinC;
    if (p_FinC.norm() > 1e-6) {
      predicted.push_back({entry.first, camera_.Project(p_FinC)});
    }
  }
  return predicted;
}

}  // namespace vio

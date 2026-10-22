// SPDX-License-Identifier: MIT
/**
 * @file      estimator.h
 * @brief     Sliding-window Kalman filter of the IMU state, the camera rig and landmarks.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2026-09-28
 * @copyright Copyright (c) 2026 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 *
 * The window keeps the IMU pose of the last images (clones). A track that leaves the window is triangulated and
 * constrains the clones after its point is projected out (nullspace projection); tracks that span the whole window
 * can become landmarks in the state, updated with every new image. Jacobians are evaluated at first estimates. The
 * IMU mean is integrated with RK4, the covariance with a closed-form transition. Quaternions follow the JPL
 * convention; measurements are image pixels of the camera model. Papers: doc/references.md.
 *
 * State and error order: [theta, p, v, bg, ba] of the IMU, then the rig time offsets: of sensor 0 (an image stamped
 * t was exposed at t + t_d on the IMU clock) and of sensor 1 relative to sensor 0 (a stitched two-lens image
 * combines two exposures that need not be simultaneous; sensor 1 sees the rays with z <= 0, its pose is the clone
 * pose moved by the gyro over the offset, and its position moves with the velocity). When estimated, the camera-IMU
 * rotation and translation follow (R_ItoC = Exp(-dphi) R_ItoC_hat, p_IinC = p_hat + dp), then the online parameters
 * of the camera model (equirectangular: the lens-angle error). Clones [theta, p] and landmarks [alpha, beta, rho]
 * follow in the order they were added; each keeps its own covariance index. Orientation error:
 * R_GtoI = Exp(-dtheta) R_GtoI_hat.
 *
 * Stereo: a measurement can also come from camera 1, fixed to camera 0 by the rig (p_C1 = R_C0toC1 p_C0 + p_C0inC1),
 * so it depends on the same states, the camera-0 extrinsics included. A track holds the measurements of both cameras.
 */

#pragma once

#include "vio/geometry/triangulation.h"
#include "vio/scene/camera.h"
#include "vio/scene/keyframe.h"
#include "vio/sensor/rig.h"
#include "vio/util/types.h"

#include <Eigen/Dense>

#include <deque>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace vio {

struct EstimatorOptions {
  int max_clones = 11;                       // Window size
  int max_landmarks = 0;                     // Landmarks kept in the state (0: none)
  double landmark_max_linearity = 0.0;       // A long track becomes a landmark only when the window determines its
                                             // depth: linearity index 4 sigma_d / d (pixel noise) at most this; else it
                                             // updates the window like a leaving track (0: no limit)
  double landmark_relinearize_ratio = 0.0;   // Remove a landmark when |rho / rho_first_estimate - 1|, weighted by
                                             // rho^2 / mean rho^2 of the landmarks, exceeds this; its track can start
                                             // a new landmark (0: never)
  int landmark_max_gate_rejections = 1;      // Consecutive gate rejections that remove a landmark (a rejection
                                             // otherwise skips only the measurement)
  double landmark_dormant_time = 0.0;        // A landmark whose track is lost stays in the state this long (dormant),
                                             // so that it can be found again at its predicted pixel; the oldest
                                             // dormant ones make room for new landmarks [s] (0: removed when lost)
  double pixel_sigma = 1.0;                  // Pixel noise [px]
  double time_offset_sigma = 0.0;            // Prior sigma of each rig time offset [s] (0: fixed)
  double time_offset_walk = 0.0;             // Random walk of each rig time offset [s/sqrt(s)]
  double extrinsic_rotation_sigma = 0.0;     // Prior sigma of the camera-IMU rotation [rad]
  double extrinsic_translation_sigma = 0.0;  // Prior sigma of the camera-IMU translation [m] (both 0: fixed)
  double camera_param_sigma = 0.0;           // Prior sigma of the online parameters of the camera model (0: fixed)
  TriangulationOptions triangulation;        // Triangulation of the tracks

  bool Check() const;
};

// What happened to a track in the last update
enum class TrackOutcome {
  kAccepted,       // Used by the window update
  kTooFew,         // Measurements at fewer than two clone times
  kTriangulation,  // Rejected by the triangulation (see TrackReport::triangulation)
  kProjection,     // Point where the projection has no derivative
  kGate,           // Rejected by the chi-square gate
  kLandmark,       // Landmark in the state (initialized or updated with this image)
};

// One track in the last update (for display and the landmark tracker)
struct TrackReport {
  int id = -1;
  TrackOutcome outcome = TrackOutcome::kTooFew;
  TriangulationStatus triangulation = TriangulationStatus::kSuccess;
  bool has_point = false;                           // point is valid
  Eigen::Vector3d point = Eigen::Vector3d::Zero();  // Triangulated or landmark position [m]
  double nis = 0.0;                                 // chi2 / dof at the gate
};

// Per-image statistics
struct EstimatorStatistics {
  int candidates = 0;                     // Tracks selected for the window update
  int too_few_measurements = 0;           // Measurements at fewer than two clone times
  int triangulation_failed = 0;           // Rejected by the triangulation (sum of the below)
  int triangulation_ill_conditioned = 0;  // Linear system condition number too large
  int triangulation_linear_depth = 0;     // Linear solution outside the depth limits
  int triangulation_refined_depth = 0;    // Refined solution outside the depth limits
  int triangulation_baseline = 0;         // Distance / baseline ratio too large
  int triangulation_not_finite = 0;       // NaN in the solution
  int projection_failed = 0;              // Point where the projection has no derivative
  int gate_rejected = 0;                  // Rejected by the chi-square gate
  int accepted = 0;                       // Used in the update
  int update_rows = 0;                    // Rows after compression
  int gate_tested = 0;                    // Tracks that reached the gate
  double nis_sum = 0.0;                   // Sum of chi2 / dof over tested tracks (consistency: mean ~ 1)
  double nis_accepted_sum = 0.0;          // Sum of chi2 / dof over accepted tracks
  int landmarks_in_state = 0;             // Landmarks in the state after the image
  int landmarks_updated = 0;              // Landmarks updated with this image
  int landmarks_gate_rejected = 0;        // Landmark measurements rejected by the gate
  int landmarks_lost = 0;                 // Landmarks no longer tracked (removed)
  int landmarks_initialized = 0;          // New landmarks
  int landmarks_undetermined = 0;         // Long tracks not made landmarks (depth undetermined): window updates
  int landmarks_relinearized = 0;         // Landmarks removed because rho left its first estimate
  int landmarks_dormant = 0;              // Landmarks kept in the state without an observation in this image
  int landmarks_found_again = 0;          // Landmarks updated again after images without an observation
  int landmarks_removed = 0;              // Landmarks that left the state (diagnostics of their life below)
  double landmark_life_sum = 0.0;         // Sum of their times in the state [s]
  double landmark_updates_sum = 0.0;      // Sum of their accepted updates
  int landmark_life_histogram[6] = {};    // Their times in the state: < 1, 1-2, 2-5, 5-10, 10-30, >= 30 s
};

/**
 * @brief Sliding-window filter of the visual-inertial state
 *
 * Usage: Initialize() with an initial IMU state, AddImu() for every IMU sample (camera clock), ProcessImage() for every
 * image after the initialization time.
 */
class Estimator {
 public:
  /**
   * @param options Estimator parameters
   * @param camera Camera (its parameters are refined when camera_param_sigma > 0)
   * @param rig Camera-IMU rig (extrinsics and time offsets are refined when their sigmas are > 0)
   * @param imu_noise IMU noise densities
   * @param gravity_magnitude [m/s^2]
   * @param camera1 Stereo: the second camera (model kInvalid: monocular)
   */
  Estimator(const EstimatorOptions& options, const Camera& camera, const Rig& rig, const ImuNoise& imu_noise,
            double gravity_magnitude, const Camera& camera1 = Camera());

  /**
   * @brief Set the initial IMU state and covariance; the rig and the camera start at their configured values, the
   *        time offsets at zero with the prior sigma, uncorrelated
   * @param covariance Covariance [theta, p, v, bg, ba]
   */
  void Initialize(Timestamp timestamp, const Eigen::Vector4d& q_GtoI, const Eigen::Vector3d& p_IinG,
                  const Eigen::Vector3d& v_IinG, const Eigen::Vector3d& gyro_bias, const Eigen::Vector3d& accel_bias,
                  const Eigen::Matrix<double, 15, 15>& covariance);

  /**
   * @brief Start the time offsets at an estimate (right after Initialize(), before the first image); the initial
   *        state is then taken at the sensor-0 exposure of the initialization image
   */
  void SetTimeOffsets(double time_offset, double sensor_time_offset);

  /**
   * @brief Update with one measured component of the accelerometer bias (e.g. from a period at rest)
   * @param direction Unit vector in the IMU frame
   * @param value Measured bias along the direction [m/s^2]
   * @param sigma Measurement sigma [m/s^2]
   * @return false if the update failed (see failure())
   */
  bool UpdateAccelBiasComponent(const Eigen::Vector3d& direction, double value, double sigma);

  // Add one IMU sample (camera clock, time ordered)
  void AddImu(const ImuSample& sample);

  /**
   * @brief Propagate to an image, clone, update with the tracks and landmarks, slide the window
   * @param timestamp Image time [s]
   * @param observations Tracked features of the image
   * @return false if the estimator cannot continue (see failure())
   */
  bool ProcessImage(Timestamp timestamp, const std::vector<PixelObservation>& observations);

  // ============ State ============

  bool IsInitialized() const { return initialized_; }
  Timestamp timestamp() const { return time_; }  // State time: sensor-0 exposure of the last image (IMU clock)
  const Eigen::Vector4d& orientation() const { return q_; }  // q_GtoI
  const Eigen::Vector3d& position() const { return p_; }     // p_IinG
  const Eigen::Vector3d& velocity() const { return v_; }
  const Eigen::Vector3d& gyro_bias() const { return bg_; }
  const Eigen::Vector3d& accel_bias() const { return ba_; }
  const Rig& rig() const { return rig_; }                    // Current rig estimate
  const Camera& camera() const { return camera_; }           // Current camera estimate
  const Eigen::MatrixXd& covariance() const { return P_; }
  int NumClones() const { return static_cast<int>(clones_.size()); }
  const EstimatorStatistics& statistics() const { return statistics_; }
  const std::vector<TrackReport>& track_reports() const { return track_reports_; }
  const std::string& failure() const { return failure_; }

  // The clone that left the window in the last ProcessImage() as the oldest one, with its final estimate
  const std::optional<Keyframe>& marginalized_keyframe() const { return marginalized_; }

  // (q_GtoI, p_IinG) of every clone, oldest first
  std::vector<std::pair<Eigen::Vector4d, Eigen::Vector3d>> ClonePoses() const;

  // (track id, p_FinG) of every landmark
  std::vector<std::pair<int, Eigen::Vector3d>> Landmarks() const;

  /**
   * @brief Predicted pixel of every landmark in the next image, from the state moved to its exposure with the
   *        bias-corrected gyro and the velocity
   * @param timestamp Image time [s]
   * @param R_GtoC Output predicted camera orientation at the exposure
   */
  std::vector<PixelObservation> PredictLandmarks(Timestamp timestamp, Eigen::Matrix3d* R_GtoC) const;

 private:
  static constexpr int kTimeOffset = 15;        // Covariance index of the sensor-0 time offset
  static constexpr int kSensorTimeOffset = 16;  // Covariance index of the sensor-1 time offset
  static constexpr int kExtrinsic = 17;         // Covariance index of the camera-IMU rotation, then translation

  struct Clone {
    double t;                                  // Image time [s]
    double t_capture;                          // Sensor-0 exposure t + t_d on the IMU clock [s]
    Eigen::Vector4d q;                         // q_GtoI
    Eigen::Vector3d p;                         // p_IinG
    Eigen::Vector4d q_fej;                     // First estimate of q_GtoI
    Eigen::Vector3d p_fej;                     // First estimate of p_IinG
    Eigen::Vector3d v;                         // Velocity at the exposure [m/s]
    int index;                                 // Covariance index of [theta, p]
    std::vector<PixelObservation> observations;  // Features of the image (keyframe report)
    std::vector<Eigen::Vector3d> rays;           // Their unit rays
  };

  // Motion of the body from the sensor-0 to the sensor-1 exposure of one clone
  struct SensorMotion {
    double dt = std::numeric_limits<double>::quiet_NaN();           // Sensor time offset it was computed for [s]
    Eigen::Matrix3d R_back_to_front = Eigen::Matrix3d::Identity();  // R_{I(sensor 0) <- I(sensor 1)}
    Eigen::Vector3d w_back = Eigen::Vector3d::Zero();               // Body rate at the sensor-1 exposure [rad/s]
  };

  // A landmark: inverse depth (alpha, beta, rho) from an anchor (geometry/inverse_depth.h)
  struct Landmark {
    Eigen::Vector3d theta;      // (alpha, beta, rho)
    Eigen::Vector3d theta_fej;  // First estimate of theta
    Eigen::Vector3d c;          // Anchor point (constant) [m]
    Eigen::Matrix3d R_BtoG;     // Ray frame (constant)
    int index;                  // Covariance index
    int rejections = 0;         // Consecutive gate rejections
    double last_seen = 0.0;     // Image time of the last update or of the initialization [s]
    bool dormant = false;       // Its track was lost; kept in the state to be found again
    double t_init = 0.0;        // Image time of the initialization [s] (diagnostics)
    int updates = 0;            // Accepted updates (diagnostics)
  };

  // Measurements of one feature track
  struct Track {
    std::vector<double> times;             // Observation times [s]
    std::vector<Eigen::Vector2d> pixels;   // Pixel coordinates
    std::vector<Eigen::Vector3d> bearings; // Unit rays in the frame of the measuring camera
    std::vector<int> cameras;              // Measuring camera (0, or 1 of a stereo rig)
    bool used = false;                     // Consumed by an update (removed at cleanup)
    int frames = 0;                        // Images since the track started
  };

  // Number of camera-IMU extrinsic states [rotation, translation] after the time offsets (0: fixed)
  int ExtrinsicSize() const {
    return (options_.extrinsic_rotation_sigma > 0.0 || options_.extrinsic_translation_sigma > 0.0) ? 6 : 0;
  }
  // Number of camera-parameter states after the extrinsics: the online parameters of the camera model (0: fixed)
  int CameraParamSize() const { return (options_.camera_param_sigma > 0.0) ? camera_.NumOnlineParams() : 0; }
  // Calibration states after the time offsets: extrinsics, then the camera parameters
  int CalibrationSize() const { return ExtrinsicSize() + CameraParamSize(); }

  // ---- imu_propagation.cc ----

  // Propagate the state and covariance to the exposure of an image and append a clone
  bool PropagateToImage(double t);

  // One IMU interval between two samples: the IMU state (and its first estimate) moves to the end of the interval;
  // F is the error-state transition and Qd the discrete noise of the interval
  void PropagateInterval(const ImuSample& start, const ImuSample& end, Eigen::Matrix<double, 15, 15>* F,
                         Eigen::Matrix<double, 15, 15>* Qd);

  // Body rotation R_{I(t0) <- I(t1)} from the bias-corrected gyro (either time order); w_end: body rate at t1
  Eigen::Matrix3d GyroRotation(double t0, double t1, Eigen::Vector3d& w_end) const;

  // Body motion from the sensor-0 to the sensor-1 exposure of clone c (cached while the offset is unchanged)
  const SensorMotion& SensorOneMotion(size_t c) const;

  /**
   * @brief IMU pose when the sensor that measured a ray was exposed
   * @param c Clone index in the window
   * @param ray Measured ray in the camera frame
   * @param first_estimate Move the first estimate of the clone instead of the current estimate
   * @return Motion from the clone to the exposure, nullptr for sensor 0 (the clone pose itself)
   */
  const SensorMotion* SensorPose(size_t c, const Eigen::Vector3d& ray, bool first_estimate, Eigen::Matrix3d& R_GtoI,
                                 Eigen::Vector3d& p_IinG) const;

  // ---- visual_update.cc ----

  // Update with tracks leaving the window, in the given order
  void UpdateWithTracks(const std::vector<int>& ids);

  // Update with the landmarks seen in this image; remove the lost and rejected ones
  bool UpdateLandmarks();

  // Add tracks that span the whole window to the state as landmarks
  bool InitializeLandmarks(const std::vector<int>& ids);

  // Linearized measurements of a track: residual r = H_f dp_F + H_x dx + n, where the columns of H_x belong to the
  // error states listed in `states` (covariance indices): the clones in order of first appearance, then the sensor-1
  // time offset and the calibration states
  struct TrackSystem {
    Eigen::MatrixXd H_f;
    Eigen::MatrixXd H_x;
    Eigen::VectorXd r;
    std::vector<int> states;
  };

  // One pixel measurement of a point from clone c, linearized: residual at the current estimate, Jacobians at the
  // first estimates w.r.t. the point, the clone [theta, p], the sensor-1 time offset and the calibration states
  struct MeasurementJacobian {
    Eigen::Vector2d r;
    Eigen::Matrix<double, 2, 3> H_point;
    Eigen::Matrix<double, 2, 6> H_clone;
    Eigen::Vector2d H_sensor;              // Zero for sensor 0
    Eigen::MatrixXd H_calibration;         // 2 x CalibrationSize()
  };

  // False if the projection has no derivative at the first estimate; camera: the measuring camera (0, or 1 of a
  // stereo rig)
  bool LinearizeMeasurement(size_t c, const Eigen::Vector3d& bearing, const Eigen::Vector2d& pixel,
                            const Eigen::Vector3d& p_FinG, const Eigen::Vector3d& p_FinG_fej, MeasurementJacobian* m,
                            int camera = 0) const;

  // Camera pose of camera k (0, or 1 of a stereo rig) for an IMU pose
  CameraPose CameraPoseOf(int camera, const Eigen::Matrix3d& R_GtoI, const Eigen::Vector3d& p_IinG) const;

  // Covariance indices of the six states of a clone block starting at index
  static std::vector<int> CloneStates(int index);
  // Covariance indices of the sensor-1 time offset and the calibration states
  std::vector<int> TailStates() const;

  // Keep the measurements of a track at clone times and triangulate it
  bool TriangulateTrack(Track& track, TrackReport& report, Eigen::Vector3d& p_FinG);

  // Residual at the current estimate and Jacobians at the first estimates of a track (measurements at clone times);
  // false if the projection has no derivative
  bool LinearizeTrack(const Track& track, const Eigen::Vector3d& p_FinG, TrackSystem* system) const;

  // Chi-square gate of residual r with Jacobian H over the error states `states`
  bool PassesGate(const Eigen::MatrixXd& H, const Eigen::VectorXd& r, const std::vector<int>& states,
                  TrackReport& report);

  // Remove landmarks from the state (one covariance pass) and record their lives in the statistics
  void RemoveLandmarks(const std::vector<int>& ids, double t);

  static Eigen::Vector3d LandmarkPosition(const Landmark& landmark, const Eigen::Vector3d& theta);
  static Eigen::Matrix3d LandmarkJacobian(const Landmark& landmark, const Eigen::Vector3d& theta);

  // ---- estimator.cc ----

  // Measurement Jacobian over the error state that is zero outside a few columns: every row has coefficients over the
  // `shared` states, and every block of block_rows rows has its own block_cols states starting at its block_index (the
  // landmark update: shared = newest clone and tail states, own = the landmark of the block)
  struct SparseJacobian {
    std::vector<int> shared;       // Covariance indices of the shared columns
    Eigen::MatrixXd H_shared;      // rows x shared.size()
    int block_rows = 0;            // Rows per block (0: no own columns)
    int block_cols = 0;            // Own columns per block
    std::vector<int> block_index;  // Covariance index of the first own column of every block
    Eigen::MatrixXd H_blocks;      // rows x block_cols; block k in rows [k block_rows, (k + 1) block_rows)
  };

  // EKF update; false if the covariance lost definiteness
  bool KalmanUpdate(const SparseJacobian& H, const Eigen::VectorXd& r);

  // Add an error-state correction to the state
  void ApplyCorrection(const Eigen::VectorXd& dx);

  // Tracks of one image for the window update (in update order) and for new landmarks
  struct TrackSelection {
    std::vector<int> window;
    std::vector<int> landmarks;
  };

  // Choose the tracks of an image; dormant landmarks may leave the state to make room for new ones
  TrackSelection SelectTracks(Timestamp timestamp);

  // Remove a clone (0: oldest) from the state
  void RemoveClone(size_t c);

  // Remove blocks (index, size) from the covariance and shift the indices after them
  void RemoveFromCovariance(std::vector<std::pair<int, int>> blocks);

  // x with P(chi2_dof <= x) = probability
  static double ChiSquaredQuantile(int dof, double probability);

  EstimatorOptions options_;
  ImuNoise imu_noise_;
  double gravity_magnitude_;
  Camera initial_camera_;             // Configured camera and rig (start of the estimate)
  Rig initial_rig_;
  std::vector<double> chi2_table_;    // 95% chi-square quantiles by dof

  bool initialized_;
  double time_;                       // State time [s]
  Eigen::Vector4d q_;                 // q_GtoI
  Eigen::Vector3d p_;                 // p_IinG
  Eigen::Vector3d v_;                 // v_IinG
  Eigen::Vector3d bg_;                // Gyroscope bias
  Eigen::Vector3d ba_;                // Accelerometer bias
  Rig rig_;                           // Current rig estimate (extrinsics, time offsets)
  Camera camera_;                     // Current camera estimate (parameters)
  Camera camera1_;                    // Stereo: the second camera (fixed; model kInvalid: monocular)
  Eigen::Vector4d q_fej_;             // First estimates of the IMU pose and velocity
  Eigen::Vector3d p_fej_;
  Eigen::Vector3d v_fej_;
  Eigen::MatrixXd P_;                 // Covariance [imu (15), time offsets (2), calibration (extrinsics 6, camera
                                      // parameter 1, when estimated), then clones (6) and landmarks (3)]
  std::deque<Clone> clones_;          // Oldest first
  mutable std::deque<SensorMotion> sensor_motion_;  // Sensor-1 motion cache, one per clone
  std::map<int, Landmark> landmarks_; // By track id

  std::vector<ImuSample> imu_;        // IMU buffer
  std::map<int, Track> tracks_;       // Feature tracks by id
  EstimatorStatistics statistics_;    // Statistics of the last image
  std::vector<TrackReport> track_reports_;  // Per-track outcome of the last update
  std::optional<Keyframe> marginalized_;    // Clone that left the window in the last image
  std::string failure_;               // Reason when ProcessImage() returns false
};

}  // namespace vio

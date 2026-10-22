// SPDX-License-Identifier: MIT
/**
 * @file      initializer.h
 * @brief     Initial IMU state (orientation, velocity, biases) from a start at rest or a moving start.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2026-09-28
 * @copyright Copyright (c) 2026 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 *
 * The image disparity of the older and the newer half of a window chooses the method. At rest followed by a jerk,
 * the mean specific force of the still half gives gravity and the biases. Moving, a linear system in the first IMU
 * frame (point positions, velocity, gravity) with the gravity-magnitude constraint is solved from the rays and the
 * preintegrated IMU, then refined by maximum likelihood; its covariance is recovered and inflated. The rays work
 * anywhere on the sphere. Papers: doc/references.md.
 */

#pragma once

#include "vio/scene/camera.h"
#include "vio/sensor/rig.h"
#include "vio/util/types.h"

#include <Eigen/Dense>

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace vio {

struct InitializerOptions {
  double window_time = 2.0;               // Data window [s]
  double excitation_accel_std = 1.5;      // Accel sample std separating still / excited [m/s^2]
  double max_still_disparity = 10.0;      // Mean pixel disparity separating still / moving [px]
  bool rotation_compensated_disparity = false;  // The disparity of the still / moving test is measured after turning
                                                // the first ray by the gyro, so that only translation counts as
                                                // moving (a hand-held device rotates in place; the moving start
                                                // needs translation)
  int max_features = 50;                  // Tracker feature target while initializing (a moving start needs 75 % of
                                          // it as tracks in the window)

  // Moving start
  int refine_max_iterations = 50;         // Maximum-likelihood refinement iterations (0: no refinement)
  int num_poses = 6;                      // Poses spread over the window
  double min_rotation_deg = 10.0;         // Minimum summed gyro rotation over the window [deg]
  double covariance_scale_orientation = 10.0;  // Scale of the recovered state's covariance
  double covariance_scale_velocity = 100.0;
  double covariance_scale_gyro_bias = 10.0;
  double covariance_scale_accel_bias = 100.0;
  double min_covariance_rcond = 1e-12;                        // Covariance recovery threshold
  Eigen::Vector3d gyro_bias_prior = Eigen::Vector3d::Zero();   // Gyro bias linearization point [rad/s]
  Eigen::Vector3d accel_bias_prior = Eigen::Vector3d::Zero();  // Accel bias linearization point [m/s^2]
  double pixel_sigma = 1.0;               // Pixel noise [px]

  // Startup calibration
  double rest_accel_bias_sigma = 0.0;     // Sigma of the accel-bias component along gravity measured from a start at
                                          // rest to the initialization [m/s^2] (0: not used)

  bool Check() const;
};

/**
 * Result of one initialization attempt. State and covariance are at 'timestamp'. Covariance order is
 * [theta, p, v, bg, ba].
 */
struct InitializerResult {
  bool success = false;
  std::string method;  // "at rest", "moving", or empty when none was tried
  std::string reason;  // Why the attempt stopped (empty on success)

  double timestamp = -1.0;  // [s]
  Eigen::Vector4d q_GtoI = Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);  // JPL quaternion
  Eigen::Vector3d p_IinG = Eigen::Vector3d::Zero();
  Eigen::Vector3d v_IinG = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
  Eigen::Vector3d accel_bias = Eigen::Vector3d::Zero();
  Eigen::Matrix<double, 15, 15> covariance = Eigen::Matrix<double, 15, 15>::Zero();

  // Diagnostics
  double disparity_old = -1.0;   // Mean disparity of the older half [px]
  double disparity_new = -1.0;   // Mean disparity of the newer half [px]
  int disparity_tracks_old = 0;
  int disparity_tracks_new = 0;
  double accel_std_old = -1.0;   // At rest: accel std of the older half
  double accel_std_new = -1.0;   // At rest: accel std of the newer half
  int num_tracks = 0;            // Tracks in the window
  int num_valid_tracks = 0;      // Moving: tracks used by the linear system
  int num_poses = 0;             // Moving: pose times
  double rotation_deg = -1.0;    // Moving: summed gyro rotation [deg]
  double gravity_lambda = 0.0;   // Moving: Lagrange multiplier of |g| = G
  double gravity_norm_error = -1.0;                              // Moving: | |g| - G | of the linear solution
  Eigen::Vector3d linear_velocity_I0 = Eigen::Vector3d::Zero();  // Moving: v of the linear solution
  int num_refined_features = 0;
  int refine_iterations = 0;
  double refine_initial_cost = 0.0;
  double refine_final_cost = 0.0;
};

/**
 * @brief Visual-inertial initializer
 *
 * Usage: AddImu() for every IMU sample and AddImage() for every tracked image (camera clock), TryInitialize() after
 * each image until it succeeds.
 */
class Initializer {
 public:
  /**
   * @param options Initialization parameters
   * @param camera Camera model of the observations
   * @param rig Camera-IMU extrinsics
   * @param imu_noise IMU noise densities
   * @param gravity_magnitude [m/s^2]
   */
  Initializer(const InitializerOptions& options, const Camera& camera, const Rig& rig, const ImuNoise& imu_noise,
              double gravity_magnitude);

  void AddImu(const ImuSample& sample);
  void AddImage(Timestamp timestamp, const std::vector<PixelObservation>& observations);

  // Try to initialize with the data buffered so far
  InitializerResult TryInitialize();

 private:
  struct Track {
    std::vector<double> times;              // Observation times [s]
    std::vector<Eigen::Vector2d> pixels;    // Pixel coordinates
    std::vector<Eigen::Vector3d> bearings;  // Unit rays in the camera frame
  };

  bool StartAtRest(InitializerResult& result) const;
  bool StartMoving(InitializerResult& result);

  // Remove track measurements at or before t and drop empty tracks
  void DropMeasurementsUpTo(double t);

  // Newest observation time over all tracks (-1 when there is none)
  double NewestImageTime() const;

  // Mean pixel disparity of the tracks between their first observation after `after` and their last observation
  // before `before` (no bound when not given); mean -1 when no track has both
  void MeanTrackMotion(std::optional<double> before, std::optional<double> after, double* mean, int* count) const;

  InitializerOptions options_;
  Camera camera_;
  Eigen::Matrix3d R_ItoC_;
  Eigen::Vector3d p_IinC_;
  ImuNoise imu_noise_;
  double gravity_magnitude_;
  std::vector<ImuSample> imu_;    // IMU buffer
  std::map<int, Track> tracks_;   // Feature tracks by id
};

}  // namespace vio

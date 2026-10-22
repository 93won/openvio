// SPDX-License-Identifier: MIT
/**
 * @file      visual_inertial_odometry.cc
 * @brief     Implements the odometry engine.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2026-09-28
 * @copyright Copyright (c) 2026 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "vio/odometry/visual_inertial_odometry.h"

#include "vio/geometry/lie_group.h"
#include "vio/util/logging.h"

namespace vio {

bool VisualInertialOdometryOptions::Check() const {
  if (!camera.Check()) {
    return false;
  }
  REQUIRE_OPTION(camera.NumSensors() <= 2);
  if (IsStereo()) {
    // Stereo: two single-sensor cameras (the two-lens 360-degree camera is not paired)
    if (!camera1.Check()) {
      return false;
    }
    REQUIRE_OPTION(camera.NumSensors() == 1 && camera1.NumSensors() == 1);
    REQUIRE_OPTION(rig.p_C0inC1.norm() > 0.0);
  }
  REQUIRE_OPTION(estimator.camera_param_sigma == 0.0 || camera.NumOnlineParams() > 0);
  REQUIRE_OPTION(imu_noise.gyro > 0.0 && imu_noise.accel > 0.0);
  REQUIRE_OPTION(imu_noise.gyro_walk > 0.0 && imu_noise.accel_walk > 0.0);
  REQUIRE_OPTION(gravity_magnitude > 0.0);
  return feature_tracker.Check() && landmark_tracker.Check() && initializer.Check() && estimator.Check();
}

// The initializer and the startup calibration use the camera without the online corrections of its model (the
// estimator refines them after the initialization)
VisualInertialOdometry::VisualInertialOdometry(const VisualInertialOdometryOptions& options)
    : options_(options),
      feature_tracker_(options.feature_tracker, options.camera),
      landmark_tracker_(options.landmark_tracker, options.camera),
      initializer_(options.initializer, options.camera.Plain(), options.rig, options.imu_noise,
                   options.gravity_magnitude),
      startup_calibration_(std::make_unique<StartupCalibration>(options.camera.Plain(), options.rig)),
      estimator_(options.estimator, options.camera, options.rig, options.imu_noise, options.gravity_magnitude,
                 options.camera1) {
  // While initializing, the tracker targets the initializer's feature count
  feature_tracker_.SetMaxFeatures(options.initializer.max_features);
  if (options.IsStereo()) {
    feature_tracker_.EnableStereo(options.camera1, options.rig.R_C0toC1, options.rig.p_C0inC1);
  }
}

void VisualInertialOdometry::AddImu(const ImuSample& sample) {
  ImuSample camera_clock = sample;
  camera_clock.timestamp = ToCameraClock(sample.timestamp);
  if (!estimator_.IsInitialized()) {
    initializer_.AddImu(camera_clock);
    startup_calibration_->AddImu(camera_clock);
  }
  estimator_.AddImu(camera_clock);
}

bool VisualInertialOdometry::ProcessImage(Timestamp timestamp, const cv::Mat& image, int index,
                                          const cv::Mat& image1) {
  report_ = FrameReport();
  report_.index = index;
  report_.timestamp = timestamp;

  // Track features, then find lost landmarks at their predicted pixels
  auto frame = std::make_shared<Frame>(timestamp, index, image);
  feature_tracker_.TrackFeatures(frame.get(), previous_.get());
  cv::Mat image_f;
  const bool find_landmarks = options_.estimator.landmark_dormant_time > 0.0 && estimator_.IsInitialized();
  if (find_landmarks) {
    image.convertTo(image_f, CV_32F);
    Eigen::Matrix3d R_GtoC;
    const std::vector<PixelObservation> predictions = estimator_.PredictLandmarks(frame->timestamp(), &R_GtoC);
    landmark_tracker_.FindLandmarks(predictions, R_GtoC, image_f, feature_tracker_, frame.get());
  }
  std::vector<PixelObservation>& observations = report_.observations;
  observations.reserve(frame->features().size());
  for (const auto& feature : frame->features()) {
    observations.push_back({feature->id, Eigen::Vector2d(feature->pixel.x, feature->pixel.y)});
  }
  previous_ = frame;

  // Initialize; the initialization image itself is not used by the estimator
  if (!estimator_.IsInitialized()) {
    startup_calibration_->AddImage(timestamp, observations);
    initializer_.AddImage(timestamp, observations);
    init_result_ = initializer_.TryInitialize();
    report_.initialization = init_result_;
    if (init_result_.success) {
      estimator_.Initialize(init_result_.timestamp, init_result_.q_GtoI, init_result_.p_IinG, init_result_.v_IinG,
                            init_result_.gyro_bias, init_result_.accel_bias, init_result_.covariance);

      // Start the estimated time offsets at what the images seen so far say
      if (options_.estimator.time_offset_sigma > 0.0) {
        double offset0 = 0.0;
        double offset1 = 0.0;
        if (startup_calibration_->EstimateSensorTimeOffsets(&offset0, &offset1)) {
          if (options_.camera.NumSensors() > 1) {
            estimator_.SetTimeOffsets(offset0, offset1 - offset0);
            LOG_INFO("Startup sensor time offsets: sensor 0 {:.2f} ms, sensor 1 {:.2f} ms", 1e3 * offset0,
                     1e3 * offset1);
          } else {
            // One sensor: no offset of a second sensor
            estimator_.SetTimeOffsets(offset0, 0.0);
            LOG_INFO("Startup time offset: {:.2f} ms", 1e3 * offset0);
          }
        } else {
          LOG_INFO("Startup sensor time offsets: not observable yet, starting at zero");
        }
      }
      // Accel-bias component along gravity when the data start at rest
      if (options_.initializer.rest_accel_bias_sigma > 0.0) {
        Eigen::Vector3d direction;
        double bias = 0.0;
        double elapsed = 0.0;
        if (startup_calibration_->EstimateRestAccelBias(options_.gravity_magnitude, estimator_.velocity().z(),
                                                        &direction, &bias, &elapsed)) {
          estimator_.UpdateAccelBiasComponent(direction, bias, options_.initializer.rest_accel_bias_sigma);
          LOG_INFO("Accel bias along gravity from rest: {:.3f} m/s^2 over {:.1f} s", bias, elapsed);
        }
      }
      startup_calibration_.reset();

      feature_tracker_.SetMaxFeatures(options_.feature_tracker.max_features);
      report_.initialized = true;
    }
    return true;
  }
  report_.initialization = init_result_;

  // Stereo: the camera-1 pixels of the features join the camera-0 ones
  if (options_.IsStereo() && !image1.empty()) {
    std::vector<PixelObservation> stereo = observations;
    const std::vector<PixelObservation> camera1 = feature_tracker_.MatchStereo(*frame, image1);
    stereo.insert(stereo.end(), camera1.begin(), camera1.end());
    report_.has_state = estimator_.ProcessImage(timestamp, stereo);
  } else {
    report_.has_state = estimator_.ProcessImage(timestamp, observations);
  }
  if (report_.has_state && find_landmarks) {
    std::vector<int> landmark_ids;
    for (const auto& landmark : estimator_.Landmarks()) {
      landmark_ids.push_back(landmark.first);
    }
    std::vector<int> updated_ids;
    for (const auto& track : estimator_.track_reports()) {
      if (track.outcome == TrackOutcome::kLandmark) {
        updated_ids.push_back(track.id);
      }
    }
    const Eigen::Matrix3d R_CtoG =
        (estimator_.rig().R_ItoC * jpl::ToRotation(estimator_.orientation())).transpose();
    landmark_tracker_.UpdatePatches(*frame, image_f, landmark_ids, updated_ids, R_CtoG);
  }
  return report_.has_state;
}

}  // namespace vio

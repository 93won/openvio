// SPDX-License-Identifier: MIT
/**
 * @file      visual_inertial_odometry.h
 * @brief     Odometry engine: one image through tracking, initialization and estimation in a fixed order.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2026-09-28
 * @copyright Copyright (c) 2026 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 *
 * Per image: track features; find lost landmarks at their predicted pixels (once initialized); before the
 * initialization, feed the startup calibration and the initializer and try to initialize (the initialization image
 * itself is not used by the estimator); afterwards, the estimator processes the image and the landmark patches are
 * stored. The IMU must be added up to one sample past CaptureTime() of the image before ProcessImage().
 *
 * Stereo (camera1 configured): after the initialization, every feature of the camera-0 image is also matched in the
 * camera-1 image, and the estimator gets both pixels; tracking, landmark search and initialization use camera 0.
 */

#pragma once

#include "vio/estimators/estimator.h"
#include "vio/estimators/initializer.h"
#include "vio/estimators/startup_calibration.h"
#include "vio/feature/feature_tracker.h"
#include "vio/feature/landmark_tracker.h"
#include "vio/scene/camera.h"
#include "vio/scene/frame.h"
#include "vio/sensor/rig.h"
#include "vio/util/types.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace vio {

struct VisualInertialOdometryOptions {
  Camera camera;
  Camera camera1;  // Second camera of a stereo rig (model kInvalid: monocular)
  Rig rig;
  ImuNoise imu_noise;
  double gravity_magnitude = 9.81;  // [m/s^2]
  FeatureTrackerOptions feature_tracker;
  LandmarkTrackerOptions landmark_tracker;
  InitializerOptions initializer;
  EstimatorOptions estimator;

  bool IsStereo() const { return camera1.model_id != CameraModelId::kInvalid; }
  bool Check() const;
};

// What happened with the last image
struct FrameReport {
  int index = -1;                               // Image index
  Timestamp timestamp = 0.0;                    // Image time [s]
  bool initialized = false;                     // The estimator was initialized with this image
  bool has_state = false;                       // The estimator processed this image (its state is valid)
  std::vector<PixelObservation> observations;   // Tracked features of the image
  InitializerResult initialization;             // Last initialization attempt
};

class VisualInertialOdometry {
 public:
  explicit VisualInertialOdometry(const VisualInertialOdometryOptions& options);

  // Add one IMU sample (IMU clock)
  void AddImu(const ImuSample& sample);

  /**
   * @brief Process one image
   * @param timestamp Image time (camera clock) [s]
   * @param image Grayscale image at the camera resolution
   * @param index Image index
   * @param image1 Stereo: grayscale camera-1 image taken with image (empty: none)
   * @return false if the estimator stopped (see failure())
   */
  bool ProcessImage(Timestamp timestamp, const cv::Mat& image, int index, const cv::Mat& image1 = cv::Mat());

  // IMU time on the camera clock
  Timestamp ToCameraClock(Timestamp imu_time) const { return imu_time - options_.rig.clock_offset; }

  // Latest exposure time of an image (over its sensors) with the estimated time offsets; an image can be processed
  // once the IMU (camera clock) has passed it
  Timestamp CaptureTime(Timestamp image_time) const {
    return image_time + estimator_.rig().time_offset + std::max(0.0, estimator_.rig().sensor_time_offset);
  }

  const FrameReport& report() const { return report_; }
  const Estimator& estimator() const { return estimator_; }
  const LandmarkTracker& landmark_tracker() const { return landmark_tracker_; }
  const std::string& failure() const { return estimator_.failure(); }

 private:
  VisualInertialOdometryOptions options_;
  FeatureTracker feature_tracker_;
  LandmarkTracker landmark_tracker_;
  Initializer initializer_;
  std::unique_ptr<StartupCalibration> startup_calibration_;  // Until the initialization
  Estimator estimator_;
  std::shared_ptr<Frame> previous_;                          // Previous frame (tracking)
  InitializerResult init_result_;                            // Last initialization attempt
  FrameReport report_;
};

}  // namespace vio

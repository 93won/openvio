// SPDX-License-Identifier: MIT
/**
 * @file      odometry_pipeline.h
 * @brief     Runs the odometry over a dataset: reading, IMU and image order, trajectory output and callbacks.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2026-10-05
 * @copyright Copyright (c) 2026 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 *
 * Dataset folder: images/ (or panocam/) with PNG images sorted by name, cam_timestamps.txt (one time per image) and
 * imu_data.csv (header line, then timestamp, ax, ay, az, gx, gy, gz). Like a live system, an image is processed once
 * the IMU has passed its capture time. Output: one TUM line per processed image (time x y z qx qy qz qw of the IMU
 * pose at the sensor-0 exposure).
 */

#pragma once

#include "vio/odometry/visual_inertial_odometry.h"
#include "vio/scene/keyframe.h"

#include <opencv2/core.hpp>

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace vio {

class OdometryPipeline {
 public:
  enum CallbackType {
    kInitializedCallback,  // The estimator was initialized with the current image
    kFrameCallback,        // An image was processed (odometry().report(), image())
    kKeyframeCallback,     // The oldest clone left the window with its final estimate (keyframe())
  };

  /**
   * @param options Odometry options
   * @param dataset_path Dataset folder
   * @param output_path Trajectory file (TUM format); empty: none
   */
  OdometryPipeline(const VisualInertialOdometryOptions& options, const std::string& dataset_path,
                   const std::string& output_path);

  void AddCallback(CallbackType type, std::function<void()> callback);

  /**
   * @brief Process the whole dataset (or until Stop())
   * @return false if the dataset cannot be read, the estimator stopped or it never initialized
   */
  bool Run();

  // Ask Run() to stop after the current image (e.g. from a callback)
  void Stop() { stop_ = true; }

  // Valid inside the callbacks
  const VisualInertialOdometry& odometry() const { return *odometry_; }
  const cv::Mat& image() const { return image_; }
  const Keyframe& keyframe() const { return *odometry_->estimator().marginalized_keyframe(); }

 private:
  void Callback(CallbackType type) const;

  VisualInertialOdometryOptions options_;
  std::string dataset_path_;
  std::string output_path_;
  std::map<CallbackType, std::vector<std::function<void()>>> callbacks_;
  std::unique_ptr<VisualInertialOdometry> odometry_;
  cv::Mat image_;
  bool stop_ = false;
};

}  // namespace vio

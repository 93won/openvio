// SPDX-License-Identifier: MIT
/**
 * @file      online_odometry.h
 * @brief     Runs the odometry on streamed IMU samples and images in the order of the offline tool (ROS 1 and 2).
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2026-10-23
 * @copyright Copyright (c) 2026 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 *
 * The ROS nodes hand every message to this class from their callbacks, which only queue it; one worker thread feeds
 * the odometry in the order of `vio odometry`: before each camera-0 image, IMU samples are added until one has passed
 * the image's exposure, and a camera-1 image pairs with the camera-0 image of the same time stamp. Time stamps are read
 * as "sec.nanosec" text and IMU values in single precision, as the offline reader reads its files, so a bag of a
 * prepared dataset gives the same trajectory. After every image with a state the callback receives the pose; Stop()
 * processes the queued images with the queued IMU.
 */

#pragma once

#include "vio/odometry/visual_inertial_odometry.h"

#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace vio_ros {

// Time stamp of a message
struct Stamp {
  int64_t sec = 0;
  uint32_t nanosec = 0;

  double Seconds() const;  // Parsed from its "sec.nanosec" text, as the offline reader parses its time files
  int64_t Nanoseconds() const { return sec * 1000000000LL + nanosec; }
};

// 8-bit gray image of a sensor_msgs/Image, decoded as the offline reader decodes a PNG: 16-bit samples keep their
// high byte, color becomes gray; empty for other encodings (supported: mono8, mono16, bgr8, rgb8)
cv::Mat GrayImage(const std::string& encoding, uint32_t width, uint32_t height, uint32_t step, bool big_endian,
                  const uint8_t* data);

// Estimate after an image
struct Pose {
  Eigen::Quaterniond q_ItoG;  // IMU orientation in the world
  Eigen::Vector3d p_IinG;     // IMU position in the world [m]
  Eigen::Vector3d v_inI;      // Velocity in the IMU frame [m/s]
};

class OnlineOdometry {
 public:
  // Called by the worker thread after every image with a state: its stamp, the gray camera-0 image and the estimate
  using StateCallback = std::function<void(const Stamp& stamp, const cv::Mat& image,
                                           const vio::VisualInertialOdometry& odometry, const Pose& pose)>;

  /**
   * @param options Odometry options (config file)
   * @param output_path TUM trajectory written as `vio odometry` writes it (empty: none)
   * @param callback Receives every estimate
   */
  OnlineOdometry(const vio::VisualInertialOdometryOptions& options, const std::string& output_path,
                 StateCallback callback);
  ~OnlineOdometry();

  bool stereo() const { return options_.IsStereo(); }

  void AddImu(const Stamp& stamp, const Eigen::Vector3d& gyro, const Eigen::Vector3d& accel);
  void AddImage(const Stamp& stamp, const cv::Mat& gray);   // Camera 0
  void AddImage1(const Stamp& stamp, const cv::Mat& gray);  // Camera 1 of a stereo rig

  // Process the queued images with the queued IMU and stop the worker (also done by the destructor)
  void Stop();

 private:
  struct ImuEntry {
    Stamp stamp;
    vio::ImuSample sample;
  };
  struct ImageEntry {
    Stamp stamp;
    cv::Mat gray;
  };

  bool ReadyLocked() const;
  void Run();
  void Process(const ImageEntry& entry, cv::Mat image1);

  vio::VisualInertialOdometryOptions options_;
  std::unique_ptr<vio::VisualInertialOdometry> odometry_;  // Used by the worker only
  StateCallback callback_;
  std::ofstream out_;
  int index_ = 0;        // Images given to the odometry
  bool failed_ = false;  // The odometry stopped

  std::mutex mutex_;
  std::condition_variable wake_;
  std::deque<ImuEntry> imu_;
  std::deque<ImageEntry> images_;
  std::map<int64_t, cv::Mat> images1_;  // Camera-1 images by time stamp [ns]
  int64_t latest_image1_ = INT64_MIN;
  bool stop_ = false;
  std::thread worker_;
};

}  // namespace vio_ros

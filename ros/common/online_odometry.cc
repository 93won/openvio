// SPDX-License-Identifier: MIT
/**
 * @file      online_odometry.cc
 * @brief     Implements the streamed odometry of the ROS nodes.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2026-10-23
 * @copyright Copyright (c) 2026 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "online_odometry.h"

#include "vio/geometry/lie_group.h"
#include "vio/util/logging.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <stdexcept>

namespace vio_ros {

double Stamp::Seconds() const {
  char text[40];
  std::snprintf(text, sizeof(text), "%lld.%09u", static_cast<long long>(sec), nanosec);
  return std::strtod(text, nullptr);
}

cv::Mat GrayImage(const std::string& encoding, uint32_t width, uint32_t height, uint32_t step, bool big_endian,
                  const uint8_t* data) {
  const int rows = static_cast<int>(height);
  const int cols = static_cast<int>(width);
  auto* bytes = const_cast<uint8_t*>(data);
  if (encoding == "mono8" || encoding == "8UC1") {
    return cv::Mat(rows, cols, CV_8UC1, bytes, step).clone();
  }
  if (encoding == "mono16" || encoding == "16UC1") {
    cv::Mat gray(rows, cols, CV_8UC1);
    const int high = big_endian ? 0 : 1;
    for (int y = 0; y < rows; ++y) {
      const uint8_t* row = data + static_cast<size_t>(y) * step;
      uint8_t* out = gray.ptr<uint8_t>(y);
      for (int x = 0; x < cols; ++x) {
        out[x] = row[2 * x + high];
      }
    }
    return gray;
  }
  if (encoding == "bgr8" || encoding == "rgb8") {
    cv::Mat gray;
    cv::cvtColor(cv::Mat(rows, cols, CV_8UC3, bytes, step), gray,
                 encoding == "bgr8" ? cv::COLOR_BGR2GRAY : cv::COLOR_RGB2GRAY);
    return gray;
  }
  return cv::Mat();
}

OnlineOdometry::OnlineOdometry(const vio::VisualInertialOdometryOptions& options, const std::string& output_path,
                               StateCallback callback)
    : options_(options),
      odometry_(std::make_unique<vio::VisualInertialOdometry>(options)),
      callback_(std::move(callback)) {
  if (!output_path.empty()) {
    out_.open(output_path);
    if (!out_) {
      throw std::runtime_error("cannot open the trajectory output '" + output_path + "'");
    }
    out_ << std::fixed << std::setprecision(9) << std::unitbuf;
  }
  worker_ = std::thread(&OnlineOdometry::Run, this);
}

OnlineOdometry::~OnlineOdometry() { Stop(); }

void OnlineOdometry::Stop() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
  }
  wake_.notify_one();
  if (worker_.joinable()) {
    worker_.join();
  }
}

void OnlineOdometry::AddImu(const Stamp& stamp, const Eigen::Vector3d& gyro, const Eigen::Vector3d& accel) {
  ImuEntry entry;
  entry.stamp = stamp;
  entry.sample.timestamp = stamp.Seconds();
  // Single precision, as the offline reader reads the IMU file
  entry.sample.gyro = gyro.cast<float>().cast<double>();
  entry.sample.accel = accel.cast<float>().cast<double>();
  std::lock_guard<std::mutex> lock(mutex_);
  imu_.push_back(entry);
  wake_.notify_one();
}

void OnlineOdometry::AddImage(const Stamp& stamp, const cv::Mat& gray) {
  std::lock_guard<std::mutex> lock(mutex_);
  images_.push_back({stamp, gray});
  wake_.notify_one();
}

void OnlineOdometry::AddImage1(const Stamp& stamp, const cv::Mat& gray) {
  std::lock_guard<std::mutex> lock(mutex_);
  images1_[stamp.Nanoseconds()] = gray;
  latest_image1_ = std::max(latest_image1_, stamp.Nanoseconds());
  wake_.notify_one();
}

bool OnlineOdometry::ReadyLocked() const {
  // The front image can be processed when the IMU has passed its exposure and, for a stereo rig, its camera-1 image
  // (if any) has arrived; when stopping, every queued image can
  if (images_.empty()) {
    return false;
  }
  if (stop_) {
    return true;
  }
  const Stamp& stamp = images_.front().stamp;
  if (options_.IsStereo() && latest_image1_ < stamp.Nanoseconds()) {
    return false;
  }
  return !imu_.empty() &&
         odometry_->ToCameraClock(imu_.back().sample.timestamp) > odometry_->CaptureTime(stamp.Seconds());
}

void OnlineOdometry::Run() {
  std::unique_lock<std::mutex> lock(mutex_);
  while (true) {
    wake_.wait(lock, [this] { return stop_ || ReadyLocked(); });
    if (!ReadyLocked()) {
      return;  // Stopping with no image left
    }
    const ImageEntry entry = images_.front();
    images_.pop_front();
    const double image_time = entry.stamp.Seconds();

    // IMU up to the first sample past the exposure, as the offline tool adds it
    while (!imu_.empty()) {
      const vio::ImuSample sample = imu_.front().sample;
      imu_.pop_front();
      odometry_->AddImu(sample);
      if (odometry_->ToCameraClock(sample.timestamp) > odometry_->CaptureTime(image_time)) {
        break;
      }
    }

    // Camera-1 image of the same time stamp; older ones are never used
    cv::Mat image1;
    if (options_.IsStereo()) {
      const int64_t key = entry.stamp.Nanoseconds();
      const auto it = images1_.find(key);
      if (it != images1_.end()) {
        image1 = it->second;
      }
      images1_.erase(images1_.begin(), images1_.upper_bound(key));
    }

    lock.unlock();
    Process(entry, image1);
    lock.lock();
  }
}

void OnlineOdometry::Process(const ImageEntry& entry, cv::Mat image1) {
  if (failed_) {
    return;
  }
  cv::Mat image = entry.gray;
  if (image.cols != options_.camera.width || image.rows != options_.camera.height) {
    cv::resize(image, image, cv::Size(options_.camera.width, options_.camera.height));
  }
  if (!image1.empty() && (image1.cols != options_.camera1.width || image1.rows != options_.camera1.height)) {
    cv::resize(image1, image1, cv::Size(options_.camera1.width, options_.camera1.height));
  }
  if (!odometry_->ProcessImage(entry.stamp.Seconds(), image, index_++, image1)) {
    LOG_ERROR("Odometry stopped at image {}: {}", index_ - 1, odometry_->failure());
    failed_ = true;
    return;
  }
  const vio::FrameReport& report = odometry_->report();
  if (report.initialized) {
    LOG_INFO("Initialized ({}) at image {}", report.initialization.method, index_ - 1);
  }
  if (!report.has_state) {
    return;
  }
  const vio::Estimator& estimator = odometry_->estimator();
  const Eigen::Matrix3d R_GtoI = vio::jpl::ToRotation(estimator.orientation());
  Pose pose;
  pose.q_ItoG = Eigen::Quaterniond(R_GtoI.transpose());
  pose.p_IinG = estimator.position();
  pose.v_inI = R_GtoI * estimator.velocity();
  if (out_.is_open()) {
    const Eigen::Quaterniond& q = pose.q_ItoG;
    const Eigen::Vector3d& p = pose.p_IinG;
    out_ << estimator.timestamp() << ' ' << p.x() << ' ' << p.y() << ' ' << p.z() << ' ' << q.x() << ' ' << q.y()
         << ' ' << q.z() << ' ' << q.w() << '\n';
  }
  if (callback_) {
    callback_(entry.stamp, image, *odometry_, pose);
  }
}

}  // namespace vio_ros

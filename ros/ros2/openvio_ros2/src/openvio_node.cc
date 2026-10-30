// SPDX-License-Identifier: MIT
/**
 * @file      openvio_node.cc
 * @brief     ROS 2 node: runs OpenVIO on IMU and image topics and publishes its estimate.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2026-10-23
 * @copyright Copyright (c) 2026 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 *
 * Subscribes to "imu" (sensor_msgs/Imu), "image0" and, with a stereo config, "image1" (sensor_msgs/Image); remap them
 * to the topics of the sensor. The processing order is that of the offline tool (ros/common/online_odometry.h).
 * Publishes odometry (nav_msgs/Odometry), path (nav_msgs/Path), landmarks (sensor_msgs/PointCloud2), tracked_image
 * (sensor_msgs/Image) and the transform world_frame -> body_frame; a message is built only when someone subscribes.
 * Parameters: config_path (YAML of config/), output_path (TUM trajectory, optional), world_frame, body_frame.
 */

#include "online_odometry.h"

#include "vio/pipeline/config_loader.h"
#include "vio/util/logging.h"

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace {

using sensor_msgs::msg::Image;
using sensor_msgs::msg::Imu;

vio_ros::Stamp ToStamp(const builtin_interfaces::msg::Time& time) { return {time.sec, time.nanosec}; }

builtin_interfaces::msg::Time ToTime(const vio_ros::Stamp& stamp) {
  builtin_interfaces::msg::Time time;
  time.sec = static_cast<int32_t>(stamp.sec);
  time.nanosec = stamp.nanosec;
  return time;
}

cv::Mat Gray(const Image& msg) {
  return vio_ros::GrayImage(msg.encoding, msg.width, msg.height, msg.step, msg.is_bigendian != 0, msg.data.data());
}

class OpenVioNode : public rclcpp::Node {
 public:
  OpenVioNode() : Node("openvio") {
    const std::string config_path = declare_parameter<std::string>("config_path", "");
    const std::string output_path = declare_parameter<std::string>("output_path", "");
    world_frame_ = declare_parameter<std::string>("world_frame", "world");
    body_frame_ = declare_parameter<std::string>("body_frame", "imu");
    vio::ConfigLoader config;
    if (config_path.empty() || !config.Read(config_path) || !config.Check()) {
      throw std::runtime_error("cannot read the config '" + config_path + "' (parameter config_path)");
    }

    odometry_pub_ = create_publisher<nav_msgs::msg::Odometry>("odometry", 10);
    path_pub_ = create_publisher<nav_msgs::msg::Path>("path", 10);
    landmarks_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("landmarks", 10);
    tracked_pub_ = create_publisher<Image>("tracked_image", 10);
    tf_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    path_.header.frame_id = world_frame_;
    odometry_ = std::make_unique<vio_ros::OnlineOdometry>(
        config.odometry, output_path,
        [this](const vio_ros::Stamp& stamp, const cv::Mat& image, const vio::VisualInertialOdometry& odometry,
               const vio_ros::Pose& pose) { Publish(stamp, image, odometry, pose); });

    // Deep queues: a bag played at full rate must not lose a message while the worker is busy
    const auto qos = rclcpp::QoS(rclcpp::KeepLast(2000)).reliable();
    imu_sub_ = create_subscription<Imu>("imu", qos, [this](Imu::ConstSharedPtr msg) {
      odometry_->AddImu(ToStamp(msg->header.stamp),
                        Eigen::Vector3d(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z),
                        Eigen::Vector3d(msg->linear_acceleration.x, msg->linear_acceleration.y,
                                        msg->linear_acceleration.z));
    });
    image0_sub_ = create_subscription<Image>("image0", qos, [this](Image::ConstSharedPtr msg) {
      const cv::Mat gray = Gray(*msg);
      if (gray.empty()) {
        LOG_ERROR("Image encoding {} is not supported (mono8, mono16, bgr8, rgb8)", msg->encoding);
        return;
      }
      odometry_->AddImage(ToStamp(msg->header.stamp), gray);
    });
    if (odometry_->stereo()) {
      image1_sub_ = create_subscription<Image>("image1", qos, [this](Image::ConstSharedPtr msg) {
        const cv::Mat gray = Gray(*msg);
        if (!gray.empty()) {
          odometry_->AddImage1(ToStamp(msg->header.stamp), gray);
        }
      });
    }
    LOG_INFO("OpenVIO node: config {}, {}", config_path, odometry_->stereo() ? "stereo" : "monocular");
  }

  // The queued images are processed before the publishers go away
  ~OpenVioNode() override { odometry_->Stop(); }

 private:
  void Publish(const vio_ros::Stamp& stamp, const cv::Mat& image, const vio::VisualInertialOdometry& odometry,
               const vio_ros::Pose& pose) {
    if (!rclcpp::ok()) {
      return;  // Shutting down: the trajectory file is still written
    }
    geometry_msgs::msg::TransformStamped transform;
    transform.header.stamp = ToTime(stamp);
    transform.header.frame_id = world_frame_;
    transform.child_frame_id = body_frame_;
    transform.transform.translation.x = pose.p_IinG.x();
    transform.transform.translation.y = pose.p_IinG.y();
    transform.transform.translation.z = pose.p_IinG.z();
    transform.transform.rotation.x = pose.q_ItoG.x();
    transform.transform.rotation.y = pose.q_ItoG.y();
    transform.transform.rotation.z = pose.q_ItoG.z();
    transform.transform.rotation.w = pose.q_ItoG.w();
    tf_->sendTransform(transform);

    geometry_msgs::msg::PoseStamped stamped;
    stamped.header = transform.header;
    stamped.pose.position.x = pose.p_IinG.x();
    stamped.pose.position.y = pose.p_IinG.y();
    stamped.pose.position.z = pose.p_IinG.z();
    stamped.pose.orientation = transform.transform.rotation;
    path_.poses.push_back(stamped);

    if (odometry_pub_->get_subscription_count() > 0) {
      nav_msgs::msg::Odometry msg;
      msg.header = transform.header;
      msg.child_frame_id = body_frame_;
      msg.pose.pose = stamped.pose;
      msg.twist.twist.linear.x = pose.v_inI.x();
      msg.twist.twist.linear.y = pose.v_inI.y();
      msg.twist.twist.linear.z = pose.v_inI.z();
      odometry_pub_->publish(msg);
    }
    if (path_pub_->get_subscription_count() > 0) {
      path_.header.stamp = transform.header.stamp;
      path_pub_->publish(path_);
    }
    if (landmarks_pub_->get_subscription_count() > 0) {
      const auto landmarks = odometry.estimator().Landmarks();
      sensor_msgs::msg::PointCloud2 cloud;
      cloud.header = transform.header;
      sensor_msgs::PointCloud2Modifier modifier(cloud);
      modifier.setPointCloud2FieldsByString(1, "xyz");
      modifier.resize(landmarks.size());
      sensor_msgs::PointCloud2Iterator<float> x(cloud, "x");
      sensor_msgs::PointCloud2Iterator<float> y(cloud, "y");
      sensor_msgs::PointCloud2Iterator<float> z(cloud, "z");
      for (const auto& landmark : landmarks) {
        *x = static_cast<float>(landmark.second.x());
        *y = static_cast<float>(landmark.second.y());
        *z = static_cast<float>(landmark.second.z());
        ++x;
        ++y;
        ++z;
      }
      landmarks_pub_->publish(cloud);
    }
    if (tracked_pub_->get_subscription_count() > 0) {
      cv::Mat color;
      cv::cvtColor(image, color, cv::COLOR_GRAY2BGR);
      for (const auto& observation : odometry.report().observations) {
        cv::circle(color, cv::Point2d(observation.pixel.x(), observation.pixel.y()), 3, cv::Scalar(0, 220, 0), -1);
      }
      Image msg;
      msg.header = transform.header;
      msg.height = static_cast<uint32_t>(color.rows);
      msg.width = static_cast<uint32_t>(color.cols);
      msg.encoding = "bgr8";
      msg.step = static_cast<uint32_t>(color.step);
      msg.data.assign(color.datastart, color.dataend);
      tracked_pub_->publish(msg);
    }
  }

  std::string world_frame_;
  std::string body_frame_;
  nav_msgs::msg::Path path_;  // Used by the worker thread only
  std::unique_ptr<vio_ros::OnlineOdometry> odometry_;

  rclcpp::Subscription<Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<Image>::SharedPtr image0_sub_;
  rclcpp::Subscription<Image>::SharedPtr image1_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr landmarks_pub_;
  rclcpp::Publisher<Image>::SharedPtr tracked_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_;
};

}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  vio::InitLogging();
  int status = 0;
  try {
    auto node = std::make_shared<OpenVioNode>();
    rclcpp::spin(node);
  } catch (const std::exception& error) {
    LOG_ERROR("{}", error.what());
    status = 1;
  }
  rclcpp::shutdown();
  return status;
}

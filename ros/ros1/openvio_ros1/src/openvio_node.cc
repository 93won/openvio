// SPDX-License-Identifier: MIT
/**
 * @file      openvio_node.cc
 * @brief     ROS 1 node: runs OpenVIO on IMU and image topics and publishes its estimate.
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
 * Private parameters: ~config_path (YAML of config/), ~output_path (TUM trajectory, optional), ~world_frame,
 * ~body_frame.
 */

#include "online_odometry.h"

#include "vio/pipeline/config_loader.h"
#include "vio/util/logging.h"

#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <opencv2/imgproc.hpp>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <tf2_ros/transform_broadcaster.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace {

vio_ros::Stamp ToStamp(const ros::Time& time) { return {static_cast<int64_t>(time.sec), time.nsec}; }

ros::Time ToTime(const vio_ros::Stamp& stamp) {
  return ros::Time(static_cast<uint32_t>(stamp.sec), stamp.nanosec);
}

cv::Mat Gray(const sensor_msgs::Image& msg) {
  return vio_ros::GrayImage(msg.encoding, msg.width, msg.height, msg.step, msg.is_bigendian != 0, msg.data.data());
}

class OpenVioNode {
 public:
  OpenVioNode(ros::NodeHandle& nh, ros::NodeHandle& pnh) {
    const std::string config_path = pnh.param<std::string>("config_path", "");
    const std::string output_path = pnh.param<std::string>("output_path", "");
    world_frame_ = pnh.param<std::string>("world_frame", "world");
    body_frame_ = pnh.param<std::string>("body_frame", "imu");
    vio::ConfigLoader config;
    if (config_path.empty() || !config.Read(config_path) || !config.Check()) {
      throw std::runtime_error("cannot read the config '" + config_path + "' (parameter ~config_path)");
    }

    odometry_pub_ = nh.advertise<nav_msgs::Odometry>("odometry", 10);
    path_pub_ = nh.advertise<nav_msgs::Path>("path", 10);
    landmarks_pub_ = nh.advertise<sensor_msgs::PointCloud2>("landmarks", 10);
    tracked_pub_ = nh.advertise<sensor_msgs::Image>("tracked_image", 10);
    path_.header.frame_id = world_frame_;
    odometry_ = std::make_unique<vio_ros::OnlineOdometry>(
        config.odometry, output_path,
        [this](const vio_ros::Stamp& stamp, const cv::Mat& image, const vio::VisualInertialOdometry& odometry,
               const vio_ros::Pose& pose) { Publish(stamp, image, odometry, pose); });

    // Deep queues: a bag played at full rate must not lose a message while the worker is busy
    imu_sub_ = nh.subscribe<sensor_msgs::Imu>("imu", 2000, [this](const sensor_msgs::Imu::ConstPtr& msg) {
      odometry_->AddImu(ToStamp(msg->header.stamp),
                        Eigen::Vector3d(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z),
                        Eigen::Vector3d(msg->linear_acceleration.x, msg->linear_acceleration.y,
                                        msg->linear_acceleration.z));
    });
    image0_sub_ = nh.subscribe<sensor_msgs::Image>("image0", 2000, [this](const sensor_msgs::Image::ConstPtr& msg) {
      const cv::Mat gray = Gray(*msg);
      if (gray.empty()) {
        LOG_ERROR("Image encoding {} is not supported (mono8, mono16, bgr8, rgb8)", msg->encoding);
        return;
      }
      odometry_->AddImage(ToStamp(msg->header.stamp), gray);
    });
    if (odometry_->stereo()) {
      image1_sub_ = nh.subscribe<sensor_msgs::Image>("image1", 2000, [this](const sensor_msgs::Image::ConstPtr& msg) {
        const cv::Mat gray = Gray(*msg);
        if (!gray.empty()) {
          odometry_->AddImage1(ToStamp(msg->header.stamp), gray);
        }
      });
    }
    LOG_INFO("OpenVIO node: config {}, {}", config_path, odometry_->stereo() ? "stereo" : "monocular");
  }

  // The queued images are processed before the publishers go away
  ~OpenVioNode() { odometry_->Stop(); }

 private:
  void Publish(const vio_ros::Stamp& stamp, const cv::Mat& image, const vio::VisualInertialOdometry& odometry,
               const vio_ros::Pose& pose) {
    if (!ros::ok()) {
      return;  // Shutting down: the trajectory file is still written
    }
    geometry_msgs::TransformStamped transform;
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
    tf_.sendTransform(transform);

    geometry_msgs::PoseStamped stamped;
    stamped.header = transform.header;
    stamped.pose.position.x = pose.p_IinG.x();
    stamped.pose.position.y = pose.p_IinG.y();
    stamped.pose.position.z = pose.p_IinG.z();
    stamped.pose.orientation = transform.transform.rotation;
    path_.poses.push_back(stamped);

    if (odometry_pub_.getNumSubscribers() > 0) {
      nav_msgs::Odometry msg;
      msg.header = transform.header;
      msg.child_frame_id = body_frame_;
      msg.pose.pose = stamped.pose;
      msg.twist.twist.linear.x = pose.v_inI.x();
      msg.twist.twist.linear.y = pose.v_inI.y();
      msg.twist.twist.linear.z = pose.v_inI.z();
      odometry_pub_.publish(msg);
    }
    if (path_pub_.getNumSubscribers() > 0) {
      path_.header.stamp = transform.header.stamp;
      path_pub_.publish(path_);
    }
    if (landmarks_pub_.getNumSubscribers() > 0) {
      const auto landmarks = odometry.estimator().Landmarks();
      sensor_msgs::PointCloud2 cloud;
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
      landmarks_pub_.publish(cloud);
    }
    if (tracked_pub_.getNumSubscribers() > 0) {
      cv::Mat color;
      cv::cvtColor(image, color, cv::COLOR_GRAY2BGR);
      for (const auto& observation : odometry.report().observations) {
        cv::circle(color, cv::Point2d(observation.pixel.x(), observation.pixel.y()), 3, cv::Scalar(0, 220, 0), -1);
      }
      sensor_msgs::Image msg;
      msg.header = transform.header;
      msg.height = static_cast<uint32_t>(color.rows);
      msg.width = static_cast<uint32_t>(color.cols);
      msg.encoding = "bgr8";
      msg.step = static_cast<uint32_t>(color.step);
      msg.data.assign(color.datastart, color.dataend);
      tracked_pub_.publish(msg);
    }
  }

  std::string world_frame_;
  std::string body_frame_;
  nav_msgs::Path path_;  // Used by the worker thread only
  tf2_ros::TransformBroadcaster tf_;
  ros::Publisher odometry_pub_;
  ros::Publisher path_pub_;
  ros::Publisher landmarks_pub_;
  ros::Publisher tracked_pub_;
  std::unique_ptr<vio_ros::OnlineOdometry> odometry_;
  ros::Subscriber imu_sub_;
  ros::Subscriber image0_sub_;
  ros::Subscriber image1_sub_;
};

}  // namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "openvio");
  vio::InitLogging();
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  try {
    OpenVioNode node(nh, pnh);
    ros::spin();
  } catch (const std::exception& error) {
    LOG_ERROR("{}", error.what());
    return 1;
  }
  return 0;
}

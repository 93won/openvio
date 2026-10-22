// SPDX-License-Identifier: MIT
/**
 * @file      config_loader.cc
 * @brief     Implements the config loader.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "vio/pipeline/config_loader.h"

#include "vio/util/logging.h"

#include <opencv2/core/persistence.hpp>

#include <map>
#include <memory>
#include <set>
#include <sstream>

namespace vio {
namespace {

// Looks up dotted paths ("section.key", "section.sub.key") in the file, overrides first
class OptionReader {
 public:
  OptionReader(const cv::FileStorage& file, const std::map<std::string, std::string>& overrides)
      : file_(file), overrides_(overrides) {}

  void Read(const std::string& path, int* value) {
    const cv::FileNode node = Node(path);
    if (!node.empty()) {
      *value = static_cast<int>(node);
    }
  }
  void Read(const std::string& path, double* value) {
    const cv::FileNode node = Node(path);
    if (!node.empty()) {
      *value = static_cast<double>(node);
    }
  }
  void Read(const std::string& path, float* value) {
    const cv::FileNode node = Node(path);
    if (!node.empty()) {
      *value = static_cast<float>(static_cast<double>(node));
    }
  }
  void Read(const std::string& path, bool* value) {
    const cv::FileNode node = Node(path);
    if (!node.empty()) {
      *value = static_cast<int>(node) != 0;
    }
  }
  void Read(const std::string& path, std::string* value) {
    const cv::FileNode node = Node(path);
    if (!node.empty()) {
      *value = static_cast<std::string>(node);
    }
  }
  void Read(const std::string& path, std::vector<double>* value) {
    const cv::FileNode node = Node(path);
    if (!node.empty()) {
      value->clear();
      for (const cv::FileNode& element : node) {
        value->push_back(static_cast<double>(element));
      }
    }
  }
  void Read(const std::string& path, Eigen::Vector3d* value) {
    const cv::FileNode node = Node(path);
    if (node.isSeq() && node.size() == 3) {
      *value = Eigen::Vector3d(static_cast<double>(node[0]), static_cast<double>(node[1]),
                               static_cast<double>(node[2]));
    }
  }
  void Read(const std::string& path, std::vector<cv::Vec4f>* value) {
    const cv::FileNode node = Node(path);
    if (!node.empty()) {
      value->clear();
      for (const cv::FileNode& rectangle : node) {
        if (rectangle.size() == 4) {
          value->emplace_back(static_cast<float>(rectangle[0]), static_cast<float>(rectangle[1]),
                              static_cast<float>(rectangle[2]), static_cast<float>(rectangle[3]));
        }
      }
    }
  }
  void Read(const std::string& path, Eigen::Matrix4f* value) {
    const cv::FileNode node = Node(path);
    if (node.isSeq() && node.size() == 4) {
      for (int i = 0; i < 4; ++i) {
        const cv::FileNode row = node[i];
        if (row.isSeq() && row.size() == 4) {
          for (int j = 0; j < 4; ++j) {
            (*value)(i, j) = static_cast<float>(static_cast<double>(row[j]));
          }
        }
      }
    }
  }
  // Read a value that was single precision in the reference runs (D9 quirk: rounded through float)
  void ReadThroughFloat(const std::string& path, double* value) {
    float single = static_cast<float>(*value);
    Read(path, &single);
    *value = single;
  }

  // Overrides that named no option
  std::vector<std::string> UnusedOverrides() const {
    std::vector<std::string> unused;
    for (const auto& entry : overrides_) {
      if (used_.count(entry.first) == 0) {
        unused.push_back(entry.first);
      }
    }
    return unused;
  }

 private:
  cv::FileNode Node(const std::string& path) {
    const auto override = overrides_.find(path);
    if (override != overrides_.end()) {
      used_.insert(path);
      auto storage = std::make_unique<cv::FileStorage>("%YAML:1.0\n---\nvalue: " + override->second + "\n",
                                                       cv::FileStorage::READ | cv::FileStorage::MEMORY);
      const cv::FileNode node = (*storage)["value"];
      parsed_.push_back(std::move(storage));
      return node;
    }
    std::stringstream parts(path);
    std::string part;
    std::getline(parts, part, '.');
    cv::FileNode node = file_[part];
    while (std::getline(parts, part, '.')) {
      if (!node.isMap()) {
        return cv::FileNode();
      }
      node = node[part];
    }
    return node;
  }

  const cv::FileStorage& file_;
  std::map<std::string, std::string> overrides_;
  std::set<std::string> used_;
  std::vector<std::unique_ptr<cv::FileStorage>> parsed_;  // Storages of the parsed overrides (their nodes refer to them)
};

}  // namespace

bool ConfigLoader::Read(const std::string& path, const std::vector<std::string>& overrides) {
  std::map<std::string, std::string> values;
  for (const std::string& argument : overrides) {
    const std::string text = (argument.rfind("--", 0) == 0) ? argument.substr(2) : argument;
    const size_t equal = text.find('=');
    if (equal == std::string::npos || equal == 0) {
      LOG_ERROR("Malformed option override '{}' (expected --section.key=value)", argument);
      return false;
    }
    values[text.substr(0, equal)] = text.substr(equal + 1);
  }

  cv::FileStorage file(path, cv::FileStorage::READ);
  if (!file.isOpened()) {
    LOG_ERROR("Cannot read the options file {}", path);
    return false;
  }
  OptionReader reader(file, values);
  VisualInertialOdometryOptions& o = odometry;

  // Camera
  std::string model = CameraModelName(o.camera.model_id);
  reader.Read("camera.model", &model);
  o.camera.model_id = CameraModelFromName(model);
  reader.Read("camera.width", &o.camera.width);
  reader.Read("camera.height", &o.camera.height);
  reader.Read("camera.params", &o.camera.params);
  reader.Read("camera.polar_exclusion_ratio", &o.camera.polar_exclusion_ratio);
  reader.Read("camera.border_margin", &o.camera.border_margin);
  reader.Read("camera.mask_rectangles", &o.camera.mask_rectangles);

  // IMU (single precision values, see ReadThroughFloat)
  reader.ReadThroughFloat("imu.gyro_noise", &o.imu_noise.gyro);
  reader.ReadThroughFloat("imu.accel_noise", &o.imu_noise.accel);
  reader.ReadThroughFloat("imu.gyro_walk", &o.imu_noise.gyro_walk);
  reader.ReadThroughFloat("imu.accel_walk", &o.imu_noise.accel_walk);
  reader.ReadThroughFloat("imu.gravity_magnitude", &o.gravity_magnitude);

  // Rig: imu_from_camera maps camera to IMU, p_I = R p_C + t, so p_C = R^T p_I - R^T t
  Eigen::Matrix4f imu_from_camera = Eigen::Matrix4f::Identity();
  reader.Read("rig.imu_from_camera", &imu_from_camera);
  const Eigen::Matrix4d T_IfromC = imu_from_camera.cast<double>();
  o.rig.R_ItoC = T_IfromC.block<3, 3>(0, 0).transpose();
  o.rig.p_IinC = -o.rig.R_ItoC * T_IfromC.block<3, 1>(0, 3);
  reader.Read("rig.clock_offset", &o.rig.clock_offset);

  // Stereo: a second camera with its own model and imu_from_camera1 (the IMU frame is shared, so the pair is fixed:
  // p_C1 = R_C0toC1 p_C0 + p_C0inC1 with R_C0toC1 = R_ItoC1 R_ItoC0^T, p_C0inC1 = p_IinC1 - R_C0toC1 p_IinC0)
  std::string model1;
  reader.Read("camera1.model", &model1);
  if (!model1.empty()) {
    o.camera1.model_id = CameraModelFromName(model1);
    if (o.camera1.model_id == CameraModelId::kInvalid) {
      LOG_ERROR("Unknown camera1.model '{}'", model1);
      return false;
    }
    reader.Read("camera1.width", &o.camera1.width);
    reader.Read("camera1.height", &o.camera1.height);
    reader.Read("camera1.params", &o.camera1.params);
    reader.Read("camera1.polar_exclusion_ratio", &o.camera1.polar_exclusion_ratio);
    reader.Read("camera1.border_margin", &o.camera1.border_margin);
    reader.Read("camera1.mask_rectangles", &o.camera1.mask_rectangles);
    Eigen::Matrix4f imu_from_camera1 = Eigen::Matrix4f::Identity();
    reader.Read("rig.imu_from_camera1", &imu_from_camera1);
    const Eigen::Matrix4d T_IfromC1 = imu_from_camera1.cast<double>();
    const Eigen::Matrix3d R_ItoC1 = T_IfromC1.block<3, 3>(0, 0).transpose();
    const Eigen::Vector3d p_IinC1 = -R_ItoC1 * T_IfromC1.block<3, 1>(0, 3);
    o.rig.R_C0toC1 = R_ItoC1 * o.rig.R_ItoC.transpose();
    o.rig.p_C0inC1 = p_IinC1 - o.rig.R_C0toC1 * o.rig.p_IinC;
  }

  // Feature tracker
  FeatureTrackerOptions& tracker = o.feature_tracker;
  reader.Read("feature_tracker.max_features", &tracker.max_features);
  reader.Read("feature_tracker.quality_level", &tracker.quality_level);
  reader.Read("feature_tracker.min_distance", &tracker.min_distance);
  reader.Read("feature_tracker.grid_cols", &tracker.grid_cols);
  reader.Read("feature_tracker.grid_rows", &tracker.grid_rows);
  reader.Read("feature_tracker.max_features_per_cell", &tracker.max_features_per_cell);
  reader.Read("feature_tracker.min_features_per_cell", &tracker.min_features_per_cell);
  reader.Read("feature_tracker.window_size", &tracker.window_size);
  reader.Read("feature_tracker.max_pyramid_level", &tracker.max_pyramid_level);
  reader.Read("feature_tracker.max_iterations", &tracker.max_iterations);
  reader.Read("feature_tracker.epsilon", &tracker.epsilon);
  reader.Read("feature_tracker.max_forward_backward_error", &tracker.max_forward_backward_error);
  reader.Read("feature_tracker.gain_compensation", &tracker.gain_compensation);
  reader.Read("feature_tracker.rotation_iterations", &tracker.rotation_iterations);
  reader.Read("feature_tracker.rotation_threshold_deg", &tracker.rotation_threshold_deg);
  reader.Read("feature_tracker.remove_light_clusters", &tracker.remove_light_clusters);
  reader.Read("feature_tracker.cluster_std_ratio", &tracker.cluster_std_ratio);
  reader.Read("feature_tracker.cluster_min_intensity", &tracker.cluster_min_intensity);
  reader.Read("feature_tracker.stereo_max_epipolar_error", &tracker.stereo_max_epipolar_error);

  // Landmark tracker
  LandmarkTrackerOptions& landmark_tracker = o.landmark_tracker;
  reader.Read("landmark_tracker.search_radius", &landmark_tracker.search_radius);
  reader.Read("landmark_tracker.min_ncc", &landmark_tracker.min_ncc);
  reader.Read("landmark_tracker.patch_half_size", &landmark_tracker.patch_half_size);
  reader.Read("landmark_tracker.loss_reasons", &landmark_tracker.loss_reasons);

  // Initializer
  InitializerOptions& init = o.initializer;
  reader.Read("initializer.window_time", &init.window_time);
  reader.Read("initializer.excitation_accel_std", &init.excitation_accel_std);
  reader.Read("initializer.max_still_disparity", &init.max_still_disparity);
  reader.Read("initializer.rotation_compensated_disparity", &init.rotation_compensated_disparity);
  reader.Read("initializer.max_features", &init.max_features);
  reader.Read("initializer.refine_max_iterations", &init.refine_max_iterations);
  reader.Read("initializer.num_poses", &init.num_poses);
  reader.Read("initializer.min_rotation_deg", &init.min_rotation_deg);
  reader.Read("initializer.covariance_scale_orientation", &init.covariance_scale_orientation);
  reader.Read("initializer.covariance_scale_velocity", &init.covariance_scale_velocity);
  reader.Read("initializer.covariance_scale_gyro_bias", &init.covariance_scale_gyro_bias);
  reader.Read("initializer.covariance_scale_accel_bias", &init.covariance_scale_accel_bias);
  reader.Read("initializer.min_covariance_rcond", &init.min_covariance_rcond);
  reader.Read("initializer.gyro_bias_prior", &init.gyro_bias_prior);
  reader.Read("initializer.accel_bias_prior", &init.accel_bias_prior);
  reader.Read("initializer.pixel_sigma", &init.pixel_sigma);
  reader.Read("initializer.rest_accel_bias_sigma", &init.rest_accel_bias_sigma);

  // Estimator
  EstimatorOptions& estimator = o.estimator;
  reader.Read("estimator.max_clones", &estimator.max_clones);
  reader.Read("estimator.max_landmarks", &estimator.max_landmarks);
  reader.Read("estimator.landmark_max_linearity", &estimator.landmark_max_linearity);
  reader.Read("estimator.landmark_relinearize_ratio", &estimator.landmark_relinearize_ratio);
  reader.Read("estimator.landmark_max_gate_rejections", &estimator.landmark_max_gate_rejections);
  reader.Read("estimator.landmark_dormant_time", &estimator.landmark_dormant_time);
  reader.Read("estimator.pixel_sigma", &estimator.pixel_sigma);
  reader.Read("estimator.time_offset_sigma", &estimator.time_offset_sigma);
  reader.Read("estimator.time_offset_walk", &estimator.time_offset_walk);
  reader.Read("estimator.extrinsic_rotation_sigma", &estimator.extrinsic_rotation_sigma);
  reader.Read("estimator.extrinsic_translation_sigma", &estimator.extrinsic_translation_sigma);
  reader.Read("estimator.camera_param_sigma", &estimator.camera_param_sigma);
  TriangulationOptions& triangulation = estimator.triangulation;
  reader.Read("estimator.triangulation.max_iterations", &triangulation.max_iterations);
  reader.Read("estimator.triangulation.initial_damping", &triangulation.initial_damping);
  reader.Read("estimator.triangulation.max_damping", &triangulation.max_damping);
  reader.Read("estimator.triangulation.min_step", &triangulation.min_step);
  reader.Read("estimator.triangulation.min_relative_decrease", &triangulation.min_relative_decrease);
  reader.Read("estimator.triangulation.damping_factor", &triangulation.damping_factor);
  reader.Read("estimator.triangulation.min_depth", &triangulation.min_depth);
  reader.Read("estimator.triangulation.max_depth", &triangulation.max_depth);
  reader.Read("estimator.triangulation.max_depth_baseline_ratio", &triangulation.max_depth_baseline_ratio);
  reader.Read("estimator.triangulation.max_condition_number", &triangulation.max_condition_number);

  const std::vector<std::string> unused = reader.UnusedOverrides();
  if (!unused.empty()) {
    for (const std::string& key : unused) {
      LOG_ERROR("Unknown option '{}'", key);
    }
    return false;
  }
  return true;
}

}  // namespace vio

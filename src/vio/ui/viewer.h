// SPDX-License-Identifier: MIT
/**
 * @file      viewer.h
 * @brief     Viewer of the odometry: 3D scene, image with feature states and time-series plots.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2026-09-28
 * @copyright Copyright (c) 2026 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include "vio/estimators/estimator.h"
#include "vio/odometry/visual_inertial_odometry.h"

#include <Eigen/Dense>
#include <imgui.h>
#include <opencv2/core.hpp>

#include <string>
#include <vector>

struct GLFWwindow;

namespace vio {

// State of a tracked feature in the current image
enum class FeatureState {
  kTracking,       // Tracked, not used by an update yet
  kAccepted,       // Used by the window update of this image
  kGate,           // Rejected by the chi-square gate
  kTriangulation,  // Rejected by the triangulation
  kTooFew,         // Finished with fewer than two measurements
  kLandmark,       // Landmark in the estimator state
};

// Tracked feature drawn on the image
struct ViewerFeature {
  Eigen::Vector2d uv;  // Pixel coordinates
  FeatureState state;  // State in this image
};

// Everything the viewer shows for one image
struct ViewerFrame {
  int index = 0;                    // Image index
  double timestamp = 0.0;           // Image time [s]
  bool initialized = false;         // Estimator state is valid
  std::string status;               // Initialization or estimator status text
  cv::Mat image;                    // Grayscale image
  std::vector<ViewerFeature> features;

  Eigen::Matrix3d R_ItoG = Eigen::Matrix3d::Identity();  // Orientation (IMU -> world)
  Eigen::Vector3d p = Eigen::Vector3d::Zero();           // Position [m]
  Eigen::Vector3d v = Eigen::Vector3d::Zero();           // Velocity [m/s]
  Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
  Eigen::Vector3d accel_bias = Eigen::Vector3d::Zero();
  Eigen::Matrix<double, 15, 1> sigma = Eigen::Matrix<double, 15, 1>::Zero();  // sqrt(diag(P)) of the IMU state
  std::vector<Eigen::Matrix3d> clone_R;          // Clone orientations (IMU -> world)
  std::vector<Eigen::Vector3d> clone_p;          // Clone positions
  std::vector<Eigen::Vector3d> accepted_points;  // Points used by the update
  std::vector<Eigen::Vector3d> rejected_points;  // Triangulated points rejected by the gate
  std::vector<Eigen::Vector3d> landmarks;        // Landmarks in the estimator state
  EstimatorStatistics stats;                     // Update statistics

  // Everything to show after the odometry processed an image
  static ViewerFrame FromOdometry(const VisualInertialOdometry& odometry, const cv::Mat& image);
};

/**
 * @brief ImGui/ImPlot viewer
 *
 * Usage: Initialize() once, SetGroundTruth() if available; Push() every processed image, RenderFrame() in the loop.
 */
class Viewer {
 public:
  Viewer(int width, int height);
  ~Viewer();

  // Create the window and the GUI contexts; false if the window could not be created
  bool Initialize();

  // Ground-truth positions (aligned to the estimate while running)
  void SetGroundTruth(const std::vector<double>& times, const std::vector<Eigen::Vector3d>& positions);

  void Push(const ViewerFrame& frame);

  // Poll events and draw one GUI frame; false when the window was closed
  bool RenderFrame();

  // Save the window content at the end of the next RenderFrame()
  void RequestScreenshot(const std::string& path) { screenshot_request_ = path; }

  // Select a plot tab (0: motion, 1: updates, 2: uncertainty, 3: biases) in the next RenderFrame()
  void SelectPlotTab(int tab) { plot_tab_request_ = tab; }

  bool IsPaused() const { return paused_; }
  bool IsRealtime() const { return realtime_; }

  // true if the user asked for one step while paused (consumes the request)
  bool ConsumeStepRequest();

 private:
  void DrawScene(const ImVec2& pos, const ImVec2& size);
  void DrawImagePanel(const ImVec2& pos, const ImVec2& size);
  void DrawControlPanel(const ImVec2& pos, const ImVec2& size);
  void DrawPlotPanel(const ImVec2& pos, const ImVec2& size);

  // Rigid fit of the estimate onto the ground truth over the history
  void UpdateAlignment();

  // Ground-truth position at a time, interpolated; false outside its time range
  bool GroundTruthAt(double t, Eigen::Vector3d& p) const;

  void UploadImage(const cv::Mat& gray);

  // Save the back buffer (before the buffers are swapped)
  bool SaveScreenshot(const std::string& path);

  int width_;                       // Initial window size
  int height_;
  GLFWwindow* window_;
  unsigned int texture_;            // Image texture
  int texture_width_;
  int texture_height_;

  // Playback
  bool paused_;
  bool step_requested_;
  bool realtime_;
  std::string screenshot_request_;  // Path requested from the GUI
  int plot_tab_request_;            // Plot tab to select, -1 for none

  // Scene options and orbit camera
  bool follow_;
  bool show_ground_truth_;
  bool show_points_;
  bool show_clones_;
  bool show_grid_;
  float yaw_;
  float pitch_;
  float distance_;
  Eigen::Vector3d target_;

  // Data
  ViewerFrame current_;
  bool has_frame_;
  std::vector<double> gt_times_;
  std::vector<Eigen::Vector3d> gt_positions_;
  std::vector<double> gt_speed_;
  std::vector<double> times_;       // History of processed images with a state
  std::vector<Eigen::Vector3d> positions_;
  std::vector<Eigen::Vector3d> map_points_;  // Accumulated accepted points
  Eigen::Matrix3d align_R_;         // Estimate -> ground truth
  Eigen::Vector3d align_t_;
  bool aligned_;

  // Plot series
  std::vector<double> plot_t_;
  std::vector<double> speed_est_;
  std::vector<double> speed_gt_;
  std::vector<double> position_error_;
  std::vector<double> candidates_;
  std::vector<double> accepted_;
  std::vector<double> gate_rejected_;
  std::vector<double> triangulation_failed_;
  std::vector<double> nis_;
  std::vector<double> bg_[3];
  std::vector<double> ba_[3];
  std::vector<double> sigma_theta_;
  std::vector<double> sigma_p_;
  std::vector<double> sigma_v_;
};

}  // namespace vio

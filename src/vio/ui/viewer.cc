// SPDX-License-Identifier: MIT
/**
 * @file      viewer.cc
 * @brief     Implements the ImGui/ImPlot viewer of the odometry.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2026-09-28
 * @copyright Copyright (c) 2026 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "vio/ui/viewer.h"

#include "vio/geometry/lie_group.h"

#include <GLFW/glfw3.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <implot.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include <unordered_map>

namespace vio {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// Colors shared by the scene, the image overlay and the legend
const ImU32 kColorEstimate = IM_COL32(255, 150, 40, 255);
const ImU32 kColorGroundTruth = IM_COL32(80, 210, 110, 255);
const ImU32 kColorTracking = IM_COL32(120, 200, 255, 200);
const ImU32 kColorAccepted = IM_COL32(60, 230, 90, 255);
const ImU32 kColorGate = IM_COL32(255, 60, 60, 255);
const ImU32 kColorTriangulation = IM_COL32(255, 190, 40, 255);
const ImU32 kColorTooFew = IM_COL32(150, 150, 150, 200);
const ImU32 kColorMap = IM_COL32(200, 200, 210, 110);
const ImU32 kColorClone = IM_COL32(120, 170, 255, 220);
const ImU32 kColorLandmark = IM_COL32(240, 90, 240, 255);

ImVec4 ToVec4(ImU32 color) { return ImGui::ColorConvertU32ToFloat4(color); }

ImU32 ColorOf(FeatureState state) {
  switch (state) {
    case FeatureState::kAccepted:
      return kColorAccepted;
    case FeatureState::kGate:
      return kColorGate;
    case FeatureState::kTriangulation:
      return kColorTriangulation;
    case FeatureState::kTooFew:
      return kColorTooFew;
    case FeatureState::kLandmark:
      return kColorLandmark;
    default:
      return kColorTracking;
  }
}

}  // namespace

ViewerFrame ViewerFrame::FromOdometry(const VisualInertialOdometry& odometry, const cv::Mat& image) {
  const FrameReport& report = odometry.report();
  const Estimator& estimator = odometry.estimator();
  ViewerFrame frame;
  frame.index = report.index;
  frame.image = image;
  frame.initialized = report.has_state;

  // The state is at the sensor-0 exposure (image time plus the estimated time offset)
  frame.timestamp = frame.initialized ? estimator.timestamp() : report.timestamp;
  std::unordered_map<int, FeatureState> states;
  if (frame.initialized) {
    for (const auto& track : estimator.track_reports()) {
      FeatureState state = FeatureState::kTooFew;
      if (track.outcome == TrackOutcome::kAccepted) {
        state = FeatureState::kAccepted;
        if (track.has_point) frame.accepted_points.push_back(track.point);
      } else if (track.outcome == TrackOutcome::kGate) {
        state = FeatureState::kGate;
        if (track.has_point) frame.rejected_points.push_back(track.point);
      } else if (track.outcome == TrackOutcome::kTriangulation || track.outcome == TrackOutcome::kProjection) {
        state = FeatureState::kTriangulation;
      } else if (track.outcome == TrackOutcome::kLandmark) {
        state = FeatureState::kLandmark;
      }
      states[track.id] = state;
    }
  }
  for (const auto& observation : report.observations) {
    const auto it = states.find(observation.id);
    frame.features.push_back({observation.pixel, it == states.end() ? FeatureState::kTracking : it->second});
  }

  if (!frame.initialized) {
    const InitializerResult& init = report.initialization;
    if (report.initialized) {
      frame.status = "initialized (" + init.method + "), the estimator starts with the next image";
    } else if (estimator.IsInitialized()) {
      frame.status = "estimator stopped: " + estimator.failure();
    } else {
      frame.status = "initializing: " + (init.reason.empty() ? std::string("waiting for data") : init.reason);
    }
    return frame;
  }

  frame.status = "tracking";
  frame.R_ItoG = jpl::ToRotation(estimator.orientation()).transpose();
  frame.p = estimator.position();
  frame.v = estimator.velocity();
  frame.gyro_bias = estimator.gyro_bias();
  frame.accel_bias = estimator.accel_bias();
  frame.sigma = estimator.covariance().topLeftCorner<15, 15>().diagonal().cwiseMax(0.0).cwiseSqrt();
  for (const auto& [q, p] : estimator.ClonePoses()) {
    frame.clone_R.push_back(jpl::ToRotation(q).transpose());
    frame.clone_p.push_back(p);
  }
  for (const auto& [id, p] : estimator.Landmarks()) {
    frame.landmarks.push_back(p);
  }
  frame.stats = estimator.statistics();
  return frame;
}

Viewer::Viewer(int width, int height)
    : width_(width),
      height_(height),
      window_(nullptr),
      texture_(0),
      texture_width_(0),
      texture_height_(0),
      paused_(false),
      step_requested_(false),
      realtime_(false),
      plot_tab_request_(-1),
      follow_(true),
      show_ground_truth_(true),
      show_points_(true),
      show_clones_(true),
      show_grid_(true),
      yaw_(-2.3f),
      pitch_(0.6f),
      distance_(40.0f),
      target_(Eigen::Vector3d::Zero()),
      has_frame_(false),
      align_R_(Eigen::Matrix3d::Identity()),
      align_t_(Eigen::Vector3d::Zero()),
      aligned_(false) {}

Viewer::~Viewer() {
  if (!window_) {
    return;
  }
  if (texture_) {
    glDeleteTextures(1, &texture_);
  }
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImPlot::DestroyContext();
  ImGui::DestroyContext();
  glfwDestroyWindow(window_);
  glfwTerminate();
}

bool Viewer::Initialize() {
  if (!glfwInit()) {
    return false;
  }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  window_ = glfwCreateWindow(width_, height_, "360 VIO viewer", nullptr, nullptr);
  if (!window_) {
    glfwTerminate();
    return false;
  }
  glfwMakeContextCurrent(window_);
  glfwSwapInterval(1);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImPlot::CreateContext();
  ImGui::GetIO().IniFilename = nullptr;
  ImGui::StyleColorsDark();
  ImGui::GetStyle().WindowRounding = 0.0f;
  ImGui_ImplGlfw_InitForOpenGL(window_, true);
  ImGui_ImplOpenGL3_Init("#version 130");

  glGenTextures(1, &texture_);
  return true;
}

void Viewer::SetGroundTruth(const std::vector<double>& times, const std::vector<Eigen::Vector3d>& positions) {
  gt_times_ = times;
  gt_positions_ = positions;

  // Speed from central differences over +-0.4 s
  gt_speed_.assign(times.size(), kNaN);
  for (size_t i = 0; i < times.size(); ++i) {
    const auto a = std::lower_bound(times.begin(), times.end(), times[i] - 0.4) - times.begin();
    const auto b = std::upper_bound(times.begin(), times.end(), times[i] + 0.4) - times.begin() - 1;
    if (b > a && times[b] - times[a] > 0.4) {
      gt_speed_[i] = (positions[b] - positions[a]).norm() / (times[b] - times[a]);
    }
  }
}

bool Viewer::GroundTruthAt(double t, Eigen::Vector3d& p) const {
  if (gt_times_.size() < 2 || t < gt_times_.front() || t > gt_times_.back()) {
    return false;
  }
  const size_t i = std::upper_bound(gt_times_.begin(), gt_times_.end(), t) - gt_times_.begin();
  if (i == 0 || i >= gt_times_.size()) {
    p = (i == 0) ? gt_positions_.front() : gt_positions_.back();
    return true;
  }
  const double u = (t - gt_times_[i - 1]) / (gt_times_[i] - gt_times_[i - 1]);
  p = (1.0 - u) * gt_positions_[i - 1] + u * gt_positions_[i];
  return true;
}

void Viewer::UpdateAlignment() {
  // Rigid (rotation + translation) least-squares fit of the estimate onto the ground truth
  std::vector<Eigen::Vector3d> est;
  std::vector<Eigen::Vector3d> gt;
  for (size_t i = 0; i < times_.size(); ++i) {
    Eigen::Vector3d g;
    if (GroundTruthAt(times_[i], g)) {
      est.push_back(positions_[i]);
      gt.push_back(g);
    }
  }
  if (est.size() < 10) {
    return;
  }
  Eigen::Vector3d mean_e = Eigen::Vector3d::Zero();
  Eigen::Vector3d mean_g = Eigen::Vector3d::Zero();
  for (size_t i = 0; i < est.size(); ++i) {
    mean_e += est[i];
    mean_g += gt[i];
  }
  mean_e /= static_cast<double>(est.size());
  mean_g /= static_cast<double>(est.size());
  Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
  for (size_t i = 0; i < est.size(); ++i) {
    cov += (gt[i] - mean_g) * (est[i] - mean_e).transpose();
  }
  // Too little motion to define the rotation
  if (cov.norm() < 1.0) {
    return;
  }
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
  if ((svd.matrixU() * svd.matrixV().transpose()).determinant() < 0.0) {
    D(2, 2) = -1.0;
  }
  align_R_ = svd.matrixU() * D * svd.matrixV().transpose();
  align_t_ = mean_g - align_R_ * mean_e;
  aligned_ = true;

  // Position error of the whole history with the new alignment
  for (size_t i = 0; i < times_.size() && i < position_error_.size(); ++i) {
    Eigen::Vector3d g;
    position_error_[i] = GroundTruthAt(times_[i], g) ? (align_R_ * positions_[i] + align_t_ - g).norm() : kNaN;
  }
}

void Viewer::UploadImage(const cv::Mat& gray) {
  cv::Mat rgb;
  cv::cvtColor(gray, rgb, cv::COLOR_GRAY2RGB);
  glBindTexture(GL_TEXTURE_2D, texture_);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rgb.cols, rgb.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb.data);
  texture_width_ = rgb.cols;
  texture_height_ = rgb.rows;
}

void Viewer::Push(const ViewerFrame& frame) {
  current_ = frame;
  has_frame_ = true;
  if (!frame.image.empty()) {
    UploadImage(frame.image);
  }
  if (!frame.initialized) {
    return;
  }

  times_.push_back(frame.timestamp);
  positions_.push_back(frame.p);
  map_points_.insert(map_points_.end(), frame.accepted_points.begin(), frame.accepted_points.end());
  const size_t max_points = 60000;
  if (map_points_.size() > max_points) {
    map_points_.erase(map_points_.begin(), map_points_.end() - max_points);
  }

  // Time series
  const EstimatorStatistics& s = frame.stats;
  plot_t_.push_back(frame.timestamp - times_.front());
  speed_est_.push_back(frame.v.norm());
  double gt_speed = kNaN;
  if (gt_times_.size() >= 2 && frame.timestamp >= gt_times_.front() && frame.timestamp <= gt_times_.back()) {
    const size_t i = std::upper_bound(gt_times_.begin(), gt_times_.end(), frame.timestamp) - gt_times_.begin();
    gt_speed = gt_speed_[std::min(i, gt_speed_.size() - 1)];
  }
  speed_gt_.push_back(gt_speed);
  Eigen::Vector3d g;
  position_error_.push_back((aligned_ && GroundTruthAt(frame.timestamp, g))
                                ? (align_R_ * frame.p + align_t_ - g).norm()
                                : kNaN);
  candidates_.push_back(s.candidates - s.too_few_measurements);
  accepted_.push_back(s.accepted);
  gate_rejected_.push_back(s.gate_rejected);
  triangulation_failed_.push_back(s.triangulation_failed);
  nis_.push_back(s.accepted > 0 ? s.nis_accepted_sum / s.accepted : kNaN);
  for (int k = 0; k < 3; ++k) {
    bg_[k].push_back(frame.gyro_bias(k));
    ba_[k].push_back(frame.accel_bias(k));
  }
  sigma_theta_.push_back(frame.sigma.segment<3>(0).norm() * 180.0 / M_PI);
  sigma_p_.push_back(frame.sigma.segment<3>(3).norm());
  sigma_v_.push_back(frame.sigma.segment<3>(6).norm());

  if (positions_.size() < 60 || positions_.size() % 30 == 0) {
    UpdateAlignment();
  }
}

bool Viewer::ConsumeStepRequest() {
  const bool requested = step_requested_;
  step_requested_ = false;
  return requested;
}

bool Viewer::RenderFrame() {
  if (!window_ || glfwWindowShouldClose(window_)) {
    return false;
  }
  glfwPollEvents();
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  if (!ImGui::GetIO().WantCaptureKeyboard) {
    if (ImGui::IsKeyPressed(ImGuiKey_Space)) {
      paused_ = !paused_;
    }
    if (ImGui::IsKeyPressed(ImGuiKey_N)) {
      step_requested_ = true;
    }
  }

  // Layout: 3D scene and plots on the left, image and status on the right
  const ImVec2 display = ImGui::GetIO().DisplaySize;
  const float left_width = std::floor(display.x * 0.62f);
  const float right_width = display.x - left_width;
  const float scene_height = std::floor(display.y * 0.62f);
  const float image_height = std::min(display.y * 0.5f, right_width * 0.5f + 60.0f);
  DrawScene(ImVec2(0.0f, 0.0f), ImVec2(left_width, scene_height));
  DrawPlotPanel(ImVec2(0.0f, scene_height), ImVec2(left_width, display.y - scene_height));
  DrawImagePanel(ImVec2(left_width, 0.0f), ImVec2(right_width, image_height));
  DrawControlPanel(ImVec2(left_width, image_height), ImVec2(right_width, display.y - image_height));

  ImGui::Render();
  int fb_width = 0;
  int fb_height = 0;
  glfwGetFramebufferSize(window_, &fb_width, &fb_height);
  glViewport(0, 0, fb_width, fb_height);
  glClearColor(0.08f, 0.08f, 0.10f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

  if (!screenshot_request_.empty()) {
    SaveScreenshot(screenshot_request_);
    screenshot_request_.clear();
  }
  glfwSwapBuffers(window_);
  return true;
}

bool Viewer::SaveScreenshot(const std::string& path) {
  int width = 0;
  int height = 0;
  glfwGetFramebufferSize(window_, &width, &height);
  cv::Mat rgb(height, width, CV_8UC3);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, rgb.data);
  cv::Mat bgr;
  cv::flip(rgb, rgb, 0);
  cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
  return cv::imwrite(path, bgr);
}

void Viewer::DrawScene(const ImVec2& pos, const ImVec2& size) {
  const ImGuiWindowFlags flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse |
                                 ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;
  ImGui::SetNextWindowPos(pos);
  ImGui::SetNextWindowSize(size);
  ImGui::Begin("3D view  (left drag: rotate, right drag: pan, wheel: zoom)", nullptr, flags);

  const ImVec2 origin = ImGui::GetCursorScreenPos();
  const ImVec2 canvas = ImGui::GetContentRegionAvail();
  ImGui::InvisibleButton("scene", canvas, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
  const ImGuiIO& io = ImGui::GetIO();

  // Orbit camera around the target (world z is up)
  if (ImGui::IsItemHovered() && io.MouseWheel != 0.0f) {
    distance_ = std::clamp(distance_ * std::pow(0.88f, io.MouseWheel), 1.0f, 2000.0f);
  }
  if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
    yaw_ -= io.MouseDelta.x * 0.008f;
    pitch_ = std::clamp(pitch_ + io.MouseDelta.y * 0.008f, -1.5f, 1.5f);
  }
  const Eigen::Vector3d eye_dir(std::cos(pitch_) * std::cos(yaw_), std::cos(pitch_) * std::sin(yaw_),
                                std::sin(pitch_));
  const Eigen::Vector3d forward = -eye_dir;
  const Eigen::Vector3d right = forward.cross(Eigen::Vector3d::UnitZ()).normalized();
  const Eigen::Vector3d up = right.cross(forward);
  if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Right)) {
    const double scale = distance_ * 0.0015;
    target_ += (-io.MouseDelta.x * right + io.MouseDelta.y * up) * scale;
    follow_ = false;
  }
  if (follow_ && current_.initialized) {
    target_ = current_.p;
  }
  const Eigen::Vector3d eye = target_ + distance_ * eye_dir;
  const double focal = 0.5 * canvas.y / std::tan(30.0 * M_PI / 180.0);
  const ImVec2 center(origin.x + 0.5f * canvas.x, origin.y + 0.5f * canvas.y);
  auto project = [&](const Eigen::Vector3d& p, ImVec2& out) {
    const Eigen::Vector3d d = p - eye;
    const double z = d.dot(forward);
    if (z < 0.05) {
      return false;
    }
    out = ImVec2(static_cast<float>(center.x + focal * d.dot(right) / z),
                 static_cast<float>(center.y - focal * d.dot(up) / z));
    return true;
  };

  ImDrawList* draw = ImGui::GetWindowDrawList();
  draw->PushClipRect(origin, ImVec2(origin.x + canvas.x, origin.y + canvas.y), true);
  draw->AddRectFilled(origin, ImVec2(origin.x + canvas.x, origin.y + canvas.y), IM_COL32(18, 20, 26, 255));
  auto line = [&](const Eigen::Vector3d& a, const Eigen::Vector3d& b, ImU32 color, float width) {
    ImVec2 pa, pb;
    if (project(a, pa) && project(b, pb)) {
      draw->AddLine(pa, pb, color, width);
    }
  };
  auto axes = [&](const Eigen::Matrix3d& R, const Eigen::Vector3d& p, double length, float width) {
    line(p, p + length * R.col(0), IM_COL32(230, 70, 70, 255), width);
    line(p, p + length * R.col(1), IM_COL32(70, 230, 70, 255), width);
    line(p, p + length * R.col(2), IM_COL32(80, 120, 255, 255), width);
  };

  // Ground grid below the trajectory
  if (show_grid_) {
    double ground = target_.z() - 1.0;
    for (const auto& p : positions_) {
      ground = std::min(ground, p.z() - 0.5);
    }
    const double spacing = std::pow(10.0, std::floor(std::log10(std::max(1.0, distance_ / 4.0))));
    const int half = 10;
    const double cx = std::round(target_.x() / spacing) * spacing;
    const double cy = std::round(target_.y() / spacing) * spacing;
    for (int k = -half; k <= half; ++k) {
      const ImU32 color = (k % 5 == 0) ? IM_COL32(70, 75, 90, 255) : IM_COL32(45, 48, 58, 255);
      line(Eigen::Vector3d(cx + k * spacing, cy - half * spacing, ground),
           Eigen::Vector3d(cx + k * spacing, cy + half * spacing, ground), color, 1.0f);
      line(Eigen::Vector3d(cx - half * spacing, cy + k * spacing, ground),
           Eigen::Vector3d(cx + half * spacing, cy + k * spacing, ground), color, 1.0f);
    }
  }

  // Accumulated map points and the points of the last update
  if (show_points_) {
    ImVec2 q;
    for (const auto& p : map_points_) {
      if (project(p, q)) {
        draw->AddRectFilled(ImVec2(q.x - 1.0f, q.y - 1.0f), ImVec2(q.x + 1.0f, q.y + 1.0f), kColorMap);
      }
    }
    for (const auto& p : current_.accepted_points) {
      if (project(p, q)) {
        draw->AddCircleFilled(q, 2.5f, kColorAccepted);
      }
    }
    for (const auto& p : current_.rejected_points) {
      if (project(p, q)) {
        draw->AddCircleFilled(q, 2.5f, kColorGate);
      }
    }
    for (const auto& p : current_.landmarks) {
      if (project(p, q)) {
        draw->AddCircleFilled(q, 3.5f, kColorLandmark);
      }
    }
  }

  // Ground truth (aligned into the estimate frame) and the estimate
  if (show_ground_truth_ && aligned_ && !times_.empty()) {
    Eigen::Vector3d previous;
    bool has_previous = false;
    for (size_t i = 0; i < gt_times_.size(); ++i) {
      if (gt_times_[i] < times_.front() || gt_times_[i] > times_.back()) {
        continue;
      }
      const Eigen::Vector3d p = align_R_.transpose() * (gt_positions_[i] - align_t_);
      if (has_previous) {
        line(previous, p, kColorGroundTruth, 2.0f);
      }
      previous = p;
      has_previous = true;
    }
  }
  for (size_t i = 1; i < positions_.size(); ++i) {
    line(positions_[i - 1], positions_[i], kColorEstimate, 2.0f);
  }

  // Clones and the current pose with its velocity
  if (current_.initialized) {
    const double scale = 0.03 * distance_;
    if (show_clones_) {
      for (size_t i = 0; i < current_.clone_p.size(); ++i) {
        ImVec2 q;
        if (project(current_.clone_p[i], q)) {
          draw->AddCircle(q, 3.0f, kColorClone);
        }
      }
    }
    axes(current_.R_ItoG, current_.p, 2.0 * scale, 3.0f);
    line(current_.p, current_.p + current_.v, IM_COL32(255, 240, 80, 255), 2.0f);
  }
  draw->PopClipRect();

  // Legend
  ImGui::SetCursorScreenPos(ImVec2(origin.x + 10.0f, origin.y + 8.0f));
  ImGui::TextColored(ToVec4(kColorEstimate), "estimate");
  ImGui::SameLine();
  ImGui::TextColored(ToVec4(kColorGroundTruth), "  ground truth (rigid-aligned)");
  ImGui::SameLine();
  ImGui::TextColored(ToVec4(kColorAccepted), "  update points");
  ImGui::SameLine();
  ImGui::TextColored(ToVec4(kColorLandmark), "  landmarks");
  ImGui::SameLine();
  ImGui::TextColored(ToVec4(kColorMap), "  map");
  ImGui::SameLine();
  ImGui::TextColored(ImVec4(1.0f, 0.94f, 0.3f, 1.0f), "  velocity");
  ImGui::End();
}

void Viewer::DrawImagePanel(const ImVec2& pos, const ImVec2& size) {
  const ImGuiWindowFlags flags =
      ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar;
  ImGui::SetNextWindowPos(pos);
  ImGui::SetNextWindowSize(size);
  ImGui::Begin("Image", nullptr, flags);
  if (texture_width_ > 0) {
    const float width = ImGui::GetContentRegionAvail().x;
    const float scale = width / static_cast<float>(texture_width_);
    const ImVec2 image_size(width, texture_height_ * scale);
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    ImGui::Image(static_cast<ImTextureID>(texture_), image_size);

    ImDrawList* draw = ImGui::GetWindowDrawList();
    int counts[6] = {0, 0, 0, 0, 0, 0};
    for (const auto& feature : current_.features) {
      const ImVec2 q(origin.x + static_cast<float>(feature.uv.x()) * scale,
                     origin.y + static_cast<float>(feature.uv.y()) * scale);
      const ImU32 color = ColorOf(feature.state);
      if (feature.state == FeatureState::kTracking) {
        draw->AddCircle(q, 2.5f, color);
      } else {
        draw->AddCircleFilled(q, 3.5f, color);
      }
      counts[static_cast<int>(feature.state)]++;
    }
    ImGui::TextColored(ToVec4(kColorTracking), "tracking %d", counts[0]);
    ImGui::SameLine();
    ImGui::TextColored(ToVec4(kColorAccepted), "  accepted %d", counts[1]);
    ImGui::SameLine();
    ImGui::TextColored(ToVec4(kColorGate), "  gate %d", counts[2]);
    ImGui::SameLine();
    ImGui::TextColored(ToVec4(kColorTriangulation), "  triangulation %d", counts[3]);
    ImGui::SameLine();
    ImGui::TextColored(ToVec4(kColorTooFew), "  short %d", counts[4]);
    ImGui::SameLine();
    ImGui::TextColored(ToVec4(kColorLandmark), "  landmark %d", counts[5]);
  }
  ImGui::End();
}

void Viewer::DrawControlPanel(const ImVec2& pos, const ImVec2& size) {
  const ImGuiWindowFlags flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse;
  ImGui::SetNextWindowPos(pos);
  ImGui::SetNextWindowSize(size);
  ImGui::Begin("Status", nullptr, flags);

  if (ImGui::Button(paused_ ? "Play (space)" : "Pause (space)")) {
    paused_ = !paused_;
  }
  ImGui::SameLine();
  if (ImGui::Button("Step (n)")) {
    step_requested_ = true;
  }
  ImGui::SameLine();
  ImGui::Checkbox("real time", &realtime_);
  ImGui::SameLine();
  if (ImGui::Button("Screenshot")) {
    screenshot_request_ = "vio_viewer_" + std::to_string(current_.index) + ".png";
  }
  ImGui::Checkbox("follow", &follow_);
  ImGui::SameLine();
  ImGui::Checkbox("ground truth", &show_ground_truth_);
  ImGui::SameLine();
  ImGui::Checkbox("points", &show_points_);
  ImGui::SameLine();
  ImGui::Checkbox("clones", &show_clones_);
  ImGui::SameLine();
  ImGui::Checkbox("grid", &show_grid_);
  ImGui::Separator();

  ImGui::Text("image %d   t = %.3f s", current_.index, current_.timestamp);
  ImGui::TextWrapped("%s", current_.status.c_str());
  if (!current_.initialized) {
    ImGui::End();
    return;
  }

  const ViewerFrame& f = current_;
  if (ImGui::BeginTable("state", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV)) {
    ImGui::TableSetupColumn("");
    ImGui::TableSetupColumn("x");
    ImGui::TableSetupColumn("y");
    ImGui::TableSetupColumn("z");
    ImGui::TableHeadersRow();
    auto row = [](const char* name, const Eigen::Vector3d& v, const char* format) {
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::TextUnformatted(name);
      for (int k = 0; k < 3; ++k) {
        ImGui::TableNextColumn();
        ImGui::Text(format, v(k));
      }
    };
    row("position [m]", f.p, "%.2f");
    row("velocity [m/s]", f.v, "%.2f");
    row("gyro bias [rad/s]", f.gyro_bias, "%.4f");
    row("accel bias [m/s2]", f.accel_bias, "%.3f");
    row("sigma p [m]", f.sigma.segment<3>(3), "%.3f");
    row("sigma v [m/s]", f.sigma.segment<3>(6), "%.3f");
    ImGui::EndTable();
  }
  ImGui::Text("speed %.2f m/s   clones %zu", f.v.norm(), f.clone_p.size());

  const EstimatorStatistics& s = f.stats;
  ImGui::Separator();
  ImGui::Text("update: %d finished tracks, %d with >= 2 measurements", s.candidates,
              s.candidates - s.too_few_measurements);
  ImGui::TextColored(ToVec4(kColorAccepted), "accepted %d  (rows %d, NIS %.2f)", s.accepted, s.update_rows,
                     s.accepted > 0 ? s.nis_accepted_sum / s.accepted : 0.0);
  ImGui::TextColored(ToVec4(kColorGate), "gate rejected %d", s.gate_rejected);
  ImGui::TextColored(ToVec4(kColorTriangulation), "triangulation failed %d  (cond %d, depth %d/%d, baseline %d)",
                     s.triangulation_failed, s.triangulation_ill_conditioned, s.triangulation_linear_depth,
                     s.triangulation_refined_depth, s.triangulation_baseline);
  ImGui::TextColored(ToVec4(kColorLandmark), "landmarks %d in state (updated %d, new %d, gate %d, lost %d)",
                     s.landmarks_in_state, s.landmarks_updated, s.landmarks_initialized, s.landmarks_gate_rejected,
                     s.landmarks_lost);

  if (aligned_ && !position_error_.empty()) {
    double sum = 0.0;
    int count = 0;
    for (double e : position_error_) {
      if (!std::isnan(e)) {
        sum += e * e;
        count++;
      }
    }
    ImGui::Separator();
    ImGui::Text("position error now %.2f m   RMSE so far %.2f m (rigid alignment)", position_error_.back(),
                count > 0 ? std::sqrt(sum / count) : 0.0);
  }
  ImGui::End();
}

void Viewer::DrawPlotPanel(const ImVec2& pos, const ImVec2& size) {
  const ImGuiWindowFlags flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse;
  ImGui::SetNextWindowPos(pos);
  ImGui::SetNextWindowSize(size);
  ImGui::Begin("Plots", nullptr, flags);

  const int n = static_cast<int>(plot_t_.size());
  const double* t = plot_t_.data();
  const float half_width = ImGui::GetContentRegionAvail().x * 0.5f - 4.0f;
  auto plot = [&](const char* title, const char* y_label,
                  std::initializer_list<std::tuple<const char*, const std::vector<double>*, ImU32>> series) {
    if (ImPlot::BeginPlot(title, ImVec2(half_width, ImGui::GetContentRegionAvail().y))) {
      ImPlot::SetupAxes("time since init [s]", y_label, ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
      for (const auto& item : series) {
        ImPlotSpec spec;
        spec.LineColor = ToVec4(std::get<2>(item));
        spec.LineWeight = 1.5f;
        ImPlot::PlotLine(std::get<0>(item), t, std::get<1>(item)->data(), n, spec);
      }
      ImPlot::EndPlot();
    }
  };

  auto tab_flags = [&](int tab) {
    return (plot_tab_request_ == tab) ? ImGuiTabItemFlags_SetSelected : ImGuiTabItemFlags_None;
  };
  if (ImGui::BeginTabBar("plots")) {
    if (ImGui::BeginTabItem("Motion", nullptr, tab_flags(0))) {
      plot("Speed", "m/s", {{"estimate", &speed_est_, kColorEstimate}, {"ground truth", &speed_gt_, kColorGroundTruth}});
      ImGui::SameLine();
      plot("Position error (rigid-aligned)", "m", {{"error", &position_error_, IM_COL32(255, 90, 90, 255)}});
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Updates", nullptr, tab_flags(1))) {
      plot("Tracks per update", "count",
           {{">= 2 measurements", &candidates_, kColorTracking},
            {"accepted", &accepted_, kColorAccepted},
            {"gate", &gate_rejected_, kColorGate},
            {"triangulation", &triangulation_failed_, kColorTriangulation}});
      ImGui::SameLine();
      plot("NIS of accepted tracks (consistent ~ 1)", "chi2 / dof", {{"mean NIS", &nis_, IM_COL32(200, 150, 255, 255)}});
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Uncertainty", nullptr, tab_flags(2))) {
      plot("Position / velocity sigma", "m, m/s", {{"sigma p", &sigma_p_, kColorEstimate}, {"sigma v", &sigma_v_, kColorClone}});
      ImGui::SameLine();
      plot("Orientation sigma", "deg", {{"sigma theta", &sigma_theta_, IM_COL32(230, 200, 90, 255)}});
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Biases", nullptr, tab_flags(3))) {
      plot("Gyro bias", "rad/s",
           {{"x", &bg_[0], IM_COL32(230, 70, 70, 255)},
            {"y", &bg_[1], IM_COL32(70, 230, 70, 255)},
            {"z", &bg_[2], IM_COL32(80, 120, 255, 255)}});
      ImGui::SameLine();
      plot("Accel bias", "m/s^2",
           {{"x", &ba_[0], IM_COL32(230, 70, 70, 255)},
            {"y", &ba_[1], IM_COL32(70, 230, 70, 255)},
            {"z", &ba_[2], IM_COL32(80, 120, 255, 255)}});
      ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
  }
  plot_tab_request_ = -1;
  ImGui::End();
}

}  // namespace vio

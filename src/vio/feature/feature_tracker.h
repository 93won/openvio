// SPDX-License-Identifier: MIT
/**
 * @file      feature_tracker.h
 * @brief     Tracks features from image to image and detects new ones.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 *
 * Per image: pyramidal optical flow with a forward-backward check, removal of the features that left the trackable
 * region, a robust rotation check on the rays (wrong tracks do not follow the camera rotation), removal of feature
 * clusters on light sources, a cap of features per grid cell (longest tracks first), and detection of new corners
 * away from the kept features.
 */

#pragma once

#include "vio/scene/camera.h"
#include "vio/scene/frame.h"

#include <opencv2/core.hpp>

#include <map>
#include <memory>
#include <vector>

namespace vio {

struct FeatureTrackerOptions {
  int max_features = 200;                   // Feature target (after initialization)
  float quality_level = 0.01f;              // Corner quality relative to the best corner of the image
  float min_distance = 30.0f;               // Minimum distance between features [px]

  int grid_cols = 20;                       // Feature grid
  int grid_rows = 10;
  int max_features_per_cell = 10;           // Longest tracks kept per cell
  int min_features_per_cell = 0;            // > 0: the cap per cell shares the feature target over the occupied cells,
                                            // between this and max_features_per_cell (0: fixed cap)

  int window_size = 21;                     // Optical flow window [px]
  int max_pyramid_level = 3;                // Optical flow pyramid levels above the image
  int max_iterations = 30;                  // Optical flow iterations per level
  double epsilon = 0.01;                    // Optical flow convergence [px]
  double max_forward_backward_error = 0.5;  // Track back to the previous image and drop points that do not return
                                            // within this distance [px] (0: no check)
  bool gain_compensation = false;           // Scale the previous image by the brightness ratio of the two images
                                            // before the optical flow (auto exposure changes the gain between images)

  int rotation_iterations = 1000;           // Robust rotation check: samples of three rays
  float rotation_threshold_deg = 2.0f;      // Robust rotation check: inlier angle [deg]

  bool remove_light_clusters = true;        // Remove clustered features (a cell whose feature positions spread less
                                            // than cluster_std_ratio of the cell diagonal)
  float cluster_std_ratio = 0.25f;
  int cluster_min_intensity = 250;          // > 0: only clusters whose features mostly see a pixel this bright in their
                                            // window (light sources) are removed

  double stereo_max_epipolar_error = 2.0;   // Stereo: a camera-1 match may leave the epipolar plane of its camera-0
                                            // ray by this much [px at the camera-1 image scale]

  bool Check() const;
};

// Why a feature of the previous image is missing from the current one
enum class TrackLoss {
  kNotFound = 0,      // Optical flow failed (status, forward-backward check)
  kMasked = 1,        // Moved into a polar band or a mask rectangle
  kOutlier = 2,       // Rejected by the robust rotation check
  kGridCap = 3,       // Trimmed by the per-cell cap
  kLightCluster = 4,  // Removed with a cluster on a light source
  kBorder = 5,        // Moved into the border margin (the left/right edge is the back-lens axis of the image)
};
constexpr int kNumTrackLosses = 6;

// Feature found outside the tracker under an existing id (a landmark re-found at its predicted pixel)
struct RecoveredFeature {
  int id;             // Track id
  cv::Point2f pixel;  // Position [px]
  int track_count;    // Track length (grid priority)
  int replaces;       // Tracked feature on the same point that it continues (-1: none)
};

class FeatureTracker {
 public:
  FeatureTracker(const FeatureTrackerOptions& options, const Camera& camera);

  /**
   * @brief Track the features of the previous frame into the frame and detect new ones
   * @param frame Current frame (its image)
   * @param previous Previous frame (null for the first image)
   */
  void TrackFeatures(Frame* frame, const Frame* previous);

  /**
   * @brief Add re-found features; the features they replace and the features detected in this image within the
   *        minimum distance of one are dropped
   * @param frame Current frame (after TrackFeatures)
   * @param recovered Re-found features
   */
  void AddRecoveredFeatures(const std::vector<RecoveredFeature>& recovered, Frame* frame) const;

  void SetMaxFeatures(int max_features) { max_features_ = max_features; }

  // Stereo rig: the second camera and its fixed pose relative to camera 0 (p_C1 = R_C0toC1 p_C0 + p_C0inC1)
  void EnableStereo(const Camera& camera1, const Eigen::Matrix3d& R_C0toC1, const Eigen::Vector3d& p_C0inC1);

  /**
   * @brief Camera-1 pixel of every feature of the frame: optical flow from the camera-0 image into the camera-1 image,
   *        started at the pixel of the ray at infinity, kept when the flow returns to the camera-0 pixel, the match
   *        lies on the epipolar plane of the baseline and the two rays meet in front of both cameras
   * @param frame Current frame (after TrackFeatures, whose pyramid it uses)
   * @param image1 Camera-1 image taken with the frame's image
   * @return Observations with camera = 1
   */
  std::vector<PixelObservation> MatchStereo(const Frame& frame, const cv::Mat& image1) const;

  // Features of the previous frame lost by the last TrackFeatures(), by id
  const std::map<int, TrackLoss>& track_losses() const { return track_losses_; }

 private:
  std::vector<cv::Point2f> DetectFeatures(const cv::Mat& image, const cv::Mat& mask) const;
  cv::Mat CreateDetectionMask(const std::vector<std::shared_ptr<Feature>>& features) const;
  // Optical flow with the image pyramids (cv::buildOpticalFlowPyramid) of both images
  void TrackOpticalFlow(const cv::Mat& prev_image, const std::vector<cv::Mat>& prev_pyramid, const cv::Mat& curr_image,
                        const std::vector<cv::Mat>& curr_pyramid, const std::vector<cv::Point2f>& prev_points,
                        std::vector<cv::Point2f>& curr_points, std::vector<uchar>& status) const;
  // Pyramid of an image for the optical flow (levels and borders as cv::calcOpticalFlowPyrLK builds them internally;
  // the optical flow computes the image derivatives itself, which is faster and smaller than storing them)
  void BuildPyramid(const cv::Mat& image, std::vector<cv::Mat>& pyramid) const;
  void RejectRotationOutliers(const std::vector<cv::Point2f>& prev_points,
                              const std::vector<cv::Point2f>& curr_points, std::vector<uchar>& inlier_mask) const;
  void RemoveLightClusters(Frame* frame) const;
  void LimitFeaturesPerCell(Frame* frame) const;
  void RecordLost(const std::vector<std::shared_ptr<Feature>>& previous, const Frame& frame, TrackLoss loss);

  FeatureTrackerOptions options_;
  Camera camera_;
  int max_features_;                        // Current feature target

  cv::Size lk_window_size_;                 // Optical flow
  cv::TermCriteria lk_criteria_;
  double lk_min_eig_threshold_;

  cv::Mat region_mask_;                     // Detection: polar bands and mask rectangles excluded
  cv::Mat border_mask_;                     // Detection: left/right border excluded
  cv::Mat detection_mask_;                  // Detection: region and border masks combined
  cv::Rect detection_roi_;                  // Detection: image part searched for corners (mask bounding box + margin)

  // Rotation check: a ray pair is an inlier when acos(cos) < threshold, i.e. when the clamped cosine is at least this
  // float (the boundary of the float acos, found once; NaN: compare the angles)
  float rotation_min_cos_;

  std::vector<cv::Mat> prev_pyramid_;       // Optical flow pyramid of the last tracked image
  int prev_pyramid_index_ = -1;             // Its image index

  bool stereo_ = false;                     // Stereo rig (EnableStereo)
  Camera camera1_;
  Eigen::Matrix3f R_C0toC1_ = Eigen::Matrix3f::Identity();
  Eigen::Vector3f p_C0inC1_ = Eigen::Vector3f::Zero();

  int next_feature_id_;
  std::map<int, TrackLoss> track_losses_;
};

}  // namespace vio

// SPDX-License-Identifier: MIT
/**
 * @file      feature_tracker.cc
 * @brief     Implements the feature tracker.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "vio/feature/feature_tracker.h"

#include "vio/util/logging.h"

#include <Eigen/Dense>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <set>

namespace vio {
namespace {

// Rotation R with b2 ~ R b1 from three ray correspondences (SVD of the cross-covariance)
Eigen::Matrix3f EstimateRotation(const std::array<const Eigen::Vector3f*, 3>& rays1,
                                 const std::array<const Eigen::Vector3f*, 3>& rays2) {
  Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
  for (size_t i = 0; i < 3; ++i) {
    H += *rays2[i] * rays1[i]->transpose();
  }
  Eigen::JacobiSVD<Eigen::Matrix3f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3f U = svd.matrixU();
  Eigen::Matrix3f V = svd.matrixV();
  Eigen::Matrix3f R = U * V.transpose();
  // Proper rotation (det(R) = 1)
  if (R.determinant() < 0.0f) {
    V.col(2) *= -1.0f;
    R = U * V.transpose();
  }
  return R;
}

// Cosine of the angle between two unit rays, clamped to [-1, 1]
float ClampedCosine(const Eigen::Vector3f& ray1, const Eigen::Vector3f& ray2) {
  const float cos_angle = ray1.dot(ray2);
  return std::max(-1.0f, std::min(1.0f, cos_angle));
}

float FloatFromBits(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

uint32_t BitsFromFloat(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

// Smallest float cosine c with std::acos(c) < threshold, so that the angle test of a clamped cosine is the comparison
// c >= result. Found by bisection over the floats of [0, 1] (their bit patterns increase with their values) and checked
// over 1000 floats on both sides of the boundary, beyond which the float acos is far from the threshold. NaN when the
// threshold is not in (0, 90 deg) or the check fails: the angles are compared then.
float MinInlierCosine(float threshold) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  if (!(threshold > 0.0f) || !(threshold < std::acos(0.0f))) {
    return nan;
  }
  uint32_t lo = BitsFromFloat(0.0f);  // acos >= threshold
  uint32_t hi = BitsFromFloat(1.0f);  // acos < threshold
  while (hi - lo > 1) {
    const uint32_t mid = lo + (hi - lo) / 2;
    if (std::acos(FloatFromBits(mid)) < threshold) {
      hi = mid;
    } else {
      lo = mid;
    }
  }
  for (uint32_t k = 0; k < 1000; ++k) {
    if (hi + k <= BitsFromFloat(1.0f) && !(std::acos(FloatFromBits(hi + k)) < threshold)) {
      return nan;
    }
    if (lo >= k && std::acos(FloatFromBits(lo - k)) < threshold) {
      return nan;
    }
  }
  return FloatFromBits(hi);
}

// cv::goodFeaturesToTrack of OpenCV 4.6 with the minimum-eigenvalue score, block 3 and aperture 3, written out in one
// pass: the gradients come from cv::Sobel as there, and the 3 x 3 sums of the gradient products are added directly in
// double, where OpenCV keeps running column sums in double (the same float except in rare low-score pixels). The
// quality threshold, the 3 x 3 maximum test, the sort order and the minimum-distance grid are those of OpenCV.
std::vector<cv::Point2f> DetectCorners(const cv::Mat& image, const cv::Mat& mask, int max_corners,
                                       double quality_level, double min_distance) {
  std::vector<cv::Point2f> corners;
  const int w = image.cols;
  const int h = image.rows;
  if (w == 0 || h == 0) {
    return corners;
  }

  // Gradients scaled as in cv::cornerMinEigenVal: 1 / (2^(aperture - 1) * block * 255)
  const double scale = 1.0 / (4.0 * 3.0 * 255.0);
  cv::Mat dx, dy;
  cv::Sobel(image, dx, CV_32F, 1, 0, 3, scale, 0, cv::BORDER_DEFAULT);
  cv::Sobel(image, dy, CV_32F, 0, 1, 3, scale, 0, cv::BORDER_DEFAULT);

  // Sums over 3 columns of the products dx dx, dx dy, dy dy (mirrored at the edges as BORDER_REFLECT_101), kept for
  // the last 3 rows
  std::vector<float> products(3 * static_cast<size_t>(w));
  std::vector<double> row_sums(9 * static_cast<size_t>(w));
  const int left = cv::borderInterpolate(-1, w, cv::BORDER_REFLECT_101);
  const int right = cv::borderInterpolate(w, w, cv::BORDER_REFLECT_101);
  auto compute_row_sums = [&](int y) {
    const float* gx = dx.ptr<float>(y);
    const float* gy = dy.ptr<float>(y);
    float* pxx = products.data();
    float* pxy = pxx + w;
    float* pyy = pxy + w;
    for (int x = 0; x < w; ++x) {
      pxx[x] = gx[x] * gx[x];
      pxy[x] = gx[x] * gy[x];
      pyy[x] = gy[x] * gy[x];
    }
    double* sums = row_sums.data() + static_cast<size_t>(y % 3) * 3 * w;
    for (int c = 0; c < 3; ++c) {
      const float* p = products.data() + static_cast<size_t>(c) * w;
      double* s = sums + static_cast<size_t>(c) * w;
      for (int x = 1; x < w - 1; ++x) {
        s[x] = static_cast<double>(p[x - 1]) + static_cast<double>(p[x]) + static_cast<double>(p[x + 1]);
      }
      s[0] = static_cast<double>(p[left]) + static_cast<double>(p[0]) + static_cast<double>(p[std::min(1, w - 1)]);
      if (w > 1) {
        s[w - 1] = static_cast<double>(p[w - 2]) + static_cast<double>(p[w - 1]) + static_cast<double>(p[right]);
      }
    }
  };

  // Minimum eigenvalue of the summed products (the float expression of cv::cornerMinEigenVal)
  cv::Mat eig(h, w, CV_32F);
  compute_row_sums(0);
  if (h > 1) {
    compute_row_sums(1);
  }
  for (int y = 0; y < h; ++y) {
    if (y >= 1 && y + 1 < h) {
      compute_row_sums(y + 1);
    }
    const int y0 = cv::borderInterpolate(y - 1, h, cv::BORDER_REFLECT_101);
    const int y2 = cv::borderInterpolate(y + 1, h, cv::BORDER_REFLECT_101);
    const double* s0 = row_sums.data() + static_cast<size_t>(y0 % 3) * 3 * w;
    const double* s1 = row_sums.data() + static_cast<size_t>(y % 3) * 3 * w;
    const double* s2 = row_sums.data() + static_cast<size_t>(y2 % 3) * 3 * w;
    float* e = eig.ptr<float>(y);
    for (int x = 0; x < w; ++x) {
      const float a = static_cast<float>(s0[x] + s1[x] + s2[x]) * 0.5f;
      const float b = static_cast<float>(s0[w + x] + s1[w + x] + s2[w + x]);
      const float c = static_cast<float>(s0[2 * w + x] + s1[2 * w + x] + s2[2 * w + x]) * 0.5f;
      e[x] = (a + c) - std::sqrt((a - c) * (a - c) + b * b);
    }
  }

  // Scores below the quality threshold, relative to the best score of the mask, become 0 (cv::threshold TOZERO in
  // float)
  double max_value = 0.0;
  cv::minMaxLoc(eig, nullptr, &max_value, nullptr, nullptr, mask);
  const float threshold = static_cast<float>(max_value * quality_level);
  for (int y = 0; y < h; ++y) {
    float* e = eig.ptr<float>(y);
    for (int x = 0; x < w; ++x) {
      e[x] = e[x] > threshold ? e[x] : 0.0f;
    }
  }

  // Candidates: mask pixels with a nonzero score that is the maximum over its 3 x 3 neighbourhood (cv::dilate), from
  // the maxima over 3 columns of the rows above, at and below
  struct Candidate {
    float value;
    int y;
    int x;
  };
  std::vector<Candidate> candidates;
  std::vector<float> row_max(3 * static_cast<size_t>(w));
  auto compute_row_max = [&](int y) {
    const float* e = eig.ptr<float>(y);
    float* r = row_max.data() + static_cast<size_t>(y % 3) * w;
    for (int x = 1; x < w - 1; ++x) {
      r[x] = std::max(std::max(e[x - 1], e[x]), e[x + 1]);
    }
  };
  if (h >= 3) {
    compute_row_max(0);
    compute_row_max(1);
  }
  for (int y = 1; y < h - 1; ++y) {
    compute_row_max(y + 1);
    const float* r0 = row_max.data() + static_cast<size_t>((y - 1) % 3) * w;
    const float* r1 = row_max.data() + static_cast<size_t>(y % 3) * w;
    const float* r2 = row_max.data() + static_cast<size_t>((y + 1) % 3) * w;
    const float* e = eig.ptr<float>(y);
    const uchar* m = mask.ptr<uchar>(y);
    for (int x = 1; x < w - 1; ++x) {
      if (m[x] != 0 && e[x] != 0.0f && e[x] == std::max(std::max(r0[x], r1[x]), r2[x])) {
        candidates.push_back({e[x], y, x});
      }
    }
  }

  // Candidates in the order of OpenCV's sort, by score and then by position, both descending (its pointer order),
  // taken from a heap: the selection usually stops long before the last candidate
  auto comes_after = [](const Candidate& p, const Candidate& q) {
    if (p.value != q.value) {
      return p.value < q.value;
    }
    return p.y != q.y ? p.y < q.y : p.x < q.x;
  };
  std::make_heap(candidates.begin(), candidates.end(), comes_after);
  auto next_candidate = [&, end = candidates.end()]() mutable {
    std::pop_heap(candidates.begin(), end, comes_after);
    --end;
    return *end;
  };

  // Greedy selection with the minimum distance, over a grid of cells of that size
  if (min_distance < 1.0) {
    for (size_t i = 0; i < candidates.size(); ++i) {
      const Candidate candidate = next_candidate();
      corners.emplace_back(static_cast<float>(candidate.x), static_cast<float>(candidate.y));
      if (max_corners > 0 && static_cast<int>(corners.size()) == max_corners) {
        break;
      }
    }
    return corners;
  }
  const int cell_size = cvRound(min_distance);
  const int grid_width = (w + cell_size - 1) / cell_size;
  const int grid_height = (h + cell_size - 1) / cell_size;
  std::vector<std::vector<cv::Point2f>> grid(static_cast<size_t>(grid_width) * grid_height);
  const double min_distance2 = min_distance * min_distance;
  for (size_t i = 0; i < candidates.size(); ++i) {
    const Candidate candidate = next_candidate();
    const int x = candidate.x;
    const int y = candidate.y;
    const int x_cell = x / cell_size;
    const int y_cell = y / cell_size;
    bool good = true;
    for (int yy = std::max(0, y_cell - 1); yy <= std::min(grid_height - 1, y_cell + 1) && good; ++yy) {
      for (int xx = std::max(0, x_cell - 1); xx <= std::min(grid_width - 1, x_cell + 1) && good; ++xx) {
        for (const cv::Point2f& kept : grid[static_cast<size_t>(yy) * grid_width + xx]) {
          const float ddx = x - kept.x;
          const float ddy = y - kept.y;
          if (ddx * ddx + ddy * ddy < min_distance2) {
            good = false;
            break;
          }
        }
      }
    }
    if (good) {
      grid[static_cast<size_t>(y_cell) * grid_width + x_cell].emplace_back(static_cast<float>(x),
                                                                            static_cast<float>(y));
      corners.emplace_back(static_cast<float>(x), static_cast<float>(y));
      if (max_corners > 0 && static_cast<int>(corners.size()) == max_corners) {
        break;
      }
    }
  }
  return corners;
}

// Median intensity of an 8-bit image over the nonzero pixels of a mask, interpolated within its histogram bin (-1
// without such pixels)
double MaskedMedian(const cv::Mat& image, const cv::Mat& mask) {
  std::array<int, 256> histogram{};
  int count = 0;
  for (int y = 0; y < image.rows; ++y) {
    const uchar* value = image.ptr<uchar>(y);
    const uchar* valid = mask.ptr<uchar>(y);
    for (int x = 0; x < image.cols; ++x) {
      if (valid[x] != 0) {
        ++histogram[value[x]];
        ++count;
      }
    }
  }
  if (count == 0) {
    return -1.0;
  }
  const double half = 0.5 * count;
  int below = 0;
  for (int v = 0; v < 256; ++v) {
    if (histogram[v] > 0 && below + histogram[v] >= half) {
      return v - 0.5 + (half - below) / histogram[v];
    }
    below += histogram[v];
  }
  return 255.0;
}

}  // namespace

bool FeatureTrackerOptions::Check() const {
  REQUIRE_OPTION(max_features > 0);
  REQUIRE_OPTION(quality_level > 0.0f && quality_level < 1.0f);
  REQUIRE_OPTION(min_distance >= 0.0f);
  REQUIRE_OPTION(grid_cols > 0 && grid_rows > 0);
  REQUIRE_OPTION(max_features_per_cell > 0);
  REQUIRE_OPTION(min_features_per_cell >= 0);
  REQUIRE_OPTION(window_size > 0 && window_size % 2 == 1);
  REQUIRE_OPTION(max_pyramid_level >= 0);
  REQUIRE_OPTION(max_iterations > 0);
  REQUIRE_OPTION(epsilon > 0.0);
  REQUIRE_OPTION(max_forward_backward_error >= 0.0);
  REQUIRE_OPTION(rotation_iterations > 0);
  REQUIRE_OPTION(rotation_threshold_deg > 0.0f);
  REQUIRE_OPTION(cluster_std_ratio >= 0.0f);
  REQUIRE_OPTION(stereo_max_epipolar_error > 0.0);
  return true;
}

FeatureTracker::FeatureTracker(const FeatureTrackerOptions& options, const Camera& camera)
    : options_(options),
      camera_(camera),
      max_features_(options.max_features),
      lk_window_size_(options.window_size, options.window_size),
      lk_criteria_(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, options.max_iterations, options.epsilon),
      lk_min_eig_threshold_(1e-4),  // OpenCV default
      next_feature_id_(0) {
  // Detection mask: polar bands, then the mask rectangles
  const int width = camera_.width;
  const int height = camera_.height;
  const float ratio = camera_.polar_exclusion_ratio;
  region_mask_ = cv::Mat::ones(height, width, CV_8U) * 255;
  const int top_rows = static_cast<int>(height * ratio);
  const int bottom_start = static_cast<int>(height * (1.0f - ratio));
  if (top_rows > 0) {
    region_mask_(cv::Rect(0, 0, width, top_rows)) = 0;
  }
  if (bottom_start < height) {
    region_mask_(cv::Rect(0, bottom_start, width, height - bottom_start)) = 0;
  }
  const float width_f = static_cast<float>(width);
  const float height_f = static_cast<float>(height);
  for (const cv::Vec4f& r : camera_.mask_rectangles) {
    const cv::Rect rect(cv::Point(cvRound(r[0] * width_f), cvRound(r[2] * height_f)),
                        cv::Point(cvRound(r[1] * width_f), cvRound(r[3] * height_f)));
    region_mask_(rect & cv::Rect(0, 0, region_mask_.cols, region_mask_.rows)) = 0;
  }

  // Detection mask: the border margin at every image edge, where tracks end (Camera::IsNearBorder)
  const int margin = camera_.border_margin;
  border_mask_ = cv::Mat::ones(height, width, CV_8UC1) * 255;
  if (margin > 0) {
    border_mask_(cv::Rect(0, 0, margin, height)) = 0;
    border_mask_(cv::Rect(width - margin, 0, margin, height)) = 0;
    border_mask_(cv::Rect(0, 0, width, std::min(margin, height))) = 0;
    border_mask_(cv::Rect(0, std::max(0, height - margin), width, std::min(margin, height))) = 0;
  }
  cv::bitwise_and(region_mask_, border_mask_, detection_mask_);

  // Corners are searched only around the rows and columns the mask leaves: its bounding box with a margin of three
  // pixels, so that the corner score of every pixel the detection uses sees the same neighbours as in the full image
  // (Sobel aperture 3 and block 3, then a 3 x 3 maximum)
  constexpr int kMargin = 3;
  detection_roi_ = cv::boundingRect(detection_mask_);
  detection_roi_ = cv::Rect(detection_roi_.x - kMargin, detection_roi_.y - kMargin, detection_roi_.width + 2 * kMargin,
                            detection_roi_.height + 2 * kMargin) &
                   cv::Rect(0, 0, width, height);

  const float threshold_rad = options_.rotation_threshold_deg * M_PI / 180.0f;
  rotation_min_cos_ = MinInlierCosine(threshold_rad);
}

void FeatureTracker::BuildPyramid(const cv::Mat& image, std::vector<cv::Mat>& pyramid) const {
  cv::buildOpticalFlowPyramid(image, pyramid, lk_window_size_, options_.max_pyramid_level, false);
}

void FeatureTracker::TrackFeatures(Frame* frame, const Frame* previous) {
  frame->SetGrid(options_.grid_cols, options_.grid_rows, options_.max_features_per_cell);
  const cv::Mat& current_image = frame->image();
  track_losses_.clear();

  // The pyramid of every image is built once: it serves the optical flow from the previous image, the
  // forward-backward check and the optical flow into the next image
  std::vector<cv::Mat> curr_pyramid;
  BuildPyramid(current_image, curr_pyramid);

  // First image: detect only
  if (previous == nullptr || previous->features().empty()) {
    const std::vector<cv::Point2f> detected = DetectFeatures(current_image, cv::Mat());
    for (const auto& pt : detected) {
      auto feature = std::make_shared<Feature>(next_feature_id_++, pt);
      feature->ray = camera_.PixelToRay(pt);
      frame->AddFeature(feature);
    }
    frame->AssignFeaturesToGrid();
    frame->LimitFeaturesPerCell();
    prev_pyramid_ = std::move(curr_pyramid);
    prev_pyramid_index_ = frame->index();
    return;
  }

  // Optical flow from the previous image
  const auto& prev_features = previous->features();
  std::vector<cv::Point2f> prev_points;
  prev_points.reserve(prev_features.size());
  for (const auto& feature : prev_features) {
    prev_points.push_back(feature->pixel);
  }
  if (prev_pyramid_index_ != previous->index() || prev_pyramid_.empty()) {
    BuildPyramid(previous->image(), prev_pyramid_);
    prev_pyramid_index_ = previous->index();
  }
  std::vector<cv::Point2f> curr_points;
  std::vector<uchar> status;
  TrackOpticalFlow(previous->image(), prev_pyramid_, current_image, curr_pyramid, prev_points, curr_points, status);

  // Keep the tracks that stay in the trackable region
  std::vector<cv::Point2f> good_prev_points;
  std::vector<cv::Point2f> good_curr_points;
  std::vector<std::shared_ptr<Feature>> good_prev_features;
  for (size_t i = 0; i < status.size(); ++i) {
    if (status[i] && camera_.IsTrackable(curr_points[i])) {
      good_prev_points.push_back(prev_points[i]);
      good_curr_points.push_back(curr_points[i]);
      good_prev_features.push_back(prev_features[i]);
    } else if (!status[i]) {
      track_losses_[prev_features[i]->id] = TrackLoss::kNotFound;
    } else {
      track_losses_[prev_features[i]->id] = camera_.IsMasked(curr_points[i]) ? TrackLoss::kMasked : TrackLoss::kBorder;
    }
  }

  // Robust rotation check
  std::vector<uchar> inlier_mask;
  if (good_curr_points.size() >= 3) {
    RejectRotationOutliers(good_prev_points, good_curr_points, inlier_mask);
  } else {
    inlier_mask.resize(good_curr_points.size(), 1);
  }

  // Tracked features
  for (size_t i = 0; i < inlier_mask.size(); ++i) {
    if (inlier_mask[i]) {
      const auto& prev_feature = good_prev_features[i];
      auto feature = std::make_shared<Feature>(prev_feature->id, good_curr_points[i]);
      feature->ray = camera_.PixelToRay(good_curr_points[i]);
      feature->track_count = prev_feature->track_count + 1;
      frame->AddFeature(feature);
    } else {
      track_losses_[good_prev_features[i]->id] = TrackLoss::kOutlier;
    }
  }

  // Clusters on light sources, then the per-cell cap
  RemoveLightClusters(frame);
  RecordLost(prev_features, *frame, TrackLoss::kLightCluster);
  LimitFeaturesPerCell(frame);

  // New features in the empty regions
  if (frame->features().size() < static_cast<size_t>(max_features_)) {
    const cv::Mat mask = CreateDetectionMask(frame->features());
    const std::vector<cv::Point2f> new_points = DetectFeatures(current_image, mask);
    for (const auto& pt : new_points) {
      auto feature = std::make_shared<Feature>(next_feature_id_++, pt);
      feature->ray = camera_.PixelToRay(pt);
      frame->AddFeature(feature);
    }
    LimitFeaturesPerCell(frame);
  }

  // Tracked features that the per-cell cap removed
  RecordLost(prev_features, *frame, TrackLoss::kGridCap);

  // The pyramid of this image serves the optical flow into the next one
  prev_pyramid_ = std::move(curr_pyramid);
  prev_pyramid_index_ = frame->index();
}

void FeatureTracker::RecordLost(const std::vector<std::shared_ptr<Feature>>& previous, const Frame& frame,
                                TrackLoss loss) {
  std::set<int> kept;
  for (const auto& feature : frame.features()) {
    kept.insert(feature->id);
  }
  for (const auto& feature : previous) {
    if (kept.count(feature->id) == 0 && track_losses_.count(feature->id) == 0) {
      track_losses_[feature->id] = loss;
    }
  }
}

std::vector<cv::Point2f> FeatureTracker::DetectFeatures(const cv::Mat& image, const cv::Mat& mask) const {
  cv::Mat combined_mask;  // a new buffer for the combination (detection_mask_ must not change)
  if (mask.empty()) {
    combined_mask = detection_mask_;
  } else {
    cv::bitwise_and(detection_mask_, mask, combined_mask);
  }
  std::vector<cv::Point2f> corners;
  if (detection_roi_.area() == 0) {
    return corners;
  }
  corners = DetectCorners(image(detection_roi_), combined_mask(detection_roi_), max_features_,
                          options_.quality_level, options_.min_distance);
  for (auto& corner : corners) {
    corner.x += static_cast<float>(detection_roi_.x);
    corner.y += static_cast<float>(detection_roi_.y);
  }
  return corners;
}

void FeatureTracker::TrackOpticalFlow(const cv::Mat& prev_image, const std::vector<cv::Mat>& prev_pyramid,
                                      const cv::Mat& curr_image, const std::vector<cv::Mat>& curr_pyramid,
                                      const std::vector<cv::Point2f>& prev_points,
                                      std::vector<cv::Point2f>& curr_points, std::vector<uchar>& status) const {
  if (prev_points.empty()) {
    return;
  }

  // Auto exposure scales the whole image from one image to the next, and the optical flow assumes constant
  // brightness: the previous image is scaled by the ratio of the median intensities of the two images over the region
  // pixels that neither image clips (a clipped pixel does not scale)
  const std::vector<cv::Mat>* prev = &prev_pyramid;
  std::vector<cv::Mat> scaled_pyramid;
  if (options_.gain_compensation && prev_image.type() == CV_8UC1 && curr_image.type() == CV_8UC1) {
    cv::Mat unclipped_prev;
    cv::Mat unclipped_curr;
    cv::inRange(prev_image, 1, 254, unclipped_prev);
    cv::inRange(curr_image, 1, 254, unclipped_curr);
    const cv::Mat valid = region_mask_ & unclipped_prev & unclipped_curr;
    const double median_prev = MaskedMedian(prev_image, valid);
    const double median_curr = MaskedMedian(curr_image, valid);
    if (median_prev > 0.0 && median_curr > 0.0) {
      cv::Mat scaled;
      prev_image.convertTo(scaled, CV_8U, median_curr / median_prev);
      BuildPyramid(scaled, scaled_pyramid);
      prev = &scaled_pyramid;
    }
  }

  std::vector<float> error;
  cv::calcOpticalFlowPyrLK(*prev, curr_pyramid, prev_points, curr_points, status, error, lk_window_size_,
                           options_.max_pyramid_level, lk_criteria_, 0, lk_min_eig_threshold_);

  // Forward-backward check: track back to the previous image and drop points that do not return
  if (options_.max_forward_backward_error > 0.0) {
    std::vector<cv::Point2f> back_points = prev_points;
    std::vector<uchar> back_status;
    cv::calcOpticalFlowPyrLK(curr_pyramid, *prev, curr_points, back_points, back_status, error, lk_window_size_,
                             options_.max_pyramid_level, lk_criteria_, cv::OPTFLOW_USE_INITIAL_FLOW,
                             lk_min_eig_threshold_);
    for (size_t i = 0; i < status.size(); ++i) {
      if (status[i] &&
          (!back_status[i] || cv::norm(back_points[i] - prev_points[i]) > options_.max_forward_backward_error)) {
        status[i] = 0;
      }
    }
  }
}

void FeatureTracker::RejectRotationOutliers(const std::vector<cv::Point2f>& prev_points,
                                            const std::vector<cv::Point2f>& curr_points,
                                            std::vector<uchar>& inlier_mask) const {
  if (prev_points.size() < 3) {
    inlier_mask.resize(prev_points.size(), 1);
    return;
  }
  const size_t num = prev_points.size();
  std::vector<Eigen::Vector3f> prev_rays(num);
  std::vector<Eigen::Vector3f> curr_rays(num);
  for (size_t i = 0; i < num; ++i) {
    prev_rays[i] = camera_.PixelToRay(prev_points[i]);
    curr_rays[i] = camera_.PixelToRay(curr_points[i]);
  }

  // A pair is an inlier of rotation R when the angle between R b1 and b2 is below the threshold: compared as the
  // clamped cosine against the boundary of the float acos (same decision without an acos per pair)
  const float threshold_rad = options_.rotation_threshold_deg * M_PI / 180.0f;
  const bool by_cosine = !std::isnan(rotation_min_cos_);
  auto is_inlier = [&](const Eigen::Matrix3f& R, size_t i) {
    const Eigen::Vector3f rotated = R * prev_rays[i];
    const float cos_angle = ClampedCosine(rotated, curr_rays[i]);
    return by_cosine ? cos_angle >= rotation_min_cos_ : std::acos(cos_angle) < threshold_rad;
  };

  // Random samples of three rays; a fixed sample stream makes a dataset evaluation reproducible. The model with the
  // most inliers wins (the first of equal ones).
  std::mt19937 gen(0x360u);
  std::uniform_int_distribution<> dis(0, static_cast<int>(num) - 1);
  int best_inliers = 0;
  Eigen::Matrix3f best_R = Eigen::Matrix3f::Identity();
  for (int iter = 0; iter < options_.rotation_iterations; ++iter) {
    std::array<int, 3> indices;
    int count = 0;
    while (count < 3) {
      const int idx = dis(gen);
      if (std::find(indices.begin(), indices.begin() + count, idx) == indices.begin() + count) {
        indices[count++] = idx;
      }
    }
    const Eigen::Matrix3f R =
        EstimateRotation({&prev_rays[indices[0]], &prev_rays[indices[1]], &prev_rays[indices[2]]},
                         {&curr_rays[indices[0]], &curr_rays[indices[1]], &curr_rays[indices[2]]});
    if (std::abs(R.determinant() - 1.0f) > 0.1f) {
      continue;
    }
    int num_inliers = 0;
    for (size_t i = 0; i < num; ++i) {
      num_inliers += is_inlier(R, i) ? 1 : 0;
    }
    if (num_inliers > best_inliers) {
      best_inliers = num_inliers;
      best_R = R;
    }
  }
  inlier_mask.assign(num, 1);
  if (best_inliers > 0) {
    for (size_t i = 0; i < num; ++i) {
      inlier_mask[i] = is_inlier(best_R, i) ? 1 : 0;
    }
  }
}

void FeatureTracker::AddRecoveredFeatures(const std::vector<RecoveredFeature>& recovered, Frame* frame) const {
  if (recovered.empty()) {
    return;
  }
  // Replaced tracks end here; detections of this image near a re-found feature give way to it
  const float min_distance2 = options_.min_distance * options_.min_distance;
  std::vector<std::shared_ptr<Feature>> kept;
  for (const auto& feature : frame->features()) {
    bool drop = false;
    for (const auto& r : recovered) {
      const cv::Point2f d = feature->pixel - r.pixel;
      drop = drop || r.replaces == feature->id || (feature->track_count == 0 && d.dot(d) < min_distance2);
    }
    if (!drop) {
      kept.push_back(feature);
    }
  }
  frame->ClearFeatures();
  for (const auto& feature : kept) {
    frame->AddFeature(feature);
  }
  for (const auto& r : recovered) {
    auto feature = std::make_shared<Feature>(r.id, r.pixel);
    feature->ray = camera_.PixelToRay(r.pixel);
    feature->track_count = r.track_count;
    frame->AddFeature(feature);
  }
  frame->AssignFeaturesToGrid();
}

cv::Mat FeatureTracker::CreateDetectionMask(const std::vector<std::shared_ptr<Feature>>& features) const {
  // No new corner within min_distance of a kept feature
  cv::Mat mask = region_mask_.clone();
  for (const auto& feature : features) {
    cv::circle(mask, feature->pixel, static_cast<int>(options_.min_distance), 0, -1);
  }
  return mask;
}

void FeatureTracker::RemoveLightClusters(Frame* frame) const {
  if (!options_.remove_light_clusters) {
    return;
  }
  const auto& features = frame->features();
  if (features.size() < 4) {
    return;
  }

  const cv::Mat& image = frame->image();
  const int img_width = image.cols;
  const int img_height = image.rows;
  const int grid_cols = options_.grid_cols;
  const int grid_rows = options_.grid_rows;
  const float cell_width = static_cast<float>(img_width) / grid_cols;
  const float cell_height = static_cast<float>(img_height) / grid_rows;
  const float grid_diagonal = std::sqrt(cell_width * cell_width + cell_height * cell_height);
  const float std_threshold = grid_diagonal * options_.cluster_std_ratio;

  // Features per cell
  std::vector<std::vector<bool>> is_clustered(grid_rows, std::vector<bool>(grid_cols, false));
  std::vector<std::vector<std::vector<size_t>>> grid_feature_indices(grid_rows,
                                                                     std::vector<std::vector<size_t>>(grid_cols));
  for (size_t idx = 0; idx < features.size(); ++idx) {
    const cv::Point2f pt = features[idx]->pixel;
    const int col = std::min(static_cast<int>(pt.x / cell_width), grid_cols - 1);
    const int row = std::min(static_cast<int>(pt.y / cell_height), grid_rows - 1);
    grid_feature_indices[row][col].push_back(idx);
  }

  // A cell of at least four features is clustered when their positions spread less than the threshold
  for (int row = 0; row < grid_rows; ++row) {
    for (int col = 0; col < grid_cols; ++col) {
      const auto& cell_indices = grid_feature_indices[row][col];
      if (cell_indices.size() >= 4) {
        std::vector<cv::Point2f> cell_points;
        for (size_t idx : cell_indices) {
          cell_points.push_back(features[idx]->pixel);
        }
        cv::Point2f mean(0.0f, 0.0f);
        for (const auto& pt : cell_points) {
          mean += pt;
        }
        mean.x /= cell_points.size();
        mean.y /= cell_points.size();
        float variance = 0.0f;
        for (const auto& pt : cell_points) {
          const float dx = pt.x - mean.x;
          const float dy = pt.y - mean.y;
          variance += (dx * dx + dy * dy);
        }
        variance /= cell_points.size();
        const float std_dev = std::sqrt(variance);
        if (std_dev < std_threshold) {
          is_clustered[row][col] = true;
        }
      }
    }
  }

  // A feature sees a light source when its optical flow window holds a pixel at or above cluster_min_intensity
  const int half = options_.window_size / 2;
  const cv::Rect bounds(0, 0, img_width, img_height);
  auto sees_light = [&](size_t idx) {
    const cv::Point2f pt = features[idx]->pixel;
    const cv::Rect window =
        cv::Rect(cvRound(pt.x) - half, cvRound(pt.y) - half, 2 * half + 1, 2 * half + 1) & bounds;
    double max_value = 0.0;
    cv::minMaxLoc(image(window), nullptr, &max_value);
    return max_value >= options_.cluster_min_intensity;
  };

  // With cluster_min_intensity > 0 only clusters on light sources are removed: most features of the cell see a light
  // (the corners of a blooming light share its errors). Clusters on texture are kept.
  if (options_.cluster_min_intensity > 0) {
    for (int row = 0; row < grid_rows; ++row) {
      for (int col = 0; col < grid_cols; ++col) {
        if (!is_clustered[row][col]) {
          continue;
        }
        const auto& cell_indices = grid_feature_indices[row][col];
        const size_t lights =
            static_cast<size_t>(std::count_if(cell_indices.begin(), cell_indices.end(), sees_light));
        if (2 * lights < cell_indices.size()) {
          is_clustered[row][col] = false;
        }
      }
    }
  }

  // Remove the features of the clustered cells
  std::vector<bool> keep(features.size(), true);
  for (int row = 0; row < grid_rows; ++row) {
    for (int col = 0; col < grid_cols; ++col) {
      if (!is_clustered[row][col]) {
        continue;
      }
      for (size_t idx : grid_feature_indices[row][col]) {
        keep[idx] = false;
      }
    }
  }
  std::vector<std::shared_ptr<Feature>> filtered_features;
  for (size_t idx = 0; idx < features.size(); ++idx) {
    if (keep[idx]) {
      filtered_features.push_back(features[idx]);
    }
  }
  frame->ClearFeatures();
  for (const auto& feature : filtered_features) {
    frame->AddFeature(feature);
  }
}

void FeatureTracker::LimitFeaturesPerCell(Frame* frame) const {
  // The feature target shared over the occupied cells: dense where few cells see features (lights at night),
  // sparse where texture is everywhere
  const int min_cap = options_.min_features_per_cell;
  if (min_cap > 0) {
    const cv::Mat& image = frame->image();
    const float cell_width = static_cast<float>(image.cols) / options_.grid_cols;
    const float cell_height = static_cast<float>(image.rows) / options_.grid_rows;
    std::vector<char> occupied(static_cast<size_t>(options_.grid_cols * options_.grid_rows), 0);
    for (const auto& feature : frame->features()) {
      const cv::Point2f pt = feature->pixel;
      if (pt.x < 0.0f || pt.y < 0.0f || pt.x >= image.cols || pt.y >= image.rows) {
        continue;
      }
      const int col = std::min(static_cast<int>(pt.x / cell_width), options_.grid_cols - 1);
      const int row = std::min(static_cast<int>(pt.y / cell_height), options_.grid_rows - 1);
      occupied[static_cast<size_t>(row * options_.grid_cols + col)] = 1;
    }
    const int count = static_cast<int>(std::count(occupied.begin(), occupied.end(), 1));
    const int cap = (count > 0) ? std::clamp((max_features_ + count - 1) / count,
                                             std::min(min_cap, options_.max_features_per_cell),
                                             options_.max_features_per_cell)
                                : options_.max_features_per_cell;
    frame->SetGrid(options_.grid_cols, options_.grid_rows, cap);
  }
  frame->AssignFeaturesToGrid();
  frame->LimitFeaturesPerCell();
}

void FeatureTracker::EnableStereo(const Camera& camera1, const Eigen::Matrix3d& R_C0toC1,
                                  const Eigen::Vector3d& p_C0inC1) {
  stereo_ = true;
  camera1_ = camera1;
  R_C0toC1_ = R_C0toC1.cast<float>();
  p_C0inC1_ = p_C0inC1.cast<float>();
}

std::vector<PixelObservation> FeatureTracker::MatchStereo(const Frame& frame, const cv::Mat& image1) const {
  std::vector<PixelObservation> matches;
  const auto& features = frame.features();
  if (!stereo_ || features.empty() || prev_pyramid_index_ != frame.index()) {
    return matches;
  }

  // Start every match at the pixel of its ray at infinity: the rotation and the lens difference between the cameras
  // are then removed, and the optical flow only has to find the parallax of the baseline
  std::vector<cv::Point2f> points0;
  std::vector<cv::Point2f> points1;
  points0.reserve(features.size());
  points1.reserve(features.size());
  const cv::Rect2f image_rect(0.0f, 0.0f, static_cast<float>(camera1_.width), static_cast<float>(camera1_.height));
  for (const auto& feature : features) {
    points0.push_back(feature->pixel);
    const cv::Point2f guess = camera1_.RayToPixel(R_C0toC1_ * feature->ray);
    points1.push_back((std::isfinite(guess.x) && std::isfinite(guess.y) && image_rect.contains(guess)) ? guess
                                                                                                         : feature->pixel);
  }
  std::vector<cv::Mat> pyramid1;
  BuildPyramid(image1, pyramid1);
  std::vector<uchar> status;
  std::vector<float> error;
  cv::calcOpticalFlowPyrLK(prev_pyramid_, pyramid1, points0, points1, status, error, lk_window_size_,
                           options_.max_pyramid_level, lk_criteria_, cv::OPTFLOW_USE_INITIAL_FLOW,
                           lk_min_eig_threshold_);
  std::vector<cv::Point2f> back = points0;
  std::vector<uchar> back_status;
  cv::calcOpticalFlowPyrLK(pyramid1, prev_pyramid_, points1, back, back_status, error, lk_window_size_,
                           options_.max_pyramid_level, lk_criteria_, cv::OPTFLOW_USE_INITIAL_FLOW,
                           lk_min_eig_threshold_);

  // Epipolar plane in camera 1: through its center, the camera-0 center p_C0inC1 and the camera-0 ray turned into
  // camera 1; the match ray may leave it by the angle of stereo_max_epipolar_error pixels
  const float max_sin = static_cast<float>(
      std::sin(options_.stereo_max_epipolar_error / camera1_.PixelsPerRadian()));
  matches.reserve(features.size());
  for (size_t i = 0; i < features.size(); ++i) {
    if (!status[i] || !back_status[i] || !camera1_.IsTrackable(points1[i])) {
      continue;
    }
    if (options_.max_forward_backward_error > 0.0 &&
        cv::norm(back[i] - points0[i]) > options_.max_forward_backward_error) {
      continue;
    }
    const Eigen::Vector3f ray1 = camera1_.PixelToRay(points1[i]);
    const Eigen::Vector3f a = R_C0toC1_ * features[i]->ray;
    const Eigen::Vector3f normal = p_C0inC1_.cross(a);
    if (normal.norm() < 1e-6f || std::abs(normal.normalized().dot(ray1)) > max_sin) {
      continue;
    }
    // The two rays meet in front of both cameras: closest points p_C0inC1 + s a and u ray1 with s, u > 0 (nearly
    // parallel rays see a far point and pass)
    const float aa = a.dot(a);
    const float ab = a.dot(ray1);
    const float bb = ray1.dot(ray1);
    const float det = aa * bb - ab * ab;
    if (det > 1e-9f) {
      const float s = (ab * ray1.dot(p_C0inC1_) - bb * a.dot(p_C0inC1_)) / det;
      const float u = (aa * ray1.dot(p_C0inC1_) - ab * a.dot(p_C0inC1_)) / det;
      if (s <= 0.0f || u <= 0.0f) {
        continue;
      }
    }
    matches.push_back({features[i]->id, Eigen::Vector2d(points1[i].x, points1[i].y), 1});
  }
  return matches;
}

}  // namespace vio

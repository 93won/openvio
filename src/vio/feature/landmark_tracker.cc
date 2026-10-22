// SPDX-License-Identifier: MIT
/**
 * @file      landmark_tracker.cc
 * @brief     Implements the landmark tracker.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2026-10-04
 * @copyright Copyright (c) 2026 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "vio/feature/landmark_tracker.h"

#include "vio/util/logging.h"

#include <Eigen/Dense>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <set>

namespace vio {

bool LandmarkTrackerOptions::Check() const {
  REQUIRE_OPTION(search_radius >= 0.0);
  REQUIRE_OPTION(min_ncc > -1.0 && min_ncc <= 1.0);
  REQUIRE_OPTION(patch_half_size > 0);
  REQUIRE_OPTION(loss_reasons >= 0 && loss_reasons < (1 << kNumTrackLosses));
  return true;
}

LandmarkTracker::LandmarkTracker(const LandmarkTrackerOptions& options, const Camera& camera)
    : options_(options), camera_(camera) {}

cv::Mat LandmarkTracker::TangentPatch(const cv::Mat& image, const Eigen::Vector3d& c, const Eigen::Vector3d& e1,
                                      int half) const {
  const double f = camera_.PixelsPerRadian();
  const Eigen::Vector3d e2 = c.cross(e1);
  const int size = 2 * half + 1;
  cv::Mat map_x(size, size, CV_32F);
  cv::Mat map_y(size, size, CV_32F);
  for (int y = 0; y < size; ++y) {
    for (int x = 0; x < size; ++x) {
      const Eigen::Vector3d d = c + ((x - half) * e1 + (y - half) * e2) / f;
      const cv::Point2f uv = camera_.RayToPixel(d.cast<float>());
      map_x.at<float>(y, x) = uv.x;
      map_y.at<float>(y, x) = uv.y;
    }
  }
  // Across the left/right edge when the image continues there (e.g. 360 degrees of longitude), else zero outside
  cv::Mat patch;
  cv::remap(image, patch, map_x, map_y, cv::INTER_LINEAR,
            camera_.WrapsHorizontally() ? cv::BORDER_WRAP : cv::BORDER_CONSTANT);
  return patch;
}

const cv::Mat& LandmarkTracker::Patch(LandmarkPatch& patch) const {
  if (patch.patch.empty()) {
    patch.patch = TangentPatch(patch_image_, patch.c, patch.e1, options_.patch_half_size);
  }
  return patch.patch;
}

void LandmarkTracker::FindLandmarks(const std::vector<PixelObservation>& predictions, const Eigen::Matrix3d& R_GtoC,
                                    const cv::Mat& image, const FeatureTracker& tracker, Frame* frame) {
  std::set<int> tracked;
  for (const auto& feature : frame->features()) {
    tracked.insert(feature->id);
  }
  std::set<int> landmarks;
  for (const auto& prediction : predictions) {
    landmarks.insert(prediction.id);
  }

  const int half = options_.patch_half_size;
  const int radius = static_cast<int>(std::lround(options_.search_radius));
  const double f = camera_.PixelsPerRadian();
  constexpr double kSamePoint = 3.0;  // Two features this close [px at the center scale] observe the same point
  const float same_point_cos = static_cast<float>(std::cos(kSamePoint / f));
  std::vector<RecoveredFeature> recovered;
  found_.clear();
  const auto& losses = tracker.track_losses();
  for (const auto& prediction : predictions) {
    const auto patch = patches_.find(prediction.id);
    if (tracked.count(prediction.id) > 0 || patch == patches_.end()) {
      continue;
    }
    // Latest loss: in this image, or kept from the image where it was lost
    const auto loss = losses.find(prediction.id);
    if (loss != losses.end()) {
      patch->second.loss = static_cast<int>(loss->second);
    }
    const cv::Point2f uv(static_cast<float>(prediction.pixel.x()), static_cast<float>(prediction.pixel.y()));
    if (patch->second.loss < 0 || !(options_.loss_reasons & (1 << patch->second.loss)) || !camera_.IsTrackable(uv)) {
      continue;
    }
    statistics_.searched++;

    // Stored patch axes turned into this camera, on the tangent plane of the predicted ray
    const Eigen::Vector3d c = camera_.PixelToRay(uv).cast<double>().normalized();
    Eigen::Vector3d e1 = R_GtoC * patch->second.e1_G;
    e1 -= e1.dot(c) * c;
    if (e1.norm() < 1e-6) {
      continue;
    }
    e1.normalize();

    // Normalized cross-correlation over the search window, sub-pixel peak by parabola fits
    const cv::Mat region = TangentPatch(image, c, e1, half + radius);
    cv::Mat score;
    cv::matchTemplate(region, Patch(patch->second), score, cv::TM_CCOEFF_NORMED);
    double best = 0.0;
    cv::Point peak;
    cv::minMaxLoc(score, nullptr, &best, nullptr, &peak);
    if (best < options_.min_ncc) {
      continue;
    }
    statistics_.matched++;
    auto vertex = [](float left, float middle, float right) {
      const float d = left - 2.0f * middle + right;
      return (d < 0.0f) ? 0.5f * (left - right) / d : 0.0f;
    };
    double x = peak.x - radius;
    double y = peak.y - radius;
    if (peak.x > 0 && peak.x < score.cols - 1) {
      x += vertex(score.at<float>(peak.y, peak.x - 1), score.at<float>(peak.y, peak.x),
                  score.at<float>(peak.y, peak.x + 1));
    }
    if (peak.y > 0 && peak.y < score.rows - 1) {
      y += vertex(score.at<float>(peak.y - 1, peak.x), score.at<float>(peak.y, peak.x),
                  score.at<float>(peak.y + 1, peak.x));
    }
    const cv::Point2f found = camera_.RayToPixel((c + (x * e1 + y * c.cross(e1)) / f).cast<float>());
    if (!camera_.IsTrackable(found)) {
      continue;
    }

    // A tracked feature on the same point continues as the landmark, unless it is a landmark itself or another
    // landmark took it (the point would be observed twice)
    const Eigen::Vector3f ray_found = camera_.PixelToRay(found);
    int replaces = -1;
    bool conflict = false;
    for (const auto& feature : frame->features()) {
      if (feature->track_count > 0 && ray_found.dot(feature->ray) > same_point_cos) {
        conflict = conflict || landmarks.count(feature->id) > 0 || replaces >= 0;
        replaces = feature->id;
      }
    }
    for (const auto& r : recovered) {
      conflict = conflict || (replaces >= 0 && r.replaces == replaces) ||
                 ray_found.dot(camera_.PixelToRay(r.pixel)) > same_point_cos;
    }
    if (conflict) {
      statistics_.conflicts++;
      continue;
    }
    int track_count = patch->second.track_count;
    if (replaces >= 0) {
      statistics_.merged++;
      for (const auto& feature : frame->features()) {
        if (feature->id == replaces) {
          track_count = std::max(track_count, feature->track_count);
        }
      }
    }
    recovered.push_back({prediction.id, found, track_count, replaces});
    found_[prediction.id] = patch->second.loss;
    statistics_.added++;
    statistics_.added_by_loss[patch->second.loss]++;
    statistics_.offset_sum += std::hypot(x, y);
  }
  tracker.AddRecoveredFeatures(recovered, frame);
}

void LandmarkTracker::UpdatePatches(const Frame& frame, const cv::Mat& image, const std::vector<int>& landmark_ids,
                                    const std::vector<int>& updated_ids, const Eigen::Matrix3d& R_CtoG) {
  std::map<int, std::shared_ptr<Feature>> features;
  for (const auto& feature : frame.features()) {
    features[feature->id] = feature;
  }
  const std::set<int> updated(updated_ids.begin(), updated_ids.end());
  for (int id : updated_ids) {
    const auto found = found_.find(id);
    if (found != found_.end()) {
      statistics_.accepted_by_loss[found->second]++;
    }
  }

  // Landmarks initialized or updated with this image get a new patch, cut from this image when first needed; the
  // others keep their last one, cut now from the previous image if it was not needed yet
  std::map<int, LandmarkPatch> patches;
  for (int id : landmark_ids) {
    const auto feature = features.find(id);
    if (feature == features.end() || updated.count(id) == 0) {
      const auto old = patches_.find(id);
      if (old != patches_.end()) {
        Patch(old->second);
        patches.emplace(id, old->second);
      }
      continue;
    }
    const Eigen::Vector3d c = camera_.PixelToRay(feature->second->pixel).cast<double>().normalized();
    const Eigen::Vector3d east = Eigen::Vector3d::UnitY().cross(c);
    if (east.norm() < 1e-6) {
      continue;
    }
    LandmarkPatch& patch = patches[id];
    patch.c = c;
    patch.e1 = east.normalized();
    patch.e1_G = R_CtoG * east.normalized();
    patch.track_count = feature->second->track_count;
  }
  patches_.swap(patches);
  patch_image_ = image;
}

}  // namespace vio

// SPDX-License-Identifier: MIT
/**
 * @file      landmark_tracker.h
 * @brief     Finds landmarks again at their predicted pixels after the feature tracker lost them.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2026-10-04
 * @copyright Copyright (c) 2026 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 *
 * Every landmark updated with an image stores a tangent-plane (gnomonic) patch around its ray, so the patch has no
 * stretch of the equirectangular image. When the feature tracker loses a landmark that stays in the estimator, the
 * patch, turned into the current camera, is matched by normalized cross-correlation around the landmark's predicted
 * pixel; a match is added to the frame under the landmark id.
 */

#pragma once

#include "vio/feature/feature_tracker.h"
#include "vio/scene/camera.h"
#include "vio/scene/frame.h"
#include "vio/util/types.h"

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <map>
#include <vector>

namespace vio {

struct LandmarkTrackerOptions {
  double search_radius = 10.0;  // Search radius around the predicted pixel on the tangent plane [px at the image
                                // center scale]
  double min_ncc = 0.8;         // Minimum normalized cross-correlation of a match
  int patch_half_size = 12;     // Half size of the stored patch [px]
  int loss_reasons = 63;        // Losses after which a landmark is searched, bit mask of TrackLoss: 1 not found,
                                // 2 masked, 4 outlier, 8 grid cap, 16 light cluster, 32 border

  bool Check() const;
};

class LandmarkTracker {
 public:
  // Counts since the start
  struct Statistics {
    int searched = 0;                             // Lost landmarks with a patch and a trackable predicted pixel
    int matched = 0;                              // Patch found above min_ncc
    int conflicts = 0;                            // Match on a point another landmark observes (not added)
    int added = 0;                                // Added to the frame under the landmark id
    int merged = 0;                               // Added ones that continue a tracked feature on the same point
    double offset_sum = 0.0;                      // Sum of the distances of the added matches from the prediction [px]
    int added_by_loss[kNumTrackLosses] = {};      // Added, by the loss of the landmark (TrackLoss)
    int accepted_by_loss[kNumTrackLosses] = {};   // Added and used by the estimator (gate passed)
  };

  LandmarkTracker(const LandmarkTrackerOptions& options, const Camera& camera);

  /**
   * @brief Find the landmarks the frame does not track at their predicted pixels and add them to the frame
   * @param predictions Predicted pixel of every landmark in the estimator
   * @param R_GtoC Predicted camera orientation of the frame
   * @param image Image of the frame (CV_32F)
   * @param tracker Feature tracker that tracked the frame (losses, adding)
   * @param frame Frame after tracking
   */
  void FindLandmarks(const std::vector<PixelObservation>& predictions, const Eigen::Matrix3d& R_GtoC,
                     const cv::Mat& image, const FeatureTracker& tracker, Frame* frame);

  /**
   * @brief Store the patch of every landmark initialized or updated with the frame; the others keep their last one
   * @param frame Frame
   * @param image Image of the frame (CV_32F); kept (shared, not copied) until the next call, the patches are cut from
   *              it when needed
   * @param landmark_ids Landmarks in the estimator after the frame
   * @param updated_ids Landmarks initialized or updated with the frame
   * @param R_CtoG Camera orientation of the frame
   */
  void UpdatePatches(const Frame& frame, const cv::Mat& image, const std::vector<int>& landmark_ids,
                     const std::vector<int>& updated_ids, const Eigen::Matrix3d& R_CtoG);

  const Statistics& statistics() const { return statistics_; }

 private:
  // Tangent-plane patch: samples at c + (x e1 + y e2) / f for x, y in [-half, half] px, e2 = c x e1, f = image scale
  cv::Mat TangentPatch(const cv::Mat& image, const Eigen::Vector3d& c, const Eigen::Vector3d& e1, int half) const;

  // Latest appearance of a landmark. The patch is cut from the image of the last UpdatePatches call when it is first
  // needed (a landmark that stays tracked never needs it), at the latest when the next call keeps it
  struct LandmarkPatch {
    cv::Mat patch;         // (2 patch_half_size + 1)^2, CV_32F; empty until cut
    Eigen::Vector3d c;     // Ray of the patch center in that camera
    Eigen::Vector3d e1;    // First patch axis (east) in that camera
    Eigen::Vector3d e1_G;  // First patch axis in the world frame
    int track_count = 0;   // Track length when stored (grid priority when found again)
    int loss = -1;         // Why the tracker lost it (TrackLoss; -1: tracked)
  };

  // The patch of a landmark, cut from the image of the last UpdatePatches call if not yet
  const cv::Mat& Patch(LandmarkPatch& patch) const;

  LandmarkTrackerOptions options_;
  Camera camera_;
  std::map<int, LandmarkPatch> patches_;  // By landmark id
  cv::Mat patch_image_;                   // Image of the last UpdatePatches call (CV_32F)
  std::map<int, int> found_;              // Landmarks found in the current frame and their losses
  Statistics statistics_;
};

}  // namespace vio

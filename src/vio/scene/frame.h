// SPDX-License-Identifier: MIT
/**
 * @file      frame.h
 * @brief     An image with its tracked features and the per-cell cap of the feature grid.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include "vio/util/types.h"

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <memory>
#include <vector>

namespace vio {

// A tracked feature in one image
struct Feature {
  Feature(int feature_id, const cv::Point2f& feature_pixel) : id(feature_id), pixel(feature_pixel) {}

  int id;                                        // Track id
  cv::Point2f pixel;                             // Pixel coordinates
  Eigen::Vector3f ray = Eigen::Vector3f::Zero(); // Unit ray in the camera frame
  int track_count = 0;                           // Images the feature was tracked through
};

class Frame {
 public:
  /**
   * @param timestamp Image time [s]
   * @param index Image index
   * @param image Grayscale image (copied)
   */
  Frame(Timestamp timestamp, int index, const cv::Mat& image);

  Timestamp timestamp() const { return timestamp_; }
  int index() const { return index_; }
  const cv::Mat& image() const { return image_; }
  const std::vector<std::shared_ptr<Feature>>& features() const { return features_; }

  void AddFeature(std::shared_ptr<Feature> feature) { features_.push_back(std::move(feature)); }
  void ClearFeatures() { features_.clear(); }

  /**
   * @brief Set the feature grid
   * @param cols Grid columns
   * @param rows Grid rows
   * @param max_per_cell Features kept per cell
   */
  void SetGrid(int cols, int rows, int max_per_cell);

  // Assign the features to the grid cells by pixel
  void AssignFeaturesToGrid();

  // Keep the longest tracks of every cell (max_per_cell) and drop the others
  void LimitFeaturesPerCell();

 private:
  Timestamp timestamp_;
  int index_;
  cv::Mat image_;
  int width_;
  int height_;
  std::vector<std::shared_ptr<Feature>> features_;

  int grid_cols_;
  int grid_rows_;
  int max_features_per_cell_;
  std::vector<std::vector<std::vector<int>>> grid_;  // [row][col]: feature indices
};

}  // namespace vio

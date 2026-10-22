// SPDX-License-Identifier: MIT
/**
 * @file      frame.cc
 * @brief     Implements the frame and its feature grid.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "vio/scene/frame.h"

#include <algorithm>

namespace vio {

Frame::Frame(Timestamp timestamp, int index, const cv::Mat& image)
    : timestamp_(timestamp),
      index_(index),
      image_(image.clone()),
      width_(image.cols),
      height_(image.rows),
      grid_cols_(20),
      grid_rows_(10),
      max_features_per_cell_(4) {
  grid_.resize(grid_rows_, std::vector<std::vector<int>>(grid_cols_));
}

void Frame::SetGrid(int cols, int rows, int max_per_cell) {
  grid_cols_ = cols;
  grid_rows_ = rows;
  max_features_per_cell_ = max_per_cell;
  grid_.clear();
  grid_.resize(grid_rows_, std::vector<std::vector<int>>(grid_cols_));
}

void Frame::AssignFeaturesToGrid() {
  for (auto& row : grid_) {
    for (auto& cell : row) {
      cell.clear();
    }
  }
  if (features_.empty()) {
    return;
  }

  const float cell_width = static_cast<float>(width_) / grid_cols_;
  const float cell_height = static_cast<float>(height_) / grid_rows_;
  for (size_t i = 0; i < features_.size(); ++i) {
    const auto& feature = features_[i];
    if (!feature) {
      continue;
    }
    const cv::Point2f pixel = feature->pixel;
    if (pixel.x < 0 || pixel.x >= width_ || pixel.y < 0 || pixel.y >= height_) {
      continue;
    }
    const int grid_x = std::min(static_cast<int>(pixel.x / cell_width), grid_cols_ - 1);
    const int grid_y = std::min(static_cast<int>(pixel.y / cell_height), grid_rows_ - 1);
    grid_[grid_y][grid_x].push_back(static_cast<int>(i));
  }
}

void Frame::LimitFeaturesPerCell() {
  for (int row = 0; row < grid_rows_; ++row) {
    for (int col = 0; col < grid_cols_; ++col) {
      auto& cell = grid_[row][col];
      if (cell.size() <= static_cast<size_t>(max_features_per_cell_)) {
        continue;
      }
      // Longest tracks first
      std::sort(cell.begin(), cell.end(), [this](int a, int b) {
        if (a >= static_cast<int>(features_.size()) || b >= static_cast<int>(features_.size())) return false;
        if (!features_[a] || !features_[b]) return false;
        return features_[a]->track_count > features_[b]->track_count;
      });
      cell.resize(max_features_per_cell_);
    }
  }

  // Keep the features that stayed in a cell
  std::vector<bool> keep_feature(features_.size(), false);
  for (const auto& row : grid_) {
    for (const auto& cell : row) {
      for (int idx : cell) {
        if (idx >= 0 && idx < static_cast<int>(features_.size())) {
          keep_feature[idx] = true;
        }
      }
    }
  }
  std::vector<std::shared_ptr<Feature>> filtered_features;
  for (size_t i = 0; i < features_.size(); ++i) {
    if (keep_feature[i]) {
      filtered_features.push_back(features_[i]);
    }
  }
  features_ = std::move(filtered_features);

  // Reassign with the new indices
  AssignFeaturesToGrid();
}

}  // namespace vio

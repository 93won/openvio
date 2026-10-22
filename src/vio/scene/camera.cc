// SPDX-License-Identifier: MIT
/**
 * @file      camera.cc
 * @brief     Implements the camera checks and its untracked image regions.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "vio/scene/camera.h"

#include "vio/util/logging.h"

namespace vio {

bool Camera::Check() const {
  REQUIRE_OPTION(model_id != CameraModelId::kInvalid);
  REQUIRE_OPTION(static_cast<int>(params.size()) == CameraModelNumParameters(model_id));
  REQUIRE_OPTION(width > 0 && height > 0);
  REQUIRE_OPTION(polar_exclusion_ratio >= 0.0f && polar_exclusion_ratio < 0.5f);
  REQUIRE_OPTION(border_margin >= 0 && 2 * border_margin < width);
  for (const cv::Vec4f& r : mask_rectangles) {
    REQUIRE_OPTION(r[0] <= r[1] && r[2] <= r[3]);
  }
  return true;
}

bool Camera::IsMasked(const cv::Point2f& pixel) const {
  const float v_ratio = pixel.y / static_cast<float>(height);
  if ((v_ratio < polar_exclusion_ratio) || (v_ratio > (1.0f - polar_exclusion_ratio))) {
    return true;
  }
  const float x = pixel.x / static_cast<float>(width);
  const float y = pixel.y / static_cast<float>(height);
  for (const cv::Vec4f& r : mask_rectangles) {
    if (x >= r[0] && x < r[1] && y >= r[2] && y < r[3]) {
      return true;
    }
  }
  return false;
}

bool Camera::IsNearBorder(const cv::Point2f& pixel) const {
  const float margin = static_cast<float>(border_margin);
  return (pixel.x < margin) || (pixel.x > static_cast<float>(width) - margin) || (pixel.y < margin) ||
         (pixel.y > static_cast<float>(height) - margin);
}

}  // namespace vio

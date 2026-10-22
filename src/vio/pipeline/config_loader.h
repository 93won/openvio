// SPDX-License-Identifier: MIT
/**
 * @file      config_loader.h
 * @brief     Reads the options from a YAML file with command-line overrides and checks them.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-25
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 *
 * YAML sections: camera, imu, rig, feature_tracker, landmark_tracker, initializer, estimator (with
 * estimator.triangulation); camera1 (with rig.imu_from_camera1) makes a stereo rig. A key missing from the file keeps
 * its default. An override "--section.key=value" replaces a key and is parsed by the same YAML reader as the file (the
 * same number for the same text); an unknown key fails.
 */

#pragma once

#include "vio/odometry/visual_inertial_odometry.h"

#include <string>
#include <vector>

namespace vio {

class ConfigLoader {
 public:
  /**
   * @brief Read the options
   * @param path YAML file
   * @param overrides Overrides "--section.key=value" (or without the dashes)
   * @return false if the file cannot be read, an override is malformed or names an unknown key
   */
  bool Read(const std::string& path, const std::vector<std::string>& overrides = {});

  // Check every option
  bool Check() const { return odometry.Check(); }

  VisualInertialOdometryOptions odometry;
};

}  // namespace vio

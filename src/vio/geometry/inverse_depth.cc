// SPDX-License-Identifier: MIT
/**
 * @file      inverse_depth.cc
 * @brief     Implements the inverse-depth point parameterization.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2026-09-28
 * @copyright Copyright (c) 2026 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "vio/geometry/inverse_depth.h"

namespace vio {

Eigen::Vector3d InverseDepthPoint(const Eigen::Vector3d& anchor, const Eigen::Matrix3d& R_BtoG,
                                  const Eigen::Vector3d& theta) {
  return anchor + R_BtoG * Eigen::Vector3d(theta(0), theta(1), 1.0) / theta(2);
}

Eigen::Matrix3d InverseDepthJacobian(const Eigen::Matrix3d& R_BtoG, const Eigen::Vector3d& theta) {
  const double rho = theta(2);
  Eigen::Matrix3d J;
  J << 1.0 / rho, 0.0, -theta(0) / (rho * rho),
       0.0, 1.0 / rho, -theta(1) / (rho * rho),
       0.0, 0.0, -1.0 / (rho * rho);
  return R_BtoG * J;
}

}  // namespace vio

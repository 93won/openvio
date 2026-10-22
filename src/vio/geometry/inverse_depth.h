// SPDX-License-Identifier: MIT
/**
 * @file      inverse_depth.h
 * @brief     Points parameterized by inverse depth from an anchor.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2026-09-28
 * @copyright Copyright (c) 2026 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 *
 * Sphere-complete inverse depth: p_FinG = c + R_BtoG [alpha, beta, 1] / rho. The anchor point c and the ray frame
 * R_BtoG (anchor ray along +z) are constants, so theta = (alpha, beta, rho) is a reparameterization of the point that
 * stays linear at low parallax and is regular in every direction of the sphere.
 */

#pragma once

#include <Eigen/Dense>

namespace vio {

/**
 * @brief Point of an inverse-depth parameter
 * @param anchor Anchor point c [m]
 * @param R_BtoG Ray frame of the anchor
 * @param theta (alpha, beta, rho)
 * @return p_FinG
 */
Eigen::Vector3d InverseDepthPoint(const Eigen::Vector3d& anchor, const Eigen::Matrix3d& R_BtoG,
                                  const Eigen::Vector3d& theta);

/**
 * @brief Jacobian of the point w.r.t. (alpha, beta, rho)
 * @param R_BtoG Ray frame of the anchor
 * @param theta (alpha, beta, rho)
 * @return d p_FinG / d theta
 */
Eigen::Matrix3d InverseDepthJacobian(const Eigen::Matrix3d& R_BtoG, const Eigen::Vector3d& theta);

}  // namespace vio

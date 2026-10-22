// SPDX-License-Identifier: MIT
/**
 * @file      lie_group.h
 * @brief     Rotations on SO(3) and unit quaternions in the JPL convention.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2026-09-28
 * @copyright Copyright (c) 2026 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 *
 * A quaternion is stored as [x, y, z, w] in the JPL convention. For q = q_AtoB:
 *   Rot(q) = (2w^2 - 1) I - 2w [v]x + 2 v v^T   maps A-frame vectors into B
 *   Rot(p (x) q) = Rot(p) Rot(q)
 *   Rot([sin(t/2) n; cos(t/2)]) = Exp(-t n)    (Exp = standard Rodrigues map)
 */

#pragma once

#include <Eigen/Dense>

#include <cmath>

namespace vio {

/**
 * @brief Skew-symmetric matrix of a vector
 * @param v Input vector
 * @return [v]x such that [v]x w = v x w
 */
inline Eigen::Matrix3d Skew(const Eigen::Vector3d& v) {
  Eigen::Matrix3d m;
  m << 0.0, -v.z(), v.y(),
       v.z(), 0.0, -v.x(),
       -v.y(), v.x(), 0.0;
  return m;
}

namespace jpl {

/**
 * @brief Quaternion product
 * @param p Left quaternion [x, y, z, w]
 * @param q Right quaternion [x, y, z, w]
 * @return p (x) q, with Rot(p (x) q) = Rot(p) Rot(q)
 */
inline Eigen::Vector4d Multiply(const Eigen::Vector4d& p, const Eigen::Vector4d& q) {
  const Eigen::Vector3d pv = p.head<3>();
  const Eigen::Vector3d qv = q.head<3>();
  Eigen::Vector4d r;
  r.head<3>() = p.w() * qv + q.w() * pv - pv.cross(qv);
  r.w() = p.w() * q.w() - pv.dot(qv);
  return r;
}

/**
 * @brief Inverse of a unit quaternion
 * @param q Unit quaternion [x, y, z, w]
 * @return Conjugate [-x, -y, -z, w]
 */
inline Eigen::Vector4d Inverse(const Eigen::Vector4d& q) { return Eigen::Vector4d(-q.x(), -q.y(), -q.z(), q.w()); }

/**
 * @brief Normalize a quaternion and choose the sign with w >= 0
 * @param q Quaternion [x, y, z, w]
 * @return Unit quaternion with non-negative scalar part
 */
inline Eigen::Vector4d NormalizePositive(Eigen::Vector4d q) {
  if (q.w() < 0.0) {
    q = -q;
  }
  return q / q.norm();
}

/**
 * @brief Quaternion rate matrix: dq/dt = 1/2 Omega(w) q for q = q_GtoI and body rate w
 * @param w Angular velocity in the body frame
 * @return [[-[w]x, w], [-w^T, 0]]
 */
inline Eigen::Matrix4d Omega(const Eigen::Vector3d& w) {
  Eigen::Matrix4d m;
  m.block<3, 3>(0, 0) = -Skew(w);
  m.block<3, 1>(0, 3) = w;
  m.block<1, 3>(3, 0) = -w.transpose();
  m(3, 3) = 0.0;
  return m;
}

/**
 * @brief Top-left 3x3 block of the left-product matrix L(q), where q (x) p = L(q) p
 * @param q Quaternion [x, y, z, w]
 * @return w I - [v]x
 */
inline Eigen::Matrix3d LeftProductBlock(const Eigen::Vector4d& q) {
  return q.w() * Eigen::Matrix3d::Identity() - Skew(q.head<3>());
}

/**
 * @brief Top-left 3x3 block of the right-product matrix R(p), where q (x) p = R(p) q
 * @param p Quaternion [x, y, z, w]
 * @return w I + [v]x
 */
inline Eigen::Matrix3d RightProductBlock(const Eigen::Vector4d& p) {
  return p.w() * Eigen::Matrix3d::Identity() + Skew(p.head<3>());
}

/**
 * @brief Rotation matrix of a quaternion
 * @param q Unit quaternion q_AtoB [x, y, z, w]
 * @return R_AtoB
 */
inline Eigen::Matrix3d ToRotation(const Eigen::Vector4d& q) {
  const Eigen::Vector3d v = q.head<3>();
  const double w = q.w();
  return (2.0 * w * w - 1.0) * Eigen::Matrix3d::Identity() - 2.0 * w * Skew(v) + 2.0 * v * v.transpose();
}

/**
 * @brief Quaternion of a rotation matrix (inverse of ToRotation)
 *
 * With T = trace(C) = 4w^2 - 1 and C_ii = 2w^2 - 1 + 2v_i^2 every squared component follows
 * from the diagonal; the others follow from C - C^T = -4w[v]x and C + C^T = 2(2w^2 - 1)I + 4vv^T.
 * The branch with the largest pivot is used for accuracy.
 *
 * @param C Rotation matrix R_AtoB
 * @return q_AtoB [x, y, z, w] with w >= 0
 */
inline Eigen::Vector4d FromRotation(const Eigen::Matrix3d& C) {
  const double T = C.trace();
  Eigen::Vector4d q;
  if (T >= C(0, 0) && T >= C(1, 1) && T >= C(2, 2)) {
    const double w = 0.5 * std::sqrt(1.0 + T);
    q << (C(1, 2) - C(2, 1)) / (4.0 * w),
         (C(2, 0) - C(0, 2)) / (4.0 * w),
         (C(0, 1) - C(1, 0)) / (4.0 * w),
         w;
  } else if (C(0, 0) >= C(1, 1) && C(0, 0) >= C(2, 2)) {
    const double x = 0.5 * std::sqrt(1.0 + 2.0 * C(0, 0) - T);
    q << x,
         (C(0, 1) + C(1, 0)) / (4.0 * x),
         (C(0, 2) + C(2, 0)) / (4.0 * x),
         (C(1, 2) - C(2, 1)) / (4.0 * x);
  } else if (C(1, 1) >= C(2, 2)) {
    const double y = 0.5 * std::sqrt(1.0 + 2.0 * C(1, 1) - T);
    q << (C(0, 1) + C(1, 0)) / (4.0 * y),
         y,
         (C(1, 2) + C(2, 1)) / (4.0 * y),
         (C(2, 0) - C(0, 2)) / (4.0 * y);
  } else {
    const double z = 0.5 * std::sqrt(1.0 + 2.0 * C(2, 2) - T);
    q << (C(0, 2) + C(2, 0)) / (4.0 * z),
         (C(1, 2) + C(2, 1)) / (4.0 * z),
         z,
         (C(0, 1) - C(1, 0)) / (4.0 * z);
  }
  if (q.w() < 0.0) {
    q = -q;
  }
  return q.normalized();
}

}  // namespace jpl

/**
 * @brief SO(3) exponential (Rodrigues formula)
 * @param phi Rotation vector
 * @return Exp(phi) = I + sin|phi|/|phi| [phi]x + (1 - cos|phi|)/|phi|^2 [phi]x^2
 */
inline Eigen::Matrix3d ExpSO3(const Eigen::Vector3d& phi) {
  const double t = phi.norm();
  const Eigen::Matrix3d K = Skew(phi);
  if (t < 1e-10) {
    return Eigen::Matrix3d::Identity() + K + 0.5 * K * K;
  }
  return Eigen::Matrix3d::Identity() + (std::sin(t) / t) * K + ((1.0 - std::cos(t)) / (t * t)) * K * K;
}

/**
 * @brief SO(3) logarithm computed through the quaternion (accurate for small angles)
 * @param R Rotation matrix
 * @return phi such that ExpSO3(phi) = R
 */
inline Eigen::Vector3d LogSO3(const Eigen::Matrix3d& R) {
  // Rot([sin(t/2) n; cos(t/2)]) = Exp(-t n), hence log(R) = -t n
  const Eigen::Vector4d q = jpl::FromRotation(R);
  const double s = q.head<3>().norm();
  if (s < 1e-12) {
    return -2.0 * q.head<3>() / q.w();
  }
  return -2.0 * std::atan2(s, q.w()) * q.head<3>() / s;
}

/**
 * @brief Right Jacobian of SO(3)
 * @param phi Rotation vector
 * @return Jr(phi)
 */
inline Eigen::Matrix3d RightJacobianSO3(const Eigen::Vector3d& phi) {
  const double t = phi.norm();
  const Eigen::Matrix3d K = Skew(phi);
  if (t < 1e-8) {
    return Eigen::Matrix3d::Identity() - 0.5 * K + (1.0 / 6.0) * K * K;
  }
  return Eigen::Matrix3d::Identity() - ((1.0 - std::cos(t)) / (t * t)) * K + ((t - std::sin(t)) / (t * t * t)) * K * K;
}

/**
 * @brief Inverse of the right Jacobian of SO(3)
 * @param phi Rotation vector
 * @return Jr(phi)^-1
 */
inline Eigen::Matrix3d RightJacobianInverseSO3(const Eigen::Vector3d& phi) {
  const double t = phi.norm();
  const Eigen::Matrix3d K = Skew(phi);
  if (t < 1e-8) {
    return Eigen::Matrix3d::Identity() + 0.5 * K + (1.0 / 12.0) * K * K;
  }
  const double c = 1.0 / (t * t) - (1.0 + std::cos(t)) / (2.0 * t * std::sin(t));
  return Eigen::Matrix3d::Identity() + 0.5 * K + c * K * K;
}

/**
 * @brief Rotation R_GtoI whose third column is the unit gravity direction expressed in I
 *
 * The frame is completed with cross products: x is perpendicular to gravity and to the I axis (x or y) that has the
 * smaller component along gravity, y = z x x. This choice fixes the yaw, which gravity leaves free.
 *
 * @param gravity_in_I Gravity (up) vector expressed in frame I
 * @return R_GtoI with columns [x, y, z] expressed in I
 */
inline Eigen::Matrix3d GravityAlignedRotation(const Eigen::Vector3d& gravity_in_I) {
  const Eigen::Vector3d z = gravity_in_I.normalized();
  const Eigen::Vector3d reference =
      (std::abs(z.x()) < std::abs(z.y())) ? Eigen::Vector3d::UnitX() : Eigen::Vector3d::UnitY();
  const Eigen::Vector3d x = z.cross(reference).normalized();
  Eigen::Matrix3d R_GtoI;
  R_GtoI << x, z.cross(x).normalized(), z;
  return R_GtoI;
}

}  // namespace vio

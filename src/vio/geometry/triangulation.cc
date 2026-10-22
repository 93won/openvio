// SPDX-License-Identifier: MIT
/**
 * @file      triangulation.cc
 * @brief     Implements the triangulation of a point from unit rays.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2026-10-06
 * @copyright Copyright (c) 2026 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 *
 * References:
 *   [1] R. Hartley, A. Zisserman, "Multiple View Geometry in Computer Vision", 2nd ed., 2004, Ch. 12 (triangulation).
 *   [2] A. I. Mourikis, S. I. Roumeliotis, "A Multi-State Constraint Kalman Filter for Vision-aided Inertial
 *       Navigation", ICRA 2007, Appendix (inverse-depth Gauss-Newton estimate of a feature).
 *   [3] D. W. Marquardt, "An Algorithm for Least-Squares Estimation of Nonlinear Parameters", SIAM J. Appl. Math.
 *       11(2), 1963 (damping of the Gauss-Newton matrix).
 */

#include "vio/geometry/triangulation.h"

#include "vio/geometry/lie_group.h"
#include "vio/util/logging.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace vio {
namespace {

// One observation of the inverse-depth problem. A' is the anchor camera frame and C' the observation's camera frame,
// both rotated so that their measured ray is the +z axis.
struct RayObservation {
  Eigen::Matrix3d R_AtoC;  // A' -> C'
  Eigen::Vector3d p_AinC;  // Origin of A' in C'
  Eigen::Vector3d p_CinA;  // Origin of C' in A'
};

/**
 * Reprojection error of one point in inverse depth ([2], Appendix): theta = (alpha, beta, rho) = (x / z, y / z, 1 / z)
 * of the point in A'. In C' the point is (R_AtoC (alpha, beta, 1) + rho p_AinC) / rho, so its normalized-plane
 * coordinates are h_xy / h_z with h = R_AtoC (alpha, beta, 1) + rho p_AinC; the measurement is (0, 0) in every C'.
 */
class InverseDepthProblem {
 public:
  explicit InverseDepthProblem(const std::vector<RayObservation>& observations) : observations_(observations) {}

  // Sum of the squared normalized-plane errors
  double Cost(const Eigen::Vector3d& theta) const {
    double cost = 0.0;
    for (const RayObservation& o : observations_) {
      const Eigen::Vector3d h = Projection(o, theta);
      const double u = h.x() / h.z();
      const double v = h.y() / h.z();
      cost += u * u + v * v;
    }
    return cost;
  }

  // Gauss-Newton normal equations at theta: H = sum J^T J, g = -sum J^T e (e: normalized-plane error)
  void Linearize(const Eigen::Vector3d& theta, Eigen::Matrix3d* H, Eigen::Vector3d* g) const {
    H->setZero();
    g->setZero();
    for (const RayObservation& o : observations_) {
      const Eigen::Vector3d h = Projection(o, theta);
      const Eigen::Vector2d e(h.x() / h.z(), h.y() / h.z());
      // dh/dtheta = [R_AtoC e1, R_AtoC e2, p_AinC], and d(h_xy / h_z) = (dh_xy - e dh_z) / h_z
      Eigen::Matrix3d dh;
      dh << o.R_AtoC.col(0), o.R_AtoC.col(1), o.p_AinC;
      const Eigen::Matrix<double, 2, 3> J = (dh.topRows<2>() - e * dh.row(2)) / h.z();
      *H += J.transpose() * J;
      *g -= J.transpose() * e;
    }
  }

 private:
  static Eigen::Vector3d Projection(const RayObservation& o, const Eigen::Vector3d& theta) {
    return o.R_AtoC * Eigen::Vector3d(theta.x(), theta.y(), 1.0) + theta.z() * o.p_AinC;
  }

  const std::vector<RayObservation>& observations_;
};

/**
 * Levenberg-Marquardt ([3]): the Gauss-Newton matrix with its diagonal scaled by (1 + damping). A step that does not
 * raise the cost is taken and divides the damping by the factor; any other step is discarded and multiplies it. The
 * search ends after a step with a relative cost decrease below min_relative_decrease (that step is kept), after
 * max_iterations taken steps, once the damping reaches max_damping, or when the last taken step is shorter than
 * min_step.
 */
Eigen::Vector3d MinimizeInverseDepth(const TriangulationOptions& options, const InverseDepthProblem& problem,
                                     Eigen::Vector3d theta) {
  double damping = options.initial_damping;
  double last_step = std::numeric_limits<double>::infinity();
  int taken = 0;
  bool linearize = true;
  Eigen::Matrix3d H;
  Eigen::Vector3d g;
  double cost = problem.Cost(theta);
  while (taken < options.max_iterations && damping < options.max_damping && last_step > options.min_step) {
    if (linearize) {
      problem.Linearize(theta, &H, &g);
    }
    Eigen::Matrix3d H_damped = H;
    H_damped.diagonal() *= 1.0 + damping;
    const Eigen::Vector3d step = H_damped.colPivHouseholderQr().solve(g);
    const double trial = problem.Cost(theta + step);
    if (!(trial <= cost)) {
      damping *= options.damping_factor;
      linearize = false;
      continue;
    }
    theta += step;
    if ((cost - trial) / cost < options.min_relative_decrease) {
      break;
    }
    cost = trial;
    damping /= options.damping_factor;
    last_step = step.norm();
    ++taken;
    linearize = true;
  }
  return theta;
}

}  // namespace

bool TriangulationOptions::Check() const {
  REQUIRE_OPTION(max_iterations >= 0);
  REQUIRE_OPTION(initial_damping > 0.0);
  REQUIRE_OPTION(max_damping >= initial_damping);
  REQUIRE_OPTION(min_step >= 0.0);
  REQUIRE_OPTION(min_relative_decrease >= 0.0);
  REQUIRE_OPTION(damping_factor > 1.0);
  REQUIRE_OPTION(min_depth > 0.0);
  REQUIRE_OPTION(max_depth > min_depth);
  REQUIRE_OPTION(max_depth_baseline_ratio > 0.0);
  REQUIRE_OPTION(max_condition_number > 1.0);
  return true;
}

Eigen::Matrix3d RotationToPlusZ(const Eigen::Vector3d& b) {
  const Eigen::Vector3d z = Eigen::Vector3d::UnitZ();
  const Eigen::Vector3d axis = b.cross(z);
  const double s = axis.norm();
  const double c = b.dot(z);
  if (s < 1e-12) {
    if (c > 0.0) {
      return Eigen::Matrix3d::Identity();
    }
    // Opposite direction: half turn about x
    return Eigen::Vector3d(1.0, -1.0, -1.0).asDiagonal();
  }
  return ExpSO3(axis / s * std::atan2(s, c));
}

TriangulationStatus TriangulatePoint(const TriangulationOptions& options,
                                     const std::vector<Eigen::Vector3d>& rays,
                                     const std::vector<CameraPose>& poses,
                                     Eigen::Vector3d* p_FinG) {
  // The last observation is the anchor; A is its camera frame
  const CameraPose& anchor = poses.back();

  // Linear estimate ([1]): the point closest to every observation ray in the least-squares sense. With the projector
  // P_i = I - d_i d_i^T onto the plane orthogonal to the ray direction d_i through the camera center c_i (both in A),
  // sum_i P_i p = sum_i P_i c_i.
  Eigen::Matrix3d normal = Eigen::Matrix3d::Zero();
  Eigen::Vector3d rhs = Eigen::Vector3d::Zero();
  for (size_t i = 0; i < poses.size(); ++i) {
    const Eigen::Matrix3d R_CtoA = anchor.R_GtoC * poses[i].R_GtoC.transpose();
    const Eigen::Vector3d center = anchor.R_GtoC * (poses[i].p_CinG - anchor.p_CinG);
    const Eigen::Vector3d direction = (R_CtoA * rays[i]).normalized();
    const Eigen::Matrix3d projector = Eigen::Matrix3d::Identity() - direction * direction.transpose();
    normal += projector;
    rhs += projector * center;
  }
  const Eigen::Vector3d p_FinA = normal.colPivHouseholderQr().solve(rhs);
  const Eigen::Vector3d eigenvalues = Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(normal).eigenvalues();
  const double condition = eigenvalues(2) / eigenvalues(0);
  if (std::isnan(p_FinA.norm())) {
    return TriangulationStatus::kNotFinite;
  }
  if (std::abs(condition) > options.max_condition_number) {
    return TriangulationStatus::kIllConditioned;
  }
  const double linear_depth = p_FinA.dot(rays.back());  // Along the anchor ray
  if (linear_depth < options.min_depth || linear_depth > options.max_depth) {
    return TriangulationStatus::kLinearDepth;
  }

  // Refinement in inverse depth ([2]). Turning the anchor frame and every observation frame so that the measured ray
  // is +z keeps the parameterization defined for rays anywhere on the sphere.
  const Eigen::Matrix3d R_AtoAp = RotationToPlusZ(rays.back());
  std::vector<RayObservation> observations(poses.size());
  for (size_t i = 0; i < poses.size(); ++i) {
    const Eigen::Matrix3d R_AtoC = poses[i].R_GtoC * anchor.R_GtoC.transpose();
    const Eigen::Vector3d p_CinA = anchor.R_GtoC * (poses[i].p_CinG - anchor.p_CinG);
    RayObservation& o = observations[i];
    o.R_AtoC = RotationToPlusZ(rays[i]) * R_AtoC * R_AtoAp.transpose();
    o.p_CinA = R_AtoAp * p_CinA;
    o.p_AinC = -o.R_AtoC * o.p_CinA;
  }
  const Eigen::Vector3d p_FinAp = R_AtoAp * p_FinA;
  const Eigen::Vector3d theta0(p_FinAp.x() / p_FinAp.z(), p_FinAp.y() / p_FinAp.z(), 1.0 / p_FinAp.z());
  const Eigen::Vector3d theta = MinimizeInverseDepth(options, InverseDepthProblem(observations), theta0);
  const Eigen::Vector3d p_refined(theta.x() / theta.z(), theta.y() / theta.z(), 1.0 / theta.z());

  // Largest distance of a camera center from the line of sight of the anchor through the point
  const Eigen::Vector3d sight = p_refined.normalized();
  double baseline = 0.0;
  for (const RayObservation& o : observations) {
    baseline = std::max(baseline, (o.p_CinA - o.p_CinA.dot(sight) * sight).norm());
  }

  if (std::isnan(p_refined.norm())) {
    return TriangulationStatus::kNotFinite;
  }
  if (p_refined.z() < options.min_depth || p_refined.z() > options.max_depth) {
    return TriangulationStatus::kRefinedDepth;
  }
  if (p_refined.norm() / baseline > options.max_depth_baseline_ratio) {
    return TriangulationStatus::kBaseline;
  }
  *p_FinG = anchor.R_GtoC.transpose() * (R_AtoAp.transpose() * p_refined) + anchor.p_CinG;
  return TriangulationStatus::kSuccess;
}

}  // namespace vio

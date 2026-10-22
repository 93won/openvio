// SPDX-License-Identifier: MIT
/**
 * @file      imu_interval.cc
 * @brief     Implements the IMU samples of a time interval.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2026-10-06
 * @copyright Copyright (c) 2026 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "vio/sensor/imu_interval.h"

#include <cmath>

namespace vio {
namespace {

// The signal at time t on the segment between two samples
ImuSample OnSegment(const ImuSample& from, const ImuSample& to, double t) {
  const double s = (t - from.timestamp) / (to.timestamp - from.timestamp);
  ImuSample sample;
  sample.timestamp = t;
  sample.gyro = (1.0 - s) * from.gyro + s * to.gyro;
  sample.accel = (1.0 - s) * from.accel + s * to.accel;
  return sample;
}

}  // namespace

std::vector<ImuSample> ImuSamplesOverInterval(const std::vector<ImuSample>& buffer, double t0, double t1,
                                              bool extrapolate_end) {
  std::vector<ImuSample> knots;
  for (size_t k = 1; k < buffer.size(); ++k) {
    const ImuSample& from = buffer[k - 1];
    const ImuSample& to = buffer[k];
    if (from.timestamp < t0) {
      // Segment that starts before the interval: it opens the interval when it reaches past t0
      if (to.timestamp > t0) {
        knots.push_back(OnSegment(from, to, t0));
      }
      continue;
    }
    if (to.timestamp <= t1) {
      knots.push_back(from);
      continue;
    }
    // Segment that leaves the interval: close it at t1
    if (from.timestamp <= t1) {
      knots.push_back(from);
      if (from.timestamp != t1) {
        knots.push_back(OnSegment(from, to, t1));
      }
    } else if (k >= 2) {
      // The interval lies inside the previous segment, which opened it
      knots.push_back(OnSegment(buffer[k - 2], from, t1));
    }
    break;
  }
  if (extrapolate_end && !knots.empty() && knots.back().timestamp != t1) {
    knots.push_back(OnSegment(buffer[buffer.size() - 2], buffer.back(), t1));
  }

  std::vector<ImuSample> result;
  result.reserve(knots.size());
  for (size_t k = 0; k < knots.size(); ++k) {
    const bool zero_step = k + 1 < knots.size() && std::abs(knots[k + 1].timestamp - knots[k].timestamp) < 1e-12;
    if (!zero_step) {
      result.push_back(knots[k]);
    }
  }
  return result;
}

}  // namespace vio

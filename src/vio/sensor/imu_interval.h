// SPDX-License-Identifier: MIT
/**
 * @file      imu_interval.h
 * @brief     IMU samples of a time interval, for piecewise integration.
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2026-10-06
 * @copyright Copyright (c) 2026 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include "vio/util/types.h"

#include <vector>

namespace vio {

/**
 * @brief Knots of the IMU signal over [t0, t1], the signal being linear between buffer samples
 *
 * The first knot is the signal at t0 when t0 falls strictly between two samples (otherwise the first sample at or
 * after t0); then every sample whose successor is still at or before t1; the last knot is t1, taken on the segment
 * that crosses t1. A knot closer than 1e-12 s to the next one is dropped.
 *
 * @param buffer Samples in time order
 * @param t0 Interval start [s]
 * @param t1 Interval end [s]
 * @param extrapolate_end When the buffer ends before t1, close the interval with the line through the last two samples
 * @return Knots in time order (empty when the buffer has fewer than two samples or starts after t1)
 */
std::vector<ImuSample> ImuSamplesOverInterval(const std::vector<ImuSample>& buffer, double t0, double t1,
                                              bool extrapolate_end);

}  // namespace vio

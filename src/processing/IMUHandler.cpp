/**
 * @file      IMUHandler.cpp
 * @brief     Implements IMU data preintegration and bias management
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-09-08
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 *
 * @par Mathematical Foundation
 * The IMU preintegration theory and mathematical formulations implemented in this file
 * are based on the work presented in:
 * "IMU Preintegration on Manifold for Efficient Visual-Inertial Maximum-a-Posteriori Estimation"
 * by C. Forster, L. Carlone, F. Dellaert, and D. Scaramuzza (RSS 2015)
 */

#include "processing/IMUHandler.h"
#include "processing/Optimizer.h"
#include "database/MapPoint.h"
#include "util/EurocUtils.h"
#include "util/Config.h"
#include "database/Frame.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <ceres/ceres.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace lightweight_vio {

IMUPreintegration::IMUPreintegration() {
    reset();
}

void IMUPreintegration::reset() {
    delta_R = Eigen::Matrix3f::Identity();
    delta_V = Eigen::Vector3f::Zero();
    delta_P = Eigen::Vector3f::Zero();
    
    J_Rg = Eigen::Matrix3f::Zero();
    J_Vg = Eigen::Matrix3f::Zero();
    J_Va = Eigen::Matrix3f::Zero();
    J_Pg = Eigen::Matrix3f::Zero();
    J_Pa = Eigen::Matrix3f::Zero();
    
    covariance = Eigen::Matrix<float, 15, 15>::Zero();
    
    gyro_bias = Eigen::Vector3f::Zero();
    accel_bias = Eigen::Vector3f::Zero();
    
    dt_total = 0.0;
}

bool IMUPreintegration::is_valid() const {
    return dt_total > 0.0 && !delta_R.hasNaN() && !delta_V.hasNaN() && !delta_P.hasNaN();
}

IMUHandler::IMUHandler() 
    : m_gyro_bias(Eigen::Vector3f::Zero())
    , m_accel_bias(Eigen::Vector3f::Zero())
    , m_gravity(0.0f, 0.0f, -9.81f)  // Default gravity pointing down
    , m_Rgw(Eigen::Matrix3f::Identity())
    , m_gravity_aligned(false)
    , m_initialized(false) {
    
    // Load noise parameters from config
    const auto& config = Config::getInstance();
    m_gyro_noise = static_cast<float>(config.m_gyro_noise_density);
    m_accel_noise = static_cast<float>(config.m_accel_noise_density);
    m_gyro_bias_noise = static_cast<float>(config.m_gyro_random_walk);
    m_accel_bias_noise = static_cast<float>(config.m_accel_random_walk);
    
    if (Config::getInstance().m_enable_debug_output) {
        spdlog::info("[IMU_HANDLER] Initialized with config parameters:");
        spdlog::info("  - Gyro noise: {:.6e} rad/s/√Hz", m_gyro_noise);
        spdlog::info("  - Accel noise: {:.6e} m/s²/√Hz", m_accel_noise);
        spdlog::info("  - Gyro bias noise: {:.6e} rad/s²/√Hz", m_gyro_bias_noise);
        spdlog::info("  - Accel bias noise: {:.6e} m/s³/√Hz", m_accel_bias_noise);
    }
}

void IMUHandler::reset() {
    m_gyro_bias = Eigen::Vector3f::Zero();
    m_accel_bias = Eigen::Vector3f::Zero();
    m_gravity = Eigen::Vector3f(0.0f, 0.0f, -9.81f);
    m_Rgw = Eigen::Matrix3f::Identity();
    m_gravity_aligned = false;
    m_initialized = false;
    
    spdlog::info("[IMU_HANDLER] Reset to initial state");
}

void IMUHandler::set_bias(const Eigen::Vector3f& gyro_bias, const Eigen::Vector3f& accel_bias) {
    m_gyro_bias = gyro_bias;
    m_accel_bias = accel_bias;
    
    
}

void IMUHandler::get_bias(Eigen::Vector3f& gyro_bias, Eigen::Vector3f& accel_bias) const {
    gyro_bias = m_gyro_bias;
    accel_bias = m_accel_bias;
}

std::shared_ptr<IMUPreintegration> IMUHandler::preintegrate(
    const std::vector<IMUData>& imu_measurements,
    double start_time,
    double end_time) {
    
    if (imu_measurements.empty()) {
        spdlog::warn("[IMU_HANDLER] No IMU measurements provided for preintegration");
        return nullptr;
    }
    
    auto preint = std::make_shared<IMUPreintegration>();
    preint->gyro_bias = m_gyro_bias;
    preint->accel_bias = m_accel_bias;
    
    // Filter measurements within time range
    std::vector<IMUData> filtered_measurements;
    for (const auto& imu : imu_measurements) {
        if (imu.timestamp >= start_time && imu.timestamp < end_time) {
            filtered_measurements.push_back(imu);
        }
    }
    
    if (filtered_measurements.empty()) {
        spdlog::warn("[IMU_HANDLER] No IMU measurements in time range [{:.6f}, {:.6f})", start_time, end_time);
        return nullptr;
    }
    
    // Integrate measurements WITHOUT gravity compensation (gravity is handled in optimization factors)
    bool use_gravity_compensation = false;  // Always false for proper preintegration
    
    for (size_t i = 0; i < filtered_measurements.size(); ++i) {
        float dt;
        if (i == 0) {
            dt = (filtered_measurements.size() > 1) ? 
                 static_cast<float>(filtered_measurements[1].timestamp - filtered_measurements[0].timestamp) : 0.005f;
        } else {
            dt = static_cast<float>(filtered_measurements[i].timestamp - filtered_measurements[i-1].timestamp);
        }
        dt = std::max(0.001f, std::min(dt, 0.02f));
        
        // Always use raw measurements for proper preintegration
        integrate_measurement(preint, filtered_measurements[i], dt);
        
        update_covariance(preint, dt);
        preint->dt_total += dt;
    }
    
    return preint;
}

void IMUHandler::integrate_measurement(
    std::shared_ptr<IMUPreintegration> preint,
    const IMUData& imu,
    float dt) {
    
    // Bias-corrected measurements
    Eigen::Vector3f gyro = imu.angular_vel - preint->gyro_bias;
    Eigen::Vector3f accel = imu.linear_accel - preint->accel_bias;
    
    // Current state
    Eigen::Matrix3f R = preint->delta_R;
    Eigen::Vector3f V = preint->delta_V;
    Eigen::Vector3f P = preint->delta_P;
    
    // Update rotation using Rodrigues formula
    Eigen::Vector3f omega_dt = gyro * dt;
    Eigen::Matrix3f dR = rodrigues(omega_dt);
    Eigen::Matrix3f Jr = right_jacobian(omega_dt);
    
    // Update Jacobians w.r.t bias. Must use the pre-update J_Rg/J_Vg/J_Va on the
    // right-hand side (they feed into J_Pg/J_Pa/J_Vg below), so J_Rg is updated last.
    Eigen::Matrix3f J_Rg_old = preint->J_Rg;
    Eigen::Matrix3f accel_skew = skew_symmetric(accel);

    preint->J_Pg = preint->J_Pg + preint->J_Vg * dt
                 - 0.5f * R * accel_skew * J_Rg_old * dt * dt;
    preint->J_Pa = preint->J_Pa + preint->J_Va * dt - 0.5f * R * dt * dt;

    preint->J_Vg = preint->J_Vg - R * accel_skew * J_Rg_old * dt;
    preint->J_Va = preint->J_Va - R * dt;

    preint->J_Rg = dR.transpose() * J_Rg_old - Jr * dt;

    // Update rotation
    preint->delta_R = R * dR;

    // Update velocity
    Eigen::Vector3f dV = R * accel * dt;
    preint->delta_V = V + dV;

    // Update position
    Eigen::Vector3f dP = V * dt + 0.5f * R * accel * dt * dt;
    preint->delta_P = P + dP;
}

void IMUHandler::integrate_measurement_with_gravity(
    std::shared_ptr<IMUPreintegration> preint,
    const IMUData& imu,
    float dt,
    const Eigen::Matrix3f& R_wb) {
    
    // Remove bias from measurements  
    Eigen::Vector3f gyro = imu.angular_vel - preint->gyro_bias;
    Eigen::Vector3f accel = imu.linear_accel - preint->accel_bias;
    
    // 🎯 중력 보상: body frame에서 중력 제거
    // 방정식: v_j_w - v_i_w = g_w*dt + R_wb*∫(a_compensated)dt
    // 따라서: a_compensated = a_measured - g_body, 여기서 g_body = R_wb^T * g_w
    // m_gravity = [0, 0, -9.81]이므로 실제로는 더해야 함
    Eigen::Vector3f gravity_body = R_wb.transpose() * m_gravity;
    Eigen::Vector3f accel_compensated = accel + gravity_body;

    // Current state
    Eigen::Matrix3f R = preint->delta_R;
    Eigen::Vector3f V = preint->delta_V;
    Eigen::Vector3f P = preint->delta_P;
    
    // Update rotation using Rodrigues formula
    Eigen::Vector3f omega_dt = gyro * dt;
    Eigen::Matrix3f dR = rodrigues(omega_dt);
    Eigen::Matrix3f Jr = right_jacobian(omega_dt);
    
    // Update Jacobians w.r.t bias. Must use the pre-update J_Rg/J_Vg/J_Va on the
    // right-hand side (they feed into J_Pg/J_Pa/J_Vg below), so J_Rg is updated last.
    Eigen::Matrix3f J_Rg_old = preint->J_Rg;
    Eigen::Matrix3f accel_skew = skew_symmetric(accel_compensated);

    preint->J_Pg = preint->J_Pg + preint->J_Vg * dt
                 - 0.5f * R * accel_skew * J_Rg_old * dt * dt;
    preint->J_Pa = preint->J_Pa + preint->J_Va * dt - 0.5f * R * dt * dt;

    preint->J_Vg = preint->J_Vg - R * accel_skew * J_Rg_old * dt;
    preint->J_Va = preint->J_Va - R * dt;

    preint->J_Rg = dR.transpose() * J_Rg_old - Jr * dt;

    // Update rotation
    preint->delta_R = R * dR;

    // Update velocity (중력 보상된 가속도 사용)
    Eigen::Vector3f dV = R * accel_compensated * dt;
    preint->delta_V = V + dV;

    // Update position (중력 보상된 가속도 사용)
    Eigen::Vector3f dP = V * dt + 0.5f * R * accel_compensated * dt * dt;
    preint->delta_P = P + dP;
}

void IMUHandler::update_covariance(std::shared_ptr<IMUPreintegration> preint, float dt) {
    
    // Get current bias-corrected measurements for this integration step
    // Note: This should be called DURING integration, not after
    // The current implementation assumes bias-corrected accel is available
    
    // ============================================================================
    // STEP 1: Construct noise covariance matrices (6x6)
    // ============================================================================
    
    // IMU noise covariance (gyro + accel)
    Eigen::Matrix<float, 6, 6> Nga = Eigen::Matrix<float, 6, 6>::Zero();
    Nga.block<3,3>(0,0) = Eigen::Matrix3f::Identity() * (m_gyro_noise * m_gyro_noise);  // σ²_g
    Nga.block<3,3>(3,3) = Eigen::Matrix3f::Identity() * (m_accel_noise * m_accel_noise); // σ²_a
    
    // Random walk noise (bias evolution)
    Eigen::Matrix<float, 6, 6> NgaWalk = Eigen::Matrix<float, 6, 6>::Zero();  
    NgaWalk.block<3,3>(0,0) = Eigen::Matrix3f::Identity() * (m_gyro_bias_noise * m_gyro_bias_noise * dt); // σ²_bg * dt
    NgaWalk.block<3,3>(3,3) = Eigen::Matrix3f::Identity() * (m_accel_bias_noise * m_accel_bias_noise * dt); // σ²_ba * dt
    
    // ============================================================================
    // STEP 2: Construct state transition matrix A (9x9) and noise mapping B (9x6)
    // ============================================================================
    // State vector: [δR, δV, δP] (9 DoF for position/orientation/velocity increments)
    
    Eigen::Matrix<float, 9, 9> A = Eigen::Matrix<float, 9, 9>::Identity();
    Eigen::Matrix<float, 9, 6> B = Eigen::Matrix<float, 9, 6>::Zero();
    
    // Get current rotation state
    Eigen::Matrix3f dR = preint->delta_R;
    
    // Note: In a complete implementation, we need the bias-corrected acceleration
    // For this correction, we assume the acceleration used in the integration step
    // is available. In practice, this should be computed during integrate_measurement()
    
    // Approximate coupling terms (simplified - requires acceleration from current integration)
    // A.block<3,3>(3,0) = -dR * dt * skew_symmetric(accel_corrected); // δV depends on δR  
    // A.block<3,3>(6,0) = -0.5f * dR * dt * dt * skew_symmetric(accel_corrected); // δP depends on δR
    A.block<3,3>(6,3) = Eigen::Matrix3f::Identity() * dt; // δP depends on δV
    
    // Noise mapping matrix B
    // Rotation noise (from gyroscope measurements)
    // B.block<3,3>(0,0) = right_jacobian * dt; // Requires proper right Jacobian computation
    
    // Velocity noise (from accelerometer measurements)  
    B.block<3,3>(3,3) = dR * dt;
    
    // Position noise (from accelerometer measurements)
    B.block<3,3>(6,3) = 0.5f * dR * dt * dt;
    
    // ============================================================================
    // STEP 3: Update covariance for preintegration states (9x9 block)
    // ============================================================================
    preint->covariance.block<9,9>(0,0) = A * preint->covariance.block<9,9>(0,0) * A.transpose() + B * Nga * B.transpose();
    
    // ============================================================================  
    // STEP 4: Update bias covariance (6x6 block) - simple random walk
    // ============================================================================
    preint->covariance.block<6,6>(9,9) += NgaWalk;
    
    // ============================================================================
    // NOTES for future improvement:
    // ============================================================================
    // 1. The acceleration-dependent terms in A matrix are commented out because
    //    they require the bias-corrected acceleration from the current integration step
    // 2. The right Jacobian terms in B matrix need proper computation
    // 3. For full accuracy, integrate this directly into integrate_measurement()
}

void IMUHandler::update_preintegration_with_new_bias(
    std::shared_ptr<IMUPreintegration> preint,
    const Eigen::Vector3f& new_gyro_bias,
    const Eigen::Vector3f& new_accel_bias) {
    
    if (!preint || !preint->is_valid()) {
        return;
    }
    
    // Calculate bias change (delta)
    Eigen::Vector3f delta_bg = new_gyro_bias - preint->gyro_bias;
    Eigen::Vector3f delta_ba = new_accel_bias - preint->accel_bias;
    
    // Log initial covariance values
    float old_gyro_cov_trace = preint->covariance.block<3,3>(9, 9).trace();
    float old_accel_cov_trace = preint->covariance.block<3,3>(12, 12).trace();
    
    // spdlog::debug("[IMU_HANDLER] Updating preintegration with new bias - Old gyro: ({:.6f}, {:.6f}, {:.6f}), New gyro: ({:.6f}, {:.6f}, {:.6f}), Delta: ({:.6f}, {:.6f}, {:.6f})",
    //              preint->gyro_bias.x(), preint->gyro_bias.y(), preint->gyro_bias.z(),
    //              new_gyro_bias.x(), new_gyro_bias.y(), new_gyro_bias.z(),
    //              delta_bg.x(), delta_bg.y(), delta_bg.z());
    
    // Update using Jacobians (much faster than re-integration)
    preint->delta_R = preint->delta_R * rodrigues(preint->J_Rg * delta_bg);
    preint->delta_V = preint->delta_V + preint->J_Vg * delta_bg + preint->J_Va * delta_ba;
    preint->delta_P = preint->delta_P + preint->J_Pg * delta_bg + preint->J_Pa * delta_ba;
    
    // Update covariance: When bias changes, bias uncertainty should reset to lower values
    // since optimization has provided better bias estimates
    float bias_reduction_factor = 0.9f; // New bias estimates are more accurate
    
    // Update gyro bias covariance (indices 9-11)
    preint->covariance.block<3,3>(9, 9) *= bias_reduction_factor;
    
    // Update accel bias covariance (indices 12-14)  
    preint->covariance.block<3,3>(12, 12) *= bias_reduction_factor;
    
    // Add small diagonal regularization to prevent singular covariance
    float min_bias_variance = 1e-8f;
    for (int i = 9; i < 15; ++i) {
        preint->covariance(i, i) = std::max(preint->covariance(i, i), min_bias_variance);
    }
    
    // Update bias to new values
    preint->gyro_bias = new_gyro_bias;
    preint->accel_bias = new_accel_bias;
    
    // Log covariance update results
    float new_gyro_cov_trace = preint->covariance.block<3,3>(9, 9).trace();
    float new_accel_cov_trace = preint->covariance.block<3,3>(12, 12).trace();
    
}

void IMUHandler::estimate_initial_bias(
    const std::vector<IMUData>& imu_measurements,
    float gravity_magnitude) {
    
    if (imu_measurements.empty()) {
        spdlog::warn("[IMU_HANDLER] No IMU measurements for bias estimation");
        return;
    }
    
    spdlog::info("[IMU_HANDLER] Estimating initial bias from {} measurements", imu_measurements.size());
    
    // Estimate gyroscope bias (should be zero in static conditions)
    Eigen::Vector3f gyro_sum = Eigen::Vector3f::Zero();
    Eigen::Vector3f accel_sum = Eigen::Vector3f::Zero();
    
    for (const auto& imu : imu_measurements) {
        gyro_sum += imu.angular_vel;
        accel_sum += imu.linear_accel;
    }
    
    // Average as bias estimate
    m_gyro_bias = gyro_sum / static_cast<float>(imu_measurements.size());
    
    // 🎯 STEP 1: Estimate gravity direction from accelerometer measurements
    // In static conditions, accelerometer measures only gravity (+ bias)
    Eigen::Vector3f accel_mean = accel_sum / static_cast<float>(imu_measurements.size());
    
    // 🎯 STEP 2: Gravity direction = average accelerometer reading direction  
    Eigen::Vector3f gravity_direction = accel_mean.normalized();
    float measured_gravity_mag = accel_mean.norm();
    
    // 🎯 STEP 3: Update gravity vector (pointing opposite to measured acceleration)
    m_gravity = -gravity_direction * gravity_magnitude;
    
    // 🎯 STEP 4: Accelerometer bias = measured - expected gravity
    Eigen::Vector3f expected_accel = -m_gravity;  // Expected acceleration due to gravity
    m_accel_bias = accel_mean - expected_accel;
    
    m_initialized = true;
    
    spdlog::info("[IMU_HANDLER] ✅ Initial bias estimated:");
    spdlog::info("  - Gravity direction: ({:.6f}, {:.6f}, {:.6f})", 
                 gravity_direction.x(), gravity_direction.y(), gravity_direction.z());
    spdlog::info("  - Measured gravity magnitude: {:.6f} m/s²", measured_gravity_mag);
    spdlog::info("  - Gyro bias: ({:.6f}, {:.6f}, {:.6f})", 
                 m_gyro_bias.x(), m_gyro_bias.y(), m_gyro_bias.z());
    spdlog::info("  - Accel bias: ({:.6f}, {:.6f}, {:.6f})", 
                 m_accel_bias.x(), m_accel_bias.y(), m_accel_bias.z());
}
bool IMUHandler::estimate_gravity_with_stereo_constraints(
    const std::vector<Frame*>& frames,
    const std::vector<IMUData>& all_imu_data,
    float gravity_magnitude,
    double* initial_cost,
    double* final_cost) {
    
    if (frames.size() < 3) {
        spdlog::warn("[IMU_HANDLER] Need at least 3 frames for gravity estimation");
        return false;
    }
    
    if (Config::getInstance().m_enable_debug_output) {
        spdlog::info("[IMU_HANDLER] 🎯 Starting gravity estimation with {} frames", frames.size());
    }
    
    // Step 1: Create preintegrations (without gravity compensation first)
    std::vector<std::shared_ptr<IMUPreintegration>> preintegrations;
    for (size_t i = 0; i < frames.size() - 1; i++) {
        double start_time = static_cast<double>(frames[i]->get_timestamp()) / 1e9;
        double end_time = static_cast<double>(frames[i + 1]->get_timestamp()) / 1e9;
        
        auto preint = preintegrate(all_imu_data, start_time, end_time);
        if (!preint) continue;
        preintegrations.push_back(preint);
    }
    
    const int N = frames.size();
    const int M = preintegrations.size();
    
    // Step 2: Setup linear system for gravity estimation
    // Variables: [g_x, g_y, g_z, v0_x, v0_y, v0_z, ..., v_{N-1}_x, v_{N-1}_y, v_{N-1}_z]
    const int num_unknowns = 3 + 3 * N;
    const int num_equations = 3 * M + 3 * N;
    
    Eigen::MatrixXf A = Eigen::MatrixXf::Zero(num_equations, num_unknowns);
    Eigen::VectorXf b = Eigen::VectorXf::Zero(num_equations);
    int eq_idx = 0;
    
    // IMU constraints based on the equation: v_{i+1} - v_i - g*dt = R_wb_i * delta_V
    for (int i = 0; i < M; i++) {
        auto preint = preintegrations[i];
        Eigen::Matrix3f R_wb = frames[i]->get_Twb().block<3,3>(0,0);
        float dt = preint->dt_total;
        Eigen::Vector3f imu_vel = R_wb * preint->delta_V;

        for (int axis = 0; axis < 3; axis++) {
            // [MODIFIED] The coefficient for the gravity term should be -dt, based on the equation v_{i+1} - v_i - g*dt = ...
            A(eq_idx + axis, axis) = -dt;
            // v_i coefficient: -1
            A(eq_idx + axis, 3 + 3*i + axis) = -1.0f;
            // v_{i+1} coefficient: +1  
            A(eq_idx + axis, 3 + 3*(i+1) + axis) = 1.0f;
            // RHS: IMU velocity change
            b(eq_idx + axis) = imu_vel(axis);
        }
        eq_idx += 3;
    }
    
    // Stereo visual constraints: smooth velocity assumption (regularization)
    const float stereo_weight = 100000.0f;  // A stronger constraint
    for (int i = 0; i < N; i++) {
        Eigen::Vector3f stereo_vel = Eigen::Vector3f::Zero();
        
        // Estimate velocity from neighboring frames
        if (i > 0 && i < N-1 && i-1 < M && i < M) {
            // Central difference for intermediate frames
            Eigen::Matrix4f T_prev = frames[i-1]->get_Twb();
            Eigen::Matrix4f T_next = frames[i+1]->get_Twb();
            Eigen::Vector3f pos_diff = T_next.block<3,1>(0,3) - T_prev.block<3,1>(0,3);
            float dt_total = preintegrations[i-1]->dt_total + (i < M ? preintegrations[i]->dt_total : 0.0f);
            if (dt_total > 0.0f) {
                stereo_vel = pos_diff / dt_total;
            }
        } else if (i == 0 && N > 1) {
            // Forward difference for the first frame
            Eigen::Matrix4f T_curr = frames[0]->get_Twb();
            Eigen::Matrix4f T_next = frames[1]->get_Twb();
            Eigen::Vector3f pos_diff = T_next.block<3,1>(0,3) - T_curr.block<3,1>(0,3);
            if (M > 0) {
                stereo_vel = pos_diff / preintegrations[0]->dt_total;
            }
        } else if (i == N-1 && N > 1 && M > 0) {
            // Backward difference for the last frame
            Eigen::Matrix4f T_prev = frames[N-2]->get_Twb();
            Eigen::Matrix4f T_curr = frames[N-1]->get_Twb();
            Eigen::Vector3f pos_diff = T_curr.block<3,1>(0,3) - T_prev.block<3,1>(0,3);
            stereo_vel = pos_diff / preintegrations[M-1]->dt_total;
        }
        
        for (int axis = 0; axis < 3; axis++) {
            A(eq_idx + axis, 3 + 3*i + axis) = stereo_weight;
            b(eq_idx + axis) = stereo_weight * stereo_vel(axis);
        }
        eq_idx += 3;
    }
    
    // Calculate initial cost (before optimization) using zero gravity assumption
    if (initial_cost != nullptr) {
        Eigen::VectorXf zero_solution = Eigen::VectorXf::Zero(num_unknowns);
        Eigen::VectorXf residuals = A * zero_solution - b;
        *initial_cost = static_cast<double>(residuals.squaredNorm());
    }
    
    // Step 3: Solve the linear system using least squares
    Eigen::MatrixXf AtA = A.transpose() * A;
    Eigen::VectorXf Atb = A.transpose() * b;
    // Add damping (Tikhonov regularization) to ensure the matrix is invertible
    AtA += 1e-6f * Eigen::MatrixXf::Identity(num_unknowns, num_unknowns);
    
    Eigen::VectorXf solution = AtA.lu().solve(Atb);
    Eigen::Vector3f estimated_gravity = solution.segment<3>(0);
    
    // Calculate final cost (after optimization)
    if (final_cost != nullptr) {
        Eigen::VectorXf residuals = A * solution - b;
        *final_cost = static_cast<double>(residuals.squaredNorm());
    }
    
    // Step 4: Normalize the estimated gravity and apply the known magnitude
    if (estimated_gravity.norm() < 1.0f) {
        return false;
    }
    
    m_gravity = estimated_gravity.normalized() * gravity_magnitude;
    m_initialized = true;
    
    // 🎯 Calculate Rgw (World-to-Gravity transformation matrix)
    Eigen::Vector3f gravity_direction = m_gravity.normalized();  // The current estimated gravity direction
    Eigen::Vector3f gravity_ideal(0.0f, 0.0f, -1.0f);          // The ideal gravity direction (downward Z-axis)
    
    // [MODIFIED] The rotation axis must be from the start_vector to the target_vector.
    // This computes the rotation that transforms `gravity_direction` to align with `gravity_ideal`.
    Eigen::Vector3f rotation_axis = gravity_direction.cross(gravity_ideal);
    float rotation_axis_norm = rotation_axis.norm();
    float cos_angle = gravity_ideal.dot(gravity_direction);
    
    Eigen::Matrix3f Rgw;
    if (rotation_axis_norm < 1e-6f) {
        // Case 1: Vectors are nearly parallel or anti-parallel
        if (cos_angle > 0.0f) {
            // Already aligned
            Rgw = Eigen::Matrix3f::Identity();
        } else {
            // Anti-parallel (180-degree rotation needed). Choose an arbitrary perpendicular axis.
            Eigen::Vector3f perp_axis(1.0f, 0.0f, 0.0f);
            if (std::abs(gravity_direction.x()) > 0.9f) {
                perp_axis = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
            }
            Rgw = rodrigues(perp_axis * M_PI);
        }
    } else {
        // Case 2: General rotation using Rodrigues' formula
        float angle = std::acos(std::clamp(cos_angle, -1.0f, 1.0f));
        Eigen::Vector3f normalized_axis = rotation_axis / rotation_axis_norm;
        Eigen::Vector3f rotation_vector = normalized_axis * angle;
        Rgw = rodrigues(rotation_vector);
    }
    
    // 🎯 Normalize the rotation matrix using SVD to ensure it is a proper orthogonal matrix
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(Rgw, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f U = svd.matrixU();
    Eigen::Matrix3f V = svd.matrixV();
    
    // Ensure it's a proper rotation matrix (determinant = +1)
    Eigen::Matrix3f R_candidate = U * V.transpose();
    if (R_candidate.determinant() < 0) {
        U.col(2) *= -1.0f;
        R_candidate = U * V.transpose();
    }
    
    m_Rgw = R_candidate;
    m_gravity_aligned = true;

    if (Config::getInstance().m_enable_debug_output)
    {
        spdlog::info("[IMU_HANDLER] ✅ Gravity estimation completed:");
        spdlog::info("  Estimated gravity: ({:.4f}, {:.4f}, {:.4f}) m/s²",
                     estimated_gravity.x(), estimated_gravity.y(), estimated_gravity.z());
        spdlog::info("  Final gravity: ({:.4f}, {:.4f}, {:.4f}) m/s²",
                     m_gravity.x(), m_gravity.y(), m_gravity.z());
        spdlog::info("  Gravity direction: ({:.4f}, {:.4f}, {:.4f})",
                     gravity_direction.x(), gravity_direction.y(), gravity_direction.z());
        spdlog::info("  Gravity magnitude: {:.4f} m/s²", m_gravity.norm());
        spdlog::info("  Rgw (SVD-normalized World-to-Gravity rotation matrix):");
        spdlog::info("    [{:.6f}, {:.6f}, {:.6f}]", m_Rgw(0, 0), m_Rgw(0, 1), m_Rgw(0, 2));
        spdlog::info("    [{:.6f}, {:.6f}, {:.6f}]", m_Rgw(1, 0), m_Rgw(1, 1), m_Rgw(1, 2));
        spdlog::info("    [{:.6f}, {:.6f}, {:.6f}]", m_Rgw(2, 0), m_Rgw(2, 1), m_Rgw(2, 2));
        spdlog::info("  Matrix properties - det(Rgw): {:.6f}, orthogonality error: {:.2e}",
                     m_Rgw.determinant(), (m_Rgw * m_Rgw.transpose() - Eigen::Matrix3f::Identity()).norm());

        // Verify the transformation by applying it to the original gravity direction
        Eigen::Vector3f transformed_gravity = m_Rgw * gravity_direction;
        spdlog::info("  Verification - Transformed gravity should be close to [0, 0, -1]: ({:.4f}, {:.4f}, {:.4f})",
                     transformed_gravity.x(), transformed_gravity.y(), transformed_gravity.z());
    }

    return true;
}

bool IMUHandler::debug_velocity_comparison(
    const std::vector<Frame*>& frames,
    const std::vector<IMUData>& all_imu_data) {
    
    if (frames.size() < 2) {
        spdlog::warn("[IMU_HANDLER] Need at least 2 frames for velocity comparison");
        return false;
    }
    
    if (!m_initialized || m_gravity.norm() < 9.0f) {
        spdlog::warn("[IMU_HANDLER] Gravity not estimated, run gravity estimation first");
        return false;
    }
    
    
    float total_error = 0.0f;
    int count = 0;
    
    for (size_t i = 0; i < std::min(frames.size() - 1, size_t(5)); ++i) {
        Frame* frame1 = frames[i];
        Frame* frame2 = frames[i + 1];
        
        double start_time = static_cast<double>(frame1->get_timestamp()) / 1e9;
        double end_time = static_cast<double>(frame2->get_timestamp()) / 1e9;
        float dt = static_cast<float>(end_time - start_time);
        
        if (dt <= 0.001f) continue;
        
        // Stereo velocity
        
        // IMU velocity (raw preintegration without gravity compensation)
        auto preint = preintegrate(all_imu_data, start_time, end_time);
        if (!preint) continue;
        
        count++;
    }
    
    if (count > 0) {
        float avg_error = total_error / count;
        return avg_error < 0.5f;  // Reasonable threshold
    }
    
    return false;
}



// ==================== HELPER FUNCTIONS ====================

Eigen::Matrix3f IMUHandler::rodrigues(const Eigen::Vector3f& w) const {
    float theta = w.norm();
    if (theta < 1e-6f) {
        return Eigen::Matrix3f::Identity() + skew_symmetric(w);
    }
    
    Eigen::Vector3f axis = w / theta;
    float c = std::cos(theta);
    float s = std::sin(theta);
    
    return c * Eigen::Matrix3f::Identity() + 
           s * skew_symmetric(axis) + 
           (1.0f - c) * axis * axis.transpose();
}

Eigen::Matrix3f IMUHandler::right_jacobian(const Eigen::Vector3f& w) const {
    float theta = w.norm();
    if (theta < 1e-6f) {
        return Eigen::Matrix3f::Identity() - 0.5f * skew_symmetric(w);
    }
    
    Eigen::Vector3f axis = w / theta;
    float c = std::cos(theta);
    float s = std::sin(theta);
    
    return s / theta * Eigen::Matrix3f::Identity() + 
           (1.0f - c) / theta * skew_symmetric(axis) + 
           (theta - s) / theta * axis * axis.transpose();
}

Eigen::Matrix3f IMUHandler::skew_symmetric(const Eigen::Vector3f& v) const {
    Eigen::Matrix3f skew = Eigen::Matrix3f::Zero();
    skew(0, 1) = -v(2);
    skew(0, 2) =  v(1);
    skew(1, 0) =  v(2);
    skew(1, 2) = -v(0);
    skew(2, 0) = -v(1);
    skew(2, 1) =  v(0);
    return skew;
}

bool IMUHandler::transform_to_gravity_frame(
    const std::vector<std::shared_ptr<Frame>>& keyframes,
    const std::vector<std::shared_ptr<MapPoint>>& map_points,
    Eigen::Matrix4f& T_gw) {


    // if (Config::getInstance().m_enable_debug_output) 
    {
        spdlog::info("================================================================================");
        spdlog::info("[TRANSFORM] 🔄 Applying gravity frame transformation");
        spdlog::info("[TRANSFORM] Input Tgw (World → Gravity-aligned):");
        spdlog::info("[TRANSFORM]   Row 1: [{:.6f}, {:.6f}, {:.6f}, {:.6f}]", 
                     T_gw(0,0), T_gw(0,1), T_gw(0,2), T_gw(0,3));
        spdlog::info("[TRANSFORM]   Row 2: [{:.6f}, {:.6f}, {:.6f}, {:.6f}]", 
                     T_gw(1,0), T_gw(1,1), T_gw(1,2), T_gw(1,3));
        spdlog::info("[TRANSFORM]   Row 3: [{:.6f}, {:.6f}, {:.6f}, {:.6f}]", 
                     T_gw(2,0), T_gw(2,1), T_gw(2,2), T_gw(2,3));
        spdlog::info("[TRANSFORM]   Row 4: [{:.6f}, {:.6f}, {:.6f}, {:.6f}]", 
                     T_gw(3,0), T_gw(3,1), T_gw(3,2), T_gw(3,3));
        spdlog::info("================================================================================");
    }

    if (!m_gravity_aligned) {
        spdlog::error("[IMU_HANDLER] Gravity alignment not computed, call estimate_gravity_with_stereo_constraints first");
        return false;
    }
    
    if (keyframes.empty()) {
        spdlog::warn("[IMU_HANDLER] No keyframes to transform");
        return false;
    }
    
    if (Config::getInstance().m_enable_debug_output) {
        spdlog::info("[IMU_HANDLER] 🌍 Transforming {} keyframes and {} map points to gravity-aligned frame", 
                     keyframes.size(), map_points.size());
    }
    
    // ============================================================================
    // 1. Transform all keyframes
    // ============================================================================
    for (const auto& frame : keyframes) {
        if (!frame) continue;
        
        // Get current pose T_wb (world-to-body)
        Eigen::Matrix4f T_wb = frame->get_Twb();
        Eigen::Matrix4f T_gb = T_gw * T_wb;
        
        // Set transformed pose
        frame->set_Twb(T_gb);
        
        // Transform velocity to gravity frame
        Eigen::Vector3f velocity_world = frame->get_velocity();
        Eigen::Vector3f velocity_gravity = m_Rgw * velocity_world;
        frame->set_velocity(velocity_gravity);
    }

    
    // ============================================================================
    // 2. Transform all map points
    // ============================================================================
    int transformed_count = 0;
    for (const auto& map_point : map_points) {
        if (!map_point) continue;
        
        // Get current 3D position in world frame
        Eigen::Vector3f pos_world = map_point->get_position();
        
        // Transform to gravity frame: pos_gravity = Rgw * pos_world
        Eigen::Vector3f pos_gravity = m_Rgw * pos_world;
        
        // Set transformed position
        map_point->set_position(pos_gravity);
        transformed_count++;
    }
    
    if (Config::getInstance().m_enable_debug_output) {
        spdlog::info("[IMU_HANDLER] ✅ Successfully transformed to gravity frame:");
        spdlog::info("[IMU_HANDLER]   - {} keyframes", keyframes.size());
        spdlog::info("[IMU_HANDLER]   - {} map points", transformed_count);
        spdlog::info("[IMU_HANDLER] Gravity vector in new frame: ({:.4f}, {:.4f}, {:.4f}) m/s²", 
                     0.0f, 0.0f, -m_gravity.norm());
    }

    // ============================================================================
    // 🎯 VERIFICATION: Check if gravity is really aligned using raw IMU data
    // ============================================================================
    if (Config::getInstance().m_enable_debug_output) {
        verify_gravity_alignment_with_imu_data(keyframes);
    }
    
    return true;
}


bool IMUHandler::apply_optimization_results(
    const std::vector<Frame*>& optimized_keyframes,
    const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& optimized_biases,
    const std::vector<Eigen::Vector3f>& optimized_velocities) {
    
    if (optimized_keyframes.size() != optimized_biases.size() || 
        optimized_keyframes.size() != optimized_velocities.size()) {
        spdlog::error("[IMU_HANDLER] Size mismatch: keyframes={}, biases={}, velocities={}", 
                     optimized_keyframes.size(), optimized_biases.size(), optimized_velocities.size());
        return false;
    }
    
    spdlog::info("[IMU_HANDLER] 🎯 Applying optimization results to {} keyframes", optimized_keyframes.size());
    
    // Apply results to each keyframe
    for (size_t i = 0; i < optimized_keyframes.size(); ++i) {
        Frame* frame = optimized_keyframes[i];
        if (!frame) continue;
        
        const auto& bias_pair = optimized_biases[i];
        const Eigen::Vector3f& gyro_bias = bias_pair.first;
        const Eigen::Vector3f& accel_bias = bias_pair.second;
        const Eigen::Vector3f& velocity = optimized_velocities[i];
        
        // Set optimized values
        frame->set_gyro_bias(gyro_bias);
        frame->set_accel_bias(accel_bias);
        frame->set_velocity(velocity);
        
        // Frame optimization applied (reduced logging)
    }
    
    // Update IMUHandler's global bias with latest optimized values (summary only)
    if (!optimized_keyframes.empty() && !optimized_biases.empty()) {
        const auto& latest_bias = optimized_biases.back();
        m_gyro_bias = latest_bias.first;
        m_accel_bias = latest_bias.second;
        
        spdlog::info("[IMU_HANDLER] ✅ Updated global bias from optimization");
    }
    
    return true;
}

bool IMUHandler::initialize_first_keyframe_from_second(
    Frame* first_keyframe,
    const Frame* second_keyframe) {
    
    if (!first_keyframe || !second_keyframe) {
        spdlog::error("[IMU_HANDLER] Null keyframe provided for initialization");
        return false;
    }
    
    // Copy bias and velocity from second keyframe to first
    first_keyframe->set_gyro_bias(second_keyframe->get_gyro_bias());
    first_keyframe->set_accel_bias(second_keyframe->get_accel_bias());
    first_keyframe->set_velocity(second_keyframe->get_velocity());
    
    return true;
}

bool IMUHandler::inherit_bias_from_keyframe(
    Frame* new_frame,
    const Frame* reference_keyframe) {
    
    if (!new_frame || !reference_keyframe) {
        spdlog::error("[IMU_HANDLER] Null frame provided for bias inheritance");
        return false;
    }
    
    // Copy bias from reference keyframe
    new_frame->set_gyro_bias(reference_keyframe->get_gyro_bias());
    new_frame->set_accel_bias(reference_keyframe->get_accel_bias());
    
    // Update IMUHandler's global bias as well
    m_gyro_bias = reference_keyframe->get_gyro_bias();
    m_accel_bias = reference_keyframe->get_accel_bias();
    
    return true;
}

void IMUHandler::compute_rgw_transformation() {
    if (m_gravity.norm() < 1.0f) {
        return;
    }
    
    // 🎯 Calculate Rgw (World-to-Gravity transformation matrix)
    // =====================================================================
    // Purpose: Transform World coordinate system to Gravity-aligned coordinate system
    // Notation: Rab = transformation from frame b to frame a
    // Rgw = World → Gravity transformation (transforms World frame vectors to Gravity frame)
    // 
    // Algorithm: Calculate rotation matrix that aligns current gravity direction with ideal [0,0,-1]
    // =====================================================================
    
    Eigen::Vector3f gravity_direction = m_gravity.normalized();  // Current estimated gravity direction (normalized)
    Eigen::Vector3f gravity_ideal(0.0f, 0.0f, -1.0f);          // Ideal gravity direction (downward Z-axis)
    
    // Calculate rotation axis and angle for Rodrigues formula
    Eigen::Vector3f rotation_axis = gravity_ideal.cross(gravity_direction);  // Rotation axis = ideal × current
    float rotation_axis_norm = rotation_axis.norm();                         // Rotation axis magnitude (proportional to sin(θ))
    float cos_angle = gravity_ideal.dot(gravity_direction);                  // cos(θ) = dot product of unit vectors
    
    if (rotation_axis_norm < 1e-6f) {
        // Case 1: Gravity vectors are nearly parallel (aligned or anti-aligned)
        if (cos_angle > 0.0f) {
            // Gravity already aligned with ideal direction
            m_Rgw = Eigen::Matrix3f::Identity();
        } else {
            // 180 degree rotation needed - choose any perpendicular axis
            Eigen::Vector3f perp_axis(1.0f, 0.0f, 0.0f);  // Default to X-axis
            if (std::abs(gravity_direction.x()) > 0.9f) {
                perp_axis = Eigen::Vector3f(0.0f, 1.0f, 0.0f);  // Use Y-axis if X is nearly parallel
            }
            m_Rgw = rodrigues(perp_axis * M_PI);  // 180 degree rotation around perpendicular axis
        }
    } else {
        // Case 2: General rotation using Rodrigues formula
        // Calculate rotation angle and apply axis-angle rotation
        float angle = std::acos(std::clamp(cos_angle, -1.0f, 1.0f));  // Clamp to handle numerical errors
        Eigen::Vector3f normalized_axis = rotation_axis / rotation_axis_norm;
        Eigen::Vector3f rotation_vector = normalized_axis * angle;
        m_Rgw = rodrigues(rotation_vector); // Convert axis-angle to rotation matrix
    }

    m_gravity_aligned = true; // Mark that we have computed the transformation

    if (Config::getInstance().m_enable_debug_output)
    {
        spdlog::info("[IMU_HANDLER] ✅ Computed Rgw transformation matrix:");
        spdlog::info("  Current gravity: ({:.4f}, {:.4f}, {:.4f}) m/s²",
                     m_gravity.x(), m_gravity.y(), m_gravity.z());
        spdlog::info("  Gravity direction: ({:.4f}, {:.4f}, {:.4f})",
                     gravity_direction.x(), gravity_direction.y(), gravity_direction.z());
        spdlog::info("  Rgw matrix:");
        spdlog::info("    [{:.6f}, {:.6f}, {:.6f}]", m_Rgw(0, 0), m_Rgw(0, 1), m_Rgw(0, 2));
        spdlog::info("    [{:.6f}, {:.6f}, {:.6f}]", m_Rgw(1, 0), m_Rgw(1, 1), m_Rgw(1, 2));
        spdlog::info("    [{:.6f}, {:.6f}, {:.6f}]", m_Rgw(2, 0), m_Rgw(2, 1), m_Rgw(2, 2));

        // Verify the transformation
        Eigen::Vector3f transformed_gravity = m_Rgw * gravity_direction;
        spdlog::info("  Verification - Transformed gravity: ({:.4f}, {:.4f}, {:.4f})",
                     transformed_gravity.x(), transformed_gravity.y(), transformed_gravity.z());
    }
}

Eigen::Vector3f IMUHandler::get_gravity_compensated_velocity(
    const Frame* frame,
    float dt) const {
    
    if (!frame || dt <= 0.0f) {
        return Eigen::Vector3f::Zero();
    }
    
    // Get current velocity (may include gravity effects)
    Eigen::Vector3f velocity = frame->get_velocity();
    
    // In gravity-aligned frame, gravity is [0, 0, -9.81]
    // Remove gravity effects: v_real = v_measured - g * dt
    // But since our velocity is already preintegrated and optimized, 
    // we don't need to subtract gravity*dt here
    
    // The velocity from optimization should already be the real motion velocity
    // because the InertialGravityFactor handles gravity properly
    
    return velocity;  // Already gravity-compensated from optimization
}

bool IMUHandler::update_preintegrations_with_optimized_bias(
    const std::vector<Frame*>& frames,
    const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& optimized_biases) {
    
    if (frames.size() != optimized_biases.size()) {
        spdlog::error("[IMU_HANDLER] Size mismatch: frames={}, biases={}", 
                     frames.size(), optimized_biases.size());
        return false;
    }
    
    int updated_count = 0;
    for (size_t i = 0; i < frames.size(); ++i) {
        Frame* frame = frames[i];
        if (!frame) continue;
        
        // Try to get preintegration data from different sources
        auto preint = frame->get_imu_preintegration_from_last_keyframe();
        if (!preint) {
            preint = frame->get_imu_preintegration_from_last_frame();
        }
        
        if (!preint) {
            spdlog::debug("[IMU_HANDLER] Frame {}: No preintegration data available", frame->get_frame_id());
            continue;
        }
        
        const auto& bias_pair = optimized_biases[i];
        const Eigen::Vector3f& new_gyro_bias = bias_pair.first;
        const Eigen::Vector3f& new_accel_bias = bias_pair.second;
        
        // Update preintegration with new bias using Jacobians (fast method)
        update_preintegration_with_new_bias(preint, new_gyro_bias, new_accel_bias);
        
        updated_count++;
        
        // Preintegration updated (reduced logging)
        updated_count++;
    }
    
    return updated_count > 0;
}

void IMUHandler::set_gravity_aligned_coordinate_system() {
    if (!m_gravity_aligned) {
        spdlog::warn("[IMU_HANDLER] Gravity alignment not yet computed, cannot set aligned coordinate system");
        return;
    }
    
    // After transforming to gravity-aligned frame, set Rgw to Identity
    // This indicates we are now working in the gravity-aligned coordinate system
    m_Rgw = Eigen::Matrix3f::Identity();
    
    // Update gravity vector to point in standard downward direction
    m_gravity = Eigen::Vector3f(0.0f, 0.0f, -9.81f);
    
    if (Config::getInstance().m_enable_debug_output) {
        spdlog::info("[IMU_HANDLER] ✅ Set coordinate system to gravity-aligned frame");
        spdlog::info("  - Rgw is now Identity (no further transformation needed)");
        spdlog::info("  - Gravity vector: (0.0, 0.0, -9.81) m/s²");
        std::cout<<"\n";
    }
}

bool IMUHandler::verify_gravity_alignment_with_imu_data(
    const std::vector<std::shared_ptr<Frame>>& keyframes) {
    
    if (keyframes.empty()) {
        spdlog::warn("[IMU_HANDLER] No keyframes provided for verification");
        return false;
    }
    
    spdlog::info("[IMU_HANDLER] 🔍 Verifying gravity alignment using raw IMU data...");
    
    // Collect all IMU measurements from keyframes
    std::vector<Eigen::Vector3f> world_accelerations;
    int total_imu_samples = 0;
    int keyframes_with_imu = 0;
    
    for (const auto& frame : keyframes) {
        if (!frame) continue;
        
        // Get IMU data since last keyframe
        const auto& imu_data = frame->get_imu_data_since_last_keyframe();
        if (imu_data.empty()) continue;
        
        keyframes_with_imu++;
        
        // Get frame's body-to-world rotation (after gravity alignment)
        Eigen::Matrix4f T_gb = frame->get_Twb();
        Eigen::Matrix3f R_gb = T_gb.block<3,3>(0,0);
        
        // Get frame's bias
        Eigen::Vector3f gyro_bias = frame->get_gyro_bias();
        Eigen::Vector3f accel_bias = frame->get_accel_bias();
        
        // Transform each IMU acceleration to world frame
        for (const auto& imu : imu_data) {
            // Remove bias from body frame acceleration
            Eigen::Vector3f accel_body = imu.linear_accel - accel_bias;
            
            // Transform to world (gravity-aligned) frame
            Eigen::Vector3f accel_world = R_gb * accel_body;
            
            world_accelerations.push_back(accel_world);
            total_imu_samples++;
        }
    }
    
    if (world_accelerations.empty()) {
        spdlog::warn("[IMU_HANDLER] No IMU data available for verification");
        return false;
    }
    
    // Compute statistics
    Eigen::Vector3f mean_accel = Eigen::Vector3f::Zero();
    for (const auto& accel : world_accelerations) {
        mean_accel += accel;
    }
    mean_accel /= static_cast<float>(world_accelerations.size());
    
    // Compute standard deviation
    Eigen::Vector3f variance = Eigen::Vector3f::Zero();
    for (const auto& accel : world_accelerations) {
        Eigen::Vector3f diff = accel - mean_accel;
        variance += diff.cwiseProduct(diff);
    }
    variance /= static_cast<float>(world_accelerations.size());
    Eigen::Vector3f std_dev = variance.cwiseSqrt();
    
    // Expected gravity in gravity-aligned frame
    Eigen::Vector3f expected_gravity(0.0f, 0.0f, -9.81f);
    
    // Compute error
    Eigen::Vector3f error = mean_accel - expected_gravity;
    float error_norm = error.norm();
    float error_percentage = (error_norm / 9.81f) * 100.0f;
    
    // Log results
    spdlog::info("[IMU_HANDLER] ═══════════════════════════════════════════════════════");
    spdlog::info("[IMU_HANDLER]   GRAVITY ALIGNMENT VERIFICATION RESULTS");
    spdlog::info("[IMU_HANDLER] ═══════════════════════════════════════════════════════");
    spdlog::info("[IMU_HANDLER] Total keyframes: {}, Keyframes with IMU: {}", 
                 keyframes.size(), keyframes_with_imu);
    spdlog::info("[IMU_HANDLER] Total IMU samples analyzed: {}", total_imu_samples);
    spdlog::info("[IMU_HANDLER] ───────────────────────────────────────────────────────");
    spdlog::info("[IMU_HANDLER] Mean acceleration (world frame):");
    spdlog::info("[IMU_HANDLER]   X: {:.4f} m/s² (σ={:.4f})", mean_accel.x(), std_dev.x());
    spdlog::info("[IMU_HANDLER]   Y: {:.4f} m/s² (σ={:.4f})", mean_accel.y(), std_dev.y());
    spdlog::info("[IMU_HANDLER]   Z: {:.4f} m/s² (σ={:.4f})", mean_accel.z(), std_dev.z());
    spdlog::info("[IMU_HANDLER] ───────────────────────────────────────────────────────");
    spdlog::info("[IMU_HANDLER] Expected gravity: (0.0000, 0.0000, -9.8100) m/s²");
    spdlog::info("[IMU_HANDLER] ───────────────────────────────────────────────────────");
    spdlog::info("[IMU_HANDLER] Error from expected:");
    spdlog::info("[IMU_HANDLER]   ΔX: {:.4f} m/s²", error.x());
    spdlog::info("[IMU_HANDLER]   ΔY: {:.4f} m/s²", error.y());
    spdlog::info("[IMU_HANDLER]   ΔZ: {:.4f} m/s²", error.z());
    spdlog::info("[IMU_HANDLER]   ||error||: {:.4f} m/s² ({:.2f}%)", error_norm, error_percentage);
    spdlog::info("[IMU_HANDLER] ───────────────────────────────────────────────────────");
    
    // Determine if alignment is good (threshold: 5% error)
    bool is_aligned = error_percentage < 5.0f;
    
    if (is_aligned) {
        spdlog::info("[IMU_HANDLER] ✅ PASSED: Gravity is properly aligned!");
        spdlog::info("[IMU_HANDLER]    Error within acceptable range (<5%)");
    } else {
        spdlog::warn("[IMU_HANDLER] ⚠️  WARNING: Gravity alignment error is high (>{:.1f}%)", error_percentage);
        spdlog::warn("[IMU_HANDLER]    This may indicate:");
        spdlog::warn("[IMU_HANDLER]      1. Sensor is moving during initialization");
        spdlog::warn("[IMU_HANDLER]      2. Incorrect camera extrinsics (T_BC)");
        spdlog::warn("[IMU_HANDLER]      3. Bias estimation issues");
    }
    
    spdlog::info("[IMU_HANDLER] ═══════════════════════════════════════════════════════");
    std::cout << "\n";
    
    return is_aligned;
}


} // namespace lightweight_vio

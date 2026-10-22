#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Seungwon Choi. All rights reserved.
"""Finite-difference checks for the estimator's inertial and equirectangular pose Jacobians."""

import numpy as np


def skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def exp_so3(v):
    angle = np.linalg.norm(v)
    k = skew(v)
    return np.eye(3) + (np.sin(angle) / angle) * k + ((1 - np.cos(angle)) / angle**2) * k @ k


def log_so3(r):
    return np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / 2


def propagate(state, gyro, accel, gravity, dt):
    r, p, v, bg, ba = state
    w, a = gyro - bg, accel - ba
    world_accel = r @ exp_so3(w * dt / 2) @ a + gravity
    return (r @ exp_so3(w * dt), p + v * dt + world_accel * dt * dt / 2,
            v + world_accel * dt, bg.copy(), ba.copy())


def test_inertial():
    r = exp_so3(np.array([0.3, -0.2, 0.1]))
    state = (r, np.array([1., 2., 3.]), np.array([0.4, -0.1, 0.2]),
             np.array([0.01, -0.02, 0.005]), np.array([0.1, -0.05, 0.02]))
    gyro, accel, gravity = np.array([0.2, -0.3, 0.1]), np.array([0.4, -9.4, 0.2]), np.array([0., 0., -9.805])
    dt, eps = 0.001, 1e-6
    w, a = gyro - state[3], accel - state[4]
    r_mid = r @ exp_so3(w * dt / 2)
    f = np.zeros((15, 15))
    f[0:3, 0:3], f[0:3, 9:12] = -skew(w), -np.eye(3)
    f[3:6, 6:9] = np.eye(3)
    f[6:9, 0:3], f[6:9, 12:15] = -r_mid @ skew(a), -r_mid
    transition = np.eye(15) + f * dt + f @ f * dt * dt / 2
    base = propagate(state, gyro, accel, gravity, dt)
    numeric = np.zeros((15, 15))
    for j in range(15):
        perturbed = [item.copy() for item in state]
        if j < 3:
            perturbed[0] = r @ exp_so3(np.eye(3)[j] * eps)
        else:
            block, component = divmod(j - 3, 3)
            perturbed[block + 1][component] += eps
        after = propagate(perturbed, gyro, accel, gravity, dt)
        numeric[:, j] = np.concatenate((log_so3(base[0].T @ after[0]),
                                        *(after[k] - base[k] for k in range(1, 5)))) / eps
    error = np.max(np.abs(transition - numeric))
    assert error < 2e-5, error
    return error


def project(r, p, point, camera_rotation, camera_offset, width, height):
    x, y, z = camera_rotation.T @ (r.T @ (point - p) - camera_offset)
    return np.array([width * (0.5 + np.arctan2(x, z) / (2 * np.pi)),
                     height * (0.5 + np.arcsin(y / np.sqrt(x*x + y*y + z*z)) / np.pi)])


def test_observation():
    r = exp_so3(np.array([0.2, -0.3, 0.1]))
    camera_rotation = exp_so3(np.array([0.01, 0.02, -0.03]))
    p, point, offset = np.array([1., 2., 3.]), np.array([5., -1., 9.]), np.array([0.01, 0.02, 0.03])
    width, height, eps = 1280, 640, 1e-6
    q = r.T @ (point - p)
    x, y, z = camera_rotation.T @ (q - offset)
    xz2, l2 = x*x + z*z, x*x + y*y + z*z
    j = np.array([[width*z/(2*np.pi*xz2), 0, -width*x/(2*np.pi*xz2)],
                  [-height*x*y/(np.pi*l2*np.sqrt(xz2)), height*np.sqrt(xz2)/(np.pi*l2),
                   -height*y*z/(np.pi*l2*np.sqrt(xz2))]])
    chain = j @ camera_rotation.T
    analytic = np.column_stack((chain @ skew(q), -chain @ r.T, chain @ r.T))
    numeric = np.zeros((2, 9))
    for k in range(9):
        delta = np.eye(3)[k % 3] * eps
        if k < 3:
            plus = project(r @ exp_so3(delta), p, point, camera_rotation, offset, width, height)
            minus = project(r @ exp_so3(-delta), p, point, camera_rotation, offset, width, height)
        elif k < 6:
            plus = project(r, p + delta, point, camera_rotation, offset, width, height)
            minus = project(r, p - delta, point, camera_rotation, offset, width, height)
        else:
            plus = project(r, p, point + delta, camera_rotation, offset, width, height)
            minus = project(r, p, point - delta, camera_rotation, offset, width, height)
        numeric[:, k] = (plus - minus) / (2 * eps)
    error = np.max(np.abs(analytic - numeric))
    assert error < 1e-6, error
    return error


if __name__ == "__main__":
    print(f"inertial_jacobian_max_error={test_inertial():.3g}")
    print(f"observation_jacobian_max_error={test_observation():.3g}")

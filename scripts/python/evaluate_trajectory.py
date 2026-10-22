#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Seungwon Choi. All rights reserved.
"""Evaluate metric VIO positions against PanoAir TUM-format ground truth."""

import argparse
import numpy as np


def load_positions(path):
    rows = np.loadtxt(path, comments="#", usecols=(0, 1, 2, 3))
    rows = np.atleast_2d(rows)
    if len(rows) < 3 or np.any(np.diff(rows[:, 0]) <= 0):
        raise ValueError(f"invalid or unsorted trajectory: {path}")
    return rows


def align(estimate, reference, allow_scale=False):
    x = estimate - estimate.mean(axis=0)
    y = reference - reference.mean(axis=0)
    u, singular, vt = np.linalg.svd(x.T @ y)
    fix = np.diag([1, 1, np.linalg.det(vt.T @ u.T)])
    rotation = vt.T @ fix @ u.T
    scale = np.sum(singular * np.diag(fix)) / np.sum(x * x) if allow_scale else 1.0
    translation = reference.mean(axis=0) - scale * rotation @ estimate.mean(axis=0)
    return (scale * estimate @ rotation.T + translation), scale, rotation


def evaluate(estimate, truth, start, rpe_dt):
    times = estimate[:, 0]
    valid = (times >= max(start, truth[0, 0])) & (times <= truth[-1, 0])
    times, positions = times[valid], estimate[valid, 1:]
    if len(times) < 3:
        raise ValueError("too few overlapping poses")
    gt = np.column_stack([np.interp(times, truth[:, 0], truth[:, axis]) for axis in (1, 2, 3)])
    aligned, _, rotation = align(positions, gt)
    ate = np.sqrt(np.mean(np.sum((aligned - gt) ** 2, axis=1)))
    indices = np.searchsorted(times, times + rpe_dt)
    valid_pair = (indices < len(times))
    pairs = np.flatnonzero(valid_pair)
    indices = indices[valid_pair]
    close = np.abs(times[indices] - times[pairs] - rpe_dt) < 0.05
    pairs, indices = pairs[close], indices[close]
    rpe = np.sqrt(np.mean(np.sum(((aligned[indices] - aligned[pairs]) -
                                  (gt[indices] - gt[pairs])) ** 2, axis=1))) if len(pairs) else np.nan

    initial_windows = {}
    for seconds in (2, 5):
        count = np.searchsorted(times, times[0] + seconds, side="right")
        if count >= 3:
            _, scale, _ = align(positions[:count], gt[:count], allow_scale=True)
            initial_windows[f"first_{seconds}s_scale_correction"] = scale
            initial_windows[f"first_{seconds}s_estimated_distance_ratio"] = 1.0 / scale
            initial_windows[f"first_{seconds}s_scale_error_pct"] = abs(1.0 / scale - 1.0) * 100
            initial_windows[f"first_{seconds}s_gt_displacement_m"] = np.linalg.norm(gt[count-1] - gt[0])

    # First 10 m of GT displacement after start: scale to multiply VIO by to match GT.
    distance = np.linalg.norm(gt - gt[0], axis=1)
    crossing = np.flatnonzero(distance >= 10.0)
    if len(crossing) and crossing[0] >= 3:
        finish = crossing[0] + 1
        _, early_scale, _ = align(positions[:finish], gt[:finish], allow_scale=True)
        early_error = abs(1.0 / early_scale - 1.0) * 100
        early_end = times[finish - 1]
    else:
        early_scale = early_error = early_end = np.nan
    return dict(poses=len(times), start=times[0], end=times[-1], ate_rmse_m=ate,
                rpe_rmse_m=rpe, rpe_dt_s=rpe_dt, **initial_windows,
                first_10m_scale_correction=early_scale,
                first_10m_estimated_distance_ratio=1.0 / early_scale,
                first_10m_scale_error_pct=early_error,
                first_10m_end_s=early_end)


def self_check():
    t = np.linspace(0, 20, 101)
    truth = np.column_stack([t, t, np.sin(t), np.cos(t)])
    estimate = truth.copy()
    estimate[:, 1:] = truth[:, 1:] * 2 + np.array([3, -2, 1])
    aligned, scale, _ = align(estimate[:, 1:], truth[:, 1:], allow_scale=True)
    assert abs(scale - 0.5) < 1e-12 and np.max(np.abs(aligned - truth[:, 1:])) < 1e-11
    metric = evaluate(estimate, truth, 0, 1)
    assert metric["ate_rmse_m"] > 1 and abs(metric["first_10m_scale_correction"] - 0.5) < 1e-12


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("estimate", nargs="?")
    parser.add_argument("ground_truth", nargs="?")
    parser.add_argument("--start", type=float, default=12.5)
    parser.add_argument("--rpe-dt", type=float, default=1.0)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print("self-check: pass")
    else:
        if not args.estimate or not args.ground_truth:
            parser.error("estimate and ground_truth are required")
        for name, value in evaluate(load_positions(args.estimate),
                                    load_positions(args.ground_truth), args.start, args.rpe_dt).items():
            print(f"{name}={value}")

#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Seungwon Choi. All rights reserved.
"""Write the VIO inputs into sequences in the ASL folder format (EuRoC MAV, TUM-VI), without copying the images.

For every sequence directory SEQ (holding mav0/) given on the command line, SEQ/mav0/ gets
  images/             link to cam0/data
  images1/            link to cam1/data (stereo: a camera-1 image pairs with the camera-0 image of the same name, i.e.
                      the same time stamp; a camera-0 image without one is processed monocular)
  cam_timestamps.txt  image times [s] from cam0/data.csv
  imu_data.csv        timestamp,ax,ay,az,gx,gy,gz from imu0/data.csv (timestamp,w,a)
  gt.txt              ground truth in TUM format (time x y z qx qy qz qw) from state_groundtruth_estimate0
                      (EuRoC) or mocap0 (TUM-VI)
so that SEQ/mav0/ is a dataset directory. A monocular config uses camera 0; a stereo config (camera1) also images1/.
"""

import argparse
import pathlib


def ns_to_seconds(text):
    ns = int(text)
    return f"{ns // 1_000_000_000}.{ns % 1_000_000_000:09d}"


def rows(path):
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            yield [value.strip() for value in line.split(",")]


def prepare(seq_dir):
    mav = seq_dir / "mav0"
    stamps = [row[0] for row in rows(mav / "cam0" / "data.csv")]
    images = sorted(p.name for p in (mav / "cam0" / "data").glob("*.png"))
    if [s + ".png" for s in stamps] != images:
        raise ValueError(f"cam0/data.csv does not match cam0/data/ ({len(stamps)} vs {len(images)})")
    link = mav / "images"
    if not link.exists():
        link.symlink_to("cam0/data")
    pairs = 0
    if (mav / "cam1" / "data").is_dir():
        images1 = set(p.name for p in (mav / "cam1" / "data").glob("*.png"))
        pairs = sum(1 for name in images if name in images1)
        link1 = mav / "images1"
        if not link1.exists():
            link1.symlink_to("cam1/data")
    (mav / "cam_timestamps.txt").write_text("".join(ns_to_seconds(s) + "\n" for s in stamps))

    imu = ["timestamp,ax,ay,az,gx,gy,gz\n"]
    for t, wx, wy, wz, ax, ay, az in (row[:7] for row in rows(mav / "imu0" / "data.csv")):
        imu.append(",".join([ns_to_seconds(t), ax, ay, az, wx, wy, wz]) + "\n")
    (mav / "imu_data.csv").write_text("".join(imu))

    source = next((mav / name / "data.csv" for name in ("state_groundtruth_estimate0", "mocap0")
                   if (mav / name / "data.csv").exists()), None)
    poses = 0
    if source is not None:
        gt = ["# timestamp tx ty tz qx qy qz qw\n"]
        for t, x, y, z, qw, qx, qy, qz in (row[:8] for row in rows(source)):
            gt.append(" ".join([ns_to_seconds(t), x, y, z, qx, qy, qz, qw]) + "\n")
        (mav / "gt.txt").write_text("".join(gt))
        poses = len(gt) - 1
    return len(stamps), len(imu) - 1, poses, pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sequences", type=pathlib.Path, nargs="+", help="sequence directories holding mav0/")
    args = parser.parse_args()
    for seq in args.sequences:
        try:
            images, imu, poses, pairs = prepare(seq)
            print(f"{seq.name}: {images} images ({pairs} with a camera-1 image), {imu} IMU samples, "
                  f"{poses} ground-truth poses")
        except (OSError, ValueError) as error:
            print(f"{seq.name}: skipped ({error})")


if __name__ == "__main__":
    main()

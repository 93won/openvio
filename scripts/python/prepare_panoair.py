#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Seungwon Choi. All rights reserved.
"""Write the VIO inputs next to the raw PanoAir images, without copying or linking them.

For every seqN/ under the PanoAir root, seqN/frame_sequences/ gets
  cam_timestamps.txt  image times [s] from rgb.txt
  imu_data.csv        timestamp,ax,ay,az,gx,gy,gz from imu0.txt (timestamp,omega,alpha)
  gt.txt              copy of SeqN_GT.txt
so that seqN/frame_sequences/ is a dataset directory (images are read from panocam/).
"""

import argparse
import pathlib
import shutil


def ns_to_seconds(text):
    ns = int(text)
    return f"{ns // 1_000_000_000}.{ns % 1_000_000_000:09d}"


def prepare(seq_dir):
    frames = seq_dir / "frame_sequences"
    images = sorted(p.name for p in (frames / "panocam").glob("*.png"))
    stamps = [line.split()[0] for line in (frames / "rgb.txt").read_text().splitlines() if line.strip()]
    if [s + ".png" for s in stamps] != images:
        raise ValueError(f"{seq_dir.name}: rgb.txt does not match panocam/ ({len(stamps)} vs {len(images)})")
    (frames / "cam_timestamps.txt").write_text("".join(ns_to_seconds(s) + "\n" for s in stamps))

    rows = ["timestamp,ax,ay,az,gx,gy,gz\n"]
    for line in (frames / "imu0.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        t, wx, wy, wz, ax, ay, az = line.split(",")
        rows.append(",".join([ns_to_seconds(t), ax, ay, az, wx, wy, wz]) + "\n")
    (frames / "imu_data.csv").write_text("".join(rows))

    gt = next(seq_dir.glob("Seq*_GT.txt"))
    shutil.copyfile(gt, frames / "gt.txt")
    return len(stamps), len(rows) - 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=pathlib.Path, help="PanoAir root with seq1 ... seqN")
    args = parser.parse_args()
    seqs = sorted((d for d in args.root.glob("seq*") if d.is_dir() and d.name[3:].isdigit()), key=lambda d: int(d.name[3:]))
    for seq in seqs:
        try:
            images, imu = prepare(seq)
            print(f"{seq.name}: {images} images, {imu} IMU samples")
        except (OSError, ValueError, StopIteration) as error:
            print(f"{seq.name}: skipped ({error})")


if __name__ == "__main__":
    main()

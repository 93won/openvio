# Release notes

## v1.0.0 (2026-10-22)

First release: a filter-based visual-inertial odometry for 360° cameras, which also runs pinhole and fisheye cameras,
monocular or stereo. Getting started: [doc/getting_started.md](doc/getting_started.md).

### Cameras

| | Camera models |
|---|---|
| Monocular | `EQUIRECTANGULAR` (360°, two lenses), `PINHOLE_RADTAN`, `FISHEYE_EQUIDISTANT` |
| Stereo | `PINHOLE_RADTAN`, `FISHEYE_EQUIDISTANT` |

### Features

- Sliding-window Kalman filter (20 poses) with up to 150 inverse-depth landmarks in the state
- Online calibration: camera-IMU time offset (per 360° lens) and extrinsics, 360° stitching error
- Startup calibration of the time offsets; initialization at rest or while moving
- Front end: optical flow with outlier checks, gain compensation, light-cluster removal
- Landmarks lost by the tracker are searched again at their predicted pixels
- Stereo matching with forward-backward and epipolar checks
- `vio odometry`, `vio gui`, scripts to download, prepare and evaluate; byte-identical reruns

### Accuracy

Mean ATE [m] (SE(3) alignment, median of the runs per sequence). No sequence failed.

| | PanoAir (17 flights) | EuRoC MAV (11) | TUM-VI indoor (20) |
|---|---|---|---|
| Mono | 1.14 | 0.122 | 0.736 |
| Stereo | – | 0.084 | 0.771 |

### Run time

Intel Core i9-13900KF desktop, Ubuntu 24.04, GCC 13.3, Release build (`-O3 -march=x86-64-v3`); each run pinned to 2
dedicated cores. Per image, mean over six EuRoC MAV and TUM-VI sequences; CPU usage = CPU time / wall time.

| | Time | CPU usage | Peak memory |
|---|---|---|---|
| Mono | 13.1 ms | 1.31 of 2 cores | 38.0 MB |
| Stereo | 20.5 ms | 1.40 of 2 cores | 44.1 MB |

360° camera (PanoAir, 1280 x 640): 20.2 ms, 1.60 of 2 cores, 55.2 MB.

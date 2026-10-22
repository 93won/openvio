# OpenVIO

Visual-inertial odometry for 360° equirectangular cameras (stitched two-lens images), pinhole and fisheye
cameras, monocular or stereo.

![OpenVIO Demo](docs/demo.gif)

**Getting started: [doc/getting_started.md](doc/getting_started.md)** · Datasets: [doc/datasets.md](doc/datasets.md) ·
Release notes: [RELEASE_NOTES.md](RELEASE_NOTES.md)

## Dependencies

- **C++17**, **CMake** >= 3.16
- **OpenCV** >= 4.0, **Eigen3** >= 3.3, **libpng**
- **Ceres Solver** and **spdlog** (included in `thirdparty/`)
- **GLFW** >= 3.3 and OpenGL for the viewer (optional; `vio gui` is built when GLFW is found; ImGui and ImPlot are
  included)

## Build

```bash
git clone https://github.com/93won/openvio.git
cd openvio
./build.sh
```

The tool is `build/bin/vio`. Runs are reproducible: the build fixes the rounding (`-ffp-contract=off`,
`-march=x86-64-v3`, fixed cache sizes for Eigen's products), the solvers run on one thread and the random seed is fixed.

## Run

```bash
# Odometry over a dataset, trajectory in TUM format (time x y z qx qy qz qw of the IMU)
build/bin/vio odometry --dataset_path /data/PanoAir/seq4/frame_sequences --config_path config/panoair.yaml \
    --output_path seq4.tum

# The same with the viewer (3D scene, image with feature states, plots; ground truth from gt.txt when present)
build/bin/vio gui --dataset_path /data/PanoAir/seq4/frame_sequences --config_path config/panoair.yaml

# Stereo: a config with a second camera
build/bin/vio odometry --dataset_path /data/EuRoC/MH_01_easy/mav0 --config_path config/euroc_stereo.yaml \
    --output_path MH_01_easy.tum

# Any option of the config file can be overridden on the command line
build/bin/vio odometry ... --estimator.max_landmarks=100 --feature_tracker.max_features=250
```

A dataset folder holds `images/` (or `panocam/`) with PNG images sorted by name, `cam_timestamps.txt` (one time per
image) and `imu_data.csv` (a header line, then `timestamp, ax, ay, az, gx, gy, gz`); a stereo dataset also `images1/`.
`scripts/python/download_dataset.py` downloads and prepares EuRoC MAV and TUM-VI sequences,
`scripts/python/prepare_panoair.py` prepares the PanoAir flights, and `scripts/python/evaluate_trajectory.py` scores a
trajectory against the ground truth ([doc/getting_started.md](doc/getting_started.md)).

| Config | |
|---|---|
| `config/panoair.yaml` | PanoAir: landmarks lost by the tracker are searched again at their predicted pixels (`--estimator.landmark_dormant_time=0`: no search, a landmark leaves the state when its track is lost) |
| `config/euroc.yaml`, `config/tum_vi.yaml` | EuRoC MAV, TUM-VI, monocular |
| `config/euroc_stereo.yaml`, `config/tum_vi_stereo.yaml` | EuRoC MAV, TUM-VI, stereo |

## Code layout

```
src/vio/
├── util/          logging, basic types (time, IMU samples and noise, pixel observations)
├── geometry/      SO(3) and quaternions, triangulation, inverse depth
├── sensor/        camera models (equirectangular, pinhole, fisheye), run-time dispatch, camera-IMU rig
├── scene/         camera, frame, keyframe
├── feature/       feature tracker, landmark tracker (finds lost landmarks again)
├── estimators/    estimator (sliding-window filter), initializer, startup calibration
├── odometry/      visual-inertial odometry engine (one image in a fixed order)
├── pipeline/      odometry pipeline (dataset, outputs, callbacks), config loader
├── ui/            viewer
└── apps/          vio (odometry, gui)
```

The pipeline reports three events through callbacks: the initialization, every processed image, and every keyframe
(an image that leaves the estimator's window with its final pose).

## Documentation

| Document | |
|---|---|
| [doc/getting_started.md](doc/getting_started.md) | build, get a dataset, run and evaluate |
| [doc/datasets.md](doc/datasets.md) | PanoAir, EuRoC MAV and TUM-VI: download, folder format, dataset settings |
| [doc/configuration.md](doc/configuration.md) | the config file, command-line overrides, stereo |
| [doc/camera_models.md](doc/camera_models.md) | the camera models and how to add one |
| [doc/architecture.md](doc/architecture.md) | modules, the processing of one image, determinism |
| [doc/references.md](doc/references.md) | the papers behind each method, dataset references |

## Datasets

PanoAir: Y. Wu et al., arXiv:2604.00852, [github.com/YYWumm/PanoAir](https://github.com/YYWumm/PanoAir). Pre-parsed
flights: [Google Drive](https://drive.google.com/drive/folders/1Cry1wAP2cYRwB4armNdCuuae5ZyYmsxm?usp=drive_link).
EuRoC MAV and TUM-VI: see [doc/references.md](doc/references.md).

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

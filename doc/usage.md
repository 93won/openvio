# Quick start

## 1. Build

```bash
git clone https://github.com/93won/openvio.git
cd openvio
./build.sh
```

The tool is `build/bin/vio`. Dependencies:

- C++17, CMake >= 3.16
- OpenCV >= 4.0, Eigen3 >= 3.3, libpng
- Ceres Solver and spdlog (included in `thirdparty/`)
- GLFW >= 3.3 and OpenGL for the viewer (optional)

## 2. Get a dataset

```bash
python3 scripts/python/download_dataset.py euroc /data/EuRoC MH_01_easy   # EuRoC MAV, one sequence
python3 scripts/python/download_dataset.py tumvi /data/TUM-VI room1       # TUM-VI, one sequence
```

Without sequence names the script fetches all 11 EuRoC MAV sequences or the 20 TUM-VI indoor sequences, and prepares
each for `vio`. PanoAir is downloaded by hand, see [datasets.md](datasets.md#panoair).

## 3. Run

```bash
# Monocular
build/bin/vio odometry --dataset_path /data/EuRoC/MH_01_easy/mav0 --config_path config/euroc.yaml \
    --output_path MH_01_easy.tum

# Stereo
build/bin/vio odometry --dataset_path /data/EuRoC/MH_01_easy/mav0 --config_path config/euroc_stereo.yaml \
    --output_path MH_01_easy_stereo.tum

# With the viewer (3D scene, image with feature states, plots; the ground truth from gt.txt)
build/bin/vio gui --dataset_path /data/EuRoC/MH_01_easy/mav0 --config_path config/euroc_stereo.yaml
```

| Dataset | `--dataset_path` | `--config_path` |
|---|---|---|
| PanoAir | `/data/PanoAir/seqN/frame_sequences` | `config/panoair.yaml` |
| EuRoC MAV | `/data/EuRoC/SEQ/mav0` | `config/euroc.yaml`, stereo `config/euroc_stereo.yaml` |
| TUM-VI | `/data/TUM-VI/SEQ/mav0` | `config/tum_vi.yaml`, stereo `config/tum_vi_stereo.yaml` |

The trajectory is the IMU pose in TUM format (`time x y z qx qy qz qw`), one line per image after the initialization.
Any config key can be overridden on the command line, e.g. `--estimator.max_landmarks=100`
([configuration.md](configuration.md)).

## 4. Evaluate

```bash
python3 scripts/python/evaluate_trajectory.py MH_01_easy.tum /data/EuRoC/MH_01_easy/mav0/gt.txt
```

It prints the ATE (RMSE of the positions after a rigid alignment) and the RPE over 1 s. `--start` (12.5 s) skips the
start of a PanoAir flight; EuRoC and TUM-VI times are absolute, so their whole run is scored. The TUM-VI motion capture
covers only its room: outside the room the ground truth has gaps, which this script interpolates across, so only the
TUM-VI room sequences are scored correctly with it.

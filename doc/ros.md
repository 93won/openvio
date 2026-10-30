# ROS 1 / ROS 2

OpenVIO can run on image and IMU topics from a sensor or a recorded bag. The ROS nodes use the same estimator as
the standalone tool. Use [Quick start](usage.md) for the standalone tool.

## 1. Build

Install ROS for the Ubuntu version below. The base installation is enough. Use a fresh terminal for each ROS
distribution. Do not source another ROS distribution or an old OpenVIO installation before the build.

| ROS | Distribution | Ubuntu | Build command | Verification |
|---|---|---|---|---|
| ROS 1 | Noetic | 20.04 | `./build_ros1.sh` | Built; RViz checked with EuRoC MH_01 mono |
| ROS 2 | Humble | 22.04 | `./build_ros2.sh humble` | Not tested |
| ROS 2 | Jazzy | 24.04 | `./build_ros2.sh jazzy` | Built; RViz checked with EuRoC MH_01 mono |

```bash
git clone https://github.com/93won/openvio.git
cd openvio
./build_ros1.sh    # ROS 1
# OR
./build_ros2.sh jazzy     # ROS 2 (use humble on Ubuntu 22.04)
```

The build needs C++17, CMake >= 3.16 and the system Python for ROS. Each script installs missing system libraries
and ROS packages with `apt` and asks for `sudo` if necessary. `build_ros1.sh` uses `catkin_make_isolated`. `build_ros2.sh` uses `colcon`.
Noetic also needs Eigen 3.4; the script downloads it into its own build directory.

| Build | Script | Output | Setup file |
|---|---|---|---|
| Standalone | `./build.sh` | `build/` | Not required |
| ROS 1 Noetic | `./build_ros1.sh` | `build_ros/noetic/` | `build_ros/noetic/install/setup.bash` |
| ROS 2 Humble | `./build_ros2.sh humble` | `build_ros/humble/` | `build_ros/humble/install/setup.bash` |
| ROS 2 Jazzy | `./build_ros2.sh jazzy` | `build_ros/jazzy/` | `build_ros/jazzy/install/setup.bash` |

Neither ROS script calls `build.sh` or writes to `build/`. Each ROS distribution has its own CMake cache and
installation. Run the same command again for an incremental build. `build_ros1.sh` always builds Noetic.
For `build_ros2.sh`, an omitted distribution uses `ROS_DISTRO`, or the only
Humble/Jazzy installation under `/opt/ros`.

## 2. Run

The commands below assume that the current directory is the repository root. Source the setup file in every new
terminal. Replace `jazzy` with `humble` for ROS 2 Humble.

### ROS 1

Terminal 1:

```bash
source build_ros/noetic/install/setup.bash
roslaunch openvio_ros1 openvio.launch dataset:=euroc
```

Terminal 2:

```bash
source build_ros/noetic/install/setup.bash
rosbag play -d 2 /data/MH_01_easy.bag
```

The delay gives the node time to connect before the first messages arrive. Start the node before the bag.

### ROS 2

Terminal 1:

```bash
source build_ros/jazzy/install/setup.bash
ros2 launch openvio_ros2 openvio.launch.py dataset:=euroc
```

Terminal 2:

```bash
source build_ros/jazzy/install/setup.bash
ros2 bag play /data/MH_01_easy
```

EuRoC and TUM-VI official bags use ROS 1 format. Convert a bag before ROS 2 playback, for example with
[rosbags](https://pypi.org/project/rosbags/):

```bash
rosbags-convert --src /data/MH_01_easy.bag --dst /data/MH_01_easy
```

### Stereo and other sensors

```bash
# TUM-VI stereo
roslaunch openvio_ros1 openvio.launch dataset:=tum_vi stereo:=true
ros2 launch openvio_ros2 openvio.launch.py dataset:=tum_vi stereo:=true

# Own sensor: supply its calibration and topic names
roslaunch openvio_ros1 openvio.launch config:=/data/sensor.yaml imu:=/sensor/imu image0:=/sensor/image
ros2 launch openvio_ros2 openvio.launch.py config:=/data/sensor.yaml imu:=/sensor/imu image0:=/sensor/image
```

Use one ROS command for the installed distribution. See [Configuration](configuration.md) for the sensor config.
Stereo inputs must have the same timestamp on the two image topics. Add `image1:=/sensor/right/image` for stereo.

| Launch argument | Default | Purpose |
|---|---|---|
| `dataset` | `euroc` | `euroc` or `tum_vi` config |
| `stereo` | `false` | Select the dataset's stereo config |
| `config` | Empty | Explicit config path; overrides `dataset` and `stereo` |
| `output_path` | Empty | Optional TUM trajectory file, e.g. `output_path:=/data/run.tum` |
| `imu` | `/imu0` | IMU input topic |
| `image0` | `/cam0/image_raw` | Camera 0 input topic |
| `image1` | `/cam1/image_raw` | Camera 1 input topic for stereo |
| `rviz` | `true` | Start RViz; use `rviz:=false` without a display |

Stop playback before you stop the node. On Ctrl-C, the node processes the queued images with the available IMU
samples and closes the trajectory file.

## 3. Check RViz

The launch command opens RViz with the supplied layout. The fixed frame is `world`; the view follows `imu`.
The estimator needs suitable motion before it can initialize. An empty view before initialization is expected.

| Display | Content |
|---|---|
| Trajectory | Blue line for the estimated path |
| Landmarks | Orange points for the landmarks in the estimator |
| Tracked features | Camera image with green feature points |
| TF | `world` to `imu` coordinate transform |

Use the mouse wheel to zoom, the left button to rotate and the middle button to move the view. If the image is too
small, drag the lower edge of the `Tracked features` panel down.

| Problem | Check |
|---|---|
| No image or path | Check the input topics with `rostopic list` or `ros2 topic list`; match the launch arguments |
| Images arrive but there is no path | Wait for motion and check the node terminal for initialization or errors |
| ROS 1 misses the start of a bag | Start the node first and use `rosbag play -d 2` |
| ROS 2 receives no data across computers | Use the same `ROS_DOMAIN_ID`; check the ROS discovery and network settings |
| Build reports a different sourced ROS | Open a fresh terminal and build the required distribution |
| A dependency install fails | Fix the reported package error, then run the build command again |

## 4. Topics and output

| Direction | Topic | Type | Content |
|---|---|---|---|
| Input | `imu` | `sensor_msgs/Imu` | Angular velocity [rad/s] and acceleration [m/s²] |
| Input | `image0`, `image1` | `sensor_msgs/Image` | `mono8`, `mono16`, `bgr8` or `rgb8` |
| Output | `odometry` | `nav_msgs/Odometry` | IMU pose in `world`; velocity in the IMU frame |
| Output | `path` | `nav_msgs/Path` | Estimated trajectory |
| Output | `landmarks` | `sensor_msgs/PointCloud2` | Landmarks in `world` |
| Output | `tracked_image` | `sensor_msgs/Image` | Image with tracked features |
| Output | `/tf` | `tf2_msgs/TFMessage` | `world` to `imu` transform |

ROS 2 message type names include `/msg/`, for example `sensor_msgs/msg/Imu`. The node parameters are `config_path`,
`output_path`, `world_frame` (default `world`) and `body_frame` (default `imu`). ROS 1 uses private parameters.
The node creates output messages when a subscriber needs them.

The optional trajectory file contains the IMU pose in TUM format (`time x y z qx qy qz qw`). The callbacks queue the
inputs, and one worker processes them in the same order as the standalone tool. Identical inputs give the same
trajectory. Official bags can differ slightly from the dataset files in timestamps or IMU values, so those inputs
do not require byte-identical trajectories.

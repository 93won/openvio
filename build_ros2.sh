#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Seungwon Choi. All rights reserved.

# Build the ROS 2 Humble or Jazzy node. Standalone builds use build.sh.
set -eo pipefail
cd "$(dirname "$0")"
if [ "$#" -gt 1 ]; then
    echo "Usage: ./build_ros2.sh [humble|jazzy]" >&2
    exit 1
fi
DISTRO=${1:-${ROS_DISTRO:-}}
if [ -z "$DISTRO" ]; then
    INSTALLED=()
    for candidate in humble jazzy; do
        [ ! -f "/opt/ros/$candidate/setup.bash" ] || INSTALLED+=("$candidate")
    done
    if [ "${#INSTALLED[@]}" -ne 1 ]; then
        echo "Select ROS 2: ./build_ros2.sh humble|jazzy" >&2
        exit 1
    fi
    DISTRO=${INSTALLED[0]}
fi
case "$DISTRO" in
    humble|jazzy) ;;
    *) echo "Unsupported ROS 2 distribution: $DISTRO (use humble or jazzy)" >&2; exit 1 ;;
esac
PACKAGES="rclcpp sensor_msgs nav_msgs geometry_msgs tf2_ros launch_ros rviz2"
if [ -n "${ROS_DISTRO:-}" ] && [ "$ROS_DISTRO" != "$DISTRO" ]; then
    echo "ROS $ROS_DISTRO is sourced. Open a fresh terminal to build $DISTRO." >&2
    exit 1
fi
if [ ! -f "/opt/ros/$DISTRO/setup.bash" ]; then
    echo "Install ROS $DISTRO first (/opt/ros/$DISTRO/setup.bash not found)." >&2
    exit 1
fi
# shellcheck disable=SC1090
source "/opt/ros/$DISTRO/setup.bash"
BUILD_DIR="$PWD/build_ros/$DISTRO"
echo "Building ROS $DISTRO into $BUILD_DIR"

bash -e scripts/install_deps.sh
MISSING=()
for package in $PACKAGES; do
    ros2 pkg prefix "$package" > /dev/null 2>&1 || MISSING+=("ros-$DISTRO-${package//_/-}")
done
if ! command -v colcon > /dev/null; then
    MISSING+=(python3-colcon-common-extensions)
fi
if [ "${#MISSING[@]}" -gt 0 ]; then
    sudo apt-get update
    sudo apt-get install -y "${MISSING[@]}"
fi

CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release -DSUITESPARSE=OFF -DPython3_EXECUTABLE=/usr/bin/python3)
colcon --log-base "$BUILD_DIR/log" build --base-paths ros/ros2 --build-base "$BUILD_DIR/build" \
    --install-base "$BUILD_DIR/install" --executor sequential --cmake-args "${CMAKE_ARGS[@]}"

echo "Build complete. In each new terminal:"
echo "  source $BUILD_DIR/install/setup.bash"
echo "  ros2 launch openvio_ros2 openvio.launch.py dataset:=euroc"

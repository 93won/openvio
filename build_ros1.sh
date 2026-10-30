#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Seungwon Choi. All rights reserved.

# Build the ROS 1 Noetic node. Standalone builds use build.sh.
set -eo pipefail
cd "$(dirname "$0")"
if [ "$#" -ne 0 ]; then
    echo "Usage: ./build_ros1.sh" >&2
    exit 1
fi
DISTRO=noetic
PACKAGES="catkin roscpp sensor_msgs nav_msgs geometry_msgs tf2_ros rviz"
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
    rospack find "$package" > /dev/null 2>&1 || MISSING+=("ros-$DISTRO-${package//_/-}")
done
if [ "${#MISSING[@]}" -gt 0 ]; then
    sudo apt-get update
    sudo apt-get install -y "${MISSING[@]}"
fi

CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release -DSUITESPARSE=OFF -DPython3_EXECUTABLE=/usr/bin/python3)
# Noetic's system Eigen is 3.3; keep Eigen 3.4 local to this ROS build.
EIGEN_VERSION=$(awk '/define EIGEN_(WORLD|MAJOR)_VERSION/{printf "%s.", $3}' \
    /usr/include/eigen3/Eigen/src/Core/util/Macros.h)
if [ "$EIGEN_VERSION" != "3.4." ]; then
    if [ ! -f "$BUILD_DIR/eigen/share/eigen3/cmake/Eigen3Config.cmake" ]; then
        mkdir -p "$BUILD_DIR"
        /usr/bin/python3 - "$BUILD_DIR/eigen-3.4.0.tar.gz" <<'PY'
import shutil
import sys
import urllib.request
url = 'https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz'
with urllib.request.urlopen(url, timeout=60) as response, open(sys.argv[1], 'wb') as target:
    shutil.copyfileobj(response, target)
PY
        tar --no-same-owner -xzf "$BUILD_DIR/eigen-3.4.0.tar.gz" -C "$BUILD_DIR"
        cmake -S "$BUILD_DIR/eigen-3.4.0" -B "$BUILD_DIR/eigen-build" \
            -DCMAKE_INSTALL_PREFIX="$BUILD_DIR/eigen" -DBUILD_TESTING=OFF
        cmake --install "$BUILD_DIR/eigen-build"
    fi
    CMAKE_ARGS+=("-DEigen3_DIR=$BUILD_DIR/eigen/share/eigen3/cmake")
fi
catkin_make_isolated --source ros/ros1 --build "$BUILD_DIR/build" --devel "$BUILD_DIR/devel" \
    --install-space "$BUILD_DIR/install" --install --cmake-args "${CMAKE_ARGS[@]}" \
    -DPYTHON_EXECUTABLE=/usr/bin/python3

echo "Build complete. In each new terminal:"
echo "  source $BUILD_DIR/install/setup.bash"
echo "  roslaunch openvio_ros1 openvio.launch dataset:=euroc"

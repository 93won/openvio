#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Seungwon Choi. All rights reserved.

# Installs the system libraries OpenVIO and the bundled Ceres Solver need (used by build.sh, build_ros1.sh and build_ros2.sh)

DEPS=""

# OpenCV, libpng, Eigen
if ! pkg-config --exists opencv4 2>/dev/null; then
    DEPS="$DEPS libopencv-dev"
fi
if ! pkg-config --exists libpng 2>/dev/null; then
    DEPS="$DEPS libpng-dev"
fi
if [ ! -d /usr/include/eigen3 ]; then
    DEPS="$DEPS libeigen3-dev"
fi

# glog
if ! pkg-config --exists libglog 2>/dev/null && [ ! -f /usr/include/glog/logging.h ]; then
    DEPS="$DEPS libgoogle-glog-dev"
fi

# gflags
if ! pkg-config --exists gflags 2>/dev/null && [ ! -f /usr/include/gflags/gflags.h ]; then
    DEPS="$DEPS libgflags-dev"
fi

# SuiteSparse (for Ceres)
if [ ! -f /usr/include/suitesparse/cholmod.h ] && [ ! -f /usr/include/cholmod.h ]; then
    DEPS="$DEPS libsuitesparse-dev"
fi

# BLAS/LAPACK
if [ ! -f /usr/lib/x86_64-linux-gnu/libblas.so ] && [ ! -f /usr/lib/libblas.so ]; then
    DEPS="$DEPS libblas-dev liblapack-dev"
fi

if [ -n "$DEPS" ]; then
    echo "Installing missing dependencies:$DEPS"
    sudo apt-get update
    sudo apt-get install -y $DEPS
else
    echo "All dependencies found."
fi

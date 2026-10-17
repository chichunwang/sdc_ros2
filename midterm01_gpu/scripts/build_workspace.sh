#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
set -u
cd /root/midterm
export CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_WORKERS:-2}"
export MAKEFLAGS="-j${BUILD_WORKERS:-2} -l${BUILD_WORKERS:-2}"
colcon build --symlink-install --parallel-workers "${BUILD_WORKERS:-2}" "$@" \
    --cmake-args -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-120}"


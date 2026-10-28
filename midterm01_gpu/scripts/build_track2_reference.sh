#!/usr/bin/env bash
# Frozen source only; compile the old CPU oracle for synthetic comparisons.
set -euo pipefail
SOURCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:?Provide a fresh output build directory}"
mkdir -p "$BUILD_DIR"
REF="$SOURCE_ROOT/track2/reference_5839746c"
nvcc -O2 -std=c++17 -arch=sm_120 -Xcompiler -fPIC -I"$REF/include" \
  -c "$REF/src/nearest_neighbor.cu" -o "$BUILD_DIR/nearest.o"
g++ -O2 -shared -std=c++17 -fPIC -I"$REF/include" -I/usr/include/pcl-1.12 \
  -I/usr/include/eigen3 -I/usr/local/cuda-12.8/include \
  "$REF/src/sensor_guided.cpp" "$REF/src/geometry_weighting.cpp" \
  "$SOURCE_ROOT/track2/reference_compare.cpp" "$BUILD_DIR/nearest.o" \
  -L/usr/local/cuda-12.8/lib64 -Wl,-rpath,/usr/local/cuda-12.8/lib64 \
  -lpcl_common -lpcl_kdtree -lcudart -o "$BUILD_DIR/liboriginal_guided.so"

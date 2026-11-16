#!/usr/bin/env bash
# 在 MASt3R-SLAM 容器內執行一個指令（一次性容器；repo 唯讀、只有 outputs 可寫）
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec docker run --rm --gpus all --shm-size 16g --user "$(id -u):$(id -g)" \
  -v "$ROOT/..:/workspace:ro" -v "$ROOT/outputs:/workspace/midterm02/outputs:rw" \
  -e TORCH_HOME=/workspace/midterm02/outputs/cache/torch -e HOME=/tmp/mast3r-home \
  -e HF_HUB_OFFLINE=1 midterm02_mast3r:cu128 "$@"

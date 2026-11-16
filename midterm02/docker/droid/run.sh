#!/usr/bin/env bash
# 在 DROID-SLAM 容器內執行一個指令（一次性容器；repo 唯讀、只有 outputs 可寫，與主 SLAM 容器相同）
# 用法：docker/droid/run.sh python3 outputs/cache/dev27_droid.py <名稱> [...]
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec docker run --rm --gpus all --shm-size 8g --user "$(id -u):$(id -g)" \
  -v "$ROOT/..:/workspace:ro" -v "$ROOT/outputs:/workspace/midterm02/outputs:rw" \
  -e TORCH_HOME=/workspace/midterm02/outputs/cache/torch \
  -w /workspace/midterm02 midterm02_droid:cu128 "$@"

#!/usr/bin/env bash
# 在 VGGT 容器內執行一個指令（一次性容器；repo 唯讀、只有 outputs 可寫；HF 權重快取在 outputs/cache/hf）
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec docker run --rm --gpus all --shm-size 8g --user "$(id -u):$(id -g)" \
  -v "$ROOT/..:/workspace:ro" -v "$ROOT/outputs:/workspace/midterm02/outputs:rw" \
  -e HF_HOME=/workspace/midterm02/outputs/cache/hf -e HOME=/tmp/vggt-home \
  -w /workspace/midterm02 midterm02_vggt:cu128 "$@"

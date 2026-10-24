#!/usr/bin/env bash
# 僅配準 frame 0；每輪保留來源快照並開啟 RViz。
set -euo pipefail
GPU_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_HOST=""
if (($#)); then
  if [[ $# -ne 2 || "$1" != --config || ! -f "$2" ]]; then
    echo '用法：scripts/run_frame0.sh [--config 主機端JSON路徑]' >&2
    exit 2
  fi
  CONFIG_HOST="$(realpath -- "$2")"
fi
mkdir -p "$GPU_ROOT/outputs"
exec 9>"$GPU_ROOT/outputs/.gpu.lock"
flock -n 9 || { echo 'GPU 工作執行中，未啟動另一輪。' >&2; exit 1; }
RUN_HOST="$(python3 "$GPU_ROOT/scripts/create_run.py" --track 03 --kind test)"
RUN_CONTAINER="/workspace/midterm01_gpu/outputs/run/$(basename "$RUN_HOST")"
CONFIG_CONTAINER="$RUN_CONTAINER/source/config/frame0.json"
if [[ -n "$CONFIG_HOST" ]]; then
  cp -- "$CONFIG_HOST" "$RUN_HOST/frame0.input.json"
  CONFIG_CONTAINER="$RUN_CONTAINER/frame0.input.json"
fi
printf '%q ' "$0" "$@" > "$RUN_HOST/invocation.txt"
printf '\n' >> "$RUN_HOST/invocation.txt"
echo "Run: $RUN_HOST"
docker inspect midterm01_track3 --format '{{.Image}}' > "$RUN_HOST/container_image.txt"
"$GPU_ROOT/scripts/dev.sh" exec python -m pip freeze > "$RUN_HOST/environment_freeze.txt"
"$GPU_ROOT/scripts/dev.sh" exec python "$RUN_CONTAINER/source/scripts/start_rviz.py" --run-dir "$RUN_CONTAINER"
"$GPU_ROOT/scripts/dev.sh" exec env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ROS_LOG_DIR="$RUN_CONTAINER/ros_log" \
  python "$RUN_CONTAINER/source/scripts/register_frame0.py" \
  --config "$CONFIG_CONTAINER" --run-dir "$RUN_CONTAINER" \
  2>&1 | tee "$RUN_HOST/console.log"

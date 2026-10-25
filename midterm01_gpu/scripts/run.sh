#!/usr/bin/env bash
# 手動 full 或已獲幀數授權的短段入口；建立 immutable snapshot 後執行。
set -euo pipefail
GPU_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
INVOCATION_ARGS=("$@")
CONFIG_ARGS=()
CONFIG_NAME=track3.json
if [[ "${1:-}" == --config ]]; then
  [[ $# -ge 3 && -f "$2" ]] || { echo '--config 必須指定存在的 JSON 檔及執行模式。' >&2; exit 2; }
  CONFIG_ARGS=(--config "$2")
  CONFIG_NAME=selected_config.json
  shift 2
fi
RUN_KIND=test
case "${1:-}" in
  --full) [[ $# == 1 ]] || { echo '用法：scripts/run.sh --full' >&2; exit 2; }; RUN_KIND=full ;;
  --max-frames) [[ $# == 2 && "${2:-}" =~ ^[0-9]+$ ]] && ((10#${2} > 0 && 10#${2} < 3142)) || { echo '用法：scripts/run.sh --max-frames N，N 必須1..3141' >&2; exit 2; } ;;
  *) echo '用法：scripts/run.sh [--config PATH] --full 或 --max-frames N（v5：官方前視 PLY + IMU）' >&2; exit 2 ;;
esac
mkdir -p "$GPU_ROOT/outputs"
exec 9>"$GPU_ROOT/outputs/.gpu.lock"
flock -n 9 || { echo 'GPU 工作執行中，未啟動另一輪。' >&2; exit 1; }
RUN_HOST="$(python3 "$GPU_ROOT/scripts/create_run.py" --kind "$RUN_KIND" "${CONFIG_ARGS[@]}")"
RUN_CONTAINER="/workspace/midterm01_gpu/outputs/run/$(basename "$RUN_HOST")"
echo "Run: $RUN_HOST"
printf '%q ' "${INVOCATION_ARGS[@]}" > "$RUN_HOST/invocation.txt"
printf '\n' >> "$RUN_HOST/invocation.txt"
docker inspect midterm01_track3 --format '{{.Image}}' > "$RUN_HOST/container_image.txt"
"$GPU_ROOT/scripts/dev.sh" exec python -m pip freeze > "$RUN_HOST/environment_freeze.txt"
"$GPU_ROOT/scripts/dev.sh" exec python "$RUN_CONTAINER/source/scripts/start_rviz.py" --run-dir "$RUN_CONTAINER"
"$GPU_ROOT/scripts/dev.sh" exec env \
  ROS_LOG_DIR="$RUN_CONTAINER/ros_log" \
  python "$RUN_CONTAINER/source/scripts/run_track3.py" \
  --config "$RUN_CONTAINER/source/config/$CONFIG_NAME" --run-dir "$RUN_CONTAINER" --publish-rviz "$@" \
  2>&1 | tee "$RUN_HOST/console.log"

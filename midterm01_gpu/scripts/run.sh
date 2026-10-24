#!/usr/bin/env bash
# 手動 full 或已獲幀數授權的短段入口；建立 immutable snapshot 後執行。
set -euo pipefail
GPU_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_KIND=test
case "${1:-}" in
  --full) RUN_KIND=full ;;
  --max-frames) [[ "${2:-}" =~ ^[0-9]+$ ]] && ((10#${2} > 0 && 10#${2} < 3142)) || { echo '短段幀數必須1..3141' >&2; exit 2; } ;;
  *) echo '用法：scripts/run.sh --full [--source ply|loger] 或 --max-frames N [--source ply|loger]' >&2; exit 2 ;;
esac
mkdir -p "$GPU_ROOT/outputs"
exec 9>"$GPU_ROOT/outputs/.gpu.lock"
flock -n 9 || { echo 'GPU 工作執行中，未啟動另一輪。' >&2; exit 1; }
RUN_HOST="$(python3 "$GPU_ROOT/scripts/create_run.py" --kind "$RUN_KIND")"
RUN_CONTAINER="/workspace/midterm01_gpu/outputs/run/$(basename "$RUN_HOST")"
echo "Run: $RUN_HOST"
printf '%s\n' "$*" > "$RUN_HOST/invocation.txt"
docker inspect midterm01_track3 --format '{{.Image}}' > "$RUN_HOST/container_image.txt"
"$GPU_ROOT/scripts/dev.sh" exec python -m pip freeze > "$RUN_HOST/environment_freeze.txt"
"$GPU_ROOT/scripts/dev.sh" exec python "$RUN_CONTAINER/source/scripts/start_rviz.py" --run-dir "$RUN_CONTAINER"
"$GPU_ROOT/scripts/dev.sh" exec env \
  ROS_LOG_DIR="$RUN_CONTAINER/ros_log" \
  python "$RUN_CONTAINER/source/scripts/run_track3.py" \
  --config "$RUN_CONTAINER/source/config/track3.json" --run-dir "$RUN_CONTAINER" --publish-rviz "$@" \
  2>&1 | tee "$RUN_HOST/console.log"

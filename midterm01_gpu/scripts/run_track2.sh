#!/usr/bin/env bash
# 由使用者啟動 full；短段需要當次明確幀數授權。
set -euo pipefail
GPU_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_KIND=test
case "${1:-}" in
  --full) [[ $# == 1 ]] || exit 2; RUN_KIND=full ;;
  --max-frames) [[ $# == 2 && "${2:-}" =~ ^[0-9]+$ ]] && ((10#${2} > 0 && 10#${2} < 3299)) || { echo 'N 必須 1..3298' >&2; exit 2; } ;;
  *) echo '用法：scripts/run_track2.sh --full 或 --max-frames N' >&2; exit 2 ;;
esac
mkdir -p "$GPU_ROOT/outputs"
exec 9>"$GPU_ROOT/outputs/.gpu.lock"
flock -n 9 || { echo 'GPU 工作執行中，未啟動另一輪。' >&2; exit 1; }
RUN_HOST="$(python3 "$GPU_ROOT/scripts/create_track2_run.py" --kind "$RUN_KIND")"
RUN_CONTAINER="/workspace/midterm01_gpu/outputs/run/$(basename "$RUN_HOST")"
echo "Run: $RUN_HOST"
printf '%q ' "$@" > "$RUN_HOST/invocation.txt"
printf '\n' >> "$RUN_HOST/invocation.txt"
docker inspect midterm01_track3 --format '{{.Image}}' > "$RUN_HOST/container_image.txt"
"$GPU_ROOT/scripts/dev.sh" exec python -m pip freeze > "$RUN_HOST/environment_freeze.txt"
"$GPU_ROOT/scripts/dev.sh" exec python "$RUN_CONTAINER/source/scripts/start_track2_rviz.py" --run-dir "$RUN_CONTAINER"
"$GPU_ROOT/scripts/dev.sh" exec python "$RUN_CONTAINER/source/scripts/run_track2.py" \
  --run-dir "$RUN_CONTAINER" --publish-rviz "$@" 2>&1 | tee "$RUN_HOST/console.log"

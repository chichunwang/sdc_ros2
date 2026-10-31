#!/usr/bin/env bash
# 共用研究入口，使用者平常使用各 Track 固定入口。
set -euo pipefail
GPU_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TRACK="${1:-}"
[[ "$TRACK" =~ ^[123]$ ]] || { echo 'Track 必須是 1、2 或 3' >&2; exit 2; }
shift
MODE="${1:---check}"
[[ $# -le 1 && ( "$MODE" == --check || "$MODE" == --full ) ]] || {
  echo "用法：./scripts/run_external_track${TRACK}.sh --check 或 --full" >&2; exit 2;
}
mkdir -p "$GPU_ROOT/outputs"
exec 9>"$GPU_ROOT/outputs/.gpu.lock"
flock -n 9 || { echo '其他 Track 工作執行中，未啟動本輪。' >&2; exit 1; }
RUN_HOST="$(python3 "$GPU_ROOT/scripts/external_track3.py" prepare "$MODE" "$TRACK")"
RUN_CONTAINER="/workspace/midterm01_gpu/outputs/run/$(basename "$RUN_HOST")"
echo "Run: $RUN_HOST"
finish() {
  local code=$?
  trap - EXIT
  printf '%s\n' "$code" > "$RUN_HOST/launcher_exit_code.txt"
  python3 "$RUN_HOST/source/scripts/external_track3.py" finalize "$RUN_HOST" "$code"
  exit "$code"
}
trap finish EXIT
docker inspect midterm01_track3 --format '{{.Image}}' > "$RUN_HOST/container_image.txt"
"$GPU_ROOT/scripts/dev.sh" exec python -m pip install --disable-pip-version-check \
  --target "$RUN_CONTAINER/deps" 'polars==1.44.2' 2>&1 | tee "$RUN_HOST/dependencies.log"
"$GPU_ROOT/scripts/dev.sh" exec env \
  PYTHONPATH="$RUN_CONTAINER/deps" PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR="$RUN_CONTAINER/mplconfig" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python "$RUN_CONTAINER/source/scripts/external_track3.py" execute "$RUN_CONTAINER" "$MODE" \
  2>&1 | tee "$RUN_HOST/console.log"

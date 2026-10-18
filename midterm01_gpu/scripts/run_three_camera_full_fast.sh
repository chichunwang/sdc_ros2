#!/usr/bin/env bash
# 獨立加速入口：相同模型／尺度／取樣規則，CUDA PCA 與 voxel 整數排序。
set -euo pipefail
RIG_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RIG_PROJECT"
RIG_RESUME=""
RIG_DRY=0
case "${1:-}" in
  --resume) RIG_RESUME="${2:?請提供先前印出的 run 目錄}"; [[ $# == 2 ]] || exit 2 ;;
  --dry-run) RIG_DRY=1; [[ $# == 1 ]] || exit 2 ;;
  "") ;;
  *) echo '用法：bash scripts/run_three_camera_full_fast.sh [--dry-run | --resume <run 目錄>]' >&2; exit 2 ;;
esac
RIG_PY="$RIG_PROJECT/outputs/ml_runtime/venv/bin/python"
RIG_CANDIDATE="$RIG_PROJECT/outputs/runs/rig_bias_validation_2610181142/candidate_final.json"
if [[ "$RIG_DRY" == 0 ]]; then
  if pgrep -f '[p]ython.*scripts/run_rig_(full|fast)\.py' >/dev/null; then
    echo '已有三相機重建程序；請等目前這輪完成後再啟動另一輪。' >&2; exit 2;
  fi
  [[ "$(docker inspect -f '{{.State.Running}}' sdc_midterm01_gpu)" == true ]] || {
    echo '請先執行 bash scripts/dev.sh up 啟動 GPU 容器。' >&2; exit 2;
  }
  if docker exec sdc_midterm01_gpu pgrep -x localization >/dev/null; then
    echo 'GPU 容器已有 localization；請等它完成後再執行。' >&2; exit 2;
  fi
fi
if [[ "$RIG_DRY" == 1 ]]; then
  RIG_RUN="$RIG_PROJECT/outputs/runs/three_camera_full_fast_preview"
elif [[ -n "$RIG_RESUME" ]]; then
  RIG_RUN="$(realpath "$RIG_RESUME")"
  [[ -f "$RIG_RUN/candidate.json" ]] || { echo '找不到該 run 的 candidate.json' >&2; exit 2; }
  RIG_CANDIDATE="$RIG_RUN/candidate.json"
else
  RIG_RUN="$(python3 -c 'from pathlib import Path; import sys; sys.path.insert(0,"scripts/accuracy"); from benchmark import create_run_directory; print(create_run_directory(Path("outputs/runs").resolve(),"three_camera_full_fast"))')"
  cp "$RIG_CANDIDATE" "$RIG_RUN/candidate.json"
  RIG_CANDIDATE="$RIG_RUN/candidate.json"
fi
echo "Run: $RIG_RUN"
echo '這是驗證未通過尺度的三相機實驗候選；既有最佳 CSV 不變，不自動提交。'
for RIG_TRACK in 1 2; do
  if [[ "$RIG_TRACK" == 1 ]]; then
    RIG_COUNT=3011
    RIG_PRIOR="$RIG_PROJECT/outputs/runs/track1_sensor_guided_heading_rts_geometry_directional_cuda_robust_icp_d2d_3011f_70rv9cj4"
    RIG_MAP="$RIG_PROJECT/outputs/runs/cuda_ndt_20261017_nlvdk4j5/cache_track1_leaf1.pcd"
  else
    RIG_COUNT=3299
    RIG_PRIOR="$RIG_PROJECT/outputs/runs/track2_sensor_guided_heading_rts_geometry_directional_cuda_robust_icp_d2d_3299f_7o_atg6_"
    RIG_MAP="$RIG_PROJECT/outputs/runs/three_camera_dev_2610181055/cache_track2_leaf1.pcd"
  fi
  RIG_ARGS=(--track "$RIG_TRACK" --frames "$RIG_COUNT" --data-root "$RIG_PROJECT/../midterm01/data"
    --checkpoint "$RIG_PROJECT/outputs/ml_runtime/checkpoint" --model-repo "$RIG_PROJECT/outputs/ml_runtime/map-anything"
    --prior-run "$RIG_PRIOR" --candidate "$RIG_CANDIDATE" --output "$RIG_RUN/track$RIG_TRACK"
    --localize --map-cache "$RIG_MAP")
  if [[ "$RIG_DRY" == 1 ]]; then RIG_ARGS+=(--dry-run)
  elif [[ -d "$RIG_RUN/track$RIG_TRACK" ]]; then RIG_ARGS+=(--resume)
  fi
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$RIG_PY" scripts/run_rig_fast.py "${RIG_ARGS[@]}"
  if [[ "$RIG_DRY" == 0 ]]; then cp "$RIG_RUN/track$RIG_TRACK/result_$RIG_TRACK.csv" "$RIG_RUN/result_$RIG_TRACK.csv"; fi
done
if [[ "$RIG_DRY" == 0 ]]; then
  echo "完成：$RIG_RUN/result_1.csv"
  echo "完成：$RIG_RUN/result_2.csv"
fi

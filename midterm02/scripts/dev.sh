#!/usr/bin/env bash
# 管理 Midterm02 專用容器 midterm02_vslam（不影響 midterm01 容器與 Docker daemon）
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export VSLAM_UID="$(id -u)" VSLAM_GID="$(id -g)"
export DISPLAY="${DISPLAY:-:0}"   # SSH 進來時沒有 DISPLAY，預設用本機桌面 :0
cd "$ROOT"
mkdir -p outputs/run outputs/cache
COMPOSE=(docker compose -p midterm02_vslam -f docker-compose.yml)
case "${1:-help}" in
  build)         exec docker build -t midterm02_vslam:py39-gpu . ;;
  build-learned) exec docker build --build-arg INSTALL_LEARNED=1 -t midterm02_vslam:py39-gpu . ;;
  up)    exec "${COMPOSE[@]}" up -d ;;
  down)  exec "${COMPOSE[@]}" down ;;
  shell) exec "${COMPOSE[@]}" exec vslam bash ;;
  exec)  shift; exec "${COMPOSE[@]}" exec -T vslam "$@" ;;
  rviz)  # 啟動（或重啟）RViz 視窗與 bridge
         "${COMPOSE[@]}" --profile rviz up -d --force-recreate rviz ;;
  rviz-stop) exec "${COMPOSE[@]}" --profile rviz rm -sf rviz ;;
  replay)  # 把跑完的 run（DROID／P1／F1…）重播到 RViz：scripts/dev.sh replay outputs/run/<run> [--ref <run>] [--step 2] [--fps 30]
         shift
         if [ -z "$(docker ps -q -f name=^midterm02_rviz$ -f status=running)" ]; then
           "${COMPOSE[@]}" --profile rviz up -d rviz
         fi
         "${COMPOSE[@]}" up -d vslam
         exec "${COMPOSE[@]}" exec -T vslam python scripts/rviz_replay.py "$@" ;;
  slam)  # 手動執行 SLAM 的標準入口：一定開 RViz。用法：scripts/dev.sh slam --track 01 [--frames N] [--experiment NAME]
         shift
         if [ -z "$(docker ps -q -f name=^midterm02_rviz$ -f status=running)" ]; then
           "${COMPOSE[@]}" --profile rviz up -d rviz
         fi
         "${COMPOSE[@]}" up -d vslam
         exec "${COMPOSE[@]}" exec -T vslam python vslam/run_slam.py --rviz rviz:5600 "$@" ;;
  gpu)   exec "${COMPOSE[@]}" exec -T vslam python scripts/gpu_check.py ;;
  test)  exec "${COMPOSE[@]}" exec -T vslam python -m unittest discover -s tests -v ;;
  *) echo '用法：scripts/dev.sh {build|build-learned|up|down|shell|exec COMMAND...|gpu|test|rviz|rviz-stop|slam ARGS...|replay RUN [ARGS...]}' ;;
esac

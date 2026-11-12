#!/usr/bin/env bash
# 管理 Midterm02 專用容器 midterm02_vslam（不影響 midterm01 容器與 Docker daemon）
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export VSLAM_UID="$(id -u)" VSLAM_GID="$(id -g)"
cd "$ROOT"
mkdir -p outputs/run
COMPOSE=(docker compose -p midterm02_vslam -f docker-compose.yml)
case "${1:-help}" in
  build) exec docker build -t midterm02_vslam:py39 . ;;
  up)    exec "${COMPOSE[@]}" up -d ;;
  down)  exec "${COMPOSE[@]}" down ;;
  shell) exec "${COMPOSE[@]}" exec vslam bash ;;
  exec)  shift; exec "${COMPOSE[@]}" exec -T vslam "$@" ;;
  test)  exec "${COMPOSE[@]}" exec -T vslam python -m unittest discover -s tests -v ;;
  *) echo '用法：scripts/dev.sh {build|up|down|shell|exec COMMAND...|test}' ;;
esac

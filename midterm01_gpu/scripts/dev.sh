#!/usr/bin/env bash
set -euo pipefail
GPU_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export TRACK3_UID="$(id -u)" TRACK3_GID="$(id -g)"
export TRACK3_XAUTHORITY="${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}"
cd "$GPU_ROOT"
case "${1:-help}" in
  build) docker build --build-arg "BASE_IMAGE=${TRACK3_BASE_IMAGE:-sdc_midterm01_gpu_ros2:cuda12.8}" -f Dockerfile -t midterm01_track3:cuda12.8 .
    exec docker build -f Dockerfile.loger -t midterm01_track3:loger . ;;
  up) exec docker compose -p midterm01_track3 -f docker-compose.yml up -d ;;
  down) exec docker compose -p midterm01_track3 -f docker-compose.yml down ;;
  shell) exec docker compose -p midterm01_track3 -f docker-compose.yml exec track3 /track3_entrypoint.sh bash ;;
  exec) shift; exec docker compose -p midterm01_track3 -f docker-compose.yml exec -T track3 /track3_entrypoint.sh "$@" ;;
  *) echo '用法：scripts/dev.sh {build|up|down|shell|exec COMMAND...}' ;;
esac

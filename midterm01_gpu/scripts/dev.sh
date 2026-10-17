#!/usr/bin/env bash
set -euo pipefail
gpu_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cpu_only=false
if [[ "${1:-}" == --cpu ]]; then
    cpu_only=true
    shift
fi
action="${1:-help}"
if (( $# > 0 )); then shift; fi

usage() {
    cat <<'USAGE'
在 Ubuntu 主機執行：
  bash scripts/dev.sh init              建立本環境的 .env 與輸出目錄，不建立或複製 data
  bash scripts/dev.sh config            顯示 GPU Compose 設定與共用資料掛載
  bash scripts/dev.sh build             建置 ROS Humble + CUDA 12.8 映像
  bash scripts/dev.sh up                啟動獨立 GPU 容器，需要主機 NVIDIA CDI
  bash scripts/dev.sh shell             進入容器
  bash scripts/dev.sh build-workspace   Release 編譯獨立 ROS workspace
  bash scripts/dev.sh check-data [args] 唯讀檢查共用資料，含 --strict-count
  bash scripts/dev.sh cuda-check        執行真實 CUDA kernel 並驗證結果
  bash scripts/dev.sh doctor            查看主機 GPU、Docker CDI 與容器狀態
  bash scripts/dev.sh status            查看本 Compose project
  bash scripts/dev.sh stop              只停止本 GPU project
  bash scripts/dev.sh --cpu up          明確以無 GPU 模式啟動，僅供建置／診斷

主機首次啟用 GPU：sudo bash scripts/setup_host_gpu.sh（不重啟 Docker）
USAGE
}

case "$action" in
    help|-h|--help) usage; exit 0 ;;
    init)
        mkdir -p "$gpu_dir/workspace/src" "$gpu_dir/outputs/runs" \
            "$gpu_dir/outputs/submissions" "$gpu_dir/outputs/figures"
        if [[ ! -e "$gpu_dir/.env" ]]; then cp "$gpu_dir/.env.example" "$gpu_dir/.env"; fi
        printf 'Initialized %s; data stays in ../midterm01/data.\n' "$gpu_dir"
        exit 0 ;;
    config|build|up|shell|check-data|build-workspace|cuda-check|doctor|status|stop) ;;
    *) usage >&2; exit 2 ;;
esac

compose=(docker compose --project-name sdc_midterm01_gpu \
    --project-directory "$gpu_dir" --file "$gpu_dir/docker-compose.yml")
if [[ -f "$gpu_dir/.env" ]]; then compose+=(--env-file "$gpu_dir/.env"); fi
if [[ "$cpu_only" == false ]]; then compose+=(--file "$gpu_dir/docker-compose.gpu.yml"); fi

case "$action" in
    config) "${compose[@]}" config "$@" ;;
    build)
        if ! docker image inspect sdc_midterm01_ros2:latest >/dev/null 2>&1; then
            printf 'Build the existing midterm01 ROS image first; no existing image was replaced.\n' >&2
            exit 1
        fi
        "${compose[@]}" build "$@" ;;
    up)
        if [[ "$cpu_only" == false ]]; then
            if ! command -v nvidia-ctk >/dev/null || ! nvidia-ctk cdi list 2>/dev/null | grep -qx 'nvidia.com/gpu=all'; then
                printf 'Host NVIDIA CDI is missing. Run: sudo bash %q\n' "$gpu_dir/scripts/setup_host_gpu.sh" >&2
                exit 1
            fi
        else
            printf 'Starting explicit CPU-only development mode; GPU runtime is not enabled.\n'
        fi
        "${compose[@]}" up -d --no-build "$@" ;;
    shell) "${compose[@]}" exec midterm01_gpu bash "$@" ;;
    build-workspace) "${compose[@]}" exec -T midterm01_gpu bash /root/scripts/build_workspace.sh "$@" ;;
    check-data) "${compose[@]}" exec -T midterm01_gpu python3 /root/reference/check_data.py \
        --data-root /root/data --config /root/config/tracks.json "$@" ;;
    cuda-check) "${compose[@]}" exec -T midterm01_gpu midterm-cuda-probe "$@" ;;
    status) "${compose[@]}" ps "$@" ;;
    stop) "${compose[@]}" stop "$@" ;;
    doctor)
        nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
        docker info --format 'CDI spec directories: {{json .CDISpecDirs}}'
        if command -v nvidia-ctk >/dev/null; then nvidia-ctk cdi list; else printf 'NVIDIA Container Toolkit: not installed\n'; fi
        "${compose[@]}" ps ;;
esac


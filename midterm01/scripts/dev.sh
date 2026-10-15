#!/usr/bin/env bash
# Run on the remote Ubuntu host. Use bash explicitly after a Git checkout.
set -euo pipefail

midterm_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
repo_dir="$(cd -- "$midterm_dir/.." && pwd)"
gui_enabled=false
if [[ "${1:-}" == "--gui" ]]; then
    gui_enabled=true
    shift
fi
action="${1:-help}"
if (( $# > 0 )); then
    shift
fi

usage() {
    cat <<'USAGE'
Run on the remote Ubuntu host:
  bash scripts/dev.sh init              Create local folders and .env if absent
  bash scripts/dev.sh config            Show resolved Docker mounts/settings
  bash scripts/dev.sh build             Build HW2 base if missing, then midterm image
  bash scripts/dev.sh up                Start the midterm container without rebuilding
  bash scripts/dev.sh shell             Enter the midterm container
  bash scripts/dev.sh check-data [args]  Check dataset structure inside the container
  bash scripts/dev.sh build-workspace   Build ROS packages after importing the template
  bash scripts/dev.sh status            Show this Compose project's containers
  bash scripts/dev.sh stop              Stop this Compose project's container
  bash scripts/dev.sh --gui up          Enable the optional Ubuntu X11 overlay

HW2 uses its own container. No HW2 container is stopped or recreated by this script.
USAGE
}

case "$action" in
    help|-h|--help)
        usage
        exit 0
        ;;
    init)
        mkdir -p -- "$midterm_dir/workspace/src" \
            "$midterm_dir/data/Track1" "$midterm_dir/data/Track2" \
            "$midterm_dir/outputs/runs" "$midterm_dir/outputs/submissions" \
            "$midterm_dir/outputs/figures"
        if [[ ! -e "$midterm_dir/.env" ]]; then
            cp -- "$midterm_dir/.env.example" "$midterm_dir/.env"
        fi
        printf 'Initialized %s\n' "$midterm_dir"
        printf 'Review .env on Ubuntu and set MIDTERM_DATA_DIR to your actual dataset root.\n'
        exit 0
        ;;
    config|build|up|shell|check-data|build-workspace|status|stop)
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

if ! command -v docker >/dev/null 2>&1; then
    printf 'Docker is unavailable in this shell. Run Docker actions on the remote Ubuntu host.\n' >&2
    exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
    printf 'Docker Compose v2 is required (docker compose).\n' >&2
    exit 1
fi

compose=(docker compose --project-name sdc_midterm01 \
    --project-directory "$midterm_dir" --file "$midterm_dir/docker-compose.yml")
if [[ -f "$midterm_dir/.env" ]]; then
    compose+=(--env-file "$midterm_dir/.env")
fi
if [[ "$gui_enabled" == true ]]; then
    compose+=(--file "$midterm_dir/docker-compose.gui.yml")
fi

case "$action" in
    config)
        "${compose[@]}" config "$@"
        ;;
    build)
        if ! docker image inspect sdc_hw2_ros2:latest >/dev/null 2>&1; then
            # Send only the two files used by the HW2 Dockerfile, not its large dataset.
            tar -C "$repo_dir/hw2" -cf - Dockerfile ros_entrypoint.sh \
                | docker build --tag sdc_hw2_ros2:latest -
        fi
        "${compose[@]}" build "$@"
        ;;
    up)
        "${compose[@]}" up -d --no-build "$@"
        ;;
    shell)
        "${compose[@]}" exec midterm01 bash "$@"
        ;;
    check-data)
        "${compose[@]}" exec -T midterm01 python3 /root/scripts/check_data.py "$@"
        ;;
    build-workspace)
        "${compose[@]}" exec -T midterm01 bash /root/scripts/build_workspace.sh "$@"
        ;;
    status)
        "${compose[@]}" ps "$@"
        ;;
    stop)
        "${compose[@]}" stop "$@"
        ;;
esac


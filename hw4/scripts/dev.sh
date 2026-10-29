#!/usr/bin/env bash
# 從 Ubuntu 主機呼叫；沿用 HW2 的 Compose project/service。
set -euo pipefail
hw4_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
compose=(docker compose -f "$hw4_dir/../hw2/docker-compose.yml")
command_name="${1:-help}"
if (($#)); then shift; fi

find_bag() {
  if (($#)); then
    local resolved
    resolved="$(realpath -e -- "$1")"
    case "$resolved/" in
      "$hw4_dir/colcon_ws/data/"*) ;;
      *) echo 'Bag 必須放在 hw4/colcon_ws/data/ 內。' >&2; return 1 ;;
    esac
    [[ -f "$resolved/metadata.yaml" ]] || { echo '此目錄缺少 metadata.yaml' >&2; return 1; }
    if [[ "$resolved" == "$hw4_dir/colcon_ws/data" ]]; then
      printf '/ros_ws/hw4/data\n'
    else
      printf '/ros_ws/hw4/data/%s\n' "${resolved#"$hw4_dir/colcon_ws/data/"}"
    fi
  else
    local -a metadata_files=()
    [[ -d "$hw4_dir/colcon_ws/data" ]] || { echo '請先建立 colcon_ws/data 並解壓 bag。' >&2; return 1; }
    mapfile -d '' -t metadata_files < <(find "$hw4_dir/colcon_ws/data" -type f -name metadata.yaml -print0)
    if ((${#metadata_files[@]} != 1)); then
      echo "找到 ${#metadata_files[@]} 個 bag；請明確指定含 metadata.yaml 的主機目錄。" >&2
      return 1
    fi
    find_bag "$(dirname -- "${metadata_files[0]}")"
  fi
}

case "$command_name" in
  recreate)
    mkdir -p "$hw4_dir/colcon_ws/data" "$hw4_dir/colcon_ws/result"
    "${compose[@]}" up -d --no-build --pull never --force-recreate ros2_dev
    ;;
  deps)
    "${compose[@]}" exec --user root ros2_dev bash -c \
      'apt-get update && apt-get install -y python3-numpy python3-matplotlib python3-reportlab fonts-arphic-uming python3-colcon-common-extensions'
    ;;
  shell) "${compose[@]}" exec ros2_dev bash ;;
  build)
    "${compose[@]}" exec ros2_dev bash -c \
      'source /opt/ros/humble/setup.bash && cd /ros_ws/hw4 && colcon build --packages-select fusion'
    ;;
  launch)
    # 在主機建立可讀目錄，每次執行保留不同結果。
    run_dir="$(mktemp -d "$hw4_dir/colcon_ws/result/run_$(TZ=Asia/Taipei date +%y%m%d_%H%M%S)_XXXX")"
    chmod 755 "$run_dir"
    git -C "$hw4_dir/.." rev-parse HEAD > "$run_dir/commit.txt"
    git -C "$hw4_dir/.." status --short > "$run_dir/git_status.txt"
    git -C "$hw4_dir/.." diff --binary HEAD -- hw4 hw2/docker-compose.yml > "$run_dir/source_changes.patch"
    printf 'Ubuntu 結果位置：%s\n' "$run_dir"
    "${compose[@]}" exec ros2_dev bash -c \
      'source /opt/ros/humble/setup.bash && cd /ros_ws/hw4 && source install/setup.bash && exec ros2 launch fusion EKF_fusion.launch.py "result_dir:=$1" "${@:2}"' \
      bash "/ros_ws/hw4/result/$(basename -- "$run_dir")" "$@"
    ;;
  info|play|inspect)
    bag_path="$(find_bag "$@")"
    if [[ "$command_name" == info ]]; then
      "${compose[@]}" exec ros2_dev bash -c \
        'source /opt/ros/humble/setup.bash && exec ros2 bag info "$1"' bash "$bag_path"
    elif [[ "$command_name" == inspect ]]; then
      # scripts 不額外掛載，透過 stdin 傳送唯讀檢查程式。
      "${compose[@]}" exec -T ros2_dev bash -c \
        'source /opt/ros/humble/setup.bash && exec python3 - "$1"' bash "$bag_path" \
        < "$hw4_dir/scripts/inspect_bag.py"
    else
      "${compose[@]}" exec ros2_dev bash -c \
        'source /opt/ros/humble/setup.bash && exec ros2 bag play "$1" --delay 5' bash "$bag_path"
    fi
    ;;
  test)
    "${compose[@]}" exec -T ros2_dev bash -c \
      'source /opt/ros/humble/setup.bash && cd /ros_ws/hw4 && source install/setup.bash && exec python3 -m unittest discover -s /ros_ws/hw4_tools/tests -v'
    ;;
  report)
    "${compose[@]}" exec ros2_dev bash -c \
      'cd /ros_ws/hw4_tools && exec python3 scripts/prepare_submission.py'
    ;;
  status) "${compose[@]}" ps ;;
  help)
    echo '用法：bash scripts/dev.sh {recreate|deps|build|launch|info|inspect|play|test|report|shell|status}'
    echo 'info/inspect/play 可附上 data 內含 metadata.yaml 的 Ubuntu 主機目錄。'
    echo 'launch 可加 use_rviz:=false；預設開 RViz2。'
    ;;
  *) echo "未知指令：$command_name" >&2; exit 2 ;;
esac

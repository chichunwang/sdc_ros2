# Midterm01 CPU 手動操作手冊（歷史重現）

2026-10-17 後續更新：使用者指定 Midterm01 統一使用 [GPU 工作區](../../midterm01_gpu/doc/27_PROJECT_OVERVIEW.md)。日常編輯／建置／新版完整 Track／結果讀取／chmod 請依 [GPU 操作手冊](../../midterm01_gpu/doc/08_MANUAL_WORKFLOW.md)。本文以下命令只供原 CPU 環境重現，不套到 GPU；GPU dev.sh 沒有 `--gui`，benchmark 的 `--frames` 也不接受 0。CPU Track1 既有完整結果已核對，見 [完成紀錄](CPU_TRACK1_COMPLETED_20261017.md)。

2026-10-17，依目前 checkout 的程式與 launch 介面整理。Docker／colcon 與無 GUI 首跑已驗證；本手冊的桌面 RViz 操作尚未現場驗證。原始資料保持唯讀，每次測試使用新的輸出目錄。

## 1. 工作位置與編輯架構

在 Ubuntu 主機用 VS Code 開啟 `/home/ricky/sdc_ros2`。主機修改會經 bind mount 直接出現在容器，不必重新複製程式。

| 工作 | Ubuntu 主機位置 | 容器位置 |
| --- | --- | --- |
| 程式 | `midterm01/workspace/src/localization/` | `/root/midterm/src/localization/` |
| 輸入資料 | `midterm01/data/` | `/root/data/`（唯讀） |
| 執行結果 | `midterm01/outputs/` | `/root/outputs/` |

| 要改什麼 | 編輯檔案／函式 |
| --- | --- |
| 主流程、前一幀初值、品質判定與 fallback | `src/localization.cpp` 的 `RunLocalization()` |
| NDT 配準；未來可加入其他 scan matcher | `src/localization.cpp` 的 `AlignNDT()` |
| 清理／下採樣、局部地圖 | `FilterInputCloud()`、`CropSubmap()`、`LoadMap()` |
| 新成員、新方法、預設參數 | `include/localization/localization.h` |
| 新 ROS 參數 | C++ `LoadParameters()` 的 declare/get；需從 launch 傳入時同步修改 launch |
| 啟動、參數轉接、RViz 開關 | `launch/localization.launch.py` |
| RViz 畫面 | `config/localization.rviz`，或在 RViz 中 Save Config As |
| 編譯／套件依賴 | `CMakeLists.txt`、`package.xml`；系統套件另寫入 `midterm01/Dockerfile` |
| TXT 轉 CSV | `processing_codes/txt_to_idx_x_y_yaw.py` |

目前流程：索引並按時間排序 PLY → 縮放／濾波 → 依上一個 pose 裁切地圖 → NDT → 接受 candidate 或保留前一 pose → 發布 cloud／TF／path → 寫 TXT。

只改數值時先用 launch 參數；改流程才改 C++。新增功能可先保留在既有 package，等介面穩定再拆檔。新增 `.cpp` 時需要把它加進 CMake 的 target；沒有指定第二個 package 的必要。

NDT 輸出目前是 `T_map_source`，直接接受最終 transformation，不再乘一次初始矩陣。先保留前相機尺度 3.0 做可比較的 R0；尺度不是已校準結論。`bag_path` 只會讀感測器，目前沒有融合。

## 2. 啟動與編譯：在 Ubuntu 主機

```bash
cd /home/ricky/sdc_ros2
git status --short
cd midterm01
bash scripts/dev.sh up
bash scripts/dev.sh build-workspace --parallel-workers 2
bash scripts/dev.sh shell
```

最後一行進入容器。`build-workspace` 雖然從主機輸入，實際 colcon 在容器執行。C++ 改完要重新 build，之後各容器終端重新 source workspace。只換 launch 參數不必重建 Docker。

只有修改 Dockerfile／系統依賴時，才在沒有測試執行的時候回到 Ubuntu 主機：

```bash
cd /home/ricky/sdc_ros2/midterm01
bash scripts/dev.sh build
bash scripts/dev.sh up
bash scripts/dev.sh build-workspace --parallel-workers 2
```

若要保留 GUI overlay，上面的 `up` 改成 `bash scripts/dev.sh --gui up`。套用不同 Compose 設定可能重建容器，先結束目前測試；掛載的原始資料和 outputs 仍在主機。

## 3. 一次測試的完整命令：在容器

以下可整段貼進容器 Bash。預設先跑 Track1 單幀；之後每次重新執行整段，只修改最前面的五個設定。每次生成新目錄，不覆寫首跑紀錄。

```bash
source /opt/ros/humble/setup.bash
cd /root/midterm
source install/setup.bash

TRACK=1
FRAMES=1
LABEL=E01_ndt
USE_RVIZ=false
KEEP_ALIVE=false

RUN_DIR="/root/outputs/runs/${LABEL}_track${TRACK}_n${FRAMES}_$(date +%Y%m%dT%H%M%S_%N)"
mkdir "$RUN_DIR"
cp -a src/localization "$RUN_DIR/source"
printf 'Results: %s\n' "$RUN_DIR"

python3 - "$TRACK" "$RUN_DIR" <<'PY'
from pathlib import Path
import csv, re, sys
base = Path(f'/root/data/Track{sys.argv[1]}/raw_ply')
frames = []
for p in base.glob('*.ply'):
    match = re.fullmatch(r'(\d+)_ts_(\d+)', p.stem)
    assert match and int(match[2]) > 0, p.name
    frames.append((int(match[2]), int(match[1]), p.name))
frames.sort()
assert len(frames) == len({r[0] for r in frames}), 'Duplicate timestamp'
with (Path(sys.argv[2]) / 'input_frames.csv').open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['output_row', 'timestamp_ns', 'filename_idx', 'filename'])
    w.writerows((i, *r) for i, r in enumerate(frames))
PY

cmd=(ros2 launch localization localization.launch.py
  "data_path:=/root/data/Track${TRACK}"
  "save_path:=$RUN_DIR" save_file:=result.txt
  "max_frames:=$FRAMES" "use_rviz:=$USE_RVIZ"
  "keep_alive:=$KEEP_ALIVE"
  publish_global_map:=false
  ply_scale:=3.0
  map_voxel_leaf_size:=1.0 input_voxel_leaf_size:=0.3
  min_input_points:=10 submap_half_size:=100.0 submap_half_height:=30.0
  ndt_resolution:=8.0 ndt_step_size:=0.8
  ndt_transformation_epsilon:=0.05 ndt_max_iterations:=30)
printf '%q ' "${cmd[@]}" > "$RUN_DIR/command.sh"
printf '\n' >> "$RUN_DIR/command.sh"

TIMEFORMAT='elapsed_seconds=%R'
{ time "${cmd[@]}"; } 2>&1 | tee "$RUN_DIR/run.log"
run_exit=${PIPESTATUS[0]}
printf '%s\n' "$run_exit" > "$RUN_DIR/exit_code.txt"
printf 'Exit code: %s; results: %s\n' "$run_exit" "$RUN_DIR"
```

保持 `map_cache_file` 與 `bag_path` 未指定，沿用空字串停用；資料唯讀不需改權限。`min_input_points` 目前尚未用於點數門檻，單改它不會生效。

在 Ubuntu 主機另存這次版本（把 `你的run目錄名` 換成剛印出的名稱）：

```bash
cd /home/ricky/sdc_ros2
RUN_HOST="midterm01/outputs/runs/你的run目錄名"
git rev-parse HEAD > "$RUN_HOST/commit.txt"
git status --short > "$RUN_HOST/git_status.txt"
git diff --binary > "$RUN_HOST/changes.patch"
```

這些紀錄應在下一次編輯前保存；未追蹤程式不在 git diff 內，上述 `source/` 快照也要保留。

依序做相同設定的 Track1／Track2 單幀、Track1／Track2 前 50 幀，再考慮完整資料。`FRAMES=0` 表示全部可解析 PLY，Track1 預期 3011、Track2 預期 3299。中途錯誤或 Ctrl-C 都可能留下不完整 TXT。

本次既有 50 幀耗時約 Track1 322.5 秒、Track2 118.5 秒，包含地圖讀取。Track1 原始地圖為 ASCII，初始化較久；從 50 幀不能精確推算整條 Track 時間。需要 cache 時另行設計以 Track／map hash／voxel 區分的 outputs 路徑；本次先用停用 cache 的可比較流程。

## 4. 桌面 RViz

在 **Ubuntu 圖形桌面的 Terminal** 輸入以下主機命令。純 SSH／TTY 的 DISPLAY 可能為空；不要猜填 `:0`。目前代理工具所在終端沒有 DISPLAY，不能用它證明 GUI 可用。

```bash
cd /home/ricky/sdc_ros2/midterm01
printenv DISPLAY
xhost +SI:localuser:root
bash scripts/dev.sh --gui up
bash scripts/dev.sh shell
```

這個 `xhost` 設定讓本機 root 容器使用既有 X11／XWayland 顯示；若指令無法連上 display，先處理桌面 session。不要用開放所有連線的 `xhost +`。`--gui up` 可能重建容器，必須在測試前做。

容器重新 source 後，使用第 3 節整段測試，把設定改成：

```bash
TRACK=1
FRAMES=50
LABEL=E01_view
USE_RVIZ=true
KEEP_ALIVE=true
```

這是視覺檢查用的新 run。跑完保留最後一幀，觀察完 Ctrl-C；批次評測用 `USE_RVIZ=false`、`KEEP_ALIVE=false`。GUI keep_alive 的牆鐘耗時包含等待你關閉的時間，不能當成配準效能。

RViz 先檢查以下內容：

| 畫面 | Display／topic |
| --- | --- |
| 影像重建點雲 | PLY RGB：`/image_point_cloud` |
| 局部 LiDAR 地圖 | Submap：`/image_point_cloud_submap` |
| 估計姿態 | Estimated Pose：`/transformed_image_point_cloud_pose` |
| 累積軌跡 | Localization Path：`/localization_path` |
| 全域 LiDAR 地圖 | Global Map：`/map_pc`；只有 `publish_global_map=true` 才發布 |

現有 RViz 設定 `Fixed Frame=image_point_cloud`，偏向跟隨感測器看局部。若觀察世界座標中的軌跡，把 **Global Options → Fixed Frame 改為 `map`**。Views 的 Target Frame 可保留 `image_point_cloud` 來跟隨觀察；切換 Fixed Frame 會清空當前畫面，最好在定位開始前設定。固定世界座標與相機視角是不同設定，見 [ROS 2 Humble RViz 官方說明](https://github.com/ros2/ros2_documentation/blob/humble/source/Tutorials/Intermediate/RViz/RViz-User-Guide/RViz-User-Guide.rst)。

如果點雲太粗，把 PLY RGB 的 Style 改 Points，Size (Pixels) 設 2–3；目前 Spheres 0.5 m 可能遮住錯位。先看牆面、路面是否重合，以及軌跡有無跳躍；看不見資料時先查看各 Display 的 Status、Topic、TF 與 QoS。兩個 PointCloud2 的設定應為 Reliable／Transient Local。

每次只跑一個定位節點，避免同名 topics 與 TF 混在一起。`2D Pose Estimate` 目前沒有對應訂閱者，不會更改初始 pose；RViz 也不控制 C++ 的逐幀迴圈。需要初始姿態請改實驗輸入或程式介面，不要修改原始 init_guess.txt。

如果 RViz 太晚啟動，PointCloud2 可收到保留的最後訊息，但 TF／pose／path 不保證完整補送；優先從 launch 同時啟動。完成桌面測試後，在主機撤回授權：

```bash
xhost -SI:localuser:root
```

舊 `visualization.cpp` 是 Radar／PCD 介面，不能直接用它播放這次的 result.txt。要回看過去的 RGB 點雲與 TF，需事先錄製處理後 ROS topics（包含 `/tf`，以及必要的點雲、pose、path）或另做專用回放器；只有 CSV 無法還原點雲。可參考 [ROS 2 Humble 錄製／回放文件](https://github.com/ros2/ros2_documentation/blob/humble/source/Tutorials/Beginner-CLI-Tools/Recording-And-Playing-Back-Data/Recording-And-Playing-Back-Data.rst)。

## 5. 輸出與檢查

定位程式自動寫 `$RUN_DIR/result.txt`，不需在 RViz 按 Save。每行是一個 3×4 矩陣，共 12 個值；沒有 timestamp。第 3 節另存完整輸入索引 `input_frames.csv`，短序列的結果依序對應其中前 N 筆。

在跑完批次測試的同一個容器終端：

```bash
python3 - "$RUN_DIR/result.txt" "$FRAMES" "$TRACK" <<'PY'
from pathlib import Path
import math, sys
path, frames, track = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
expected = frames if frames else {1: 3011, 2: 3299}[track]
rows = [list(map(float, line.split())) for line in path.read_text().splitlines()]
assert len(rows) == expected, (len(rows), expected)
assert all(len(r) == 12 and all(math.isfinite(v) for v in r) for r in rows)
print(f'PASS: {len(rows)} poses, 12 finite values each')
PY

python3 src/localization/processing_codes/txt_to_idx_x_y_yaw.py \
  "$RUN_DIR/result.txt" "$RUN_DIR/result.csv"
```

CSV 為 `idx,x,y,yaw`，目前 yaw 是弧度、idx 由 0 起。50 幀 CSV 僅供診斷，不能當成完整比賽提交；正式提交前還需依官方 sample／公告確認 ID 集合、frame、尺度與 yaw 單位。完整 Track1／Track2 分別需 3011／3299 資料列，不含表頭。

## 6. 比較修改前後

固定同一 Track、同一段時間／檔名序列、尺度、地圖 voxel、編譯設定、cache 與 GUI 模式，每次只改一個因素。`ndt_resolution:=8.0` 改成另一值是實驗，不是已知會改善。

先比較輸出完整性、fallback、耗時，再看相鄰 pose 的突跳、點雲重合與軌跡；最後用 Kaggle 實際分數判斷競賽效果。收斂與 PCL fitness 都不是正式 ATE，GNSS 也不能直接當作評分真值。

現有前 50 幀參考檔：

```text
/root/outputs/runs/handoff_20261017/track1_50frames/diagnostic.csv
/root/outputs/runs/handoff_20261017/track2_50frames/diagnostic.csv
```

在新 50 幀測試的容器終端，以下程式比較與舊版的差異；先確認兩者 frame 順序一致，不只看 idx 相同：

```bash
python3 - \
  "/root/outputs/runs/handoff_20261017/track${TRACK}_50frames/diagnostic.csv" \
  "$RUN_DIR/result.csv" <<'PY'
import csv, math, sys
def read(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ['idx', 'x', 'y', 'yaw']
        rows = [(int(r['idx']), float(r['x']), float(r['y']), float(r['yaw'])) for r in reader]
    assert rows and all(all(math.isfinite(v) for v in r[1:]) for r in rows)
    assert len({r[0] for r in rows}) == len(rows)
    return rows
a, b = read(sys.argv[1]), read(sys.argv[2])
assert len(a) == len(b), 'Frame count differs'
assert [r[0] for r in a] == [r[0] for r in b], 'IDs differ'
xy = [(u[1]-v[1])**2 + (u[2]-v[2])**2 for u, v in zip(a, b)]
yaw = [math.atan2(math.sin(u[3]-v[3]), math.cos(u[3]-v[3])) for u, v in zip(a, b)]
print('XY RMS difference (m):', math.sqrt(sum(xy)/len(xy)))
print('Yaw RMS difference (deg):', math.degrees(math.sqrt(sum(v*v for v in yaw)/len(yaw))))
for label, rows in [('Reference', a), ('New', b)]:
    steps = [math.hypot(v[1]-u[1], v[2]-u[2]) for u, v in zip(rows, rows[1:])]
    print(label, 'max adjacent XY step (m):', max(steps, default=0))
print('Differences from a previous estimate are not ground-truth errors.')
PY
```

RMS difference 只能表示新舊結果差多少；小不一定準、大不一定錯。舊版已有 Track1 約 0.796 m 的相鄰 3D 平移步長，以及 Track2 約 0.507 rad 的首末 yaw 變化，值得優先檢查。上面程式計算的是 XY 步長，與 3D 步長不同。

完整結果驗證後，由你在對應 Kaggle Track 上傳候選 CSV，記錄檔案 SHA256、版本、參數、提交時間與實際分數到 `experiments/README.md`。確認公式／方向與助教 baseline 數字後才寫「超過 baseline」。

建議你的第一輪操作：保持程式不變，重跑兩條前 50 幀並打開 RViz 熟悉資料；下一輪只改一項參數，做同樣比較。確定要改進的失敗現象後，再實作運動預測、品質門檻或其他方法。

# Midterm01：影像點雲與 LiDAR 地圖定位

單人完成兩條 Track：把影像重建的 3D 點雲對齊各自的 LiDAR 地圖，估計每幀車輛 pose，提交 Kaggle，並交付程式、報告與簡報。

手動編輯、建置、單幀／50 幀／完整測試、RViz、轉檔與結果比較，請依 [手動開發操作手冊](docs/MANUAL_WORKFLOW.md) 操作。

## Ubuntu 實測進度（2026-10-17）

已在 Ubuntu 完成 Git／資料核對、Docker 與 ROS package 建置；兩條 Track 的單幀和 50 幀測試均完成，輸出列數／有限值通過，NDT fallback 均為 0。單幀耗時 Track1 97.6／Track2 14.9 秒；50 幀耗時 Track1 322.5／Track2 118.5 秒（含地圖讀取）。兩條 Track 單幀與 50 幀實跑的參數、耗時及驗證見 [首跑報告](docs/UBUNTU_FIRST_RUN_20261017.md)，完整證據在 `outputs/runs/handoff_20261017/`。未取得 Kaggle 分數；尺度與車體提交 frame 尚未完成官方確認。

## 交接背景（2026-10-16，歷史紀錄）

**下一位執行者為 Ubuntu 裡的 Codex。模板與資料已建立，請從現有資料檢查與 NDT 參考版開始。**

| 項目 | 目前狀態與證據 |
| --- | --- |
| 官方題目 | [比賽 PDF](<docs/SDC 2026 Midterm01 Competition -  Localization.pdf>)，本次讀取全部 24 頁，另核對 p.19–22 的公式與交付圖像 |
| ROS package | `workspace/src/localization/` 已有 12 個程式／設定檔；包含基礎 NDT、地圖處理與 CSV 轉換 |
| Ubuntu 資料 | 使用者確認位於 `~/sdc_ros2/midterm01/data`，截圖有兩條 Track 與兩組 MapAnything 點雲 |
| 資料完整性 | 尚未由本次工具在 Ubuntu 檢查檔案數量、非空、timestamp、幾何或 sample submission |
| 執行環境 | 有獨立 Docker 與建置腳本；既有 Ubuntu／ROS 環境由使用者確認，本次未連線查驗或建置 |
| 定位與評分 | 尚無本次可核對的實跑紀錄、兩條 Track 分數或已超過 baseline 的證據 |

本次可直接讀寫的是 Windows checkout；其中 `data/Track1`、`data/Track2` 為空，不能據此判定 Ubuntu 缺少資料。先保留 Ubuntu 已建立的資料，不重新下載或搬移。

閱讀根目錄 [AGENTS.md](../AGENTS.md)，再看 [題目分析與交付要求](docs/README.md)、[Kaggle 資料與提交規格](docs/KAGGLE_DATASET_SPEC.md)、[資料樹與檢查](data/README.md)。[資源清單](docs/DOWNLOADS.md) 用於查漏補缺。

## 專案結構與三層路徑

```text
midterm01/
├── Dockerfile / docker-compose.yml / docker-compose.gui.yml
├── .env.example                  # Ubuntu 資料根目錄與容器設定
├── config/tracks.json            # 只供 check_data.py 使用
├── workspace/src/localization/   # 目前 NDT package、launch、轉換工具
├── scripts/
│   ├── dev.sh                    # Ubuntu 主機：容器管理與檢查入口
│   ├── build_workspace.sh        # 容器：colcon build
│   └── check_data.py             # 主機或容器：基本檔案／數量檢查
├── data/                         # Ubuntu 已建立，Git 忽略大型內容
├── outputs/                      # runs、submissions、figures，Git 忽略結果
├── docs/                         # 官方 PDF、題目分析、資源與 Kaggle 規格
├── experiments/README.md         # 實驗版本與分數紀錄
└── reports/README.md             # 報告與簡報安排
```

| Ubuntu 主機來源（相對 `midterm01/`） | Midterm01 容器 | 權限 |
| --- | --- | --- |
| `./workspace` | `/root/midterm` | 讀寫 |
| `.env` 的 `MIDTERM_DATA_DIR`，預設 `./data` | `/root/data` | 唯讀 |
| `./config` | `/root/config` | 唯讀 |
| `./scripts` | `/root/scripts` | 唯讀 |
| `./outputs` | `/root/outputs` | 讀寫 |

Ubuntu 專案已知為 `~/sdc_ros2`；絕對路徑先用 `realpath` 確認。Windows 的 `C:\...` 不可作為 Ubuntu Docker bind mount 來源。Docker 管理指令在 **Ubuntu 主機** 執行；ROS／colcon 在 **配置好的容器** 執行。直接登入 Ubuntu 和 Remote SSH 終端機均可。

獨立 Compose project 為 `sdc_midterm01`、service 為 `midterm01`、預設容器名 `sdc_midterm01`、ROS_DOMAIN_ID 為 1。HW2 的 `ros2_container` 及工作區保留原用途。

## Ubuntu 已建立的資料

以下依使用者截圖與文字確認，**不是本次遠端檔案完整性檢查結果**：

```text
~/sdc_ros2/midterm01/data/
├── Track1/
│   ├── raw_ply/
│   ├── Track1_images/
│   ├── init_guess.txt
│   ├── map.pcd
│   ├── metadata.yaml
│   └── Track1.db3
├── Track2/
│   ├── raw_ply/
│   ├── Track2_images/
│   ├── init_guess.txt
│   ├── map.pcd
│   ├── metadata.yaml
│   └── Track2.db3
├── kaggle_track01/Track1_Mapanything_ply/
│   ├── ply/
│   ├── index.csv
│   └── report.json
└── kaggle_track02/Track2_Mapanything_ply/
    ├── ply/
    ├── index.csv
    └── report.json
```

前相機 `raw_ply/` 是目前程式預設輸入。三相機點雲的實際資料夾名如上，與 PDF 泛稱的 `3cameras_ply/` 不同；不必改名。程式尚未讀取 MapAnything 的 `index.csv`／`report.json`，需核對檔名、尺度、座標與官方 frame 對應後，才改用 `ply_path`。截圖未顯示 `camera_info/` 與官方 sample submission；先搜尋，不直接判定缺少，也不把 MapAnything 索引當成 submission。

## Ubuntu 主機：交接後先檢查

先同步必要文件／程式並保留兩端既有修改。Git 不會同步資料、`.env` 與輸出；未追蹤的 PDF／文件也需明確納入同步。

```bash
cd ~/sdc_ros2
git status --short
cd midterm01
pwd
realpath data
ls -ld data/Track1 data/Track2 \
  data/kaggle_track01/Track1_Mapanything_ply \
  data/kaggle_track02/Track2_Mapanything_ply
python3 scripts/check_data.py --data-root "$(realpath data)" \
  --strict-count --output outputs/runs/data_check_host.json
```

預期前相機 PLY 分別為 **3011／3299**。該工具只檢查 map／init_guess 存在且非空，以及資料夾第一層的 PLY 非空與數量；**不查 bag、校正、timestamp、MapAnything、ID 或 CSV**。更完整的唯讀核對指令見 [資料檢查](data/README.md)。若檢查失敗，先查真正的檔案／解壓層級，不覆寫既有資料。

## Ubuntu 主機：設定與建置

```bash
cd ~/sdc_ros2/midterm01
bash scripts/dev.sh init
```

`init` 只建立目錄，並在 `.env` 不存在時複製範例，不會下載或清空資料。目前資料位置可沿用 `MIDTERM_DATA_DIR=./data`；若改用其他位置，填入 Ubuntu 的實際根目錄，該目錄需同時含 Track1／Track2。確認 `.env` 後：

```bash
bash scripts/dev.sh config
bash scripts/dev.sh build
bash scripts/dev.sh up
bash scripts/dev.sh status
bash scripts/dev.sh check-data --strict-count \
  --output /root/outputs/runs/data_check_container.json
bash scripts/dev.sh build-workspace
bash scripts/dev.sh shell
```

先檢查既有容器與 `config` 解析結果，再 `up`。`build` 缺少 `sdc_hw2_ros2:latest` 時，會使用 `hw2/Dockerfile` 與 entrypoint 建置基底；然後建立 Midterm01 映像。沿用 Ubuntu 既有 Docker 權限，平常需要 sudo 時只對 Docker 動作加 sudo，`init` 以一般帳號執行。`build-workspace` 會確認 package 後執行 `colcon build --symlink-install`。

CMake 依賴含 ROS 2 Humble、PCL、Eigen、Boost、`rosbag2_cpp`／`rosbag2_storage` 等，以 `CMakeLists.txt` 和 `package.xml` 為準；缺少依賴時依實際建置錯誤補齊，不把 Windows 上無法編譯當成 Ubuntu 結果。

## 容器：先用無 GUI 的 NDT 單幀

2026-10-17 已修正 `localization.launch.py`：`use_rviz` 預設 false、`map_cache_file` 預設空字串，輸出預設在 outputs。直接執行 executable 的首跑方式仍適用：不啟用 bag、不指定 cache、不開 RViz，輸出寫到新的獨立 run。

以下為可重現的命令範例；實際測試使用首跑報告中的獨立目錄與完整參數檔。`smoke_track01` 若已存在，換成新 run 名稱，程式會覆寫同名結果。

```bash
# 以下都在 Midterm01 容器內
source /opt/ros/humble/setup.bash
cd /root/midterm
source install/setup.bash
ros2 run localization localization --ros-args \
  -p map_path:=/root/data/Track1 \
  -p ply_path:=/root/data/Track1/raw_ply \
  -p init_guess_file:=/root/data/Track1/init_guess.txt \
  -p save_path:=/root/outputs/runs/smoke_track01 \
  -p save_file:=result.txt \
  -p max_frames:=1 \
  -p keep_alive:=false
```

C++ 的 `map_cache_file` 預設為空，所以此命令停用快取；`bag_path` 預設為空，也不會讀 bag。這可先驗證讀檔、NDT 與寫入，不能證明 pose 正確。確認輸出有 1 行、每行 12 個有限數值，再看配準前後疊圖與座標。短序列改用新的 run 目錄及 `max_frames:=50`。

## 容器：完整兩條 Track 與轉檔

在資料契約、單幀與短序列通過後，使用新的 run 名稱保存 R0。以下 `R0_ndt_20261016` 為命名例，請依實際日期／版本調整。**`ply_scale=3.0` 等預設值尚需實測確認，尤其不能直接套用到 MapAnything。**

```bash
# 容器內；延續已 source 的環境
for track in 1 2; do
  ros2 run localization localization --ros-args \
    -p map_path:="/root/data/Track${track}" \
    -p ply_path:="/root/data/Track${track}/raw_ply" \
    -p init_guess_file:="/root/data/Track${track}/init_guess.txt" \
    -p save_path:="/root/outputs/runs/R0_ndt_20261016/track${track}" \
    -p save_file:=result.txt \
    -p max_frames:=0 \
    -p keep_alive:=false || break
done
```

`max_frames=0` 表示處理所有可解析的 PLY；中途出錯／被中斷可能只有部分結果。執行成功不代表筆數或 ID 完整，必須另外檢查。保存原始碼版本與未提交差異、完整參數、輸出、時間及失敗 frame，參考 [實驗紀錄](experiments/README.md)。

官方轉換工具實際位於 `processing_codes/`；2026-10-17 已移除未使用的 pandas／NumPy／SciPy imports 並驗證轉換，目前只需 Python 標準函式庫：

```bash
# 容器內；先確認兩條完整結果與轉檔依賴
mkdir -p /root/outputs/submissions/R0_ndt_20261016
python3 src/localization/processing_codes/txt_to_idx_x_y_yaw.py \
  /root/outputs/runs/R0_ndt_20261016/track1/result.txt \
  /root/outputs/submissions/R0_ndt_20261016/result_1.csv
python3 src/localization/processing_codes/txt_to_idx_x_y_yaw.py \
  /root/outputs/runs/R0_ndt_20261016/track2/result.txt \
  /root/outputs/submissions/R0_ndt_20261016/result_2.csv
```

CSV 表頭順序 **`idx,x,y,yaw`**，Track01 **3011** 列、Track02 **3299** 列，均不含表頭。轉檔程式以行號產生從 0 起的 ID，yaw 以弧度輸出；這是程式行為，正式 ID 集合／順序、座標與單位仍須核對官方 sample／公告。確認沒有缺漏、重複、NaN、Inf 或額外 index 欄後，再做 Kaggle 提交並記錄實際分數。

## 已核對的程式行為與風險

| 項目 | 現況與接手時的處理 |
| --- | --- |
| PLY 時間 | 第一層 `.ply` 檔名需含 `_ts_<正整數奈秒>`，否則略過；依時間排序，沒有重複時間檢查，也不讀 index.csv |
| 初始姿態 | `init_guess.txt` 前四列為 4×4 矩陣，容許逗號；只初始化第一幀。實際檔案、剛體矩陣與 frame 仍待核對 |
| 尺度與外參 | 所有 source XYZ 先乘 `ply_scale`，預設 3.0；目前主流程未另套相機到車體外參。先確認點雲定義，避免漏套或重複套用 |
| 濾波與局部地圖 | source 清理非有限點、voxel 0.3；map voxel 1.0；依上幀 pose 裁切半寬 100／半高 30。這些已是現有流程 |
| NDT | resolution 8.0、step 0.8、epsilon 0.05、最多 30 iterations。以原始 source 和初始 pose 配準，最終矩陣直接作為 candidate，不再乘一次初始 pose |
| 品質與回退 | 目前只查收斂及有限值；NDT 失敗保留前一 pose。`min_input_points=10` 已宣告但未用於點數判斷；空 submap、壞 PLY 等會終止 |
| GNSS／IMU | `bag_path` 才啟用；要求兩個 topic 分別為 PoseStamped／Imu。依 bag timestamp 讀不晚於 PLY 的最新訊息，尚未融合，也未限制訊息過期 |
| 快取 | launch 的 `map_cache_file` 預設空字串停用；若手動啟用，需指定 outputs 中區分 Track／leaf／map 版本的路徑。已有快取時跳過下採樣，不檢查版本是否吻合 |
| RViz／批次 | launch 的 `use_rviz` 預設 false；`keep_alive` 預設 true，跑完仍 spin。批次明確指定 `keep_alive:=false` |
| 結果 | 每幀 12 數值（3×4 矩陣按列攤平），不帶時間／ID；順序錯誤會一路傳到 CSV。預設開檔會覆寫同名結果 |
| 舊視覺化 | `visualization.cpp` 仍讀 Radar／PCD 與占位路徑，不能直接回放本次 PLY；定位節點本身已有 cloud／TF／path |

RViz 需要時先在 **Ubuntu 主機** 確認既有顯示環境，再使用 `bash scripts/dev.sh --gui up` 加入 X11 overlay；`DISPLAY`、X11 socket 與授權要現場核對。容器內可另啟動 `rviz2 -d /root/midterm/src/localization/config/localization.rviz` 配合定位節點；觀察 submap 即可，需全域地圖時才設 `publish_global_map:=true`。

## 接手後的完成順序

1. 保存 Ubuntu 資料檢查報告，核對 raw_ply 與 MapAnything 的時間、尺度、座標、初始矩陣和 sample。
2. 編譯並完成無 GUI 單幀／短序列，處理依賴與快取／launch 落差。
3. 保存目前 NDT 的兩條 Track 完整 R0 和 Kaggle 分數，與助教 baseline 分開記錄。
4. 依失敗片段比較初始姿態預測、品質門檻、重試／回退；再考慮 ICP、多尺度、三相機或 IMU／GNSS，逐項實驗。
5. 固定最佳版本，整理兩條 Track CSV、可重現 README、報告及 4–6 分鐘簡報。

官方 PDF 公告：Kaggle 與簡報寄送 **2026-10-29 09:00**、課堂簡報 **10/29**、E3 程式與報告 **2026-11-05 23:59**；時區與最新公告另核對。兩條 Track 都須超過 baseline。報告需說明自己的貢獻與解題過程，不能把模板既有功能列為自創。詳細評分與交付命名見 [題目分析](docs/README.md)。

## 本次交接文件驗證

已在 Windows 通過 Markdown 本地連結、程式區塊配對、Bash 範例與現有腳本的語法檢查、內嵌 Python 語法、Python 檔案語法、package XML、範例 ROS 參數名稱及 `git diff --check`。`dev.sh help`／`check_data.py --help` 可執行。這些屬靜態與入口檢查，不代表 Ubuntu Docker 建置、C++ 編譯、資料完整性或實際定位已通過。

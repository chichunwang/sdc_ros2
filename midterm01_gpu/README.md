# Midterm01 Track3 GPU 定位

容器名稱 **`midterm01_track3`**。Ubuntu 主機使用 Docker Compose；容器內為 ROS 2 Humble、CUDA 12.8、CuPy 13.6、PyTorch 2.8 cu128，支援本機 RTX 5060（sm_120）。新程式在本目錄，既有 CPU 程式及原始資料保留不動。

預設流程：**官方 LoGeR PLY → CUDA Gaussian voxel 粗配準 → CUDA point-to-plane ICP → IMU／constant-turn Kalman 融合 → 每幀 TXT + Kaggle CSV**。不使用 GNSS、未來frame、離線修軌跡或榜分反推。模型與限制見 [算法說明](doc/ALGORITHM.md)，規則摘要見 [midterm01/AGENTS.md](../midterm01/AGENTS.md)。

**v2 已有真實點雲執行證據，但首段就未可靠配準；目前不建議原樣重跑 full 或提交其輸出。**`track03_full_2610241610` 的可讀前綴到 idx 216，僅 idx 7 採用地圖；該 run 的 `test_info.json` 仍記為 `running`／`frame_count=0`，不是完整 full 成果，也沒有 Kaggle 成績。24 項合成檢查通過不代表真實精度。首幀軸向與幾何分析見[報告](doc/49_DATA_FIRST_FRAME_AUDIT_20261024.md)。scale=3 是沿用 CPU 範例、尚未標定；評分 frame／yaw 單位仍須依 Kaggle 模板核對，也未測得真實資料相對 CPU 的加速比。

## 執行（Ubuntu 主機）

容器已建立，可直接查看或進入：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
./scripts/dev.sh up
./scripts/dev.sh shell
```

完整定位依專案授權規則由使用者親自啟動：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
./scripts/run.sh --full
```

`run.sh` 每次定位（含短段與 full）都預設先啟動 RViz。它使用每輪 run 目錄中的設定複本 `rviz.runtime.rviz`；啟動 log/status 留在 run 目錄。若 RViz GUI 無法啟動，定位不會開始；定位結束後 RViz 保持開啟，畫面是最後收到的資料，不表示 run 成功。可觀察 `/track3/map`、`/track3/submap`、`/track3/current_cloud`、`/track3/pose`、`/track3/path`，以及 `map → track3_camera` TF。地圖只發布一次，點雲以 map frame 顯示；pose 與 CSV 使用同一 SE(3) 轉換，點雲使用同一轉換與本幀尺度。**`map_accepted=0` 期間，path 是 IMU／運動模型預測，不是可靠的地圖定位軌跡。**地圖可以看出道路走廊的大致形狀與轉向，但不能當作逐幀真值，也不能單靠俯視形狀判定pose正確。

每次自動建立唯一的 `outputs/run/track03_full_YYMMDDHHmm/`（碰撞加`_0002`），啟動前凍結程式快照、設定、來源SHA、commit與未提交差異，定位使用快照執行。`flock`防止這個入口同時跑兩份GPU工作。不要同時從其他入口啟動GPU定位。

快照 runner 由自己的檔案位置選取 `track3` 套件，保留 ROS entrypoint 已設定的 `PYTHONPATH`。若曾遇到 `ModuleNotFoundError: rclpy`，更新後重新執行上述命令建立新 run；不修改或重用失敗 run 的快照。

完成後的重點檔案：

| 檔案 | 用途 |
| --- | --- |
| `result_3.csv` | `idx,x,y,yaw`，3142筆、0-based，yaw沿用範例的弧度 |
| `poses.txt` | 每行12個row-major 3×4矩陣值，不含時間戳 |
| `frame_manifest.csv` | 原始索引、整數奈秒、來源檔及SHA256 |
| `timings.csv` | 載入／配準／總frame毫秒、接受狀態與scale |
| `observations.jsonl` | coverage、cost、NIS、退化檢查、IMU數量與elapsed_seconds |
| `test_info.json` | 成敗、版本、分階段耗時、失敗原因、輸出SHA；未有分數時為null |
| `source/` | 本輪使用的程式與設定快照 |
| `analysis/` | 成功full自動產生3張PNG及PDF、README與SHA manifest |

逐幀timing表頭固定為`idx,original_idx,timestamp_ns,backend,load_ms,registration_ms,total_ms,map_accepted,scale`。console的idx是original_idx。準備、暖機、配準、graph（本版無graph）時間分開記錄；畫圖時間在analysis_status.json。第一次配準的lazy JIT編譯列入配準耗時。

失敗或短段預設不產完整分析圖。若定位成功而畫圖失敗，定位狀態及原輸出不變，查看`analysis_status.json`與`analysis.log`。修復後用原run的快照重試（只畫圖，不重跑定位）：

```bash
./scripts/dev.sh exec python outputs/run/track03_full_YYMMDDHHmm/source/scripts/analyze_run.py \
  --run-dir outputs/run/track03_full_YYMMDDHHmm
```

Kaggle提交、選取最終submission、E3上傳與寄信由使用者操作。

## 短段與檢查

以下真實短段命令供使用者手動測試；助理不自行執行未授權定位：

```bash
./scripts/run.sh --max-frames 30
```

只讀資料檢查與合成測試也必須建立新run；不會定位真實資料：

```bash
RUN_DIR="$(python3 scripts/create_run.py --kind test --track 00)"
RUN_PATH="outputs/run/$(basename "$RUN_DIR")"
./scripts/dev.sh exec python scripts/inspect_data.py --record-dir "$RUN_PATH"
./scripts/dev.sh exec env ROS_DOMAIN_ID=34 python -m pytest -q -p no:cacheprovider tests --basetemp "$RUN_PATH/pytest"
```

資料檢查已確認3142個PLY、前視影像與PLY時間戳逐一相同、11307筆IMU；左／右前影像時間戳不同，不能未同步就按索引硬合併。原始data以整個repo唯讀掛載，只有GPU outputs可寫。ASCII地圖約57M點，初次讀圖／SHA計算可能耗時；不回寫地圖cache到data。

## 失敗 run 與目前算法版本

既有 `track03_full_2610241507` 輸出到第2776筆後因局部地圖的稀疏voxel例外中止；其中只有12幀採用地圖觀測，最後一次是 `idx=569`，`570–2775` 連續2206幀未採用。失敗 run 的來源 SHA 共31筆且相符，原始輸出保留。授權的真實診斷上限為30幀：第一次診斷在第0幀後因 NumPy `bool_` 序列化失敗，仍計入一幀；修正後的 `track03_test_2610241547` 實際跑 `idx=0–28` 共29筆（unique 29），未跑 `idx=29`。29筆中2筆更新地圖；25個細 ICP `backtracking_failed` 案例在小於 `1/8` 步長時仍有下降cost的候選。這輪是加儀表的 v1 診斷，不是精度改善證據；CSV前29列與失敗 full 完全相同。紀錄見 `outputs/run/track03_test_2610241535/{failure_audit,diagnostic_summary,authorization_usage}.json` 與 `outputs/run/track03_test_2610241547/`。

目前設定版本為 `track3_cuda_voxel_v2_armijo_turn`。它對稀疏voxel採本幀拒絕並保留原cache；各配準階段使用最多20次二分試算的 Armijo 回溯，只接受足夠下降的候選，且收斂檢查使用未乘線搜尋比例的proposal。coverage、NIS與可觀測性門檻未放寬。IMU模型加入平面constant-turn的圓弧積分與gyro bias／共變異傳播；gyro過期後切回constant-velocity，不外推舊轉率。合成檢查涵蓋上述機制；之後的真實 Track3 full 可讀到 idx 0–216，但只有 idx 7 採用地圖，顯示 v2 從首段就未可靠配準，後續 idx 216 以後狀況未知。完整定位仍由使用者手動啟動；此 v2 快照已有首段失敗證據，**不建議原樣重跑**：`./scripts/run.sh --full`。

## LoGeR

已下載並解壓至 `vendor/LoGeR/`，含checkpoint、老師修改版程式、來源manifest。外層與內層ZIP均已刪除；下載與SHA證據保存在環境run。`vendor/`與`outputs/`不加入Git，也不假設會跨電腦同步。

官方已提供`raw_ply`，**預設定位不需再跑LoGeR**。如要自行重建，使用已整合的因果frontend：

```bash
./scripts/run.sh --full --source loger
# 或使用者手動短段
./scripts/run.sh --max-frames 10 --source loger
```

只用當前及過去2張前視影像，rectify及mask後推論，只保存當前FLU點雲到本run的`reconstructed_ply/`。來源與checkpoint SHA另記`loger_frontend.json`。重建方式與老師offline W32不同，未進行真實影像或精度比較，通常比直接讀PLY慢。8GB GPU採混合bfloat16，TTT的lr_fc與輸出heads保留FP32；記憶體不足時明確失敗。

不要直接拿vendor的`run_pipeline.sh`重建正式逐幀定位輸入：其W32視窗含對早期frame而言的未來資訊，且預設路徑屬老師電腦。原vendor MANIFEST中run_pipeline.sh本來有一筆checksum不符，已保存失敗證據；checkpoint與其公告SHA一致，沒有改寫vendor掩蓋差異。

## 在另一台電腦重建

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
# 若沒有既有本機Humble/CUDA12.8 base，先建立專用base（不動CPU容器）
docker build -f Dockerfile.base -t midterm01_track3_base:cuda12.8 .
# 或直接使用既有 sdc_midterm01_gpu_ros2:cuda12.8
TRACK3_BASE_IMAGE=midterm01_track3_base:cuda12.8 ./scripts/dev.sh build
./scripts/dev.sh up
```

需先由主機安裝可用的NVIDIA driver／NVIDIA Container Toolkit；本次已實際驗證GPU passthrough，不重啟Docker daemon。現有base的來源ID保存在環境run。重新下載LoGeR（尚未存在時）：

```bash
RUN_DIR="$(python3 scripts/create_run.py --kind test --track 00)"
python3 scripts/download_loger.py --record-dir "$RUN_DIR"
```

下載程式需要Python requests，拒絕覆蓋既有vendor，驗證完整大小／CRC後才刪ZIP。依賴版本在requirements.txt及Dockerfile.loger；完整pip freeze與映像ID保存於每輪run。程式是Python ROS 2 bag離線讀取器，無需colcon建置；修改Python後下一輪建立新快照即可。

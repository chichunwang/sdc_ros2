# Midterm01 GPU 主工作區

2026-10-17 起，Midterm01 後續開發、建置與測試統一在 **`/home/ricky/sdc_ros2/midterm01_gpu`**。原 `midterm01/` 保留 CPU 參考程式、既有結果、官方文件與共用資料。

使用者指定的首要目的是學習、自主推導與真實世界應用：先理解問題並選擇適合的算法，必要時才依物理／統計或標定依據調參。後續討論與實作遵循 [學習與算法改進原則](doc/DEVELOPMENT_PRINCIPLES.md)，以可解釋、可驗證及可泛化的方法為主。

目前最佳流程使用 **`--profile sensor_guided --sensor-fusion heading_rts --geometry-mode directional` → `cuda_robust_icp`**：GNSS／IMU 時變 heading 先驗、GPU 最近鄰搜尋、CPU 方向性正則化四自由度細修。程式為重現舊結果，省略 profile 仍使用 `legacy`／`cuda_ndt`，省略 sensor-fusion 仍是 `fixed`，geometry 預設仍為 `off`；推薦命令必須明確帶三個選項。

| 項目 | 已有證據／範圍 |
| --- | --- |
| GPU 環境 | ROS 2 Humble、CUDA 12.8、RTX 5060 `sm_120`；CDI 與真實 CUDA kernel／4096 個結果核對已通過 |
| 新版完整序列 | heading_rts＋directional Track1 3011、Track2 3299 筆，使用者手動完成，本機驗證通過 |
| Track1 Kaggle | 最新使用者截圖 **1.15275**，前最佳 **1.16951**；同圖 baseline **1.79075**，當下第 3 名 |
| Track2 Kaggle | 最新使用者截圖 **0.30576**，前最佳 **0.31967**；同圖 baseline **1.92587**，當下第 1 名 |
| 舊 CPU Track1 | 完整 3011 筆已核對；先前空目錄是 root:root／700 權限造成，已修正可讀 |

日常編輯、跑指定幀數、找結果與 chmod 請看 [GPU 操作手冊](doc/MANUAL_WORKFLOW.md)。可直接貼上的新版命令在 [run_sensor_guided.txt](doc/run_sensor_guided.txt)；[command_paste.txt](doc/command_paste.txt) 保留使用者手動記錄。後續完整測試及 Kaggle 提交由使用者手動執行並回報結果。

改進前的診斷見 [濾波分析](doc/FILTER_ANALYSIS_20261017.md)：既有流程已平滑 GNSS 與 map correction；獨立離線比較工具可另存局部多項式／RTS 後處理候選，這些後處理候選沒有 Kaggle 分數。下列已評分的 heading_rts 是感測器先驗融合改動。

已實作 [感測器融合](doc/SENSOR_FUSION_20261017.md)：`heading_rts` 以有品質門檻的 GNSS course 校正 IMU 慢變 heading bias。使用者已完成兩條完整測試與 Kaggle 提交，分數相對截圖中的前最佳降低 **4.43%／43.00%**；變更、run、CSV SHA256 與成績證據見 [進步紀錄](doc/HEADING_RTS_RESULTS_20261017.md)。排名是截圖快照，不是即時查榜。`motion_rts` 額外估 GNSS 位置／速度，目前僅有短測，沒有其 Kaggle 成績。

已實作並評分的 [幾何診斷與方向性配準](doc/GEOMETRY_DIRECTIONAL_20261017.md)：明確加 `--geometry-mode directional`，以局部表面 PCA 品質及法向／切向 metric 調整地圖殘差。使用者兩條完整測試／提交回報 **1.15275／0.30576**，比前最佳降低 **1.43%／4.35%**，見 [本輪成績、比較與最佳 CSV](doc/GEOMETRY_RESULTS_20261017.md)。完整手動命令見 [run_geometry_directional.txt](doc/run_geometry_directional.txt)。預設仍是 `off`；`diagnostic` 只加診斷，不沿用 directional 的分數。

directional Track2 3299 幀中，有 252 幀拒絕地圖細修並輸出當幀先驗，完整輸出檢查仍通過。第 410 幀的舊 `scan update failed` 是幾何支持保護，詳見 [警告調查與診斷修正](doc/GEOMETRY_FALLBACK_20261017.md)。新程式會記錄候選拒絕原因，保留原配準目標與門檻；此診斷修改不獨立歸功於本輪分數進步。

## 日常使用：Ubuntu 主機

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
bash scripts/dev.sh up

# 修改 C++／CUDA 後重編；只改 --frames 不需重編。
bash scripts/dev.sh build-workspace --packages-select cuda_ndt_backend localization

# 依序完成兩條前 50 幀測試。
python3 scripts/benchmark.py --track 1 --frames 50 --profile sensor_guided --sensor-fusion heading_rts --geometry-mode directional
python3 scripts/benchmark.py --track 2 --frames 50 --profile sensor_guided --sensor-fusion heading_rts --geometry-mode directional
```

以下完整測試由使用者依序手動執行：

```bash
python3 scripts/benchmark.py --track 1 --frames 3011 --profile sensor_guided --sensor-fusion heading_rts --geometry-mode directional
python3 scripts/benchmark.py --track 2 --frames 3299 --profile sensor_guided --sensor-fusion heading_rts --geometry-mode directional
```

`--frames` 只接受正整數，預設 50；benchmark **不以 0 表示全部**。Track 與幀數只選依 timestamp 排序的前 N 幀，不需修改程式。每次產生唯一 run，終端印出主機 `Run:` 路徑；已有 localization 執行時拒絕啟動第二輪。本機已建置，不必每次重建 Docker；執行中不要重建 workspace 或更換容器設定。

## 主機與容器路徑

| 主機來源（相對本目錄） | 容器位置 | 用途／權限 |
| --- | --- | --- |
| `../midterm01/data` | `/root/data` | 唯一原始資料，唯讀 |
| `./workspace` | `/root/midterm` | GPU 程式、build、install、log，讀寫 |
| `../midterm01/config` | `/root/config` | Track 基本檢查設定，唯讀 |
| `../midterm01/scripts/check_data.py` | `/root/reference/check_data.py` | 共用檢查工具，唯讀 |
| `./scripts` | `/root/scripts` | GPU 工具，唯讀 |
| `./outputs` | `/root/outputs` | GPU run、提交候選與圖，讀寫 |

Compose **project** 為 `sdc_midterm01_gpu`、**service** 為 `midterm01_gpu`、**預設容器** 為 `sdc_midterm01_gpu`，映像 `sdc_midterm01_gpu_ros2:cuda12.8`，ROS domain 2；原 CPU domain 1。資料根目錄由 `.env` 的 `MIDTERM_GPU_DATA_DIR` 設定，預設 `../midterm01/data`。`create_host_path: false` 在來源不存在時直接報錯。

GPU workspace 是獨立程式快照，與 CPU 不自動同步，也不共用 build／install。主機 VS Code 開啟 `/home/ricky/sdc_ros2`，可同時檢視兩側原始碼與結果。

**兩個容器的 `/root/outputs` 對應不同目錄**：GPU 是主機 `midterm01_gpu/outputs`，CPU 是 `midterm01/outputs`。你貼的 CPU run 在主機位置：

```text
/home/ricky/sdc_ros2/midterm01/outputs/runs/first_test_9vBOV7le/track1_n0_xLOSykY4
```

原始 `result.txt` 有 3011 行；目前該 run 沒有 CSV，需轉檔才會有。主機讀取權限修復與轉檔見 [操作手冊](doc/MANUAL_WORKFLOW.md#cpu-結果看不到與-chmod)，完成證據見 [CPU 紀錄](../midterm01/docs/CPU_TRACK1_COMPLETED_20261017.md)。

## 輸出與檢查

新版 run 名稱包含 `sensor_guided_cuda_robust_icp`。先查看 `validation.json` 的 `returncode=0`、`output_complete_and_finite=true`、空 `validation_errors` 與預期筆數，再看 `guidance.csv`／軌跡／Kaggle 分數。`result.txt` 是 12 數值矩陣；`diagnostic.csv` 是 `idx,x,y,yaw`，yaw 為弧度。完整 CSV 才能作提交候選；50 幀 CSV 僅供診斷。

每輪另保存完整參數／命令、commit／未提交差異、來源與 binary SHA256、`frames.csv`、`sensor_prior.csv/json`、逐幀時間、GPU 取樣與 log。新版以獨立前處理讀取 bag；ROS `bag_path` 空字串不代表沒使用 GNSS／IMU。`guidance.csv` 記錄地圖修正與目標函數，拒絕更新時輸出當下感測器先驗；CUDA 執行錯誤直接失敗，不切換 CPU 後端。

`pipeline_elapsed_seconds` = `sensor_preparation_seconds` + ROS `elapsed_seconds`；不含建置、版本保存及 map cache 雜湊核對。新版 `timings.csv.ndt_ms` 是沿用欄名，實際為 **CUDA 搜尋＋CPU 正則化細修**；`fitness_ms=0`。`cuda_converged_frames`／`cuda_seed_used_frames` 是舊 D2D 欄位，新版為 0 屬正常。不得把收斂、fitness、GNSS 或地圖殘差當真值精度。

map 快取預設停用。需要時明確加 `--map-cache outputs/runs/<實驗目錄>/cache_track1_leaf1.pcd`，只能在 GPU outputs 下，且核對 Track、原 map SHA256、leaf sidecar。首次建立仍讀原 map；比較速度須使用相同快取策略。Track1 ASCII map 讀取成本大，命中快取的時間不能與未快取結果直接計算 GPU 加速倍率。

## 環境建置與診斷

本機既有環境已通過 GPU runtime，以下為重建／移機入口：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
bash scripts/dev.sh init
bash scripts/dev.sh config
bash scripts/dev.sh build
bash scripts/dev.sh up
bash scripts/dev.sh cuda-check
bash scripts/dev.sh build-workspace
```

Dockerfile 延伸已存在的 CPU 基底 `sdc_midterm01_ros2:latest`，安裝 CUDA 12.8 的 nvcc、cudart 開發套件與 CCCL，編譯 `sm_120` probe；不替換主機 NVIDIA 驅動。缺少 CPU 基底時依 [CPU README](../midterm01/README.md) 的基底建置方式處理。`BUILD_WORKERS` 預設 2，限制 colcon 與各 package 編譯工作數。

若移機後尚未設定 NVIDIA CDI，在該主機執行一次 `sudo bash scripts/setup_host_gpu.sh`，再 `up`／`cuda-check`。依 [NVIDIA 安裝文件](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) 與 [CDI 文件](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html)，腳本設定 Toolkit base 與 CDI，不重啟 Docker daemon。`bash scripts/dev.sh --cpu up` 僅供無 GPU 建置／診斷，不能執行新版 GPU 定位或宣稱 GPU 通過。`cuda-check` 須完成配置、傳輸、kernel 與結果核對，只有 nvidia-smi／nvcc 輸出不算。

```bash
bash scripts/dev.sh doctor
bash scripts/dev.sh status
bash scripts/dev.sh shell
bash scripts/dev.sh stop
```

上述管理入口只操作 GPU project。共用資料基本檢查可用 `bash scripts/dev.sh check-data --strict-count`；要保存 JSON 時依操作手冊建立唯一 run。GUI 尚未現場驗證；GPU dev.sh 沒有 `--gui`，不要套 CPU 手冊的 GUI 指令。

## 文件與歷史比較

| 文件 | 用途 |
| --- | --- |
| [GPU 操作手冊](doc/MANUAL_WORKFLOW.md) | 編輯、建置、幀數、輸出、chmod、CPU 轉檔 |
| [heading_rts 進步紀錄](doc/HEADING_RTS_RESULTS_20261017.md) | 本輪 1.16951／0.31967、改動、完整 run、SHA256 與截圖證據 |
| [SENSOR_GUIDED_20261017.md](SENSOR_GUIDED_20261017.md) | 目前演算法、完整序列、本地限制與使用者回報成績 |
| [架構教學](doc/ALGORITHM_WALKTHROUGH_20261017.md) | 模板 → CUDA D2D → 感測器約束 ICP 的演進 |
| [SETUP_20261017.md](SETUP_20261017.md) | 首建與初始無 GPU 診斷的歷史紀錄 |
| [CUDA_NDT_20261017.md](CUDA_NDT_20261017.md) | 舊 CUDA D2D 整合、Release、快取與速度比較 |
| [ACCURACY_COMPARISON_20261017.md](ACCURACY_COMPARISON_20261017.md) | 舊 CPU／D2D 前 50 幀幾何診斷，不是新版精度 |
| [官方要求](../midterm01/docs/README.md) | PDF、期限、格式、報告與簡報 |

重現舊短序列時才明確使用 `--profile legacy`，例如 `--backend pcl_ndt` 或 `--backend cuda_ndt --cuda-mode D2D`。舊 D2D 完整序列已有漂移／姿態翻轉與 CUDA 記憶體錯誤紀錄，不能當作目前推薦候選；原 PCL 同公式的 CUDA 移植尚未實作。

保持 `p_map = T_map_source × (ply_scale × p_raw)`、初始矩陣／外參只套一次。尺度、GNSS frame 與車體提交 frame 的完整契約仍待官方確認。所有大型資料、`.env`、build／install／log 與輸出依既有 Git 忽略規則保存；最佳 CSV／證據另備份。

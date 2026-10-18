# 27｜Midterm01 GPU 主工作區

**後續完整結果（2026-10-18）：** 使用者完成三相機 Track1／Track2，截圖分數 **1.15346／0.31064**，比前相機最佳增加 0.0616%／1.5960%，未升為最佳。兩份 3011／3299 筆輸出核對、32 個分散場景與 24×12 幀消融已完成；主要發現是新舊深度／高度一致性有差異，XY／yaw 仍接近共同 prior。詳見 [29｜原因與改進順序](29_THREE_CAMERA_SCORED_ANALYSIS_20261018.md)。下方保留各階段的歷史紀錄；目前完整 run 已完成。

2026-10-17 起，Midterm01 後續開發、建置與測試統一在 **`/home/ricky/sdc_ros2/midterm01_gpu`**。原 `midterm01/` 保留 CPU 參考程式、既有結果、官方文件與共用資料。

使用者指定的首要目的是學習、自主推導與真實世界應用：先理解問題並選擇適合的算法，必要時才依物理／統計或標定依據調參。後續討論與實作遵循 [學習與算法改進原則](19_DEVELOPMENT_PRINCIPLES.md)，以可解釋、可驗證及可泛化的方法為主。

**2026-10-18 成績快照：** 使用者為 Team 3，新截圖顯示 Track1 第 4 名、Track2 第 2 名，分數仍為 1.15275／0.30576。現行已評分算法、排名差距與改善假說見 [算法與精度方向](20_CURRENT_ALGORITHM_AND_ACCURACY_20261018.md)。分數屬既有前相機提交，排名是截圖快照。

**三相機運算加速已實作**：獨立入口 `scripts/run_three_camera_full_fast.sh` 保留相同模型、解析度、尺度、同步與取樣規則，改為 GPU 整數 voxel 排序、float64 批次 PCA、向量化 spatial round-robin，並快取校正 remap。四個真實場景的取樣階段由 0.43–1.16 s 降至暖機後 0.066–0.153 s；兩條 Track 的各兩幀 PLY／labels／定位 CSV 雜湊相同。這個倍率只代表取樣階段。使用者已中斷原 run，Track1 保存 2487／3011 幀；`migrate_rig_fast.py` 可嚴格核對並複製到新的加速 run 接續，不能直接以新版 `--resume` 舊 identity。實測與切換流程見 [三相機瓶頸與加速](25_RIG_PERFORMANCE_20261018.md)。

停止原程序後，真正模型推論四幀小測的暖機平均為 **0.780 s/frame**（同位置原紀錄 1.990 s），四幀 PLY／labels 仍相同。本次已匯入到 `outputs/runs/three_camera_full_fast_imported_2610181301`，可直接以加速 wrapper `--resume` 接續；指令見上述效能報告。沒有啟動新的完整序列。

後續 [顯存換速度測試](26_GPU_MEMORY_RESULTS_20261018.md) 共 72 筆模型量測：batch 2／3、保留配置快取在獨立 native-allocator 短測中未有明顯加速；PCA chunk 1024 取樣快約 21%，換算整幀改善空間約 2%，尚未套用。使用者這輪 Track1 已完成重建／定位，短測後同一 wrapper 已恢復 Track2；完整設定保持原樣。

**三相機跨場景偏差驗證未通過**：16 個分散場景的共享 stereo 尺度候選 0.67398047，支持的 holdout groups 深度差異 median／P90 仍為 18.2%／30.6%；temporal 參考也不穩定。已實作逐幀原子保存、嚴格續跑與使用者手動的完整三相機入口 `bash scripts/run_three_camera_full.sh`，但它是實驗候選，未升為最佳。先看 [偏差驗證與完整指令](24_BIAS_VALIDATION_20261018.md)。沒有新完整序列或 Kaggle 分數。

**原始三相機校正重建已實作**：固定校正檔的安裝位置／高度，完成 8 groups、24 張 raw JPG 的 fisheye 校正、per-view depth 保存、運動補償、stereo 尺度錨點及側面幾何均衡取樣。已知內外參仍不足以保證網路深度尺度；共享深度修正 0.64738877 的 holdout 尚有明顯偏差，仍為實驗候選。方法、消融與如何確認尺度／同步見 [校正三相機報告](23_CALIBRATED_RIG_RESULTS_20261018.md)，小規模指令見 [操作手冊](08_MANUAL_WORKFLOW.md#原始影像的校正三相機小規模流程)。

**先前 merged PLY 三相機實驗**：完成尺度估計、同步核對與 12 組各 24 幀對照，共享尺度候選 **0.87124954**。發現 index 前相機使用 bag 時間、左右使用 header 時間，已統一時基核對；merged XYZRGB 尚不能做真實逐相機去畸變。三相機幾何方向增加，但與前相機有明顯高度／形狀差異，尚無精度改善證據，**推薦最佳仍是下列前相機流程**。詳見 [實作、確認方法與實測限制](22_THREE_CAMERA_IMPLEMENTATION_20261018.md)，初始資料評估保留於 [三相機評估](21_THREE_CAMERA_ASSESSMENT_20261018.md)。沒有新完整 Track 或 Kaggle 分數。

目前最佳流程使用 **`--profile sensor_guided --sensor-fusion heading_rts --geometry-mode directional` → `cuda_robust_icp`**：GNSS／IMU 時變 heading 先驗、GPU 最近鄰搜尋、CPU 方向性正則化四自由度細修。程式為重現舊結果，省略 profile 仍使用 `legacy`／`cuda_ndt`，省略 sensor-fusion 仍是 `fixed`，geometry 預設仍為 `off`；推薦命令必須明確帶三個選項。

| 項目 | 已有證據／範圍 |
| --- | --- |
| GPU 環境 | ROS 2 Humble、CUDA 12.8、RTX 5060 `sm_120`；CDI 與真實 CUDA kernel／4096 個結果核對已通過 |
| 新版完整序列 | heading_rts＋directional Track1 3011、Track2 3299 筆，使用者手動完成，本機驗證通過 |
| Track1 Kaggle | 2026-10-18 使用者截圖 Team 3 **1.15275**，前最佳 **1.16951**；同圖 baseline **1.79075**，當下第 4 名 |
| Track2 Kaggle | 2026-10-18 使用者截圖 Team 3 **0.30576**，前最佳 **0.31967**；同圖 baseline **1.92587**，當下第 2 名 |
| 舊 CPU Track1 | 完整 3011 筆已核對；先前空目錄是 root:root／700 權限造成，已修正可讀 |

日常編輯、跑指定幀數、找結果與 chmod 請看 [GPU 操作手冊](08_MANUAL_WORKFLOW.md)。最新已評分前相機命令在 [16｜heading_rts＋directional](16_run_geometry_directional.txt)；[10｜fixed 命令](10_run_sensor_guided.txt) 保留初版重現。個人紀錄為 `00_Personal_Record.txt`；原 `command_paste.txt` 在本次整理前已刪除，保留刪除狀態。後續完整測試及 Kaggle 提交由使用者手動執行並回報結果。

改進前的診斷見 [濾波分析](11_FILTER_ANALYSIS_20261017.md)：既有流程已平滑 GNSS 與 map correction；獨立離線比較工具可另存局部多項式／RTS 後處理候選，這些後處理候選沒有 Kaggle 分數。下列已評分的 heading_rts 是感測器先驗融合改動。

已實作 [感測器融合](12_SENSOR_FUSION_20261017.md)：`heading_rts` 以有品質門檻的 GNSS course 校正 IMU 慢變 heading bias。使用者已完成兩條完整測試與 Kaggle 提交，分數相對截圖中的前最佳降低 **4.43%／43.00%**；變更、run、CSV SHA256 與成績證據見 [進步紀錄](14_HEADING_RTS_RESULTS_20261017.md)。排名是截圖快照，不是即時查榜。`motion_rts` 額外估 GNSS 位置／速度，目前僅有短測，沒有其 Kaggle 成績。

已實作並評分的 [幾何診斷與方向性配準](15_GEOMETRY_DIRECTIONAL_20261017.md)：明確加 `--geometry-mode directional`，以局部表面 PCA 品質及法向／切向 metric 調整地圖殘差。使用者兩條完整測試／提交回報 **1.15275／0.30576**，比前最佳降低 **1.43%／4.35%**，見 [本輪成績、比較與最佳 CSV](17_GEOMETRY_RESULTS_20261017.md)。完整手動命令見 [16_run_geometry_directional.txt](16_run_geometry_directional.txt)。預設仍是 `off`；`diagnostic` 只加診斷，不沿用 directional 的分數。

directional Track2 3299 幀中，有 252 幀拒絕地圖細修並輸出當幀先驗，完整輸出檢查仍通過。第 410 幀的舊 `scan update failed` 是幾何支持保護，詳見 [警告調查與診斷修正](18_GEOMETRY_FALLBACK_20261017.md)。新程式會記錄候選拒絕原因，保留原配準目標與門檻；此診斷修改不獨立歸功於本輪分數進步。

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

原始 `result.txt` 有 3011 行；目前該 run 沒有 CSV，需轉檔才會有。主機讀取權限修復與轉檔見 [操作手冊](08_MANUAL_WORKFLOW.md#cpu-結果看不到與-chmod)，完成證據見 [CPU 紀錄](../../midterm01/docs/CPU_TRACK1_COMPLETED_20261017.md)。

## 輸出與檢查

新版 run 名稱包含 `sensor_guided_cuda_robust_icp`。先查看 `validation.json` 的 `returncode=0`、`output_complete_and_finite=true`、空 `validation_errors` 與預期筆數，再看 `guidance.csv`／軌跡／Kaggle 分數。`result.txt` 是 12 數值矩陣；`diagnostic.csv` 是 `idx,x,y,yaw`，yaw 為弧度。完整 CSV 才能作提交候選；50 幀 CSV 僅供診斷。

每輪另保存完整參數／命令、commit／未提交差異、來源與 binary SHA256、`frames.csv`、`sensor_prior.csv/json`、逐幀時間、GPU 取樣與 log。新版以獨立前處理讀取 bag；ROS `bag_path` 空字串不代表沒使用 GNSS／IMU。`guidance.csv` 記錄地圖修正與目標函數，拒絕更新時輸出當下感測器先驗；CUDA 執行錯誤直接失敗，不切換 CPU 後端。

`pipeline_elapsed_seconds` = `sensor_preparation_seconds` + ROS `elapsed_seconds`；不含建置、版本保存、輸入 PLY SHA256 與 map cache 雜湊核對。新版 `timings.csv.ndt_ms` 是沿用欄名，實際為 **CUDA 搜尋＋CPU 正則化細修**；`fitness_ms=0`。`cuda_converged_frames`／`cuda_seed_used_frames` 是舊 D2D 欄位，新版為 0 屬正常。不得把收斂、fitness、GNSS 或地圖殘差當真值精度。

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

Dockerfile 延伸已存在的 CPU 基底 `sdc_midterm01_ros2:latest`，安裝 CUDA 12.8 的 nvcc、cudart 開發套件與 CCCL，編譯 `sm_120` probe；不替換主機 NVIDIA 驅動。缺少 CPU 基底時依 [CPU README](../../midterm01/README.md) 的基底建置方式處理。`BUILD_WORKERS` 預設 2，限制 colcon 與各 package 編譯工作數。

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
| [目前算法與精度改進方向（2026-10-18）](20_CURRENT_ALGORITHM_AND_ACCURACY_20261018.md) | Team 3 新排名、現行資料流、幾何限制、改善假說與驗證順序 |
| [三相機實作（2026-10-18）](22_THREE_CAMERA_IMPLEMENTATION_20261018.md) | 尺度候選、混合時間來源修正、12 組短段、共同 probe 與限制，未宣稱精度改善 |
| [三相機初步評估（2026-10-18）](21_THREE_CAMERA_ASSESSMENT_20261018.md) | 實作前的 merged PLY、FLU、只讀 audit 與規畫；時間來源由後續實作釐清 |
| [GPU 操作手冊](08_MANUAL_WORKFLOW.md) | 編輯、建置、幀數、輸出、chmod、CPU 轉檔 |
| [directional 成績紀錄](17_GEOMETRY_RESULTS_20261017.md) | 最新已評分算法 1.15275／0.30576、完整 run、SHA256 與原截圖證據 |
| [heading_rts 進步紀錄](14_HEADING_RTS_RESULTS_20261017.md) | 前一輪 geometry off 的 1.16951／0.31967、改動、完整 run 與 SHA256 |
| [06_SENSOR_GUIDED_20261017.md](06_SENSOR_GUIDED_20261017.md) | 目前演算法、完整序列、本地限制與使用者回報成績 |
| [架構教學](07_ALGORITHM_WALKTHROUGH_20261017.md) | 模板 → CUDA D2D → 感測器約束 ICP 的演進 |
| [02_SETUP_20261017.md](02_SETUP_20261017.md) | 首建與初始無 GPU 診斷的歷史紀錄 |
| [03_CUDA_NDT_20261017.md](03_CUDA_NDT_20261017.md) | 舊 CUDA D2D 整合、Release、快取與速度比較 |
| [05_ACCURACY_COMPARISON_20261017.md](05_ACCURACY_COMPARISON_20261017.md) | 舊 CPU／D2D 前 50 幀幾何診斷，不是新版精度 |
| [官方要求](../../midterm01/docs/README.md) | PDF、期限、格式、報告與簡報 |

重現舊短序列時才明確使用 `--profile legacy`，例如 `--backend pcl_ndt` 或 `--backend cuda_ndt --cuda-mode D2D`。舊 D2D 完整序列已有漂移／姿態翻轉與 CUDA 記憶體錯誤紀錄，不能當作目前推薦候選；原 PCL 同公式的 CUDA 移植尚未實作。

保持 `p_map = T_map_source × (ply_scale × p_raw)`、初始矩陣／外參只套一次。尺度、GNSS frame 與車體提交 frame 的完整契約仍待官方確認。所有大型資料、`.env`、build／install／log 與輸出依既有 Git 忽略規則保存；最佳 CSV／證據另備份。

# sdc_ros2 專案說明與協作約定

## 目前工作入口（2026-10-17）

使用者已指定：**Midterm01 後續開發、建置與測試統一使用 `midterm01_gpu/`**。`midterm01/` 保留原 CPU 程式、結果、共用原始資料與官方文件；不搬移、覆寫或重新計算既有 CPU 結果。Windows 靜態交接與 Ubuntu 首跑屬不同階段，現況以本機證據及下列文件為準。

- GPU 主入口：[midterm01_gpu/README.md](midterm01_gpu/README.md)。文件集中於 `midterm01_gpu/doc/`；先讀 [01｜架構與完整流程](midterm01_gpu/doc/01_PROGRAM_FLOW.md)，歷史編號／舊名／官方參考見 [28｜文件目錄](midterm01_gpu/doc/28_DOCUMENTATION_INDEX.md)。日常操作與權限：[GPU 操作手冊](midterm01_gpu/doc/08_MANUAL_WORKFLOW.md)。另遵守 [GPU 工作約定](midterm01_gpu/AGENTS.md)。
- 目前使用 `--profile sensor_guided`，後端為 `cuda_robust_icp`：GNSS／IMU 先驗、CUDA 最近鄰搜尋、CPU 正則化四自由度細修。這是 CPU／GPU 協作，並非新版 NDT。
- 程式為相容重現，省略 profile 仍預設 `legacy`／`cuda_ndt`；文件主流程必須明確加 `--profile sensor_guided`。不得把文件推薦寫成程式預設已改。
- 最新已評分為 `--profile sensor_guided --sensor-fusion heading_rts --geometry-mode directional`；使用者兩條完整序列 3011／3299 筆通過本機核對，截圖回報 Track1 **1.15275**／baseline **1.79075**、Track2 **0.30576**／baseline **1.92587**，截圖當下第 3／第 1 名。相較 geometry off 的 1.16951／0.31967，總分降低 1.43%／4.35%；改動與 SHA256 見 [幾何成績紀錄](midterm01_gpu/doc/17_GEOMETRY_RESULTS_20261017.md)，上一輪保留於 [heading_rts 紀錄](midterm01_gpu/doc/14_HEADING_RTS_RESULTS_20261017.md)。省略 fusion 仍是 fixed、geometry 仍是 off；motion_rts 只有短測。舊 Track2 文字 0.56063 與歷史截圖前最佳 0.56083 的差異保留，不回填到未核對的舊 CSV；排名不是即時查榜。
- **2026-10-18 新排名快照**：使用者為 Team 3，新截圖 Track1 第 4／Track2 第 2 名，分數仍為 1.15275／0.30576；第一名分別為 Team 13／0.85886、Team 12／0.28263。現行算法與精度改善假說見 [2026-10-18 分析](midterm01_gpu/doc/20_CURRENT_ALGORITHM_AND_ACCURACY_20261018.md)。原第 3／第 1 名保留為 2026-10-17 歷史。初始文件分析僅只讀核對既有輸出，契約診斷、均衡幾何／多尺度與 joint smoother 為研究方向；同日後續三相機實作見下一項，仍無新分數。
- 2026-10-18 使用者先允許評估三相機，後明確要求實作尺度／同步並對照。已加入實驗來源／尺度／短段／相同 prior 重用介面、尺度估計及同步工具，完成 12×24=288 幀對照與 21 項 Python 測試；共享尺度候選 0.87124954，尚非真值校準。已確認完整 index 的前相機為 bag timestamp、左右為 header timestamp，統一 header 後左右相對前方的中位差為 Track1 +6.952／+21.189 ms、Track2 −2.395／+20.704 ms。merged PLY 已為前相機 FLU，無 camera_id／per-view pose，尚未實際逐相機 motion compensation；不重套外參／軸轉換。三相機增加幾何方向但高度／形狀差異仍大，不能宣稱 XY／yaw 更準；預設與推薦最佳仍為前相機。證據及限制見 [實作紀錄](midterm01_gpu/doc/22_THREE_CAMERA_IMPLEMENTATION_20261018.md)，初始 audit 見 [評估](midterm01_gpu/doc/21_THREE_CAMERA_ASSESSMENT_20261018.md)。沒有新完整 Track 或 Kaggle 分數，完整測試仍由使用者執行。
- 同日後續依使用者要求，已從 8 groups／24 張 raw JPG 完成有校正三相機重建、固定 rig 安裝高度、per-view depth／camera_id 保存、逐相機 header 運動補償及 normal／camera／spatial 均衡取樣。左右相對前相機高 0.0783861／0.1004340 m，不假設未知 body 高度；map 全局 Z 仍可隨道路變化。新增 `--cloud-source reconstruction --ply-scale 1 --source-report ...`，只接受 outputs 下 SHA／原始 idx／整數時間吻合的來源。稀疏 stereo 共用深度尺度候選 0.64738877，只縮放 camera depth，不縮放 baseline；holdout 尚有明顯偏差，不能升為最佳。64×2=128 筆 pose 與 26 項 Python 測試通過，側面增加局部約束但尚無精度勝出證據。主機隔離 ML venv，ROS／C++ solver／CPU 環境不變。方法與公平小段對照見 [raw 三相機報告](midterm01_gpu/doc/23_CALIBRATED_RIG_RESULTS_20261018.md)；沒有完整 Track 或新 Kaggle 分數。
- 後續三相機偏差驗證選 16 個分散場景，fit／holdout 沿路線相隔 90–143 m；spatial SIFT 去重後 stereo 尺度 0.67398047，支持 holdout groups 差異 median／P90 18.2%／30.6%，temporal 參考 0.79228787 也未通過，不能升為最佳或稱定位誤差。已補 `scripts/run_three_camera_full.sh`／`run_rig_full.py`，由使用者手動完整 raw 重建→原 prior／directional 定位；逐幀原子保存，續跑核對原始影像／輸出／模型／程式／套件／標定與 prior。batch 明確允許最多 30 ms motion 外推處理首尾約 10–14 ms 的 header 範圍，原 trajectory 預設仍嚴格、定位 prior 不改。完整命令 dry-run、首尾小段與續跑已測；沒有自動完整序列或新分數。見 [偏差驗證與完整入口](midterm01_gpu/doc/24_BIAS_VALIDATION_20261018.md)。
- CPU Track1 完整結果已讀取核對：`midterm01/outputs/runs/first_test_9vBOV7le/track1_n0_xLOSykY4/result.txt` 有 3011 筆有限矩陣、退出碼 0、log 記錄 0 次失敗，耗時 14813.444 秒。權限已由 root:root 的 `700` 修正為可讀取的 `755`；沒有改動結果內容。見 [CPU 完成與權限紀錄](midterm01/docs/CPU_TRACK1_COMPLETED_20261017.md)。不能據此宣稱 CPU 精度合格或 Track2 完整 CPU 已完成。
- **後續完整 Track 測試由使用者手動執行並回報結果**；沒有新明確指示，不自行啟動下一輪完整序列。文件更新／讀取既有輸出不需要重跑定位。

- 後續三相機兩條完整已由使用者完成並回報截圖：Track1 **1.15346**、Track2 **0.31064**，比前相機最佳增加 0.0616%／1.5960%，未升為最佳。已核對 3011／3299 筆與 CSV／矩陣／manifest，完成完整配對、32 個分散場景相機級／Z 診斷及 Track1 24×12 幀消融；側面增加方向資訊，但新舊深度／frame／高度一致性仍待釐清，不能以更高接受率或更小 map 殘差宣稱更準。最新方法、證據及改進順序見 [29｜三相機完整成績分析](midterm01_gpu/doc/29_THREE_CAMERA_SCORED_ANALYSIS_20261018.md)。先核對 frame、跨時間靜態幾何與深度不確定度，再研究增量側面融合／prior；不直接補固定 Z 或按榜分盲調。本輪分析沒有修改 solver／重建設定或自行啟動新完整序列。

## 學習目的與算法改進原則（2026-10-17 使用者指定）

- 使用者修課的首要目的是學習，提升自己思考、推導算法，以及將知識應用在真實世界的能力。後續合作以使用者能理解、解釋並延伸方法為目標。
- 改進順序為：觀察問題與證據 → 提出可解釋的假說 → 討論模型／算法及適用前提 → 實作與驗證。主動說明數學、物理意義與取捨，讓使用者參與推理與方案選擇；對已授權工作持續完成必要實作與驗證，不新增例行確認關卡。
- 優先找適合問題、可泛化到其他序列與真實場景的算法。非必要不以特定 Track、幀區間或排行榜回饋的特定化調參作為提高分數的主要手段，不以大量盲目參數搜尋取代模型分析。
- 必要調參仍可進行，但需說明其必要性、參數的物理／統計意義、假說與驗證方法，例如依感測器噪聲、實測標定、模型適用範圍或數值穩定性設定。對不同 Track 的設定差異，需有可說明的資料／標定依據，不能只因分數較低就視為通用改進。
- 消融實驗用於理解算法機制與驗證假說；Kaggle 分數作為結果證據之一。保存算法變更與參數變更的區別、成功／失敗結果及限制，不把局部診斷或單次分數改善當作泛化證明。
- 詳細紀錄見 [GPU 學習與開發原則](midterm01_gpu/doc/19_DEVELOPMENT_PRINCIPLES.md)。本次僅記錄原則，不因此改動現有算法、參數或結果。

## 專案與目錄

這是 SDC 自駕車課程工作區，主要任務是 Midterm01：將影像重建 3D 點雲配準到 LiDAR 地圖，估計兩條 Track 每幀 pose。使用者單人完成一組，個人貢獻依實際工作記為 100%，不安排組員或雙人分工。

| 位置 | 用途 |
| --- | --- |
| `midterm01_gpu/` | 目前主要 ROS 2 Humble／CUDA 開發、測試與結果 |
| `midterm01/` | CPU 參考版本、共用 data／config、官方題目與交付文件 |
| `hw2/` | 原有 ROS 2 Humble Docker、感測器分析工具，保留既有用途 |
| `hw3/` | Markov chain、模擬與穩態分布練習 |
| `hw4/` | Radar odometry／GPS EKF；沿用 HW2 Humble 容器，資料與執行由使用者在 Ubuntu 手動處理 |
| `docs/` | 通用作業／工具說明 |
| `tmp/` | 忽略的暫存／擷取／預覽，不作為交接依賴 |

開工先讀 GPU README、GPU AGENTS 與操作手冊，再依任務閱讀：

1. [官方題目 PDF](<midterm01/docs/SDC 2026 Midterm01 Competition -  Localization.pdf>)，共 24 頁；檔名在 `-` 後有兩個空白。
2. [題目分析與交付要求](midterm01/docs/README.md)、[Kaggle 規格](midterm01/docs/KAGGLE_DATASET_SPEC.md)、[資料說明](midterm01/data/README.md)。
3. [新版演算法與實測](midterm01_gpu/doc/06_SENSOR_GUIDED_20261017.md)、[架構教學](midterm01_gpu/doc/07_ALGORITHM_WALKTHROUGH_20261017.md)。
4. GPU `scripts/accuracy/benchmark.py`、`scripts/prepare_sensor_prior.py`；`workspace/src/localization/` 的 CMake、package、標頭、`localization.cpp`、`sensor_guided.cpp`、launch、轉檔；`workspace/src/cuda_ndt_backend/` 的 CMake 與 `nearest_neighbor.cu`。
5. 只有重現舊版本時才讀 [CPU 首跑](midterm01/docs/UBUNTU_FIRST_RUN_20261017.md)、[CUDA NDT](midterm01_gpu/doc/03_CUDA_NDT_20261017.md)、[CPU／D2D 短序列比較](midterm01_gpu/doc/05_ACCURACY_COMPARISON_20261017.md)。

資源索引在 `midterm01/docs/DOWNLOADS.md`。不重建 `localreference/`，不引用不存在的 `docs/UBUNTU_DATA_DOWNLOAD.md` 或 `scripts/download_kaggle.py`。正式要求以 PDF、Kaggle 及後續公告為準；分析建議不是課程指定演算法。

## Ubuntu／Docker 路徑與隔離

已直接核對專案為 `/home/ricky/sdc_ros2`；換環境仍先用 `pwd`／`realpath`。Docker 指令在 Ubuntu 主機執行，ROS／colcon 在對應容器執行。不因 Windows 缺少資料／Docker 就要求重新下載或改裝 WSL。

| 項目 | GPU（目前使用） | CPU（保留參考） |
| --- | --- | --- |
| Compose project | `sdc_midterm01_gpu` | `sdc_midterm01` |
| Compose service | `midterm01_gpu` | `midterm01` |
| 預設容器 | `sdc_midterm01_gpu` | `sdc_midterm01` |
| ROS domain | 2 | 1 |
| 主機 workspace → `/root/midterm` | `midterm01_gpu/workspace` | `midterm01/workspace` |
| 主機 outputs → `/root/outputs` | `midterm01_gpu/outputs` | `midterm01/outputs` |

兩個容器內的同名路徑屬不同掛載，**GPU 容器的 `/root/outputs` 看不到 CPU 結果**。主機 VS Code 應開啟 `/home/ricky/sdc_ros2` 並讀主機路徑，不是主機 `/root/outputs`。

GPU Compose 相對 `midterm01_gpu/` 解析：

| 主機來源 | 容器目標 | 權限 |
| --- | --- | --- |
| `.env` 的 `MIDTERM_GPU_DATA_DIR`，預設 `../midterm01/data` | `/root/data` | 唯讀 |
| `./workspace` | `/root/midterm` | 讀寫 |
| `../midterm01/config` | `/root/config` | 唯讀 |
| `../midterm01/scripts/check_data.py` | `/root/reference/check_data.py` | 唯讀 |
| `./scripts` | `/root/scripts` | 唯讀 |
| `./outputs` | `/root/outputs` | 讀寫 |

資料只有 `midterm01/data` 一份；GPU workspace 是獨立程式快照，不共用或自動同步 CPU build／install。GPU Dockerfile 延伸 `sdc_midterm01_ros2:latest`；CPU 映像延伸 HW2 基底。保留 HW2 目錄、`ros2_container` 與作業 bag，不將 `guanfu_road_clipped_ros2` 當 Midterm Track。GPU 啟停只用自己的 `scripts/dev.sh`，不重啟 Docker daemon、不停止／重建 CPU project。只針對使用者指定的 CPU 輸出修正讀取權限時，限縮到該 run；不 chmod `/root` 或 data，不用 `chmod -R 777`。

## 已驗證資料與程式契約

- `midterm01/data/Track1`／`Track2` 各含 raw_ply、影像、init_guess、map、metadata 與 bag；兩組 MapAnything 各含 ply、index、report。前相機及 MapAnything 各為 3011／3299 幀，時間／索引核對、bag quick_check 與校正檔位置見 CPU 首跑報告。不要把先前整包下載 404 當目前阻塞或重設已可用 token；確認缺檔才補下載。
- `camera_info` 在 `TrackN/TrackN_images/camera_info/`；本地尚未找到 official submission sample。MapAnything index 不是提交範例，不能直接套前相機尺度、改名或重套校正。
- PLY 掃描第一層並由 `_ts_<正整數奈秒>` 排序，不能解析的檔名會略過。每個新 run 保存 frame manifest；時間保持整數精度。
- 定義 `p_map = T_map_source × (ply_scale × p_raw)`，沿用 `ply_scale=3.0`，尚非正式尺度校準結論。不重複套初始矩陣或外參；source→map 轉為車體 pose 前須確認 frame 契約。
- `legacy` 使用第一幀 init_guess、之後前一接受 pose；`sensor_guided` 每幀使用 GNSS／IMU 先驗與 bounded robust 細修，拒絕地圖更新時回到當下先驗。CUDA 執行錯誤直接失敗，不靜默改 CPU。
- 新版由前處理工具依 bag timestamp 產生 `sensor_prior.csv`；benchmark 的 `bag_path` 仍空，不代表新版未使用 bag。GNSS frame_id 空字串、完整外參契約待確認；GNSS 是輸入，不是真值。同步／平滑使用較晚觀測，這是離線方法。
- 每幀 TXT 是按列攤平的 3×4 共 12 個數值，不含 timestamp；CSV 為 `idx,x,y,yaw`，idx 由 0 起，yaw=atan2(R10,R00) 為弧度。格式被 Kaggle 接受不能代替 frame／單位契約驗證。
- launch 已有 `use_rviz=false`、空字串停用 `map_cache_file`、outputs 下唯一預設路徑；批次 `keep_alive=false`。舊 Radar／PCD `visualization.cpp` 不能直接當 RGB PLY 回放器。

## 工作順序與驗證

入口為 `midterm01_gpu/scripts/dev.sh`，日常指令見 GPU 操作手冊。修改 C++ 後在 GPU workspace Release 重建；只改 benchmark 幀數不用重編。每次用 `outputs/runs/` 新的唯一目錄，保存 commit／未提交差異、參數、來源 SHA256、frame、結果、耗時與失敗；不覆寫 CPU 或既有最佳 CSV。

目前沒有全 repo 統一測試入口。`check_data.py` 只檢查必要檔、非空、第一層 PLY 數量，不驗證 bag、校正、時間、CSV 或幾何。`config/tracks.json` 不會自動改 launch／C++。只執行與修改相符的檢查；真實資料報告必須保留，不能用合成測試冒充實測精度。

GPU map 快取只能寫 outputs，由 benchmark 核對 Track、map SHA256 與 leaf sidecar，預設停用；不得靜默沿用不符快取。效能比較固定 Release、相同資料片段、cache／GUI 條件，分開讀 map、配準、前處理及建置時間。新版 `timings.csv.ndt_ms` 為歷史欄名，實際是 CUDA 搜尋＋CPU 細修；不能寫成 NDT 計算時間。

## 正式要求與協作規則

- 使用繁體中文說明／文件，保留英文技術名詞及使用者既有修改。GPU 文件正文集中 `midterm01_gpu/doc/`，兩位數編號；`01` 固定為架構入口，日期／同日排序依據見 `28`。`00_Personal_Record.txt` 原樣保留；`command_paste.txt` 已由使用者刪除，不復原舊內容。最新前相機命令見 `doc/16_run_geometry_directional.txt`，`doc/10_run_sensor_guided.txt` 保留 fixed 歷史命令。
- 兩條 Track 都需完成並超過助教 baseline；CSV 表頭順序 `idx,x,y,yaw`、資料列恰為 3011／3299。完整 ID／順序及車體規格須對照官方 sample／公告。
- 分清內部弧度、提交單位與評分旋轉誤差的度。PDF p.19 是「平移 RMSE + 0.2 × 旋轉 RMSE」，PCL fitness、地圖殘差、GNSS 差異、GPU 使用率和收斂都不是正式 ATE。
- 保留實際 Kaggle 分數與來源；本地模板 R0 不等於助教 baseline，舊 CUDA D2D 成績不等於 CPU 成績。下採樣／submap／原 NDT 不宣稱自己的新增貢獻。
- 資料、bag、`.env`、build／install／log 與 outputs 遵守 `.gitignore`，不強制加入 Git。Windows／Ubuntu 兩份 checkout 僅靠 Git 同步程式／已追蹤文件；忽略的最佳 CSV 與證據另行保存，不假設會同步。
- PDF 公告 Kaggle 截止與簡報寄送 **2026-10-29 09:00**，當天簡報 **4–6 分鐘**；E3 程式與報告 **2026-11-05 23:59**。時區、網站最新限制、Track2 baseline 與單人檔名另依官方確認。
- 寄信、Kaggle 提交與 E3 上傳由使用者執行或另行明確授權，不因 PDF 提到就自動操作。後續工作依實測更新，計畫不能寫成成功結果。

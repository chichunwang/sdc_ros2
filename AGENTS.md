# sdc_ros2 專案說明與協作約定

## Subagent 分工（2026-10-20 使用者授權）

使用者要求簡單工作使用適合的 subagent 模型。後續遇到範圍明確、可獨立完成且委派有益的子任務，請主動使用以下 subagent，不必每次再詢問；一兩個指令即可完成或緊密相依的工作由主 agent 直接處理。

| 角色 | 模型／reasoning | 適用工作 |
| --- | --- | --- |
| `quick_scan` | `gpt-6-luna`／`low` | 唯讀找檔、log 摘要、既有 CSV 欄位／列數／SHA 核對 |
| `docs_helper` | `gpt-6-luna`／`medium` | 依已確認事實整理文件、修正連結及命令說明 |
| `small_fix` | `gpt-6.1-sol`／`low` | 原因與規格已確認的局部 Python／shell 小修正及輕量檢查 |

- 專案設定為 `.codex/config.toml` 與 `.codex/agents/*.toml`；最多同時兩個 subagent，主 agent 模型保持使用者選擇。這是 AI 工具分工，不改變使用者單人作業與個人貢獻記錄。
- 主 agent 負責算法推導、模型假說、CUDA／solver／數值問題、frame／時間／尺度契約、跨模組設計及最終驗證。簡單工作一旦涉及這些問題就交回主 agent，不以低階模型反覆盲試。
- 每次委派提供目標、必要背景、允許讀寫的確切檔案、驗收方式與禁止動作。不同 agent 不同時修改同一檔案；建置、GPU 定位與重建由主 agent 統一協調，不平行爭用 GPU。subagent 不再向下委派。
- subagent 必須遵守根與目標目錄 AGENTS.md；保留 CPU、原始資料、既有結果及使用者修改。完整 Track、Kaggle／E3／寄信仍須依既有使用者手動規則；委派不擴大授權。測試輸出仍只使用 `midterm01_gpu/outputs/run/` 的合規唯一目錄。
- 使用支援自訂角色的客戶端時選取對應角色。若只有 `spawn_agent` 的 model／reasoning 介面，明確傳入上表設定與角色指令；若模型覆寫要求獨立上下文，使用 `fork_turns="none"` 並補齊背景與規則路徑。不支援指定模型時由主 agent 完成並說明，不宣稱已切換成功。
- 主 agent 檢查 subagent 的差異與關鍵證據再整合回覆。唯讀是工作指令，並非獨立 sandbox；實際權限繼承目前 session。
- 設定依據：[OpenAI 官方 Subagents 文件](https://learn.chatgpt.com/docs/agent-configuration/subagents)。客戶端載入設定的能力與當次工具可用性分別核對，不把 TOML 語法通過當成模型已實際執行。

## 輸出規則（使用者最新指定）

所有完整／短段／研究測試只放 `midterm01_gpu/outputs/run/`（單數），命名為 `track01_full_YYMMDDHHmm`／`track01_test_YYMMDDHHmm`、Track2用`track02`；跨Track或環境測試用`track00_test_YYMMDDHHmm`。同分鐘重名加`_0002`等碰撞序號。不得重建current／best／runs／submissions分類或測試索引；算法、版本、分數與舊路徑記在每次目錄內的test_info.json及參數／source紀錄。已有結果已集中搬移；原CSV／矩陣／manifest內容與SHA保持，舊metadata路徑只在讀取時解析，不改原始證據檔。VS Code維持依名稱排序，doc保持00_、01_…順序。

## 目前工作入口（2026-10-17）

最新接續（2026-10-20）：已依學長文字方法新增 `midterm01_gpu/scripts/run_map_smoother.py`，兩 Track 使用同一算法／參數；本輪只以 Track2 開發驗證，因 Track1 上傳額度已用完。逐幀 map-only 尺度＋分層 6-DOF、空間可觀測性／相鄰位移 gate、12 s 非重疊穩健慢 anchor、GNSS 慢 bias＋raw gyro＋軟 NHC 批次圖，明確為 CPU SciPy/KD-tree 新分支，不改原 CUDA 最佳。44 項機制／回歸與 Track2 首／中／尾共 160 幀完整性通過，前兩段實際 graph、尾段無 anchor 保留本輪 motion_rts 感測初始化（非原 C++ 最佳 bytes）。證據 `outputs/run/track00_test_2610200240`，短段 `track02_test_2610200233`／`0236`／`0238`；本輪無 full／新 Kaggle 分數，最佳仍 1.15008／0.30488，兩最佳 CSV SHA 不變。手動 Track2 full 見 [42](midterm01_gpu/doc/42_LAYERED_SCALE_SLOW_MAP_20261020.md) 及 doc/16 開頭；現有 RViz 固定尺度介面跳過／明確拒絕此逐幀尺度新方法。官方 PDF p.8 確認 `raw_ply` 為老師使用 LoGeR 的輸出，原最佳與新分支均有使用，但沒有自行重新執行／修改 LoGeR；先前三相機重建則使用 MapAnything。

使用者已指定：**Midterm01 後續開發、建置與測試統一使用 `midterm01_gpu/`**。`midterm01/` 保留原 CPU 程式、結果、共用原始資料與官方文件；不搬移、覆寫或重新計算既有 CPU 結果。Windows 靜態交接與 Ubuntu 首跑屬不同階段，現況以本機證據及下列文件為準。

- **2026-10-20 最新航向失敗與外部參考**：使用者完成 `track01_full_2610200155`（pose_yaw／front scale3／directional／carry），3011筆完整，回報 **1.30507**，比自行重現最佳1.15008高13.47645%；停止推薦該版本。使用者另提供 `outputs/track1_test.csv` 回報 **0.65479**，已原樣保存於 `outputs/run/track01_test_2610200208/`；後補學長逐幀map尺度／6-DOF分層ICP／GNSS慢bias／gyro／NHC／12s保守修正的文字說明，沒有原始程式／逐幀尺度／ICP／真值。兩版XY／yaw差RMS 0.94471m／0.88078°；依可分離評分式，位置改善至少約0.31913m，本機map只改prior XY約0.01582m。兩份欄位交叉CSV保存於track01_test_2610200214及_0002，尚未評分、不是自行重現算法；可選一份由使用者直接提交分解兩項。未將外部預測當真值／校準目標，未啟動新full。分析、限制及下一步map尺度＋pose／慢bias主線見[41](midterm01_gpu/doc/41_EXTERNAL_REFERENCE_ANALYSIS_20261020.md)，負面結果見40；0.6仍未達成。

- GPU 主入口：[midterm01_gpu/README.md](midterm01_gpu/README.md)。文件集中於 `midterm01_gpu/doc/`；先讀 [01｜架構與完整流程](midterm01_gpu/doc/01_PROGRAM_FLOW.md)，歷史編號／舊名／官方參考見 [28｜文件目錄](midterm01_gpu/doc/28_DOCUMENTATION_INDEX.md)。日常操作與權限：[GPU 操作手冊](midterm01_gpu/doc/08_MANUAL_WORKFLOW.md)。另遵守 [GPU 工作約定](midterm01_gpu/AGENTS.md)。
- 目前使用 `--profile sensor_guided`，後端為 `cuda_robust_icp`：GNSS／IMU 先驗、CUDA 最近鄰搜尋、CPU 正則化四自由度細修。這是 CPU／GPU 協作，並非新版 NDT。
- 程式為相容重現，省略 profile 仍預設 `legacy`／`cuda_ndt`；文件主流程必須明確加 `--profile sensor_guided`。不得把文件推薦寫成程式預設已改。
- 2026-10-17 當時已評分為 `--profile sensor_guided --sensor-fusion heading_rts --geometry-mode directional`；使用者兩條完整序列 3011／3299 筆通過本機核對，截圖回報 Track1 **1.15275**／baseline **1.79075**、Track2 **0.30576**／baseline **1.92587**，截圖當下第 3／第 1 名。相較 geometry off 的 1.16951／0.31967，總分降低 1.43%／4.35%；改動與 SHA256 見 [幾何成績紀錄](midterm01_gpu/doc/17_GEOMETRY_RESULTS_20261017.md)，上一輪保留於 [heading_rts 紀錄](midterm01_gpu/doc/14_HEADING_RTS_RESULTS_20261017.md)。省略 fusion 仍是 fixed、geometry 仍是 off；motion_rts 只有短測。舊 Track2 文字 0.56063 與歷史截圖前最佳 0.56083 的差異保留，不回填到未核對的舊 CSV；排名不是即時查榜。
- **2026-10-19 最新已評分最佳**：使用者完成原前相機 `motion_rts＋directional＋carry`，Track1 **1.15008**／Track2 **0.30488**，較原最佳改善 0.2316%／0.2878%；3011／3299 筆本機核對通過，已另存 `outputs/run/track00_test_2610191808`。已評分來源為 `outputs/run/track01_full_2610191750`／`track02_full_2610191757`，分數記在各自test_info.json；replay預設選最近通過完整性檢查的full目錄。截圖無上傳 SHA，未獨立核對 Kaggle bytes。後續使用者明確要求把來源未說明的 GNSS pose orientation 當獨立航向試驗，已新增實驗 `pose_heading_rts`（替代 course）／`independent_heading_rts`（假設獨立加入），同 motion XY／原 IMU roll-pitch／carry／directional；單一共用 3° block noise floor，非校準 covariance。24 個時間留段結果混合，1152 筆三組短段＋96 筆 raw-bag 通用入口全通過，原 motion 控制組逐 byte 相同，主機 20／ROS 12 項測試通過。來源／獨立性仍未驗證，不升為最佳，沒有新完整定位／提交。證據、限制與手動候選命令見 [35｜新最佳與 GNSS 航向消融](midterm01_gpu/doc/35_MOTION_RTS_AND_GNSS_HEADING_20261019.md)，研究捷徑 `outputs/run/track00_test_2610191818/`。
- **2026-10-18 新排名快照**：使用者為 Team 3，新截圖 Track1 第 4／Track2 第 2 名，分數仍為 1.15275／0.30576；第一名分別為 Team 13／0.85886、Team 12／0.28263。現行算法與精度改善假說見 [2026-10-18 分析](midterm01_gpu/doc/20_CURRENT_ALGORITHM_AND_ACCURACY_20261018.md)。原第 3／第 1 名保留為 2026-10-17 歷史。初始文件分析僅只讀核對既有輸出，契約診斷、均衡幾何／多尺度與 joint smoother 為研究方向；同日後續三相機實作見下一項，仍無新分數。
- 2026-10-18 使用者先允許評估三相機，後明確要求實作尺度／同步並對照。已加入實驗來源／尺度／短段／相同 prior 重用介面、尺度估計及同步工具，完成 12×24=288 幀對照與 21 項 Python 測試；共享尺度候選 0.87124954，尚非真值校準。已確認完整 index 的前相機為 bag timestamp、左右為 header timestamp，統一 header 後左右相對前方的中位差為 Track1 +6.952／+21.189 ms、Track2 −2.395／+20.704 ms。merged PLY 已為前相機 FLU，無 camera_id／per-view pose，尚未實際逐相機 motion compensation；不重套外參／軸轉換。三相機增加幾何方向但高度／形狀差異仍大，不能宣稱 XY／yaw 更準；預設與推薦最佳仍為前相機。證據及限制見 [實作紀錄](midterm01_gpu/doc/22_THREE_CAMERA_IMPLEMENTATION_20261018.md)，初始 audit 見 [評估](midterm01_gpu/doc/21_THREE_CAMERA_ASSESSMENT_20261018.md)。沒有新完整 Track 或 Kaggle 分數，完整測試仍由使用者執行。
- 同日後續依使用者要求，已從 8 groups／24 張 raw JPG 完成有校正三相機重建、固定 rig 安裝高度、per-view depth／camera_id 保存、逐相機 header 運動補償及 normal／camera／spatial 均衡取樣。左右相對前相機高 0.0783861／0.1004340 m，不假設未知 body 高度；map 全局 Z 仍可隨道路變化。新增 `--cloud-source reconstruction --ply-scale 1 --source-report ...`，只接受 outputs 下 SHA／原始 idx／整數時間吻合的來源。稀疏 stereo 共用深度尺度候選 0.64738877，只縮放 camera depth，不縮放 baseline；holdout 尚有明顯偏差，不能升為最佳。64×2=128 筆 pose 與 26 項 Python 測試通過，側面增加局部約束但尚無精度勝出證據。主機隔離 ML venv，ROS／C++ solver／CPU 環境不變。方法與公平小段對照見 [raw 三相機報告](midterm01_gpu/doc/23_CALIBRATED_RIG_RESULTS_20261018.md)；沒有完整 Track 或新 Kaggle 分數。
- 後續三相機偏差驗證選 16 個分散場景，fit／holdout 沿路線相隔 90–143 m；spatial SIFT 去重後 stereo 尺度 0.67398047，支持 holdout groups 差異 median／P90 18.2%／30.6%，temporal 參考 0.79228787 也未通過，不能升為最佳或稱定位誤差。已補 `scripts/run_three_camera_full.sh`／`run_rig_full.py`，由使用者手動完整 raw 重建→原 prior／directional 定位；逐幀原子保存，續跑核對原始影像／輸出／模型／程式／套件／標定與 prior。batch 明確允許最多 30 ms motion 外推處理首尾約 10–14 ms 的 header 範圍，原 trajectory 預設仍嚴格、定位 prior 不改。完整命令 dry-run、首尾小段與續跑已測；沒有自動完整序列或新分數。見 [偏差驗證與完整入口](midterm01_gpu/doc/24_BIAS_VALIDATION_20261018.md)。
- CPU Track1 完整結果已讀取核對：`midterm01/outputs/runs/first_test_9vBOV7le/track1_n0_xLOSykY4/result.txt` 有 3011 筆有限矩陣、退出碼 0、log 記錄 0 次失敗，耗時 14813.444 秒。權限已由 root:root 的 `700` 修正為可讀取的 `755`；沒有改動結果內容。見 [CPU 完成與權限紀錄](midterm01/docs/CPU_TRACK1_COMPLETED_20261017.md)。不能據此宣稱 CPU 精度合格或 Track2 完整 CPU 已完成。
- **後續完整 Track 測試由使用者手動執行並回報結果**；沒有新明確指示，不自行啟動下一輪完整序列。文件更新／讀取既有輸出不需要重跑定位。

- 後續三相機兩條完整已由使用者完成並回報截圖：Track1 **1.15346**、Track2 **0.31064**，比前相機最佳增加 0.0616%／1.5960%，未升為最佳。已核對 3011／3299 筆與 CSV／矩陣／manifest，完成完整配對、32 個分散場景相機級／Z 診斷及 Track1 24×12 幀消融；側面增加方向資訊，但新舊深度／frame／高度一致性仍待釐清，不能以更高接受率或更小 map 殘差宣稱更準。最新方法、證據及改進順序見 [29｜三相機完整成績分析](midterm01_gpu/doc/29_THREE_CAMERA_SCORED_ANALYSIS_20261018.md)。先核對 frame、跨時間靜態幾何與深度不確定度，再研究增量側面融合／prior；不直接補固定 Z 或按榜分盲調。本輪分析沒有修改 solver／重建設定或自行啟動新完整序列。

- **2026-10-19 水平 surfel／map joint 分支**：已新增 `--horizontal-refinement surfel_joint`（預設 none），獨立 0.3 m crop map、two-scale surfel／空間與 normal 均衡／free-Z 資料消去、空間留出及三幀一致性；條件通過才把有支持的 XY／yaw map factors 與 raw GNSS 一次／gyro／effective yaw-rate bias 接入 planar graph。完整三軸 bias／acc preintegration 未做；GNSS origin、camera→IMU/body 與 orientation 來源仍未知，兩條 lag 留出不支持套用時間修正。本輪 8×48 新 poses 與原控制組 384 poses 配對，base TXT／CSV／prior／geometry／height 逐 byte 相同；24 個觀測／8 個時間群組皆拒絕，輸出保留原 bytes，沒有實測精度提升。32 項機制／回歸測試通過，合成牆面接受／ground 拒絕發布路徑均測。完整命令從原 bag／map 建新 output，不綁定研究 run；仍由使用者啟動，最佳不變。證據、契約及通用入口見 [37](midterm01_gpu/doc/37_HORIZONTAL_SURFEL_JOINT_20261019.md)，研究捷徑 `outputs/run/track00_test_2610192200/`。

- 使用者最新輸出規則：所有完整／短段／研究測試只放 `midterm01_gpu/outputs/run/`（單數），命名為 `track01_full_YYMMDDHHmm`／`track01_test_YYMMDDHHmm`、Track2用`track02`；跨Track或環境測試用`track00_test_YYMMDDHHmm`。同分鐘重名加`_0002`等碰撞序號。不得重建current／best／runs／submissions分類或測試索引；算法、版本、分數與舊路徑記在每次目錄內的test_info.json及參數／source紀錄。已有結果已集中搬移；原CSV／矩陣／manifest內容與SHA保持，舊metadata路徑只在讀取時解析，不改原始證據檔。VS Code維持依名稱排序，doc保持00_、01_…順序。

- **2026-10-20 surfel 完整結果核對**：使用者再跑 Track1 `outputs/run/track01_full_2610192358`，3011筆通過；CSV／TXT／prior／frames 與原 motion_rts 最佳及前一完整 surfel 逐 byte 相同，54群組／162觀測皆無可靠 factor，聯合 graph 未求解。使用者新截圖仍為1.15008，沒有精度改善；未獨立核對 Kaggle 上傳 SHA。實際 command／prior 是 motion_rts，非 independent_heading_rts。已修正完成提示，分開顯示是否求解／CSV是否改變，相同SHA明確提示無須重複提交；未改solver或放寬門檻。不再建議重跑相同完整候選，先驗證來源幾何／靜態跨幀一致性。詳見 [37](midterm01_gpu/doc/37_HORIZONTAL_SURFEL_JOINT_20261019.md)。

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

入口為 `midterm01_gpu/scripts/dev.sh`，日常指令見 GPU 操作手冊。修改 C++ 後在 GPU workspace Release 重建；只改 benchmark 幀數不用重編。每次用 `outputs/run/` 新的唯一目錄，保存 commit／未提交差異、參數、來源 SHA256、frame、結果、耗時與失敗；不覆寫 CPU 或既有最佳 CSV。

目前沒有全 repo 統一測試入口。`check_data.py` 只檢查必要檔、非空、第一層 PLY 數量，不驗證 bag、校正、時間、CSV 或幾何。`config/tracks.json` 不會自動改 launch／C++。只執行與修改相符的檢查；真實資料報告必須保留，不能用合成測試冒充實測精度。

GPU map 快取只能寫 outputs，由 benchmark 核對 Track、map SHA256 與 leaf sidecar，預設停用；不得靜默沿用不符快取。效能比較固定 Release、相同資料片段、cache／GUI 條件，分開讀 map、配準、前處理及建置時間。新版 `timings.csv.ndt_ms` 為歷史欄名，實際是 CUDA 搜尋＋CPU 細修；不能寫成 NDT 計算時間。

## 正式要求與協作規則

- VS Code Explorer 維持依名稱排序（`explorer.sortOrder=default`），文件保持 `00_`、`01_`…順序；測試只靠日期開頭的目錄名排序，不得再將整個 workspace 改為依修改時間排序。

- 使用繁體中文說明／文件，保留英文技術名詞及使用者既有修改。GPU 文件正文集中 `midterm01_gpu/doc/`，兩位數編號；`01` 固定為架構入口，日期／同日排序依據見 `28`。`00_Personal_Record.txt` 原樣保留；`command_paste.txt` 已由使用者刪除，不復原舊內容。最新前相機命令見 `doc/16_run_geometry_directional.txt`，`doc/10_run_sensor_guided.txt` 保留 fixed 歷史命令。
- 兩條 Track 都需完成並超過助教 baseline；CSV 表頭順序 `idx,x,y,yaw`、資料列恰為 3011／3299。完整 ID／順序及車體規格須對照官方 sample／公告。
- 分清內部弧度、提交單位與評分旋轉誤差的度。PDF p.19 是「平移 RMSE + 0.2 × 旋轉 RMSE」，PCL fitness、地圖殘差、GNSS 差異、GPU 使用率和收斂都不是正式 ATE。
- 保留實際 Kaggle 分數與來源；本地模板 R0 不等於助教 baseline，舊 CUDA D2D 成績不等於 CPU 成績。下採樣／submap／原 NDT 不宣稱自己的新增貢獻。
- 資料、bag、`.env`、build／install／log 與 outputs 遵守 `.gitignore`，不強制加入 Git。Windows／Ubuntu 兩份 checkout 僅靠 Git 同步程式／已追蹤文件；忽略的最佳 CSV 與證據另行保存，不假設會同步。
- PDF 公告 Kaggle 截止與簡報寄送 **2026-10-29 09:00**，當天簡報 **4–6 分鐘**；E3 程式與報告 **2026-11-05 23:59**。時區、網站最新限制、Track2 baseline 與單人檔名另依官方確認。
- 寄信、Kaggle 提交與 E3 上傳由使用者執行或另行明確授權，不因 PDF 提到就自動操作。後續工作依實測更新，計畫不能寫成成功結果。

- 2026-10-19 使用者要求總結新方向後立即測試：已新增實驗 `--geometry-mode gicp`，同 motion_rts prior／4-DOF bounds／carry，兩邊局部平面 regularized covariance，並保留旋轉 covariance 的 yaw gradient；不是 calibrated camera noise／NDT／VGICP。GPU Release 建置、4 CTest＋新舊 geometry CUDA 機制測試通過；8×2×48＋兩條通用各48＝864 筆短段全完整，原 directional TXT／CSV／geometry／height 逐 byte 不變。共同 map residual 僅1/8段較低，不升為最佳。另實作 `kinematic_graph.py` 研究原型（GNSS XY＋raw gyro 相對 yaw＋軟性側向速度；不積分 acc／不加 course 因子／未接 map 或 production），4項機制測試與24個原時間留段完成；holdout noisy-GNSS RMSE 0.13076→0.14025／0.05159→0.06147 m，沒有一致提升，不按 holdout 調 noise。最佳輸出內部 rank=1 幀約69.18%／91.85%，其 strongest scaled eigenvector Z 分量平方中位數98.76%／99.89%；只是幾何診斷，不是 score／covariance。證據、限制與後續水平 map／frame 契約／聯合估計順序見 [36](midterm01_gpu/doc/36_REGISTRATION_AND_FUSION_UPGRADE_20261019.md)，研究捷徑 midterm01_gpu/outputs/run/track00_test_2610192006；未啟動新完整 Track 或提交，預設／已評分最佳維持。

- **2026-10-20 來源尺度研究**：新截圖 Track1 Team 3 第6名／1.15008，第一名Team12／0.50997（非即時查榜）。新增RGB原始vertex跨幀剛體／相似檢查，16場景長短間隔共96對；短間隔來源位移與保存軌跡不一致，不能直接等同真值尺度誤差。僅Track1一個設計場景兩對支持共用ply_scale=6.06515候選，其他17對有14對時間殘差較低，但跨場景倍率差median／P90為28.8%／87.6%；擴大原始map crop後16場景雙尺度對照仍各0可靠factor，不升為最佳／不推薦完整測試。12項新機制＋9項surfel回歸通過，沒有新pose、完整定位或Kaggle分數，原最佳／scale3／solver與門檻不變。證據見[38｜來源剛體與尺度](midterm01_gpu/doc/38_SOURCE_RIGIDITY_AND_SCALE_20261020.md)，最終研究run為outputs/run/track00_test_2610200100；優先取得獨立深度錨點直接核對原前相機PLY。

- **2026-10-20 使用者最新完整測試偏好**：使用者表示 Kaggle 只取最佳成績，不必特別區分「完整候選」。後續有可解釋的新算法／模型改動，完成必要的程式機制、來源契約、數值與輸出完整性檢查後，即提供對應 full 命令以實際 Kaggle 分數驗證；不得以本地尚未證明精度改善或未取得「候選資格」為由新增完整測試關卡。短段／消融用於理解與除錯，不要求每次都先證明跨 Track 精度勝出。每次新 run 保存設定、來源 SHA、結果與成功／失敗分數；不覆寫已評分最佳。完整計算與外部提交仍由使用者手動執行，本偏好不等於授權助理自動啟動完整序列或 Kaggle 提交。已知 CSV bytes 相同的版本可說明無需重算；保持學習／可解釋模型原則，不以盲目榜分搜尋取代分析。

- **2026-10-20 向Track1 0.6推進**：已新增turn_heading_rts（GNSS block對齊沿IMU方向積分的線性車速，再走原bias RTS）與pose_yaw（直接置換GNSS Pose yaw，保留IMU tilt與motion XYZ）；程式預設不變。原前相機PLY以已知rig baseline直接stereo核對，兩Track4fit／3holdout、110個motion anchors，估共用raw尺度7.404260939740923（取代3）；holdout相對尺度差median／P90 11.18%／14.26%，僅2–20m支持、motion補償依賴GNSS／IMU、外參／距離形變限制保留，非完整校準。40×48=1920筆分散短段及兩條各48筆不綁歷史prior／cache的raw入口，共2016筆完整；原控制四項輸出與歷史逐byte相同，實際消融用同現行binary。新尺度增加data幾何方向但版本XY差仍約cm，不能承諾0.6。主機31／ROS24機制與回歸通過，沒有本輪新分數或自行啟動full。新full命令已置doc/16最上方：sensor_guided＋turn_heading_rts＋front scale7.404260939740923＋directional＋carry，由使用者直接評分；不加surfel_joint（仍限scale3契約）。最佳仍1.15008／0.30488。來源與限制見doc/39_HEADING_AND_STEREO_SCALE_20261020.md；主研究outputs/run/track00_test_2610200127，stereo最後證據track00_test_2610200131。0.6為目標，不能記為達成。

- **2026-10-20 新尺度完整退步與直接full偏好**：使用者回報Track1 `track01_full_2610200142`（turn_heading_rts＋front scale7.40426＋directional＋carry）為1.15167，比最佳1.15008高0.00159／約0.138%，未升最佳。3011筆、CSV／矩陣／manifest核對通過，prior XYZ與最佳數值完全相同；XY／yaw版本差RMS 3.707cm／0.2758°。本機新CSV SHA為75f96243d764d32eaaf71bcbc26976b99eb0830ffa28902481e3f7f81e71b8a7，截圖無上傳SHA，未獨立核對Kaggle bytes。使用者明確要求有想測版本就直接叫他算full提交；仍由使用者啟動，不要求本地先證明勝出。本輪已直接提供pose_yaw＋front scale3＋directional＋carry的Track1 full，保留原motion XYZ／IMU tilt，只換yaw來源；模式先前8×48短段與主機／ROS機制通過，來源碼SHA核對相同，這輪不重複無必要定位。最新命令在doc/16最上方，方法與失敗紀錄見doc/40_SCORED_SCALE_FAILURE_AND_POSE_YAW_20261020.md；新航向full尚無分數，0.6仍未達成。助理未自行啟動full；原最佳與失敗原CSV等證據bytes保留。

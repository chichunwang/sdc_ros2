# Midterm01 GPU 工作區

文件已集中到 **[doc/](doc/28_DOCUMENTATION_INDEX.md)**，以兩位數編號排列。

常用命令速查：[50｜指令表](doc/50_COMMAND_REFERENCE.md)（Track3 v2 full、可選短段、資料盤點、結果與圖表）。

2026-10-23 [Track3 前向 IMU Kalman／配準拒絕／逐幀尺度](doc/52_TRACK3_CAUSAL_KALMAN_SCALE_20261023.md) 的現行入口已更新為 v2；初始化修正與驗證見 [53](doc/53_TRACK3_INITIALIZATION_REPAIR_20261023.md)。v1 full `track03_full_2610231521` 使用者回報 **82.17**，3142 筆輸出完整，但地圖觀測採用 0 筆、速度為零、位置固定初值；不推薦原樣重跑。v2 的 59 項檢查均為 CPU／CUDA 機制或合成測試，沒有真實 v2 定位、新分數或精度改善證據。配準由程式自動完成，`initial-scale 3` 只是 PLY XYZ 的未校準起始倍率。只有 full 提交產生正式分數；短段可選、不設 full 資格門檻。建議 full 命令見 [50](doc/50_COMMAND_REFERENCE.md)，由使用者手動執行；score <1.29 是目標，不保證達成。

2026-10-23 Track3 本機資料已盤點：3142 個 PLY 的原始索引與 bag 前相機時間逐筆吻合，map／初始矩陣存在，bag 無 GNSS。已開放既有 map-only `legacy` 的 Track3 短段入口並完成乾跑；尚未實際定位，尺度與提交 frame／yaw 契約未驗證。手動 50 幀命令與因果審查見 [49｜Track3 因果 baseline](doc/49_TRACK3_CAUSAL_BASELINE_20261023.md)。此入口不使用歷史 RTS／graph；下文非因果方法警示仍有效。

Python／Matplotlib 的 full run 事後圖表，隨每次成功完成的 full run 寫入該 run 的 `analysis/`；格式、來源與限制見 [08｜操作手冊](doc/08_MANUAL_WORKFLOW.md#full-run-事後圖表)。定位後圖表只讀診斷，不修正軌跡。舊教學素材與圖表說明保留於 [doc/analysis/README.md](doc/analysis/README.md)，不補繪歷史 run。

2026-10-23 Track3 資料說明更新：使用者轉貼的 Kaggle 原文已確認三相機 H265、`/imu/data`（orientation／acceleration）、CSV 表頭 `idx,x,y,yaw` 與 3142 筆資料（不含表頭；含表頭共 3143 筆紀錄）。完整原文與來源範圍見 [資料規格](../midterm01/docs/KAGGLE_DATASET_SPEC.md)。公告確認不代表本機 bag／sample 或因果 runner 已驗證。

**現行規則（2026-10-23）：**[00｜整合新版（31 頁）](<doc/Competition Regulations/00_SDC_2026_Midterm01_Localization_整合新版_20261022.pdf>) 與 [01｜因果定位細則（4 頁）](<doc/Competition Regulations/01_Mid01_Updates_Rules.pdf>) 優先；[90](<doc/Competition Regulations/90_舊版_已取代_可刪除_SDC_2026_Midterm01_Localization_24頁.pdf>)／[91](<doc/Competition Regulations/91_舊版_已取代_可刪除_Mid01_Updates_Adjustments_20261021.pdf>) 僅保留歷史。因果規則禁止未來資訊回修 pose、RTS／backward smoothing 與定位後離線修正，所有 Track 適用；Track1／2 GNSS 仍可輸入，但 RTS 不合規。現有 RTS、map batch graph 與離線 joint 分支不可用於正式提交，舊分數只作歷史紀錄。規則與程式盤點見 [48](doc/48_RULES_AND_CODE_AUDIT_20261023.md)；此文件更新不授權新 full。

**歷史分數提示：**1.15008／0.30488、0.77 及其相關方法均來自含非因果平滑／離線 graph 的舊實驗，僅保留歷史證據，不是現行規則下的合規最佳或提交推薦。不得因這些分數啟動新的 full。

2026-10-21 完整結果與實作盤點（歷史）：[47](doc/47_IMPLEMENTATION_STATUS_AND_TRACK2_SCORE_20261021.md)。Track2 `track02_full_2610212125` 已完成 3299 幀，慢變 map graph 實際求解並套用；使用者回報顯示分數 **0.77**，較當時最佳約 **0.30** 差。新分數只記顯示精度，不反推隱藏小數。GNSS／IMU 的 Kalman＋RTS 前處理與新 GNSS＋map batch graph 為不同模組。

2026-10-21 GPU PCA OOM 已修正：[46｜記憶體界限](doc/46_CUDA_PCA_MEMORY_BOUND_20261021.md)。原 full 在 idx 653 的大型 `eigh` 工作區配置失敗，現每批 256 個矩陣、維持全部點與 float64／原參數。18 項回歸、跨失敗點 24 幀與同 full 輸入數值配對通過；修正後完整 Track 仍由使用者手動執行，原 run 保留。

2026-10-20 GPU 優先加速：[44｜CUDA 尺度配準](doc/44_CUDA_SCALE_ACCELERATION_20261020.md)。新入口 `scripts/run_map_smoother_gpu.py` 將最近鄰、PCA／法向、分層配對、patch 均衡與 voxel 搬 GPU，維持兩 Track 同模型／參數與固定逐幀時間格式。當時 51 項機制／回歸、96 幀 GPU 端到端通過；八個分散影格配準實測約 2.87 倍，非 full 效能或精度保證。手動命令在 [16](doc/16_run_geometry_directional.txt) 開頭。讀檔、小型 solver／graph 與少量精度邊界裁決仍明列為 CPU；尚無本輪新分數。

2026-10-20 新實驗：[42｜逐幀尺度、分層 6-DOF 與慢變 map 修正](doc/42_LAYERED_SCALE_SLOW_MAP_20261020.md)。Track1／Track2 共用同一算法與參數；本輪用 Track2 驗證及提供手動 full 命令，因 Track1 上傳額度已用完。新入口為 `scripts/run_map_smoother.py`，由原始 bag／map 與老師提供的前相機 LoGeR `raw_ply` 開始，未重新執行 LoGeR。44 項測試與首／中／尾共 160 幀完整性通過，其中兩段實際求解 graph；這是明確的 SciPy／KD-tree CPU 研究分支，尚無新 Kaggle 分數，原 CUDA 流程與已評分最佳保持。

目前整體進度、感測器座標契約與下一步算法路線見 [43｜開發進度與座標問題](doc/43_PROGRESS_FRAME_AND_ROADMAP_20261020.md)。

2026-10-20 航向／尺度評分更新：[40｜尺度退步與 pose_yaw 完整評分](doc/40_SCORED_SCALE_FAILURE_AND_POSE_YAW_20261020.md)。`turn_heading_rts`＋尺度7.40426得1.15167；`pose_yaw`＋尺度3得1.30507，兩者都差於最佳1.15008，pose_yaw 已停止推薦且不重跑。先看[41｜外部方法分析](doc/41_EXTERNAL_REFERENCE_ANALYSIS_20261020.md)；[16｜命令紀錄](doc/16_run_geometry_directional.txt)開頭為本輪新分支命令，後面保留已失敗命令作歷史。最新外部方法 score 0.65479 的 CSV 已保存，但原始算法／ICP／尺度與真值證據不足，不能稱為本專案已重現或新最佳。

2026-10-20 來源尺度研究：[38｜來源剛體一致性與尺度](doc/38_SOURCE_RIGIDITY_AND_SCALE_20261020.md)。新截圖 Track1 第 6 名／1.15008；已完成 96 對來源跨幀檢查與 16 場景尺度／map 對照。共用尺度候選改善部分時間殘差，但跨場景不一致、可靠 map factors 仍為 0，不替換最佳或推薦完整重跑。12 項新機制＋9 項 surfel 回歸通過。

所有完整與短段測試統一放 `outputs/run/`，例如 `track01_full_2610192232`、`track02_full_2610192240`、`track01_test_YYMMDDHHmm`。算法／版本／分數記在各次目錄內 `test_info.json`／`params.json`，不另分最近／最佳目錄，不新增測試索引。


- **第一次讀： [01｜程式架構圖與完整演算法流程](doc/01_PROGRAM_FLOW.md)**。從主機／container、Python 入口、C++ solver 一路讀到 CSV，也包含三相機分支。
- 日常指令：[08｜操作手冊](doc/08_MANUAL_WORKFLOW.md)。歷史已評分流程曾明確加 `--profile sensor_guided --sensor-fusion motion_rts --geometry-mode directional --height-mode carry`；它不合現行因果規則。省略選項的程式預設仍是 legacy／fixed／off。
- 結果回放：[30｜RViz 回放與座標朝向](doc/30_RVIZ_REPLAY_20261019.md)。Ubuntu 桌面執行 `python3 scripts/replay.py --track 1`，讀既有 pose／點雲，不重算定位。
- 高度浮動：[32｜Z 回授與防漂移候選](doc/32_HEIGHT_DRIFT_ANALYSIS_20261019.md)。新增實驗 `--height-mode anchored`；短測未一致改善貼合，預設 carry／推薦最佳保留。
- 幾何高度／傾斜候選：[33｜平面高度與有界姿態](doc/33_SURFACE_HEIGHT_REFERENCE_20261019.md)。`surface`／`surface_tilt` 使用獨立細 map、品質檢查與同一靜態表面假說；已提供 `--full` 通用完整命令，自行從原始資料準備並另存新 output，無須指定歷史 run。仍須區分幾何一致性與正式精度。
- 位置先驗研究：[34｜位置先驗與水平觀測](doc/34_POSITION_PRIOR_AND_HORIZONTAL_UPDATE_20261019.md)。原前相機／姿態／carry／directional 的 `motion_rts` 消融、GNSS 留段與 864 筆短段 pose 已完成；後續使用者完整回報 1.15008／0.30488，已升為最佳，見 [35｜新成績與航向實驗](doc/35_MOTION_RTS_AND_GNSS_HEADING_20261019.md)。視覺／立面觀測未通過融合檢查，未接入新的 joint solver。研究入口：`outputs/run/track00_test_2610191627/README.md`。
- 配準／融合升級：[36｜瓶頸與已執行的 GICP／跨幀融合實驗](doc/36_REGISTRATION_AND_FUSION_UPGRADE_20261019.md)。GICP 864 筆短段與聯合 GNSS／gyro 原型 24 個留段已測，尚未一致改善；原最佳與預設保留，研究入口 `outputs/run/track00_test_2610192006/`。
- 專案現況與原 README 完整內容：[27｜專案總覽](doc/27_PROJECT_OVERVIEW.md)。所有歷史文件、舊名對照與官方／CPU 文件索引：[28｜文件目錄](doc/28_DOCUMENTATION_INDEX.md)。

水平 map／joint 分支見 [37｜細 map 水平觀測與跨幀聯合估計](doc/37_HORIZONTAL_SURFEL_JOINT_20261019.md)。`--horizontal-refinement surfel_joint` 已實作，但短段及使用者兩條完整測試都沒有可靠 map factor，聯合求解跳過，CSV 與原 motion_rts 最佳逐 byte 相同。2026-10-20 使用者回報 Track1 仍 1.15008；這輪未提升精度，不建議重跑／重複提交相同候選。benchmark 現在明確提示未產生新提交內容；詳細拒絕原因與未確認外參見 37。

**歷史自行重現最低分（非現行合規候選）：Track1 1.15008、Track2 0.30488，motion_rts＋directional＋carry。** 已評分來源在 `outputs/run/track01_full_2610191750`／`track02_full_2610191757`，分數記於各自test_info.json；[35](doc/35_MOTION_RTS_AND_GNSS_HEADING_20261019.md) 的 pose_heading_rts／independent_heading_rts 消融分支尚無回報分數，不含已評分退步的 pose_yaw（見40）。外部參考0.65479與欄位對照見41，尚未重現其定位算法。

三相機完整成績：Track1 **1.15346**、Track2 **0.31064**，均未超過原前相機最佳 1.15275／0.30576。兩條輸出與 24 組消融已核對，原因與改進順序見 [29｜三相機完整成績分析](doc/29_THREE_CAMERA_SCORED_ANALYSIS_20261018.md)。

2026-10-19 高度／傾斜候選兩條完整已由使用者完成，Track1 回報 **1.15519**，較原最佳增加 **0.2117%**；Track2 本輪尚無回報分數。Track1 Z 範圍雖由 4.86 降至 0.92 m，提交 XY／yaw 並未因此更準。完整配對與下一步先驗／跨幀觀測方向見 [33 最新評分更新](doc/33_SURFACE_HEIGHT_REFERENCE_20261019.md#完整評分失敗與下一步使用者後續回報)，此高度候選未升為最佳；後續位置版本最佳見 35。

後續開發使用本 GPU workspace；原始資料唯讀共用 `../midterm01/data`。完整 Track 由使用者手動執行，協作約定見 [AGENTS.md](AGENTS.md)。

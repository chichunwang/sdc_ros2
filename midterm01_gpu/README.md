# Midterm01 GPU 工作區

文件已集中到 **[doc/](doc/28_DOCUMENTATION_INDEX.md)**，以兩位數編號排列。

2026-10-20 本輪定位更新：[39｜向Track1 0.6推進：航向與原前相機stereo尺度](doc/39_HEADING_AND_STEREO_SCALE_20261020.md)。新增 `turn_heading_rts` 與 `pose_yaw`，並把原前相機 stereo 推導的固定尺度 `7.404260939740923` 接入定位短測；新 full 命令已放在 [16｜常用指令](doc/16_run_geometry_directional.txt) 最上方，使用者可直接執行評分。0.6為目標，尚無本輪分數；已評分最佳保持1.15008／0.30488。

2026-10-20 最新研究：[38｜來源剛體一致性與尺度](doc/38_SOURCE_RIGIDITY_AND_SCALE_20261020.md)。新截圖 Track1 第 6 名／1.15008；已完成 96 對來源跨幀檢查與 16 場景尺度／map 對照。共用尺度候選改善部分時間殘差，但跨場景不一致、可靠 map factors 仍為 0，不替換最佳或推薦完整重跑。12 項新機制＋9 項 surfel 回歸通過。

所有完整與短段測試統一放 `outputs/run/`，例如 `track01_full_2610192232`、`track02_full_2610192240`、`track01_test_YYMMDDHHmm`。算法／版本／分數記在各次目錄內 `test_info.json`／`params.json`，不另分最近／最佳目錄，不新增測試索引。


- **第一次讀： [01｜程式架構圖與完整演算法流程](doc/01_PROGRAM_FLOW.md)**。從主機／container、Python 入口、C++ solver 一路讀到 CSV，也包含三相機分支。
- 日常指令：[08｜操作手冊](doc/08_MANUAL_WORKFLOW.md)。目前已評分流程明確加 `--profile sensor_guided --sensor-fusion motion_rts --geometry-mode directional --height-mode carry`；省略選項的程式預設仍是 legacy／fixed／off。
- 結果回放：[30｜RViz 回放與座標朝向](doc/30_RVIZ_REPLAY_20261019.md)。Ubuntu 桌面執行 `python3 scripts/replay.py --track 1`，讀既有 pose／點雲，不重算定位。
- 高度浮動：[32｜Z 回授與防漂移候選](doc/32_HEIGHT_DRIFT_ANALYSIS_20261019.md)。新增實驗 `--height-mode anchored`；短測未一致改善貼合，預設 carry／推薦最佳保留。
- 幾何高度／傾斜候選：[33｜平面高度與有界姿態](doc/33_SURFACE_HEIGHT_REFERENCE_20261019.md)。`surface`／`surface_tilt` 使用獨立細 map、品質檢查與同一靜態表面假說；已提供 `--full` 通用完整命令，自行從原始資料準備並另存新 output，無須指定歷史 run。仍須區分幾何一致性與正式精度。
- 位置先驗研究：[34｜位置先驗與水平觀測](doc/34_POSITION_PRIOR_AND_HORIZONTAL_UPDATE_20261019.md)。原前相機／姿態／carry／directional 的 `motion_rts` 消融、GNSS 留段與 864 筆短段 pose 已完成；後續使用者完整回報 1.15008／0.30488，已升為最佳，見 [35｜新成績與航向實驗](doc/35_MOTION_RTS_AND_GNSS_HEADING_20261019.md)。視覺／立面觀測未通過融合檢查，未接入新的 joint solver。研究入口：`outputs/run/track00_test_2610191627/README.md`。
- 配準／融合升級：[36｜瓶頸與已執行的 GICP／跨幀融合實驗](doc/36_REGISTRATION_AND_FUSION_UPGRADE_20261019.md)。GICP 864 筆短段與聯合 GNSS／gyro 原型 24 個留段已測，尚未一致改善；原最佳與預設保留，研究入口 `outputs/run/track00_test_2610192006/`。
- 專案現況與原 README 完整內容：[27｜專案總覽](doc/27_PROJECT_OVERVIEW.md)。所有歷史文件、舊名對照與官方／CPU 文件索引：[28｜文件目錄](doc/28_DOCUMENTATION_INDEX.md)。

水平 map／joint 分支見 [37｜細 map 水平觀測與跨幀聯合估計](doc/37_HORIZONTAL_SURFEL_JOINT_20261019.md)。`--horizontal-refinement surfel_joint` 已實作，但短段及使用者兩條完整測試都沒有可靠 map factor，聯合求解跳過，CSV 與原 motion_rts 最佳逐 byte 相同。2026-10-20 使用者回報 Track1 仍 1.15008；這輪未提升精度，不建議重跑／重複提交相同候選。benchmark 現在明確提示未產生新提交內容；詳細拒絕原因與未確認外參見 37。

**目前最佳：Track1 1.15008、Track2 0.30488，motion_rts＋directional＋carry。** 已評分來源在 `outputs/run/track01_full_2610191750`／`track02_full_2610191757`，分數記於各自test_info.json；GNSS pose orientation 的替代／獨立融合實驗見 [35](doc/35_MOTION_RTS_AND_GNSS_HEADING_20261019.md)，尚無新分數。

三相機完整成績：Track1 **1.15346**、Track2 **0.31064**，均未超過原前相機最佳 1.15275／0.30576。兩條輸出與 24 組消融已核對，原因與改進順序見 [29｜三相機完整成績分析](doc/29_THREE_CAMERA_SCORED_ANALYSIS_20261018.md)。

2026-10-19 高度／傾斜候選兩條完整已由使用者完成，Track1 回報 **1.15519**，較原最佳增加 **0.2117%**；Track2 本輪尚無回報分數。Track1 Z 範圍雖由 4.86 降至 0.92 m，提交 XY／yaw 並未因此更準。完整配對與下一步先驗／跨幀觀測方向見 [33 最新評分更新](doc/33_SURFACE_HEIGHT_REFERENCE_20261019.md#完整評分失敗與下一步使用者後續回報)，此高度候選未升為最佳；後續位置版本最佳見 35。

後續開發使用本 GPU workspace；原始資料唯讀共用 `../midterm01/data`。完整 Track 由使用者手動執行，協作約定見 [AGENTS.md](AGENTS.md)。

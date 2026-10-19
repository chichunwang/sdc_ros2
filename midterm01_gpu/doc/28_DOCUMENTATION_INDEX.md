# 28｜文件目錄、開發時間序與舊名對照

整理日期：2026-10-18；回放索引更新：2026-10-19（Asia/Taipei）。Midterm01 GPU 的說明正文集中在 `/home/ricky/sdc_ros2/midterm01_gpu/doc/`；文件名使用兩位數前綴。**先讀 [01｜程式架構與完整流程](01_PROGRAM_FLOW.md)**，再依問題選讀下面的文件。

## 編號與保存規則

- `01` 是使用者指定的總架構入口，優先於歷史時間序。
- `02–19` 對應 2026-10-17 的環境、NDT、sensor_guided、濾波／融合、方向性配準及後續開發原則；`20–26` 對應 2026-10-18 的分析、三相機與效能實驗。
- 日期以文件與既有實驗紀錄為準。同日沒有可核對的精確建立時刻時，依已知開發先後與依賴排列，操作／命令／後端說明附在相關階段；不把檔案 mtime 當成事件時間。
- `08` 操作手冊、`09` 輸出指南、`19` 開發原則與 `27` 專案總覽會持續更新，編號不是它們最後更新時間。`27` 保存整理當時原 GPU README 的完整內容；`28` 是本次新增目錄。
- `29` 是整理期間同步新增的三相機完整評分／診斷文件，接續納入此目錄；它沒有改變 `01` 作為架構入口的安排。
- `30` 為 2026-10-19 新增的既有結果 RViz 回放、座標朝向與 GUI 操作。
- 歷史文件的「目前／尚未」保留當時語境。判斷現在實際流程先讀 `01`；判斷舊結果看該輪日期、run、參數與 SHA256。
- `00_Personal_Record.txt` 是你的個人紀錄，保留原名、原內容，不加入歷史報告排序。原 `command_paste.txt` 在本次整理之前已被刪除；此次保留刪除狀態，沒有用 Git 舊內容復原。
- 根目錄、CUDA package、outputs 的 `README.md` 保留短入口；`AGENTS.md` 保留工具辨識所需的位置與名稱。它們連到這個集中目錄。`CMakeLists.txt` 是建置程式，不是應改名的 documentation。
- CPU／官方原始文件仍保留在原位置，下面提供中央索引。HW2、HW3 與通用 CLI 文件有各自用途，不併入 Midterm01 GPU 的開發歷史。

## 全部編號文件

表中的「原位置」相對 `midterm01_gpu/`，供你對照 IDE 舊分頁與舊名稱；使用「文件」欄的新連結開啟。

| 編號／日期 | 文件與用途 | 原位置 |
| --- | --- | --- |
| 01／2026-10-18 新增 | [程式架構圖與完整演算法流程](01_PROGRAM_FLOW.md)：從 container、Python、C++、CUDA 到 CSV；含三相機分支 | 本次新增 |
| 02／2026-10-17 | [GPU 首建紀錄](02_SETUP_20261017.md)：環境隔離、CDI／CUDA 核對與初期限制 | `SETUP_20261017.md` |
| 03／2026-10-17 | [首次 CUDA NDT 整合](03_CUDA_NDT_20261017.md)：D2D 後端、Release、cache 與速度實測 | `CUDA_NDT_20261017.md` |
| 04／2026-10-17，持續更新 | [CUDA 後端說明](04_CUDA_BACKEND_REFERENCE.md)：上游版本／授權、最近鄰與歷史 NDT 分工 | `workspace/src/cuda_ndt_backend/README.md` |
| 05／2026-10-17 | [CPU／CUDA 前 50 幀比較](05_ACCURACY_COMPARISON_20261017.md)：舊後端的幾何與軌跡診斷 | `ACCURACY_COMPARISON_20261017.md` |
| 06／2026-10-17 | [sensor_guided 初版與實測](06_SENSOR_GUIDED_20261017.md)：由舊失效轉向先驗引導、完整 run 與早期成績 | `SENSOR_GUIDED_20261017.md` |
| 07／2026-10-17 | [老師模板到 GPU 的演進教學](07_ALGORITHM_WALKTHROUGH_20261017.md)：原有／新增功能、初版公式與失效機制 | `doc/ALGORITHM_WALKTHROUGH_20261017.md` |
| 08／2026-10-17 起，持續更新 | [GPU 操作手冊](08_MANUAL_WORKFLOW.md)：建置、幀數、來源、結果、權限與三相機命令 | `doc/MANUAL_WORKFLOW.md` |
| 09／2026-10-17 起，持續更新 | [輸出指南](09_OUTPUTS_GUIDE.md)：runs／submissions／figures、保存與命名 | `outputs/README.md` |
| 10／2026-10-17 | [fixed＋geometry off 歷史命令](10_run_sensor_guided.txt)：初版重現，非最新推薦組合 | `doc/run_sensor_guided.txt` |
| 11／2026-10-17 | [濾波分析](11_FILTER_ANALYSIS_20261017.md)：IIR 滯後、後處理消融、時變 heading 假說 | `doc/FILTER_ANALYSIS_20261017.md` |
| 12／2026-10-17 | [GNSS／IMU 感測器融合](12_SENSOR_FUSION_20261017.md)：heading bias Kalman／RTS、motion_rts 與相關性限制 | `doc/SENSOR_FUSION_20261017.md` |
| 13／2026-10-17 | [heading_rts＋geometry off 歷史命令](13_run_sensor_fusion.txt)：融合階段重現 | `doc/run_sensor_fusion.txt` |
| 14／2026-10-17 | [heading_rts 完整進步紀錄](14_HEADING_RTS_RESULTS_20261017.md)：1.16951／0.31967、來源與 SHA256 | `doc/HEADING_RTS_RESULTS_20261017.md` |
| 15／2026-10-17 | [方向性配準設計](15_GEOMETRY_DIRECTIONAL_20261017.md)：PCA、法向／切向 metric、幾何診斷 | `doc/GEOMETRY_DIRECTIONAL_20261017.md` |
| 16／2026-10-17 | [heading_rts＋directional 手動命令](16_run_geometry_directional.txt)：前相機命令，開頭已更新 motion_rts；保留 heading 歷史 | `doc/run_geometry_directional.txt` |
| 17／2026-10-17 | [方向性配準完整成績](17_GEOMETRY_RESULTS_20261017.md)：1.15275／0.30576、最佳 CSV 與證據 | `doc/GEOMETRY_RESULTS_20261017.md` |
| 18／2026-10-17 | [幾何 fallback 調查](18_GEOMETRY_FALLBACK_20261017.md)：第 410 幀、候選拒絕原因與 log 修正 | `doc/GEOMETRY_FALLBACK_20261017.md` |
| 19／2026-10-17 | [學習與開發原則](19_DEVELOPMENT_PRINCIPLES.md)：觀察、假說、模型推導、驗證與泛化 | `doc/DEVELOPMENT_PRINCIPLES.md` |
| 20／2026-10-18 | [現行算法與精度方向](20_CURRENT_ALGORITHM_AND_ACCURACY_20261018.md)：Team 3 排名快照與改善假說 | `doc/CURRENT_ALGORITHM_AND_ACCURACY_20261018.md` |
| 21／2026-10-18 | [三相機初步評估](21_THREE_CAMERA_ASSESSMENT_20261018.md)：merged PLY、FLU 與初始只讀 audit | `doc/THREE_CAMERA_ASSESSMENT_20261018.md` |
| 22／2026-10-18 | [三相機尺度／同步實作](22_THREE_CAMERA_IMPLEMENTATION_20261018.md)：混合時間來源、尺度候選與 12 組短段 | `doc/THREE_CAMERA_IMPLEMENTATION_20261018.md` |
| 23／2026-10-18 | [raw 三相機校正重建](23_CALIBRATED_RIG_RESULTS_20261018.md)：固定 rig、per-view depth、運動補償與公平對照 | `doc/CALIBRATED_RIG_RESULTS_20261018.md` |
| 24／2026-10-18 | [跨場景偏差與完整入口](24_BIAS_VALIDATION_20261018.md)：fit／holdout、凍結尺度、原子保存與續跑 | `doc/BIAS_VALIDATION_20261018.md` |
| 25／2026-10-18 | [三相機瓶頸與加速](25_RIG_PERFORMANCE_20261018.md)：fast 入口、取樣加速、一致性與匯入舊幀 | `doc/RIG_PERFORMANCE_20261018.md` |
| 26／2026-10-18 | [顯存換速度實測](26_GPU_MEMORY_RESULTS_20261018.md)：batch／cache／PCA chunk，已測範圍與未採用設定 | `doc/GPU_MEMORY_RESULTS_20261018.md` |
| 27／2026-10-18 整理，持續更新 | [專案總覽](27_PROJECT_OVERVIEW.md)：原 GPU README 完整內容與狀態紀錄 | `README.md` |
| 28／2026-10-18 新增 | [本文件](28_DOCUMENTATION_INDEX.md)：中央目錄、閱讀路線、舊名與參考索引 | 本次新增 |
| 29／2026-10-18 | [三相機完整成績與原因診斷](29_THREE_CAMERA_SCORED_ANALYSIS_20261018.md)：後續評分、完整配對、camera／Z probe 與消融 | 整理期間同步新增 |
| 30／2026-10-19 | [RViz 結果回放與座標朝向](30_RVIZ_REPLAY_20261019.md)：既有 pose／PLY 重演、GUI、XYZ／RPY／TF 與逐幀操作 | 本次新增 |
| 31／2026-10-19 | [回放播放與高度診斷](31_REPLAY_PLAYBACK_AND_HEIGHT_20261019.md)：連續播放 TF 丟雲修正、64 幀前後對照、保留原 pose 的高度／形狀分析 | 使用者回報後新增 |
| 32／2026-10-19 | [Z 漂移與高度模式對照](32_HEIGHT_DRIFT_ANALYSIS_20261019.md)：累積回授、anchored 候選、12 組短段與負面結果；未替換最佳 | 使用者要求分析／修改後新增 |
| 33／2026-10-19 | [來源／LiDAR 平面高度參考](33_SURFACE_HEIGHT_REFERENCE_20261019.md)：獨立細 map、surface／surface_tilt；含後續完整 Track1 1.15519 退步、兩條配對與下一步優化順序 | 接續 32 的幾何驗證 |
| 34／2026-10-19 | [位置先驗與水平觀測更新](34_POSITION_PRIOR_AND_HORIZONTAL_UPDATE_20261019.md)：只換 XY 的 motion_rts 消融、GNSS 留段、864 筆短段、影像／立面與外參拒絕證據；joint 未接入 | 使用者授權下一波開發 |

| 35／2026-10-19 | [motion_rts 新最佳與 GNSS 航向消融](35_MOTION_RTS_AND_GNSS_HEADING_20261019.md)：1.15008／0.30488、輸入 quaternion、假設獨立／替代因子及留段限制 | 後續分數與航向實驗 |

| 36／2026-10-19 | [配準與融合升級](36_REGISTRATION_AND_FUSION_UPGRADE_20261019.md)：最佳完整輸出的方向／界線診斷、有界 GICP 864 筆短段、最小 GNSS／gyro 聯合原型與 24 留段，未升為最佳 | 使用者要求分析後即測試 |
| 37／2026-10-19 | [細 map 水平觀測與跨幀聯合估計](37_HORIZONTAL_SURFEL_JOINT_20261019.md)：0.3 m surfel、frame／lag audit、可靠 map factors 接入 planar graph、拒絕保留與通用完整命令；尚無真實精度提升 | 使用者指定下一階段實作與驗證 |
| 38／2026-10-20 | [來源剛體一致性與尺度](38_SOURCE_RIGIDITY_AND_SCALE_20261020.md)：最新 Track1 第 6 名快照、96 對 RGB／3D 跨幀檢查、共用尺度與 16 場景 map 消融；候選未通過，不替換最佳 | 使用者要求持續優化 Track1 |
| 39／2026-10-20 | [向Track1 0.6推進：航向與原前相機stereo尺度](39_HEADING_AND_STEREO_SCALE_20261020.md)：turn_heading_rts／pose_yaw接入通用定位、原PLY的stereo固定尺度7.40426、分散短段與full指令；尚無本輪正式分數 | 使用者指定0.6目標，完整評分仍由使用者執行 |

## 開發脈絡

```text
2026-10-17
  GPU 環境／CDI／CUDA
    → 首次 CUDA D2D NDT、CPU 對照
    → 發現漂移／姿態失效，改 sensor_guided + robust ICP
    → 分析既有平滑與 heading 的時變差異
    → heading bias KF／RTS（heading_rts）與完整評分
    → directional map 殘差與完整評分
    → fallback 診斷、記錄學習與算法改進原則
2026-10-18
  新排名與精度分析
    → 現成 merged 三相機 audit、尺度／同步短段
    → raw 三相機校正重建、per-view motion、均衡取樣
    → 分散場景偏差驗證、完整／續跑入口
    → 相同算法取樣加速、匯入既有幀
    → 顯存／batch／chunk 對照
    → 本次架構教學與文件集中整理
```

序列輸出完整、效能更快、內部幾何更好、Kaggle 分數更低，是不同種類的證據；查看各階段時保留這個區別。三相機本機已有完整輸出，見 `01` 的實際 run 追蹤，後續評分與原因診斷見 [29](29_THREE_CAMERA_SCORED_ANALYSIS_20261018.md)；較早報告所記的短測／執行中狀態仍保留原貌，沒有把前相機成績回填給三相機。

## 按你的問題選讀

| 想回答的問題 | 閱讀路線 |
| --- | --- |
| 從哪支程式開始？每一步讀什麼、輸出什麼？ | 01 → 08 |
| 跑完後怎麼看 RViz、座標與朝向？ | [30｜回放操作與座標契約](30_RVIZ_REPLAY_20261019.md) |
| 模板、舊 GPU 與現在方法差在哪？ | 07 → 03／05 → 06 → 12 → 15 |
| Kalman filter 在哪？為什麼估 heading bias？ | 01 第 5 節 → 11 → 12 |
| cost、先驗權重、normal 與四自由度如何推導？ | 01 第 8 節 → 15 → 07 的初版公式對照 |
| 哪版真的有 Kaggle 成績？ | 14 → 17 → 20 → 29；保留日期與 run／hash |
| 為什麼有 WARN／INFO，還是有完整 pose？ | 01 第 8–9 節 → 18 |
| 三相機怎麼走到我開的 result_2.csv？ | 01 第 10 節 → 23 → 24 → 25 |
| 為什麼顯存增加卻沒變快？ | 25 → 26 |
| 只要可貼上的最新前相機命令？ | 08 → 16；10／13 是較早消融版本 |

## 官方、CPU 與其他必要參考的中央索引

這些檔案保留原位置，避免改動官方原件、CPU 參考環境與共用資料說明；從這裡即可集中找到它們。

| 文件 | 用途 |
| --- | --- |
| [官方題目 PDF](<../../midterm01/docs/SDC 2026 Midterm01 Competition -  Localization.pdf>) | 正式任務、評分、格式、簡報與交付要求；檔名中 `-` 後是兩個空白 |
| [官方要求分析](../../midterm01/docs/README.md) | 題目頁次、交付與算法建議的區別 |
| [Kaggle 資料規格](../../midterm01/docs/KAGGLE_DATASET_SPEC.md) | Track、輸入與 submission 契約的已知／待核對事項 |
| [共用資料 README](../../midterm01/data/README.md) | 唯一 data 目錄、資料結構與來源 |
| [資源下載索引](../../midterm01/docs/DOWNLOADS.md) | 官方資源位置與缺檔確認 |
| [CPU 首跑紀錄](../../midterm01/docs/UBUNTU_FIRST_RUN_20261017.md) | Ubuntu 初次資料／bag／單幀與短測證據 |
| [CPU Track1 完整與權限](../../midterm01/docs/CPU_TRACK1_COMPLETED_20261017.md) | 3011 筆 TXT、退出碼、指定 run 權限修復 |
| [CPU 操作手冊](../../midterm01/docs/MANUAL_WORKFLOW.md) | 原 CPU 環境重現；日常改用本目錄 08 |
| [CPU 根 README](../../midterm01/README.md)、[原 workspace 說明](../../midterm01/workspace/src/README.md) | 原環境與模板來源 |
| [歷史實驗索引](../../midterm01/experiments/README.md)、[報告交付說明](../../midterm01/reports/README.md)、[CPU 輸出說明](../../midterm01/outputs/README.md) | CPU／早期交付與結果保存，成績文字按原時點解讀 |
| [repo 協作約定](../../AGENTS.md)、[GPU 協作約定](../AGENTS.md) | 開工入口、手動完整測試、資料與環境隔離 |

本次只更新上述文件中指向搬移文件的連結，沒有搬走 CPU／官方原件。大型 outputs、模型與既有 CSV 不由 Git 同步；文件裡能點到的本機證據，移機時仍須另外備份。

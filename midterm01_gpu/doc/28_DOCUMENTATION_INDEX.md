# 28｜文件目錄、開發時間序與舊名對照

整理日期：2026-10-18；規則索引更新：2026-10-23（Asia/Taipei）。Midterm01 GPU 的說明正文集中在 `/home/ricky/sdc_ros2/midterm01_gpu/doc/`；一般文件使用兩位數前綴。現行官方規則與歷史原件集中於 `Competition Regulations/`；核心入口及因果規則程式盤點見 [48](48_RULES_AND_CODE_AUDIT_20261023.md)。**先讀 [01｜程式架構與完整流程](01_PROGRAM_FLOW.md)**，再依問題選讀下面的文件。

## 編號與保存規則

**現行規則：**[00 整合新版（31 頁）](<Competition Regulations/00_SDC_2026_Midterm01_Localization_整合新版_20261022.pdf>)、[01 因果定位細則（4 頁）](<Competition Regulations/01_Mid01_Updates_Rules.pdf>)；[90](<Competition Regulations/90_舊版_已取代_可刪除_SDC_2026_Midterm01_Localization_24頁.pdf>)／[91](<Competition Regulations/91_舊版_已取代_可刪除_Mid01_Updates_Adjustments_20261021.pdf>)為保留歷史。因果規則禁止未來資訊回修、RTS／backward smoothing 及定位後離線軌跡修正，所有 Track 適用；程式盤點見 [48](48_RULES_AND_CODE_AUDIT_20261023.md)。

- 2026-10-23 起，每次成功 full run 的事後唯讀圖表放在該 run 的 `analysis/`；舊教學程式、素材與 MATLAB 證據仍保留於 [analysis/README.md](analysis/README.md)，作歷史教學內容，不補畫舊 run。輸出契約與限制見 [08](08_MANUAL_WORKFLOW.md#full-run-事後圖表)；學習原則見 [19](19_DEVELOPMENT_PRINCIPLES.md)。

- `01` 是使用者指定的總架構入口，優先於歷史時間序。
- `000` 與 `001` 是官方原始 PDF 的固定置頂例外：`000` 為原競賽說明，`001` 為 2026-10-21 補充規則；其他一般文件仍用兩位數編號。規則解讀與專案影響以 [45｜競賽補充規則](45_COMPETITION_UPDATES_20261021.md) 為入口。
- `02–19` 對應 2026-10-17 的環境、NDT、sensor_guided、濾波／融合、方向性配準及後續開發原則；`20–26` 對應 2026-10-18 的分析、三相機與效能實驗。
- 日期以文件與既有實驗紀錄為準。同日沒有可核對的精確建立時刻時，依已知開發先後與依賴排列，操作／命令／後端說明附在相關階段；不把檔案 mtime 當成事件時間。
- `08` 操作手冊、`09` 輸出指南、`19` 開發原則與 `27` 專案總覽會持續更新，編號不是它們最後更新時間。`27` 保存整理當時原 GPU README 的完整內容；`28` 是本次新增目錄。
- `29` 是整理期間同步新增的三相機完整評分／診斷文件，接續納入此目錄；它沒有改變 `01` 作為架構入口的安排。
- `30` 為 2026-10-19 新增的既有結果 RViz 回放、座標朝向與 GUI 操作。
- 歷史文件的「目前／尚未」保留當時語境。判斷現在實際流程先讀 `01`；判斷舊結果看該輪日期、run、參數與 SHA256。
- `00_Personal_Record.txt` 是你的個人紀錄，保留原名、原內容，不加入歷史報告排序。原 `command_paste.txt` 在本次整理之前已被刪除；此次保留刪除狀態，沒有用 Git 舊內容復原。
- 根目錄、CUDA package、outputs 的 `README.md` 保留短入口；`AGENTS.md` 保留工具辨識所需的位置與名稱。它們連到這個集中目錄。`CMakeLists.txt` 是建置程式，不是應改名的 documentation。
- CPU 參考文件保留在原位置；官方原始 PDF 集中於本目錄的 `000`／`001`，下面提供中央索引。HW2、HW3 與通用 CLI 文件有各自用途，不併入 Midterm01 GPU 的開發歷史。

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
| 39／2026-10-20 | [向Track1 0.6推進：航向與原前相機stereo尺度](39_HEADING_AND_STEREO_SCALE_20261020.md)：turn_heading_rts／pose_yaw接入通用定位、原PLY的stereo固定尺度7.40426、分散短段與full指令；後續完整評分 Track1 1.15167，較最佳1.15008退步約0.138%，未升為最佳 | 已評分歷史結果，完整數據見40 |
| 40／2026-10-20 | [尺度與航向完整評分失敗](40_SCORED_SCALE_FAILURE_AND_POSE_YAW_20261020.md)：scale7.40426得1.15167、pose_yaw＋scale3得1.30507；兩者未勝最佳，pose_yaw停止推薦 | 使用者回報完整評分後更新 |
| 41／2026-10-20 | [外部方法分析](41_EXTERNAL_REFERENCE_ANALYSIS_20261020.md)：外部 score 0.65479 CSV、方法口述與目前缺少的重現證據 | 外部參考 CSV 與方法說明 |
| 42／2026-10-20 | [分層尺度配準與慢變 map 修正](42_LAYERED_SCALE_SLOW_MAP_20261020.md)：兩 Track 共用逐幀尺度／6-DOF／12 s anchor／GNSS 慢 bias 與 gyro／NHC graph；本輪 Track2 短測與手動 full 命令 | 依學長文字方法建立的新研究分支，尚無新分數 |
| 43／2026-10-20 | [開發進度、座標契約與研究路線](43_PROGRESS_FRAME_AND_ROADMAP_20261020.md)：目前已評分最佳、研究分支狀態、GNSS／IMU／LiDAR／相機點雲的座標關係，以及通往可驗證最佳算法的步驟 | 本專案進度與座標問題入口 |
| 44／2026-10-20 | [CUDA 尺度配準加速](44_CUDA_SCALE_ACCELERATION_20261020.md)：CUDA／CPU 工作邊界、配對數值與 registration 計時、接受率限制及端到端待辦 | GPU 加速分支；完整輸入路徑尚待驗證 |
| 45／2026-10-21 | [競賽補充規則與專案影響](45_COMPETITION_UPDATES_20261021.md)：Track3、排名計分、隱藏真值使用限制、期限及資料格式待確認項 | 依補充公告整理；按原件與公告分清已知／未知 |
| 46／2026-10-21 | [CUDA PCA 記憶體界限與分批修正](46_CUDA_PCA_MEMORY_BOUND_20261021.md)：full idx 653 OOM、256 矩陣批次、工作區峰值與同輸入回歸；固定輸出不變 | 原失敗 run 保留，修正後 full 尚待使用者執行 |
| 47／2026-10-21 | [目前實作盤點與 Track2 最新回報](47_IMPLEMENTATION_STATUS_AND_TRACK2_SCORE_20261021.md)：說明原最佳與新 full 的方法差異、Kalman／RTS 與 ICP／batch graph 邊界、已做實驗及未完成契約 | Track2 新 full 完整性通過；使用者回報顯示 0.77，未勝 0.30488 歷史最佳 |
| 48／2026-10-23 | [規則與程式稽核](48_RULES_AND_CODE_AUDIT_20261023.md)：現行因果條款、Track3 公告已知／未知項目及本機入口盤點 | 依 10/22 現行 PDF 與本機靜態審查 |
| 49／2026-10-23 | [Track3 因果 baseline 與手動短段入口](49_TRACK3_CAUSAL_BASELINE_20261023.md)：分清靜態因果路徑、尚待使用者執行的 50 幀短段與未確認的正式提交契約 | 資料／CLI／乾跑及靜態審查；沒有執行定位 |
| 50／2026-10-23，持續更新 | [指令表](50_COMMAND_REFERENCE.md)：Track3 因果 v2 full 手動命令、可選短段、dry-run／audit-only、歷史 legacy 命令、輸出檢查與事後圖表重試 | 手動命令速查；真實 v2 效果未驗證 |
| 51／2026-10-23 | [Track3 異常跳動診斷](51_TRACK3_FAILURE_DIAGNOSIS_20261023.md)：legacy run 的收斂跳點、IMU 證據界限與因果驗證建議 | 唯讀診斷；未修改 solver 或 CSV |
| 52／2026-10-23 | [Track3 因果 Kalman 與逐幀尺度配準](52_TRACK3_CAUSAL_KALMAN_SCALE_20261023.md)：11-state 前向模型、尺度／幾何檢查、v1 full 失敗及現行 v2 full 命令 | v2 通過合成／機制測試，尚無真實定位或新分數 |
| 53／2026-10-23 | [Track3 初始化修正與驗證](53_TRACK3_INITIALIZATION_REPAIR_20261023.md)：v1 啟動失敗幾何線索、v2 初始化與拒絕條件、59 項檢查及其合成證據界限 | 合成／機制檢查通過；真實 v2 定位未驗證 |

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
| 如何查看新 full run 的 Python 診斷圖？ | 成功 run 目錄下的 `analysis/` → [輸出契約](08_MANUAL_WORKFLOW.md#full-run-事後圖表) |
| 如何讀取舊 Python／Matplotlib 教學素材？ | [歷史教學入口](analysis/README.md) → 01 → 12 |
| 目前開發到哪裡、GNSS／IMU／LiDAR／相機點雲各在哪個座標，哪個座標準？ | [43｜進度與座標契約](43_PROGRESS_FRAME_AND_ROADMAP_20261020.md) |
| 跑完後怎麼看 RViz、座標與朝向？ | [30｜回放操作與座標契約](30_RVIZ_REPLAY_20261019.md) |
| 模板、舊 GPU 與現在方法差在哪？ | 07 → 03／05 → 06 → 12 → 15 |
| Kalman filter 在哪？為什麼估 heading bias？ | 01 第 5 節 → 11 → 12 |
| cost、先驗權重、normal 與四自由度如何推導？ | 01 第 8 節 → 15 → 07 的初版公式對照 |
| 哪版真的有 Kaggle 成績？ | 14 → 17 → 20 → 29；保留日期與 run／hash |
| 為什麼有 WARN／INFO，還是有完整 pose？ | 01 第 8–9 節 → 18 |
| 三相機怎麼走到我開的 result_2.csv？ | 01 第 10 節 → 23 → 24 → 25 |
| 為什麼顯存增加卻沒變快？ | 25 → 26 |
| 最新外部方法與本地版本差異？ | 41；full 命令記錄見16，已失敗的pose_yaw命令僅供歷史參照 |
| Track3 因果 Kalman／尺度方法採用哪些狀態、假設與驗證順序？ | [52｜因果 Kalman 與逐幀尺度配準](52_TRACK3_CAUSAL_KALMAN_SCALE_20261023.md)；v2 初始化修正與合成檢查見 [53](53_TRACK3_INITIALIZATION_REPAIR_20261023.md) |
| Track3 如何檢視因果 baseline、手動跑新分支短段並判讀限制？ | [49｜因果 baseline](49_TRACK3_CAUSAL_BASELINE_20261023.md) → [52｜新方法與驗證順序](52_TRACK3_CAUSAL_KALMAN_SCALE_20261023.md)；現行規則與程式盤點見48 |
| 常用指令、Track3 因果短段／歷史 legacy 命令、輸出檢查與 full 圖表重試？ | [50｜指令表](50_COMMAND_REFERENCE.md)；新方法限制見52 |

## 官方、CPU 與其他必要參考的中央索引

CPU 參考文件與共用資料說明保留原位置；原規則 PDF 已移入本 GPU 文件目錄，補充 PDF 也放在此處。從這裡可集中找到它們。

| 文件 | 用途 |
| --- | --- |
| [00｜整合新版規則（31 頁）](<Competition Regulations/00_SDC_2026_Midterm01_Localization_整合新版_20261022.pdf>) | 現行整合規則，包含 10/22 因果定位條文 |
| [01｜因果定位細則（4 頁）](<Competition Regulations/01_Mid01_Updates_Rules.pdf>) | 所有 Track 適用；禁止未來資訊回修、RTS／backward smoothing、定位後離線軌跡修正 |
| [48｜規則與程式稽核](48_RULES_AND_CODE_AUDIT_20261023.md) | 現行限制及本機非因果入口盤點；舊算法與分數不代表合規最佳 |
| [90｜已取代的原競賽規則（24 頁）](<Competition Regulations/90_舊版_已取代_可刪除_SDC_2026_Midterm01_Localization_24頁.pdf>) | 歷史原件保留；不作現行規則入口 |
| [91｜已取代的 10/21 補充公告（10 頁）](<Competition Regulations/91_舊版_已取代_可刪除_Mid01_Updates_Adjustments_20261021.pdf>) | 歷史原件保留；現行解讀以 00／01 為準 |
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

原競賽 PDF 已由 `midterm01/docs/` 搬至本目錄，連結指向同一原件；補充 PDF 亦保存在本目錄。CPU 參考文件仍保留原位置。大型 outputs、模型與既有 CSV 不由 Git 同步；文件裡能點到的本機證據，移機時仍須另外備份。

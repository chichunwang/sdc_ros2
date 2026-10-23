# 48｜10/22 新規則與現有程式合規審查

**同日後續：**Track3 本機資料已完成唯讀盤點，並開放既有 map-only `legacy` 的手動短段入口；50 幀乾跑通過，沒有實際定位或提交。接續證據、因果資料路徑及命令見 [49](49_TRACK3_CAUSAL_BASELINE_20261023.md)。本篇原始審查中「本機 bag／初始矩陣未讀取」「Track3 尚無入口」等狀態以此接續為準；官方 sample／提交 frame／yaw／尺度與實際前綴檢查仍未確認，RTS 與 batch graph 的阻擋判定不變。

核對日期：2026-10-23（Asia/Taipei）。**目前推薦過的 `motion_rts`、CPU／GPU map smoother 與 `surfel_joint` 流程不符合新公布的因果定位要求；`fixed` 也不是合規替代。既有 1.15008／0.30488／0.77 等分數保留為歷史，不再稱為現行合規最佳。**本輪完成文件整理、程式審查與規則盤點工具修正，沒有改造定位算法或建立新的合規提交結果。

## 1. 原件、改名與舊版標記

四份 PDF 均已抽取全文；10/22 四頁細則全部頁面與新版 Track3 關鍵頁另經渲染目視核對。本輪依使用者提供的本機官方文件；公開網路查詢未找到可核實的對應公告，未宣稱已登入 E3 或核對 Kaggle 即時設定。

| 現在檔名（位於 `doc/Competition Regulations/`） | 頁數 | 判定與原檔名 |
| --- | ---: | --- |
| [00_SDC_2026_Midterm01_Localization_整合新版_20261022.pdf](<Competition Regulations/00_SDC_2026_Midterm01_Localization_整合新版_20261022.pdf>) | 31 | 現行整合文件；原 `SDC 2026 Midterm01 Competition -  Localization.pdf`，含 10/22 摘要 |
| [01_Mid01_Updates_Rules.pdf](<Competition Regulations/01_Mid01_Updates_Rules.pdf>) | 4 | 現行中英文細則；原 `mid01 Updates Rules.pdf`，不能因整合文件有摘要就刪除 |
| [90_舊版_已取代_可刪除_SDC_2026_Midterm01_Localization_24頁.pdf](<Competition Regulations/90_舊版_已取代_可刪除_SDC_2026_Midterm01_Localization_24頁.pdf>) | 24 | 原 `00_SDC 2026 Midterm01 Competition -  Localization.pdf`；規範已由整合新版取代 |
| [91_舊版_已取代_可刪除_Mid01_Updates_Adjustments_20261021.pdf](<Competition Regulations/91_舊版_已取代_可刪除_Mid01_Updates_Adjustments_20261021.pdf>) | 10 | 原 `01_mid01 Updates & Adjustments.pdf`；規範已整合，部分 Track3 資訊已更新 |

**沒有完全相同的重複 PDF。**「可刪除」表示就查閱現行規則而言已被新版覆蓋，並非 SHA 相同；本輪只改名，沒有刪除歷史原件、下載連結或來源證據。91 的中文頁是英文條文翻譯，未發現獨立新增規範；新版 public 比例、private baseline 與三 Track 交付要求優先。歷史分析文件仍可能連到 90／91 以保留當時來源，若日後真正刪除，歷史原件連結也會失效；現行規則工具只依賴 00／01。

| 原件 | 改名前後相同的 SHA256 |
| --- | --- |
| 00 新版 | `cdf6467c50562a03e3a34d4ab26edb4db8ace84da9ca22a726728c0ada8ea047` |
| 01 細則 | `cf60236b87f6a89b12f1b182dfd27b20a356023636877677ef0b608c2e8b14fd` |
| 90 舊原規則 | `8f1bcd2d1e3a43f7420d96d601d51561f2c5ad426b662b62534d296dd0a2bace` |
| 91 舊補充 | `4c2fd524a82299f912fae40e5d2fdcf5f7349f4af14641e6f84ac70e79f3feac` |

## 2. 現行規則與舊解讀的差異

| 項目 | 核實內容與來源頁碼 |
| --- | --- |
| 因果定位，適用所有 Track | 01 p.1／3：當前 pose 只能使用當前與過去資訊，不得取得未來 frame 後回頭修正舊 pose；允許 KF／EKF、causal moving average 等方法 |
| 禁止離線軌跡後處理 | 01 p.1／3：禁止定位後整段平移／旋轉、非因果平滑、人工調 pose，以及 pipeline 正常流程以外的後處理。將 backward smoothing 放進 pipeline 仍違反獨立的因果要求 |
| 禁止榜分反推真值 | 01 p.1／3、00 p.27：正常算法開發、模型比較、參數調整仍允許；不能利用多次系統性擾動提交與 public 分數反推 GT 再改輸出 |
| 最終選擇與重現 | 01 p.2／4：10/29 09:00 前選取合規 Kaggle CSV；TA 會重跑提交程式，允許浮點、版本、硬體造成的合理差異；確認違規或無法合理重現時該 Track 0 分 |
| Track1／2 GNSS | 00 p.25：輸入合法，不等於所有 GNSS 融合／離線平滑方法皆合法 |
| Track3 幀數與 GNSS | 00 p.9 明列 **3142** 幀；p.28 明確 **不提供 GNSS pose**。一般資料列表 p.9 對 bag 含 GNSS 的泛稱不能覆蓋 Track3 專屬條款 |
| Track3 public／private | 00 p.28：public 為前 **5–40%**，其餘為 private，private 用於最終評分排名；private score **≤1.40** 才取得基本分。取代 91 的前 10–30% 舊說明，不自行推定精確切分比例 |
| 排名權重 | 00 p.21：各 Track 名次基數分別乘 **0.3／0.3／0.4** 後相加；報告 30、簡報 20，照原規則記錄 |
| Kaggle／簡報／E3 | 00 p.17：每天最多 15 次提交、每幀須有 pose；p.22：4–6 分鐘簡報，10/29 課堂報告，當日 09:00 前寄投影片；p.23：E3 11/05 23:59，交程式、不含原始資料、報告、README 及 `result_1.csv`／`result_2.csv`／`result_3.csv` |
| Track2 地圖時間欄位 | 00 p.26：`tl/th` 為誤留且可靠性未驗證的資訊，文字為審慎使用警示；本專案繼續採定位僅 XYZ、不由時間欄反推軌跡的政策 |

日期時間依 PDF 文字記錄；文件未逐項明示時區或數值重現容忍門檻，不自行新增。後續使用者於 2026-10-23 轉貼 Track3 Kaggle 說明，已確認三個相機 H265 topic、`/imu/data`（orientation／acceleration），以及 CSV 欄序 `idx,x,y,yaw`、3142 筆資料（不含表頭，含表頭共 3143 筆紀錄）；完整原文見 [Kaggle 資料規格](../../midterm01/docs/KAGGLE_DATASET_SPEC.md)。這是官方公告文字的確認，未登入網頁或讀取本機 bag 核驗。官方 sample、idx 起算／完整 ID 集合、yaw 單位／座標系，以及本機 bag schema／初始 pose／frame／尺度仍待核對；不能將公告 IMU topic 等同已驗證實際 gyro／外參或新 runner。

## 3. 判斷方法：檢查資訊何時可取得

令 `T_i` 為第 i 幀 source→map pose，`P_1:i` 為截至當前幀的提供點雲，`z_≤t_i` 為當下之前已取得的感測量測，`M` 為提供的靜態地圖。合規依賴應為：

`T_i = F(M, init, P_1:i, z_≤t_i, 已固定的算法設定)`。

地圖配準殘差例如 `r = nᵀ(R_i s_i p + t_i − q)`，使用當前點雲、靜態 map 與歷史狀態，本身可以因果執行。GPU kNN／PCA、逐幀 ICP、使用上一幀校正的 EMA 也不因名稱或硬體而違規。關鍵是初值、尺度、filter、門檻與結果發布是否間接依賴尚未取得的資訊。

可檢驗預測是**前綴不變性**：保持截止時間之前的全部輸入相同，只改其後資料，已輸出的舊 pose 必須不變。亦應比較只提供前 K 幀與提供完整資料的前 K 幀結果，並檢查各感測器實際使用的最大整數 timestamp。單純先把整包資料讀進記憶體不直接證明違規；使用其未來內容影響舊估計才是問題。計算速度慢於即時也不直接等於非因果。

## 4. 程式審查發現

以下判定針對把該流程的結果用於**正式評分**。保留不合規方法供歷史重現或離線分析，不等於已證明有違規提交；本輪沒有登入 Kaggle 核對最終選擇。

| 嚴重度／發現 | 直接證據 | 對正式結果的影響 |
| --- | --- | --- |
| **阻擋：RTS 雙向平滑** | [sensor_fusion.py](../scripts/sensor_fusion.py) `bias_rts` L84–128、`position_rts` L190–243：反向迴圈用 i+1 的 smoothed state 修正 i；`fuse_priors` 呼叫位置 RTS／heading RTS | 已評分 `motion_rts`、`heading_rts`、`pose_heading_rts`、`independent_heading_rts`、`turn_heading_rts` 不合因果規則；`pose_yaw` 仍呼叫位置 RTS |
| **阻擋：fixed 也使用未來與整段估計** | [prepare_sensor_prior.py](../scripts/prepare_sensor_prior.py) L53–70：中心 Savitzky–Golay、中心 gradient、整段 GNSS／IMU heading offset median；L22 雙端位置插值，L70 SLERP | 把 fusion 改成 `fixed` 不能解決問題；量測右端若晚於當前時刻，也屬未來依賴。整段估 bias 再套回首幀不是在線校準 |
| **阻擋：GPU／CPU map smoother batch graph** | [run_map_smoother_gpu.py](../scripts/run_map_smoother_gpu.py) L230 `motion_rts`、L302–304 整段 `smooth_trajectory`；[run_map_smoother.py](../scripts/run_map_smoother.py) L220、L270–272；[slow_map_graph.py](../scripts/slow_map_graph.py) L44–84 相鄰兩端接受及 12 s block 中位 anchor，L179–188 全狀態 least_squares，L192 起整段輸出 | 即使前面逐幀幾何成立，後段仍利用後來觀測改較早 pose；無 anchor 時保留的 `motion_rts` 本身也不合規。0.77 那輪不作合規候選 |
| **阻擋：定位後 joint 回寫** | [accuracy/benchmark.py](../scripts/accuracy/benchmark.py) L616 起定位後啟動 `refine_horizontal_joint.py`；[refine_horizontal_joint.py](../scripts/refine_horizontal_joint.py) L192–229 整段求解，通過時覆寫 `result.txt`／`diagnostic.csv` | `surfel_joint` 套用時違反因果與離線後處理限制；歷史未套用／bytes 相同也不能消除 base `motion_rts` 的非因果 |
| **阻擋：高度／姿態參考中心濾波** | [prepare_surface_height.py](../scripts/prepare_surface_height.py) L17–30 雙端補值與 ±0.25 s 中位數；L142–145 套於高度及 roll／pitch | `surface`／`surface_tilt` 額外含未來資料；CSV 不含 Z／roll／pitch 也不能推定不影響其後 XY／yaw 配準 |
| **阻擋：現成軌跡後處理匯出** | [trajectory_filter_lab.py](../scripts/trajectory_filter_lab.py) L53–71 居中局部多項式、L82–114 RTS；L242–283 讀完整既有 pose 後產生 SG／RTS／EMA 候選 CSV／TXT | 此入口的後處理候選不能作正式結果。EMA 核心是 causal，但本工具在定位完成後另行處理，而且 base 也可能非因果；不因此取得合規資格 |
| **受牽連：三相機重建 prior** | [run_rig_full.py](../scripts/run_rig_full.py) L65–69 要求 `heading_rts` prior，L128–129 交給 `PriorTrajectory`；[calibrated_rig_geometry.py](../scripts/calibrated_rig_geometry.py) L70–106 使用雙端軌跡插值／SLERP 做運動補償 | 現有三相機實作繼承非因果來源，不能直接宣稱合規。跨時間尺度標定及影像時間可用性也需逐項審查；固定算法參數本身並非禁止項目 |
| **未完成：提交可重現性認證** | 現有 run 有 SHA、參數與 manifest，但歷史分數沒有獨立核對 Kaggle 上傳 bytes，Track3 尚無已驗證 runner／sample | 保存 hash、列數通過及退出碼 0 都不能取代因果性與最終 submission 重現檢查 |

### 已核對可保留的部分與界限

- C++ [sensor_guided.cpp](../workspace/src/localization/src/sensor_guided.cpp) L284–324 的上一幀 correction／EMA 本身僅使用歷史與當前資料；問題來自傳入的非因果 prior 等上游。不能僅憑該函式是逐幀就認證整條 pipeline。
- C++ [localization.cpp](../workspace/src/localization/src/localization.cpp) L326–345 raw bag reader 只消費 timestamp ≤當前幀的訊息；`legacy` 前一 pose 初始化的結構未見上述 backward smoothing。**本輪沒有重跑 legacy，也未證明其輸出或精度符合完整提交要求，不將它自動升為推薦。**
- 地圖讀取／快取主要使用 `pcl::PointXYZ` 或 Python binary XYZ；未在已追查的正式定位入口發現 `tl/th` 軌跡推導。這是所審路徑的結論，非對所有歷史檔案作無限範圍保證。
- [analyze_external_reference.py](../scripts/analyze_external_reference.py) 的 `reject_score_probing` 仍在讀檔／建立 run 前拒絕交叉提交與分項反推參數。版本差異、未發布的 SE(2) 診斷屬分析用途；未發現主定位 runner 把 Kaggle score 當殘差、校正目標或真值輸入。外部參考 CSV 來源算法仍不明，不能直接當本專案可重現的正式結果。

## 5. 本輪實際驗證與修改範圍

證據放在 [`outputs/run/track00_test_2610231309/`](../outputs/run/track00_test_2610231309/)，沒有建立其他 runs／best／submission 分類。包括改名前後 SHA／路徑 manifest、原始抽取全文、關鍵頁 PNG、修改前程式 SHA、合成反例程式與結果、規則工具盤點／局部回歸紀錄。

五個**合成**反例保持截止時刻以前量測不變，只改未來輸入；較早輸出最大改變如下，僅證明資料依賴，**不是實測定位誤差、改善量或生產參數**：

| 機制 | 較早輸出最大差異 |
| --- | ---: |
| `position_rts` | 0.0189079222 m |
| `bias_rts` | 0.8478878848° |
| `fixed` 全段 heading median | 2° |
| 雙端位置插值 | 0.1 m |
| surface 補值／中心 median | 0.5 m |

本輪修正 `competition_rules.py` 的現行原件路徑／SHA、規則版本與已公告 Track3 契約；`check_competition_rules.py` 將檔案盤點 `inventory_checks_passed` 與是否能提交分開。**目前三條 Track 均無已驗證因果 runner，工具 `checks_passed=false` 並退出 2**；這是刻意回報阻擋項目，不代表程式執行崩潰。此工具維護已審規則契約，不會自動分析所有 solver，也不認證輸出或精度。10/22 後的 `gnss_smoother_supported=false` 指正式合規支援，歷史程式仍保留。

同步修復連結與主要文件入口的歷史提示；`analyze_external_reference.py` 僅更新規則參照，原有禁用行為保留。**未修改定位 solver、CUDA 核心、算法參數或現存最佳 CSV；未啟動 Docker／GPU、完整 Track、外部提交或寄信。**

實際檢查結果：9 項針對性規則檢查通過（包含缺檔／錯誤 SHA 的失敗檢測、Track3 已知與未知契約分開、GNSS 許可不冒充 smoother 合規、榜分反推入口仍拒絕，以及只有三個規則相關 Python 檔變動）。三條 Track 的盤點命令均如預期退出 2；Track1／2 純檔案盤點通過，觀察到 3011／3299 幀；Track3 候選 `raw_ply` 位置觀察到 0 幀，不能據此排除其他尚未盤點的下載位置。四份 PDF 改名前後 SHA 相同；已核對的 18 份既有結果／metadata bytes 未變。主要規則與審查連結無失效，歷史命令檔 16 沒有未註解命令，`git diff --check` 通過。這些檢查不代表定位或精度通過。

## 6. 後續算法應如何改造（本輪尚未實作）

先建立前向的 `x_i=[位置,速度,航向,必要bias]` 狀態：只用 timestamp ≤當前幀的 GNSS／IMU 預測與更新，再以當前點雲對提供地圖的可靠殘差更新當前 pose。初始化速度不能偷用首幀以後量測；heading block 若要用完一段資料，只能在 block 結束後影響當前及未來，不能將觀測掛回 block 中心修改過去。

保留逐幀 GPU 配準、可觀測性與 outlier 檢查；移除往回套用的 RTS、對稱時間窗、全段回算 bias 與 batch trajectory 重寫。歷史狀態可供估計當前狀態，但已發布的結果須固定。Track3 要另建無 GNSS 的初值／motion／失敗恢復流程，不能用假 GNSS 或其他 Track prior 替代。

驗證依序是合成因果反例、真實短段前綴截斷／未來擾動、最大已用感測 timestamp、逐幀輸出不回寫，以及乾淨環境重現相同來源／設定的 CSV。精度需另作正常算法比較；通過因果檢查不等於更準。完成短段驗證後才整理使用者手動完整 Track 命令；本輪不推薦再跑既有不合規 full。

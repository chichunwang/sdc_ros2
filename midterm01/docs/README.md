**2026-10-23 現行規則：**請以 GPU 文件中的 [00 整合新版（31 頁）](../../midterm01_gpu/doc/Competition%20Regulations/00_SDC_2026_Midterm01_Localization_%E6%95%B4%E5%90%88%E6%96%B0%E7%89%88_20261022.pdf)、[01 因果定位細則（4 頁）](../../midterm01_gpu/doc/Competition%20Regulations/01_Mid01_Updates_Rules.pdf) 及 [48 規則與程式稽核](../../midterm01_gpu/doc/48_RULES_AND_CODE_AUDIT_20261023.md) 為準。所有 Track 禁止未來資訊回修、RTS／backward smoothing 及定位後離線軌跡修正；Track1／2 GNSS 輸入合法不代表 RTS 合規。以下歷史內容不作現行提交推薦。

# Midterm01 題目分析與交付要求

更新：2026-10-23。以下保留當時歷史成績與原規則記錄；現行規則以 [00 整合新版](../../midterm01_gpu/doc/Competition%20Regulations/00_SDC_2026_Midterm01_Localization_%E6%95%B4%E5%90%88%E6%96%B0%E7%89%88_20261022.pdf)、[01 因果定位細則](../../midterm01_gpu/doc/Competition%20Regulations/01_Mid01_Updates_Rules.pdf)及[48 規則與程式稽核](../../midterm01_gpu/doc/48_RULES_AND_CODE_AUDIT_20261023.md)為準，10/21 歷史規則分析見 [45](../../midterm01_gpu/doc/45_COMPETITION_UPDATES_20261021.md)。使用者單人完成，實際貢獻記為 100%。

**目前開發／建置／測試統一使用 [GPU 工作區](../../midterm01_gpu/doc/27_PROJECT_OVERVIEW.md)**，日常命令見 [GPU 操作手冊](../../midterm01_gpu/doc/08_MANUAL_WORKFLOW.md)。新版為 `sensor_guided`／`cuda_robust_icp`，不是舊 CPU NDT 或首次 CUDA D2D。

2026-10-17 當時，新版 Track1／Track2 完整序列已完成本機檢查；截圖回報 Track1 Kaggle **1.22369**、同圖 baseline **1.79075**，Track2 分數尚未回報。這是歷史成績快照，不是現行最佳紀錄。CPU Track1 另有完整 3011 筆結果，已修正 run 讀取權限，見 [CPU 完成紀錄](CPU_TRACK1_COMPLETED_20261017.md)。各版本成績與證據不能混用。

## 官方要求與證據來源

現行規則以 [00 整合新版](../../midterm01_gpu/doc/Competition%20Regulations/00_SDC_2026_Midterm01_Localization_%E6%95%B4%E5%90%88%E6%96%B0%E7%89%88_20261022.pdf)、[01 因果定位細則](../../midterm01_gpu/doc/Competition%20Regulations/01_Mid01_Updates_Rules.pdf)及[48 規則與程式稽核](../../midterm01_gpu/doc/48_RULES_AND_CODE_AUDIT_20261023.md)為準。Track1／2 的歷史幀數、輸入與格式仍分別適用於各自 Track；Track3 已公告 3142 幀、無 GNSS pose，並有 IMU topics 與 CSV 欄位規格。貼文未指定完整 idx 集合、起算方式、yaw 單位／座標系；本機資料與 runner 支援均未因此核驗。原 10/21 解讀見[45](../../midterm01_gpu/doc/45_COMPETITION_UPDATES_20261021.md)，只作歷史參考。

| 項目 | 題目要求／規格 | 來源 |
| --- | --- | --- |
| 定位任務 | 把影像重建點雲配準到 LiDAR map，估計每幀 pose | PDF p.5–6 |
| Track01／02 資料 | Track01 3011 幀、Track02 3299 幀，各自地圖；保留原規則歷史契約 | 原 PDF p.8；既有資料紀錄見首跑報告 |
| 必做方法 | 自行設計 scan matching；可修改模板、用其他套件或從零實作 | PDF p.12、15 |
| 計分門檻 | 原規則的 baseline／排名描述依現行公告更新 | 原 PDF p.15、20；現行規則見 00／01 |
| 可用額外輸入 | Track1／2 GNSS 仍合法；Track3 無 GNSS pose，公告三相機與 `/imu/data` | 原 PDF p.8、15；Track3 topics 見 [Kaggle 原文紀錄](KAGGLE_DATASET_SPEC.md) |
| Track03 公告資料 | 3142 幀；本機資料／runner 支援未核驗，未登入核驗 Kaggle 頁面 | 使用者轉貼 Kaggle 文字；無 GNSS pose 見 00 p.28 |
| CSV | 三 Track 均為 `idx,x,y,yaw`；資料列 Track01／02／03 分別為 3011／3299／3142，不含表頭；Track03 含表頭共 3143 列 | [Kaggle 原文紀錄](KAGGLE_DATASET_SPEC.md) |
| 結果與文件 | 三個 Track 的 E3 CSV 為 `result_1.csv`、`result_2.csv`、`result_3.csv`；另含程式、執行 README、報告與 4–6 分鐘簡報 | PDF p.21–23；檔名見 00 p.23 |

PDF 把 `localization.cpp` 標成 TODO，但取得的 package 已含基礎 NDT；原 R0 已有實跑紀錄，新版改善須與原有功能分開描述。官方提供重建點雲，LoGeR 原始碼為選用，不必先重跑影像重建。

## 目前實作與證據

| 版本 | 行為／文件 |
| --- | --- |
| 原 CPU PCL NDT | 第一幀 init_guess，後續前一接受 pose；bag 可讀但未融合。單幀／前 50 幀見 [Ubuntu 首跑](UBUNTU_FIRST_RUN_20261017.md)，完整 Track1 見 [CPU 完成紀錄](CPU_TRACK1_COMPLETED_20261017.md) |
| 首次 CUDA D2D | 與 PCL 不同目標函數／最佳化；曾完整追蹤漂移。見 [歷史比較](../../midterm01_gpu/doc/05_ACCURACY_COMPARISON_20261017.md) |
| 目前 sensor_guided | GNSS／IMU 同步先驗、CUDA 最近鄰、CPU robust 四自由度細修，拒絕時回當下先驗。見 [新版實測與限制](../../midterm01_gpu/doc/06_SENSOR_GUIDED_20261017.md) |

三階段資料流、原有功能與新增貢獻見 [演算法教學](../../midterm01_gpu/doc/07_ALGORITHM_WALKTHROUGH_20261017.md)。新版 profile 必須明確選用，CLI 未指定時仍是 legacy。兩個 workspace 與 outputs 獨立，原始資料唯一放 `midterm01/data`，GPU 唯讀共用。

Track01／02 前相機與 MapAnything 的數量、時間、index、bag／影像對應與 camera_info 位置已保存於首跑報告；Track01／02 sample 本地搜尋尚未找到，Track03 sample 未核對。Track03 公告 CSV 為 3142 資料列、含表頭 3143 紀錄。PLY 只掃第一層，以檔名整數奈秒排序；MapAnything index 不直接作 submission，也不直接沿用前相機尺度或重套外參。原始資料與檢查入口見 [data/README](../data/README.md)。

launch 的 RViz 開關、空字串停用快取、outputs 唯一路徑、移除轉檔未使用 import 均已完成；舊 Windows 階段的相反敘述不再是待辦。GPU benchmark 有來源／frame／耗時／GPU／驗證記錄，詳細檔案用途見操作手冊。

## 座標、單位與資料契約

配準求出的首先是點雲座標到地圖的轉換。定義 `T_A_B` 表示將 B 中的點轉到 A：

```text
p_map = T_map_source × p_source
T_map_body = T_map_source × inverse(T_body_source)
```

第二式只在 source 尚未轉到 body、且已確認該外參定義時適用。若資料已做轉換，不再重複套用。應檢查初始矩陣與重建點雲是否具有相同 source 定義，以及提交要求的究竟是哪個車體參考點。

PLY 名稱、bag timestamp、header timestamp 與 Kaggle idx 分別核對，不把不同欄位視為同一時間或 ID。MapAnything index 中的奈秒應保留整數／字串精度，不先轉為浮點數再比對。切換三相機時保留原資料夾與 index，先驗證可用的 `ply_path` 與逐幀對應。

目前轉檔 yaw 為弧度；PDF 評分旋轉誤差以度描述。兩者不能直接混用，也不能只因評分寫 degree 就自行修改提交 yaw，須依官方 sample／補充說明確認。

## 評分與實驗證據

原 PDF p.19 的公式經圖像核對，是兩個 RMSE 相加；10/21 補充另更新競賽排名配分，現行規則以[45](../../midterm01_gpu/doc/45_COMPETITION_UPDATES_20261021.md)為準：

$$
E_n = \sqrt{\frac{1}{n}\sum_{i=1}^{n}\|\Delta p_i\|^2}
+ 0.2\sqrt{\frac{1}{n}\sum_{i=1}^{n}\|\Delta\theta_i\|^2}
$$

平移誤差以公尺、旋轉誤差以度，誤差越低越好；不是將兩種誤差放入同一根號。原 PDF 未附完整評分程式，實際維度、角度環繞、轉換與 ID 對應仍依官方評分規格。補充公告禁止利用排行榜分數反推隱藏真值；一般算法／參數比較仍可進行，但不得用榜分推測隱藏標籤。Track2 的 `tl/th` 說明是警示，並非明文全面禁止；本專案採保守政策，只提交 XYZ 位置、不使用 `tl/th`。

- **助教 baseline** 是課程計分門檻。Track1 使用者截圖為 1.79075，新版提交為 1.22369；Track2 baseline／分數仍待回報。來源與 CSV SHA256 見新版實測，不將本機殘差當競賽分數。
- **自己的 R0** 是原 CPU 前相機 NDT，已有單幀／短序列與指定 Track1 完整結果，不等於助教 baseline。原有清理、下採樣、局部地圖與 NDT 不能說成自己的新增貢獻。
- **PCL 收斂／fitness** 用於配準診斷，不是正式 ATE。GNSS 允許作為輸入，PDF 沒有保證它是評分 ground truth。

修改後依需要從單幀／固定短片段驗證；後續完整測試由使用者手動執行並回報。保存 commit／未提交差異、完整參數、資料來源、frame 對應、輸出、耗時、失敗／fallback 與 Kaggle 分數。Track01／02 為 3011／3299 幀；Track03 公告 3142 幀，完整實驗時間應由實測估算。實驗表在 [experiments/README](../experiments/README.md)。

目前已加入感測器先驗、robust ICP、CUDA 搜尋與修正門檻。後續取捨依失敗案例及使用者完整測試／Kaggle 分數，新增改動須有開關與對照；沒有消融實驗時，不把整組修改的成績歸因於某一項或單純 GPU 加速。

## 官方期限與配分

下表照錄原 PDF，年份依本次 2026 課程。2026-10-21 補充公告未改動既有期限；工作安排使用台灣時間，PDF 未明示 Kaggle 截止時區，網站最新日期／時區及補交流程仍須另確認。排名依補充公告的基數 56、52、48、42、36、30、late 22、低於基準 0，再乘 `r=(0.3,0.3,0.4)`；競賽上限 56、報告 30、簡報 20，合計 106，照錄且不正規化。詳見[45](../../midterm01_gpu/doc/45_COMPETITION_UPDATES_20261021.md)。

| 事項 | PDF 公告 | 頁碼 |
| --- | --- | --- |
| 分組登記 | 2026-10-16 15:00；單人一組已由使用者決定，登記狀態另查 | p.23 |
| Kaggle 截止 | **2026-10-29 09:00**；每日提交上限公告為 15 | p.16 |
| 簡報寄送 | **2026-10-29 09:00** 前寄簡報檔或連結到 `neal516.en14@nycu.edu.tw` | p.21 |
| 課堂簡報 | **2026-10-29，4–6 分鐘** | p.21 |
| E3 程式與報告 | **2026-11-05 23:59** | p.22 |
| Late submission | 公告列 11 分與 11/05 23:59，適用方式另依課程說明 | p.20 |

原 PDF p.20 的逐 Track 級距是歷史規則，已由 10/21 補充公告的排名基數取代。現行補充列出基數 56、52、48、42、36、30、late 22、低於 baseline 0，並乘 `r=(0.3,0.3,0.4)`；競賽 56、報告 30、簡報 20 合計 106，原樣記載、不正規化。計分細節及未明確處見[45](../../midterm01_gpu/doc/45_COMPETITION_UPDATES_20261021.md)。

寄送、Kaggle 提交與 E3 上傳是交付事項，不表示本次已操作或授權後續代理自動對外寄信。

## 報告、簡報與 E3

PDF p.21 要求 pipeline、自己的貢獻、問題與解法、工作分配及比例等。單人依實際內容列出工作並記為 100%。僅使用開源專案而沒有實質修改或原創貢獻，該貢獻項目為零分；因此需保留「問題、設計、對照結果」證據。

建議報告依任務與資料、方法與座標、自己的改善、各 Track 實驗、失敗分析、重現方法排列；納入 Track3 已公告的 topics 與 CSV 列數，並標明尚未核對的 idx／yaw 契約。簡報控制在 4–6 分鐘，重點放自己的設計與效果。詳見 [reports/README](../reports/README.md)。

原 PDF p.22 的 Track1／2 命名範本（歷史）：

```text
<student ID1>_<student ID2>_localization.zip
localization_report_<student ID1>_<student ID2>.pdf
```

單人學號填法未另列，繳交前依課程確認，不自行補第二個學號。壓縮檔至少包含：

- GPU `workspace/src/localization/` 完整程式資料夾，排除原始資料集；本版本另依賴自行撰寫的 `workspace/src/cuda_ndt_backend/`，一併交付。
- GPU Dockerfile／Compose、`scripts/`（含 sensor prior／benchmark）、必要設定、固定上游版本／SHA256 與授權。
- 三個 Track 的 E3 CSV 檔名依 00 p.23 為 `result_1.csv`、`result_2.csv`、`result_3.csv`。此為交付契約記錄，不代表已產生或提交任何檔案。
- 報告 PDF，可選附簡報影片連結。
- 能重現各 Track 的 README、依賴、版本與參數；Track3 的資料來源／runner 支援及未確認座標契約仍須另行核實。

交付前確認各已發布 Track 的 CSV 表頭／完整 ID／行數／有限值、地圖未混用、最佳 CSV 與版本相符、Kaggle 實際成績已保存。這段 2026-10-17 的紀錄只可依當時使用者截圖宣稱新版 Track1 提交優於同圖 baseline；不作為現行 Track3 狀態或目前最佳分數。

# Midterm01 題目分析與交付要求

更新：2026-10-17。正式要求依同目錄 [官方 PDF](<SDC 2026 Midterm01 Competition -  Localization.pdf>) 全部 24 頁與 [Kaggle 原文紀錄](KAGGLE_DATASET_SPEC.md)；本次未登入網站查最新公告。使用者單人完成，實際貢獻記為 100%。

**目前開發／建置／測試統一使用 [GPU 工作區](../../midterm01_gpu/doc/27_PROJECT_OVERVIEW.md)**，日常命令見 [GPU 操作手冊](../../midterm01_gpu/doc/08_MANUAL_WORKFLOW.md)。新版為 `sensor_guided`／`cuda_robust_icp`，不是舊 CPU NDT 或首次 CUDA D2D。

新版兩條完整序列已完成本機檢查。使用者截圖回報 Track1 Kaggle **1.22369**、同圖 baseline **1.79075**；Track2 分數尚未回報。CPU Track1 另有完整 3011 筆結果，已修正 run 讀取權限，見 [CPU 完成紀錄](CPU_TRACK1_COMPLETED_20261017.md)。各版本成績與證據不能混用。

## 官方要求與證據來源

| 項目 | 題目要求／規格 | 來源 |
| --- | --- | --- |
| 定位任務 | 把影像重建點雲配準到 LiDAR map，估計每幀 pose | PDF p.5–6 |
| 兩條 Track | Track01 3011 幀、Track02 3299 幀，各自地圖 | PDF p.8 |
| 必做方法 | 自行設計 scan matching；可修改模板、用其他套件或從零實作 | PDF p.12、15 |
| 計分門檻 | 必須超過助教 baseline；兩條 Track 分別計排名 | PDF p.15、20 |
| 可用額外輸入 | 三相機、IMU、GNSS；助教 baseline 只用前相機點雲 | PDF p.8、15 |
| CSV | `idx,x,y,yaw`；資料列分別恰為 3011／3299，不含表頭 | [Kaggle 原文紀錄](KAGGLE_DATASET_SPEC.md) |
| 結果與文件 | 兩條最佳 CSV、程式、執行 README、報告與 4–6 分鐘簡報 | PDF p.21–22 |

PDF 把 `localization.cpp` 標成 TODO，但取得的 package 已含基礎 NDT；原 R0 已有實跑紀錄，新版改善須與原有功能分開描述。官方提供重建點雲，LoGeR 原始碼為選用，不必先重跑影像重建。

## 目前實作與證據

| 版本 | 行為／文件 |
| --- | --- |
| 原 CPU PCL NDT | 第一幀 init_guess，後續前一接受 pose；bag 可讀但未融合。單幀／前 50 幀見 [Ubuntu 首跑](UBUNTU_FIRST_RUN_20261017.md)，完整 Track1 見 [CPU 完成紀錄](CPU_TRACK1_COMPLETED_20261017.md) |
| 首次 CUDA D2D | 與 PCL 不同目標函數／最佳化；曾完整追蹤漂移。見 [歷史比較](../../midterm01_gpu/doc/05_ACCURACY_COMPARISON_20261017.md) |
| 目前 sensor_guided | GNSS／IMU 同步先驗、CUDA 最近鄰、CPU robust 四自由度細修，拒絕時回當下先驗。見 [新版實測與限制](../../midterm01_gpu/doc/06_SENSOR_GUIDED_20261017.md) |

三階段資料流、原有功能與新增貢獻見 [演算法教學](../../midterm01_gpu/doc/07_ALGORITHM_WALKTHROUGH_20261017.md)。新版 profile 必須明確選用，CLI 未指定時仍是 legacy。兩個 workspace 與 outputs 獨立，原始資料唯一放 `midterm01/data`，GPU 唯讀共用。

前相機與 MapAnything 的數量、時間、index、bag／影像對應與 camera_info 位置已保存於首跑報告；官方 submission sample 本地尚未找到。PLY 只掃第一層，以檔名整數奈秒排序；MapAnything index 不直接作 submission，也不直接沿用前相機尺度或重套外參。原始資料與檢查入口見 [data/README](../data/README.md)。

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

PDF p.19 的公式經圖像核對，是兩個 RMSE 相加：

$$
E_n = \sqrt{\frac{1}{n}\sum_{i=1}^{n}\|\Delta p_i\|^2}
+ 0.2\sqrt{\frac{1}{n}\sum_{i=1}^{n}\|\Delta\theta_i\|^2}
$$

平移誤差以公尺、旋轉誤差以度，誤差越低越好；不是將兩種誤差放入同一根號。PDF 未附完整評分程式，實際維度、角度環繞、轉換與 ID 對應仍依官方評分規格。

- **助教 baseline** 是課程計分門檻。Track1 使用者截圖為 1.79075，新版提交為 1.22369；Track2 baseline／分數仍待回報。來源與 CSV SHA256 見新版實測，不將本機殘差當競賽分數。
- **自己的 R0** 是原 CPU 前相機 NDT，已有單幀／短序列與指定 Track1 完整結果，不等於助教 baseline。原有清理、下採樣、局部地圖與 NDT 不能說成自己的新增貢獻。
- **PCL 收斂／fitness** 用於配準診斷，不是正式 ATE。GNSS 允許作為輸入，PDF 沒有保證它是評分 ground truth。

修改後依需要從單幀／固定短片段驗證；後續完整測試由使用者手動執行並回報。保存 commit／未提交差異、完整參數、資料來源、frame 對應、輸出、耗時、失敗／fallback 與 Kaggle 分數。兩條 Track 共 6310 幀，完整實驗時間應由實測估算。實驗表在 [experiments/README](../experiments/README.md)。

目前已加入感測器先驗、robust ICP、CUDA 搜尋與修正門檻。後續取捨依失敗案例及使用者完整測試／Kaggle 分數，新增改動須有開關與對照；沒有消融實驗時，不把整組修改的成績歸因於某一項或單純 GPU 加速。

## 官方期限與配分

下表照錄 PDF，年份依本次 2026 課程。工作安排使用台灣時間；PDF 未明示 Kaggle 截止時區，網站最新日期／時區及補交流程仍須另確認。

| 事項 | PDF 公告 | 頁碼 |
| --- | --- | --- |
| 分組登記 | 2026-10-16 15:00；單人一組已由使用者決定，登記狀態另查 | p.23 |
| Kaggle 截止 | **2026-10-29 09:00**；每日提交上限公告為 15 | p.16 |
| 簡報寄送 | **2026-10-29 09:00** 前寄簡報檔或連結到 `neal516.en14@nycu.edu.tw` | p.21 |
| 課堂簡報 | **2026-10-29，4–6 分鐘** | p.21 |
| E3 程式與報告 | **2026-11-05 23:59** | p.22 |
| Late submission | 公告列 11 分與 11/05 23:59，適用方式另依課程說明 | p.20 |

PDF p.20：競賽最多 56 分、報告 30 分、簡報 20 分。每條 Track 為第 1 名 28、第 2 名 26、第 3 名 24、Top 50% 21、Top 75% 18、其他超過 baseline 15、late submission 11、低於 baseline 0；兩條 Track 排名分數相加。公告上限合計 106，照錄而不自行正規化。

寄送、Kaggle 提交與 E3 上傳是交付事項，不表示本次已操作或授權後續代理自動對外寄信。

## 報告、簡報與 E3

PDF p.21 要求 pipeline、自己的貢獻、問題與解法、工作分配及比例等。單人依實際內容列出工作並記為 100%。僅使用開源專案而沒有實質修改或原創貢獻，該貢獻項目為零分；因此需保留「問題、設計、對照結果」證據。

建議報告依任務與資料、方法與座標、自己的改善、兩條 Track 實驗、失敗分析、重現方法排列。簡報控制在 4–6 分鐘，重點放自己的設計與效果。詳見 [reports/README](../reports/README.md)。

PDF p.22 的正式命名範本：

```text
<student ID1>_<student ID2>_localization.zip
localization_report_<student ID1>_<student ID2>.pdf
```

單人學號填法未另列，繳交前依課程確認，不自行補第二個學號。壓縮檔至少包含：

- GPU `workspace/src/localization/` 完整程式資料夾，排除原始資料集；本版本另依賴自行撰寫的 `workspace/src/cuda_ndt_backend/`，一併交付。
- GPU Dockerfile／Compose、`scripts/`（含 sensor prior／benchmark）、必要設定、固定上游版本／SHA256 與授權。
- 兩條 Track 最佳提交 `result_1.csv`、`result_2.csv`。
- 報告 PDF，可選附簡報影片連結。
- 能重現兩條 Track 的 README、依賴、版本及最佳參數。

交付前確認 CSV 表頭／完整 ID／行數／有限值、兩條地圖未混用、最佳 CSV 與版本相符、Kaggle 實際成績已保存。目前只可依使用者截圖宣稱指定新版 Track1 提交優於同圖 baseline；Track2 與最終交付仍待完成。

**2026-10-23 現行規則提示：**因果定位適用所有 Track；不得以未來資訊回修 pose、使用 RTS／backward smoothing，或在定位後離線修正軌跡。Track1／2 GNSS 仍可作輸入，但 RTS 不因此合規。本文相關算法、命令與成績均為歷史紀錄，不作現行正式提交推薦；規則與程式影響見 [48](48_RULES_AND_CODE_AUDIT_20261023.md)。

# 45｜比賽補充規則、Track3 與專案調整

核對：2026-10-21（Asia/Taipei）。依使用者提供的兩份本機 PDF，讀取原公告 24 頁與補充公告 10 頁，並檢視補充公告全部頁面；未以此宣稱已查詢 Kaggle 即時規則或排名。

## 正式規則放在哪裡

| 文件 | 用途 | SHA256 |
| --- | --- | --- |
| [000｜原比賽規則](<Competition Regulations/90_舊版_已取代_可刪除_SDC_2026_Midterm01_Localization_24頁.pdf>)，24 頁 | 任務、輸入、評分公式、簡報與 E3 交付 | `8f1bcd2d1e3a43f7420d96d601d51561f2c5ad426b662b62534d296dd0a2bace` |
| [001｜2026-10-21 補充公告](<Competition Regulations/91_舊版_已取代_可刪除_Mid01_Updates_Adjustments_20261021.pdf>)，10 頁 | GNSS 說明、地圖時間欄位、榜分限制、新 Track3 與排名配分 | `4c2fd524a82299f912fae40e5d2fdcf5f7349f4af14641e6f84ac70e79f3feac` |

兩份原件集中在 `midterm01_gpu/doc/`，使用 `000`／`001` 排在其他文件之前；開發文件仍按 `00_`、`01_`…排列，`01` 仍是程式架構入口。舊 `midterm01/docs/` 的原 PDF 已搬走，現位置的 bytes 與 Git 中原件 SHA 一致，不重建第二份原件。補充公告明確更新的部分優先，其餘沿用原公告；未交代的契約保留待確認。

## 補充公告在說什麼

### 1. Track1／2 的 GNSS 方法仍合法

補充 p.2／7 明言 GNSS sensor fusion 符合原競賽規則。老師的顧慮是：若 GNSS 已提供接近正解的全域 pose，榜分可能主要反映 GNSS 品質，難以評估點雲配準能力。因此新增不提供 GNSS pose 的 Track3，並未追溯禁止 Track1／2 的 GNSS。

本專案的 `motion_rts＋directional＋carry`、GNSS bias／gyro／NHC smoother 在 Track1／2 仍可保留。1.15008／0.30488 是原已回報分數，沒有因本公告被作廢；本輪也沒有產生新定位分數。GNSS 仍只是量測輸入，不是真值。

### 2. Track2 地圖的 tl／th 是未移除的時間欄位

補充 p.3／8 表示地圖預處理誤留時間欄位，可能透露點的觀測時間與建圖軌跡，但可靠性及定位效益未驗證。公告要求審慎使用、不可直接視為可靠定位依據；它沒有寫成「所有使用皆禁止」。

本專案採 **地圖定位僅使用 XYZ** 的實作政策，不從 `tl/th` 重建行駛軌跡或產生 pose prior。核對入口：C++ `localization.cpp::LoadMap` 與 `map_modified.cpp` 使用 `pcl::PointXYZ`；Python `three_camera_calibration.py::load_binary_map` 只接受 binary XYZ 快取。保留原始 PCD 與既有 SHA，不為移除額外欄位覆寫資料。

### 3. 一般模型比較可以，利用榜分反推隱藏誤差不可以

補充 p.4／9 區分正常參數／模型比較與 leaderboard probing。系統性調整輸出軌跡平移／旋轉、用多次評分反推隱藏 ground truth，再修正預測，不應作正式評分／排名依據。這限制的是利用評分服務推算未知誤差的做法，不是禁止可解釋的算法實驗。

本專案先前 [41](41_EXTERNAL_REFERENCE_ANALYSIS_20261020.md) 建議交叉本機與外部 XY／yaw，再提交分解位置與航向分數。公告沒有逐字列出這種欄位組合；但其目的就是用額外榜分推算未知誤差分項，因此**本專案採保守處理，撤回該提交建議並停用生成入口**。這是根據公告作出的專案決策，不將它描述為官方逐字禁止所有軌跡融合。

既有兩份交叉 CSV 與研究 JSON／SHA 保留作歷史證據，不推薦正式提交，也不把外部預測或榜分推導出的分項當訓練目標、尺度標定或 solver 調整依據。離線版本差異、未套到輸出的 SE(2) 對照與運動診斷仍可用於理解資料。正常從觀測模型、噪聲、幾何與數值證據提出算法，再由使用者手動評分的流程繼續。

### 4. 新增 Track3，重點轉向無 GNSS 的地圖定位

補充 p.5／10：

| 項目 | 已公告內容 | 本機／待確認事項 |
| --- | --- | --- |
| 輸入 | front／front-left／front-right 影像、front-camera PLY、LiDAR map | 尚無 `midterm01/data/Track3`；不假定 PLY producer／尺度與 Track1／2 相同 |
| GNSS | 不提供 GNSS pose | 現有 GNSS 初始化／bias graph 不能直接套用 |
| 其他先驗 | 未在補充中列出 | IMU、初始 pose、時間與 frame／校正契約均待實際資料確認 |
| 發布 | 預定 2026-10-22 上課發布，可能調整 | 這是公告計畫，不是已下載或已確認發布 |
| Public | 軌跡前段約 10–30% | 精確比例未知，不能自行固定切成 20% |
| Private | 其餘後段，競賽截止後公布最終成績 | 不能把 public 的局部改善當全序列泛化證據 |
| 截止 | 與 Track1／2 相同，10/29 09:00 | 未改簡報及 E3 既有期限；官方時區／後續調整仍需確認 |
| 提交 | 補充未提供 sample、欄位與幀數 | 不把 3011／3299、yaw 單位或 `result_3.csv` 命名當已公告契約 |

Public／Private 的上述分段是對 **Track3** 的說明，沒有據此改寫 Track1／2 的評分切分。

### 5. 排名配分改為三條 Track 加權

補充 p.6 取代原公告 p.20 的雙 Track 配分。每條 Track 先依名次取基數，再乘權重 `r`；Track1／2 各 0.3，Track3 為 0.4。

| 排名級距 | 基數 | Track1 | Track2 | Track3 |
| --- | ---: | ---: | ---: | ---: |
| 第 1 | 56 | 16.8 | 16.8 | 22.4 |
| 第 2 | 52 | 15.6 | 15.6 | 20.8 |
| 第 3 | 48 | 14.4 | 14.4 | 19.2 |
| Top 50% | 42 | 12.6 | 12.6 | 16.8 |
| Top 75% | 36 | 10.8 | 10.8 | 14.4 |
| 其他超過 baseline | 30 | 9.0 | 9.0 | 12.0 |
| Late submission | 22 | 6.6 | 6.6 | 8.8 |
| Below baseline | 0 | 0 | 0 | 0 |

三條排名分數相加，最高仍 56；報告 30、簡報 20，公告上限合計 106，照錄不自行正規化。late 截止仍列 11/05 23:59，不將它誤認為正常競賽截止延期。百分位及同分的細部實作未另外公告。原 p.19 的 ATE 公式未被本次更新取代；地圖殘差、接受率與 GNSS 差異仍不能當正式分數。

## 專案已完成的調整

- 統一 PDF 位置及文件／程式引用，更新 README、架構入口、操作手冊、中央索引、協作規則與交付說明。
- 新增 [`competition_rules.py`](../scripts/competition_rules.py)：保存已核對的 PDF SHA、三條 Track 權重、已知幀數與 GNSS 契約。Track3 幀數為 `None`，不使用 Track2 的 3299 作預設。
- 新增 [`check_competition_rules.py`](../scripts/check_competition_rules.py)：唯讀核對原件 SHA、候選 PLY 檔名／整數時間與 PCD header，列出不使用的 map 欄位。檢查不是 pose、bag、提交或精度認證；Track3 在契約及 runner 尚未就緒時回傳非零退出碼，即使候選路徑裡有檔案也不誤報支援。
- run 命名工具支援 `track03_test_YYMMDDHHmm`／`track03_full_YYMMDDHHmm` 與碰撞序號。這是輸出路徑基礎支援；現行 benchmark／CPU、GPU smoother 的定位選項仍只支援 Track1／2，未加入假 GNSS 或套用舊 Track 的預設幀數。
- `analyze_external_reference.py` 保留離線比較，但拒絕 `--write-crossovers`／`--reported-reference-yaw-rmse`，停止產生交叉提交檔與從榜分推算未知分項。舊參數會在讀取輸入／建立 run 前報錯，既有研究結果不改寫。

主機規則盤點入口（不需 Docker、不跑定位）：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
python3 scripts/check_competition_rules.py --track 3
```

現況應顯示 Track3 未就緒並以 2 結束。`--track 1`／`--track 2` 可盤點已存在資料；這條命令不替代正式 submission validation。

## 算法接下來如何調整

Track1／2 保留已有合法 GNSS 主線。Track3 的首要改變是把全域位置觀測交回 map registration：先由官方提供的初始條件（若有）或全域幾何搜尋取得初值，再以 map 幾何支持估計每幀 pose／尺度，透過通過驗證的相鄰幀相對運動維持連續性。

概念目標可寫成 `Σ ρ(r_map(T_i,s_i)²) + Σ ρ(||Log(Z_ij⁻¹ T_i⁻¹ T_j)||²_Ω)`。`T_i` 是 source→map pose，`s_i` 是重建尺度，`Z_ij` 是有來源契約及尺度驗證的相對觀測；不是直接把單目平移當公尺量測。移除 GNSS 量測與 GNSS bias 狀態，不以零值或舊 Track 軌跡替代。IMU／gyro 因子只有在實際提供且時間、軸向、外參確認後才加入。

目前分層尺度／6-DOF GPU 幾何核心可研究重用；其 crop 初值、相鄰 GNSS 位移 gate、無 anchor 時返回 sensor 初始化及慢 bias graph 都依賴 GNSS，不能只刪一個因子就稱無 GNSS 方法。當 map 缺乏水平可觀測性時，平面殘差再小也可能沿路面滑動；需要非地面方向、時間一致性與重定位策略，而非放寬接受條件。

取得 Track3 後，先核對資料／時間／尺度與官方 sample，再做可解釋的短段 GPU 配準及退化／丟失後恢復檢查，最後提供完整命令由使用者執行。內部驗證應涵蓋分散時間及後段，避免只按 public 前段選模型；不靠盲目榜分搜尋。**本輪完成規則整理與契約支援，未宣稱已實作或驗證 Track3 無 GNSS 定位 solver。**

## 本輪證據

`outputs/run/track00_test_2610211638/` 保存兩份原件 SHA、抽取文字、補充公告逐頁 PNG、修改前差異與既有最佳／交叉 CSV 的保護 SHA，以及本輪規則檢查與回歸結果。未執行完整 Track、Kaggle／E3 提交或寄信；個人紀錄及歷史輸出 bytes 保留。

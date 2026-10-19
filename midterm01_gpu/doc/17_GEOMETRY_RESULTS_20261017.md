# 17｜方向性配準完整測試與 Kaggle 成績

紀錄日期：**2026-10-17（Asia/Taipei）**。使用者手動完成兩條 Track 並提交 Kaggle，於本輪回報分數及兩張排行榜截圖。本次僅讀取既有輸出、保存最佳 CSV 與更新文件，沒有重跑定位或代為提交。

2026-10-18 追加：使用者確認為 Team 3，新截圖排名已變為 Track1 第 4／Track2 第 2 名，最佳分數仍為 1.15275／0.30576。新快照與改進方向見 [目前算法與精度分析](20_CURRENT_ALGORITHM_AND_ACCURACY_20261018.md)；下表第 3／第 1 名保留為 2026-10-17 的歷史證據。

同日後續已依使用者要求完成 [三相機尺度／同步與短段對照](22_THREE_CAMERA_IMPLEMENTATION_20261018.md)：共享尺度候選 0.87124954、釐清混合 bag／header 時間，12 組短段均完整，但尚無精度改善或新 Kaggle 分數。本文最佳 CSV、分數與原始 provenance 不變，不將它們歸到三相機候選。

2026-10-19 後續：位置單項 motion_rts 已由使用者完整評分 **1.15008／0.30488** 並升為最佳，見 [35｜新最佳與航向消融](35_MOTION_RTS_AND_GNSS_HEADING_20261019.md)。本文的 heading_rts CSV／provenance 與歷史分數原樣保存。

## 成績

| 項目 | Track1 | Track2 |
| --- | ---: | ---: |
| 前最佳：heading_rts／geometry off | 1.16951 | 0.31967 |
| 本次：heading_rts／geometry directional | **1.15275** | **0.30576** |
| 分數降低量 | 0.01676 | 0.01391 |
| 相對前最佳降低 | **1.43%** | **4.35%** |
| 截圖當下排名 | 第 3 名 | 第 1 名 |
| 同圖 baseline | 1.79075 | 1.92587 |

百分比為 `(前最佳 − 本次分數) / 前最佳`，不是位置或姿態精度的提升比例。兩條均刷新個人最佳，但改善幅度比上一輪 heading_rts 小。排名只代表使用者截圖當下，沒有查詢即時排行榜。

使用者另提供評分公式截圖，與官方題目 p.19 一致：

```text
Score = sqrt(mean(||Δp||²)) + 0.2 × sqrt(mean(||Δθ||²))
      = 平移 RMSE（公尺） + 0.2 × 旋轉 RMSE（度）
```

在另一項不變時，旋轉 RMSE 降低 1° 可降低 0.2 分，平移 RMSE 降低 0.1 m 可降低 0.1 分。目前只有總分，不能反推出兩個 RMSE 各是多少，也不能把兩個降低量直接記為公尺。內部／目前輸出 CSV 的 yaw 是弧度，評分的旋轉誤差用度；不因公式顯示 degree 就直接修改 CSV 單位。評分誤差需要與真值比較，以下 map correction RMS、地圖殘差與 GNSS 差異都不能代替 Δp／Δθ。

原始截圖保留在對話附件；本機保存的是回報轉錄、run 證據與雜湊。Kaggle 截圖不含上傳檔案 hash，以下綁定依據為使用者回報、IDE 指定的完整 run 與本機核對；未從 Kaggle 獨立下載 submission metadata 核對。

## 這次已評分的是哪組改動

```text
--profile sensor_guided --sensor-fusion heading_rts --geometry-mode directional
```

保留 GNSS／IMU heading_rts、CUDA 最近鄰及 CPU 正則化四自由度求解。相對前最佳，新增局部地圖 PCA 的表面品質、法向／切向 anisotropic residual，以及至少 10 個可靠 source 對應／6 個不同 target 的接受保護。normal gain=2、tangent weight=0.05；尺度、先驗權重、correction smoothing 與上限沿用原設定。完整設計見 [方向性配準](15_GEOMETRY_DIRECTIONAL_20261017.md)。

這兩個完整 run 使用的 localization binary SHA256 均為 `390a7250a1a181b9050b774fcf3baff932325f487e68e890450cdb03c8835e4c`。兩者都在 [警告診斷修正](18_GEOMETRY_FALLBACK_20261017.md) **之前**產生；之後新增拒絕原因與候選支持數，不是這輪分數進步的來源。診斷修正曾以相同先驗重播前 450 幀，定位及幾何輸出逐 byte 相同，沒有把短測誤列為新的完整已評分 run。

## 完整輸出核對

| 項目 | Track1 | Track2 |
| --- | ---: | ---: |
| CSV／TXT 筆數 | 3011 | 3299 |
| 退出碼／validation_errors | 0／空 | 0／空 |
| 完整有限值檢查 | 通過 | 通過 |
| 接受 map 更新 | 3011 | 3047 |
| 返回當幀先驗 | 0 | 252 |
| 幾何 effective rank=1 比例 | 70.28% | 91.97% |
| 前處理＋ROS（s） | 337.422 | 155.719 |
| CUDA 搜尋＋CPU 細修／幾何（s） | 33.031 | 16.705 |
| map cache | 停用 | 停用 |

逐筆核對 CSV 表頭 `idx,x,y,yaw`、連續 0-based idx、有限數值，以及 CSV 與 TXT 的 X／Y／yaw 一致。`timings.csv.ndt_ms` 是歷史欄名，不把以上細修時間稱為 NDT 耗時。map 更新拒絕不等於丟失輸出；Track2 的第 410 幀原因見警告調查紀錄。

## 為何改善幅度有限：目前能支持的判斷

兩版的 timestamp、prior X／Y／yaw 逐筆一致；新的配準降低了地圖對 XY／yaw 的修正量：

| 全序列修正量 RMS，並非真值誤差 | Track1 off → directional | Track2 off → directional |
| --- | ---: | ---: |
| XY（m） | 0.06915 → 0.01661 | 0.04576 → 0.00993 |
| yaw（°） | 0.30349 → 0.07029 | 0.18462 → 0.03989 |
| Z（m） | 0.03830 → 0.04775 | 0.01633 → 0.02101 |

地圖法向資訊也大多以高度為主：Track1 **2998／3011** 幀、Track2 **3190／3299** 幀的 scaled normal information 對角線滿足 `Hzz > Hxx + Hyy`。這是依既定狀態尺度計算的局部指標，不能當作 covariance 或真值誤差分解；rank=1 本身也不是拒絕更新的條件。

**推論**：這輪主要效果之一可能是減少弱方向的地圖拉動，因而取得較保守的 XY／yaw 修正；不能由總分斷言剩餘誤差都來自 GNSS、yaw 或單一因素，也不能把所有降低量單獨歸功於某個門檻。較少修正與較多回退本身都不是精度改善保證。

使用者後續指定 [學習與算法改進原則](19_DEVELOPMENT_PRINCIPLES.md)：下一步先從時間／座標契約、幾何弱方向及觀測模型提出可解釋的算法假說，與使用者討論推導，再設計對應實驗。XY／yaw 消融可用來理解各模組的影響，並非立即啟動依排行榜反覆調權重的計畫；單一總分仍不能還原絕對平移／旋轉 RMSE。GNSS frame_id／完整外參仍有未確認事項，不在缺少依據時直接套固定 lever arm 或擬合一個看似改善的偏移。現有總分不足以選定位置濾波、heading 模型或更大的 map 權重；本輪沒有自動加入新算法或啟動完整實驗。

## run、最佳 CSV 與保存方式

原完整 run：

- Track1：`outputs/run/track01_full_2610172233/`。
- Track2：`outputs/run/track02_full_2610172238/`。

新的唯一保存目錄為 `outputs/run/track00_test_2610172250/`：

- [Track1 result_1.csv](../outputs/run/track00_test_2610172250/result_1.csv)
- [Track2 result_2.csv](../outputs/run/track00_test_2610172250/result_2.csv)
- [成績／來源／參數與 SHA256](../outputs/run/track00_test_2610172250/provenance.json)
- [兩版修正量與幾何比較](../outputs/run/track00_test_2610172250/comparison_not_accuracy.json)

| 檔案 | SHA256 |
| --- | --- |
| Track1 CSV | `dfa1900d095e654ebc31048eda78b53ba789616644b909ff4a1f100b5ce92834` |
| Track2 CSV | `a462479d0a5fee5cb9be4e883fd8dc0a56e27b394b0949d7136e02cf41a396d5` |

兩份是原 CSV 的 byte 相同副本；各 Track evidence 另保存原參數、validation、frame／prior／geometry、來源快照及 binary hash。原 heading_rts 最佳、原完整 run 與其 validation 都保持不變。新目錄指標為 `outputs/run/track00_test_2610192315/historical_notes/latest_geometry_submission.txt`。outputs 被 Git 忽略，跨環境須另備份 CSV 與證據。

日常推薦明確加三個選項；程式預設仍為 legacy／fixed／geometry off，沒有修改預設。手動完整命令見 [16_run_geometry_directional.txt](16_run_geometry_directional.txt)，較舊 heading_rts 已評分流程仍可依原文件重現。


## 2026-10-18 原始三相機實驗

新增 [校正三相機與公平對照](23_CALIBRATED_RIG_RESULTS_20261018.md)：使用 raw JPG、固定內外參／高度、逐相機時間補償、稀疏 stereo 尺度錨點及側面幾何取樣。既有 C++ directional 目標不變，沒有新完整 Track 或 Kaggle 分數；本文件的最佳分數與原 CSV 不因此更換。

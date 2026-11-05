# Track2 v5.16.4：回到原配對，單獨拆查表面 fallback

**v5.16.4 已完成 3299 幀，現在可上傳 [CSV](../outputs/run/track02_full_2611050123/result_2.csv) 測試；尚無正式分數。** [完整圖表](../outputs/run/track02_full_2611050123/analysis/README.md)。

v5.16.1 正式 **0.44629**、v5.16.3 **0.44657**，都比 v5.10 **0.42699** 差。兩版停止推薦重複提交。[分數與原 CSV SHA](../outputs/run/track00_test_2611050118/score_receipt.json)。最終目標 **0.30488** 尚未達成。

## 原因假說與本輪變因

兩個退步版共用 v5.16.1 表面 fallback；它在原最近 map 點不可靠時，改用附近法向相容的表面。這增加了配對並降低部分共同模型殘差，但較低殘差沒有轉化為正式精度。可能是新增配對更貼近有重建偏差的結構，或經由高度／尺度與接受分支改變了整段融合。這些都是待驗證假說；僅憑兩個榜分相近無法確定原因。

本輪版本 `track2_v510_turn_patch_v5_16_4_perf2` **只關閉表面 fallback**，從 v5.10 原地圖配對出發，保留已到達 gyro 的因果轉彎與路面最細層視窗支持。目的為拆解分支影響，不是對榜分掃參數。

| 組件 | v5.16.3 | v5.16.4 |
| --- | --- | --- |
| 配準範圍 | 40 m | 40 m |
| 原 map 配對／融合與品質門檻 | 保留 | 保留 |
| `V5161_ENABLED` 表面 fallback | True | **False** |
| `V5162_ENABLED` 因果轉彎 | True | True |
| `V5163_ENABLED` 完整路面視窗 | True | True |

[`parameters.py`](../track2/parameters.py) 可直接調整這些選擇。runner 的來源說明已依表面分支實際值產生，不再把關閉的組件寫成啟用。生成端到端 runner 檢查同時覆蓋轉彎／表面四種組合。

## 驗證計畫與狀態

先完成完整 Track2 CPU／CUDA 機制與輸出契約檢查，再依既有持續授權執行單一完整 3299 幀。800／1600 幀只讀取該次 full 已完成前綴，不額外定位。成功後生成標準圖並檢視，核對來源、frame manifest、輸出 SHA，以及實際影像回饋下的 prior 重播。

共同模型成本只用來檢查幾何與分支差，**不再視為精度改善證明**。需特別觀察後段 2643–2671、2773–2864 的 rank 0 與 map 無更新區間，分開判讀幾何支撐、尺度／高度與 solver 停止原因。

本輪不讀目標 CSV 作定位或擬合，不使用未來影格回修，不變更 map 時間欄位或 GNSS frame 假設。正式分數仍需新 CSV 的實際評分，沒有預填改善結論。

## 證據

- [本輪紀錄](../outputs/run/track00_test_2611050118/test_info.json) 保存修改前 v5.16.3 source 快照、來源 SHA、參數、檢查 log 與正式分數收據；新程式的檢查 SHA 將另外收據連到 full 快照。
- [上一版完整診斷](97_TRACK2_V5163_ROAD_PATCH_20261105.md) 保留原始影像支持、完整圖表與正式退步結果。

## 目前實際進度與額外研究

完整 suite **319 passed、50.50 秒**，1 項既有 Axes3D warning。首輪 318 passed／1 failed 僅為舊版本字串的 metadata assertion，修正後整組通過，兩次 log 均保留。full [track02_full_2611050123](../outputs/run/track02_full_2611050123/test_info.json) 已完成，定位 1578.76 秒、配準 1479.84 秒；215 個來源、6 個輸出及 7 個圖表來源 SHA 核對完成，9 組 full 圖已檢視。800／1600 幀 prior 對各自紀錄的精確重播誤差均為 0，GNSS 預測與 v5.16.3 相同，符合只改 map 配對。800 幀最大 XY 版本差 0.14320 m，這是版本差而非真值誤差。

已排除「取消 map correction mean 衰退」的直接方案：保留原 Q 的 random-walk 模型，在凍結舊 map 觀測的因果重播中，v5.10 的逐筆預測平均 NLL 0.63887→0.69467，v5.16.3 為 0.62842→0.66491，兩者都較差，故未套用。原控制 filter 均值／covariance 重播誤差在 1e-14 量級。[數據](../outputs/run/track00_test_2611050118/bias_memory_replay.json) · [圖](../outputs/run/track00_test_2611050118/analysis/02_bias_memory_replay.png)。這是條件於舊配準的預測診斷，不是新 full 或精度真值。

另發現高度 MAP center 重複使用 prior 的問題，正在 [v5.16.5 暫存研究](99_TRACK2_V5165_DATA_HEIGHT_20261105.md) 獨立驗證；目前 full 來源不受影響。

## 完整結果與判讀限制

全段 **XY prior 與旋轉**和 v5.16.3 完全相同；Z 由各版前幀高度回饋決定，最大差 **0.12678 m**。473 筆下一個 GNSS 預測也相同：XY RMSE 0.0804418 m，這不是 pose 精度。map 更新 **2047→1753**，表面 fallback 配對為 0，符合分支對照。

v5.16.4−v5.16.3 的 XY 差中位數 **0.04487 m**、95 百分位 **0.15283 m**、最大 **0.26854 m（3182）**，yaw 最大差 **0.21787°**。版本差沒有正負精度含義。原始後段 2643–2671、2773–2864 仍都沒有 map 更新；3154–3211 的 map 更新則 10→1，幾何弱區尚未解決。

固定 19 幀、相同 v5.10 raw sample／prior／height／權重，v5.16.4 在共同原模型相對 v5.10 為 **5 降／14 升，平均 +0.00032839**；表面模型為 **6 降／13 升，平均 +0.00032807**。沒有本機整體優勢證據，不預測 Kaggle 改善；新 CSV 用來隔離共用配對分支，正式分數仍待實際測試。

[五版本比較](../outputs/run/track00_test_2611050118/analysis/09_five_way_common_cost.png) · [數據](../outputs/run/track00_test_2611050118/five_way_common_quality.json) · [研究圖表索引](../outputs/run/track00_test_2611050118/analysis/README.md) · [來源與可測檔案收據](../outputs/run/track00_test_2611050118/submission_review.json)。本輪 15 組研究 PNG／PDF，PNG 均已檢視；無新正式分數，**0.30488 未達成**。

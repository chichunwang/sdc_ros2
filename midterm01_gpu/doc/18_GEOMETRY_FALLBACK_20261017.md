# 18｜Track2 第 410 幀警告調查與診斷修正

後續使用者提交原 directional 完整結果，回報 Track1 **1.15275**／Track2 **0.30576**，見 [成績紀錄](17_GEOMETRY_RESULTS_20261017.md)。以下保留當時警告調查與診斷修正的證據；不把新分數歸功於後續 log 修改。

日期：2026-10-17。使用者回報 directional 版本約 500 幀附近出現警告，後提供確切訊息：

```text
cuda_robust_icp scan update failed at frame 410; using current sensor prior.
```

這是地圖更新被拒絕後使用**當幀感測器先驗**，並非程序中止。調查時先確認使用者正在執行的完整測試，待其自然結束後才修改／重建程式；沒有停止該測試，也沒有另外啟動完整 Track。

## 原完整測試的實際結果

使用者手動 run：

```text
outputs/runs/track2_sensor_guided_heading_rts_geometry_directional_cuda_robust_icp_d2d_3299f_7o_atg6_
```

原 `validation.json`、log、result 與 CSV 保持原樣：

| 項目 | 結果 |
| --- | ---: |
| Track／筆數 | Track2／3299 |
| profile／fusion／geometry | sensor_guided／heading_rts／directional |
| 退出碼 | 0 |
| `output_complete_and_finite` | true |
| `validation_errors` | [] |
| 接受地圖更新 | 3047 |
| 退回當幀先驗 | 252（7.64%） |
| ROS 執行時間（不含前處理／建置） | 154.282 秒 |
| CUDA／registration runtime ERROR | log 未出現 |

最早連續退回區間為 **idx 282–417**，之後還有零星及後段區間。所有 idx 從 0 起；第 499／500 幀都有接受更新，實際警告沒有集中在 idx 500。

idx 410 的 source 共 791 點、Euclidean inliers 774 點，但可靠對應只有 10 點，而且都落在 **1 個不同的 target 地圖點**。現有 directional 門檻至少為 10 個可靠對應、6 個不同 target，因此拒絕更新。重複對應到同一個 map 點不構成足夠的空間支持；本次不為了消除警告而放寬門檻。

退回時 `dx=dy=dz=dyaw=0`，`final_cost=prior_cost`。XY／方向繼續使用當幀 GNSS／IMU，Z 沿用既有高度先驗規則；不是讓整台車停在上一幀 pose。這仍可能影響精度，需看實際 Kaggle 分數，不能把完整輸出或正常退回當作精度保證。

## 找到並修正的診斷問題

1. 舊 log 把所有拒絕統一寫成 `scan update failed`，無法區分幾何保護、一般對應不足與真正 runtime 錯誤。
2. 舊程式拒絕候選後，會用返回先驗的 geometry 覆寫候選 geometry；因此舊 `geometry.csv` 不能準確還原候選為何被拒絕。原完整 run 有 251 個拒絕幀在返回先驗處也不滿足支持門檻，但 idx 2851 的返回先驗滿足門檻，不能據此替它捏造精確拒絕原因。

修正已實作於 GPU `sensor_guided.h/.cpp`、`localization.cpp`、`geometry_report.py` 與 benchmark：

- 每次細修重設 metrics，所有返回路徑記錄明確 `update_status`。
- `guidance.csv` 附加 `update_status,candidate_inliers,candidate_reliable_points,candidate_unique_targets`。這些支持數在拒絕前保存；原 `geometry.csv` 繼續描述實際返回的 pose。
- 幾何支持不足顯示 `INFO Map update skipped`，列出候選可靠對應數／不同地圖點數與門檻。其他拒絕仍為 WARN，真正 runtime 錯誤仍為 ERROR 並失敗退出；CUDA 不會靜默改 CPU。
- 結尾區分接受更新、sensor-prior fallback、幾何支持不足與其他拒絕；進度列以 `sensor_prior` 表示返回先驗。
- `validation.json.guidance_not_accuracy` 新增拒絕原因計數；`failed_alignments` 保留歷史欄位，新增 scope 說明。幾何摘要也標明返回 pose 的不足幀數不等於候選拒絕原因數。
- 報表核對接受旗標、候選數量關係、拒絕原因與支持門檻，以及拒絕後沒有殘留 correction。歷史 CSV 缺少新欄位時標示 unavailable，不反推理由或改寫舊結果。

**本次修正診斷，不變更配準目標、法向／切向權重、支持門檻、感測器融合或輸出格式。** 沒有新 Kaggle 分數；已評分的 geometry off 最佳 1.16951／0.31967 仍獨立保存。

## 驗證

GPU workspace Release 重建完成。2 個 CTest 與兩個額外真實 CUDA NN 執行版本通過；15 項 Python 測試通過。新案例包含：20 個 source 全對應同一個可靠地圖點仍須拒絕、沒有可靠表面與沒有重疊的原因不同、重用 metrics 的早退不保留舊狀態，以及候選與返回 pose 支持數不同時，報表仍使用候選拒絕原因。

另以**原完整 run 的相同先驗前綴**重播 idx 0–449，共 450 幀，沒有重新估計截短資料的 heading bias：

- 退出碼 0、完整 450 筆、驗證錯誤為空。
- `result.txt` 與原完整測試的前 450 行逐 byte 相同。
- `geometry.csv` 與原完整測試的表頭＋前 450 行逐 byte 相同。
- 309 幀接受更新；62 幀 `insufficient_reliable_geometry`；79 幀 `insufficient_unique_targets`。這段 141 次退回都已由新候選紀錄確認原因。
- idx 410 確認 `insufficient_unique_targets`，候選 reliable_points=10、unique_targets=1；輸出當幀先驗。

另以正式 benchmark 入口完成 Track2 directional 前 50 幀整合測試，run 為 `track2_sensor_guided_heading_rts_geometry_directional_cuda_robust_icp_d2d_50f_s9nfbcjs`：退出碼 0、空驗證錯誤、50 次接受更新，`validation.json` 已正確寫出新增原因摘要與歷史欄位 scope。

新 log 實例：

```text
Map update skipped at frame 410: insufficient_unique_targets (candidate reliable_points=10/10, unique_targets=1/6); using current sensor prior.
```

此為數值回歸與診斷驗證，沒有宣稱重播後精度提升。450 幀重播不是完整提交檔。原完整 3299 筆 `diagnostic.csv` 仍是該候選可供使用者提交的檔案，不需只因原 WARN 或本次診斷修正就重新跑完整序列。

證據保存在 [本次修正目錄](../outputs/runs/geometry_fallback_fix_gxsz1a7r)，包含修改前快照、建置／測試 log、可重現的 `replay_prefix.py`、[450 幀重播驗證](../outputs/runs/geometry_fallback_fix_gxsz1a7r/replay_track2_450/validation.json)、來源／binary SHA256 及原最佳／完整 run 的保護雜湊。主機 Python 沒有 numpy 的首次測試 log 另留存；完整 Python 測試使用具備專案依賴的 GPU 容器執行。outputs 被 Git 忽略，需另行備份。

# 05｜CPU／CUDA 前 50 幀比較（2026-10-17）

> 歷史比較：以下僅評估原 CPU PCL NDT 與舊 CUDA D2D 的前 50 幀，不含目前 `sensor_guided`／`cuda_robust_icp`。新版完整序列與使用者回報的 Track1 成績見 [06_SENSOR_GUIDED_20261017.md](06_SENSOR_GUIDED_20261017.md)。下文未取得分數的範圍是這組短序列比較。

目前 CUDA D2D 不能視為原 CPU PCL NDT 的等精度替代。兩條 Track 的 CUDA 軌跡都有較大的相鄰 pose 跳動；Track1 的點到地圖殘差略低，Track2 則較高。沒有已確認的 pose 真值或 Kaggle 分數，因此以下是幾何貼合與穩定性診斷，不是正式 ATE，也不能據此保證 CPU 的絕對定位精度。

## 實測結果

| Track | 後端 | 共同地圖殘差 RMS（m） | 最大相鄰平移（m） | 最大相鄰 yaw（度） |
| --- | --- | ---: | ---: | ---: |
| 1 | CPU PCL NDT／Release | 0.625 | 0.796 | 0.740 |
| 1 | CUDA D2D／LM | 0.588 | 2.074 | 7.050 |
| 2 | CPU PCL NDT／Release | 0.475 | 0.262 | 3.464 |
| 2 | CUDA D2D／LM | 0.599 | 2.084 | 17.443 |

CPU／CUDA 同幀 pose 的平移差異 RMS 分別為 Track1 4.803 m、Track2 7.807 m；yaw 差異 RMS 為 5.167／31.720 度。這些數字只表示兩個後端的差異，CPU 沒有被當作真值。

每條 Track 比較相同的起始 50 幀，約 1.8 秒。兩邊均使用原始 raw PLY、尺度 3.0、input voxel 0.3 m；以輸出的 source→map 矩陣套用一次，未另套初始矩陣、外參或事後軌跡對齊。評估使用同一份完整 1 m 下採樣 map，彙整所有處理後 source 點的單向最近鄰距離。CPU Release 輸出與原始 R0 逐 byte 相同。較低點雲殘差不保證車輛 pose 較準，起始短序列也不代表完整 Track。

完整結果與圖表：

- [詳細報告](../outputs/run/track00_test_2610171613/REPORT.md)
- [軌跡與相鄰 pose 跳動](../outputs/run/track00_test_2610171613/trajectory_comparison.png)
- [共同地圖殘差](../outputs/run/track00_test_2610171613/map_residual_comparison.png)
- [點雲 XY 疊圖](../outputs/run/track00_test_2610171613/pointcloud_overlay.png)
- [來源與參數](../outputs/run/track00_test_2610171613/config.json)、[數值結果](../outputs/run/track00_test_2610171613/summary.json)、[驗證紀錄](../outputs/run/track00_test_2610171613/validation.json)

這些輸出遵守原有 Git 忽略規則，只保存在本機；重現工具為 `scripts/accuracy/alignment_evaluator.cpp`、`scripts/accuracy/CMakeLists.txt` 與 `scripts/accuracy/analyze.py`。各 Track 的 `command.json` 保存實際評估命令，`config.json` 保存所選完成 run 的來源與參數；分析使用凍結副本，沒有改動定位程式或重跑原 CPU 任務。

## 保留 CPU 演算法的 GPU 移植方向

目前的 fast_gicp CUDA D2D／LM 是另一套目標函數、鄰域與最佳化實作；差異不能只歸因於 GPU 或浮點精度。即使切換其 P2D 模式，也不會自動變成原 PCL NDT。

可以保留原 PCL NDT 的數學演算法，改成 CPU／GPU 協作：

| 工作 | 保留的行為與執行位置 |
| --- | --- |
| 地圖 voxel 統計、有效 voxel 與鄰域規則 | 先沿用 PCL 建立的資料與搜尋語義，避免同時改動演算法 |
| 每點 score、梯度與 Hessian 的計算和加總 | 以相同公式移植到 CUDA，依精度需求使用 double |
| 6×6 Newton 解算、More–Thuente 線搜尋、停止條件 | 先保留原 CPU 流程，反覆呼叫 GPU 計算核心 |
| 初值、尺度與 source→map 定義 | 沿用原有規則 |

原 PCL 實作可核對其 [computeTransformation／computeDerivatives 原始碼](https://github.com/PointCloudLibrary/pcl/blob/pcl-1.12.1/registration/include/pcl/registration/impl/ndt.hpp)。這需要新增 CUDA kernel、資料配置與傳輸及鄰域查詢，無法單靠執行參數切換。移植時應先在同一固定 pose 比對 score／梯度／Hessian，再比較逐步最佳化及兩條 Track 的相同 50 幀；平行加總順序可能造成浮點差異，驗證應使用明確容差，不能預先承諾逐 bit 相同。

此同演算法 CUDA 後端尚未實作。它也不會直接消除目前 Track1 原始 ASCII 地圖讀取／下採樣約 77 秒的耗時；該部分與 NDT 配準計算需分別評估。速度實測與快取限制見 [03_CUDA_NDT_20261017.md](03_CUDA_NDT_20261017.md)。

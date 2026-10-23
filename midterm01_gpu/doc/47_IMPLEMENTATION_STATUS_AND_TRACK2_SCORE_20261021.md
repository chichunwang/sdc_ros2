**2026-10-23 現行規則提示：**因果定位適用所有 Track；不得以未來資訊回修 pose、使用 RTS／backward smoothing，或在定位後離線修正軌跡。Track1／2 GNSS 仍可作輸入，但 RTS 不因此合規。本文相關算法、命令與成績均為歷史紀錄，不作現行正式提交推薦；規則與程式影響見 [48](48_RULES_AND_CODE_AUDIT_20261023.md)。

# 47｜目前實作盤點與 Track2 最新回報（2026-10-21）

## 最新完整 run 與分數

使用者手動執行的 Track2 `outputs/run/track02_full_2610212125` 通過完整性檢查：3299／3299 幀、有限輸出、完整 frame set、validation returncode 0。逐幀 map accepted 2581、rejected 718；時間連續性檢查標記 1343 幀；形成 11 個慢 map anchor。八狀態 batch graph 有執行並套用，SciPy solver 成功、nfev=5。map 幀累計 955.472 秒，全流程 986.816 秒。CSV SHA256：`b769f2a1d95aeb238cd93d6c0769b94d55514ba443fdd75f3d74bb671a526ec9`。來源：[run validation](../outputs/run/track02_full_2610212125/validation.json)、[參數](../outputs/run/track02_full_2610212125/params.json)。

使用者回報 Kaggle 最新顯示 **0.77**；助教目前隱藏更多小數，因此底層精確值未知，也未核對 Kaggle 上傳 bytes。既有最佳 **0.30488**（約顯示 0.30）是歷史紀錄，仍由 Track2 `track02_full_2610191757` 的 `motion_rts＋directional＋carry` 取得。依目前「分數越低越好」的指標，0.77 沒有超過該最佳。完整運行成功只代表完成與輸出有效，不代表精度提升；最新 run 不取代最佳紀錄。

## 目前實際方法與原最佳的差別

最新 run 使用 `map_scale_slow_cuda_v1`：以本輪 raw bag 產生 `motion_rts` 感測器先驗作配準初始化；逐幀用 map fit 估 6-DOF pose 與尺度，再以可靠幾何及時間條件建立慢 map anchor，最後把 GNSS、raw gyro、NHC 與慢 map bias 放進 batch graph。它不只是 PCA OOM 分批修正，也不是舊最佳換 GPU 後的同一條固定尺度流程。

原最佳則是 `sensor_guided＋motion_rts＋directional＋carry`、固定 `ply_scale=3`：GNSS／IMU 先驗引導 [既有 benchmark](../scripts/accuracy/benchmark.py) 的 CUDA 最近鄰與有界 robust directional ICP，再由既有 solver 做受先驗限制的 4-DOF 細修。它沒有用 Kalman filter 融合 ICP/map 觀測；ICP 是先驗上的 map 修正。原最佳及 run 來源見 [35｜motion_rts 與航向消融](35_MOTION_RTS_AND_GNSS_HEADING_20261019.md)。

## 專案已實作的定位元件

| 階段 | 實作與作用 | 狀態／邊界 |
| --- | --- | --- |
| 感測器與時間 | 從 ROS bag 讀 GNSS／IMU，以整數奈秒時間對影像查詢時刻插值／同步。 | GNSS、IMU、camera/body frame 與完整外參契約仍未嚴格標定。 |
| 感測器 Kalman／RTS | [`sensor_fusion.py`](../scripts/sensor_fusion.py) 的 `position_rts` 是 `[x,y,vx,vy]` 常速度 KF＋RTS，GNSS XY 作位置量測；`bias_rts` 是一維 heading offset random-walk KF＋RTS，區塊 GNSS course 對 IMU quaternion yaw 提供 bias 觀測。 | 兩者都在定位前產生先驗，沒有 ICP/map 量測。`motion_rts` 組合位置與 heading 先驗。 |
| 原始點雲與 map | 導入課程提供的 LoGeR `raw_ply`，執行有限值／範圍篩選、voxel 取樣及 map 局部鄰域處理；由 PCA 估局部法向供幾何配對。 | 專案未自行執行或訓練 LoGeR。 |
| 原最佳配準 | CUDA kNN 加 CPU 小型有界 robust directional ICP；利用 map 法向區分表面方向，受 GNSS／IMU pose prior 約束。 | 原最佳 4-DOF 細修、固定尺度；不是全 6-DOF 狀態圖。 |
| 最新 map fit | 分層 ground／non-ground、多尺度 seed 的 point-to-plane ICP，估計 Sim(3) 七個參數：6-DOF 加 log-scale。接受前做空間支持／holdout、水平可觀測性及相鄰 GNSS 位移檢查。 | 尺度是逐幀候選觀測；gate 通過與 accepted 數量不是精度證據。程式與設計見 [map_scale_icp.py](../scripts/map_scale_icp.py)、[42｜分層尺度與慢圖](42_LAYERED_SCALE_SLOW_MAP_20261020.md)。 |
| 慢 anchor 與 batch graph | [`slow_map_graph.py`](../scripts/slow_map_graph.py) 在 12 秒非重疊視窗內，對可靠 map-minus-GNSS 修正取穩健中位數，形成慢 anchor。八狀態為 `[x,y,vx,vy,yaw,gyro_bias,gnss_bias_x,gnss_bias_y]`；每筆 GNSS 只用一次，加入 25 秒 OU 型 GNSS bias、raw gyro 相對 yaw、軟 NHC、加速度平滑過程因子與有界 map loss，以稀疏 SciPy least-squares 聯合求解。 | 新分支使用 batch graph 與 raw gyro 相對角增量；原 `motion_rts` 使用 IMU quaternion 航向，未積分 raw gyro。沒有加速度積分或完整三軸 IMU preintegration。共享噪聲／時間常數不是嚴格標定 covariance。 |
| GPU／CPU 分工與輸出 | 新入口 [`run_map_smoother_gpu.py`](../scripts/run_map_smoother_gpu.py) 在 CUDA 做 float64 kNN、PCA／法向、配對與 patch 均衡、voxel；[`cuda_scale_geometry.py`](../scripts/cuda_scale_geometry.py) 的 `eigh` 每批最多 256 點。CPU 保留 PLY ASCII／磁碟 I/O、小型 fit solver、稀疏 graph 與少量精確邊界裁決。逐幀輸出計時、diagnostics、完整性 validation、CSV／SHA 與唯一 run；RViz 可回放既有固定尺度結果。 | CUDA 無靜默 CPU fallback。舊 RViz 結果回放不支援這個逐幀尺度新方法。記憶體修正與配對證據見 [46｜PCA 記憶體界限](46_CUDA_PCA_MEMORY_BOUND_20261021.md)。 |

## 已實作但未採為目前最佳的實驗

- 三相機 MapAnything 重建、rig/header 時間整理、逐 camera 運動補償與尺度檢查已實作；曾完整評分但未勝前相機最佳。稀疏 stereo／靜態跨幀幾何仍是診斷或候選，不能稱為完整尺度校準，見 [23](23_CALIBRATED_RIG_RESULTS_20261018.md)、[29](29_THREE_CAMERA_SCORED_ANALYSIS_20261018.md)、[39](39_HEADING_AND_STEREO_SCALE_20261020.md)。
- GICP 有界配準消融已實作；`surfel_joint`、height anchored／surface／surface_tilt 等分支及完整輸出曾測試，但未取代原最佳。`surfel_joint` 完整 run 沒有可靠 map factor，不能把功能接通當成有效修正，見 [36](36_REGISTRATION_AND_FUSION_UPGRADE_20261019.md)、[37](37_HORIZONTAL_SURFEL_JOINT_20261019.md)、[33](33_SURFACE_HEIGHT_REFERENCE_20261019.md)。
- `pose_yaw`、`pose_heading_rts`、`independent_heading_rts` 等航向試驗有實作，但來源／獨立性未核實或完整分數退步，未採用。不得將其描述為目前方法，見 [35](35_MOTION_RTS_AND_GNSS_HEADING_20261019.md)、[40](40_SCORED_SCALE_FAILURE_AND_POSE_YAW_20261020.md)。

## 尚未實作或尚未完成的契約

尚未自行訓練／修改 LoGeR；未完成 camera–IMU–body 與 GNSS origin 的完整外參契約及嚴格噪聲協方差標定；沒有加速度積分或完整三軸 IMU preintegration；目前沒有不依賴 GNSS pose 的 Track3 runner；也未重現學長所述方法的原始程式與逐幀中間結果。Track1 0.6 是目標，並非已達成結果。上述空缺不能由本次 Track2 完整輸出或分數顯示值補推。

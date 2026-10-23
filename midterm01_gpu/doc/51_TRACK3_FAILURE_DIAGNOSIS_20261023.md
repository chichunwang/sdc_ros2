# Track3 異常跳動診斷與後續驗證順序（2026-10-23）

## 本輪證據與界限

使用者截圖回報 Team 3 score **106.72**；截圖是分數來源，上傳 bytes 未獨立核對。只讀診斷對象為 [Track3 full run](../outputs/run/track03_full_2610231431/) 與 [診斷結果](../outputs/run/track03_test_2610231442/diagnosis.json)。本機 run 輸出 3142/3142 幀且有限，21 幀 registration 未收斂；這些完整性與收斂記錄不代表定位準確。診斷記錄來源檔案 SHA 未變。

該 run 使用 `legacy`、`cuda_ndt` D2D、固定 `ply_scale=3.0`，source 為 front，未使用 IMU／GNSS、RTS 或 batch graph。尺度明確標為未校準。該 run 保存的 `AlignNDT` 在 `align` 後檢查 `hasConverged()`、Hessian 有限且非零，以及 pose／fitness 有限，但不檢查相鄰運動是否合理（[執行時原始碼快照](../outputs/run/track03_full_2610231431/localization_sources/src/localization.cpp)）。這些數值條件成立，仍可能接受錯誤配準。

診斷中，idx 22→23 僅隔 0.036017315 s，平移 21.449 m、yaw 變化 13.561°，但 `converged=true`；idx 388→389 隔 0.035551498 s，平移 15.330 m、yaw 變化 −170.242°、3D 旋轉 174.591°，同樣 `converged=true`。後一對最近過去 IMU 3D 旋轉為 0.012029°。全序列相鄰平移大於 1 m 有 463 次、大於 5 m 有 61 次，yaw 變化絕對值大於 30° 有 91 次。這些是診斷統計，不是已驗證的生產 gate 或真值比較。

IMU 共 11307 筆，bag 時間間隔中位數約 9.87 ms，gyro、orientation、acceleration 數值皆有限；三種 covariance 欄位每筆皆全零。依 ROS 2 `sensor_msgs/Imu` 定義，全零表示 covariance 未提供，不能當作零誤差（[官方訊息定義](https://raw.githubusercontent.com/ros2/common_interfaces/humble/sensor_msgs/msg/Imu.msg)）。診斷採最近過去 bag timestamp 配對；時間對齊與 IMU／相機外參未校準，所以 IMU 只能作待驗證的運動觀測，不能視為真值。

初始矩陣方向、map/frame 關係、scale 3 的適用性，以及官方 sample 的 idx／yaw 單位與座標系仍待確認。它們是待查假說，現有證據不能斷言唯一根因，也不能據此提出新的尺度值。

## 建議的因果驗證順序

1. 先核對初始 source 到 map 的矩陣方向、座標 frame 與時間關係，並確認 submission sample、idx 與 yaw 契約。未確認前，輸出格式也不能視為已認證符合提交契約。
2. 建立前向因果預測：以 gyro 增量與歷史速度形成當前預測；map 配準若對預測產生不合理創新，或局部幾何退化，拒絕該次 map 更新。拒絕時仍輸出當前預測 pose，不凍結在舊 pose。這是設計建議，尚未實作。
3. 再研究逐幀尺度與 robust point-to-plane 配準：分開處理地面／立面，平衡空間及法向支持，檢查尺度、XY、yaw 的可觀測性，並用空間 held-out 評估。所有濾波只向前更新，不回修舊 pose。GPU 可評估重用既有 kNN／PCA 計算；不直接接入舊 RTS 或 graph 入口。
4. 驗證先從首 100 幀檢查初始化，再從 idx 0 前向跑至少 400 與 800 幀前綴，涵蓋既有異常點 389 與 751。不可任意從中段開始並用異常 pose 當初值。比較基線、運動預測、拒絕 map 更新及加入尺度估計的版本，檢查相同前綴在不同截斷長度下是否一致（prefix consistency）。拒絕率、殘差及軌跡平順度都不單獨構成精度證明。

本文件只記錄診斷與建議：未實作 solver，未執行定位或 GPU 工作，未提交 Kaggle，亦未修改 CSV。後續完整定位仍由使用者手動啟動。

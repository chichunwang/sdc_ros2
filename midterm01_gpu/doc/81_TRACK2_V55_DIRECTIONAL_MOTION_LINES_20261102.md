# Track2 v5.5：方向性運動約束與線形幾何

## 目前狀態與使用方式

v5.5 版本字串為 `track2_causal_directional_motion_line_features_v5_5_perf2`。本輪以凍結的 v5.4 為基礎開發；**175 項生成／CUDA regression 通過（22.34 秒）**，三組機制 PNG/PDF 已產生。實際 live 套用與來源 SHA 以[套用收據](/home/ricky/sdc_ros2/midterm01_gpu/outputs/run/track00_test_2611021701/apply_receipt.json)為準；尚無 v5.5 真實定位或新分數。

正式 full 仍由使用者在 Ubuntu 主機手動啟動：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
./scripts/run_track2.sh --full
```

本文件不會啟動定位。v5.5 的模型與限制見下文；實際 live 套用狀態請依本 run 的最終收據判讀。

## 觀察與假說

v5.4 的 Track2 full run `track02_full_2611021544` 已完成 3299 幀；使用者回報 Public Score 為 `0.59002`。記錄見 [v5.4 full run](</home/ricky/sdc_ros2/midterm01_gpu/outputs/run/track02_full_2611021544/test_info.json>)。目標分數 `0.30488` 的 CSV 僅作為 hash／來源 provenance 記錄，沒有讀取其 pose 值，也沒有拿它擬合參數：[目標 CSV SHA／來源紀錄](/home/ricky/sdc_ros2/midterm01_gpu/outputs/run/track00_test_2611021701/baseline_inventory.json)。分數是使用者回報，不是本地獨立真值。

既有前綴診斷曾記錄：前 826 幀中，11.08 秒輸出 yaw 改變 `1.91°`，側向跳約 `10.5 cm` 主要來自 rebase；另有 `19.54 s` 的 sensor prior＋map 更新合計 `8.2 cm`。縱向 411/423 比 prior 弱。這些是特定前綴診斷，不是 ground truth，也不能拆解或解釋 Public Score。

本輪假說是：若同一影格可連續累加多種 heading 校正，方向小偏差可能被多次改動；在車輛正常行駛、側滑不大時，側向速度可作為帶不確定度的弱約束；地圖中的線狀幾何可以提供垂直於線方向的位置／姿態資訊，而沿線方向本身不應被當成可觀測約束。這些機制用來限制不合理的改動與補充幾何資訊，不代表已知靜態語義，也不保證定位或分數改善。

## v5.5 改動

### 1. 以逐步 likelihood 更新消耗 heading 校正預算

`CausalSensorPrior` 的 `heading_update` 對每筆新 GNSS-pair／course（由運動方向估計的航向）量測只建立一次 factor，保存 `id`、`H`、`observation`、`R` 與 `remaining_fraction=1`。首 pose 初始化沒有先前發布的 pose，因此不受後續每幀 `0.5°` bias-correction budget 限制。

第一個 pose 發布後，每個 frame 先消耗先前尚有剩餘 fraction 的 heading factors，再接收新的 pair/course 量測。該 frame 內所有 factor 共用 `0.5°` 校正預算；每一步選擇 `f <= remaining_fraction`，使實際 IMU bias 步幅不超出當前尚餘預算。該步使用的有效觀測方差為：

```text
R_effective = (R_original + H Q_bias H^T * factor_age) / f
```

`f=0` 時不更新均值與 covariance，保留剩餘資訊；`f>0` 時使用完整 Kalman gain `K` 一起更新 IMU／GNSS 兩個 heading bias，並以該更新執行 Joseph covariance update。未消耗的 fraction 保留為 pending；每筆 factor 的已消耗 fraction 總和不得超過 `1`。factor 超過既有 `8 s` 時窗後，丟棄尚未使用的 fraction 並記錄。重複持有的 GNSS heading/course 不建立新 factor，也不回寫任何過去 pose。

這個方法逐步吸收新量測的 likelihood information，不再把超出單幀步幅預算的量測 residual 直接丟棄。`0.5°` 是沿用 map yaw model floor 量級的工程預算，不是由真值估得的最佳值。對靜態線性系統且 process noise 為零，累計消耗 `fraction=1` 會精確等同原本的一次 Gaussian posterior 更新。存在 process noise 或量測 staleness 時，則是當幀慢 bias 的近似，資訊投入量不超原量測且額外膨脹 variance，作工程保守處理；沒有證明完整動態／跨狀態 covariance 一致性，也不是完整的 delayed-state 或 joint EKF。

`HEADING_MAX_BIAS_RATE_DEGPS` 保留供 legacy 設定使用，但不再作為 v5.5 跨 course-window 的 rate；v5.5 由每幀 fraction budget 取代它，避免硬限幅遺失量測資訊並造成過慢暖機。

### 2. 新 GNSS 時刻加入軟側向速度觀測

相機時刻有新 GNSS 資料時，才加入一次側向速度 pseudo-observation（虛擬觀測）；同一 GNSS 輸出不會在後續每個 camera frame 重複當成新觀測。正常行駛時車體側向速度通常較小，因此把它當成 soft constraint，而非硬性零速度條件。

基底標準差為 `0.3 m/s`，是以 GNSS `0.3 m`、`1 s` 解析度換算的工程量級。方差另加入 yaw 與未知 lever arm（感測器至車體參考點的力臂）不確定度：

```text
sigma_v_left^2 = (0.3 m/s)^2
                 + speed^2 * yaw_variance
                 + angular_rate^2 * unknown_lever_variance
```

此處沿用現有未知外參／lever arm 的均值與標準差，不代表已完成標定。更新保留 `NISsoft`／`NIShard` 品質閘門與 Joseph covariance；不更新 heading 兩個 bias，也不是完整的 joint GNSS/map/height EKF。側向近零是帶假設與不確定度的運動近似，不是車輛必然無側滑的事實。

### 3. 依前、側、yaw 分別限制實際 map gain

`CorrectionFilter` 對 map-correction proposal 在當幀 sensor forward、left、yaw 的 actual gain 分別限幅，並以縮放後的 actual `K` 更新 Joseph covariance。縱向沿用原有的 `0.5 m/s`、前進距離比例 `0.1` 與 turn-lever 項；側向新增 `0.2 m/s` 與 turn-lever 項。側向預算不會因為車輛縱向速度較快而放寬。

若進入 partial mode，重新投影後會再次檢查更新 budget。這是對實際改動量的限制，不代表 sensor proposal 本身已被證明準確。

### 4. 在 plane 外加入互斥 line class

CUDA `PlaneGeometry` 在既有 plane class 之外新增獨立 line class。線候選需符合：PCA 第二／最大 eigenvalue ratio `<= 0.10`、至少 6 個鄰點、達到最小 extent，且方向在垂直方向 `15°` 內。source 與 map 都必須被分類為 line，兩者方向差也須 `<= 15°`；source 點先依 ground scale 換算成公尺再分類。

線特徵不提供沿線方向的殘差。線方向為 `d` 時，line projector 為 `P = I - d d^T`；plane 法向為 `n` 時，plane projector 為 `n n^T`。兩類都由 CUDA robust H/g 累加。同一個點只能屬於 plane 或 line 其中一類，不能重複計入。混合殘差沿用既有 scale Schur、likelihood、actual proposal validation 與 height-once 流程。

這是由 PCA 形狀與方向建立的幾何分類，不是語義辨識；不宣稱確認 pole、路牌等物件為靜態，也不宣稱已完整涵蓋所有轉角 feature。

## 範圍與限制

- decoder、camera、ROS、runner 與 Track1／Track3／config 相對本輪凍結 source 維持 SHA 不變。
- 本輪沒有下載資料、讀取 raw／bag、跑真實 frame 或啟動 full。
- GNSS 側向速度是 soft pseudo-observation；line class 只使用幾何方向資訊。v5.5 不等於完整的 joint GNSS、map、height 濾波器。
- 尚無 v5.5 真實 Track 定位或 Public Score，不能宣稱達到 `0.30488`，也不預測分數改善。

## 驗證紀錄

第一輪整組檢查為 164 passed、2 failed。其後的測試修正包含：舊 plane-only fixture 中有符合新 line 條件的 pole 點，故在 plane-null 測試隔離時關閉 line class，並另設 line 測試；舊 slow-heading 測試以硬限幅版的立即改動速度為前提，已改成檢查逐步 fraction 消耗與 heading posterior 行為。新測試涵蓋 factor fraction 總和不超過 `1`、靜態單次 Gaussian posterior 等價、prefix 不變，以及有限 gain／covariance；沒有移除精度或因果檢查。

之後發現 Python 3.10 不接受 `np.r_[ground, *poles]` 寫法，已改為 `np.concatenate`。局部檢查 33 passed（5.82 秒），硬限增益版整組 174 passed（22.41 秒）；進一步加入單次 likelihood 分數消耗後，175 passed（22.12 秒）。修正混合殘差的診斷命名後，最終整組 **175 passed（22.34 秒）**。先前 log 全部保留，未用中間結果代替最終結果。

本輪 [分析與三組圖表](/home/ricky/sdc_ros2/midterm01_gpu/outputs/run/track00_test_2611021701/analysis/README.md)包括以下生成證據：

| 生成案例 | v5.4 | v5.5 | 解讀 |
| --- | ---: | ---: | --- |
| 一筆 3° heading 量測的最大單幀校正 | 2.9703° | 0.5° | 完整吸收後均值 2.9703°、std 0.9950° 相同；精確等價案例為零 process noise |
| 含 0.3 m GNSS 噪聲直行，全段 lateral position RMSE | 0.3013 m | 0.1851 m | 車輛模型生成案例，不是真實 Track2 誤差 |
| 15 m/s 下交替 map noise 的最大 lateral correction step | 0.0647 m | 0.0100 m | 0.05 s 更新間隔；實際 gain／Joseph 同步縮放 |
| 多深度垂直線場景的 XY 誤差 | 0.3081 m | 0.0345 m | v5.5 得到 36 個有效 line matches；關閉 line 的 ablation 仍為 0.3081 m |

上述數字不能推算 Public Score。實際 PLY 是否包含足夠可靠線結構、動態物體及深度形變的影響，仍需使用者後續 full 結果檢查。舊 `median_plane_residual_m` 為相容欄位；混合幾何啟用時數值代表 plane／line residual norm，並另記 `median_geometric_residual_m`。

## 參考

- [v5.4 場景證據](80_TRACK2_V54_SCENE_EVIDENCE_20261102.md)
- [Track2 操作入口](../track2/README.md)
- [本輪 v5.5 candidate 與 run 證據](</home/ricky/sdc_ros2/midterm01_gpu/outputs/run/track00_test_2611021701/>)

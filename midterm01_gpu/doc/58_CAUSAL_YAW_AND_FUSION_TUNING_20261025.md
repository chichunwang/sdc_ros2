# 因果 yaw 濾波與融合信任度調整

本文件說明 Track3 v6 新增的 yaw robust weighting、可手動調整的 gyro／地圖配準信任度，以及相機高度對 scale 的作用。這些參數仍是工程假設，不是量測校正，也不保證提高分數。要比較參數時，一次改一項，保留每個 run 的來源與參數快照。

目前程式版本：`track3_cuda_v6_causal_yaw_robust_fusion`。

## 手動調整位置

直接編輯 [track3/imu_parameters.py](../track3/imu_parameters.py) 中的 `GYRO_NOISE_DENSITY_RAD_S_SQRT_HZ`、`GYRO_BIAS_RANDOM_WALK_RAD_S_SQRT_S`、`MAP_POSITION_STD_M`、`MAP_ROTATION_STD_DEG`、`YAW_ROBUST_ENABLED` 與 `YAW_HUBER_SIGMA`。這是含 IMU run 使用的 Python 參數入口；runner 會將有效值寫入每次 run 的 `parameters.json`，程式快照與 SHA 保存在同一 run。

| Python 常數 | 預設 | 調大／調小的方向 |
| --- | ---: | --- |
| `GYRO_NOISE_DENSITY_RAD_S_SQRT_HZ` | `0.02` `(rad/s)/√Hz` | 調小會降低 gyro prediction 的過程不確定性，通常使濾波器更相信 gyro；調大會讓地圖量測更容易修正 prediction。它不會改變 gyro 讀值，也會影響後續 covariance 與 NIS gate。 |
| `GYRO_BIAS_RANDOM_WALK_RAD_S_SQRT_S` | `0.001` `(rad/s)/√s` | 調大代表容許 gyro bias 隨時間變動得更快，可能較快吸收 drift，也可能讓 bias 更不穩定。它不同於 gyro 白雜訊，不能只當作整體 gyro 信任度旋鈕。 |
| `MAP_POSITION_STD_M` | `(0.5, 0.5, 0.5)` m | 每項是 map X/Y/Z 的標準差；調小某方向就更相信該方向的 registration。 |
| `MAP_ROTATION_STD_DEG` | `(5.0, 5.0, 5.0)` deg | 每項是 map 軸小角度誤差向量的標準差，不是 Euler roll/pitch/yaw，也不是角速度。車體接近水平時第 3 項近似 yaw 標準差；有傾斜時實際 yaw uncertainty 由 Jacobian 投影整個 3×3 姿態 covariance。 |
| `YAW_ROBUST_ENABLED` | `True` | 設為 `False` 可做舊／新機制的對照；其他參數不變。 |
| `YAW_HUBER_SIGMA` | `3.0` | 這是無單位的標準化 innovation 門檻，不是度數或最大 yaw rate。調小會較早降低不一致 map yaw 的權重；調大則容許較大的差異維持原權重。 |

例如，要單獨增加 map yaw 量測信任度，可把 `MAP_ROTATION_STD_DEG = (5.0, 5.0, 5.0)` 改成 `(5.0, 5.0, 2.0)`。小角度量測資訊約與 `1/σ²` 成正比，所以 map Z 軸角度資訊會增加 `(5/2)² = 6.25` 倍；這不是 Kalman 最後 yaw 修正量的固定倍數，因為結果還取決於 ESKF prediction covariance、cross-covariance 與 NIS gate。這個數值只是展示怎麼改，不是精度建議。

要看有效參數，檢查 run 的 `parameters.json` 內 `imu.fusion_parameter_values`。不要同時修改 JSON 和 Python 參數並假設兩者都會生效：IMU fusion noise 與 registration 標準差由這個 Python 入口解析。

## 新增的 yaw robust weighting

每一幀先有 IMU／視覺 ESKF prediction，再有本幀 camera-to-map registration candidate。演算法比較兩者目前的 map yaw：

\[
\delta_\psi=\operatorname{wrap}(\psi_{map}-\psi_{pred}),\qquad
u=\frac{|\delta_\psi|}{\sqrt{S_\psi}},
\]

\(S_\psi\) 是 prediction 與 map orientation measurement 投影到 heading 後的合成 innovation variance。小角度 map-axis rotation error 對 heading 的 projection 使用 `atan2(R[1,0], R[0,0])` 的 Jacobian，因此能處理相機姿態的 roll/pitch；forward 軸接近垂直時 yaw 不穩定，程式不執行降權。

若 `u ≤ YAW_HUBER_SIGMA`，map covariance 不變。超過門檻時，Huber 權重為

\[
w=\min\left(1,\frac{\texttt{YAW\_HUBER\_SIGMA}}{u}\right).
\]

程式沿著 map yaw 方向以 rank-one 正半定項增加量測 orientation covariance，使其 yaw 投影 variance 變成原來的 `1/w` 倍。姿態其他方向保留原 covariance。接著，轉換到 IMU 原點的 lever-arm covariance、6D NIS 檢查，以及原本 Joseph-form ESKF 更新照常執行。這是一筆 registration measurement 的一次更新；沒有再加一筆重複 yaw measurement。

初始 map bootstrap 不降權，讓第一筆幾何錨點仍能建立狀態。被拒絕的 registration 不進濾波更新；stationary hold 保留原錨點也不做 yaw measurement update。正常大角度轉彎只要 gyro prediction 和 map candidate 一致，innovation 就小，因而不會因為絕對 heading 大而遭降權。

這個 Huber covariance inflation 是簡化的 robust ESKF 作法。相關方法可參考 [robust error-state Kalman filtering 論文](https://link.springer.com/article/10.1186/s13634-024-01172-w)；目前程式沒有宣稱重現該論文的完整演算法。它只處理當前 prediction 和 map yaw 間的大型不一致，不會自行校正固定 gyro bias、外參或整體 heading offset。若 gyro 有系統性偏差，它也可能錯誤降低合理 map yaw 的權重。小於門檻的持續偏差不會被這項機制修正。

每幀診斷位於 `pose_trace.jsonl` 的 `inertial.update.yaw_constraint`，也包含於 observations 的 inertial update。可查看 `innovation_deg`、`normalized_innovation`、`weight`、`effective_map_std_deg`、`reason` 和 `measurement_used`。`weight=1` 表示沒有降權；`reason=initial_bootstrap_not_downweighted`、`stationary_pose_hold` 或 `registration_not_accepted` 表示未套用 yaw 降權。Bootstrap 仍會建立地圖錨點；其他情況是否使用量測，請看 `measurement_used`。

## 固定相機高度與尺度

若已知的是**相機光心離當地路面法向距離** \(H\)，而本幀點雲可可靠估計相機到同一地面的重建距離 \(d_k\)，則可用

\[
s_k\approx H/d_k
\]

提供該幀 raw PLY units 到公尺的尺度觀測。固定且準確的 (H) 能增加尺度資訊，也可能降低 scale 與 translation 互相吸收的程度。假設高度 (H) 有相對誤差時，尺度也會帶入約相同量級的相對誤差；高度錯誤會讓所有尺度 estimate 帶偏。

量測時要估 local ground plane，並以相機高度沿地面法向的距離計算。相機 pitch、道路坡度、起伏、地面錯分及遮擋都會影響 `d_k`。固定相機安裝高度不表示固定的 map Z，也不表示在所有幀都能從 raw 點雲準確找出地面。

水平地面本身不能觀測 yaw：繞地面法向轉動不改變地面平面。因此已知高度可以幫助 metric scale，不能單獨解決 heading。本次沒有把外部程式的 `H=1.573 m` 寫入 v6；外部程式註解稱它來自 Track1 GNSS 管線校正，並非本車 Track3 的相機高度實測。若日後用高度先驗，應先取得本車相機光心到路面的校正值，再用因果、逐幀的方式融合。

## 競賽規範與操作

10/22 更新規則明確允許 Kalman／Extended Kalman Filter 與只用當前及歷史資料的 causal filter；禁止未來 frame 回修舊 pose、backward smoothing 及使用完整序列做非因果修正。官方文字見[規則更新 PDF 第 1 頁](<../../midterm01/docs/Competition Regulations/01_Mid01_Updates_Rules.pdf>)。這項 yaw weight 只在當前迭代狀態與當前 registration 之間調整 measurement covariance；不讀未來 frame，不回寫已輸出的 pose。

程式已提供 v6 設定與可供手動修改的參數入口。使用者完成參數修改後，可以親自啟動 full：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
./scripts/run.sh --full
```

`run.sh` 會建立新的 immutable source snapshot，並在 run 中保留有效參數、SHA 與 pose diagnostics。若 container 尚未啟動，先於 Ubuntu 主機執行 `./scripts/dev.sh up`。這是執行指令說明；本文件不表示已有 v6 full 成績或精度改善。

合成機制與生成資料 CUDA suite 已完成，**136 項通過**，耗時 **46.26 秒**；另有 1 個既有 Matplotlib `Axes3D` warning。結果見 [pytest.xml](../outputs/run/track00_test_2610251412/pytest.xml)、[yaw 機制圖](../outputs/run/track00_test_2610251412/analysis/01_synthetic_yaw_mechanism.png) 與[分析說明](../outputs/run/track00_test_2610251412/analysis/README.md)。

這些是合成機制及 CUDA 檢查，沒有執行真實 Track3 定位，也沒有產生新的 full run 或成績；因此不能據此宣稱真實 heading 或 Kaggle 分數有所改善。

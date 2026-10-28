# v8.1 靜止時估計陀螺儀零偏

目前 v8.1 加入一個範圍有限的 Kalman filter：車輛已由影像、陀螺儀和點雲檢查確認靜止時，估計 gyro 的固定零偏，讓之後的角速度積分少累積一點系統性轉向誤差。它不替換 Monte Carlo，不平滑輸出的 pose，也不使用 accelerometer 積分位置。

這是有物理觀測根據的修正，尚未證明能改善 Track3 精度或分數。它在序列前段尚未確認靜止前不會學到 bias，因此不能期待這項 bias 校正改善首次停車之前的成績；首幀選擇流程保持原樣。完整定位由你手動啟動：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
./scripts/run.sh --full
```

## 為什麼先估 gyro bias

v8.1 讀取 gyro 角速度來預測旋轉，但原先把 `initial_gyro_bias_radps` 固定為零。若感測器有小的固定偏移，直接積分會讓旋轉誤差隨時間累積。已經確認靜止時，車體角速度接近零，所以 gyro 讀值可以用來估計它的 bias。

直接把所有 gyro 讀值低通濾波會延遲真實轉彎；把地圖配準結果再餵進另一個 pose filter，也需要可信的量測 covariance，且容易和已有的配準／選擇機制重複計入同一份地圖資訊。因此這次只在靜止區間估 bias；移動時的角速度仍即時進入原本的旋轉積分。Kalman filter 的狀態只代表三軸 gyro bias，不代表完整車體 pose。

這種靜止條件很重要。IMU 單獨看到低角速度不足以證明車輛停住；本程式沿用既有的影像、raw gyro、近距離 3D 支持與停車判定。OpenVINS 的零速度更新說明也指出，靜止偵測是濾波能否安全使用的關鍵，影像與 IMU 證據可以互相補足（[Zero Velocity Update](https://docs.openvins.com/update-zerovelocity.html)）。

## 每一幀怎麼更新

令 $b$ 為 gyro bias，$P$ 為其每軸變異數。兩者都在 IMU 座標系中表示。經過時間 $\Delta t$ 後，bias 使用隨機漫步模型：

$$
 b^- = b, \qquad P^- = P + q_b^2\Delta t,
$$

其中 $q_b$ 是設定的 bias random-walk noise density。這表示短時間內預期 bias 不會突然跳動，但時間拉長時允許它慢慢改變。

只有在上一節的靜止判斷成立，而且收到本幀新的、依時間排列的 IMU 樣本時，才從該區間的 raw gyro 形成加權平均 $z$。樣本依它覆蓋的時間長度加權；若影像邊界只用到 IMU 樣本間隔的一小部分，量測變異數會採保守上界：

$$
 R = \frac{\sigma_g^2}{T}, \qquad r=z-b^-, \qquad S=P^-+R,
$$

其中 $\sigma_g$ 是 gyro white-noise density，$T$ 是這次真正使用的新樣本支援秒數，$r$ 是量測與目前 bias 的差。Kalman gain 和 bias 更新為：

$$
 K=P^-/S, \qquad b=b^-+Kr.
$$

每軸各自使用一維更新；更新後的 covariance 用 Joseph 形式保持數值穩定：

$$
 P=(1-K)^2P^-+K^2R.
$$

更新前會先做 raw-rate 門檻與三自由度 normalized innovation squared（NIS）檢查。raw gyro 門檻沿用 `motion_parameters.py` 的 0.02 rad/s；NIS 機率沿用 IMU 設定中的 0.999。若檢查未過，該次觀測不改 bias。CUDA 上計算加權均值、innovation、NIS、gain 與 covariance；時間戳和輸出序列由 CPU 控制。

第一幀沒有前一段時間可以作為靜止量測，因此不更新 bias。沒有新 IMU 樣本時不重用先前的 sample。移動時不從 raw gyro 估 bias；filter 只延續已知 bias 並增加 covariance。Monte Carlo 旋轉候選使用傳播後的 bias covariance 決定探索寬度，因此首次停車前的後續候選也可能略有不同，不能宣稱整段輸出 bytes 不變。Monte Carlo 候選的噪音仍只探索候選，不改寫 prediction；配準成功時只提交既有 pose／scale，map 量測不回灌 bias filter。

## 怎麼看本輪證據

這輪唯讀重播了 v8.1 歷史 run 的 3142 幀 raw IMU，並沿用當時已記錄的因果 stationary 標記；重播沒有執行定位，也沒有改舊 pose。共使用 11,305 個 IMU 樣本，第一筆 bias 更新落在 idx 929，更新涵蓋 1,242 個標為靜止的幀。該次設定下最後估計為：

| 軸 | bias estimate (rad/s) | filter model std (rad/s) |
| --- | ---: | ---: |
| IMU X | -0.00795 | 0.00626 |
| IMU Y | -0.00330 | 0.00626 |
| IMU Z | -0.00489 | 0.00626 |

這些是未校正模型輸出，不是 bias 真值或定位誤差。歷史 bag 每筆 gyro covariance 都為零，代表 covariance 未知；程式使用 `imu_parameters.py` 的 noise density。stationary 標記也不是獨立的車體真值，所以重播只能確認現有資料和新 filter 的時間／數值機制可以連接，不能證明真實 bias 已估準。

合成已知偏移資料顯示，靜止時 Kalman update 會朝注入 bias 收斂；後續轉彎使用當下的 gyro 讀值減去已學到的 bias。90 幀合成案例的末端 SO(3) 旋轉誤差由 0.029916 rad 降至 0.007656 rad，約減少 74.4%；停車前已累積的角度誤差仍保留，沒有回頭改 pose。這只是合成演算法檢查。最終完整生成資料／CUDA suite 為 **192 passed／257.59 s**，另有一項既有 Matplotlib Axes3D warning，紀錄見 [pytest_attempt03.xml](../outputs/run/track00_test_2610261414/pytest_attempt03.xml)。首輪一項測試的預期忽略原有 zero-order hold（樣本到達前維持上一角速度）首個 10 ms；只修正預期後，針對性 46 項及最終 192 項全部通過。唯讀重播數值見 [imu_replay.json](../outputs/run/track00_test_2610261414/imu_replay.json)，首幀比較見 [startup_evidence.json](../outputs/run/track00_test_2610261414/startup_evidence.json)。這些診斷不會改寫 v7 或 v8.1 歷史 run。

v7 full 與 v8.1 full 的首幀都選中候選 2。唯讀比較中，兩個選中位置相差 0.03539 m；v7 common geometric cost 為 0.04692，v8.1 為 0.04714。v8.1 保留 v7 的四候選逐一完整 NDT／ICP refinement 與通過 gate 後共同評分流程，GPU 求解後的浮點差異稍改候選數值。近似的起點只表示首幀方法仍保留，不代表其中一個比較接近真值。

你回報改進版分數 1.17、v7 分數 1.05，但歷史 run metadata 沒有記錄這兩個分數，現在無法確認各自對應哪個 run；也沒有獨立 ground truth。本次沒有依分數調整 noise。加入 bias filter 前 v8.1 full 的既有輸出是 3142 幀、1,023 次 map 接受，最後一次接受在 idx 2221，因此這次 bias filter 不能當作後段退化已修好的證據。

## 開關與限制

實作開關在 `track3/imu_parameters.py` 的 `GYRO_BIAS_KALMAN_ENABLED`，目前為 `True`。設為 `False` 可關閉 stationary gyro-bias Kalman，回到沒有 bias 更新的 v8.1 行為。IMU→PLY 外參 `(roll,pitch,yaw)=(1,6,0)` 度仍是未標定假設；noise density、bias random walk 與初始 covariance 也不是 bag 真值。

這個修改主要針對停車後 gyro 積分可能累積的零偏，不修正 scale、map coverage 或長串配準拒絕，首次確認靜止前也沒有新 bias 量測可以帶來校正。下一步應由你手動跑一輪 full，再檢查靜止後轉彎的角速度、bias trace、接受／拒絕段落和對應成績；要將分數記為實驗結果，還需保留成績來源和確切 run ID。程式與設定快照、測試日誌及唯讀證據保存在本次 [test run](../outputs/run/track00_test_2610261414/test_info.json)。

## 圖表與可追溯紀錄

- [圖表說明](../outputs/run/track00_test_2610261414/analysis/README.md)：首幀共同 cost／選中位置、歷史 raw gyro／bias／模型 std、合成停車轉彎比較，均另存 PNG／PDF。
- [數值及限制](../outputs/run/track00_test_2610261414/analysis/metrics.json)、[圖表來源 SHA](../outputs/run/track00_test_2610261414/analysis/manifest.json)。
- [歷史輸出 SHA 核對](../outputs/run/track00_test_2610261414/historical_protection_after.json)：保留原 run、CSV、pose、manifest 與來源。
- 時間與噪聲離散化參考 [OpenVINS Discrete Propagation](https://docs.openvins.com/propagation_discrete.html)；本程式只實作本文推導的三軸 bias 子模型，沒有移植完整 VIO／ESKF。

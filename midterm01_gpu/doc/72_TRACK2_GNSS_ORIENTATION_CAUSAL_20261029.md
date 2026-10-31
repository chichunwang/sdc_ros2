# Track2 v5.2：GNSS 航向的因果融合

## 先說現在的問題

Track2 先前讀取 GNSS 位置，但沒有使用 `geometry_msgs/PoseStamped` 裡已存在的 orientation quaternion。資料稽核確認 GNSS 訊息帶有有效的 quaternion；不過資料存在不代表其座標定義或方向誤差已標定。live v5.2 讓 GNSS 航向與 IMU 航向各自保留偏差與不確定度，再用兩種不同觀測逐步估計。

使用者回報的三個分數如下。使用者已確認三者皆為同一 Track2 競賽的 Kaggle Public Score，可比較該評分區段的表現；它們不提供逐幀真值：

| 來源 | 回報值 | 解讀限制 |
| --- | ---: | --- |
| v5 full | 0.53683 | 使用者回報；不是獨立定位真值 |
| v5.1 full `track02_full_2610291400` | 0.51617 | 使用者回報；v5.1 同時包含地圖配準修正 |
| `result_2_submitnoncasual.csv` | 0.30488 | 使用者回報的非因果輸出，只作唯讀對照 |

Public Score 由 v5 的 0.53683 降至 v5.1 的 0.51617，下降 0.02066（約 3.85%）；距離 0.30488 尚差 0.21129。[分數確認與比較紀錄](../outputs/run/track00_test_2610292245/analysis/README.md)。

非因果 CSV 不會拿來設定 bias、offset、時間差或調整參數；它不是 ground truth，也不能作為合規輸出。v5.2 尚未跑真實定位，不能保證會得到 0.30488 或任何特定分數。

## 版本狀態與操作

v5.2 已套用至 live Track2。它保留 v5.1 地圖配準與 correction-memory，不改地圖演算法；新改動集中在 GNSS orientation 的解析、兩種航向觀測及其 covariance。完整定位仍由使用者在 Ubuntu 主機手動啟動：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
./scripts/run_track2.sh --full
```

直接編輯 [`track2/parameters.py`](../track2/parameters.py)。v5.2 的測試與 source SHA 核對均已完成，細節見下方「驗證狀態」。

## 兩種航向觀測各自回答什麼

直覺上，IMU 和 GNSS 航向都可能帶有偏差。若只看兩者差值，就只能知道它們彼此差多少，無法知道兩者一起相對地圖偏了多少。因此 v5.2 分開表示兩個 bias：

\[
b = \begin{bmatrix} b_{\mathrm{IMU}} \\ b_{\mathrm{GNSS}} \end{bmatrix}, \qquad
P_b = \operatorname{Cov}(b)
\]

採用的符號約定是 `地圖 yaw ≈ IMU yaw + b_IMU ≈ GNSS yaw + b_GNSS`。每個 GNSS 影格可提供兩種不同的資訊：

1. **同時刻的 GNSS–IMU 航向差**：配對 GNSS quaternion yaw 與該時間的 IMU yaw，得到
   \[
   z_{\mathrm{pair}}=\operatorname{wrap}(\psi_{\mathrm{GNSS}}-\psi_{\mathrm{IMU}})
   = [1,-1]b + n_{\mathrm{pair}}.
   \]
   它約束兩個 bias 的差值，不會假裝單獨知道共同的絕對偏差。其量測 variance 包含 GNSS 航向的 3° 工程標準差、IMU yaw covariance，以及配對時間間隔內兩個 bias 的 random-walk variance。
2. **過去 GNSS 位置形成的行進方向**：在已完成的 GNSS XY 時窗估計 course，再與同一過去時段的 IMU yaw 比較，得到
   \[
   z_{\mathrm{course}}=\operatorname{wrap}(\psi_{\mathrm{course}}-\psi_{\mathrm{IMU}})
   = [1,0]b + n_{\mathrm{course}}.
   \]
   只有行進方向足夠可觀測且轉向條件合格時，這個觀測才幫助約束 IMU 對地圖的航向對齊。

濾波器保存完整的 2×2 covariance。相對航向更新會建立兩個 bias 的交叉 covariance；不能把它拆回兩個互不相關的純量，也不能在重複讀到同一個 GNSS yaw 時假裝又有一筆獨立資訊。地圖 prior 仍透過既有 scalar 介面使用 `b_IMU` 及其邊際 variance `P_b[0,0]`，所以 v5.1 的地圖配準不需要改成另一種 bias API。

## 每幀因果處理步驟

1. 感測器 reader 同時檢查兩個時間界線：bag arrival timestamp 不晚於目前 delivery watermark，ROS measurement timestamp 不晚於目前 camera frame。處理一幀後不回頭重算已輸出的 pose。
2. GNSS `PoseStamped` 的有效 quaternion 轉成 yaw；以相同 measurement time 的 IMU yaw 配對。IMU 姿態歷史只含已抵達樣本；若需要短暫外推，也只使用當下已知角速度，超過設定的 IMU 時間界線便等待或拒用。
3. GNSS–IMU pair 只有在 IMU gyro norm 小於 `2 deg/s` 時才可更新。相同的 held GNSS yaw 不重複使用；不相容的方向 pair 也會拒絕。
4. GNSS XY course 從尚未採用的過去位置窗計算。時窗在 1–8 秒範圍內累積；只有方向標準差不高於 `3 deg`、IMU 歷史覆蓋完整且沒有超出轉向條件時，才形成一次 course 更新。採用過的 GNSS 樣本不再當成新資料重用。
5. 每次更新只改當下 bias mean 與 covariance。修正後的 sensor prior 傳給原有 v5.1 map registration 和因果 correction filter；不使用未來訊息、不離線回修舊 pose。

## 目前的誤差預算

下列數字是明確寫入模型的工程預算，尚未由獨立標定或真值校準，也沒有用非因果 CSV 擬合：

| 項目 | 暫用值 | 用途與限制 |
| --- | ---: | --- |
| `b_IMU` 初始標準差 | 10° | 沿用既有 IMU heading alignment 預算 |
| `b_GNSS` 初始標準差 | 10° | 假設 GNSS quaternion 也可能有偏差；不是其精度保證 |
| GNSS heading observation 標準差 | 3° | `PoseStamped` 沒有 orientation covariance，此值是工程預算 |
| heading bias random walk | 各 0.12°/√s | 允許兩種 bias 隨時間緩慢變化；不是量測結果 |
| IMU orientation covariance | 有效訊息 covariance；否則用設定的 unknown fallback | bag 的零或無效 covariance 不代表零誤差 |

兩 bias 的初始 covariance 對角線使用各自 10° 標準差；之後依 observation 和 random walk 更新，保留交叉項及共同偏差的不確定度。GNSS frame 定義、quaternion 的實際參考座標、GNSS/IMU/PLY 外參及 lever arm 仍未獨立標定。位置 course 也可能受倒車、側滑、低位移或轉彎影響，不能一律等同車頭方向。

## 驗證狀態與限制

Track2 合成 CPU／CUDA／ROS／runner 檢查 **58 passed，0 failed**；pytest 摘要 14.40 秒，XML suite time 14.402 秒。只有既有 Matplotlib `Axes3D` warning，不影響 2D 圖。這是生成資料與程式契約檢查，不是真實定位或精度證明。

v5.2 尚無真實 full、提交輸出或分數證據。v5 與 v5.1 已有 full run，分數為使用者回報；它們與非因果 CSV 的對照不能證明 v5.2 精度。

## 證據與程式快照

- [v5／v5.1 分析摘要](../outputs/run/track00_test_2610291527/analysis/summary.json)：既有 run 的唯讀比較；摘要明確標示 reference 不是 ground truth。
- [感測輸入稽核](../outputs/run/track00_test_2610291527/analysis/sensor_input_audit.json)：記錄 GNSS／IMU 訊息與 quaternion 輸入。稽核包含 475 筆 GNSS 訊息；觀察到的 GNSS quaternion norm 範圍約為 0.99999997–1.00000003。這確認 quaternion 資料存在且格式近似單位長度，不代表其 frame 或方向精度已標定。
- live [`causal_prior.py`](../track2/causal_prior.py)、[`sensors.py`](../track2/sensors.py) 與 [`parameters.py`](../track2/parameters.py)：分別實作雙 bias 融合、GNSS quaternion 因果讀取與航向預算。
- [套用收據](../outputs/run/track00_test_2610291602/apply_receipt.json)：記錄五個新增／更新檔案的舊、新 SHA256 與套用前無定位程序檢查。
- [58 項檢查 pytest.xml](../outputs/run/track00_test_2610291602/pytest.xml)：完整測試結果及時間。
- [1527 分析報告索引](../outputs/run/track00_test_2610291527/analysis/README.md)與[1602 驗證圖表索引](../outputs/run/track00_test_2610291602/analysis/README.md)：兩份索引與 PNG／PDF 已完成。

### 早期測試修正紀錄

兩次未通過的中間測試仍保留作歷史：[`track00_test_2610291553`](../outputs/run/track00_test_2610291553/pytest.xml) 的 finite-prior shrinkage 容許值設定不合理；[`track00_test_2610291559`](../outputs/run/track00_test_2610291559/pytest.xml) 原欲建立無效 quaternion，卻保留 ROS 預設 `w=1` 而成為有效 identity quaternion。修正後的最終檢查為上列 58 項全通過。

# Track2 v5.9：因果影像品質、尺度與感測器契約

v5.9 解決的是：舊影像速度常被幾何門檻擋掉；單一點雲作用距離難兼顧近處細節與遠處道路；高度估計容易把已用過的先驗再融合一次；航向與 gyro bias 的更新也要避免重複計數或把起步加速當成直行。這一版保留 v5.2 的方向性地圖記憶，讓影像、尺度及感測器訊息逐次通過可觀測性和統計檢查。

v5.9 程式凍結快照保存在 `outputs/run/track00_test_2611031541/candidate_source/`。Track2 GPU 回歸 **211 passed**，有 1 項既有 Matplotlib `Axes3D` warning，耗時 27.83 秒；117 個 Python 凍結來源 SHA 已核對一致，v5.9 已套用。v5.9 尚無真實定位或 Public Score，最新一筆有 Public Score 的 GPU Track2 full 是 v5.8，使用者回報 `0.45341`。本文件不保證能達到 `0.3`；`result_2_0.30488.csv` 只作同索引差異參考，不用來擬合參數、時間差或回修 pose。

## 從觀察到 v5.9 的改動

### 1. 先把輸出參考點與外參未知處寫清楚

本次官方規則／資料核對只找到相機之間的外參，沒有找到 GNSS／IMU 到相機的實體標定，也沒有找到 Track2 評分 pose 的參考點與 yaw 正方向契約。程式若把 GNSS 天線、IMU 或 PLY 原點當成同一點，位置可以出現固定 lever-arm 偏差；只靠 header 時戳相近也不能證明感測器物理延遲為零。v5.9 runner 會在 [`run_track2.py`](../scripts/run_track2.py) 中寫出 `reference_contract.json`，並保留各項假設的參數位置。

因此 v5.9 維持既有 `camera_ply` 輸出：`T_map_ply`、相機 FLU、位置單位公尺，yaw 依現有輸出計算 `atan2(R10,R00)` 弧度；這是保留的程式輸出契約，不能當成已證明的官方評分定義。runner 會寫 `reference_contract.json`，標出評分參考點與物理延遲仍未知、GNSS／IMU 到 PLY 尚未校準，並記錄採用的零均值與不確定度預算。沒有套用作者 Track1 的 body lever 或 yaw offset，也沒有把此項寫成完成標定。

### 2. 讓道路影像以已知 gyro 旋轉估平移

v5.8 的 full log 有 471 個影像區間候選，只有 15 次移動速度和 17 次靜止量測通過融合；最後一次移動量測在 frame 630。當中 228 個候選雖有至少 24 個 homography inlier，但全畫面 inlier 比例低於 60%。這些候選不因此就是真量測，卻顯示單一全畫面比例門檻可能把可用的稀疏路面支持也一併擋掉。frame 630 的 1.54 m 與 frame 2173 的 1.73 m 是舊版高度估計，並非相機高度真值。可查 [v5.8 full 的影像摘要](../outputs/run/track02_full_2611031406/analysis/image_speed_summary.json) 與[逐幀 observations](../outputs/run/track02_full_2611031406/observations.jsonl)。

v5.9 改用互不重疊、約 `0.25 s` 的影像區間。相鄰影格以 Lucas–Kanade 追蹤路面角點；點雲可見性與前景遮擋遮罩在 CUDA 端產生。由已到達的過去 gyro 積分相機旋轉，再只估 3D 平移；這樣不必從近似共平面的路面點同時硬解六個自由度。homography RANSAC 至少要有 24 個內點；平面支持要跨至少 3 個影像格，平移 Jacobian 要滿 rank 3，還要通過重投影、正深度、垂直位移和速度檢查。若有另一組不重疊的路面模型，兩者速度也要通過 NIS 相容檢查；最後仍由原 GNSS NIS gate 決定是否納入融合。

因此，v5.9 取消「全畫面必須有 60% 內點」的硬拒絕，改以絕對內點數、空間分布、Jacobian 資訊量、遮擋處理、競爭平面及 GNSS 相容性共同判斷。gyro 旋轉 covariance 會傳入平移速度不確定度；平面高度、法向與影像模型也保留誤差項。這是因果的局部道路速度模型，沒有地圖絕對位置或 map yaw 量測，也未用真實全段 pose 證明精度。實作見 [`image_speed.py`](../track2/image_speed.py) 與 [`road_motion.py`](../track2/road_motion.py)。

60 個合成路面 case 中，舊求解流程接受 21 個，新流程接受 59 個。這組比較使用精確提供的 gyro 旋轉、`0.12 px` 像素雜訊，沒有抽樣 gyro 誤差；它只說明此理想合成條件下的接受覆蓋，不是實際影像精度或真實定位改善證據。圖表見[合成路面運動](../outputs/run/track00_test_2611031541/analysis/01_generated_road_motion.png)、[航向 bias](../outputs/run/track00_test_2611031541/analysis/02_generated_heading_bias.png)、[高度](../outputs/run/track00_test_2611031541/analysis/03_generated_height.png)及[本輪分析說明](../outputs/run/track00_test_2611031541/analysis/README.md)。

### 3. 同一幀比較多個點雲距離，只提交一個結果

固定遠距範圍容易被遠方稀疏或錯配點影響；只看近距又可能缺少判斷行進方向所需的幾何。v5.9 讓每個當前 frame 依序試 `12、18、25、40 m` 特徵範圍，每次都從完全相同的過去濾波狀態複製起跑。近距解若已接受且有至少 rank 2 的強觀測就停止；若只有部分 rank，先選可觀測 rank 較高者，再把各解放回同一份近距原始特徵，用包含未配對點懲罰的共同成本比較。若都不接受，保留最遠距範圍的原有 prior 輸出。

無論試了幾個範圍，最後只把一個解的狀態寫回濾波器；多個範圍不會被當成多筆獨立量測重複融合。參數與流程見 [`parameters.py`](../track2/parameters.py) 和 [`metric_registration.py`](../track2/metric_registration.py)。多範圍會增加當幀計算；實際速度影響與定位精度仍需真實 full 判讀。

### 4. 將高度當成獨立資料量測，避免先驗重複計數

配準求解本身已用過相機高度先驗，因此不能再把包含該先驗的 MAP 高度當作新的獨立量測。v5.9 沿用 point-to-plane data Hessian／gradient，先以 Schur complement 消去 pose nuisance，形成當幀資料本身的高度 likelihood。只有高度可觀測，而且高度位於物理及既有搜尋界內、統計 NIS 可接受、並且在同一組 final planes 上的實際非線性 data loss 沒有增加時，才以 Joseph covariance 更新一次。

當幀 scale 使用更新後高度除以當幀地面幾何係數 `d_s`。若高度量測不通過，就沿用過去高度，不把被拒絕的估計或已含 prior 的結果重新融合。估計的不確定度與高度工程誤差仍未由獨立真值校準；細節與實作見 [`metric_registration.py`](../track2/metric_registration.py)、[`height_filter.py`](../track2/height_filter.py) 及[79｜相機高度與位置](79_TRACK2_CAMERA_HEIGHT_POSITION_20261102.md)。

### 5. 限制航向資訊消費，並只在確認靜止時學 gyro bias

GNSS course 是一段時間的估計；起步時一個窗口可能同時包含停止與加速，不能直接當作穩定直行方向。v5.9 在符合長度條件時把過去 course window 分成兩個不重疊半段，要求兩半各自可觀測且方向相容，再允許整段形成一次航向 likelihood。每筆 likelihood 用 fractional power 分批消費；同一筆觀測的資訊總量最多等於一筆，每幀 heading calibration 更新不超過既有 `0.5°` 預算。

三軸 gyro bias 只在影像與 GNSS 都接受靜止的區間更新，用當時 gyro 平均值、gyro covariance 及工程誤差 floor 做一次 3D 濾波。它不直接用 map yaw 估 bias，也不回頭改已輸出的 pose。這個作法不等於完整聯合估計：跨狀態與跨影格相關性未完整建模，相關 gyro／外參／heading noise 仍屬工程預算。實作見 [`causal_prior.py`](../track2/causal_prior.py)。

## v5.9 主要設定

| 設定 | v5.9 值 | 用途與限制 |
| --- | ---: | --- |
| `FUSION_PROFILE` | `v52_directional_memory` | 保留 v5.2 map fusion；v5.5 實驗性 profile 不啟用。 |
| `FEATURE_RANGE_CANDIDATES_M` | `(12, 18, 25, 40)` m | 逐幀由近到遠選候選，不是四筆量測。 |
| `FEATURE_RANGE_SUFFICIENT_RANK` | `2` | 足以停止擴大範圍的候選幾何 rank。 |
| `IMAGE_SPEED_INTERVAL_S` | `0.25` s | 不重疊影像速度區間；沿用已消費一次的因果視窗。 |
| `IMAGE_ROAD_MIN_CELLS` | `3` | 路面模型至少跨三格，仍需通過其餘資訊與品質閘門。 |
| `HEADING_MAX_UPDATE_DEG` | `0.5°` | 每幀 heading calibration 的最大更新量，不是車身轉速上限。 |
| `GYRO_BIAS_STATIONARY_STD_DEGPS` | `0.10 deg/s` | 靜止 gyro bias 量測的工程誤差 floor，不是規格精度。 |
| `GYRO_BIAS_RW_DEGPS_SQRT_S` | `0.02 deg/s/√s` | gyro bias 隨時間變化的工程預算，尚未校準。 |
| `OUTPUT_REFERENCE` | `camera_ply` | 保留 camera PLY 輸出點；官方 scoring reference 尚未確認。 |

其中影像區間與 heading 更新上限沿用既有設定；本次不從參照 CSV 或 Public Score 擬合外參、偏差、noise 或時間差。其他外參假設、uncertainty 與設定位置都在 [`parameters.py`](../track2/parameters.py)。

## 圖表輸出

合成檢查的路面運動、heading bias 與高度圖表已放在 [本輪 analysis/](../outputs/run/track00_test_2611031541/analysis/README.md)。成功完成 v5.9 full 時，run 的 `analysis/` 會額外產生唯讀 `08_quality_models.png/pdf`，彙整每幀使用的範圍與 inlier 支持、gyro bias 與高度；它們只供檢視，不修改 pose、CSV、參數或定位結果。

## 驗證與目前狀態

- 版本字串為 `track2_causal_quality_motion_v5_9_perf2`；`v59_enabled()` 只在 v5.2 fusion profile 下啟用新路徑，v5.5 experimental profile 未啟用。`reference_contract.json` 記錄 output／calibration 的未知事項。
- 全部 Track2 GPU tests 為 **211 passed**，有 1 項既有 Matplotlib `Axes3D` warning，耗時 27.83 秒：[pytest_gpu_0002.xml](../outputs/run/track00_test_2611031541/pytest_gpu_0002.xml)／[log](../outputs/run/track00_test_2611031541/pytest_gpu_0002.log)。這驗證程式測試，不是定位精度證據。
- 117 個凍結 Python 檔的來源 SHA 已核對一致，v5.9 已套用。Track1／2 external full 的 launcher exit code 均為 0；v5.9 真實 full 尚未執行，沒有 v5.9 real-photo PnP、Public Score 或 full analysis 結果。
- 最新一筆 Track2 使用者回報成績仍是 v5.8 的 `0.45341`。使用者目標 `0.3` 尚未達成或獲得保證；定位精度需由後續 v5.9 full 結果判讀。

完整定位仍由使用者手動執行既有命令，不新增旗標：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
./scripts/run_track2.sh --full
```

### 附錄：首輪回歸與測試範圍

最初 GPU 回歸為 `209 passed、1 failed`（27.59 秒）。追查發現，失敗的高度拒絕 fixture 使用 12 m 範圍，近距解有 rank 2，但 `scale_information=0`；這代表當幀沒有可觀測尺度，本來就不應校準高度。原 scale／tilt fixture 改固定在 40 m，並保留兩組深度支持；另新增 12 m 測試，確認不可觀測時不誤稱為高度量測。Production gate 沒有放寬。修正後的同一輪 GPU suite 為上方記錄的 211 passed。

另外一輪 CPU suite 為 52 passed、5 deselected、5.44 秒，唯讀圖表 fixture 為 1 passed。這些是不同範圍的檢查，不與 211 項 GPU regression 相加。

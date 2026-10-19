# 11｜濾波與下一階段演算法改善分析（2026-10-17）

本文保留改進前的診斷與當時分數。後續已實作 heading_rts，使用者完整測試／提交回報 Track1 **1.16951**、Track2 **0.31967**，變更與證據見 [本輪進步紀錄](14_HEADING_RTS_RESULTS_20261017.md)；本文件的後處理濾波候選沒有沿用這組分數。

可以增加濾波，但目前不建議把一階 IIR 直接加在每筆最終 pose 後面。既有 `sensor_guided` 已有兩層平滑，新增離線平滑對本次既有輸出的改動只有公分等級；值得優先研究的是隨時間變化的 heading 校正、感測器／提交 frame 契約，以及地圖配準權重。這些是研究方向，尚未取得新 Kaggle 分數。

使用者本輪文字回報最佳分數為 **Track1 1.22369、Track2 0.56063**。Track1 原候選有既有截圖／SHA256 證據；Track2 最佳分數尚未與特定 CSV／SHA256 核對，不自動歸到下列 Track2 run，也不據此推論 Track2 baseline。

## 分析範圍與既有濾波

本次只讀既有完整 pose 與原始感測器輸入，另存後處理候選；沒有啟動 ROS localization、重新配準完整 Track、改動 CPU 結果或操作 Kaggle。

| 位置 | 實際行為 | 對新增濾波的意義 |
| --- | --- | --- |
| `scripts/prepare_sensor_prior.py:make_priors()` | GNSS 5 點二次 Savitzky–Golay，接著線性內插 | 約 4 Hz GNSS 的高頻成分已被平滑；約 28 Hz pose 並不是同等數量的獨立 GNSS 觀測 |
| 同函式 | IMU quaternion SLERP；由移動、近直行片段估固定 yaw bias | SLERP 是內插，不等於去噪；固定 bias 不能描述所有隨時間變化的方向差 |
| `workspace/src/localization/src/sensor_guided.cpp:RefineSensorGuided()` | `correction = 0.75 * previous + 0.25 * current`，再檢查當幀目標 | 已有一階 IIR，濾的是相對當幀先驗的小修正量 |
| 同函式 | 舊修正 ×0.8 只作最佳化起點候選，必須改善當幀目標才用 | 不是另外直接對完整位置作一次 IIR |
| `workspace/src/localization/src/localization.cpp` | XY／方向每幀重設為當幀先驗，高度沿用前幀 map 細修值 | 本版的時間記憶主要在修正量與高度，並非完整運動狀態估計器 |

目前 map 與感測器的權重使用固定值：GNSS XY 0.5 m、高度 1 m、yaw 3°、點雲尺度 0.3 m；不是依每幀品質校準出的共變異數。地圖殘差以 source 點數平均後加先驗，所以增加點數本身不會自動提高地圖總權重。

## 為什麼直接加 IIR 可能更差

若最終位置使用 `y[k] = (1-alpha) y[k-1] + alpha x[k]`，等速運動下穩態時間落後約為 `((1-alpha)/alpha) * dt`。本資料 `dt` 中位數約 0.036 秒；`alpha=0.25` 時落後約 0.108 秒，在 10 m/s 下就是約 1.08 m 的位置差。這是運動造成的滯後，即使完全沒有量測雜訊也會出現。

這與目前只平滑小量 map correction 的做法不同。若一定要因果運作，應使用含速度／角速度的預測與更新模型，不能只把上一個位置拿來平均。此次競賽方法本來就是離線，優先比較雙向或中心式平滑更合理。

## 既有輸出的比較結果

來源均為先前已完成的 `sensor_guided/cuda_robust_icp` run：

- Track1：`outputs/run/track01_full_2610171659/`，3011 幀、108.322 秒。
- Track2：`outputs/run/track02_full_2610171703/`，3299 幀、118.683 秒；尚未證明它對應使用者回報的 0.56063。

下表是**濾波後相對原答案的 RMS 改動，不是定位誤差，也不是分數改善量**。局部多項式使用 0.5 秒視窗、二次擬合及實際非等間距 timestamp；RTS 設定詳見工具／JSON，僅是初始實驗值。

| 方法 | Track1 XY 改動（m） | Track1 yaw 改動（°） | Track2 XY 改動（m） | Track2 yaw 改動（°） |
| --- | ---: | ---: | ---: | ---: |
| 原始輸出 | 0 | 0 | 0 | 0 |
| 只平滑 XY | 0.01342 | 0 | 0.01710 | 0 |
| 只平滑 yaw | 0 | 0.04599 | 0 | 0.04550 |
| XY／yaw 局部多項式 | 0.01342 | 0.04599 | 0.01710 | 0.04550 |
| XY／yaw Kalman + RTS | 0.01962 | 0.05763 | 0.01946 | 0.05028 |
| 最終 pose 一階 IIR，alpha=0.25 | **1.09112** | 0.11878 | **1.21797** | 0.11054 |

這表示溫和平滑產生的答案與現有答案很接近，單靠這個變更大幅改善的依據不足；不能把「濾掉的差值」全部當作原來的雜訊。未知真值、相關誤差與低頻偏差仍可能主導分數。IIR 的大改動也不等於正式誤差增加 1.09／1.22 m，但有明確運動滯後風險。

既有 map XY correction 的 RMS 僅 Track1 0.0689 m、Track2 0.0466 m，最大值約 0.187／0.185 m；yaw correction RMS 約 0.303／0.181°。修正距離沒有接近 1.5 m 上限，因此只放寬上限並不是目前最有證據的方向；須先判斷是先驗已很好、權重太強、尺度／幾何問題，或 correspondence 的資訊不足。

完整資料與圖另存在新的唯一目錄，原始輸出不變：

- [Track1 JSON](../outputs/run/track01_test_2610172143/report.json)、[比較圖](../outputs/run/track01_test_2610172143/comparison.png)。
- [Track2 JSON](../outputs/run/track02_test_2610172143/report.json)、[比較圖](../outputs/run/track02_test_2610172143/comparison.png)。

JSON 包含原始檔 SHA256、參數、逐方法改動及低速／轉彎／移動直行分段診斷。這些 outputs 被 Git 忽略；跨 checkout 交接須另備份，不能只依賴 Git。

## 比濾波更值得優先調查的現象

沿用目前 heading bias 選樣條件：速度 >5 m/s、IMU yaw rate 絕對值 <2°/s、方向差絕對值 <15°。以 20 秒一段計算 GNSS 行進方向減 IMU yaw 的中位數：

| 時間區間（s） | Track1 方向差（°） | Track2 方向差（°） |
| --- | ---: | ---: |
| 0–20 | 3.312 | 3.199 |
| 20–40 | 3.128 | 3.965 |
| 40–60 | 4.513 | 2.517 |
| 60–80 | 4.473 | 1.220 |
| 80–100 | 3.408 | 0.235 |
| 100–結束 | 4.216 | -0.223 |

目前整包固定校正值為 Track1 +3.911°、Track2 +1.633°。Track2 的分段差值有明顯時間變化，**值得比較固定 bias 與低頻、受限的時變 bias 模型**。但 course 並非車體朝向真值；側滑、GNSS 誤差、天線參考點與同步也可能造成差值，不能直接把整條差值套回 IMU，或斷言已證明 IMU 漂移。

建議先維持目前校正作對照，再以有品質門檻的方向觀測估 `b_yaw(t)`，限制變化速度／範圍，在低速、轉彎、資料缺口時不建立不可靠的 course 更新。離線可對 bias 狀態使用 Kalman + RTS，或以 robust 分段／樣條擬合。這是**校正狀態的估計**，比直接低通完整位置更有針對性。GNSS course 和 GNSS XY 來自同一資料，融合時必須注意相關性，不能視為新增獨立感測器。

另外核對 bag timestamp 減 message header timestamp：GNSS 中位數約 Track1 1.138 ms／Track2 1.095 ms；IMU 約 0.211／0.294 ms，兩類觀測最大值均小於 4.8 ms。這不支持直接假設存在 0.1 秒的 bag/header 差；PLY 影像曝光、原始擷取流程與提交時間契約仍需另核對。10 m/s 下若真的存在 0.1 秒未建模的時間差，影響約 1 m，但這只是敏感度示例。

## 方法選擇與實驗順序

| 方法 | 此專案的適合用途 | 主要限制 |
| --- | --- | --- |
| 因果 IIR／移動平均 FIR | 小量修正或有速度預測的即時管線 | 直接濾最終位置會落後；因果對稱 FIR 也有群延遲 |
| 中心式二次局部多項式（SG 類） | 低成本的離線 XY／yaw 對照 | 視窗過長會扭曲彎道與端點，不能消除固定偏差 |
| Butterworth + `sosfiltfilt` | 指定截止頻率的離線零相位對照 | 非等間距需處理取樣；端點暫態仍存在，未納入本次實作 |
| Kalman + RTS | 有運動狀態的離線平滑；後續可估慢變 bias | Q/R 與誤差相關性會影響結果；本次工具只平滑輸出，不是完整 GNSS／IMU 融合 |
| Factor graph／batch smoothing | 結合原始 GNSS、IMU、獨立相對運動與地圖因子 | 工作較大；不能把 GNSS 生成的 pose 再當獨立 GNSS 重複融合 |

建議順序如下，**每輪只改一項，保留目前最佳檔及 SHA256**：

1. 先核對 Track2 最佳 CSV 來源，並整理兩條 Track 分數與方法對照表。現有候選先比較 `sg_yaw`、`sg_xy`，將位置與方向的效果分開；不依平滑度自動選勝者。
2. 優先研究固定／時變 yaw bias 的消融。PDF 描述的旋轉 RMSE 權重為 0.2／度，單獨少 1° 旋轉 RMSE 對總指標的影響是 0.2；不是目前已有 1° 可消除誤差的估計。
3. 檢查 source/map/車體參考點、尺度 3.0 與時間契約。確認前不重套外參、不靠 public score 盲猜全域平移／yaw offset。
4. 將目前硬編碼的 prior sigma、cloud sigma 與 alpha 開放成可記錄的參數，再研究配準品質門檻、inlier 比例／空間覆蓋及 Hessian 退化方向。point-to-plane 或 coarse-to-fine 是可比較候選，不能保證優於現有 robust point-to-point。
5. 需要更大改善時，設計帶速度、yaw rate、慢變 yaw/GNSS bias 的 joint smoother，以原始觀測／獨立相對運動作因子；保留目前 bounded robust map matching，避免回到自由配準漂移。

短序列只用於功能檢查；演算法選擇需涵蓋轉彎、直行、低速與後段，不能只看前 50 幀。完整重新定位、Kaggle 提交依既有約定由使用者執行。幾何殘差、內部目標與 GNSS 差異均不能取代正式分數。

## 獨立比較工具與可重現命令

新增 [`scripts/trajectory_filter_lab.py`](../scripts/trajectory_filter_lab.py)，讀取驗證通過的 GPU run，在 `outputs/run/trajectory_filters_trackN_<unique>/` 另存：

- `baseline`、`sg_xy`、`sg_yaw`、`sg_xy_yaw`、`rts_xy_yaw`、`ema_lag_control` 各自的 CSV／TXT。
- `frames.csv`、`report.json`、來源程式快照；加 `--plot` 產生 PNG，加 `--bag` 讀感測器作 heading 分段診斷。
- CSV 保持 `idx,x,y,yaw`，yaw 弧度；TXT 用合法 yaw 旋轉左乘原 R，保留 roll/pitch 與 Z。先展開 yaw 再濾波，不逐元素平滑矩陣。

工具不接到 benchmark 的預設流程，不改 localization profile。候選**未重新核對每幀 map 目標函數及原修正上限**，不能沿用原 run 的 validation 宣稱候選配準已通過。`ema_lag_control` 是展示滯後的對照，不建議優先提交。RTS 的觀測標準差／加速度 PSD 是實驗設定，不能解讀成已校準的感測器不確定度。

Ubuntu 主機執行，不需重編 C++，也不啟動定位：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
docker exec sdc_midterm01_gpu bash -c 'source /opt/ros/humble/setup.bash && exec python3 /root/scripts/trajectory_filter_lab.py --run /root/outputs/run/track01_full_2610171659 --track 1 --window-seconds 0.5 --bag /root/data/Track1/Track1.db3 --plot'
docker exec sdc_midterm01_gpu bash -c 'source /opt/ros/humble/setup.bash && exec python3 /root/scripts/trajectory_filter_lab.py --run /root/outputs/run/track02_full_2610171703 --track 2 --window-seconds 0.5 --bag /root/data/Track2/Track2.db3 --plot'
```

終端印出的 `/root/outputs/run/...` 對應主機本 GPU 目錄下的 `outputs/run/...`。主機目前未安裝 NumPy，因此使用既有 GPU 容器的 NumPy 1.21.5／SciPy 1.8.0／Matplotlib 3.5.1。未啟停任何容器，也不必安裝新套件。

新增數值契約測試涵蓋非等間距與大座標、已知運動／雜訊的延遲與平滑、yaw 跨 ±180°、合法旋轉與 roll/pitch/Z 保留，以及 frame／Track／非法旋轉拒絕。合成測試只證明工具契約；真實輸出比較另存，兩者不混為精度驗證。

```bash
docker exec sdc_midterm01_gpu python3 -m unittest discover -s /root/scripts/tests -p 'test_trajectory_filter_lab.py' -v
```

## 方法參考

- [SciPy Savitzky–Golay](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html)：固定樣本視窗的多項式平滑；本工具為適應 timestamp 變動，直接以真實時間建立局部擬合。
- [SciPy forward/backward filtering](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html)：雙向濾波可得零相位，官方建議優先用 SOS 形式以減少數值問題，仍須處理端點。
- [GTSAM factor graph 教學](https://gtsam.org/tutorials/intro.html)：將不同觀測與狀態關係共同表示為因子；適合作為後續 joint estimation 的方法參考，不表示已在本專案整合或驗證。

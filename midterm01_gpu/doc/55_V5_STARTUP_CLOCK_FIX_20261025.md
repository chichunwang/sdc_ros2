# 55｜v5 相機時間、初段點雲與速度修正（2026-10-25）

這份紀錄整理 v5 從 frame 0 起步時點雲逐漸偏離 LiDAR map、位置前進過快的原因與修正。使用者設定的 `IMU_RPY_DEG = (1.0, 6.0, 0.0)` 已保留；此次沒有把 IMU 外參改成另一組數值，也沒有固定尺度為 7.5。此值仍是目前逐次測試的外參假設，不等於已完成相機與 IMU 的完整標定。

## 起始時間差把視覺速度放大了約三倍

PLY 檔名中的整數奈秒是 camera message 在 bag 的抵達時間；相鄰影像的 header 才是影像擷取時間。舊實作拿抵達時間差除以視覺估出的位移。Track3 第 0、1 張影像的證據是：

| 時間／速度 | 數值 |
| --- | ---: |
| PLY/bag arrival 間隔 | 11.712403 ms |
| camera header 擷取間隔 | 35.875839 ms |
| 舊 v5 算出的首段視覺速度 | 26.328 m/s |
| 使用擷取間隔後的首段視覺速度 | 8.595 m/s |

實際移動時間約為舊分母的 3.06 倍，所以舊速度被放大到約三倍。這能解釋「點雲往前衝得太快」的主要原因；它也表示單改 IMU 角度無法修正這個速度錯誤。

修正後，`track3/time_alignment.py` 由同一個唯讀 bag 找到目前 front camera 的 header 時間。每個 IMU sample 必須同時符合：它的 bag 抵達時間不晚於目前 PLY frame，且 IMU header 時間不晚於目前 camera acquisition time。已抵達但感測時間較晚的 IMU 留到下一張影像；已錯過舊影格的遲到 sample 會丟棄，不回頭改 pose。ESKF 積分和相鄰影像速度均使用相機擷取時間。PLY 檔名、輸出 frame 順序與 `timestamp_ns` 仍保留原本的 arrival 整數奈秒。

這樣既修正了速度的時間分母，也守住因果限制：目前 pose 只使用已抵達且量測時間不晚於目前影像的 sensor data。沒有從下一張影像插值，也沒有修回先前輸出的 pose。

## 兩個後續問題也一併修正

上一版把 PnP 位移先旋到 map 座標，再當作不相關的世界座標速度量測送進 filter。這重用了 filter 前一刻自己的方向估計；方向有誤時，這種回饋會讓視覺速度與 ESKF 相互拉扯。修正版將視覺位移寫成目前 IMU frame 的速度量測 `z`，並直接和目前世界速度轉回 IMU frame 的 `Rᵀv` 比較：

```text
r = z − Rᵀv
H_v = Rᵀ
H_θ = Rᵀ[v]×
```

這讓殘差與 Jacobian 都用同一個座標表示，並把姿態誤差對速度投影的影響列入量測模型。量測仍是相鄰影格的平均位移速度近似值，使用前一張 PLY 的尺度和對應深度；NIS 通過代表 filter 接受這個模型量測，不代表速度已成為獨立真值。

舊程式移動時以因果影像距離比更新尺度，registration 又把該尺度固定；當視覺對應的 3D 支持不足或有動態物體時，錯誤比例便無法由 map registration 修正。現在移動時讓 map Sim(3) 配準共同估 pose 和 scale；只有影像、gyro、持續時間與附近 3D 支持共同符合靜止，才使用因果視覺尺度並固定尺度。靜止尺度不是每一靜止 frame 都必然有值，配準失敗也不會把未採用的候選尺度寫進狀態。

## 從 frame 0 的 80 幀證據

修正版真實短段 run：[`track03_test_2610251132`](../outputs/run/track03_test_2610251132/test_info.json)。這輪從 Track 起點重新建立 initializer 和首筆 map anchor，沒有載入舊 checkpoint，也沒有從中間 frame 開始。設定與輸入 SHA、frame 數、耗時、IMU delivery summary 及輸出 SHA 均記錄於 run；程式／參數快照保存在該 run 的 `source/`。

| 觀察 | 結果 |
| --- | ---: |
| 完成 frame | 80 |
| 接受 map update | 63/80 |
| 最長連續 map 拒絕 | 13 frames |
| run elapsed time | 219.706 s |
| idx 0 絕對 scale | 6.79345 |
| idx 79 scale | 6.70886 |
| idx 79 ESKF 速度大小 | 9.185 m/s |

idx 0 的 pose／scale 與舊 v5 起始 anchor 相同，scale 是 initializer 與首筆可信 registration 的結果，沒有在測試中手設為 7.5。這輪的 ESKF 速度是狀態估計，不是車速真值。停車、後續轉彎及完整 Track 的表現仍須另外看較長的從頭測試。

幾何圖表比較舊 v5（0913）、修正版（1132）與歷史 v4 在共同影格上的輸出。每個被比較 frame 均使用同批相機 mask 與 CUDA SOR 點，再依 frame 0 scale 取 2–60 m 深度範圍內最多 12,000 個點。點雲分別以各 run 輸出的 pose／scale 轉到 map，透過 CUDA 搜尋距離不超過 3 m 的最近 LiDAR map XYZ；圖中的中位距離以 3 m 作上限，並列出 0.5 m 內的比例。這是共同點集上的幾何一致性檢查，沒有 ground truth，也不能當成競賽定位精度。

| 原始 frame | 舊 v5 最近 map 距離中位數 | 修正版最近 map 距離中位數 | 距離小於 0.5 m：舊 → 修正版 |
| ---: | ---: | ---: | ---: |
| 0 | 0.182 m | 0.182 m | 89.2% → 89.2% |
| 25 | 0.478 m | 0.403 m | 51.3% → 54.8% |
| 79 | 1.606 m | 0.231 m | 15.1% → 91.3% |
| 299 | 3.000 m capped* | 0.236 m | 0.0% → 88.2% |

*frame 299 的修正版與 v4 最近 map 距離中位數分別為 `0.236283 m` 與 `0.163968 m`；0.5 m 內比例為 `88.1583%` 與 `96.9917%`。舊 v5 的共同 12,000 個點全部都沒有 3 m 半徑內的 map 對應，所以表上的 capped median `3.000 m` 只是搜尋上限，不是其真實距離中位數。共同點集、selection 與上限的定義見 run analysis README 和 metrics JSON。

frame 25 仍顯示修正版和歷史 v4 幾何對齊有差距（v4 中位數 0.179 m、0.5 m 內 96.3%）。frame 79 和 frame 299 的修正版均比已知失敗的舊 v5 對齊好很多，但歷史 v4 在相同影格仍更好。這些數據支持受測點雲在 frame 79、299 回到 map 近鄰搜尋範圍內，不支持「全程已解決」或「比純 NDT/ICP 精準」。接受數、中位距離與圖上重疊都不是定位真值。

圖表和完整逐幀結果：

- [起始軌跡、速度、尺度與拒絕連續數 PNG](../outputs/run/track03_test_2610251132/analysis/01_startup_comparison.png)／[PDF](../outputs/run/track03_test_2610251132/analysis/01_startup_comparison.pdf)
- [同一批點雲和 map 的重疊 PNG](../outputs/run/track03_test_2610251132/analysis/02_cloud_overlap.png)／[PDF](../outputs/run/track03_test_2610251132/analysis/02_cloud_overlap.pdf)
- [最近 map 點距離摘要](../outputs/run/track03_test_2610251132/analysis/metrics.json)；[分析來源與限制](../outputs/run/track03_test_2610251132/analysis/README.md)
- 每幀的時間、尺度及 map update 在 run 的 `observations.jsonl`、`pose_trace.jsonl`、`timings.csv`、`parameters.json` 與 `imu_sources.json`。

300 幀 run 的唯讀圖表和數值在 [`track03_test_2610251140/analysis/`](../outputs/run/track03_test_2610251140/analysis/README.md)：[起始軌跡比較 PNG](../outputs/run/track03_test_2610251140/analysis/01_startup_comparison.png)／[PDF](../outputs/run/track03_test_2610251140/analysis/01_startup_comparison.pdf)、[frame 0／79／299 點雲與 map 疊圖 PNG](../outputs/run/track03_test_2610251140/analysis/02_cloud_overlap.png)／[PDF](../outputs/run/track03_test_2610251140/analysis/02_cloud_overlap.pdf)、[最近 map 點距離 metrics](../outputs/run/track03_test_2610251140/analysis/metrics.json)。點雲圖以 3 m 搜尋半徑封頂；圖中共同點集在這個距離外時，沒有對應的點會留在視覺化內，但 metrics 以 capped distance 計算，因此舊 v5 frame 299 的 3 m 中位數不得視為實際幾何誤差。

先前純修 clock 的 run [`track03_test_2610250935`](../outputs/run/track03_test_2610250935/test_info.json) 80 幀只有 6 幀接受 map update。這表示時間分母是已確認的 bug，但只修時鐘不足以解決點雲幾何逐步分離。被手動中止的舊 run `track03_full_2610250913` 留有 stale 的 `running` metadata；舊目錄保留原樣，分析使用逐幀 observations／pose trace，不以這份 metadata 判斷完成狀態。

## 延長至 frame 0–299

從 frame 0 重新啟動的 300 幀 run [`track03_test_2610251140`](../outputs/run/track03_test_2610251140/test_info.json) 已成功結束。它沒有載入 80 幀 checkpoint，仍使用同一 frozen runtime、`IMU_RPY_DEG=(1,6,0)` 和因果時間契約。

| 觀察 | 結果 |
| --- | ---: |
| frame 範圍 | 0–299（300 幀） |
| 接受 map update | 268/300 |
| 最長連續 map 拒絕 | 13 frames |
| elapsed time | 407.543 s |
| idx 299 位置 | `(567.314498, -188.355570, 8.534851) m` |
| idx 299 scale | 7.049991 |
| idx 299 ESKF 速度大小 | 9.576343 m/s |
| idx 299 map update／連續拒絕數 | accepted／0 |

[`contract_audit.json`](../outputs/run/track03_test_2610251140/contract_audit.json) 確認 300 幀的 PLY arrival timestamp 與 frame、CSV/TXT/trace、來源及輸出 SHA 契約相符；IMU arrival 不晚於目前 PLY、IMU header 不晚於目前 camera header。300 幀 run 的前 80 個 pose 與 scale 和獨立 80 幀 run 完全一致，map decisions 相同，`poses.txt` 的前 80 行位元組也完全相同。因此這個較長前綴重現了首段 80 幀結果，沒有因為從短段接 checkpoint 而改變前綴。

這輪只測到 frame 299，沒有涵蓋 frame 950 停車或 frame 2200 以後轉彎；268/300 map 接受數、幾何距離和輸出軌跡都不是獨立真值，不能證明完整 Track 或競賽精度已解決。完整 Track 仍由使用者親自執行 `./scripts/run.sh --full`。

機制驗證在 [`track00_test_2610251134`](../outputs/run/track00_test_2610251134/) 留下首輪與針對性重跑紀錄。共有 107 個 distinct test cases：首輪 106 passed、1 failed。唯一失敗是合成 runner 將非單位外參透過 `T E inv(E)` 往返後，出現 `1.11e-16` 浮點差；該首 anchor 斷言使用 `array_equal`，並非算法失敗。僅將此 IMU 首 anchor 斷言改成 `atol=2e-15` 後，針對性重跑的兩個 runner cases 都通過（2.27 s）；其餘 106 cases 沿用首輪通過紀錄（43.22 s）。因此每個 distinct case 都有通過證據，但沒有一輪未修改來源的 107/107 全通過執行。數學 fixtures 明確指定各自的外參；另有旋轉外參與 runner 有效設定檢查，均不改使用者的 `(1,6,0)` 設定。

## 程式位置與後續測試

- 相機時間與 IMU acquisition 排序：`track3/time_alignment.py`。
- IMU→PLY 外參直接修改：`track3/imu_parameters.py`。目前 `IMU_RPY_DEG = (1.0, 6.0, 0.0)` 是使用者設定。
- body-frame 視覺速度殘差與 ESKF：`track3/inertial.py`。
- moving Sim(3) 與 rest-only 因果尺度政策：`scripts/run_track3.py`、`track3/motion_parameters.py`。
- 完整逐幀觀測：[`track03_test_2610251132/observations.jsonl`](../outputs/run/track03_test_2610251132/observations.jsonl) 與 `pose_trace.jsonl`。

後續若需驗證長時間漂移、停車解除及轉彎位置，測試須由 frame 0 重新開始，並且只使用當前或過去資料。單獨測試 frame 2200–2600 暫不進行。完整 Track 仍由使用者親自啟動 `./scripts/run.sh --full`；此份 300 幀結果不是 full 或提交成績。

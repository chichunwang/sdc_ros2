# heading_rts 完整測試與 Kaggle 進步紀錄

本文保留 heading_rts＋geometry off 的歷史成果。後續 directional 已刷新為 **1.15275／0.30576**，見 [最新幾何成績紀錄](GEOMETRY_RESULTS_20261017.md)；兩代 CSV 均獨立保存。

紀錄日期：**2026-10-17（Asia/Taipei）**。使用者手動執行完整 Track、提交 Kaggle，並在本輪提供兩張排行榜截圖。Codex 讀取既有輸出核對與保存證據，沒有重新啟動定位或操作提交。

## 成績與證據

| 項目 | Track1 | Track2 |
| --- | ---: | ---: |
| 本次 `heading_rts` 分數 | **1.16951** | **0.31967** |
| 本次截圖顯示的前最佳 | 1.22369 | 0.56083 |
| 分數降低量 | 0.05418 | 0.24116 |
| 相對前最佳降低 | **4.43%** | **43.00%** |
| 截圖當下排名 | **第 3 名** | **第 1 名** |
| 同圖助教 baseline | 1.79075 | 1.92587 |
| 相對 baseline 分數降低 | 34.69% | 83.40% |

兩條本次分數均低於各自截圖中的 baseline。排名只代表**使用者提供截圖當下**，本次沒有查詢即時排行榜。分數降低百分比為 `(舊分數 − 新分數) / 舊分數 × 100%`，不是「定位精度提高百分比」。

Track2 歷史數值保留差異：使用者先前文字回報前最佳 **0.56063**，這次截圖的 personal-best 訊息顯示 **0.56083**。上表依新截圖計算；若用前次文字值，降低量為 0.24096、相對降低 **42.98%**。先前 Track2 CSV 尚未綁定到該舊分數，不回頭改寫舊候選的成績證據。

截圖原始來源為本輪對話附件；本機另存的是分數／排名轉錄、run 證據與 SHA256，沒有假稱原始截圖已下載到本地。

## 這次實際改了什麼

上一版 `sensor_guided/fixed` 是每幀 GNSS XY＋IMU 姿態先驗，再接 bounded robust map refinement；IMU heading 用整包 GNSS／IMU 估計的一個固定 bias。新版本為同一 profile／後端加入 **`--sensor-fusion heading_rts`**，在定位前估計隨時間變化的 heading 校正。

| 改動 | 原 `fixed` | 本次 `heading_rts` |
| --- | --- | --- |
| heading 校正 | 整段一個穩健中位數 | 慢變 bias 狀態 `b(t)`，Kalman forward＋RTS backward 平滑 |
| GNSS course 觀測 | 平滑位置的相鄰數值微分 | 互不重疊的 2 秒 GNSS 區塊，以 robust 線性迴歸估速度方向 |
| 觀測品質 | 逐筆速度／轉向門檻 | 區塊樣本與跨度、速度、yaw rate／跨度、方向差及離群點門檻 |
| 觀測權重 | 合格樣本一同估固定值 | 由區塊幾何與散布估 R；大 innovation 降權 |
| 校正保護 | 固定 bias | bias ±10°、變化率最多 0.2°/s；資料不足明確退回 fixed |
| 追查與重現 | prior、frame、run 參數 | 加入逐幀 bias／模型標準差、每區塊接受原因、fusion 設定與來源快照 |

核心觀測為 `z = wrap(GNSS_course − IMU_yaw)`，慢變狀態模型為 `b[k+1] = b[k] + w`。輸出方向為 `Rz(b(t)) × R_imu_slerp(t)`，保留 IMU roll／pitch。低速／轉彎區塊不建立不可靠的 course 更新，仍使用 IMU 姿態與模型的 bias 校正。

新增與修改的程式：

- [`scripts/sensor_fusion.py`](../scripts/sensor_fusion.py)：觀測區塊、品質門檻、robust bias Kalman／RTS、校正限制與診斷；另提供可選位置／速度 RTS。
- [`scripts/prepare_sensor_prior.py`](../scripts/prepare_sensor_prior.py)：串接 fusion 模式、保留整數 timestamp，產生既有八欄 prior 及額外診斷檔。
- [`scripts/accuracy/benchmark.py`](../scripts/accuracy/benchmark.py)：新增選項、參數／來源快照與 fusion 輸出核對。
- [`scripts/tests/test_sensor_fusion.py`](../scripts/tests/test_sensor_fusion.py)：慢變 bias、離群值、低速／轉彎／倒車、yaw wrap、位置模型與時間契約測試；與既有測試合計 11 項通過。

**本次高分使用的是 `heading_rts`，沒有開啟 `motion_rts` 的 GNSS 位置／速度模型。** GNSS XY 先驗仍為原 5 點二次 Savitzky–Golay＋線性內插，後段仍為 CUDA 最近鄰＋CPU robust 四自由度細修。方向先驗改變後，map refinement 的最終 XY 也可能跟著改變。

兩條 run 的 localization／CUDA backend **binary SHA256 均與先前 fixed run 相同**。PLY scale=3.0、map/source voxel=1.0/0.3 m、配準 correction IIR alpha=0.25、既有先驗權重與修正限制沿用原版；本次主要修改 Python 感測器前處理與實驗保存。`source_sha256.json` 全表另有 README／Python bytecode 差異，沒有將全表誤記為完全相同。

尚未做各子項的獨立 Kaggle 消融，故把成果歸到**整組 heading_rts 改動**；不將全部降低量單獨歸功於 RTS、某個門檻、位置濾波或 GPU 加速。GNSS course 也不是姿態真值，這仍是會使用較晚觀測的離線方法。

## 完整測試的本機證據

| 項目 | Track1 | Track2 |
| --- | ---: | ---: |
| 完整結果列數 | 3011 | 3299 |
| 退出碼／validation_errors | 0／空 | 0／空 |
| 完整有限值檢查 | 通過 | 通過 |
| 接受 map 更新／sensor-only | 3011／0 | 3299／0 |
| GNSS heading 區塊：接受／總數 | 47／55 | 48／60 |
| innovation 降權區塊 | 6 | 0 |
| fixed bias 參考值（°） | +3.91087 | +1.63252 |
| 本次逐幀 bias 範圍（°） | +3.24471～+4.40851 | −0.09812～+3.66931 |
| 最大 XY map correction（m） | 0.18403 | 0.18892 |
| 最大 yaw map correction（°） | 0.98194 | 1.14580 |
| 感測器前處理（s） | 1.393 | 1.430 |
| ROS 執行（s） | 307.409 | 140.609 |
| 前處理＋ROS（s） | 308.802 | 142.039 |
| CUDA 搜尋＋CPU 細修（s） | 6.565 | 3.737 |
| map cache | 停用 | 停用 |

`timings.csv.ndt_ms` 是歷史欄名，以上內容為 CUDA 最近鄰＋CPU robust 細修，不能寫成 NDT 耗時。原 fixed Track1 曾命中 map cache，本次停用，總時間不能用來直接計算同條件速度倍率。JSON 的 `accuracy` 欄仍是當時 benchmark 未查 Kaggle 的本機驗證說明；後續使用者回報的分數另外保存，不覆寫歷史 validation。

原 run 與 CSV：

- Track1：[`track1_sensor_guided_heading_rts_cuda_robust_icp_d2d_3011f_f7zs6f2e/diagnostic.csv`](../outputs/runs/track1_sensor_guided_heading_rts_cuda_robust_icp_d2d_3011f_f7zs6f2e/diagnostic.csv)，[validation](../outputs/runs/track1_sensor_guided_heading_rts_cuda_robust_icp_d2d_3011f_f7zs6f2e/validation.json)。
- Track2：[`track2_sensor_guided_heading_rts_cuda_robust_icp_d2d_3299f_84_esf9m/diagnostic.csv`](../outputs/runs/track2_sensor_guided_heading_rts_cuda_robust_icp_d2d_3299f_84_esf9m/diagnostic.csv)，[validation](../outputs/runs/track2_sensor_guided_heading_rts_cuda_robust_icp_d2d_3299f_84_esf9m/validation.json)。

兩份 CSV 都已再次核對 `idx,x,y,yaw`、連續 0-based idx、3011／3299 列及有限數值，並以 byte 相同的方式另存到新的唯一保存目錄：

- [Track1 result_1.csv](../outputs/submissions/heading_rts_20261017_2akh94t6/result_1.csv)
- [Track2 result_2.csv](../outputs/submissions/heading_rts_20261017_2akh94t6/result_2.csv)
- [分數、來源、參數與 SHA256](../outputs/submissions/heading_rts_20261017_2akh94t6/provenance.json)

| 檔案 | SHA256 |
| --- | --- |
| Track1 CSV | `d7883426ea17e33fbfadf239fb42edca7722ce7d32cc24100b2fbe28ffd6e925` |
| Track2 CSV | `5648527318a1772e24df131d11179334bd26ef86c252043cdf71ce64440e54ac` |
| 本次 sensor_fusion.py | `20dd38ed34dd66188b868f95ffc466e9c2a061fe902c49453e600bab8e3d2d88` |
| 本次 prepare_sensor_prior.py | `182114fd3dab299b87b2f6d477d369192cd36ed6f1e8e5f8558689a39f029355` |
| 本次 accuracy/benchmark.py | `4560deb970fe6b62b6d556390f00bdfb92528d72abe3da68a116ba7796c4199d` |

這些雜湊綁定本機當次檔案與對話中的提交回報；截圖本身不含上傳檔案 hash，本次沒有從 Kaggle 下載 submission metadata 作獨立核對。保存目錄另記在 `outputs/latest_heading_rts_submission.txt`。outputs 依 `.gitignore` 忽略，跨環境時須另備份這兩份 CSV 與 provenance，不能假設只靠 Git 會同步。

## 重現本版

完整流程已由使用者執行；以下僅保留重現命令，之後完整測試仍由使用者手動啟動：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
python3 scripts/benchmark.py --track 1 --frames 3011 --profile sensor_guided --sensor-fusion heading_rts
python3 scripts/benchmark.py --track 2 --frames 3299 --profile sensor_guided --sensor-fusion heading_rts
```

程式省略 profile 仍是 `legacy/cuda_ndt`；指定 sensor_guided 但省略 sensor-fusion 仍是 `fixed`。本輪推薦使用已取得分數的 **heading_rts 明確選項**，沒有把推薦誤寫成程式預設已更換。`motion_rts` 目前僅有短測，不沿用本次成績。

方法細節與品質限制見 [融合設計](SENSOR_FUSION_20261017.md)，演進前的濾波分析見 [濾波診斷](FILTER_ANALYSIS_20261017.md)。

# 32｜相機點雲上下浮動：Z 回授與防漂移候選

日期：2026-10-19（Asia/Taipei）。使用者確認現象是「LiDAR 不動，相機點雲一直上下飄」。本輪核對前相機最佳，修改高度約束，完成兩條 Track 的分散短段對照。**候選減少部分高度浮動，但未一致改善 LiDAR 貼合，仍為實驗選項，不替換最佳。**

## 保存結果與來源幾何

[31｜回放修正](31_REPLAY_PLAYBACK_AND_HEIGHT_20261019.md) 已將相機點雲乘保存矩陣後發布在固定 map frame。本輪沒有移動 LiDAR 或補顯示高度；分析的是 `p_map = R_t × (3 p_raw) + t_t`。

| 完整最佳結果 | Track1 | Track2 |
| --- | ---: | ---: |
| nominal GNSS Z（m，幾乎常數） | 110.131813 | 17.344366 |
| pose Z 最低／最高（m） | 105.624535／110.485367 | 15.537108／17.950899 |
| pose Z 範圍（m） | **4.860832** | **2.413791** |

這是高度變動，不是真值誤差；道路可以有坡度，GNSS 的常數 Z 也不是實測路面。

另對六段各取 9 個固定時間分位，共 54 個來源 probe。只乘 IMU rotation、不加 pose translation，在前方 1–15 m、左右 ±5 m、相對 Z −4 至 −0.3 m，以固定 seed RANSAC 診斷下方平面候選。它不是已確認的同一路面，可能隨場景換面。

Track1 前 256 幀的來源平面相對原點高度範圍約 **0.213 m**，pose Z 同段範圍約 **4.458 m**；其餘五段來源平面範圍約 0.037–0.113 m。這支持前段數公尺上下位移主要出現在 pose Z 更新，不能全歸因於點雲自身變形；仍未排除深度、錯面匹配與 frame 問題。完整 JSON 也保存相同視野的 map 下方表面候選，沒有將它們直接當成語意路面或高度真值。

## 回授機制與修改

原本每幀 XY／rotation 回到當下 GNSS／IMU，Z 則沿用上一幀輸出：

```text
carry（保留的預設）：
prior_z = z_(t-1)
L_z = 0.5 × dz² / 1m²
|dz| ≤ 1.5m
z_t = prior_z + dz
```

限制的是這一步，而非整條軌跡相對基準的總偏移。若有偏的來源幾何或不同高度層持續給出同方向梯度，Z 可以累積移動。原 smoothing 又混合上一幀 dz，但 prior Z 已含上次輸出，會延續上一個高度增量；這與每幀重新錨定的 XY／yaw correction 語意不同。

新增 `--height-mode anchored`（C++ `sensor_height_mode`），只接受 sensor_guided／cuda_robust_icp：

```text
h_t = 當幀 nominal sensor Z
prior_z = h_t + clip(z_(t-1) − h_(t-1), −1.5m, +1.5m)
L_z = 0.5 × (prior_z + dz − h_t)² / 1m²
|prior_z + dz − h_t| ≤ 1.5m
```

新模式約束相對 nominal altitude 的**總 offset**，不重用上一幀 dz 作為新增量；仍使用既有 alpha=0.25。nominal altitude 改變時會跟隨，沒有將全局 Z 寫成常數。拒絕 map 更新時保留當下預測高度，XY／rotation 仍用當幀先驗，避免突然把高度歸回 nominal。

CUDA 搜尋盒同步擴大：總 offset 可能由 +1.5 m 移到 −1.5 m，單次 dz 可以大於 1.5 m，搜尋必須覆蓋。XY／yaw、尺度、IMU rotation、directional 目標與 CUDA NN 保留。

**這不是正式高度校準。** σ=1 m 沿用原正則化數字，1.5 m 沿用量級但改變邊界語意；本輪沒有把它們校準為感測器噪聲或道路高差。在 GNSS Z 為常數的資料上，真實道路高差可能超出這個模型，因此新模式不能升為推薦。

新 sensor_guided run 另寫 `height.csv`，保存整數 idx／timestamp、mode、nominal Z、prior Z、output Z、總 offset 與接受狀態。benchmark 核對筆數、時間、有限值、結果矩陣及 anchored 邊界。

## 12 組真實短段與負面結果

每條固定選開頭 256 幀、原序列約 1/3 與 2/3 的各 96 幀，每段兩個模式，共 **12 runs／1792 poses**。同一 Release binary、相同 PLY／尺度、逐值相同的完整 run sensor prior、同一已核對 Track／map SHA256／leaf 的 cache，其餘 solver 參數相同。全部退出 0、有限完整、validation 無錯誤。

NN RMSE 為每段 9 個固定共同 probe 的平均，距離 capped at 2 m，獨立 CPU exact NN。它不是 ATE；探針使用 0.1 raw-unit voxel representatives／最多 6000 點，不冒充 PCL voxel centroid 的逐位元重播。

| Track／原始 idx | 最大相鄰 ΔZ，carry→anchored（m） | probe NN RMSE，carry→anchored（m） |
| --- | ---: | ---: |
| 1／0–255 | 0.2324 → 0.0580 | 0.9629 → **1.1566（增加）** |
| 1／1003–1098 | 0.1366 → **0.1658（增加）** | 1.1288 → 0.8355 |
| 1／2007–2102 | 0.1949 → 0.1341 | 0.7903 → 0.7745 |
| 2／0–255 | 0.04193 → 0.01648 | 0.5706 → **0.5760（增加）** |
| 2／1099–1194 | 0.4810 → 0.2303 | 0.6489 → **0.6521（增加）** |
| 2／2199–2294 | 0.3003 → 0.1459 | 0.5184 → **0.5355（增加）** |

Track1 前段相對 nominal Z 最大偏移由 **4.4693 m 降至 0.3447 m**。但 6 段有 4 段 probe RMSE 增加，其中一段最大相鄰 ΔZ 也增加，不能宣稱全面消除浮動。

中途短段會重設高度，與完整 run 的歷史不同。另以原完整最佳 pose 作相同 probes，anchored 有 **5／6 段的 RMSE 更高**，不能藉短段 carry 的較差初始化宣稱勝過最佳。

![高度對照](../outputs/run/track00_test_2610191437/final_analysis/height_comparison.png)

證據位於新目錄 `outputs/run/track00_test_2610191437/`：

- [完整報告](../outputs/run/track00_test_2610191437/final_analysis/report.json)：來源／map 平面候選、54 個共同 probes、XY／yaw 差異、來源 hash 與限制。
- [run 對應](../outputs/run/track00_test_2610191437/pairs.json)，以及各命令 JSON／log／`experiment.py`。
- [相容核對](../outputs/run/track00_test_2610191437/legacy_prefix_compatibility.json)：兩條原最佳前 256 幀，carry 的 `result.txt`／`diagnostic.csv`／`geometry.csv` 均 **byte 相同**。

outputs 被忽略，須另行備份。原最佳、CPU 結果及個人紀錄未由本輪修改；沒有新完整 Track 或 Kaggle 分數。

## 尚待確認的物理模型

Track1 初始矩陣／第一筆 sensor prior 的 roll 為 −7.609°／+2.332°，相差約 9.94°；Track2 為 +0.007°／−0.217°。這只證明輸入不同，初始矩陣不是已驗證的 IMU→source 外參，不能由 Track1 反推適用兩條 Track 的固定 roll 補償。

目前固定 IMU roll／pitch，`ply_scale=3` 也不是正式 metric calibration。source／body／GNSS 原點與 lever arm、傾斜、尺度／深度形狀，可能讓 Z 去吸收不一致；單一 Z 平移無法修正距離相關或左右不同的形變。本輪沒有盲改 roll、scale 或相機高度。

要同時獲得穩定高度與正確貼合，後續需確認 source frame／外參，並以可確認的同一靜態表面建立沿路高度與跨幀一致性模型。map 有多層高度，不能將附近全部點的平均 Z 當路面。這是尚待驗證的工作，不寫成已完成的物理高度修正。

## 回放候選與重現

直接檢視已計算的 256 幀候選，不重算定位：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
python3 scripts/replay.py --track 1 --paused \
  --run outputs/run/track01_test_2610191438_0002
```

此候選 `--check-only` 已核對來源、整數時間、旋轉與 cache；省略 `--run` 仍回放原最佳。

重做短測，另建唯一 run：

```bash
python3 scripts/benchmark.py --track 1 --frames 256 \
  --profile sensor_guided --sensor-fusion heading_rts --geometry-mode directional \
  --height-mode anchored \
  --prior-run outputs/run/track01_full_2610172233 \
  --map-cache outputs/run/track00_test_2610171550/cache_track1_leaf1.pcd
```

改為 `--height-mode carry` 可對照；省略仍是 carry，[16｜最佳手動命令](16_run_geometry_directional.txt) 不變。本輪不支持立即啟動完整候選提交。

唯讀重產診斷：

```bash
outputs/ml_runtime/venv/bin/python scripts/analyze_height_modes.py \
  --pairs outputs/run/track00_test_2610191437/pairs.json \
  --output outputs/run/<新的唯一分析目錄>
```

## 驗證

GPU workspace Release 建置完成；3 項 CTest 全過，3 個相同 test executable 另以真實 CUDA NN 執行通過。新高度測試涵蓋偏移來源造成舊模式累積超過 3 m、anchored 逐步保持總邊界、nominal 高度改變、單步 dz 超過舊邊界但總 offset 合法，以及無重疊 fallback。合成測試僅驗證機制，實測證據是上述 1792 筆。

既有 4 項 Python geometry-report 測試通過；修改 Python／launch 的語法檢查與 `git diff --check` 通過。建置保留原 PCL_ROOT／CMP0074 開發警告；CPU 環境與原始資料不變。

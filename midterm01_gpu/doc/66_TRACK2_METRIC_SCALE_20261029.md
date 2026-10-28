# Track2：因果公尺尺度與地圖配準 v3

Track2 的 v2 路線看來平順，但每幀尺度固定為 3.0，地圖幾何修正很小，不能說點雲已對上 LiDAR 地圖。v3 讓當幀地面提供相機高度與尺度起點，再用當幀非地面結構估計尺度及地圖平面內的位置、方向。GNSS／IMU 仍用作逐幀前向 prior，不當作真值；每幀輸出後不使用未來資料回修。

一般啟動命令：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
./scripts/dev.sh up
./scripts/run_track2.sh --full
```

`run_track2.sh` 會自動開啟 RViz。參數集中在 [track2/parameters.py](../track2/parameters.py)，目前可觀察實際 scale 與 map／ground-only 狀態。本文最後一節記錄本次單次 full 啟動授權；成功與否以該新 run 的 `test_info.json` 為準。

## 觀察與原因假說

v2 把影像重建 PLY 的尺度固定乘上 3.0，因此尺度不能隨影格改變。當時已停止的 full 共輸出 1944 幀，其中 1797 幀標示 map update accepted；既有觀察顯示平移／角度修正多為毫米及小角度量級，幾何支援的有效 rank 多數為 1。這些欄位只能說明 solver 的接受狀態與局部幾何條件，不能證明點雲已與地圖重合。

一個符合現有證據的原因假說是：固定尺度排除了尺度誤差的修正自由度，而退化或單一方向幾何又限制平移修正，GNSS prior 因而維持了平順路線，但不能單獨保證點雲對齊。v3 加入地面高度尺度訊息，並讓非地面地圖結構參與同一個當幀尺度解。此假說仍須以 full 結果及獨立精度資料再評估。

## v3 每幀怎麼估計

1. 使用 Track2 官方前視相機 mask 與 rectified FOV 篩選當幀 PLY；讀取當幀資料及局部 LiDAR 地圖 XYZ，不利用地圖時間欄還原路徑。
2. 在 source PLY 與 map XYZ 分別擬合地面平面。兩平面的法向基底提供 roll／pitch（橫滾／俯仰）方向；source ground offset 是相機原點至當幀地面的 raw-unit 距離。
3. 相機對地面公尺高度起始值，由官方初始 pose 投影到局部 map ground 得到，不以 GNSS 高度或 v2 固定尺度累積出的高度代替。尺度起點為

   `scale_seed = camera_height_m / source_ground_offset_raw`

4. 以地面基底表示點雲後，擷取非地面 source／map 結構；按約 4° angular cell 給深度觀測聚合權重，避免把同一方向的大量影像點都當成獨立量測。GPU nearest-neighbor 對應與 Cauchy robust cost 用於求解 `[plane_dx, plane_dy, yaw, log_scale]` 四個變數。地面基底提供傾斜方向與高度關係，因此不把 roll、pitch、Z 再當成無約束自由度。
5. GNSS XY／yaw 是有雜訊的軟 prior。非地面 feature registration 通過接受條件時，當幀輸出 `feature_metric_update`；只有尺度資料資訊足夠（邊際化其他參數後的 `scale_log_std` 不大於門檻），才用該幀結果前向更新高度 anchor，供之後影格使用。
6. 若地面可用但非地面 feature registration 未通過，當幀可輸出 ground-based prior，但狀態標成 `ground_only_prior`、`scale_validated=false`。地面支持不足時回退到 sensor prior 及先前／未驗證尺度；固定 `PLY_SCALE=3.0` 只保留作沒有可靠地面時的未驗證 fallback。

全流程逐幀前進，不使用 RTS／backward smoothing、置中濾波、未來 frame SLERP 或定位後軌跡修正。每幀 pose 只輸出一次；尺度與 map 接受狀態分開記錄。RViz 的 `MAP UPDATE` 不等於尺度已驗證；`GROUND PRIOR` 也不表示非地面地圖 feature 已成功接受。

## 尺度和配準參數

以下是現行程式中的數值；它們是明示的工程先驗、搜尋界線及品質門檻，**不是相機／GNSS 標定結果，也不是經榜分挑出的最佳參數**。直接在 [parameters.py](../track2/parameters.py) 修改。

| 項目 | 目前值 | 用途與限制 |
| --- | ---: | --- |
| `PLY_SCALE` | `3.0` | 地面初始化失敗時的未驗證 fallback；非每幀固定尺度 |
| `METRIC_MAP_LEAF_M` | `0.25 m` | 地圖降採樣尺度 |
| `METRIC_SUBMAP_HALF_EXTENT_M` | `(60, 60, 20) m` | 局部 map XYZ 範圍 |
| `GROUND_RADIUS_M`／`GROUND_GRID_M` | `18 m`／`0.6 m` | map 地面候選半徑及網格 |
| map／source ground threshold | `0.15 m`／`0.015 raw` | 地面平面擬合距離門檻 |
| map/source ground MAD 上限 | `0.08 m`／`0.10 m` | 地面厚度品質門檻；source 至少 100 點 |
| 相機高度範圍／初始標準差 | `0.5–3.0 m`／`0.15 m` | initial pose 高度合理性界線及工程先驗，不是標定精度 |
| 高度 random walk／觀測下限 | `0.01 m/√s`／`0.05 m` | 前向高度 anchor 不確定度模型 |
| source／map 非地面最低高度 | `0.02 raw`／`0.10 m` | 排除地面附近的 feature |
| `FEATURE_MAX_RANGE_M` | `40 m` | source feature 有效距離上限 |
| `ANGULAR_CELL_RAD` | `0.07 rad`（約 4°） | 每個 angular cell 聚合深度權重；相關性模型未校準 |
| feature／位置／yaw 標準差 | `0.3 m`／`1.0 m`／`3°` | robust feature cost 及 GNSS／未知 lever 偏差的工程先驗 |
| XY／yaw 搜尋界與高度變化界 | `3 m`／`8°`／`0.6 m` | 當幀求解限制，不是量測準確度 |
| `SCALE_BOUNDS` | `(0.2, 30)` | 尺度求解範圍 |
| 配準 stages | `(3 m, 1 m, 40)`、`(1.5 m, 0.5 m, 40)`、`(0.75 m, 0.3 m, 40)` | 每階段 correspondence radius、robust 尺度及迭代上限 |
| 接受門檻 | 至少 30 matches、coverage `0.6`、median distance `≤0.5 m` | 與收斂、cost 及參數有效性共同決定 feature update |
| `MAX_SCALE_LOG_STD` | `0.15` | 控制是否以當幀非地面資訊更新後續高度 anchor |

source PLY 軸向、相機高度起點的偏差、GNSS 座標 frame／時間差／camera lever arm、IMU 到 PLY 的旋轉與零 lever 假設都未正式標定。地面可能不平、受遮蔽或有錯誤點；上述高度 prior 和 angular cell 權重也未經獨立校準。full 中若出現 `GROUND PRIOR` 或 `UNVALIDATED`，應按 observation 診斷其原因，不能把 fallback 說成 map feature update。

## 三幀真實資料對照

使用者明確授權的 frame 0、500、1000 三幀，取自已保存的 v2 causal sensor prior 與相同真實 PLY。新版由官方初始 pose 建立高度起點；對照只評估這三幀，不是連續短段，也不使用未輸出的其他真實幀。表中 baseline 為固定尺度 3.0 的 v2 pose；兩側以同一組 raw feature、angular weights、map 與 `0.75 m` final search cutoff 評分。

| frame | v3 scale | feature median capped distance（v2 → v3, m） | feature cost（v2 → v3） | coverage（v2 → v3） | v3 map／scale 狀態 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 7.20918 | 0.19896 → 0.15333 | 0.082178 → 0.036086 | 0.9635 → 0.9434 | accepted／validated |
| 500 | 8.95008 | 0.44776 → 0.18276 | 0.088960 → 0.035096 | 0.9647 → 0.9874 | accepted／validated |
| 1000 | 8.45818 | 0.50862 → 0.15742 | 0.126300 → 0.033682 | 0.7133 → 0.9884 | accepted／validated |

此固定樣本比較的 median distance 與 feature cost 三幀均下降；frame 0 coverage 則由 `0.9635` 降至 `0.9434`，需如實保留。上述是與同一份地圖幾何的一致性指標，不是 ground-truth 軌跡誤差或競賽分數；三個點通過 solver gate 也不能證明 3299 幀均準確。這三次求解 wall time 約 `0.957`、`0.461`、`0.438 s`，不是 full 每幀總耗時或平均性能。

可追溯資料：

- [三幀對照 JSON](../outputs/run/track02_test_2610290233/comparison.json)、[對照分析與圖表](../outputs/run/track02_test_2610290233/analysis/README.md)
- [18 項合成／機制檢查 XML](../outputs/run/track00_test_2610290233_0002/pytest.xml)：18 passed，8.09 秒；有一項既有 Matplotlib `Axes3D` warning
- [已停止的 v2 full run](../outputs/run/track02_full_2610290210/test_info.json)：1944 幀輸出與 source SHA 保留；不能把它當作 v3 結果

## 本次 full 授權與結果判讀

使用者本次明確授權助理在必要檢查後、於已開啟的 Ubuntu terminal 啟動 **一次** Track2 full。是否實際啟動成功、完成幀數、map 接受數、輸出 SHA 與圖表狀態，以新建 run 的 `test_info.json` 和 `analysis_status.json` 為準；run ID 應補記於本節。這次授權不延伸到之後的 full 重跑。

full 完成後自動生成的 `analysis/` 是唯讀視覺化，含軌跡、pose 變化、配準時間及融合／尺度圖與來源 manifest。若 full 中斷，保留部分輸出與停止狀態；未完成 run 不應描述為完整 Track 結果。無論輸出長度、接受率或圖表如何，都不能替代獨立 ground truth 或正式評分。

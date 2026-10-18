# 三相機完整成績、原因診斷與改進順序（2026-10-18）

**兩條三相機完整序列都沒有超過既有前相機最佳。** 已完成輸出核對、6310 幀配對診斷、兩條各 16 個分散場景的 camera-level probe，以及 Track1 的 24×12 幀消融。可以確認側面有增加幾何約束，但新的整套重建與地圖存在不同的深度／高度偏差；目前優先修正幾何一致性，保留原前相機最佳。

## 成績與來源

| Track | 前相機最佳 | 三相機 | 分數增加 | 相對增加 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.15275 | **1.15346** | +0.00071 | +0.0616% |
| 2 | 0.30576 | **0.31064** | +0.00488 | +1.5960% |

分數越低越好。來源為使用者 2026-10-18 回報的提交截圖；Track2 上傳名為 `result_2_tricam.csv`。本地候選依使用者 IDE 路徑及上下文對應，截圖本身沒有 SHA256；沒有登入查榜或代為上傳。總分無法分解出平移／旋轉各自改善或惡化多少，也沒有逐幀真值可定位是哪個片段造成分數增加。

使用者完整 run：`outputs/runs/three_camera_full_fast_imported_2610181301/`。

| 檔案 | 筆數 | SHA256 |
| --- | ---: | --- |
| `track1/result_1.csv`／根目錄副本 | 3011 | `9c37bf8618eff830eee1c20f0279bddaf997e95b19b059482b5a3ef4c6b63276` |
| `track2/result_2.csv`／根目錄副本 | 3299 | `70949f80182b7fad0ca2f8b36bd6a13667c3b02d1c383c24991d94131e44c1db` |

兩份表頭、idx 0 起連續順序、有限數值、CSV 與 3×4 TXT 的 XY／yaw、rotation 正交／det、根目錄副本、完整 source manifest／report SHA 全部通過。這排除了目前可檢查的檔案混用與轉檔問題；官方 sample／body-frame 契約仍需官方依據。舊最佳 CSV 沒有覆寫。

三相機使用相同共享深度尺度 **0.67398047**，只乘 camera depth，不縮放 baseline／安裝高度。該尺度先前的跨場景檢查仍未通過，見 [24｜偏差驗證](24_BIAS_VALIDATION_20261018.md)。GPU 加速僅改運算實作，先前小段選點／PLY／定位一致性已驗證，見 [25｜效能報告](25_RIG_PERFORMANCE_20261018.md)。

## 完整輸出揭露的作用機制

原始 sensor prior bytes、幀 timestamp、實際套用的 XY／yaw prior 完全相同。兩版都每幀重新錨定 GNSS／IMU，Z 則沿用上一幀 map-refined 高度。因此「同一份 prior CSV」不代表兩版後續使用的 Z 也相同。

| 指標（完整序列中位數） | Track1 前相機 | Track1 三相機 | Track2 前相機 | Track2 三相機 |
| --- | ---: | ---: | ---: | ---: |
| C++ voxel 後 input points | 1255 | 5069 | 641 | 4165 |
| 距離 ≤2 m 的 NN inlier 比例 | 94.91% | 70.09% | 100.00% | 90.68% |
| 符合 map PCA 門檻的可靠比例 | 32.59% | 21.04% | 54.16% | 40.93% |
| XY map correction | 0.408 cm | 1.091 cm | 0.267 cm | 1.193 cm |
| yaw map correction 絕對值 | 0.0131° | 0.0518° | 0.0064° | 0.0579° |
| 消去 Z 後最弱 XY／yaw normal information | 0.000741 | 0.035549 | 0.000351 | 0.031031 |
| 接受地圖更新／總幀數 | 3011／3011 | 3011／3011 | 3047／3299 | 3299／3299 |

information 是 data-only 局部 normal 指標，排除 prior、normal gain 與 tangent floor，不是 covariance 或定位精度。三相機雖然有效匹配**比例**下降，可靠點與 unique targets 的**絕對數量**及方向約束仍有增加，不能寫成「三相機完全沒有可靠點」。

| 三相機相對舊版的 pose 差異 | Track1 | Track2 |
| --- | ---: | ---: |
| XY 差異中位數 | **1.487 cm** | **1.419 cm** |
| yaw 絕對差異中位數 | **0.0703°** | **0.0621°** |
| XY 差異 <5 cm 的幀比例 | 91.70% | 91.82% |
| map Z 差異中位數 | **+2.127 m** | **+1.268 m** |
| map Z 差異 RMS | 2.366 m | 1.300 m |

這些是版本差異，不是誤差。兩版 XY／yaw 都主要接近同一 prior，可解釋為何整套來源改動後總分仍接近。Z 的來源偏差可以經由對應點與法向耦合到 XY／yaw，即使提交 CSV 沒有 Z，也會影響求解。

![完整配對差異](../outputs/runs/rig_track1_diagnosis_2610181322/paired_trajectories.png)

### 有效權重與限幅

目前目標的主要形式為：

`L(δ) = 0.5 δᵀΛδ + [1 / (N × 0.3²)] Σ_i robust_map_loss_i`，`δ=(dx,dy,dz,dyaw)`。

prior 使用 XY σ=0.5 m、Z σ=1 m、yaw σ=3°；directional normal gain=2。地圖項依**所有來源點數 N**正規化，失配／不可靠點保留常數 outlier loss，但不提供更新梯度。這避免優化器靠丟失匹配降低成本，同時也表示加入更多不可靠點會稀釋可用梯度，點數增加不會自動等比增加可信資訊。

由診斷矩陣估算的 normal-only information／prior 對角中位數，三相機 XY 約為 Track1 **5.7%／6.5%**、Track2 **3.8%／7.7%**，scaled yaw 約 **24.5%／21.4%**。這個代理值不含 tangent contribution，robust 權重也不完全等於求解 Hessian，不能稱為實際 posterior 比例；實際小幅 correction 則直接見於完整輸出。

兩條 Track、兩個版本都沒有碰到 1.5 m XY 或 8° yaw 的限幅。**目前證據不支持把問題歸咎於 max correction 太小。** 放大 gain 或放鬆 prior 可能把有偏的深度放大成更大的錯誤；應先證明 source 幾何可靠。

Track2 原版拒絕地圖的 252 幀，在三相機全數接受；這批 XY 版本差異中位數約 0.618 cm、yaw 約 0.0217°，比原版已接受的 3047 幀差異還小。不能僅以「接受更多幀」解釋分數好壞，亦不能斷言退步全由這 252 幀造成。

## 高度反事實診斷：大部分地圖殘差變化來自 Z

依路線累積距離的 5–95% 等分位，各 Track 固定抽 16 個場景，不按殘差或分數挑幀。用同一份 old-front／新 front／新 left／新 right 點作 common probes，分別放到兩版完整 pose，再只交換 pose 的全局 Z，保持 XY 與 rotation 不變。這是獨立 CPU exact NN／PCA 診斷，使用保存的選點、位於 C++ 第二次 voxel **之前**，不冒充逐位元重播 solver。

下表是各場景 capped NN RMSE 的平均，單位 m，僅描述與同一 1 m map 的一致性：

| Track／probe | 舊完整 pose | 三相機完整 pose | 舊 XY／yaw ＋三相機 Z |
| --- | ---: | ---: | ---: |
| 1／old-front | 0.933 | 1.307 | 1.306 |
| 1／新 front | 1.418 | 1.279 | 1.280 |
| 1／新 left | 1.383 | 1.562 | 1.562 |
| 1／新 right | 1.525 | 1.077 | 1.078 |
| 2／old-front | 0.531 | 0.996 | 0.996 |
| 2／新 front | 1.349 | 0.894 | 0.895 |
| 2／新 left | 1.314 | 1.087 | 1.088 |
| 2／新 right | 1.238 | 0.972 | 0.972 |

只交換 Z，幾乎重現所有 probe 的變化，支持「深度／frame／高度耦合」是主要待查機制。**不能由此決定哪版 Z 是真值，更不能直接加固定 −2 m。** Track1 新左相機在三相機 pose 反而較差；單一全局 Z 並沒有同時改善所有 view。可能來源包括 view-dependent 深度形狀、動態物體、地圖覆蓋、source frame 或同步／外參偏差，尚未分離。

固定 rig 安裝高度只表示左右相對前相機高 7.8／10.0 cm 不變；並沒有解決 source→body 的未知外參，也不保證網路深度與 map Z 一致。[MapAnything](https://github.com/facebookresearch/map-anything) 接受 calibration／pose 作幾何輸入；本程式固定外參反投影可保住合併契約，但尚未以跨相機／跨時間 reprojection 對預測 depth 施加共同靜態場景約束。

![只交換高度的診斷](../outputs/runs/rig_track1_diagnosis_2610181322/height_counterfactual.png)

## 同一份 depth 的側面消融

Track1 沿路線距離 20／40／60／80% 固定選四個 12 幀 window：852、1392、1877、2431。新增來源都從**這次完整重建的同一批保存點**取子集；不重新推論、不改尺度／外參。全部使用相同當前 Release binary、相同 window prior bytes、map cache 與 solver 設定，24 runs×12＝**288 poses** 全部有限、完整、退出碼 0，地圖更新全接受。

| 來源 | C++ input 中位數 | 消去 Z 後最弱信息中位數 | old-front probe | new-front probe | new-left probe | new-right probe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原 raw front | 1158.5 | 0.000856 | 1.007 | 1.396 | 1.439 | 1.441 |
| 新 depth 的 front-only | 1890.5 | 0.015023 | 1.214 | 1.280 | 1.498 | 1.145 |
| 新 depth 的三相機 | 4952 | 0.028941 | 1.224 | 1.289 | 1.506 | 1.133 |
| 新 depth 的 side-only | 3119.5 | 0.033196 | 1.205 | 1.284 | 1.475 | 1.170 |
| 新 front，匯出 600 點 | 600 | 0.013256 | 1.209 | 1.278 | 1.494 | 1.146 |
| 新三相機，匯出 600 點 | 599 | 0.029782 | 1.205 | 1.281 | 1.487 | 1.155 |

probe 數值仍不是定位誤差。C++ voxel 會合併重疊點，所以 600 匯出不等於恰好 600 有效 input。

可以確認：同 depth 的 front-only→三相機增加方向資訊，近同點數 600 控制也保留這個效果。舊 raw-front→新 front-only 就已改變幾何及 common-probe 行為；正式分數比較同時更換重建、尺度、取樣與相機數，不能將退步全歸因於側相機。

每個短 window 都重新初始化高度，無法重現完整序列累積的 Z。它們用於理解來源與側面機制，不能當完整序列 ATE 或下一份提交的勝出證明。完整舊／新 localization binary SHA 不同，既有紀錄指出後續拒絕原因診斷修正；本次 24 組消融則核對了**相同 binary**，沒有忽略這個差異。

## 改進順序與可驗證假說

### 1. 先核對 source／body 與高度契約

核對原 raw_ply 的原點、三相機 front-FLU 的原點、init_guess／提交 pose 的官方 frame，以及 GNSS／IMU→camera lever arm。目前只有 camera-to-camera 外參；不能從 2.13／1.27 m 的版本高度差反推真實相機高度或自行補固定偏移。若 frame 契約確認一致，再把尺度與深度形狀視為主要改善對象。

驗證：已知固定外參／同時基、同一靜態地標的跨 view 投影、平移單位及 optical→FLU 只套一次；保存官方來源。跨 view 不能同時解釋同一幾何時，先查原因，不提高側面權重。

### 2. 固定 rig，加入跨時間靜態一致性與共同深度尺度

現在每個三相機 group 獨立推論，稀疏 stereo 的全局倍率無法修掉非剛體深度形狀。下一個重建候選應保持 calibration baseline／相對高度固定，追蹤跨相機及跨時間的靜態特徵，以共同 3D 點的 reprojection residual 聯合估計短 window 的共享尺度／深度修正，並對 sensor motion 使用有不確定度的先驗。移動車輛／植被、遮擋、小視差不能直接當 metric 錨點。

在理想 rectified stereo，`Z=fB/d`，因此 `σ_Z ≈ Z² σ_d/(fB)`：遠點深度的不確定度快速增大。原本所有有效點幾乎共享 0.3 m 尺度的地圖噪聲，沒有反映這個差異。可用多視角／跨時間一致性與三角化條件估計深度可靠度；network confidence 尚未校準，不能直接當 inverse variance。

驗證：fit／holdout 依路線場景分離，兩條 Track 都要有支持；共同靜態點的 reprojection、尺度一致性、殘差隨距離的趨勢及左右 view 是否同時改善。只有可觀測場景才估尺度，支援不足時保留前相機，不為了通過而調鬆門檻。固定 rig 的建模可參考 [COLMAP rig 文件](https://github.com/colmap/colmap/blob/main/doc/rigs.rst)，不因此更換目前工具鏈或宣稱已完成 bundle adjustment。

### 3. 保留已評分前相機，側面作有品質門檻的增量

先保留原 raw-front 幾何作 anchor，側面通過獨立靜態一致性後再補不足的 XY／yaw 方向。這樣能隔離「新前相機深度的偏差」與「側面是否有幫助」，不必再次把整套來源一併替換。

實作時需讓 camera_id／來源不確定度進入 solver，並在 voxel 合併時一起保存或合併標籤。現在 `labels.npz` 僅供保存與診斷，C++ PLY 路徑沒有使用來源 camera_id／network confidence；來源均衡取樣不能代替配準階段的品質加權。不可在 PCL voxel 後直接沿用原 labels 索引。

可採固定於候選 pose 優化之外的來源權重，正規化各 camera 的有效支持，並保留失配常數代價，避免靠丟失匹配降低 objective。normal residual 的不確定度可依 `nᵀ R Σ_depth Rᵀ n + σ_map²` 設計；重疊／相關點要控制有效資訊量。權重需由靜態一致性與噪聲估計支持，不能因某 Track 的分數較好而指定不同 camera 權重。

驗證：同一可信 source 下做 old-front、old-front＋side、同點數版本，檢查每 camera 的可靠匹配、共用 probes、消去 Z 後 XY／yaw information，以及 prior→posterior 的更新。改善所有預定 holdout 診斷後，才由使用者手動跑完整評分。

### 4. 幾何通過後，再研究先驗與跨幀聯合求解

目前 XY／yaw correction 小，放鬆 prior 的確可能讓地圖影響更大，但同樣會放大 depth bias。先驗 σ、normal gain 及 robust cutoff 的下一輪改動，應依感測器噪聲、校正與 held-out residual 定義；不要由這兩個總分盲目搜尋。

若單幀幾何已可靠，再研究帶 sensor factors、相鄰運動與 map factors 的 joint smoother，減少每幀弱約束與 carried-Z 偏差。這屬後續算法改動，本輪沒有實作或評分。

## 已完成與尚未完成

已完成兩條完整輸出檢查、6310 幀配對、32 個分散場景的相機級／高度反事實診斷、24 組短段定位消融；新增的是分析工具與診斷來源，沒有改 C++ solver、重建尺度、現在推薦設定或舊最佳 CSV。個人紀錄與使用者文件重新編號保留。

新的 source-frame 校準、跨幀共同重建、深度不確定度融合及 joint smoother **尚未實作，沒有改善分數證據**。現階段推薦最佳仍是前相機 `sensor_guided + heading_rts + directional`。

完整證據目錄：`outputs/runs/rig_track1_diagnosis_2610181322/`。

- `track1/`、`track2/full_comparison.json`／`paired_frames.csv`：兩條完整配對；`full_outputs_validation.json`：CSV／矩陣／manifest 核對。
- `camera_probe.json`、`height_counterfactual.json`、`track2/camera_probe.json`：固定距離場景／共同 probe／交換 Z。
- `ablation_plan.json`、`ablation_runs.json`、`ablation_summary.json`、`ablation_sources/`：24 組消融來源、命令、log、結果；首次缺少 motion metadata 的前處理失敗另保留，未冒充定位結果。
- `sources/`：分析版本快照；`frozen_running_sources.json`／後續保存檢查：原重建來源碼未動；圖表 PNG／SVG 與原始 JSON。

工具：`scripts/analyze_rig_track1.py --track {1,2}`、`probe_rig_track1_geometry.py --track {1,2}`、`summarize_rig_diagnosis.py`。名稱保留 Track1 起源，但已支援兩條。使用主機 ML venv 執行；資料與 outputs 仍被 Git 忽略，跨機需另行保存。

# Track2 v5.13：影像相對幾何與地圖方向診斷

> 2026-11-04 更新：使用者已完成 v5.13 full `track02_full_2611041345`，回報分數 **0.42933**。本頁保留候選套用時的設計及測試紀錄；下文「尚無真實定位／分數」只描述當時狀態。現行 v5.14 修正、影片來源核對與完整診斷見 [91｜影像深度與速度契約](91_TRACK2_V514_RAY_SPEED_20261104.md)。

v5.13 候選想補上兩種 v5.12.1 常缺的資訊：地圖配準不接受時仍可使用的可信影像深度，以及能描述平移向量或相對航向的影像觀測。它也記錄 map 幾何實際提供哪些方向，方便分辨「點很多」和「有足夠幾何資訊」的差異。

**目前狀態：**此版本 `track2_v510_image_relative_geometry_v5_13_perf2` 已凍結於 `outputs/run/track00_test_2611041306/candidate_source/` 並套用至正式入口。v5.10 full `track02_full_2611031941` 的 Public Score `0.42699`（使用者回報）是目前最佳已知完整成績；v5.11、v5.12、v5.12.1 的歷史分數與狀態仍見 [Track2 README](../track2/README.md)。本候選沒有新分數、沒有真實定位，也沒有真實影像 PnP 結果。此版本回到 v5.10 基線，因此不帶入 v5.12.1 的 GNSS 步長限幅；不宣稱已改善首筆 GNSS 跳動。

## 為什麼加入影像相對運動與幾何診斷？

v5.12.1 的既有逐幀分析顯示，rank 0／1 合計占 90.57%；frame 1000 之後沒有成功的移動影像速度更新。472 個影像嘗試時點中，248 次因沒有可信的公尺尺度地面錨點（metric ground anchor）而在 PnP 前停止。這些是舊 full 的診斷，不是 v5.13 的結果。[前輪分析與圖表](../outputs/run/track00_test_2611041207/analysis/README.md)

舊輸出也顯示增加點數未必增加可觀測方向：frame 1833 有 3700 點，但 coverage 約 0.051；附近 31 幀中 29 幀為 rank 0、只有 1 幀接受 map。較穩定的 frame 1532 有 1406 點、coverage 約 0.239；附近 31 幀中 26 幀接受 map。這支持檢查角度與表面方向的分布，不支持單純增加點數或放寬接受條件。

## v5.13 如何使用這些資訊？

### 1. 將高度記憶品質與 map pose 接受分開

影像深度 anchor 由來源地面幾何與因果高度記憶建立。尺度以 `height / ground_offset` 計算，並檢查高度、尺度、地面法向、內點數、地面離散程度及 log-scale 不確定度。只有這組檢查通過，才從過去 PLY 建立 metric depth 和可見地面 mask。

這個檢查獨立於當幀 map pose 是否接受，也不要求當幀尺度標記為已驗證。因而 map 更新暫時失敗時，只要地面與高度記憶本身仍可信，後續影像仍可能取得相對運動所需的深度參考。品質檢查不通過時不建立可信 anchor；它不會把失敗的 map pose 當作成功結果。

### 2. 以完整畫面特徵和過去 PLY 深度估相對運動

影像追蹤使用空間分布的完整畫面特徵，不只追道路地面。前一張影像中的特徵會和前一個 PLY 的 CUDA 深度投影配對，再以相同的 3D 點／2D 像素對照兩種模型：自由旋轉和平移的 PnP，以及使用 gyro 旋轉、只解平移的模型。兩者的內點、像素殘差、分布、幾何秩和不確定度都會記錄。

預設以 gyro-fixed translation 產生 `vector_velocity`；相對航向通道使用 PnP 的旋轉。當平移與相對航向兩個通道都啟用時，消耗的影像區間交替提供其中一種量測，避免同一區間同時把兩種結果當成獨立證據。若一次嘗試沒有形成量測，可保留 anchor 延長間隔；已消耗的區間不重複融合。品質 gate 維持既有設定。

這些特徵只是「空間分布良好的 dominant rigid-motion consensus」（主要剛體運動一致群），沒有語意辨識保證；它不保證該群一定是靜態背景。內點多或重投影誤差小，也不能證明地圖座標中的運動準確。

### 3. 相對 yaw 只約束 bias 變化

相對 yaw 透過 Schmidt clone（保留相關性的舊航向錨點）比較目前與影像 anchor 的航向 bias 變化。這只能觀測兩個時刻間的 bias 差，無法辨識共同且固定的 absolute map-heading offset，因此不把相對影像旋轉當作絕對地圖航向。

GNSS course 若校準共同航向 bias，程式會同步重表達目前影像 anchor 的參考方向，維持兩個時間點的比較一致；它不會改寫已輸出的舊 pose。所有量測仍按當下時間因果更新。

### 4. 保留稀有視角，並記錄幾何資訊

超過原本 6000 點上限時，map 特徵改用角度分層抽樣，讓較少見的視角格也有機會保留；點數上限不增加。抽樣不保證法向獨立或幾何秩提升。

幾何診斷的 normal Gram matrix（法向方向資訊矩陣）只使用 plane matches，並以 robust weights 加權；有效方向與特徵值、主要法向權重因此只描述這部分平面支持。line-feature 數量另列；分距離帶的 signed residual median 也只統計 plane matches。沒有可用法向時有效方向記為 0，主要法向權重為 null。solver 每步另記固定關聯集合成本（fixed-association cost）、匹配變化和資料 Hessian（data Hessian）。這些欄位用來辨認方向不足、匹配切換和數值收斂狀況；不放寬 gate，也不是定位精度指標。

影像 anchor 診斷會分開記錄「前一個可信 anchor 是否仍保留」和「當前地面量測是否被拒絕」；如果沿用舊 anchor，影像圖的 `anchor_available` 會依舊參考是否仍可用判定，而不會把當前地面拒絕誤顯示成完全沒有 anchor。

## 版本開關與不確定度

v5.13 保留 v5.10 其餘模型，候選開關為 `V511_ENABLED=False`、`V512_ENABLED=False`、`V5121_ENABLED=False`、`V513_ENABLED=True`。v5.12.1 的步長限幅分支不帶入本候選。

高度、gyro、影像深度與模型 floor 的不確定度會傳到相對運動觀測；程式也保留外參與 lever-arm 的工程誤差預算。但高度、gyro、深度及影像之間的共用誤差尚未由完整 joint estimator 校準，不能把各項 covariance 當成彼此獨立且已校準的真實誤差。

## 驗證狀態與下一步

- `tests/test_track2*.py` 最終 GPU suite 為 **270 passed、1 項既有 Axes3D warning、41.93 秒**。其中 v5.13 新增 23 個機制案例（CPU 15、CUDA 8），正常與歷史 API 案例均通過。測試輸出：[pytest_gpu_final.xml](../outputs/run/track00_test_2611041306/pytest_gpu_final.xml)。
- 凍結了 128 個 Python source SHA：[candidate_python_manifest.json](../outputs/run/track00_test_2611041306/candidate_python_manifest.json)。
- 生成相容性檢查通過：147 個既有 v5.10 數值參數相同（僅 `VERSION`／`BASELINE_RUN_ID` metadata 不同）；關閉 v5.13 時，3 個生成 registration frame 的 pose／scale／height／height variance／accept／correction covariance 完全相同，201 個生成 sensor frame 的 pose／state／covariance／heading bias／heading covariance 完全相同。anchor-only、relative-only、geometry-only、all-on 四個獨立 ablation 場景也通過。這些是生成資料與程式相容性檢查，不是真實 Track 定位或精度證據。
- 生成 checker 已通過；新影像開啟 runner 的 7 幀與 11 幀生成前綴比較中，CSV／TXT／`pose_trace` bytes 相同。相對融合的 11 幀與 29 幀生成前綴比較中，Python pose、covariance 和 decision records 的數值前綴相同，該項沒有產生 CSV／TXT。這些只檢查程式相容性與短 prefix 輸出，不代表完整 Track 精度。
- 本輪真實定位：0 幀；真實 PnP：0 次；新分數：無。
- 本輪測試 run 的分析與圖表：[analysis README](../outputs/run/track00_test_2611041306/analysis/README.md)（含三組生成機制 PNG／PDF 與來源 manifest）。

v5.13 已套用並完成必要檢查；如要開始完整定位，由使用者手動執行：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
./scripts/run_track2.sh --full
```

新的 full 若有相應診斷資料，會在該 run 的 `analysis/` 增加 `12_image_relative` 與 `13_geometry_diagnostics` 圖表；缺少診斷時不產生該組圖。舊 full 保持原樣，不補畫新圖，也不為產圖重跑定位。

## 歷史證據範圍

v5.12.1 的 rank、coverage、影像 anchor 缺口及 frame 1532／1833 對照均來自已完成 run 的唯讀分析。它們說明加入這些機制的動機，不能當作 v5.13 已改善定位、提高準確度或取得新分數的證據。

## 模型附錄：量測真正約束什麼

令因果高度記憶為 `h`、方差為 `V_h`，來源地面到相機原點的 offset 為 `d_s`，raw plane MAD 為 `m`。影像 anchor 使用

```text
s = h / d_s
V_log_s = (V_h + HEIGHT_OBSERVATION_FLOOR_M²) / h² + (m / d_s)²
```

其中 MAD 是共用地面模型不確定度的保守代理，沒有除以點數假裝成數千個獨立深度。原地面法向 floor、MAD、inlier count、height memory variance 和 model floor 都隨 anchor 保存；只在原先的尺度不確定度與物理範圍內使用。

相對模型輸入是過去 FLU 公尺點 `P_a` 與現在 optical 像素 `u_k`，重投影殘差為 `u_k - project(R_ka P_a + t_ka)`。反轉後得到 `T_a_k`（現在 FLU 到過去 FLU）；PnP 自由求旋轉／平移，gyro 模型固定已積分的 `R_a_k`、僅求 translation。旋轉使用 SO(3) 逐時累乘，非把三軸角度或 gyro 大小當 yaw。兩模型以相同 track 的 capped pixel cost 比較，並保留其共用尺度／gyro 與 pose covariance。速度量測為

```text
v_camera_map = R_map_a * translation(T_a_k) / delta_t
v_antenna_map = v_camera_map - R_map_body * (omega_body cross lever_body)
```

相對 yaw 的觀測是視覺轉角與未加 heading bias 的 IMU 相機轉角之差，殘差為 `z_yaw - (b_now - b_anchor)`。狀態 `[b_IMU_now, b_GNSS_now, b_IMU_anchor]` 對應觀測矩陣 `[1, 0, -1]`；保留 current/anchor cross covariance，Schmidt nuisance anchor 的 gain 為 0。它不觀測共同 absolute-heading gauge。實際 gain 限幅後以同一 gain 執行 Joseph covariance 更新，並共用 v5.10 每幀航向資訊預算。

向量速度與相對 yaw 使用不重疊區間，能避免同一批像素被當成兩筆獨立量測；相鄰區間仍可能共享高度、深度模型與端點誤差，因此這不是完整跨區間相關性校準。gyro translation 的旋轉由 gyro 指定，不能把它再當獨立影像 yaw 量測。

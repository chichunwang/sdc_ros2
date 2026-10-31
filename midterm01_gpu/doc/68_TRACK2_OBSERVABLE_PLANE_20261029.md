# Track2 v5：以可觀測平面方向更新地圖修正

本文件說明現行 `track2_causal_observable_plane_v5`：逐幀 covariance、可觀測 point-to-plane 量測、因果航向與解耦限幅已實作。47 項合成／CUDA／輸出檢查通過；完整定位效果與分數仍待確認。沒有使用 noncausal reference CSV 調參。

相關 API 在 [`causal_prior.py`](../track2/causal_prior.py)、[`plane_geometry.py`](../track2/plane_geometry.py)、[`observability.py`](../track2/observability.py)、[`metric_registration.py`](../track2/metric_registration.py) 與 [`correction_filter.py`](../track2/correction_filter.py)。

## 先看問題

地圖中的一面平牆能告訴我們點雲離牆面多遠，但沿著牆面滑動時，點仍可能落在同一平面上。這時牆面在法向提供資訊，切向卻可能沒有足夠約束。如果把點到點的三個座標差都當成可信量測，solver 可能在資料實際看不清楚的方向上仍輸出很精確的 correction。

v5 改用可靠局部平面的法向殘差，並根據資料資訊矩陣判斷 map 在哪些世界座標方向真的提供新資訊。只對可觀測方向建立 map 量測；弱方向不建立假量測，由當幀 sensor prior 和其 covariance 維持估計。這個設計可以減少「平面切向沒有資訊卻更新 pose」的風險，但不會自動校準 prior、map covariance 或外參。

## 狀態與 covariance 從哪裡來

濾波狀態是地圖座標中的 correction：

`c = [dx_map (m), dy_map (m), dyaw_map (rad)]`

在此基礎上，v5 將當幀 sensor prior 表成完整的 `3×3` covariance `C_prior`，順序是 `[x_m, y_m, yaw_rad]`。它不只保留各分量的 variance，也保留由 heading、lever 和外參傳播出的 XY／yaw 交叉項。它是工程不確定度模型，不是 ground truth covariance。

| covariance 成分 | 程式來源 | 未解限制 |
| --- | --- | --- |
| GNSS XY 與速度 | forward constant-velocity Kalman filter 的 position covariance | `GNSS_STD_M` 和加速度 PSD 是工程先驗，沒有獨立標定 |
| heading | forward heading-bias variance 與 IMU orientation／gyro uncertainty | IMU covariance 若為零、無效或非正定，就使用 unknown engineering budget；body small-angle 假設未標定 |
| GNSS antenna 到 PLY 原點的 lever | lever 名目值與 lever uncertainty 經 yaw Jacobian 傳遞到位置和交叉 covariance | 本版仍用 zero lever 名目值；預算不是實際量測 |
| IMU 到 PLY 的外參 | 名目旋轉與外參角度 uncertainty 經 Jacobian 傳遞 | 本版的 identity rotation 是未標定假設 |

本版參數的 nominal `IMU_TO_PLY_RPY_DEG` 與 `GNSS_TO_PLY_LEVER_BODY_M` 都是零；`LEVER_ARM_STD_BODY_M` 是 `(0.5, 0.5, 0.5) m`，`EXTRINSIC_ROTATION_STD_DEG` 是 `(1.0, 1.0, 0.5)°`，unknown IMU orientation／gyro 的 fallback std 分別是 `(2, 2, 1)°` 與 `(0.5, 0.5, 0.5)°/s`。這些數值只代表暫用的 error budget。GNSS course 與 GNSS position 的相關性目前沒有明確估計；bag 的全零 covariance 代表未知，不能解讀成零誤差。

## 每幀如何形成部分 map 量測

### 1. 從 CUDA map 鄰域估可靠平面

[`PlaneGeometry`](../track2/plane_geometry.py) 在 LiDAR map 鄰域對點做 PCA，檢查鄰點數、第二特徵值、厚度比例與 planarity。符合條件才提供局部平面法向量 `n` 和中心點 `p`；不可靠鄰域不產生法向量測。v5 的點雲殘差為：

`r_plane = nᵀ (q - p)`

`q` 是 source 點經當幀候選 pose／scale 轉換後的位置。殘差只量沿法向的距離；程式不在平面切向加上虛構的零誤差。點鄰接搜尋、平面 PCA 與點級資訊累加在 CUDA；小型協方差和特徵分解在 CPU。

### 2. 把四維 map 資訊換到世界座標並邊際化 scale

[`MetricRegistration`](../track2/metric_registration.py) 的資料資訊是四維，包含平面局部 `[x, y, yaw, logscale]`。透過 correction Jacobian `J`，它先轉換到世界 `[dx, dy, dyaw, logscale]`。尺度 `logscale` 是未知 nuisance variable，不能假裝已知。把世界 data information 按 pose／scale 分塊後，先以 Schur complement 消去 scale：

`I_pose = I_pp - I_ps · I_ss⁻¹ · I_sp`

其中 `p` 是 pose correction 的三個方向，`s` 是 logscale。這會保留 XY／yaw／scale 之間的交互影響，並把尺度不確定度納入 pose 資訊。`observability.py` 的實作先用 block-diagonal `diag(L, scale_std)` 正規化四維資訊再做 Schur；因 pose whitening 與 scale 變換各自位於獨立 block，這與先在世界資料座標消去 scale、再用 `L` whiten pose information 等價。

### 3. 依當幀 sensor prior covariance whiten，只選可觀測 eigen-direction

令 `C_prior = L Lᵀ` 為 [`CausalSensorPrior.for_frame()`](../track2/causal_prior.py) 輸出的當幀 covariance Cholesky 分解。將邊際化後的 pose information whiten 為 `I_white = Lᵀ I_pose L`，再做 eigen-decomposition；資訊 eigenvalue 至少達 `MAP_MIN_RELATIVE_INFORMATION`（本版預設 `1.0`，並另有相對數值門檻）的方向，才產生一列 partial observation matrix `H_obs`。概念上：

`H_obs = V_goodᵀ L⁻¹`

`V_good` 是保留的可觀測 eigenvectors。未達門檻的弱方向不會輸出 measurement row；因此 map 不會在那些方向上改寫 correction mean。NIS 和 Kalman update 只使用留下的 1、2 或 3 個 measurement dimensions，而不是把缺少的資料方向補成零殘差。

每個保留方向的資料 covariance 依 eigenvalue 反比形成，再加上投影後的位置／yaw error floor，並依 coverage 與 plane residual 膨脹：`R = (diag(1 / λ_good) + H_obs C_floor H_obsᵀ) × quality_inflation`。因此保留方向較弱時量測 variance 較大，品質差時更新也會降低權重；這些仍是局部幾何模型的近似量。

這不是已校準的 ICP covariance。data Hessian 是局部線性與 angular-cell 誤差模型；弱資訊 threshold、品質膨脹和 error floors 都尚未以獨立 ground truth 標定。Point-to-point ICP 的 inverse-Hessian covariance 解讀需要謹慎，可參考 [Bonnabel、Barczyk 與 Goulette](https://arxiv.org/abs/1410.7632)；只在可靠子空間更新的設計可參考 [Zhang 等人的 degeneracy-aware lidar estimation](https://www.cs.cmu.edu/~kaess/pub/Zhang16icra.html)。這些論文支持方法上的警覺與設計方向，不代表本 point-to-plane pipeline 的 covariance 已校準。

## 部分觀測下的 NIS 與 correction 更新

[`CausalCorrectionFilter.update()`](../track2/correction_filter.py) 對 `d` 列觀測使用 `innovation = H_obs (candidate - predicted_correction)`、`S = H_obs P H_obsᵀ + R`，並計算：

`NIS = innovationᵀ S⁻¹ innovation`

NIS 門檻隨觀測自由度選擇。它們是未校準模型的 chi-square 參考門檻，不是可靠度保證。

| 觀測自由度 | soft threshold | hard rejection threshold |
| ---: | ---: | ---: |
| 1 | 3.841 | 10.828 |
| 2 | 5.991 | 13.816 |
| 3 | 7.815 | 16.266 |

未被拒絕的量測會依 NIS、motion innovation 與品質膨脹 `R`。接受後，程式依當幀 prior covariance 的 Cholesky factor `L`，以及 whitened map measurement 所張成的可觀測方向 `V_obs`，形成狀態空間 projector：

`O = L V_obs V_obsᵀ L⁻¹`

先把未限幅 gain `K₀` 投影為 `O K₀`，再分別套用 XY 與 yaw 的 correction step limit，並再次投影：

`K_cap = O D O K₀`, 其中 `D = diag(g_xy, g_xy, g_yaw)`

這個二次投影避免 row-wise 的 XY／yaw 限幅把 gain 推出 map 真正觀測到的子空間。若部分觀測的一個幾何模式同時混合 XY 與 yaw，程式再以共同的 `gain_scale_observable` 縮放 `K_cap`，使該模式不超過同時適用的更新界限。完整秩時 `O ≈ I`，共同限制不需介入，XY 與 yaw 可維持獨立限幅。最後得到的實際 gain `K_actual` 用於均值和 covariance：

`c⁺ = c⁻ + K_actual · innovation`

`P⁺ = (I - K_actual H_obs) P⁻ (I - K_actual H_obs)ᵀ + K_actual R K_actualᵀ`

這保留 Joseph covariance update 與實際 mean gain 的一致性；任意 gain 下 Joseph 形式的背景可參考 [JPL Joseph formulation](https://tmo.jpl.nasa.gov/progress_report/42-116/116c.pdf)。實際運作、部分 observation matrix、`gain_scale_observable` 與 covariance 會記在逐幀 `correction_filter` diagnostics。

若候選缺失或沒有可觀測 map 方向，短期內 correction mean 依 v4 機制 hold 0.5 秒，`P` 隨時間／運動成長；超過 hold 後 correction 以 3 秒 time constant 衰減至零，covariance 朝 anchor covariance 放回。sensor prior 每幀仍更新輸出 pose；filter 只估計 map correction，不回頭改舊 pose。

### Heading 搜尋初始化與校正 reference rebase

首個 map update 前，metric registration 以當幀 `σyaw` 建立 `0、−σyaw、+σyaw` 三種 yaw seed。yaw search bound 會依該幀不確定度調整：

`bound = max(8°, min(20°, 2 · σyaw))`

這只用到目前 frame 的 prior covariance 和已抵達輸入，不讀未來影格；首個 map update 接受後，後續搜尋使用已建立的 correction 記憶與當幀候選。

轉向 allowance 使用兩影格之間已抵達的 gyro 角速度範數積分，而不是 GNSS course 學到的 heading bias 增量。後者是感測器方向校正，不代表車輛在該段真的轉了同樣角度，因此不應擴張 correction 的物理轉向預算。

當過去 GNSS course 在當前 frame 更新 heading bias，sensor prior 的世界座標參考也可能改變：bias 的旋轉會改變 lever 導致的 XY，並改變 yaw reference。濾波器追蹤一個 `3×3` 的局部線性 reference sensitivity `B`，初值為零。production map candidate 含 sensor soft prior，因此 candidate 本身對 prior reference 有反應，不能直接假定它是 data-only 量測。metric registration 以當幀的局部 Gauss–Newton 線性化估計 candidate sensitivity：

`B_candidate = I − J H_total⁻¹ Jᵀ C_prior⁻¹`

其中 `J` 是 correction Jacobian，`H_total` 含 point-to-plane data、sensor prior 與 height 項，`C_prior` 是當幀 prior covariance。濾波器以實際 `K_actual` 與 `H_obs` 累積自身的 reference sensitivity：

`B⁺ = (I − K_actual H_obs) B + K_actual H_obs B_candidate`

若當幀 prior calibration 變化為 `δ = [δx_lever, δy_lever, δyaw_bias]`，濾波器把目前 correction mean 重表達為 `c ← c − Bδ`；上次吸收候選的內部 motion reference 則使用儲存的 `B_candidate` 重表達為 `measurement_reference ← measurement_reference − B_candidate,last δ`。已輸出的歷史 pose 檔不會修改。map 尚未學到的方向在 `B` 中仍為零，因此新 GNSS heading calibration 不會被錯誤抵銷；失聯期間 correction mean 衰減時，`B` 也乘上同一個 decay factor，避免過期 map 記憶繼續重定錨新 prior。

只有直接 data-only 的 synthetic observation 才預設 `B_candidate = I`；那代表候選完全跟著量測參考平移，並非 production map candidate 的 sensitivity。candidate sensitivity 與 filter `B` 都是局部線性 reference 追蹤，不是新增獨立量測，也不是已校準的 prior／map joint covariance；sensor prior 和 map candidate 間仍可能有未建模相關性。

## Heading course 的 causal 改法

固定 5 m/s gate 對低速資料可能太粗略：速度閾值不能直接表示一段方向估計是否真的可觀測。v5 用尚未被採納的過去 GNSS window，從 1 秒逐步延長至 8 秒；視窗需至少 5 個 samples，並以 GNSS 位移方向相對 residual scatter 計算 course std。std 不高於 `HEADING_MAX_COURSE_STD_DEG = 3°` 且未轉彎、未出現不一致 GNSS block，才將 course 當作 heading-bias 量測。

未達可觀測性時，保留／滑動尚未採用的歷史視窗等待更多資料；已接受視窗立即清除，不會再次當成獨立資料使用。首個有效過去 course 可以在目前時間對齊 heading，之後再以 rate-limited update 估慢變 bias。若 bias 被 rate limit，程式用截幅後的 effective gain 同時更新 mean 與 bias variance。因為只在當下吸收已完成視窗，起始 alignment 不會被未來資料回寫；因果結果也可能與用完整未來軌跡估出的初始航向不同。固定 `HEADING_MIN_SPEED_MPS` 只供舊 guided 對照，不是此版正式 course gate。

## 調整位置、驗證狀態與風險

主要參數在 [`track2/parameters.py`](../track2/parameters.py)：plane reliability 門檻、`MAP_MIN_RELATIVE_INFORMATION`、heading window／std、外參與 lever budget、unknown IMU covariance fallback，以及 v4 correction hold／fade 和更新限制。先確認 source 與 frames，再一次改一個物理或統計假設；不可由 noncausal reference 的均值差倒推 yaw、時間、lever 或 covariance。

[34 項 CPU-only NoCuda 檢查](../outputs/run/track00_test_2610290544/pytest.xml)通過（約 11.6 秒）；最新完整 [47 項合成／CUDA／輸出檢查](../outputs/run/track00_test_2610290847/pytest.xml)通過（18.34 秒）。合成檢查沒有執行真實短段，不能代表定位精度或分數。

[本版 CPU 機制圖表](../outputs/run/track00_test_2610290544/analysis/README.md)已建立四組 PNG/PDF。圖中的 reference 示例使用 data-only 生成量測（`B_candidate=I`），正式程式另從當幀總資訊推導 candidate sensitivity；相關機制另有數值檢查。GNSS／IMU 座標、外參、lever 與相關性仍未完整標定。成功 full 自動產生 05 修正記憶與 06 不確定度／可觀測性圖表；這些圖表不是精度證明。

主要限制包括：map normals／neighborhood 對應可能錯誤；plane/data information 只是近似；先驗與 map candidate 可能相關，但其 cross-correlation 未完整估計；GNSS／IMU axes、zero lever、identity rotation 仍未標定；過度樂觀的 uncertainty 或連續自洽錯配仍可能影響估計。此方法的目標是拒絕不可觀測方向的假資訊，不保證更高精度或特定分數。


## 本輪完整檢查

[47 項合成／CUDA／輸出檢查](../outputs/run/track00_test_2610290847/pytest.xml)通過，耗時 18.34 秒，未執行任何真實短段。[CPU 機制分析與四組 PNG/PDF](../outputs/run/track00_test_2610290544/analysis/README.md)使用生成資料，包含不確定度、退化方向、增益解耦及 calibration reference 記憶。這些不是定位精度證明。

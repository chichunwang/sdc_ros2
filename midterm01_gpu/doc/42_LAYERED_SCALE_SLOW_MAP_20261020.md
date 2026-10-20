# 42｜分層 map 尺度估計與慢變 map 修正（2026-10-20）

## 目的與目前狀態

本研究把使用者提供的學長方法描述轉成一個可檢查的實驗流程：每幀以獨立 map-only Sim(3) 配準估計來源點雲尺度與 6-DOF 位姿；只有通過幾何留出與可觀測性檢查的幀，才可能形成低頻 map 修正錨點，交由 GNSS／IMU 批次圖估計慢變偏差與航向。學長沒有提供原始程式、逐幀尺度、ICP 對應或真值，因此這是依文字線索實作的研究版本，不能稱為精確重現。

本分支與原最佳都直接使用老師提供的 `raw_ply`，沒有自行重新執行或修改 LoGeR；官方題目第 8 頁指出前相機影像使用 LoGeR。先前三相機重建分支使用 MapAnything，兩者來源不同，本文件的尺度估計針對既有前相機 `raw_ply`。見[官方題目 PDF，第 8 頁](<../../midterm01/docs/SDC 2026 Midterm01 Competition -  Localization.pdf>)。

兩條 Track 共用相同算法與參數。這一輪以 Track2 測試，是因 Track1 上傳額度已用完；這不構成 Track 專屬模型設定。當前研究實作為 Python SciPy／`cKDTree` CPU 分支，獨立於原 CUDA `sensor_guided` 定位流程；不稱 GPU ICP，也不是完整 IMU preintegration。已評分最佳仍是 `motion_rts＋directional＋carry`：Track1 1.15008、Track2 0.30488。新分支尚無完整結果或 Kaggle 分數，最佳紀錄不變。

## 逐幀 map-only 尺度與 6-DOF 配準

令來源點為 `p_i`，來源到 map 的旋轉與平移為 `R,t`，每幀未知正尺度為 `s>0`，map 點為 `q_i`、其局部單位法向為 `n_i`。最佳化變數是 `R,t,log(s)`，共七個自由度；使用點到平面誤差：

\[
r_i=n_i^\mathsf{T}(sRp_i+t-q_i),\qquad
\min_{R,t,s}\sum_i w_i\,\rho(r_i).
\]

程式不把 GNSS／IMU 殘差放進這個 map fit。感測先驗只提供配準起點、局部點雲朝向／層級初分、搜尋範圍與有限信任界；因此 map fit 不會直接被 GNSS 拉回某個位置。每個來源點的匹配從 map 最近的八個點中，選取同層、法向相容且在半徑內的候選。地面與非地面依局部 PCA 法向、相對先驗的高度和 map 局部低水平面啟發式分類；這些是幾何規則，不是語義標註，也不保證將道路、牆面或其他物體正確辨識。

配置與數值意義如下（完整值也會寫入每次 run 的 `params.json`）：

| 設定 | 預設 | 用途與限制 |
| --- | ---: | --- |
| 三個尺度起點 | 6、8.7、11 | 探索不同初始化，並非已校準尺度。每個起點都做 map-only fit，最後以訓練 patch 誤差加缺少支持的懲罰選擇。 |
| 尺度範圍 | 4–12 | 有限探索／信任界，不代表校準區間或真實不確定度。尺度必須落在界內且離邊界至少 0.02 才可能接受。 |
| 來源作用距離 | 55 m | 以最大尺度固定點雲成員資格，避免某尺度靠刪掉遠點降低誤差。 |
| 平移／旋轉信任界 | 先驗周圍 5 m／15° | 限制局部配準離開 association seed 的幅度，不是先驗殘差或真值精度。每步平移最多 0.5 m、旋轉最多 2°、log-scale 最多 0.08。 |
| 對應半徑 | 3 m、1.5 m | 兩層由寬到窄的搜尋。每層最多 14 次迭代。 |
| Huber 尺度 | 0.4 m | 壓低大點到平面殘差的影響；不是感測器標定噪聲。 |
| 空間 patch | 1.5 m | 以 map 空間格留出資料，減少相鄰點同時進入訓練與檢查。每個法向主軸最多取 160 對。 |
| 幾何支持 | train 24、holdout 8、非地面 train 12 patches | 檢查空間覆蓋；不足即拒絕該幀 map 修正。 |
| 留出誤差 | 中位數 ≤0.45 m、P90 ≤1 m | 檢查未參與擬合的空間 patches；這不是定位 ATE 或精度保證。 |
| 水平可觀測性 | 最小尺度化奇異值 ≥0.12 | 消去自由 Z、roll、pitch 與 scale 後檢查 XY／yaw 支持；另檢查 scale 在其餘自由度投影後仍有訊息。屬數值診斷，非 covariance。 |

留出以 map patch 固定切分。檢查 XY／yaw 時，將訓練 Jacobian 分成欲觀測的平移 X、Y、yaw 與 nuisance 的 Z、roll、pitch、log-scale；將水平 Jacobian 投影到 nuisance 子空間的正交補，再看三個奇異值。這可檢查地面單一平面是否讓水平位姿與尺度混淆，但不能證明 map 本身正確。接受條件還要求訓練／測試 patch 數、非地面支持、尺度非邊界、與 seed 的平移界及留出殘差同時通過。未通過者不形成 map anchor。

## 低頻錨點與八維批次圖

對通過幾何檢查的 map pose，先比較相鄰影格的 map 位移與 GNSS 位移。兩個影格都須幾何接受、間隔不超過 3 秒，且位移差不超過 `0.5 m + 1 m/s × Δt`。這個 gate 用來排除與來源運動明顯不一致的 map 片段；GNSS 仍是帶偏差的量測，不是真值。

每 12 秒切一個不重疊時間區塊，至少需 8 個觀測且時間跨度至少 1 秒。未覆蓋超過 3 秒的缺口會再切段。每個合格區塊只形成一個穩健中位數 anchor，避免重疊滑動視窗把高度相關的點重複算成許多獨立絕對觀測；不對區間端點外推。位置 anchor 表示該區塊 map 相對 GNSS 的慢變偏差，yaw anchor 則比較 map yaw 與沿用 IMU 相對角增量所形成的參考航向。

批次圖在 GNSS 時刻估計八維狀態
\[
x_k=(p^x_k,p^y_k,v^x_k,v^y_k,\psi_k,b^g_k,b^x_k,b^y_k),
\]
分別是水平位置、速度、未包角 yaw、有效陀螺 yaw-rate bias，以及 GNSS 水平慢偏差。GNSS 因子為
\[
z^{xy}_k=p^{xy}_k+b^{xy}_k+\epsilon_k,
\]
每個 GNSS 樣本只進圖一次。位置／速度由白化的常加速度型過程因子連結；原始 gyro 積分提供相鄰 yaw 差因子，並估計有效 gyro bias；車體側向速度為軟性 nonholonomic constraint（NHC），不是感測器觀測。GNSS bias 採一階 OU 型慢變模型，時間常數 25 秒。

map anchor 對 bias 使用有界 Geman–McClure 型向量損失，避免單一 map 區塊無界拉動整段解；同一 anchor 的 yaw 修正也使用有界損失。這種 robust loss 只限制離群值影響，不會替錯誤但彼此一致的 map cluster 背書。graph 使用 SciPy `least_squares` 與稀疏 Jacobian，最多 80 次函數評估；失敗或非有限解會使本輪失敗，不輸出成功定位宣稱。

預設設定的意義與限制：

| 設定 | 預設 | 意義 |
| --- | ---: | --- |
| anchor 區塊／最小跨度／觀測數 | 12 s／1 s／8 | 抑制高頻 map 噪聲並避免少數影格產生 anchor。 |
| anchor 相鄰間距與位移 gate | 3 s；0.5 m + 1 m/s × Δt | 只檢查相鄰來源運動一致性。 |
| GNSS bias 相關時間／穩態標準差 | 25 s／1.5 m | OU 慢偏差模型的共享實驗設定；非依本資料校準。初始 bias 標準差 3 m。 |
| map 位置／yaw anchor 尺度 | 0.5 m／1° | anchor 殘差的標準化尺度，不代表已測得 map covariance。 |
| 有界損失門檻 | 3 個標準化單位 | 限制單一 anchor 的影響。 |
| 共用運動模型設定 | GNSS 0.3 m、加速度 PSD 4、gyro density 0.3°/√s、bias random walk 0.02°/s/√s、側向速度 density 0.5 m/s/√s | 作為數值權重與先驗假設，均未由外部真值校準；不得解讀成此車感測器的已知噪聲規格。 |

此圖沒有 GNSS course／pose orientation 因子，也沒有加速度計積分或完整三軸慣性預積分。四元數和 gyro 被假設為同一 IMU 軸系，並假設 forward FLU 軸；camera、IMU、body 外參和 GNSS 原點契約仍未確認。若沒有任何慢 map anchor，程式回傳本輪由 raw bag 重新產生的 `motion_rts` sensor 初始化，並明確記為沒有執行 joint estimation；那不是原 C++ 已評分定位結果，也不保證 bytes 相同。若有 anchor，解只更新輸出的 XY 與 yaw，保留輸入 pose 的 Z 與 tilt。

## 執行入口與產物

完整 Track 仍由使用者手動啟動。從 GPU 專案目錄執行：

```bash
bash scripts/dev.sh up
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 outputs/ml_runtime/venv/bin/python scripts/run_map_smoother.py --track 2 --full
```

Track1 使用同一套算法時將 `--track 2` 換成 `--track 1`。腳本由原始 bag 產生本輪 `motion_rts` prior，重新匯出 raw sensors，從原始 map 建本輪細 map；不依賴歷史 run 或 map cache。每個 run 放在 `outputs/run/`，保存來源、bag/map/prior/map SHA、程式快照與 SHA、參數、逐幀 `observations.jsonl`、`graph_report.json`、輸出矩陣 `result.txt`、`diagnostic.csv`、frame manifest 與 `validation.json`。Track2 完整輸出列數為 3299，Track1 為 3011；提交欄位為 `idx,x,y,yaw`。本文件沒有執行上述命令。

## 已知驗證與未解問題

`track00_test_2610200240/verification.json` 記錄 44 項機制／回歸檢查全數通過，涵蓋 map smoother、kinematic graph、surfel joint、full CLI、輸出 layout 與 replay。兩條 Track 的 full dry-run 正常回報各自 3011／3299 幀；`--full` 與 `--frames 48`、非零 `--start-frame` 的互斥檢查都正確拒絕。機制測試包含 Sim(3) 與 8-state sparse Jacobian（含 GNSS Huber／map 有界損失）、bias 正負號、地面退化拒絕、無 map 時保留 sensor 初始化，以及有 map anchor 時實際求解並發布 proper rotation；graph Jacobian 數值微分最大差約 `8.43e-9`，這是導數檢查，不是定位誤差。

修正首輪 `track02_test_2610200231` 的 SciPy Euler batch 輸入形狀錯誤後，從原 bag／map 完成三段共 160 幀真實短測；逐段 validation 均完整、有限且通過：

| run／Track2 原始幀範圍 | 幀數／時間 | 幾何接受 | 時間 gate 通過 | Anchor | Graph 求解／套用 |
| --- | ---: | ---: | ---: | ---: | --- |
| `track02_test_2610200233`／1600–1663 | 64／63.27 s | 32 | 17 | 1 | 是／是 |
| `track02_test_2610200236`／0–47 | 48／68.72 s | 48 | 20 | 1 | 是／是 |
| `track02_test_2610200238`／3251–3298 | 48／64.13 s | 11 | 4 | 0 | 否／否 |

中段 run 的尺度接受範圍為 `7.42066–9.61275`、中位數 `8.34493`；拒絕幀的主要原因是非地面訓練 patch 不足。該段單一 anchor 跨 1.58 s，graph 用 475 個 GNSS 樣本各一次、10 次評估，cost 從 111.13 降至 5.52；輸出相對本輪 sensor 初始化的 XY／yaw RMS 差為 `0.66714 m／0.23046°`。首段亦有一個 anchor 並實際求解／套用，版本差 RMS 為 `1.77169 m／2.34901°`。尾段沒有受支持 anchor，輸出明確保留本輪 `motion_rts` sensor 初始化；這不是原 C++ 已評分最佳 bytes。所有這些差值都描述模型行為，不是真值誤差或 Kaggle 分數。短段 anchor 的觀測跨度也不能代表完整 12 秒區塊的表現。

逐幀尺度尚不支援現有 RViz replay 的單一固定尺度介面。最後版本寫入 `replay_supported=false`，預設 replay 會跳過；使用者手動指定時會明確拒絕，不會用任意 scalar 假裝還原逐幀來源點雲。新 runner 同時保存 `map_modified` binary 與來源 SHA；`track02_test_2610200238/map_modifier_identity.json` 記錄的 binary SHA256 為 `8f5153caf328bf07f67c969a9b641bcf6ca7b8b5c3245b89bf64fdf6588b6f43`，容器／主機 binary 一致、執行前後不變。既有最佳的 Track1／Track2 CSV SHA 核對仍相同，沒有被本分支改寫。

主 agent 尚未啟動 full，Kaggle 尚無新分數。完整命令已可由使用者直接執行，不需再加「本地先證明精度改善」的資格關卡；執行仍由使用者手動完成。新分支尚無 full 結果，Kaggle 最佳仍為 Track1 1.15008／Track2 0.30488。

即使後續短測完整，也只能驗證流程、輸出契約與模型行為，不能替代正式分數或真值精度。map residual、幾何可觀測性、anchor 數量和 solver 收斂都不是課程評分。外部提供的 Track1 CSV（回報 0.65479）沒有被用作訓練目標、真值或尺度校準；原始學長方法描述也沒有足夠資料可供逐幀重現。當前已評分最佳維持 Track1 1.15008／Track2 0.30488，Track1 0.6 仍是目標，未達成。

實作位置：[`map_scale_icp.py`](../scripts/map_scale_icp.py)、[`slow_map_graph.py`](../scripts/slow_map_graph.py)、[`run_map_smoother.py`](../scripts/run_map_smoother.py)；既有平面運動因子見 [`kinematic_graph.py`](../scripts/kinematic_graph.py)。

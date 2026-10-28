# Track2：以當前資料融合 GNSS、IMU 與地圖

> 這是已停止的 v1 方案紀錄。現在的正式入口已改用 [65｜5839746c 因果版](65_TRACK2_5839746C_CAUSAL.md)；參數、模型與檢查請以 doc65 為準。此處保留當時設計與失敗證據。

這次重作的問題是：舊 Track1／2 結果曾使用 RTS（向後平滑），無法代表車上當下能算出的定位。新的 Track2 每幀先處理已抵達的感測器，再把當前點雲配到地圖，最後立刻輸出相機 pose；之後不修改這一幀。這是新的獨立實作，不會覆寫舊 CSV 或更動 Track3 專用程式。

使用者下一步是在 Ubuntu 主機執行：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
./scripts/dev.sh up
./scripts/run_track2.sh --full
```

參數集中在 [track2/parameters.py](../track2/parameters.py)。結果進入新 `outputs/run/track02_full_YYMMDDHHmm/`，主結果為 `result_2.csv`。報告圖見 [三張 flowchart](../../midterm01/report/flowchart/track2_causal_gnss/README.md)。本次只完成開發、合成檢查與唯讀輸入稽核；尚無真實定位或正式成績。

## 觀察與採用機制

學長 `others code` 同時提供 causal 與 offline pipeline。其 `build_causal.py` 是無 GNSS 的前向濾波參考，不能直接當成 Track2 的 GNSS 實作；`track_sequential.py` 預設還包含反向 backfill，因此沒有直接執行學長 shell。新程式保留前向狀態、道路高度尺度、地面同類配對、近距離優先與穩健量測的想法，重新整合到逐幀 runner。

| 來源 | 本次實作 |
| --- | --- |
| 學長 `scale_from_height.py` | 當幀與之前六幀的尺度中位數，無 centered window 或未來內插 |
| 學長 `register.py`／`merge_regs.py` | 點到面 ICP、Huber、地面同類、12→15→18→25 m 與 XY 資訊檢查；點級運算移到 CUDA |
| 學長 `build_causal.py` | 前向 EKF 與 gyro bias 狀態；明確使用秒的 `v*dt` 及含 dt 的 Jacobian，使用 Joseph covariance update |
| 原 GPU 專案 | `voxel_mean`、`VoxelModel`、`GpuNearest`、官方 mask 的 CUDA 投影、camera header clock、來源 SHA 與輸出契約 |
| 本次新增 | binary PCD XYZ loader、雙時間 GNSS reader、GNSS／IMU／map 單一事件流程、獨立 Track2 runner 與圖表 |

沒有使用學長的 batch smoother、backfill、整段 bias 擬合、yaw 輸出後重建或以榜分挑選的結論。初版門檻沿用來源中的幾何定義與明示工程先驗，未用榜分調參。

## 輸入與處理順序

1. 以 `init_guess.txt` 建立初始相機姿態。地圖的 binary／ASCII loader 僅取 XYZ；Track2 的 `tl`、`th` 欄位不進定位。地圖在 GPU 以 0.2 m voxel 合併，按目前 prior 裁 local submap 並估法向量。
2. PLY 檔名保留整數 bag 奈秒；用 bag 前相機訊息取得擷取時刻。IMU／GNSS 必須同時滿足 `arrival <= 當幀 PLY bag time` 與 `header <= 當幀 camera header`。已抵達但量測較新的資料先緩存；超過已輸出影像時刻才抵達的舊資料丟棄並記數，不回修。
3. 按 sensor header 時間前向預測並更新 GNSS。到目前影像時刻後，以官方前相機 ego mask 與 rectified FOV 移除車體及視野外點，從當前 PLY 的道路高度形成尺度量測。
4. 首次可信 anchor 前用 `H=1.573 m` 作高度先驗，尺度 `s=H/h_raw` 取七幀 trailing median；以當前點雲 Sim(3)（位姿與尺度共同估計）在 yaw offsets `0,-15,-7.5,+7.5,+15°` 作候選。首次 map update 也通過濾波 innovation gate 後，才以當幀 `scale*h_raw` 建立高度 anchor。之後每幀依高度估尺度，配準固定該尺度。若道路點不足，沿用過去尺度，不向未來找值。
5. 每個 radius 的 ICP 先 0.4 m voxel／2.0 m correspondence，再 0.2 m／0.8 m；地面配地面，非地面配非地面。各候選以進入當幀的同一尺度凍結各 stage 來源點，避免尺度優化改變比較點集。Huber 降低大殘差點的影響。線搜尋以包含未配對點 penalty 的成本判斷；同 radius 合格候選才互比成本。
6. 檢查 inlier ≥0.5、非地面 0.3 m 貼合比例 ≥0.4、位置與 yaw innovation 邊界，以及以 0.1 m 單位正規化的 XY 最小資訊 ≥2。首筆 anchor 前為 6 m／30° 邊界，之後為 3 m／10°。從近到遠找到有合格候選的第一個 radius 就停止；全部失敗則維持感測器 prediction。幾何檢查通過後仍需通過 EKF innovation gate，才更新當前 pose。
7. 立即寫當幀 TXT／CSV 與診斷，進入下一幀。沒有另外一階段改 yaw，也沒有完整軌跡陣列供反向改寫。

首筆 Sim(3) 的尺度資訊以 Schur complement 邊際化，傳給濾波器的位姿 covariance 也邊際化 z／roll／pitch；不把聯合估計中的尺度當成已知精確值。未通過品質或 innovation gate 的候選不能寫入最終 pose 或建立高度 anchor。

## 模型、殘差與約束

前向擴展卡爾曼濾波（EKF）狀態為 `x=[camera_x,camera_y,yaw,gyro_bias,signed_speed]`。在小時間間隔內，gyro z 扣 bias 得 yaw rate；相機沿其水平前向移動 `speed*dt`，不假設固定前進速度。速度由 process noise 及當前 GNSS／地圖的位置更新共同估計，初始為零且有 5 m/s 標準差。

GNSS 僅使用 XY 位置。量測模型是 `z_gnss = camera_xy + R(yaw)*lever + noise`。目前 lever 設零，位置標準差 2 m，未知 lever 額外加 1 m 標準差；這些是未校正工程先驗。沒有使用 GNSS quaternion 當 heading，也沒有把整段 GNSS 當真值或用它離線對齊軌跡。GNSS／map 更新均以 innovation covariance 算 NIS，超過 25 拒絕，通過者再以 Cauchy 權重降權。這個 gate 沒有經真值校正，不能作精度保證。

點到面 ICP 殘差為 `r_i=n_iᵀ(s R p_i+t-q_i)`，未知量是剛體 pose，anchor 前另含 `log(s)`。法向量與最近鄰搜尋、殘差、Huber 權重及 H/g 累加都在 CUDA；CPU 僅求解小矩陣。法向量使用既有 GPU local Gaussian PCA 映射到地圖點的方式，與學長以 12-neighbor PCA 的實作不同。

IMU 低變化本身也可能是等速行駛，故靜止條件還要求過去 GNSS 位置視窗低速且夠新；滿足時才以零速度及 gyro 均值作 bias 量測。不用開頭數秒強設靜止，也不看後方靜止區間估開頭 bias。靜止期間仍可用當前 map 更新位置；沒有宣稱固定 pose hold。

## 目前檢查與限制

實作已涵蓋 binary PCD、CPU／CUDA H/g 對照、合成 pose／scale 恢復、濾波與尺度前綴一致性、雙時間 future buffer／late discard、拒絕量測不洩漏狀態、生成資料 runner 的 CSV／TXT／timing 契約，以及 3299 筆生成資料的唯讀 full 圖表。實際資料另做完整 3299 幀時間契約稽核，沒有執行真實配準或濾波定位。測試 run 與數值見文末附錄。

須保留的假設：

- bag 的 GNSS frame_id 為空，程式假設它已在 LiDAR map 座標；首筆位置與初始 pose 接近只支持量級檢查，不能證明完整 frame 契約。沒有 GNSS→camera lever 或 IMU→camera 標定，設定未校正外參與 noise，不從全段輸出擬合它們。
- 高度 1.573 m 與初始尺度 8.6 取自學長參考程式，不是已驗證的 Track2 物理標定。只有當前 map Sim(3) 通過兩層 gate 後才建立因果高度 anchor；坡度、遮擋與單目深度偏差仍可能影響尺度。
- GNSS PoseStamped 不含 covariance；IMU 全零 covariance 也不代表沒有雜訊。預測只使用 gyro z，假設與相機 FLU z 平行；缺 IMU 超過 0.1 s 時不外推舊 gyro，並增加 yaw uncertainty。
- 接受率、殘差、資訊量、生成測試與檔案完整皆不能當成真實精度證明。尚未執行真實短段／full，沒有新 Kaggle 分數。完整 full 由使用者手動啟動，交付與提交也由使用者操作。

## 輸出與來源保存

`result_2.csv` 為 `idx,x,y,yaw`，3299 筆、local 0-based，yaw=`atan2(R10,R00)` 弧度；`poses.txt` 每幀 12 值 row-major 3×4、不含 timestamp。`timings.csv` 固定九欄，console idx 對應原始 idx。時間戳全程整數奈秒。

每輪保存 `parameters.json`、`source_manifest.json`、Git 差異、container image ID、pip freeze、輸入與 camera SHA、frame manifest、`observations.jsonl` 的狀態/covariance/GNSS update/配準候選/elapsed_seconds、`pose_trace.jsonl` 的 prior/registered/final。準備、warmup、配準、定位總耗時分別記錄；沒有 graph optimizer，graph_seconds 為零。畫圖耗時與失敗另存 `analysis_status.json`。

full 成功才自動產生 `analysis/01_trajectory.png/pdf`、`02_pose_changes.png/pdf`、`03_registration_timing.png/pdf`、額外 `04_fusion_and_scale.png/pdf`、`README.md` 及來源 manifest。畫圖唯讀，不改定位輸出。若畫圖失敗，保留 partial analysis 與 failure log；不要為了畫圖重跑定位。

同一來源 SHA 的圖表可從 run 快照重試，`RUN` 換成本輪目錄名稱：

```bash
RUN=track02_full_YYMMDDHHmm
./scripts/dev.sh exec python "outputs/run/$RUN/source/scripts/analyze_track2.py" \
  --run-dir "outputs/run/$RUN"
```

重試會檢查原始輸出 SHA，只重建已知圖檔；來源不同、缺 manifest 或夾有其他檔案時保留現場並報錯。

## 附錄：規則與檢查證據

官方 [主題目 PDF](../../midterm01/docs/Competition%20Regulations/00_SDC_2026_Midterm01_Localization.pdf) 10/20 更新允許 Track1／2 使用 GNSS fusion；[10/22 更新](../../midterm01/docs/Competition%20Regulations/01_Mid01_Updates_Rules.pdf) 禁止 backward／non-causal smoothing 及定位後軌跡修正。本文僅說明可追溯的程式機制，不代替助教對最終重現與規則的檢查。

學長來源原檔保持原樣。每輪把參考檔複製到 `source/reference/senior/` 作來源證據，runner 不 import 或執行它們；每檔 SHA 記入 source manifest。現有 Track3 `track3/`、`config/` 與四支專用入口的原 SHA 保持相同。

- [第一輪生成檢查](../outputs/run/track00_test_2610290120/pytest.xml)：9 passed、1 failed；失敗場景單前方平面使尺度與位移耦合，joint scale 邊際化後 XY 資訊不足，保留其失敗紀錄。補第二面獨立深度的合成平面後通過，沒有調低 production 門檻。
- [第二輪生成檢查](../outputs/run/track00_test_2610290121/pytest.xml)：10 passed。
- [含圖表與完整輸入稽核的檢查](../outputs/run/track00_test_2610290123/pytest.xml)：12 passed；[檢查圖表](../outputs/run/track00_test_2610290123/analysis/README.md)、[輸入時間契約](../outputs/run/track00_test_2610290123/data_contract.json)。這是合成與唯讀檢查，不是真實 Track2 定位。
- [最終版本檢查](../outputs/run/track00_test_2610290128/pytest.xml)：12 passed，含同來源圖表重試與輸入保護；[圖表](../outputs/run/track00_test_2610290128/analysis/README.md)。另確認 source snapshot SHA 與 Track2 官方標定／mask 尺寸。僅有既有 Matplotlib Axes3D warning，二維圖表正常產生。

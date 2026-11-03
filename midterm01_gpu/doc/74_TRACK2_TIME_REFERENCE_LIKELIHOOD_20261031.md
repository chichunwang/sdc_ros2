# Track2 v5.3：時間參考與地圖似然融合

## 目前狀態與下一步

v5.3 是 Track2 的因果版本，處理 GNSS 修正造成的 sensor prior 參考點改變，以及地圖候選重複計入 sensor prior 的問題。來源 SHA 已核對並套用 live。生成資料 CPU／CUDA／runner 檢查 78 項全過，只有既有 Matplotlib `Axes3D` warning；本輪 `real_localization_frames=0`，尚無 v5.3 full 或新分數。完整定位仍由使用者手動啟動：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
./scripts/run_track2.sh --full
```

不會由文件或測試自動啟動 full。現有參數不依非因果 CSV 調整；`parameters.py` 只改版本識別，其餘 v5.2/perf1 參數與加速路徑保持原值。

## 為何調整

### 使用者回報的 Public Score

使用者確認這些是同一 Track2 Kaggle Public Score，目標是讓數值降低：

| 版本或輸出 | 回報值 | 解讀 |
| --- | ---: | --- |
| v5 | 0.53683 | full 已完成；map update 為 0 |
| v5.1 | 0.51617 | full 已完成 |
| v5.2 | 0.46434 | full 已完成；比 v5.1 低約 10.04% |
| 非因果 `result_2_submitnoncasual.csv` | 0.30488 | 只作唯讀對照，不是 ground truth |

分數改善是整體輸出上的回報，不能單獨證明某一項航向或 map update 機制造成改善，也不是定位誤差下降相同比例的證明。非因果 CSV 不用來擬合延遲、yaw offset、參數或隱藏真值。v5.3 尚無分數，不保證達到 0.30488。

### v5.2 run 與時間資料

`track02_full_2610311445` 有 3299 幀、相機時長 118.686 秒；map accepted 1717 幀，observable rank 0／1／2／3 的幀數為 1121／1901／260／17。輸出 map XY 範圍為 x=3271.666–4381.342 m、y=−1454.033–−983.676 m；yaw 範圍 −55.793° 至 −2.804°。首個 GNSS heading pair 在 frame 2 帶來 3.594° 的 yaw 變化；首個可接受的 GNSS course 更新在 frame 308（約 11.084 s）。這些是 run 統計，沒有獨立逐幀定位真值。

| 觀察 | v5.2 資料摘要 | 含義與限制 |
| --- | --- | --- |
| GNSS 到達時間 − header 時間 | median 1.095 ms，mean 1.181 ms | 訊息到達不代表 quaternion 的物理觀測 epoch 已知 |
| 相機到達時間 − header 時間 | median 7.368 ms | runner 仍以當前相機時間對齊輸出 |
| GNSS XY 更新頻率 | 約 4 Hz | 474 個相鄰 quaternion interval 中，383 次 heading 值未變；heading 改變間隔 median 1.246 s |
| 車速 >10 m/s 時最新 raw GNSS fix age | median 124.50 ms，maximum 261.11 ms | KF 已把位置與速度先驗向相機擷取時刻做 CV 外推；raw fix 與該時刻 prior 的 XY 差 median 1.516 m、maximum 3.857 m |
| 航向相關性檢視 | map candidate 對 IMU yaw rate 的最佳 lag 為 0 s，相關 0.665；GNSS yaw 最佳 lag 為 −1.5 s，相關 0.397 | GNSS yaw 長時間 held，樣本不獨立；不能據此宣稱固定延遲 |
| map XY 沿程等效時間差 | P5 −13.68 ms、P50 2.57 ms、P95 24.67 ms | 包含側向空間偏差，不能解釋成單一 clock shift |

GNSS `PoseStamped` 共用一個 header，但位置約 4 Hz 更新，orientation 經常重複維持舊值。現有證據不足以判定 quaternion 的物理觀測時間；將整筆 GNSS 平移固定秒數會把未證實的假設直接寫進 pose。相同理由也不把外部 CSV 的最佳對齊位置當成時間標定。

## v5.3 的三項改動

### 1. 把 GNSS filter 更新造成的參考位移與車輛運動分開

GNSS Kalman filter 的一次位置 innovation 會同時改位置與速度。這是濾波器修正估計的變化，不能全數解釋成車輛真的在該瞬間移動。v5.3 把 update 造成的四維 shift

\[
d = [\delta x,\delta y,\delta v_x,\delta v_y]^T
\]

沿用既有 constant-velocity transition 傳到目前相機時間：

\[
d_{camera}=F(\Delta t)d,\qquad
F(\Delta t)=\begin{bmatrix}I&\Delta t I\\0&I\end{bmatrix}.
\]

由此取得 sensor prior 的當幀 XY 參考變化；GNSS bias 更新和 lever-arm 旋轉引起的變化也記入 calibration reference delta。既有 correction filter 保存 map-memory mean 對 sensor calibration reference 的一階敏感度 (S)，在新參考到達時使用

\[
c \leftarrow c-S\,\Delta r.
\]

`S` 由實際 Kalman gain 與觀測矩陣累積；尚未被地圖學到的方向保持零敏感度，因此不會因 GNSS jump 憑空產生地圖補償。reference delta 每幀只使用一次；已輸出的舊 pose 不變。

`S` 是 correction mean 的局部一階 reference sensitivity，**不是完整的 GNSS／map 聯合 covariance**，也不是額外量測。已吸收候選的內部比較參考也會依其 sensitivity 同步重定錨，避免下一幀 motion comparison 把舊 sensor reference 當新車輛位移。GNSS 位置誤差相關性、GNSS frame、lever/extrinsic 與 heading epoch 尚未標定，因此不能把這套統計量描述成校準完成的 joint filter。

### 2. 從 prior-regularized candidate 取回 map-only likelihood

原本的配準候選是以 point-to-plane 資料和 sensor soft prior 一起求得的 MAP 解。如果把候選當成獨立的地圖量測再融合一次，就會再次把同一份 sensor prior 算進去。v5.3 在 solver 中保留不含 sensor/height prior 的 data-only Hessian (H_D) 與 gradient (g_D)，然後把局部變數轉成世界座標

\[
\xi=[\delta x,\delta y,\delta\psi,\delta\log s].
\]

將 log-scale (s) 當作 nuisance variable 邊際化；若其資訊區塊為正：

\[
H_m=H_{pp}-H_{ps}H_{ss}^{-1}H_{sp},\qquad
g_m=g_p-H_{ps}H_{ss}^{-1}g_s.
\]

其中 (p=[x,y,\psi])。再用當幀 sensor prior covariance (C=LL^T) 將 map 資訊 whiten：

\[
L^TH_mL=V\Lambda V^T,\qquad \bar g_m=L^Tg_m.
\]

只保留超過資訊門檻的 eigen-directions。若 (c_{MAP}) 是目前 solver 候選相對 sensor prior 的世界 XY/yaw correction，局部 map likelihood 的中心為

\[
c_{map}=c_{MAP}-LV_o\Lambda_o^{-1}V_o^T\bar g_m.
\]

measurement covariance 由保留方向的資訊倒數、既有誤差 floor 與 point-to-plane 殘差膨脹形成。弱方向沒有量測 row，GNSS prior 只在 filter 裡融合一次；既有 NIS、motion、step-size 與品質 gates 繼續使用。若重建出的 likelihood center 超出原配準搜尋界限，v5.3 以 `local_likelihood_outside_search` 拒絕這次 map measurement，避免把局部二次近似外推到 solver 搜尋範圍之外。

這是固定 associations 的局部二次近似，不是完整校準 likelihood。資料 Hessian 的 noise、點雲關聯及與 sensor prior 的相關性仍含工程近似；低重疊或錯誤匹配不會因這個轉換自動變可靠。

### 3. 把未知 GNSS heading epoch 表成不確定度，不硬設時間偏移

每當 GNSS quaternion 的 heading 值改變，v5.3 取兩次 heading change 之間已抵達的 IMU yaw 範圍，將該段 angular spread 平方加入 heading observation variance。若前一個 change 超出可用 IMU history，另外加入既有 10° GNSS bias variance。此處只增加不確定度，不移動 timestamp，不用未來 IMU，也不修改既有 bias/noise/limit 參數。diagnostic 明記 `physical_heading_delay_s=None` 與 `fixed_time_shift_applied=False`。

### RViz 時戳

RViz pose、cloud、TF 現在使用 camera header timestamp；原先使用的是 bag timestamp。`idx`、timings 與 manifest 欄位契約維持原樣，仍保存相機與 bag 時間供檢查。這是顯示時間戳對齊修正，不會重寫 pose 數值。

## 驗證、來源與限制

v5.2 full 使用的是 `v5.2_perf1` 套用前的演算法 source；之後才套用加速版。v5.3 保留 perf1 的 device path、iteration 與門檻，因此 v5.2 的 Public Score 不能歸因於該加速改動。v5.3 不改 10° bias、0.12°/√s random walk、3° GNSS heading observation 或其他既有工程預算，也沒有用 `.30488` 非因果 CSV 調整它們。

| 證據 | 狀態 | 可支持的結論 |
| --- | --- | --- |
| `track00_test_2610311559` | 77 passed、1 failed，14.354 s | 中間失敗是內部 JSON diagnostics 未把 `data_g` ndarray 轉成 list；失敗 run 原樣保留 |
| `track00_test_2610311605` | 78 passed、0 failed，14.478 s；1 項既有 Matplotlib `Axes3D` warning | 生成資料 likelihood/time/runner checks 通過，不是實跑精度證明 |
| v5.3 real frames / full / score | 0 / 未執行 / 未知 | 尚無真實定位或 Public Score 證據 |

程式含有 likelihood search-bound guard：局部似然中心超出原配準搜尋界限時會拒絕該次 map measurement。1605 生成檢查以強制拒用案例驗證此分支，也檢查 bag/camera 時鐘差約 7 ms 時 RViz 仍使用 camera header。套用收據記錄 7 個檔案的來源 SHA，live source 與 1605 快照一致；baseline 為 `v5.2_perf1`，套用時沒有活動定位程序，也沒有啟動 full。

可直接查看 [v5.2 full run](../outputs/run/track02_full_2610311445/test_info.json)、[全段摘要](../outputs/run/track00_test_2610311554/analysis/full_summary.json)、[run 比較摘要](../outputs/run/track00_test_2610311554/analysis/summary.json)與[分段表](../outputs/run/track00_test_2610311554/analysis/segments.csv)。[全段圖表索引](../outputs/run/track00_test_2610311554/analysis/README.md)已完成。v5.2 的 summary/reference 不是 ground truth。

v5.3 最終生成驗證證據：[1605 pytest.xml](../outputs/run/track00_test_2610311605/pytest.xml)、[1605 pytest log](../outputs/run/track00_test_2610311605/pytest.log)、[1605 test_info](../outputs/run/track00_test_2610311605/test_info.json)、[套用 SHA 收據](../outputs/run/track00_test_2610311605/apply_receipt.json)。1601 為前一輪歷史檢查（78 passed、14.441 s），最終結果以 1605 為準。失敗歷史：[1559 pytest.xml](../outputs/run/track00_test_2610311559/pytest.xml)。本輪程式快照：[causal_prior](../outputs/run/track00_test_2610311605/source/track2/causal_prior.py)、[observability](../outputs/run/track00_test_2610311605/source/track2/observability.py)、[metric_registration](../outputs/run/track00_test_2610311605/source/track2/metric_registration.py)、[parameters](../outputs/run/track00_test_2610311605/source/track2/parameters.py)、[runner](../outputs/run/track00_test_2610311605/source/scripts/run_track2.py)。[生成檢查圖表索引](../outputs/run/track00_test_2610311605/analysis/README.md)已完成。

## 操作上的邊界

- v5.3 目前沒有真實短段或 full 輸出；不能以 v5.2 的 0.46434 分數推定它會更低。
- GNSS XY CV prior、quaternion 的物理 epoch、frame/lever/extrinsic 和 noise covariance 都不是獨立標定結果。
- `reference_sensitivity` 只描述 correction mean 對小參考變化的一階反應；data-only likelihood 也只是在目前候選與固定 associations 周圍的局部模型。
- 合成測試、map-accepted 數量、trajectory smoothness、GPU 加速及模型殘差都不能單獨證明逐幀 pose 精度。
- live source 已完成 SHA 核對與套用；完整 Track 仍由使用者手動啟動，定位輸出只向前，不回頭修正舊 frame。

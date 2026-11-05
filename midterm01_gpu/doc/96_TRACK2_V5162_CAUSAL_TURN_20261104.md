# Track2 v5.16.2：causal coordinated-turn motion

## 目前狀態

`track2_v516_causal_turn_v5_16_2_perf2` 延續 v5.16.1 的雙側可靠平面 fallback 與 40 m 範圍，新增以已到達 gyro 轉速驅動的 XY coordinated-turn motion prior。3299 幀 full [`track02_full_2611042335`](../outputs/run/track02_full_2611042335/test_info.json) 已完成，定位耗時 `2259.185193303 s`、registration `2161.603214011 s`；full analysis 已完成。211 個 frozen source snapshot、135 個 live Python、6 個 output 與 7 個 analysis source 的 SHA 均核對相符；`input_sources` 與 `frame_manifest` 和 v5.16.1 full byte 相同。沒有新正式分數，使用者目標 `0.30488` 尚未達成；未對外提交。

先前的 ODE／sensor-only forecast replay、單元檢查和 suite 通過結果只驗證模型機制與預測觀測一致性，不等於完整 pipeline 的定位精度。尤其 sensor-only replay 固定了 image／height 過去值，無法預測 map／scale 驗收改變後會否讓 legacy road-PnP 多接受影像速度。

## 觀察與模型選擇

車輛的速度方向可能與車身絕對 heading 不同。若把側向速度強制設為零，或每幀直接把速度轉向絕對 heading，可能抹掉真實的側滑／轉彎動態。v5.16.2 使用目前時間區間已到達的 gyro 相對轉速旋轉 XY 速度，不強迫速度方向貼齊絕對 heading；不新增 heading measurement，也不改原 heading budget。

令 \(p=[x,y]^T\) 為位置、\(v=[v_x,v_y]^T\) 為地圖 XY 速度、\(\omega\) 為當前區間的 gyro 相對角速度，並令

\[
J=\begin{bmatrix}0&-1\\1&0\end{bmatrix},\qquad R(\theta)=\exp(J\theta).
\]

在一個時間步內將已到達的 \(\omega\) 視為常數，連續運動模型為

\[
\dot p=v,\qquad \dot v=\omega Jv+a_w,
\]

其中 \(a_w\) 是白加速度雜訊。忽略雜訊的精確離散狀態轉移為

\[
v_{k+1}=R(\omega\Delta t)v_k,\qquad
p_{k+1}=p_k+\left(\int_0^{\Delta t}R(\omega\tau)\,d\tau\right)v_k.
\]

令 \(A(\Delta t)=\int_0^{\Delta t}R(\omega\tau)\,d\tau\)，XY 子系統的 state transition 與白加速度 process covariance 為

\[
F(\Delta t)=\begin{bmatrix}I&A(\Delta t)\\0&R(\omega\Delta t)\end{bmatrix},\qquad
Q(\Delta t)=\int_0^{\Delta t}F(\tau)G S_a G^\mathsf{T}F(\tau)^\mathsf{T}\,d\tau,
\]

其中 \(G=[0\;I]^\mathsf{T}\)，\(S_a\) 是白加速度的連續時間強度。程式按常轉速積分計算 \(F\) 與 \(Q\)；\(\omega\Delta t\) 很小時用 Taylor 展開避免數值消去。這讓直線與轉彎都由同一模型連續處理。

## 因果 gyro bias 更新

gyro 零偏只從既有 image 與 GNSS 都已接受為靜止、且不重疊的時間區間估計；新估計只用於其後的 motion prediction，不回改過去 pose。rest noise \(\sigma=0.10°/s\)、bias random walk `0.02°/s/√s` 沿用 v5.9 已聲明的 engineering budgets，並非感測器真值或外參標定結果。

## Sensor-only replay 與機制檢查

研究輸出在 [`track00_test_2611042251`](../outputs/run/track00_test_2611042251/test_info.json)：

- [`nhc_prior_replay.json`](../outputs/run/track00_test_2611042251/nhc_prior_replay.json)：Non-Holonomic Constraint（NHC，非完整約束）prior replay 中，強制側向速度為零被拒絕；下一筆 GNSS 的前瞻 XY RMSE 由 `0.086055 m` 變為 `0.130133 m`。這是特定預測一致性比較，不是定位誤差真值。
- [`implemented_turn_replay.json`](../outputs/run/track00_test_2611042251/implemented_turn_replay.json)：依實際 3299 幀 sensor 順序 replay；control 精確重現 baseline，production 與獨立 prototype 的完整狀態及 covariance 最大差都是 `0`。候選的下一筆 GNSS forecast RMSE 為 `0.080632976 m`；分軸為 left `0.032019599 m` 對 `0.04368751 m`、forward `0.074002852 m` 對 `0.0741413 m`。replay 共更新 rest gyro bias 17 次。這些是預測 GNSS 的一致性數值，不代表 pose 精度。
- [`disabled_baseline_summary.json`](../outputs/run/track00_test_2611042251/disabled_baseline_summary.json)：停用新分支後，201 個 sensor frames 與 3 個 CUDA registration frames 完全等於 baseline；174 個共同參數無差異。
- 21 個 ODE／Q／Jacobian／semigroup 案例及 2 個 quarter-circle 案例通過。
### Sensor-only fixed-window forecast（非 full pipeline）

[`forecast_window_summary.json`](../outputs/run/track00_test_2611042251/forecast_window_summary.json) 固定各版本的 sensor-only replay；四個事先固定窗口中，candidate 的下一筆 GNSS XY forecast RMSE 都略低於 control：

| 固定窗口 | Control RMSE (m) | Candidate RMSE (m) |
| ---: | ---: | ---: |
| 0–799 | 0.0849645 | 0.0848252 |
| 800–1599 | 0.0983857 | 0.0890123 |
| 1600–2399 | 0.0845226 | 0.0791506 |
| 2400–3298 | 0.0758750 | 0.0694763 |

逐筆 473 個 forecast 中，candidate 有 296 筆改善、176 筆變差、1 筆相同；不能概括成每筆都改善，也不是逐幀定位真值。

- [`orientation_turn_replay.json`](../outputs/run/track00_test_2611042251/orientation_turn_replay.json) 評估由已到達 quaternion 差分求相對轉速的研究替代：XY RMSE `0.081219996 m`、left `0.033477489 m`、NLL `0.8793846`；biased-gyro 候選為 `0.080632976 m`、`0.032019599 m`、`0.6970512`。quaternion 差分結果沒有較好支持，因此未部署。
- [`historical_image_audit.json`](../outputs/run/track00_test_2611042251/historical_image_audit.json) 顯示移動影像量測很稀少：v5.10 與 v5.16.1 都只在 idx `196`、`231`、`238` 接受；v5.14 雖有更多影像通道，主要接受仍到 idx `686` 以前。這提示影像觀測可能失聯，不足以認定它是榜分差異的唯一原因。

完整 Track2 CPU/CUDA suite 為 `312 passed`、`46.00 s`，另有 1 項既有 Matplotlib Axes3D warning。首次 suite 的 `308 passed、3 failed` 紀錄仍保留：兩項缺 oracle 環境，另一項是舊 metadata expectation；修正後同範圍 suite 全數通過。最終 XML 與 log：[pytest_suite_fixedenv.xml](../outputs/run/track00_test_2611042251/pytest_suite_fixedenv.xml)、[log](../outputs/run/track00_test_2611042251/pytest_suite_fixedenv.log)。

full analysis 的 read-only contract 檢查已通過 `1 passed`、`24.64 s`，XML／log 為 [pytest_analysis.xml](../outputs/run/track00_test_2611042251/pytest_analysis.xml) 與 [pytest_analysis.log](../outputs/run/track00_test_2611042251/pytest_analysis.log)。目前已有 phase2 初圖 [01｜causal forecast validation PNG](../outputs/run/track00_test_2611042251/analysis/01_causal_forecast_validation.png)／[PDF](../outputs/run/track00_test_2611042251/analysis/01_causal_forecast_validation.pdf)；它不是 NHC 專圖，也不是 v5.16.2 的 full analysis。contract 保證圖表分析唯讀；full 的 `14_coordinated_turn` 已生成並檢視。

## Full 前綴觀察與完整 run 結果

### 800／1600 幀實際前綴

800／1600 checkpoints 已完成。兩版首筆 map update 都在 idx `4`；候選 rest gyro-bias 更新 `17` 次。map update 計數如下，順序為 control → candidate：

| checkpoint | map updates |
| ---: | ---: |
| 800 | 340 → 360 |
| 1600 | 971 → 975 |

idx `616` 候選額外接受一筆 legacy road-PnP image speed `5.237764 m/s`（該筆前 sensor speed 為 `7.799148 m/s`），其 XY update 即時拉回約 `0.234 m`；候選與 control 的最大 XY 差達 `0.370638 m`。[`audit_actual_prior_prefix` 紀錄](../outputs/run/track00_test_2611042251/audit_actual_prior_prefix.log) 顯示差異不是 coordinated-turn 單獨 replay 錯誤，而是 map／scale validation 改變後，legacy road-PnP 多接受了一筆。舊 fixed-image replay 沒有重演這條 branch；[`replay_actual_full_prior.py`](../outputs/run/track00_test_2611042251/replay_actual_full_prior.py) 以各版本自己的實際 past image／height exact 重現 prior，最大誤差為 `0`。

實際 full-prefix 的下一筆 GNSS forecast 為：

| 已到前綴 | XY RMSE control → candidate (m) | Left RMSE control → candidate (m) | Forward RMSE control → candidate (m) |
| ---: | ---: | ---: | ---: |
| 800 | 0.0849645 → 0.1076721 | — | — |
| 1600 | 0.0920080 → 0.0986620 | 0.0405674 → 0.0280047 | 0.0825819 → 0.0946041 |

這些是每個 checkpoint 以各版本實際 image／height 前綴計算的下一筆 GNSS 預測，不是 pose 真值。800 幀 XY、1600 幀 XY／forward 預測較差，left 分量較好；不能據此宣稱整體改善。證據與 checkpoint 圖見 [800](../outputs/run/track00_test_2611042251/actual_forecast_800.json)／[1600](../outputs/run/track00_test_2611042251/actual_forecast_1600.json) 及 [analysis](../outputs/run/track00_test_2611042251/analysis/README.md)。這些早期結果保留作 prefix 紀錄，不替代完整 pipeline 結果。

### 3299 幀 full 結果

完整 run 為 [`track02_full_2611042335`](../outputs/run/track02_full_2611042335/test_info.json)：3299 幀完成，定位耗時 `2259.185193303 s`，registration 耗時 `2161.603214011 s`；`analysis_status` 為 completed，耗時 `10.729403435 s`。來源快照 211／211、live Python 135、輸出 6、analysis sources 7 的 SHA 核對均相符；輸入來源與 frame manifest 和 v5.16.1 full [`track02_full_2611042153`](../outputs/run/track02_full_2611042153/test_info.json) byte 相同。9 組標準 PNG／PDF 已完成，包含 [`14_coordinated_turn`](../outputs/run/track02_full_2611042335/analysis/14_coordinated_turn.png)；圖表索引與全圖見 [full analysis README](../outputs/run/track02_full_2611042335/analysis/README.md)。

| 完整 run 指標 | Control → v5.16.2 |
| --- | ---: |
| map accepted | 2049 → 2056 |
| scale validated | 1763 → 1779 |
| XY 輸出差 median／p95 | 0.0195743／0.0591563 m |
| XY 輸出差最大值 | 0.3706383 m（idx 619） |
| yaw 輸出差最大值 | 0.172954875° |
| increment decomposition closure residual | 8.99e-13 |

這些是兩版輸出差，不是真實定位誤差。共同 19 個固定幀的 cost 比較中，原始模型有 5 幀較低、14 幀較高，平均 delta `+0.0004527549`；surface 模型有 6 幀較低、13 幀較高，平均 delta `+0.0003638248`。固定幀 cost 也不能取代 pose 真值。

[`actual_forecast_3299.json`](../outputs/run/track00_test_2611042251/actual_forecast_3299.json) 以兩版各自的實際 image 前綴重播，prior exact match 均為 0 誤差。下一筆 GNSS 觀測預測如下：

| Forecast 指標 | Control → v5.16.2 |
| --- | ---: |
| XY RMSE | 0.0860553897 → 0.0869042717 m |
| left RMSE | 0.0436875075 → 0.0320253948 m |
| forward RMSE | 0.0741412960 → 0.0807881583 m |
| NLL | 0.6984311 → 0.6985863 |

這是 forecast 與 GNSS 觀測的一致性，不是 pose ground truth：XY 與 forward 略差、left 較好，NLL 也略高，沒有整體改善證據。idx 616 的 image speed 分支仍是已知差異：legacy／free／fixed-gyro／joint road-PnP 分別為 `5.23776`／`5.24012`／`5.38459`／`5.24265 m/s`。joint 方法的 9 個生成案例通過，但在這個真實影像案例仍未解決該差異，故未採用。IPPE 第二解對應約 `206 m/s` 且 pixel fit 較差，不支持影像解歧義是此幀主因。比較資料見 [`road_pnp_models.json`](../outputs/run/track00_test_2611042251/road_pnp_models.json) 與 [idx 616 原始影像](../outputs/run/track00_test_2611042251/analysis/14_road_image_616.png)。路面支持診斷已完成：45個原PnP inliers僅8個完整finest LK `21×21` patch落在路面mask內；重新選特徵後16 tracks低於原40點門檻。[原圖與點位](../outputs/run/track00_test_2611042251/analysis/17_road_patch_support_616.png)顯示大量點在遠端路面／物體交界。這是路面假設支持不足，不是對5.24m/s真偽的真值證明。

full 的 console 只記錄到約 idx 3100：host CLI session 以 `143` 結束後，container PID `286904` 仍持續並完成了原 full；沒有重跑、source 變更或 `failure.log`，保留 raw console。這是 console 覆蓋範圍限制，不是定位失敗。最初 queued diagnostic 因 CLI 提早退出、assert 未完成而中止；重試均完成，相關 logs 保留。

目前沒有新 Public Score，目標 `0.30488` 尚未達成；full 完成、圖表完成或預測殘差變化都不能單獨證明定位精度改善。coordinated turn 僅使用當下可用 gyro rate，bias 只由已接受的靜止區間因果更新；沒有替缺失觀測創造定位資訊。後續v5.16.3只改路面光流視窗支持，詳見[97](97_TRACK2_V5163_ROAD_PATCH_20261105.md)；本v5.16.2 full快照與結果維持原樣。

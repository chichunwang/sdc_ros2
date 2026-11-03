> **歷史基準提示：**本文數據描述原始 v5.3 + perf1（1735 regression audit），不是 fix1 的執行結果。其指出的 GNSS XY rebase、方向性 memory expiry 與 map-only likelihood 局部驗收風險，已由後續 fix1 對應修改；fix1 已於 2026-10-31 18:11:48 SHA 保護套用 live。fix1 目前只有生成／CUDA 證據，沒有真實定位、full 或分數。請讀 [doc77｜方向性記憶與 map likelihood 修正](77_TRACK2_V53_FIX1_DIRECTIONAL_MEMORY_20261031.md)了解修正設計；本文原數值與歷史證據保留。

# Track2 v5.3 regression audit：GNSS reference rebase 與弱觀測方向

## 結論與目前狀態

使用者回報 v5.3 + perf1 full `track02_full_2610311612` 的 Kaggle Public Score 為 `0.80300`。移除 perf1 的 no-accel 生成測試 `track00_test_2610311713` 有 78 passed、0 failed；五個合成場景中與加速版的 pose、scale、correction、decision 全部相同，既有真實前五幀 A/B 的 CSV／TXT 也一致。這表示目前證據沒有顯示加速路徑本身造成數值差異，但不能外推到完整 3299 幀，也不代表 v5.3 的定位結果正確。

live 已恢復為 `track2_causal_time_reference_likelihood_v5_3`，並把 Track2 perf1 路徑加回；套用的 7 個檔案與 full1612 source SHA 完全一致。no-accel 對照不再是 live。這次工作沒有修改 v5.3 的定位算法，也沒有重新跑真實定位；`.80300` 仍是原 v5.3 + perf1 的回報分數，沒有證據表示已修復。不要原樣重跑同一版本期待分數改善；完整 Track 仍由使用者手動啟動。

分析發現一項較值得處理的**模型風險**：目前把 GNSS XY Kalman update 引起的位置／速度先驗變化當成 calibration reference change，用 correction-memory 的敏感度抵消。GNSS XY innovation 也可能包含真實需要保留的位置與速度修正；若 map 記憶在該方向長期沒有觀測，抵消可能逐步吃掉沿該方向的位移。這由記錄值的代數記帳及一個生成反例支持，**尚未證明它造成 `.80300`，也沒有在本輪修改算法**。

## 真實 run 的比較觀察

下列 v5.2／v5.3 比較是不同算法版本間的結果，並非只切換 perf1 的嚴格單變因實驗：

| 量 | v5.2 | v5.3 + perf1／差異 | 解讀 |
| --- | ---: | ---: | --- |
| sensor prior XY | 完全相同 | 完全相同 | XY 差異不是由這兩次 run 的 sensor prior 輸入造成 |
| sensor prior yaw mean 差異 | 基準 | 平均 `0.00808°`，最大 `0.1071°` | 小幅差異；不能解釋整段 final XY 的差距 |
| raw map candidate XY 差異 | 基準 | 平均 `0.02696 m` | map candidate 整體差異小於 final pose 差異 |
| final XY 差異 | 基準 | 平均 `0.37624 m` | correction／融合後的輸出差異較大 |
| map accepted | `1717` 幀 | `1704` 幀 | 接受數不是精度指標 |
| map correction XY 範數 | mean `0.17669 m`、P95 `0.40410 m`、max `0.58556 m` | mean `0.45493 m`、P95 `1.11354 m`、max `1.76292 m` | v5.3 correction 絕對量較大；不能單靠此數字判定每幀方向或真值誤差 |
| 絕對 yaw correction | mean `0.17202°` | mean `0.45323°` | v5.3 較大，但沒有逐幀真值 |

對已記錄的 gain、observable projector、reference sensitivity 與 decay 做線性更新記帳，最大重組誤差為 `1.33e-15`。v5.3 各分量的平均 XY 範數為：map candidate `0.19471 m`、map likelihood 額外步 `0.05097 m`、位置 reference rebase `0.28038 m`（最大 `1.34367 m`）、heading reference `0.03792 m`。這是把**實際已發生**的更新按記錄拆開，不是各自重跑的 counterfactual，也不是因果 ablation；向量分量的範數不能相加成總位移或 score 變化。

例：frame 1424 的實際 correction 為 `[-1.69627, 0.48018] m`，其中位置 reference rebase 分量為 `[-1.32142, -0.07560] m`。這指出 rebase 在該幀佔有明顯份量，但沒有獨立真值可判斷這一分量是否全錯。

## reference rebase 為何可能吃掉位置修正

設 correction-memory 的 mean 為 `c`，對 sensor calibration reference 的一階敏感度為 `S`。當 calibration reference 改變 Δr 時，現行規則用

\[
c \leftarrow c-S\,\Delta r
\]

讓既有 map correction 對座標／校準基準轉換保持一致。這個處理適用於真正的 calibration reference 變化；GNSS XY Kalman innovation 卻不必然是 calibration 變化，它也可能是 filter 根據新位置量測修正位置與速度。

在 `237` 個「該幀接受 partial-rank map 更新，同時有 GNSS XY rebase」案例中，GNSS rebase 落在當幀未納入觀測的弱方向／現行 observable subspace 之外的比例（以 prior covariance whitening）中位數是 `0.81971`，`174/237` 超過一半。`null_rebase.json` 的 projector `P` 由程式保留的 observable eigenmodes 組成；被捨棄的方向資訊較弱，但不一定是數學上的零資訊。這不表示全部舊 map memory 都無效。但目前 correction-memory 的 `last_update` 是全域計時，只要任一方向接受 map update 就重設 age，沒有分 observable mode 計齡。於是某些弱方向可能長期不在當前觀測子空間內，而其 `S` 仍保留並持續抵消該方向的 GNSS prior 更新。

Δr 中的 GNSS XY CV filter shift 不能直接視為外參或座標校準變化。要把位置 innovation 完整保留，應以明確的座標校準參數建模，或在狀態與 covariance 中保留 GNSS／map correction 的相關性；`S` 只是 correction mean 的局部一階敏感度，不是完整 joint covariance。

## 生成反例：只在弱觀測方向出現長期抵消

另建立一個 201 幀的生成場景來檢查機制：初始地圖觀測 XY／yaw 至 `0.45 s`；之後每 `0.05 s` 只觀測 y/yaw，data Hessian 的 x 方向確實沒有觀測 row。GNSS 為無噪聲 4 Hz；車輛在 `1 s` 後以 `2 m/s²` 加速。兩個分支使用相同 sensor prior，唯一差異是其中一支套用 GNSS XY reference rebase。

| 10 秒時 x | 結果 |
| --- | ---: |
| 合成真值 | `81.0000 m` |
| sensor prior | `80.9213 m` |
| 不套用 XY rebase 的輸出 | `80.9213 m` |
| 套用 XY rebase 的輸出 | `73.7319 m` |
| 移動片段 x RMSE：不套用／套用 | `0.15683 m`／`4.14843 m` |

這個反例顯示：生成場景的 x 在 `9.55 s` 沒有 data Hessian 觀測 row、global correction-memory age 卻仍是 0，`Sxx=0.99535`；因此在此生成條件下，rebase 會抵消 x 方向的位置／速度修正。此處的 4.14843 m 是**合成場景**的 x RMSE，不是實際 Track2 誤差，也不能拿來解釋 Public Score。實際 run 的 partial-rank 模式則是未納入觀測的弱方向，不應一概描述成完全沒有幾何資訊。

## map likelihood 的另一項風險

只看接受 map measurement 的幀，重建出的 likelihood center 平均多帶入 `0.04808 m` XY；經實際 Kalman gain 更新後，每幀平均追加 `0.00477 m`。現有 registration quality gate 主要驗原始 candidate，而 likelihood center 另有 search-bound gate，未再用同一組幾何匹配獨立驗其 residual。局部二次模型可能把中心推離原候選附近；這是需檢查的外推風險，並無獨立真值證明它造成某個 full 的精度或 score 變化。

GNSS heading epoch variance 對 sensor prior yaw 的影響很小。就這批證據而言，XY reference rebase 和 correction-memory 的方向性 age 更需要優先檢查；這仍是依記錄與合成反例形成的工程判斷，不是 full score 的精確因果拆解。

## 建議的下一步（尚未實作）

1. 不要把 GNSS XY position／velocity innovation 當成外參 calibration delta 來抵消。若要重定錨，限定於明確建模的外參或 heading calibration 變化；另一條路是建立含 GNSS／map cross-covariance 的 joint state，而非把 `S` 當作完整 joint covariance。
2. correction-memory 的 age 與 sensitivity 依每個 observable mode 管理。部分方向更新只刷新相應方向的 memory；沒有觀測的方向須依不確定度逐漸降權。
3. 若保留 map-only likelihood，在原候選局部 trust region 內更新，並用同一組 associations／幾何品質重新驗 likelihood center，避免只靠 search bounds 接受局部近似外推。
4. 分別用生成資料驗證弱方向、長期部分觀測、加速與 rebase，再由使用者手動啟動新版本真實 full。不要用 Public Score 或非因果 CSV 擬合隱藏真值。

上述均為待開發方向，不是已實作的修正。本文件沒有宣稱 v5.3 已改善，也沒有新增定位或 full 的啟動資格要求。

## 可追溯證據

- 恢復 v5.3 + perf1 的 SHA 收據：[1735 apply receipt](../outputs/run/track00_test_2610311735/apply_receipt.json)。七個恢復檔案 SHA 與 full1612 source 相同，live version 為 v5.3。
- 以前五個真實 frame 做的加速／no-accel A/B：[1724 analysis README](../outputs/run/track00_test_2610311724/analysis/README.md)。CSV／TXT 相同；只涵蓋前五幀，不能外推整段 full。
- regression 整段分析：[1735 regression summary](../outputs/run/track00_test_2610311735/analysis/regression_summary.json)、[mean attribution](../outputs/run/track00_test_2610311735/analysis/mean_attribution.json)、[null rebase](../outputs/run/track00_test_2610311735/analysis/null_rebase.json)、[generated degenerate summary](../outputs/run/track00_test_2610311735/analysis/generated_degenerate_summary.json)。
- [1735 分析索引與圖表](../outputs/run/track00_test_2610311735/analysis/README.md)包含三組 PNG／PDF、數值明細與來源紀錄。
- 被回報 `0.80300` 的 v5.3 + perf1 full：[track02_full_2610311612](../outputs/run/track02_full_2610311612/test_info.json)。
- no-accel 歷史檢查與未加速版本細節：[doc75](75_TRACK2_V53_NO_ACCEL_20261031.md)。


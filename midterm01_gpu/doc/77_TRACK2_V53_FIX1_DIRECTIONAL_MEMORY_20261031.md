# Track2 v5.3 fix1：校正參考、方向性記憶與 map likelihood

> **歷史狀態提示（2026-11-02）：**本文只說明 fix1 與當時的生成證據。其後使用者回報 fix1 full `track02_full_2610311843` Public Score 為 `0.46899`；目前 live 已更新為 v5.3 fix2 + perf2，避免 registration 高度先驗重複融合。fix2 的高度模型、149 項生成／CUDA 檢查與限制見 [doc79](79_TRACK2_CAMERA_HEIGHT_POSITION_20261102.md)。本文的模型與數據保留為 fix1 歷史，不代表 fix2 的真實精度；fix2 尚無 full 或新分數。

## 目前狀態與目標

fix1 對應 doc76 指出的三項風險：GNSS XY Kalman update 被錯當成 calibration reference 變化、部分方向更新會重設整體 memory expiry、以及局部 map likelihood 只檢查搜尋界限而未以最終幾何 loss 驗收。修正於 `track00_test_2610311800/candidate_source` 完成，版本名稱為 `track2_causal_time_reference_likelihood_v5_3_fix1`，並已於 2026-10-31 18:11:48 依 SHA 保護套用 live。7 個來源檔 SHA 與通過檢查的 candidate 相符，保護範圍內 36 個檔案維持零變更。修正與生成驗證已完成；正式成績待使用者手動 full，v5.3 的成績目標尚未完成。

v5.3 + perf1 full `track02_full_2610311612` 的 Public Score `0.80300` 以及 v5.2 的 `0.46434` 都是使用者回報，且來自不同算法版本。使用者目標是朝 `0.3` 改善，但 fix1 尚無真實定位、full 或 Kaggle 分數，不能聲稱已改善，也不保證達標。非因果 CSV 不當真值、不用來擬合 offset、參數或隱藏軌跡。

目前 live 已是 fix1。完整定位仍由使用者手動啟動，命令保持簡單：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
./scripts/run_track2.sh --full
```

## 修正一：校正參考只追蹤校正量

原本的 reference rebase 會把 GNSS XY Kalman filter 的位置／速度狀態 innovation，當成校正座標的 reference change，再依 correction-memory sensitivity 抵消。可是 GNSS XY innovation 也可能是需要保留的當前位置與速度修正；沿某個方向沒有地圖觀測時，抵消會吃掉該方向的先驗更新。

fix1 將 `calibration_delta` 限定為真正的 heading calibration 變化與其引起的 lever-arm 位移。GNSS XY CV filter 的狀態 innovation 仍會傳播到當前 camera 時刻、更新 sensor prior 並留在 diagnostics，但不再拿來重定位或抵消既有 map correction-memory。這是修正座標校正與位置濾波語意混用，不是新 GNSS 量測，也不會回頭改舊 pose。

## 修正二：依 map 可觀測方向管理 memory age

### 為何要分方向

Point-to-plane map observation 可能只觀測 XY／yaw 的部分組合。例如一幀更新了 y/yaw，並不代表 x 方向也重新獲得幾何約束。舊版以單一 `last_update` 計時；任何 partial-rank 接受都重設全域 age，會讓未被本幀納入的弱方向 memory 長期不衰退。

### 年齡矩陣與重置

fix1 使用固定的初始 prior Cholesky 作為白化度量，建立半正定的方向年齡矩陣 (A)。每次 prediction 依經過時間累積：

\[
A \leftarrow A + \Delta t I.
\]

若當幀接受 map update，其觀測 row-space projector 為 \(P\)，將 age 投影到未觀測子空間：這會保留該子空間內的 age 項，並移除與已觀測子空間的交叉項。此投影在本幀 measurement 接受後執行，供下一幀計齡；本幀的衰退則先使用 prediction 累積後、尚未投影的 age：

\[
A \leftarrow (I-P)A(I-P).
\]

令固定 Cholesky 矩陣為 \(L\)，將 age matrix 分解為 \(A=U\operatorname{diag}(a_i)U^T\)。每個 eigenmode 的超時量與保留係數為

\[
\Delta\tau_i=\min(\Delta t,\max(a_i-h,0)),\qquad d_i=\exp(-\Delta\tau_i/\tau),
\]

其中 \(h=0.5\,\mathrm{s}\)、\(\tau=3\,\mathrm{s}\)。定義

\[
D=U\operatorname{diag}(d_i)U^T,\quad
M=LDL^{-1},\quad
B=LU\operatorname{diag}(\sqrt{1-d_i^2})U^TL^{-1}.
\]

衰退時更新 correction mean \(c\)、reference sensitivity \(S\) 及 covariance \(\Sigma\)：

\[
c\leftarrow Mc,\quad S\leftarrow MS,\quad
\Sigma\leftarrow M\Sigma M^T+BC_{\mathrm{prior}}B^T.
\]

兩個 covariance 項都是半正定，因此方向衰退不會引入負 variance；當所有模式的 \(d_i\) 相同，公式退化為舊式 \(d^2\Sigma+(1-d^2)C_{\mathrm{prior}}\)。衰退先於當幀 reference rebase 與 map measurement，避免把本幀新資訊一併衰退。

這個 age matrix 是工程記憶模型，不是每個特徵或狀態方向精確的 last-seen 真值。固定白化度量下，當 observable 子空間旋轉時，eigenmode 會混合方向年齡；`last_update` 為相容性保留，但不再控制所有方向的 expiry。hold `0.5 s` 與 tau `3 s` 數值不變。

## 修正三：map-only likelihood 以相同幾何 loss 驗收

Point-to-plane data Hessian／gradient 對 log-scale 做 Schur marginalization 後，fix1 也保留相應的 conditional log-scale step，使 map likelihood center 與尺度 nuisance 的局部條件一致。

從 MAP candidate 重建的 map-only likelihood center 必須同時符合兩項條件：

1. 在原始 registration search bounds 之內。
2. 在相同 final point-to-plane associations 上計算的實際非線性 data loss 不增加。

超出 bounds 與 loss 增加分開記錄拒絕原因。fix1 不將 center 與 prior 混合或 damping 來偷偷拉回界限；不符合驗收條件就不接受該 likelihood observation。這讓局部二次近似受原有搜尋範圍與同一組幾何資料共同限制，但仍不表示 associations、noise 或 prior correlation 已完全標定。

## 保留範圍

- 保留 v5.3 的 causal GNSS prior、map likelihood 架構、perf1 執行路徑與 Track2 GPU 配準。
- GNSS noise、原 correction-memory hold／tau 與其他數值參數維持原值；僅更新演算法版本識別。
- Track1、Track3 不修改。
- 只向前輸出當幀 pose，不回修既有 frame；fix1 不使用 RTS 或其他 offline smoothing。

## 生成檢查與尚未驗證項目

`track00_test_2610311800` 的 CPU 核心檢查為 **47 passed**。最終 `checks_summary.json` 記錄 **83 個 unique cases 通過、0 unresolved failures**。這是跨執行彙整，不是單次未重跑的 83/83：首次整組結果為 81 passed、2 failed，原因是舊 C++ oracle 未設定 `TRACK2_REFERENCE_LIBRARY`；補上已存在的 0154 oracle 後，只重跑受影響的 5 個 case（其中 2 個 oracle、3 個 CUDA／gate 相關），5 個全通過。

另有一個早期旋轉座標測試 fixture 將 pose／H 旋轉後未同步旋轉 covariance；這是 fixture 建立錯誤，不是定位失敗。fixture 已修正，原失敗 log 保留。這些都是生成／CUDA 機制檢查，並非真實定位精度或 Kaggle 分數驗證。

兩組 old-vs-fix 生成對照均為 322 state steps（加速行駛 201 步、memory hold／decay 121 步），不含真實影格：

| 生成場景 | 原始 v5.3 | fix1 | 能說明什麼 |
| --- | ---: | ---: | --- |
| 加速且 map 長期不觀測 x：moving x RMSE | `4.1484260583 m` | `0.1568270527 m` | 不再以舊記憶抵消 GNSS XY innovation、方向性 age 衰退兩項合併的生成效果；不是 Track2 RMSE |
| 同場景 10 s x 輸出 | `73.7319287698 m` | `80.9213252626 m` | 共用 sensor prior 是 `80.9213252626 m`，合成真值是 `81 m` |
| x 的方向性記憶 | `Sxx=0.995345449` | `Sxx=0.0487362543` | x 最後 map observation 在 `0.45 s`，至 `10 s` 已有 `9.55 s` 未觀測；兩支 global successful-update age 仍為 0 |
| 只有 y/yaw map 更新的 hold/decay | x correction 維持 `0.376470588 m` | 衰退到 `0.060190022 m` | y sensitivity 兩支都維持約 `0.9999999997`，示範只衰退未獲觀測的 x 方向 |

這些數據驗證固定生成條件下的機制差異，不是實際 Track 定位精度，也不拆解 Public Score。

fix1 尚無真實影格、full、Public Score 或以非因果資料校準的證據。83 個生成／CUDA 案例通過不證明 3299 幀的誤差，也不表示 `.80300` 已改善。套用收據與檢查摘要可查 [apply receipt](../outputs/run/track00_test_2610311800/apply_receipt.json) 及 [checks summary](../outputs/run/track00_test_2610311800/checks_summary.json)；生成圖見[行駛場景 PNG](../outputs/run/track00_test_2610311800/analysis/01_generated_motion.png)／[PDF](../outputs/run/track00_test_2610311800/analysis/01_generated_motion.pdf)與[方向性 hold PNG](../outputs/run/track00_test_2610311800/analysis/02_directional_hold.png)／[PDF](../outputs/run/track00_test_2610311800/analysis/02_directional_hold.pdf)，完整圖表導讀見[1800 analysis README](../outputs/run/track00_test_2610311800/analysis/README.md)。沒有新增 full 資格門檻；full 仍由使用者手動啟動。

## 可追溯紀錄

- fix1 source、套用與 CPU／CUDA 檢查：[track00_test_2610311800 test_info](../outputs/run/track00_test_2610311800/test_info.json)、[apply receipt](../outputs/run/track00_test_2610311800/apply_receipt.json)、[checks summary](../outputs/run/track00_test_2610311800/checks_summary.json)、[47 項 CPU pytest](../outputs/run/track00_test_2610311800/pytest_cpu_02.xml)、[首次整組 pytest](../outputs/run/track00_test_2610311800/pytest_final.xml)、[oracle／CUDA／gate 重試](../outputs/run/track00_test_2610311800/pytest_oracle_and_gate.xml)。
- generated inputs 與量測：[baseline](../outputs/run/track00_test_2610311800/analysis/baseline.json)、[fix1](../outputs/run/track00_test_2610311800/analysis/fix1.json)。
- 生成圖表與說明索引：[1800 analysis README](../outputs/run/track00_test_2610311800/analysis/README.md)。
- 修正前的 regression 數據與限制：[doc76｜v5.3 regression audit](76_TRACK2_V53_REGRESSION_AUDIT_20261031.md)。
- 現行 GNSS 航向模型：[doc72](72_TRACK2_GNSS_ORIENTATION_CAUSAL_20261029.md)；時間參考與 likelihood 背景：[doc74](74_TRACK2_TIME_REFERENCE_LIKELIHOOD_20261031.md)。


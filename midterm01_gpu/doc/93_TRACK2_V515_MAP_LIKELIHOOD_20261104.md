# Track2 v5.15：從 v5.10 改善 map likelihood 與 posterior 驗收

> 2026-11-04 後續實測：v5.15 full 使用者回報 **0.51729**，已確認退步；工作版已回退 v5.10。本文件保留歷史模型與生成檢查，當前診斷與 600 幀距離對照見 [doc94](94_TRACK2_V515_REGRESSION_RANGE_20261104.md)。


## 現況與操作

v5.15 以已評分的 v5.10 full 為基線，只加入三項地圖資訊處理：從 registration 資料項建立 map likelihood、保留正但較弱的 map 資訊方向，以及在同批原始影像特徵上驗收實際 Kalman posterior 的兩種成本不增加。目的在處理 v5.14 診斷中看到的 map rank 退化，以及候選／畫面狀態不能代表最終輸出符合資料成本條件的問題。

本輪 focused CPU/CUDA 生成檢查為 **33 passed、5.96 秒**。在 `V515=False` 且歷史分支關閉時，201 個 generated sensor frames 與 3 個 CUDA registration frames 的數值、state、covariance、memory、sensitivity 都與 frozen v5.10 EXACT 相同；147 個共同、非 metadata 參數值也相同。對照收據為 [`baseline_compatibility.json`](../outputs/run/track00_test_2611041845/baseline_compatibility.json)。

機制圖已生成於本輪 run 的 `analysis/`，只呈現合成資料：[單 prior](../outputs/run/track00_test_2611041845/analysis/01_single_prior.png)／[PDF](../outputs/run/track00_test_2611041845/analysis/01_single_prior.pdf)、[連續弱資訊](../outputs/run/track00_test_2611041845/analysis/02_continuous_weak_information.png)／[PDF](../outputs/run/track00_test_2611041845/analysis/02_continuous_weak_information.pdf)、[實際 gain 回溯](../outputs/run/track00_test_2611041845/analysis/03_actual_gain_backtracking.png)／[PDF](../outputs/run/track00_test_2611041845/analysis/03_actual_gain_backtracking.pdf)。它們不能代表真實定位精度。

完整 **Track2 CPU/CUDA suite：292 passed、1 項既有 Matplotlib Axes3D warning、46.89 秒**；完整結果見 [pytest_track2_final.xml](../outputs/run/track00_test_2611041845/pytest_track2_final.xml) 與[log](../outputs/run/track00_test_2611041845/pytest_track2_final.log)。warning 是載入既有 Matplotlib 3D 軸，不影響這組 2D 機制圖。尚無真實定位、真實 PnP 或新分數。v5.10 `0.42699`、v5.13 `0.42933`、v5.14 `0.43937` 都是使用者回報的歷史成績，不證明 v5.15 的精度或榜分變化。

合成機制數值也只驗證數學行為：資訊量為 `1`、prior variance `1`、target `0.8` 的例子中，舊流程把受先驗正則化的候選再融合一次，gain 平方後得到 `0.2`；改用 data-only likelihood 單次融合得到 `0.4`，符合解析後驗。這個例子驗證 prior 不重複計入，與弱資訊 rank 截斷無關。另一個 gain 回溯例以 `1/16` gain 通過驗收，成本由 `0.01` 降至 `0.00056465`。這些是合成數值，不是 Track2 精度或誤差。

完整 Track2 由使用者手動啟動：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
./scripts/run_track2.sh --full
```

本輪 v5.15 尚未執行真實 full；這是固定的手動操作入口，不表示已取得真實定位或榜分結果。

## 觀察與原因假說

v5.14 的逐段診斷發現 map 資訊 rank 可能退化；doc92 記錄 17:18–18:00（idx 1450–1546）多為 rank 1，29:07–30:21（idx 2773–2864）則有連續 92 幀沒有 map 更新。候選狀態、畫面標示與 filter 最終輸出是不同階段，需要直接檢查融合後的 pose 在同一批影像特徵上是否符合地圖成本條件。v5.15 保留弱正資訊可避免弱方向被硬丟棄，但無法在沒有有效幾何的區段產生定位資訊；診斷細節見[92｜v5.14 影片偏離診斷](92_TRACK2_V514_VIDEO_DEVIATION_20261104.md)。這提出三個需分別處理的問題：

1. registration 的 pose 候選受 sensor prior 影響。如果把這個候選再當成不含 prior 的獨立 map measurement 融合，prior 對候選的拉動可能被算第二次。
2. 將 prior-whitened map information 用單一 rank 門檻截斷，會連正但弱的方向一起丟掉；弱方向雖應有較小權重，仍可能保留有限資訊。
3. registration 候選在 solver 固定對應下的成本較低，不代表套用 Kalman gain 後的實際 posterior 仍符合地圖資料成本條件。實際輸出需要在相同原始特徵資料上重新驗收。

這些是 v5.15 改動所針對的機制問題，不是 v5.14 分數 `0.43937` 的唯一解釋。測試與 baseline 對照用來檢查機制和基線相容性，不提供真實精度證明。

## 使用的變數與模型

registration 的 map 資料項以局部二次模型表示。令 pose 變數為

\[
p = [x, y, \psi]^T
\]

其中 `x,y` 是平面位置、`\psi` 是 yaw；令 `\ell=\log(s)` 為 nuisance variable（需在 pose likelihood 中消去的尺度變數）。資料項在目前線性化點的增量 `\delta=[\delta p,\delta\ell]` 附近寫成

\[
E_{data}(\delta) \approx \tfrac12 \delta^T H_{data}\delta + g_{data}^T\delta,
\qquad
H_{data}=\begin{bmatrix}H_{pp}&H_{p\ell}\\H_{\ell p}&H_{\ell\ell}\end{bmatrix},\quad
g_{data}=\begin{bmatrix}g_p\\g_\ell\end{bmatrix}.
\]

候選求解仍可使用 sensor prior；轉成 map likelihood 時只用 `H_data`、`g_data`，不把 prior 項混進觀測。消去尺度增量後，pose 的資料資訊與梯度為 Schur complement：

\[
\bar H = H_{pp} - H_{p\ell}H_{\ell\ell}^{-1}H_{\ell p},
\qquad
\bar g = g_p - H_{p\ell}H_{\ell\ell}^{-1}g_\ell.
\]

這保留地圖資料對 `x,y,yaw` 的可觀測資訊，同時把尺度當成未知量邊際化，不將它固定後製造額外 pose 資訊。

### 保留弱的正資訊方向

令 prior covariance 為 `P=LL^T`。在 prior-whitened 座標中考察 map 資訊矩陣 `A=L^T\bar H L` 的 eigen-directions。每個特徵值 `\lambda_i` 表示該方向相對 prior 的資訊量：正值代表資料仍提供資訊；數值 null 代表該方向沒有可用資料資訊。v5.15 保留正但弱的 eigen-direction，並只略去數值 null。對一個已 whitening 的獨立方向，資訊 `\lambda_i` 對應的 Kalman gain 直覺可寫成

\[
k_i = \frac{\lambda_i}{1+\lambda_i}.
\]

因此弱正資訊自然只產生較小更新，毋須將它和完全無資訊方向一併丟棄。既有 strong threshold `1` 仍只判斷方向性 memory 是否刷新；weak update 不重設 strong-memory age。角度支持量下限仍為 `2(rank+1)`，`rank` 使用實際保留的正資訊方向數，threshold 不因保留 weak direction 而放寬。

### 驗收套用後的 Kalman posterior

令 `x^-`、`P^-` 為更新前狀態與 covariance，`K` 為 map likelihood 對應的 Kalman gain，`\nu` 為 innovation。回溯係數 `\alpha` 乘到整個 gain：

\[
K_\alpha=\alpha K,\qquad x_\alpha=x^-+K_\alpha\nu.
\]

對應 covariance 使用同一個實際 gain 的 Joseph form：

\[
P^+_\alpha=(I-K_\alpha H_m)P^-(I-K_\alpha H_m)^T
            +K_\alpha R_m K_\alpha^T,
\]

其中 `H_m,R_m` 是 map likelihood 的等效線性量測矩陣與 covariance。計算 mean、covariance 與 sensitivity 都使用同一個 `K_\alpha`，不讓 covariance 或 sensitivity 保留未回溯的完整 gain。

## v5.15 的三個改動

### 1. 從 data-only registration 建立 map likelihood

候選 registration 可受 sensor prior 拉動；融合時改從 map 資料項的 `data_H/data_g` 建立 likelihood，並依上式消去 log-scale。這樣送入 filter 的 map likelihood 不會把候選已使用的 sensor prior 再當成獨立 map 資訊融合一次。

### 2. 保留弱正資訊，但分開管理 memory refresh

prior-whitened 矩陣中，正且非數值 null 的方向都保留；弱資訊交由連續 Kalman gain 自然降低更新量。strong threshold `1` 僅作方向性 memory age 的刷新判據，弱方向更新不會重設 strong memory age。最少角度支持仍按保留後的實際 rank 計算 `2(rank+1)`。

### 3. 對實際 posterior 做固定與重新配對成本回溯

在同一批 raw feature sample 上，分別評估固定 plane associations 的成本，以及重新配對後、含 unmatched penalty 的成本。兩種成本都須符合 `C(candidate) <= C(prior) + 1e-12`；這是「不增加」的數值容忍，不要求嚴格下降。從完整 gain 開始；只要任一成本超過界線，就將 gain 乘 `1/2` 再試，最多回溯 14 次。接受時，mean、Joseph covariance 與 sensitivity 全部使用該次實際 `K_\alpha`。若回溯後仍有成本超出界線，便原子拒絕本次 map 更新，不提交 filter state，也不改 correction memory。weak mode 可造成極小但非零更新；通過成本驗收不代表精度改善。

這使驗收對象是濾波器實際會輸出的 posterior，且固定 plane 與重新配對兩種成本都須不增加；只有 solver 候選通過不夠。

## 基線、開關與範圍

正式來源基線是 v5.10 full `track02_full_2611031941`；其 116 個 Python snapshot SHA 已完整核對。v5.15 分支要求 `V510_ENABLED=True`、`V511_ENABLED=False`、`V512_ENABLED=False`、`V5121_ENABLED=False`、`V513_ENABLED=False`、`V514_ENABLED=False`、`V515_ENABLED=True`。只啟用 v5.10 基線與本輪三項 map 改動。

以下沿用 v5.10，並非本輪調參：GNSS 模型、原有 validated-ground scalar speed、初始化搜尋、height／scale 數值模型、40 m feature range、6000-point 原抽樣、既有品質門檻、共同物理數值參數及因果輸出。本輪不改 GNSS position model。

本輪沒有移植 v5.11 多起點、v5.12／v5.12.1 的完整分支、v5.12.1 GNSS／側向步長預算、v5.13／v5.14 的影像速度回退。這是從 v5.10 基線選擇性整合 map likelihood 數學機制；不等同重新啟用整個歷史 v5.12 或 v5.4 feature bundle。那些歷史版本同時包含多項差異，其測試與榜分不能直接當成本輪三項改動的因果證據。

## 驗證狀態與限制

| 檢查項目 | 狀態 | 可支持的結論 |
| --- | --- | --- |
| focused CPU/CUDA 生成檢查 | **33 passed，5.96 秒** | 針對性生成機制案例通過；不代表真實精度 |
| v5.10 frozen baseline 相容性 | 通過 | `V515=False` 時 201 個 generated sensor frames 與 3 個 CUDA registration frames 的數值、state、covariance、memory、sensitivity 完全相同；147 個共同非 metadata 參數值相同；[baseline compatibility](../outputs/run/track00_test_2611041845/baseline_compatibility.json) |
| 完整 Track2 CPU/CUDA suite | **292 passed、1 項既有 warning、46.89 秒** | [JUnit XML](../outputs/run/track00_test_2611041845/pytest_track2_final.xml) 與[log](../outputs/run/track00_test_2611041845/pytest_track2_final.log)；warning 是 Matplotlib Axes3D 載入 |
| 本輪測試 run | `outputs/run/track00_test_2611041845/` | baseline compatibility 與三組機制圖已保存；圖只供合成檢查 |
| Track2 真實定位／真實 PnP | 0 幀／0 次 | v5.15 尚無真實 full 或 PnP 結果 |
| v5.15 Public Score | 尚無 | 不宣稱榜分或精度改善 |

本文件不把 focused／完整 suite 通過或 synthetic cost 變化寫成真實精度改善。frozen v5.10 相容性檢查與完整 Track2 CPU/CUDA suite 已通過；真實 full、PnP 與 Public Score 仍未取得。

map covariance 尚未獨立校準，錯誤配對也可能降低資料成本；doc92 中兩條灰色道路線不作車道真值，也不作強制貼線約束。

### 基線與歷史分數

- v5.10：full `track02_full_2611031941`，使用者回報 `0.42699`；本輪選用的正式算法／來源基線，116 個 Python snapshot SHA 完整。
- v5.13：full `track02_full_2611041345`，使用者回報 `0.42933`。
- v5.14：full `track02_full_2611041506`，使用者回報 `0.43937`。

以上是各版使用者回報的 Public Score；本輪沒有新成績。較早版本涉及不同算法組合，分數不能單獨用來推斷 map likelihood 的因果效果。v5.14 影片分析見[92｜v5.14 影片偏離診斷](92_TRACK2_V514_VIDEO_DEVIATION_20261104.md)。

<details>
<summary>驗證與來源附錄</summary>

首輪完整 Track2 CPU/CUDA suite 的 292 項中有 291 passed、1 個舊 weak-mode rank assertion failed，更新成 v5.15 rank 3／strong rank 0 與 V515 關閉時 legacy rank 0 後，同範圍重跑為 292 passed、1 warning、46.89 秒；另一次全 repo 診斷不宣稱通過，未修改的 CPU 參考轉檔 `midterm01/others code track1/scripts/convert_map.py` 因容器缺少 `polars` 而在 import 時失敗，與 Track2 算法無關，失敗／中斷紀錄保存在本輪 run。

本輪凍結 140 個 Python source snapshot，21 個歷史 full 的原始 SHA 仍相符；run metadata 見 [test_info.json](../outputs/run/track00_test_2611041845/test_info.json)。

</details>

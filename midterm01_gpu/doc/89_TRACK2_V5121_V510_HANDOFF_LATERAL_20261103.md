# Track2 v5.12.1：限制首筆 GNSS 與地圖更新的側向跳動

v5.12.1 以表現最好的 **v5.10** 凍結來源為數值基準，修正 Track2 更新流程，目標是處理兩種輸出不連續：首批 pose 輸出後才到達的位置資訊造成的大步長，以及 GNSS、reference rebase、map 各自一小步在車身側向累積成可見折線。這次限制 GNSS／map measurement 實際寫回的 gain，保留 Joseph covariance update，並另行限制校準反應；沒有對舊軌跡做平滑或事後修正。

**狀態：程式已凍結並套用。**最終安全 CPU subset 25 項通過（17 個新機制案例加原有 8 個 directional-memory cases，2.02 秒）；完整 CUDA suite 247 passed、1 項既有 Axes3D warning，32.34 秒。凍結 v5.10 generated baseline 檢查通過；123 個 Python source 的 SHA 已凍結。v5.12.1 沒有真實定位幀、PnP 幀或新分數。

## 目前版本和歷史分數

目前候選為 `track2_v510_causal_handoff_lateral_budget_v5_12_1_perf2`，設定 `V511_ENABLED=False`、`V512_ENABLED=False`、`V5121_ENABLED=True`。它以 v5.10 full `track02_full_2611031941` 的凍結來源為基線，保留 v5.10 的單一起點選擇、MAP center／filter、course／heading 預算、height 與 image 流程；v5.11／v5.12 的程式仍留在來源中供歷史測試，但這個候選不啟用。

| 版本 | 可確認的結果 | 解讀 |
|---|---|---|
| v5.10 `track02_full_2611031941` | 3299 幀；Public Score `0.42699`，使用者回報 | 目前已知最佳完整分數；本候選基線 |
| v5.11.1 `track02_full_2611032114` | 3299 幀；Public Score `0.43685`，使用者回報 | 比 v5.10 分數差；相同啟動 GNSS 側向跳動仍出現在診斷中 |
| v5.12 `track02_full_2611032240` | 使用者停止時已有 395 幀輸出（idx 0–394）；沒有分數 | 不完整 run；不能當成正式版本成績 |
| v5.12.1 候選 | CPU 25 passed；GPU 247 passed、1 項既有 warning；generated baseline 精確核對通過 | 已套用、沒有真實定位或新分數 |

兩張 RViz 截圖的標題都指向 `track02_full_2611032114`。畫面 scale `8.488` 與 `7.514` 各自對應多個 frame，不能用 scale 唯一定位截圖幀；以下按保存的 pose／observation 逐幀紀錄分析，不宣稱每個統計事件就是截圖中的那個時刻。

## 為什麼啟動會折出一段 Y？

舊啟動流程先輸出 prior pose，之後第一次收到 GNSS position 時，會把當前 CV 濾波器中的整筆位置 innovation 一次用掉。v5.10 的 frame 2 位移長度為 **0.6512 m**，其中車身側向分量 **−0.2301 m**；逐幀分解將這項移動記在 GNSS sensor update，該幀 map update 為 0。frame 0 和 frame 1 沒有這筆更新，所以畫出的相鄰線段先沿舊 prior，之後立即朝 GNSS 修正方向折轉。

這也說明前一輪為什麼沒有消掉啟動折線：v5.11.1 限制的是首筆 map update，在 frame 1 的 handoff 步長確實只有約 `0.01861 m`；frame 2 才到達 GNSS position，仍移動 `0.65251 m`、側向 `−0.23142 m`，map update 仍為 0。v5.12 的 data-only map likelihood 與重新配對驗收處理的是 map proposal，不是這筆延遲 GNSS likelihood；其被使用者停止的 run 在 frame 2 仍移動 `0.65248 m`、側向 `−0.23127 m`。因此這段折線來源可以從逐幀輸出追到 GNSS update，而非據圖猜初始 pose 錯了。

以上的側向值是估計 trajectory 相鄰步長在當幀車身側軸的投影；它能指出哪個更新造成折線，不能當成真實側向定位誤差或 ground truth。

## 標記路段的舊輸出還揭露兩種側向步長來源

| 來源與 frame | 側向分量 | 當幀記錄預算 | 逐幀記錄說明 |
|---|---:|---:|---|
| v5.10 frame 377 rebase | `+0.0535 m` | map lateral budget `0.00844 m` | calibration reference 改基準帶來側向步長 |
| v5.10 frame 418 rebase | `+0.0504 m` | `0.00788 m` | 同類 rebase 步長；v5.11.1 分別為 `+0.0603`／`+0.0625 m` |
| v5.10 frame 2154 map update | `+0.0623 m` | map lateral budget `0.00885 m` | 實際 map 側向修正明顯高於分開記錄的側向預算 |
| v5.10 frame 2865 map update | `+0.0615 m` | `0.00834 m` | 相同類型；v5.11.1 frame 2865 仍約 `+0.0615 m` |
| v5.10 frame 1432 GNSS | `+0.1601 m` | map update `0` | 此步長來自 GNSS position update，不是 map registration |
| v5.10 frame 2343 GNSS | `+0.1833 m` | map update `0` | 同樣在沒有 map update 的幀向左移動 |

這些案例跨不同來源：首筆或新到的 GNSS position、航向校準參考基準改變後的 `reference_sensitivity` rebase、以及 map measurement update。只調小一個共同總步長，會把原因混在一起，也可能破壞原本沿行進方向需要的修正能力。

frame 650–730 的累積分解則顯示抵銷後仍留下的側向路徑變化約 **`+0.1605 m`**：逐幀車身側向投影加總為 GNSS `+0.0316 m`、correction decay `−0.0591 m`、rebase `−0.0248 m`、map update `+0.2128 m`。這指出該窗有多種更新相加與互相抵銷，map 是最大的正向項；它是舊 filter 各項輸出的帳，不是車實際偏離道路中心的真值，也不能僅由此判斷每一幀肉眼方向。

每個事件都可由唯讀逐幀診斷檔追查： [v5.10 run](../outputs/run/track00_test_2611032253/analysis/track02_full_2611031941_motion.json)、[v5.11.1 run](../outputs/run/track00_test_2611032253/analysis/track02_full_2611032114_motion.json)、[停止於 395 幀的 v5.12 prefix](../outputs/run/track00_test_2611032253/analysis/track02_full_2611032240_motion.json)。這些是既有輸出拆解，不曾重跑定位；frame index 只用於讀出當幀成分，不會寫回 pose。

## v5.12.1 如何限制這些更新

### 延遲到達的 GNSS position 分幀使用

初始化沿用單一 2D GNSS position likelihood 和共同 power，讓兩軸 innovation 同比例縮放，保留原始修正方向；尚未完整吸收兩軸資訊時不先逐軸套用 radial gain 限制。完成初始化後，tracking 才把同一筆等向 XY covariance `R` 沿當下兩個正交方向分解成兩個 1D likelihood，各軸各自跨時間消耗 power、總量不超過 1。同一量測時刻、相同 `R` 且完整消耗 power 時，兩個 1D factor 的合併在 likelihood 代數上精確等價於原 2D observation；這只是重新表達同一筆資料，不會增加資訊。

跨多幀實際消耗時會按量測 age 加入 `q·age³/3` covariance inflation，過期時丟棄未使用 power，而且 process／state 關聯沒有完整估計。因此跨幀更新後的 posterior 不保證等於在原量測時刻一次做完整 2D Kalman update。正常 tracking 保留 GNSS 前向增益，不因側向限額一起壓掉沿行進方向的修正。

狀態採當前 CV 位置與速度 `x=[p,v]`。初始化尚使用共同 2D likelihood 時，若 GNSS 量測較目前影格早 `age` 秒，使用 `H=[I, -age·I]` 對應回量測時刻的位置；tracking 沿方向列向量 `d` 使用 `H_d=dᵀ[I, -age·I]`。兩種情況都對量測 covariance 加上 `q·age³/3` 的過程不確定度。這是保守近似，沒有完整保留 position／velocity 的 joint correlation。若這項 age uncertainty 已大於原 GNSS 位置 variance `0.3² m²`，該方向尚未用完的 power 就過期，不宣稱每筆延遲 GNSS 資訊都能被採用。

只有在初始化階段、尚未完整吸收兩個正交位置分量時，才對 GNSS 徑向修正使用 `0.5 m/s` 基準，加上既有 speed-fraction／turn-lever allowance。正常 tracking 不限制 GNSS 前向增益，保留速度改變所需的更新。GNSS 側向預算在既有 `0.2 m/s × dt` 與 turn-lever allowance 外，加入 heading／IMU／外參 yaw uncertainty 乘上 `speed × dt` 的項；GNSS CV update 仍納入實際加速度，不假設車身軸已完全校準。這是 GNSS 位置狀態的物理預算，不改 GNSS course 樣本、時間或 heading 更新。

### 將 radial、lateral 與 map 修正分開設上限

- 初始化未完整吸收兩軸資訊時，GNSS radial `0.5 m/s` 基準與既有 allowance 限制共同 2D power，兩軸同倍率以保留 innovation 方向；完整吸收後 tracking 不限前向 gain。GNSS lateral 預算包含 heading／IMU／外參 yaw uncertainty 對 `speed × dt` 的影響、`0.2 m/s × dt` 基準及 turn-lever allowance。預算限制 likelihood 經原 CV filter 計算後的實際 gain，不裁切或平滑最後輸出的軌跡座標。
- 首幀之後才到達的第一筆 map update 也納入當幀預算。Map lateral budget 與 radial／forward speed 分開計算，不讓前進速度放大可橫向移動量；並保留 map data 實際觀測到的 row space，不在未觀測方向製造更新。這和 GNSS 的速度相關側向物理誤差模型不同。舊來源雖有這條 lateral budget 分支，production v5.10／v5.2 path 未套用；v5.12.1 才啟用明確限制。
- 三種 measurement 的有限 gain 都在原 filter update 內完成，之後按 Joseph form 更新 covariance；沒有在 covariance 之外改寫 pose。
- `reference_sensitivity` 改在 prior-whitened coordinates 中以 PSD contraction 限制。這限制 yaw 校準交叉反應對 XY rebase 的放大；它是白化座標中的反應界限，不是每幀公尺側移的硬上限，也不是完整 joint EKF。當幀 map gain 的 row-space 限制仍由原 projector 負責。

GNSS course／heading 使用的原始樣本、timestamp 與更新預算未改。Map solver 的單一起點、MAP center／filter、高度與影像流程也保留 v5.10 基線。需要檢視版本開關與因果假設時，見凍結候選中的 [`track2/parameters.py`](../track2/parameters.py) 及 [Track2 操作說明](../track2/README.md)。

## 驗證與限制

### 首輪 GPU 回歸與修正紀錄

第一版 GNSS gain 設計把一筆 2D position likelihood 的 power 一起縮小，想限制側向跳動，結果也連帶降低前向加速修正。首輪 CUDA suite **1 failed、242 passed、1 項既有 warning，34.53 秒**；失敗的是原有 `test_acceleration_gnss_innovation_is_not_cancelled_by_unobserved_map_x`，生成加速案例的 RMS 為 **25.086 m**。這是候選設計的真實 regression，該版未套用；沒有跳過或放寬此測試。

首輪修正把原 2D power 拆為正交 1D likelihood，但初始化時依序吸收兩軸可能先走一段前向、再走一段側向。最終候選改為初始化共用一個 2D XY power，以同一比例縮放兩軸並保留 innovation 方向；初始化完整吸收後 tracking 才拆軸消耗，另加入 collinearity 生成案例。GNSS 側向額度納入 heading／IMU／外參 yaw uncertainty 隨 `speed × dt` 增加的項，另保留 `0.2 m/s × dt` 和 turn-lever allowance；不假設車身軸完全校準。正交 factor likelihood 等價案例與生成運動案例（15° heading uncertainty 下加速，以及 `15 m/s`、轉彎半徑 `100 m`）均要求 RMS 小於原 GNSS `0.3 m` 標準差。

首輪、修正中版本與最終候選的測試證據如下；更早的 13 項 CPU 測試屬於第一候選，不能當成目前驗證結果：

| 證據 | 結果 | 記錄 |
|---|---|---|
| 第一版完整 GPU suite | 1 failed、242 passed、1 warning，34.53 秒 | [XML](../outputs/run/track00_test_2611032253/pytest_gpu.xml) · [log](../outputs/run/track00_test_2611032253/pytest_gpu.log) |
| 早期第一候選 CPU suite | 13 passed，0.37 秒；非目前版本驗證 | [XML](../outputs/run/track00_test_2611032253/pytest_cpu.xml) · [log](../outputs/run/track00_test_2611032253/pytest_cpu.log) |
| 修正前 CPU subset | 21 passed，1.73 秒；屬於中間版本 | [XML](../outputs/run/track00_test_2611032253/pytest_cpu_0002.xml) · [log](../outputs/run/track00_test_2611032253/pytest_cpu_0002.log) |
| 中間候選完整 GPU suite | 246 passed，1 項既有 warning，32.64 秒 | [XML](../outputs/run/track00_test_2611032253/pytest_gpu_0002.xml) · [log](../outputs/run/track00_test_2611032253/pytest_gpu_0002.log) |
| 修正後 CPU subset | 24 passed，2.12 秒；16 個新機制案例加 8 個原 directional-memory cases | [XML](../outputs/run/track00_test_2611032253/pytest_cpu_0003.xml) · [log](../outputs/run/track00_test_2611032253/pytest_cpu_0003.log) |
| 最終候選 CPU subset | 25 passed，2.02 秒；17 個新機制案例加 8 個原 directional-memory cases | [XML](../outputs/run/track00_test_2611032253/pytest_cpu_0004.xml) · [log](../outputs/run/track00_test_2611032253/pytest_cpu_0004.log) |
| 最終候選完整 GPU suite | 247 passed，1 項既有 Axes3D warning，32.34 秒；123 個 Python source 已凍結 | [XML](../outputs/run/track00_test_2611032253/pytest_gpu_0003.xml) · [log](../outputs/run/track00_test_2611032253/pytest_gpu_0003.log) · [Python manifest](../outputs/run/track00_test_2611032253/candidate_python_manifest.json) |

### 凍結 v5.10 generated baseline 與 stationary 合成檢查

最終 baseline check 通過，耗時 2.594940793 秒。停用全部 v5.12.1 開關時，147 個舊參數除 `VERSION` 與 `BASELINE_RUN_ID` 這兩個 metadata 欄位外皆相同；3 個生成配準幀的 pose、scale、height、height variance、accept、covariance、correction 都逐值等於 v5.10。另有 201 個生成 sensor 幀的 pose、state、covariance、heading bias、heading covariance 與 truth 逐值等於 v5.10。三項機制各單開、以及全部同開的四組生成檢查均通過；加速案例 RMS `0.156827 m`，各正交軸累積 power 均不超過 1。這些是生成機制與相容性證據，不是真實 Track 精度證明。

generated baseline 第一次比較曾錯把 `BASELINE_RUN_ID` metadata 當數值參數而 AssertionError；檢查器後來排除 `VERSION`／`BASELINE_RUN_ID` metadata，保留[原失敗 log](../outputs/run/track00_test_2611032253/generated_baseline.log)，不將它記成算法回歸或成功。第二輪中間候選的 246 GPU suite 及 baseline 都通過（32.64 秒與 2.557614267 秒），詳見 [GPU XML](../outputs/run/track00_test_2611032253/pytest_gpu_0002.xml)、[baseline JSON](../outputs/run/track00_test_2611032253/analysis/generated_baseline_second_gpu.json) 和 [log](../outputs/run/track00_test_2611032253/generated_baseline_0002.log)；但其後又加入啟動共用 2D power 與 collinearity test，因此只以最終第三輪的 suite、manifest 和 baseline 記錄代表凍結候選。

最終 baseline artifacts： [JSON](../outputs/run/track00_test_2611032253/analysis/generated_baseline_final.json) · [log](../outputs/run/track00_test_2611032253/generated_baseline_0003.log)。只有在停用全部 v5.12.1 機制時，生成輸出才與 v5.10 exact；這不是啟用 v5.12.1 後真實定位相同的證明。

另有合成 GNSS 加獨立 image-stationary cue 檢查，耗時 `0.205555672 s`：生成案例 first-GNSS step 從 v5.10 的 `0.672080 m` 降至候選的 `0.0119588 m`；候選最大 step 為 `0.0121502 m`，末端 speed 為 `3.26649e-6 m/s`。這是生成 case 的結果，不是實際 frame 2 或真實定位改善。沒有獨立 image-rest cue、純 GNSS 啟動仍可能慢收斂並有小幅 overshoot；演算法抑制突變，不強行把軌跡拉直，也不保證所有啟動都無折線或延遲。唯讀的 v5.10 image-stationary 觀測在 frames 7、14、21、28、35、42 已接受；該 image module 本次未修改。詳見 [generated stationary JSON](../outputs/run/track00_test_2611032253/analysis/generated_stationary.json) 與 [log](../outputs/run/track00_test_2611032253/generated_stationary.log)。

候選也加入 full analysis 的 optional `11_position_information.png/pdf` 與 factor summary：按實際 position policy 顯示每個 factor 的總 power 和過期未用資訊。候選已套用，這些圖尚未由本次 full run 產生；它們診斷資訊消耗，不是精度圖。

本輪唯讀診斷圖表可查看 [startup sources PNG/PDF](../outputs/run/track00_test_2611032253/analysis/01_startup_sources.png)／[PDF](../outputs/run/track00_test_2611032253/analysis/01_startup_sources.pdf)、[screenshot candidates PNG/PDF](../outputs/run/track00_test_2611032253/analysis/02_screenshot_candidates.png)／[PDF](../outputs/run/track00_test_2611032253/analysis/02_screenshot_candidates.pdf)、[lateral sources PNG/PDF](../outputs/run/track00_test_2611032253/analysis/03_lateral_sources.png)／[PDF](../outputs/run/track00_test_2611032253/analysis/03_lateral_sources.pdf)、[generated handoff PNG/PDF](../outputs/run/track00_test_2611032253/analysis/04_generated_handoff.png)／[PDF](../outputs/run/track00_test_2611032253/analysis/04_generated_handoff.pdf)、[generated stationary PNG/PDF](../outputs/run/track00_test_2611032253/analysis/05_generated_stationary.png)／[PDF](../outputs/run/track00_test_2611032253/analysis/05_generated_stationary.pdf)，及 [lateral diagnosis JSON](../outputs/run/track00_test_2611032253/analysis/lateral_diagnosis.json)、[screenshot candidates JSON](../outputs/run/track00_test_2611032253/analysis/screenshot_candidates.json)。其中 generated plots 都是合成驗證，不可作定位精度證據。

目前候選沒有新真實定位或 PnP 幀，也沒有新的 public score；使用者回報的 v5.11.1 `0.43685` 與 v5.10 `0.42699` 仍是歷史成績。

已知限制：`q·age³/3` 是延遲位置誤差近似，不等於完整的 joint position／velocity covariance；GNSS position covariance、GNSS／camera 時間與外參沒有因此完成標定。各步驟的 body-lateral contribution 只描述 estimator 當下移動，不代表道路中心或 ground truth。這次沒有用使用者標記 frame 寫入特例，也沒有用 `result_2_0.30488.csv` 擬合位置、yaw、時間偏移或更新參數。

程式已套用並完成機制驗證；完整定位仍由使用者手動啟動：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
./scripts/run_track2.sh --full
```

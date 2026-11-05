# Track2 v5.12：地圖似然與重新配對驗收

v5.12 針對兩個已在程式模型中確認的問題補強 map update：同一筆 GNSS 資訊可能先進入配準 prior，再經 MAP 候選重複進入濾波器；而舊驗收只在固定的最終平面配對上算殘差，可能放過移動後更靠近錯誤地圖表面的 pose。候選版本改用不含明確 sensor／height prior 項的平面資料似然，並在實際有限 gain posterior pose／scale 上同時計算固定配對與重新配對的資料成本。這保留原 rank 門檻、height MAP 與各感測器更新預算。

**本文件狀態：程式已實作並套用；最終 GPU suite 230 passed，1 項既有 Matplotlib `Axes3D` warning，46.37 秒；候選來源 121 個 Python SHA 已凍結。**suite 對應 [JUnit XML](../outputs/run/track00_test_2611032122/pytest_gpu_0003.xml)、[log](../outputs/run/track00_test_2611032122/pytest_gpu_0003.log) 與 [source manifest](../outputs/run/track00_test_2611032122/candidate_python_manifest.json)。生成 ablation 亦已通過；v5.12 已套用；沒有真實 v5.12 定位幀、PnP 幀或新分數。以下把程式問題、v5.10 歷史紀錄與照片形成的風險假說分開；任何一項都不能把標記座標當成 ground truth。

## 先前承諾的 v5.11 機制有沒有實作？

有。v5.11.1 full `track02_full_2611032114` 已自然完成 3299 幀，定位耗時 2522.718837578 秒，凍結來源 186 SHA 已核對。前 300 筆紀錄中 startup 的前兩個候選集合各完成 12 個候選求解；tracking 有 212 幀嘗試 alternate 起點；有限 gain map proposal 驗收執行 90 次、拒絕 5 次。這證明三項機制有進入執行路徑；不是完整精度比較。原始計數在 [implemented_mechanisms.json](../outputs/run/track00_test_2611032122/implemented_mechanisms.json)，run 的完整狀態在 [test_info.json](../outputs/run/track02_full_2611032114/test_info.json)。

這回答的是「程式有沒有執行」，不是「標記幀有沒有變準」。在使用者標記的 13 個異常窗中，alternate retry 都沒有救回 geometry failure；穩定對照 800–1000 則有一幀（997）救回，1485–1515 沒有。這說明舊 retry 對這些異常窗沒有產生 geometry rescue，不能據此推斷失敗原因一定是起點太接近。詳細逐窗 frame list 在 [v5111_target_mechanisms.json](../outputs/run/track00_test_2611032122/v5111_target_mechanisms.json)。首筆 map 交接在同一次 full 實際移動 0.0186075116455 m，和可用預算 0.0186075116456 m 相符；這驗證 handoff 限幅有生效，不是精度證明。v5.11.1 的 handoff 修正另見 [87｜v5.11.1 啟動交接修正](87_TRACK2_V511_STARTUP_VALIDATION_20261103.md)。

## v5.10 標記路段呈現哪些差異？

比對來源是 v5.10 full `track02_full_2611031941` 的 3299 筆 observations 與 pose trace。異常中心依使用者標記為 700、1150、1220、1250、1330、1860、2146、2165、2395、2575、2715、3025、3290；每段取中心前後各 15 幀，序列末段 3290 因已到最後一幀而只有 24 幀。穩定對照區為 800–1000 和 1485–1515。

下表是逐窗的 v5.10 診斷摘要。rank、coverage、plane residual、mode fraction 都只描述演算法輸入、解空間或對地圖的局部貼合，並非真實位置誤差。mode fraction 是白化後資訊空間中該方向被保留的比例，不是方向精度百分比。`prior share*` 是理想高斯模型下的搜尋先驗比例，不能解讀成舊 full 實際把量測縮了多少。

| 區段中心 | rank 0/1/2/3 計數 | map 接受／幀數 | plane residual 中位數 (m) | features 中位數 | coverage 中位數 | 側向／yaw 保留 fraction | λ；理想 prior share* |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 700 | 12/17/2/0 | 18/31 | 0.113 | 5732 | 0.132 | 0.007 / 0.047 | 1.29；0.436 |
| 1150 | 2/28/1/0 | 24/31 | 0.096 | 1660 | 0.114 | 0.828 / 0.123 | 1.61；0.384 |
| 1220 | 1/30/0/0 | 29/31 | 0.058 | 1558 | 0.152 | 0.758 / 0.163 | 2.18；0.315 |
| 1250 | 0/30/1/0 | 27/31 | 0.059 | 1924 | 0.154 | 0.830 / 0.097 | 2.96；0.253 |
| 1330 | 0/13/17/1 | 23/31 | 0.079 | 1158 | 0.199 | 0.787 / 0.328 | 5.73；0.149 |
| 1860 | 9/20/2/0 | 14/31 | 0.127 | 3190 | 0.077 | 0.637 / 0.246 | 2.72；0.269 |
| 2146 | 4/13/13/1 | 26/31 | 0.211 | 6000 | 0.147 | 0.634 / 0.328 | 2.55；0.281 |
| 2165 | 0/5/25/1 | 29/31 | 0.197 | 6000 | 0.108 | 0.834 / 0.194 | 2.47；0.288 |
| 2395 | 0/30/1/0 | 22/31 | 0.126 | 1509 | 0.174 | 0.651 / 0.289 | 3.79；0.209 |
| 2575 | 0/31/0/0 | 28/31 | 0.041 | 1043 | 0.233 | 0.941 / 0.036 | 5.74；0.148 |
| 2715 | 9/21/1/0 | 17/31 | 0.105 | 2352 | 0.043 | 0.397 / 0.126 | 1.61；0.383 |
| 3025 | 0/29/2/0 | 28/31 | 0.126 | 821 | 0.258 | 0.771 / 0.148 | 2.56；0.281 |
| 3290 | 5/16/3/0 | 15/24 | 0.064 | 407 | 0.161 | 0.891 / 0.003 | 3.52；0.221 |
| 穩定 800–1000 | 80/120/1/0 | 99/201 | 0.137 | 2151 | 0.101 | 0.766 / 0.117 | 9.15；0.099 |
| 穩定 1485–1515 | 0/29/2/0 | 25/31 | 0.032 | 835 | 0.290 | 0.943 / 0.044 | 11.96；0.077 |

完整幾何診斷、尺度、GNSS、filter reject、finite gain、收斂及 reference shift 欄位見 [marked_windows_stats.md](../outputs/run/track00_test_2611032122/marked_windows_stats.md)；每幀 mode 摘要見 [modes_diagnosis.json](../outputs/run/track00_test_2611032122/analysis/modes_diagnosis.json)。標記影像與合成反例圖見 [01_marked_images.png](../outputs/run/track00_test_2611032122/analysis/01_marked_images.png)、[02_modes_and_residuals.png](../outputs/run/track00_test_2611032122/analysis/02_modes_and_residuals.png)、[03_generated_model_checks.png](../outputs/run/track00_test_2611032122/analysis/03_generated_model_checks.png) 和 [分析索引](../outputs/run/track00_test_2611032122/analysis/README.md)。

### 可以支持的解讀

- 秩數不能單獨標記好壞：偏移區段與穩定區段都大量出現 rank 1。800–1000 甚至有 80 幀 rank 0，仍是使用者觀察到的穩定對照；因此沒有加一刀切的低 rank／coverage／點數拒絕規則。
- 700 的側向 fraction 中位數僅 0.007，地圖平面對側向位移幾乎沒有保留有效資訊；2575 的側向 fraction 高達 0.941，但 yaw fraction 只有 0.036。這與「看起來偏左」和「yaw 可疑」分屬不同解空間方向相容，卻不能由此量出真實偏差。
- 1150、1220、1250 的 λ 中位數約 1.61、2.18、2.96，低於穩定控制 800–1000 的 9.15 和 1485–1515 的 11.96；對應 ideal prior share* 較高。這表示資料似然相對搜尋先驗較弱的假說值得檢查，並非舊輸出曾被實測縮小 38%、31%、25%。
- 2146／2165 的 plane residual 中位數為 0.211／0.197 m，coverage 0.147／0.108；同窗 features 都到 6000。照片可見高架橋與卡車，這些動態物體或錯誤關聯是風險假說，不是已證實的錯配來源。穩定的 1485–1515 只有 835 features，但 residual 0.032 m、coverage 0.290，說明不能單靠點數門檻判好壞。
- 使用者把 3290 標為左偏。該尾段 yaw fraction 中位數僅 0.003，features 407，scale validated 只有 24 幀中的 8 幀；這表示地圖幾乎無法提供 yaw 校驗能力，不能據此解釋或排除側向左偏。
- 這些指定窗口中，image-speed accepted 都是 0；但穩定對照窗口同樣為 0，所以它是目前沒有發揮的資訊來源，不能單獨解釋為何只有某些區段偏移。GNSS innovation 全窗 median 也會被大量「沒有新 GNSS 樣本」的零值稀釋；須看非零樣本和 timestamp 變化。700 窗口有 4 次非零 innovation，median 0.093 m、max 0.137 m。

候選修正聚焦能由程式與合成反例證明的資料融合漏洞，不對每個標記 frame 加座標、yaw 或 threshold 特例，也不把使用者標記或歷史 0.30488 CSV 當成真值。v5.10 舊 observation 有保存 `data_H`，但沒有保存 `data_g`、固定 association 及可重算的固定／重新配對 posterior costs，無法逐幀重建新版兩道 gate 的接受結果。

## v5.12 修正的兩個模型問題

### 1. 把含 prior 的 MAP 候選當成另一筆獨立 map 量測

舊 solver 求的是以當幀 sensor／height prior 正規化的 MAP 候選。後續濾波又將該 MAP center 當 map observation，等於可能把同一個 prior 的資訊用兩次。這會讓 fused update 重複朝 sensor prior 計入資訊，使 map 修正相對不足；不是過度相信 map proposal。

v5.12 的候選在既有固定當幀 associations 上，以 point-to-plane data Hessian／gradient `H_data, g_data` 建立 likelihood，移除明確的 sensor／height prior 項；log-scale nuisance 仍按既有 Schur 消去，再交給原本的因果 correction filter。data-only center 不直接作為輸出 pose。它仍由當幀 prior 與 height seed 選擇 association 和樣本，跨幀與高度相關性也沒有被完整估計，所以不能宣稱統計上完全獨立或 covariance 已校準。原 rank 門檻維持不變，未觀測 null direction 不會被補出資訊；height output 仍按舊 MAP height 可信度規則。

純高斯生成例子中，`C=R=1`、map measurement 為 1 m 時，避免重複 prior 可將理想後驗位置由 0.25 m 改為 0.5 m，代表減少朝 sensor prior 的重複拉回。這只是閉式生成案例用來說明重複融合，不是任何真實 frame 的量測誤差或 v5.12 真實改善幅度。

### 2. 固定平面成本下降，仍可能走向錯誤最近表面

v5.11 已將有限 gain 後實際輸出的 pose／scale 放回固定最終平面配對上重新算資料成本。問題是，這個 gate 仍沿用同一組舊對應；如果候選離開原表面並切換到另一片地圖平面，固定舊配對成本下降不保證最近表面一致性變好。

一個兩平面生成反例：地圖表面 A 在 0 m、B 在 0.5 m；來源樣本目前在 A，prior pose 為 0 m，提案移到 0.1 m。若舊 gate 固定錯配在 B，殘差平方會由 `0.25` 降至 `0.16` 而通過；重新找最近平面時，成本卻由 A 上的 `0` 升至 `0.01`。v5.12 在實際有限 gain posterior pose／scale 上同時計算固定 associations 與重新配對後的成本；兩者都須不增加，且維持原 pose／scale bounds，否則拒絕 measurement commit。對應診斷圖和數值在 [generated model checks](../outputs/run/track00_test_2611032122/analysis/03_generated_model_checks.png)。

兩道 gate 都只是在現有樣本、權重、unmatched penalty 與原 search bounds 內做檢查，不能證明整幅地圖或真實車體姿態正確。實際輸出 scale 仍遵守原 height MAP 可信度規則，以避免只改 pose 量測模型而破壞舊 pose／height 一致性。

## 實作與驗證狀態

候選版本 `track2_current_map_likelihood_v5_12_perf2` 的三個開關為 `V512_ENABLED=True`、`MAP_DATA_LIKELIHOOD_ENABLED=True`、`MAP_REMATCHED_POSTERIOR_CHECK=True`。細部機制位於凍結候選中的 `track2/metric_registration.py` 與 `track2/posterior_validation.py`，不是逐幀參數表。

9 個聚焦 CPU 機制案例通過（1.85 秒）；另有安全 CPU subset 40 passed、2 deselected（1.09 秒，與前者不能相加作 distinct total）。3299 列 generated full-analysis fixture 的唯讀分析、圖 10 支援及 retry 通過（14.7863 秒）；這是純生成 fixture，不是真實定位資料。第一輪 GPU suite 為 224 passed／45.24 秒；加入非收斂恢復搜尋與 5 個生成搜尋案例後，第二輪為 229 passed／45.65 秒。補齊 scope 記錄及非有限成本 JSON 防護後的最終 GPU suite 為 **230 passed**、1 項既有 Matplotlib `Axes3D` warning、46.37 秒；121 個 Python SHA 凍結於上述 manifest。三組各 3 幀生成 ablation 耗時 5.484477 秒：關閉 v5.12 時，三幀輸出 pose、scale、height、height variance、correction covariance、接受狀態與凍結 v5.11.1 完全一致；只開 data-only likelihood、只開 rematch cost、兩者同開的三組各 3 幀均被接受，並保留實際 posterior cost。完整摘要見 [generated_ablation.json](../outputs/run/track00_test_2611032122/analysis/generated_ablation.json)。這些是合成機制／回歸檢查，不是實際 Track 或精度證明。程式已套用；沒有 v5.12 真實定位或新分數。

新增的 `TRACKING_RETRY_AFTER_NONCONVERGENCE=True` 只在 current 與 carry 都 geometry fail、至少一個候選仍有 rank>0 且具足夠 matches／angular support、最後原因是 `iteration_limit` 或 `line_search_failed` 時，才沿 carry 的 yaw prior 追加兩個 bounded seeds（±1σ；沿用 `MAP_STARTUP_YAW_SEED_SIGMAS` 的非零項）。它不放寬原 quality gate 或 bounds；rank 0、已收斂或已合格的 frame 不擴搜。候選 solve 數記在 `nonconvergence_recovery_solves`。這能針對「有觀測卻未收斂」的情形再試起點，但沒有 v5.12 真實 full 證據顯示指定路段的 rescue 數增加，也不處理已接受但錯配的 geometry。

完整定位由使用者手動啟動：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
./scripts/run_track2.sh --full
```

### 測試執行紀錄與修正

- `pytest_cpu_0002` 標成 CPU，但 42 passed（2.88 秒）中包含 2 個 CUDA 案例，當時也曾短暫與使用者 full 共用 GPU。這是執行分類疏漏，已記錄並告知使用者；後續 CPU 測試使用 `CUDA_VISIBLE_DEVICES=-1`，CUDA suite 等 GPU lock。
- `pytest_cpu_0003` 遇到 5 個 collection errors（0.96 秒）：pytest plugin autoload 掃描 ROS `launch_testing` 測試，且當時 CUDA 被隱藏。改用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 跑限定的 CPU subset 後 40 passed。這兩次失敗／修正都保留，沒有當成 v5.12 GPU 驗證。
- 聚焦 CPU 機制案例 9 passed、subset 40 passed／2 deselected、generated fixture 通過是不同測試範圍；勿把數字相加成單次完整 suite。

## 證據與歷史基準

- 分析來源是 v5.10 full [`track02_full_2611031941`](../outputs/run/track02_full_2611031941/test_info.json)，完成 3299 幀；Public Score `0.42699` 是使用者回報，作為版本基準，不代表逐幀 ground truth。
- v5.11.1 機制執行證據、v5.12 候選、標記視窗統計及圖表集中在 [`track00_test_2611032122`](../outputs/run/track00_test_2611032122/analysis/README.md)。原始 CSV、pose trace、observations、manifest 皆唯讀；frame 座標沒有離線回寫。
- v5.10 run 的詳細欄位來源與每個標記區間統計在 [marked_windows_stats.md](../outputs/run/track00_test_2611032122/marked_windows_stats.md)；本文件的 rounded table 由該表及 `modes_diagnosis.json` 摘要，精確值以 JSON／Markdown 原始報告為準。
- 使用者報告的歷史最佳仍為 v5.10 `0.42699`。v5.12 尚無真實 full 成績；合成反例、coverage、residual、rank、接受率和測試通過都不能代替精度驗證。

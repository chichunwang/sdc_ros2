# v7｜首幀多候選配準

目前程式版本為 `track3_cuda_v7_multihypothesis_startup`。這次只改首個 map anchor 的候選評估：把 grounded initializer 產生的每個 scale／yaw 候選都送入完整 NDT→ICP，再以同一組非地面點與固定距離目標評分。只有通過既有 registration 品質閘門、0.6 非地面 coverage 及 25 m 地面平移界限的候選能勝出；全部不合格時，本幀沿用 prior，後續尚未建立 anchor 的影格仍會嘗試初始化。

官方 `init_guess.txt` 作為粗略搜尋中心、prior yaw seed 及地面高度比例 seed；初始 pose 本身加入雜訊，不是真值。最終首個 anchor 不受初始 pose 的 Kalman NIS veto，直接由合格幾何候選建立。保留初值能縮小地圖搜尋與道路重複結構的歧義；這個有限候選方法仍可能找不到正確解，完全移除初值則需要另一套全域重定位機制。本次不根據外部軌跡或分數決定初始位置。

## 依據與範圍

v4、v5 full 與既有 v6 的 1257 幀短段都從相同初始姿態開始。`init_guess` yaw 為約 145.68°，首筆幾何 map anchor yaw 為 169.207°。v5／v6 的 idx 10 map registration yaw 候選為 159.844°，ESKF prediction/final yaw 為 167.037°；該筆更新 NIS 為 29.651，高於 22.458 門檻，故 registration 未採用。idx 0 到 10 的相機 header 時間差為 0.360 s；同期間儲存的逐幀 raw gyro sample maxima × interval 合計約 1.63°，這是 sample-max envelope proxy，並非重新積分或真值。idx 10 PnP rotation-rate estimate 為 0.036 rad/s，也只是模型估計。這些證據支持檢查初始幾何候選與後續 gate，**不能證明 159.844° 或 169.207° 哪個更接近真值，也不能由 proxy 排除 IMU sampling、外參或視覺估計的影響**。完整唯讀逐幀資料與歷史 pose trace SHA 記在[起始證據](../outputs/run/track00_test_2610251509/startup_evidence.json)；時間與 PnP／gyro proxy 來源及限制記在[感測器證據](../outputs/run/track00_test_2610251509/sensor_evidence.json)。v6 短段 metadata 在 [track03_test_2610251424](../outputs/run/track03_test_2610251424/test_info.json)。

收尾 SHA 稽核發現既有 v6 run 的 `result_3.csv` 與該 run metadata 的原 SHA 不符，CSV 修改時間為本輪開始前的 14:47:56；本次保留其現況，不回寫 CSV 或舊 metadata。診斷實際使用的 `pose_trace.jsonl`、`observations.jsonl` 及其餘五份已記錄輸出均符合原 SHA。詳見本次新 run 的 [historical output check](../outputs/run/track00_test_2610251515/historical_output_check.json)；該 CSV 不作本次診斷或 v7 選候選的輸入。

ICP 是 local registration，表現依賴輸入初始對齊；這次利用現有幾何候選補足局部最佳化的起點探索。此方法不是 global-optimal registration，也沒有加入新資料來源或未來影格。[Open3D global registration 說明](https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html)也區分需要 rough alignment 的 local registration 與 global registration。

## 候選如何比較

initializer 原有的 grounded scale／yaw seeds、map/source ground planes、非地面 feature 點與角度權重都保留。v7 對每個候選完整執行既有 8／4／2 m NDT 與 1 m ICP；所有候選共用以原始 prior 為中心的同一 submap，避免候選順序改變可用地圖。

精修後使用同一份非地面 source feature、target map points 與角度權重 $w_i$ 比分，沒有將各候選 solver 的 scale prior 帶入跨候選排名：

\[
C(T,s)=\frac{\sum_i w_i\,\rho(\min(d_i^2,r^2))}{\sum_i w_i},\qquad
\rho(u)=0.3^2\log\!\left(1+\frac{u}{0.3^2}\right),\qquad r=0.75\,\mathrm{m}.
\]

距離 $d_i$ 是將同一批 raw source feature 以候選 scale $s$ 和 pose $T$ 變換後，到相同非地面 map 支持點的最近距離。超出既有最後 initializer stage 的 0.75 m 範圍仍以截斷距離計分，並留在分母中；不匹配的 feature 不會因刪除而讓候選分數虛降。0.3 m robust scale 與 0.75 m 截斷半徑沿用原有最後 initializer stage，沒有為 v7 掃參數。

通過既有 final registration gate 的候選，還須有至少 0.6 的加權非地面 feature coverage，且 grounded 平移不得超過既有 25 m 界限；在合格者中取共同幾何 cost 最低者。若所有候選都失敗，runner 拒絕首幀 registration，保留 initial pose／scale prior，不以低 cost 取代品質閘門。v7 沒有利用外部 CSV 擬合候選或調整參數。

## 輸出診斷與限制

每次新 run 的 `observations.jsonl` 在 idx 0 `initialization.refinements` 保存每個候選的精修 pose、scale、各 stage 診斷、共同 feature score、coverage、translation bound、是否合格及耗時；`initialization.selection` 記錄排名方法、選中索引及候選數。全數不合格時，可從該處確認每個失敗原因。v7 bootstrap 仍只建立首個 map pose anchor，不把 initial-to-map 的 pose 差當成車速或 IMU 量測。

IMU ESKF、yaw robust weighting、`IMU_RPY_DEG = (1.0, 6.0, 0.0)`、`IMU_XYZ_M = (0,0,0)` 與既有 noise／fusion 設定均未改；說明與參數入口見 [v6 因果 yaw 文件](58_CAUSAL_YAW_AND_FUSION_TUNING_20261025.md)。

最終 immutable snapshot 的生成資料與 CUDA regression suite 為 **141 passed，227.42 秒**，另有 1 個既有 Matplotlib Axes3D warning。[pytest.xml](../outputs/run/track00_test_2610251515/pytest.xml)、[test_info.json](../outputs/run/track00_test_2610251515/test_info.json)與[圖表 README](../outputs/run/track00_test_2610251515/analysis/README.md)保留在該 run。核對 60 個程式／設定／測試檔與受測快照 SHA 相同，見 [live source check](../outputs/run/track00_test_2610251515/live_source_check.json)。

新增的五個 case 包含：共同 raw-feature score 對錯誤 scale／heading 與缺失對應的處理；原先較低 proxy cost 的候選配準失敗時，另一個未收斂 grounded seed 仍能精修並勝出；交換候選順序仍得到相同 pose／scale；完整生成場景經預設四候選選擇；失敗低 cost 候選不能越過品質 gate；全敗保留 prior／scale。順序檢查包含於 alternate-seed case；品質 gate 以兩個 parameterized cases 檢查。

生成場景的已知 scale 為 7，選中 scale 為 7.013634，位置對合成已知真值的差為 0.02309 m。該 case 使用單一 ICP stage 隔離候選機制；一般 CUDA／NDT 與 runner 輸出契約由其餘 suite 覆蓋。注入分支的 case 中，24 m 偏移候選被拒絕，alternate 精修後位置差為 0.01334 m；24 m 是刻意生成的候選位置差，**不是舊 runner 的輸出誤差或本次 Track3 的改善幅度**。

早期 `track00_test_2610251509` 的 12 項／36 秒基礎檢查，以及 `track00_test_2610251511` 的 4 項／142.64 秒機制檢查，保留各自來源快照；最終以 `track00_test_2610251515` 為準。唯讀圖表包含 [v6 首段診斷](../outputs/run/track00_test_2610251515/analysis/01_saved_startup_diagnosis.png)與 [v7 合成候選機制](../outputs/run/track00_test_2610251515/analysis/02_synthetic_startup_selection.png)，各有 PDF、數值及來源 SHA manifest。

**目前沒有 v7 真實短段、full 或 Kaggle 成績證據。**本次真實定位處理幀數為 0；合成幾何能檢查候選分流與 gate 行為，不能證明實際首段偏移已消除。不因缺少真實精度證據另設 full 資格關卡，完成這些機制／數值／輸出檢查後即可提供下方手動 full 命令。

如需由使用者手動啟動完整 Track：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
./scripts/run.sh --full
```

定位與完成後的分析圖表會保存於新建的 `outputs/run/track03_full_YYMMDDHHmm/`；此命令由使用者執行。

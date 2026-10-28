# Track2：讓地圖修正有記憶的因果融合 v4

當幀配準失敗，只能說這一幀缺少可靠地圖更新，不能因此立刻抹掉過去已累積的 map correction。反過來說，配準雖通過幾何門檻，候選若突然偏離既有狀態，也不能直接整筆採用。v4 在每幀輸出前保留一個地圖座標的修正狀態，以協方差、運動界線和配準品質控制更新。

現行版本為 `track2_causal_metric_correction_memory_v4`。實作主要在 [`correction_filter.py`](../track2/correction_filter.py)、[`metric_registration.py`](../track2/metric_registration.py) 和 [`parameters.py`](../track2/parameters.py)。上一版地面尺度模型與疏採樣對照見 [66](66_TRACK2_METRIC_SCALE_20261029.md)；本文件只記錄新增的逐幀修正記憶。

## 看到的問題

對已保存的 v3 輸出作唯讀診斷時，frame 427／428／429 的 map update 依序接受、失敗、接受；失敗幀清掉約 0.8 m 的前一幀平移修正，使輸出短暫偏移，下一幀又把修正加回。frame 652／653／654 則顯示，在沒有新 GNSS 訊息時，當幀幾何候選也能突然移動約 1 m，再於下一幀往回變動。這表示單看 GPS 不能解釋所有跳動：配準品質合格也不保證候選與過去狀態一致。

上述座標差異只描述估計器輸出，沒有獨立 ground truth，不能據此判定哪一幀的 pose 正確。v3 full 已停止並留下 1689 幀輸出；其 `test_info.json` 仍標記 `running`，保留原樣。前 888 幀資料與唯讀診斷副本的 SHA 相符，因此可用已保存候選研究濾波行為，無須重新執行定位。歷史 run 和診斷連結收在本文末的附錄。

## 狀態代表什麼

濾波狀態是

`c = [dx_map, dy_map, dyaw_map]`

其中 XY 是地圖座標中的公尺，yaw 是地圖座標中的弧度；另保存對應的 3×3 協方差 `P`。這個狀態代表 map feature 對當幀 sensor prior 的緩變修正，**不是車輛自己的位置、速度，也不是 ground plane 座標**。車輛仍跟著新的 GNSS／IMU prior 移動，保留修正不會把車停在上一幀。

每幀先依真實時間差 `dt`、prior 的旋轉變化和因果速度預測 `P`。時間、行駛距離與轉向都會增加修正狀態的不確定度。GNSS 更新可能令 prior 移動；為免 GPS 跳點被錯當成車輛真的走得很遠，運動 gate 的行駛距離以濾波器當下的速度乘 `dt` 計算。整個更新依序是：建立當幀 sensor／ground prior 和 map candidate、預測 correction 與 `P`、比較 candidate 的 innovation 及幾何品質、融合當幀結果，然後只輸出一次。

它只使用當下及過去資訊。沒有 RTS／backward smoothing、沒有未來影格修正舊狀態，也沒有定位後改寫軌跡的介面。

## 成功配準時怎麼更新

1. 先讓現行地面／非地面 solver 產生未融合候選。候選修正轉換到世界 map XY／yaw，避免前後影格的地面座標基底改變後，仍把舊基底的平移量直接搬過來。新的輸出再放回當幀 ground-plane 模型，保留該幀的地面法向和高度關係。
2. **運動差異和統計差異各自參照不同基準。**運動差異比較本幀候選與上次被吸收的原始 map 候選，記作 `motion_innovation`；以 XY 與 yaw 各自的 allowance 正規化，再取最大值：`motion_ratio = max(norm(XY motion_innovation) / XY allowance, |yaw motion_innovation| / yaw allowance)`。這是觀測到觀測的變化，避免長時間失聯後濾波均值尚未追上候選，導致漸進重獲配準時反覆拒絕自己的更新。`motion_ratio > CORRECTION_MOTION_REJECT_MULTIPLIER`（初值 3.0）才因運動差異硬拒絕。
3. NIS 則把本幀候選與**濾波預測狀態**比較。NIS 原值大於 `CORRECTION_NIS_REJECT`（16.266）仍硬拒絕。其他候選使用 `inflation = max(1, NIS / CORRECTION_NIS_SOFT, motion_ratio²)` 放大量測協方差；NIS 偏高或運動差異中度超界，都會降低更新影響。
4. 量測協方差由**不含 GNSS prior 的資料 Hessian／Gauss–Newton 資訊矩陣**投影至 world correction，計算時保留 XY、yaw 與 log scale 的耦合。弱資訊方向會得到較大不確定度；再加上位置及角度誤差下限，並依 coverage、median residual 擴大協方差。solver objective 裡的 GNSS soft prior 不會被重算成另一份獨立 map 量測。

首筆有效候選尚沒有舊的原始 map 候選可作 motion comparison，因此跳過運動比值 gate，由初始 `P` 和 NIS 做正常 Kalman update。之後每筆候選同時和上次吸收的 raw candidate、濾波預測比較；兩者參照不同、用途也不同。

候選被接受後，濾波器再限制本幀真正寫入狀態的 correction 增量，避免長時間失聯令 `P` 變大後，重獲配準雖通過較寬的 innovation gate，首幀卻一步跳遠。XY 增量上限是 `POSITION_RATE × dt + SPEED_FRACTION × speed × dt + TURN_LEVER × prior_rotation`；yaw 增量上限是 `YAW_RATE × dt + TURN_FRACTION × prior_rotation`。若 `K × innovation` 超過任一上限，將同一 Kalman gain matrix `K` 整體乘上 `gain_scale ≤ 1`，再以縮放後的 K 同時更新 correction 均值和 Joseph covariance。這避免只限制均值、卻留下過度樂觀的 covariance。這些增量界線只限制 map correction 狀態，vehicle pose 仍按當幀 sensor／ground prior 和融合結果前進。

## 配準失敗時怎麼退讓

當幀沒有有效 map candidate，修正均值先保持原值；預測不確定度仍依時間與運動增加。最後一次有效更新後的前 0.5 秒保留修正；之後以 3 秒 time constant 的指數形式逐步將修正均值退向零，並將協方差逐步放回 sensor prior 的先驗尺度。下次收到有效候選時，持續增加的不確定度會降低保守 gate 造成的鎖死風險。

因而輸出行為是：

| 狀態 | 當幀實際輸出 | 修正記憶 |
| --- | --- | --- |
| 幾何有效且 innovation 正常 | 當幀地面 pose 加上更新後 correction | 吸收本幀候選 |
| 幾何有效、innovation 大但可接受 | 當幀 pose 加上降權後 correction | 只部分吸收候選 |
| 幾何有效、突跳超過 gate | 當幀 sensor／ground prior 加原有 correction | 拒絕該 candidate |
| 地面／配準短暫失敗 | 新 sensor prior 加原有 correction | 保持均值，擴大不確定度 |
| 連續失敗較久 | 新 sensor prior 加逐漸變小的 correction | 隨時間退回零，協方差放大 |

每幀的 `geometry_accepted` 指非地面 solver 候選通過自己的品質檢查；`accepted`／`map_accepted` 指 candidate 經時間濾波後真的被吸收。`pose_trace.jsonl` 的 `registered` 存原始 map candidate，`final` 才是該幀實際輸出。`observations.jsonl` 的 `registration.correction_filter` 記錄 `motion_reference`（上次吸收的原始候選）、`motion_innovation`、相對預測狀態的 NIS innovation、`motion_innovation_ratio`、`gain_scale`、本幀位置／yaw 更新界線，以及實際 3×3 `kalman_gain`。`measurement_weight` 只是量測協方差膨脹係數的倒數；完整實際增益看 `kalman_gain`。這些診斷只描述濾波器行為，不是 pose 真值。

高度 anchor 只有 temporal measurement 被接受，而且尺度資訊足夠時才更新。raw scale 與 ground tilt 仍從當幀幾何估計／支撐；它們沒有套用 XY／yaw correction 狀態的低通。如此避免把每幀可能真實變動的 raw 重建尺度與地面法向也硬拖向舊值。

## 參數在哪裡

直接編輯 [`parameters.py`](../track2/parameters.py)。以下數值是首版工程模型，不是用獨立真值校準的統計量；先一次改一個假設。

| 參數 | 初值 | 意義 |
| --- | ---: | --- |
| `CORRECTION_HOLD_SECONDS` | `0.5 s` | map 更新中斷時維持 correction 均值的寬限時間 |
| `CORRECTION_DECAY_SECONDS` | `3.0 s` | 超過寬限後 correction 朝零衰減的時間常數 |
| `CORRECTION_MEAS_POSITION_FLOOR_M` | `0.15 m` | XY map 量測協方差下限 |
| `CORRECTION_MEAS_YAW_FLOOR_DEG` | `0.5°` | yaw map 量測協方差下限 |
| `CORRECTION_POSITION_GATE_BASE_M` / `CORRECTION_POSITION_RATE_MPS` | `0.35 m` / `0.5 m/s` | XY 觀測運動界線；position rate 也參與每幀更新上限 |
| `CORRECTION_SPEED_FRACTION` / `CORRECTION_TURN_LEVER_M` | `0.10` / `1.5 m` | 依行駛距離／轉向放寬的位置差界線；turn lever 未標定 |
| `CORRECTION_YAW_GATE_BASE_DEG` / `CORRECTION_YAW_RATE_DEGPS` | `0.75°` / `0.5°/s` | yaw 觀測運動界線；yaw rate 也參與每幀更新上限 |
| `CORRECTION_TURN_FRACTION` | `0.10` | 依 prior 轉向放寬 yaw 差界線 |
| `CORRECTION_MOTION_REJECT_MULTIPLIER` | `3.0` | `motion_ratio` 大於此倍數時因運動差異硬拒絕；介於 1 與此值則依 ratio² 降權 |
| `CORRECTION_NIS_SOFT` / `CORRECTION_NIS_REJECT` | `7.815` / `16.266` | 3 維 NIS 軟降權和硬拒絕界線 |
| `CORRECTION_POSITION_RW_M_SQRT_S` / `_SQRT_M` | `0.10` / `0.03` | correction 隨時間／距離成長的不確定度 |
| `CORRECTION_YAW_RW_DEG_SQRT_S` / `_SQRT_M` | `0.20` / `0.05` | yaw correction 隨時間／距離成長的不確定度 |

先前調整 GNSS prior 的位置可直接看 `METRIC_POSITION_STD_M` 和 `METRIC_YAW_STD_DEG`；這會改 solver 的 sensor prior 權重，不等同於改 map correction 記憶。v4 用兩層檢查，避免只靠調大 GNSS 權重來壓住地圖 candidate 的偶發跳動。

## 驗證狀態和限制

現行 correction-memory 機制通過 29 項合成／輸出檢查，耗時 10.30 秒；唯一的 Matplotlib `Axes3D` 警告不影響本輪 2D 圖表。檢查涵蓋短暫失敗、長時間衰減、motion／NIS 軟硬 gate、失聯後逐步重獲與不延後突跳、Joseph covariance、弱資訊方向、world／plane Jacobian、尺度／傾斜、因果 prefix，以及 runner、ROS 和 full 圖表保護。pytest 證據與合成圖見 [`track00_test_2610290341`](../outputs/run/track00_test_2610290341/analysis/README.md)。

另將現行 filter 對照 888 幀已凍結的舊候選：runtime 與 replay source snapshot 的 7 個核心程式 SHA 一致。frame 428 修正突變由 0.82749 m 降為 0，估計 XY 單幀變化由 0.96942 m 降為 0.19013 m；frame 653 修正突變由 1.14247 m 降為 0.002582 m，估計 XY 單幀變化由 1.21291 m 降為 0.317887 m。frame 653 已進入長失敗衰減，所以這不是完全 hold。frame 654 後續 correction 更新量為 0.03278 m，沒有把突跳延後到下一幀。重播決策共 83 次一般更新、611 次降權、158 次 NIS 拒絕、24 次 motion 拒絕、12 次幾何拒絕。對照圖：[frame 428](../outputs/run/track00_test_2610290340_0002/analysis/01_frame_0428_correction_replay.png)、[frame 653](../outputs/run/track00_test_2610290340_0002/analysis/01_frame_0653_correction_replay.png)、[生成資料 dropout](../outputs/run/track00_test_2610290340_0002/analysis/02_generated_dropout.png)。

這些證據檢查 filter 的預期行為，不是重新配準、完整 v4 軌跡或精度證明。舊資料沒有完整 Hessian，重播量測 covariance 以 floor 和品質膨脹作 proxy；正式 runtime 使用當幀 data Hessian。舊候選也不反映 v4 新 seed 與高度回饋，沒有真值，因此重播不產生或修改提交用 pose CSV。沒有 v4 full 實際定位精度結果。

特別注意：NIS 門檻、資料 Hessian、coverage／residual 膨脹、初始 prior variance、motion envelope 和逐幀 correction update limit 都受幾何相關性、地面品質及未知感測器誤差影響，尚未標定。錯誤配準若連續且自洽，時間濾波可能逐步吸收它；若 GNSS／IMU prior 自己突跳，vehicle pose 也可能跟著 prior 改變。當幀 ground tilt 和 raw scale 有各自的失敗模式。修正變平順只表示 correction 更新較連續，不代表已知更接近真值。

成功完成 full 後，若每幀 observation 都有 v4 的 correction diagnostics，會多產生 `analysis/05_map_correction_memory.png/pdf`，呈現 XY／yaw 修正、模型不確定度、NIS 和有效 map 更新失聯時間。舊 run 若沒有逐幀 `correction_filter` 欄位就不會有此圖，也不補做歷史回填。

由使用者自行手動啟動 full：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
./scripts/run_track2.sh --full
```

每輪輸出會包含凍結 source、parameters、SHA、候選與實際 pose trace、結果 CSV，以及成功 full 後的唯讀圖表。不要在程式執行期間編輯其 frozen source；Kaggle／E3 提交仍由使用者操作。

<details>
<summary>歷史 run 與中間版檢查</summary>

- v3 full `track02_full_2610290242` 已停止，保留 1689 幀；metadata `status` 仍是 `running`，不回寫此歷史檔。
- 前 888 幀舊 candidate 的唯讀 replay 與來源摘要見 [`track00_test_2610290257`](../outputs/run/track00_test_2610290257/analysis/README.md)。它只重播修正狀態，未重跑點雲定位，也不是精度驗證。
- v4 初版 `track00_test_2610290330` 有 25 pass、2 fail；相容性修正後 `track00_test_2610290333` 有 27 pass、10.28 秒。後續 `track00_test_2610290336` 的 28 pass 是中間版，不能作為最終結果。
- `track00_test_2610290340` 有 28 pass、1 fail，原因是測試 fixture 只更新濾波 mean，未同步上次吸收的 raw candidate；修正後的現行版本結果見本節正文 29 項檢查。

</details>

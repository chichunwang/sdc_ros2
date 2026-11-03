# Track2：啟動候選與首筆 map 交接修正 v5.11.1 + perf2

版本已套用，為 `track2_startup_candidates_current_validation_v5_11_1_perf2`，基準是 v5.10 full `track02_full_2611031941`：3299 幀完成，使用者回報 Public Score `0.42699`。它保留 v5.11 的 v5.10 course／heading 檢查、startup 候選完整配準、幾何失敗時 alternate retry，以及固定 final associations 下有限 gain posterior pose+scale data-cost 驗收；另修正首次 map update 在 prior-only 輸出之後到達時略過既有限幅。使用者標記的 frame 只作視覺檢查，不當作真值，也不加入逐幀座標修正。算法原因、驗證範圍與限制見[87｜v5.11.1 啟動交接修正](../doc/87_TRACK2_V511_STARTUP_VALIDATION_20261103.md)。

**狀態：**v5.11.1 GPU Track2 suite **214 passed**、1 項既有 Matplotlib `Axes3D` warning、44.80 秒；118 個凍結 Python SHA 一致（[JUnit XML](../outputs/run/track00_test_2611032103/pytest_gpu_0002.xml)、[log](../outputs/run/track00_test_2611032103/pytest_gpu_0002.log)）。首輪為 213 passed、1 failed；唯一失敗是新測試 assertion 將 2D XY 和 3D correction 相加，已用 `[:2]` 修正測試並完整重跑，原 gain gate 沒有調整（[首輪 XML](../outputs/run/track00_test_2611032103/pytest_gpu.xml)、[log](../outputs/run/track00_test_2611032103/pytest_gpu.log)）。新增的 7 個生成 handoff cases 全數通過。關閉修正的 31 幀與 frame 0 即可接受 map 的 31 幀都與 v5.11 逐項相同；延遲首筆 map 的合成 correction 從 1.344392 m 降至 0.020000 m，恰與該場景 40 ms／0 speed／0 gyro 的既有預算相符。這是生成機制檢查，不能宣稱真實定位精度或 Y 字已在真實 full 消失。handoff 圖與數值見[圖表](../outputs/run/track00_test_2611032103/analysis/01_generated_handoff.png)／[PDF](../outputs/run/track00_test_2611032103/analysis/01_generated_handoff.pdf)及[JSON](../outputs/run/track00_test_2611032103/analysis/generated_handoff_checks.json)。

v5.11.1 已套用，程式與本文件的來源 SHA 已核對。尚未有 v5.11.1 真實定位、PnP 幀或成績。前一版 v5.11 full `track02_full_2611032051` 由使用者停止於 351 幀（idx 0–350），`score=null`；run metadata 的 `running` 狀態與 185 個 source SHA 均原樣保留（[run metadata](../outputs/run/track02_full_2611032051/test_info.json)）。目前最低的已知完整版本成績仍是 v5.10 full 的使用者回報 `0.42699`。

**執行命令：**完整定位仍由使用者手動啟動：

```bash
./scripts/run_track2.sh --full
```

v5.11.1 的生效參數位於 [`parameters.py`](parameters.py)。首次接受 map update 前的 startup 多候選、tracking alternate retry、finite-gain data-cost 驗收，以及 prior-only 啟動後首筆 map update 的 gain 限制見[87](../doc/87_TRACK2_V511_STARTUP_VALIDATION_20261103.md)；v5.10 的 course／heading 預算仍見[86](../doc/86_TRACK2_V510_V57_HEADING_20261103.md)。

下方 v5.3–v5.5 參數和機制保留作歷史說明。v5.11.1 從 v5.10 延續 v5.7 的 EPNP 影像速度、固定 40 m feature range、MAP height 與 visual update；不帶入 v5.8／v5.9 的 IPPE／逐影格 LK、多 range、data-only height 或 gyro 約束路面平移。

## 現況

perf2 於 2026-11-02 完成套用，僅改讀檔與顯示訊息的資料處理方式；全部 3299 個實際輸入的 XYZ dtype、點序與 bytes 一致，141 項 regression 及另 1 項生成 runner 等價檢查通過。六幀元件量測平均 `465.1→14.9 ms`，不包含配準、實際 ROS 發布或完整 runner，不能外推成 full 加速 31 倍。來源與保護核對見[套用收據](../outputs/run/track00_test_2610311820/apply_receipt.json)。後續 v5.4 full 使用 perf2；不同算法版本的 full 分數不能單獨歸因於加速。

使用者回報 fix1 full [`track02_full_2610311843`](../outputs/run/track02_full_2610311843/test_info.json) 的 Public Score 為 `0.46899`：3299 幀，定位耗時 3038.67 秒，未使用 perf2。fix1 保留 v5.2 雙航向 bias、v5.1 公尺尺度配準與 perf1，修正 GNSS XY rebase、方向性 correction-memory age 與 map-only likelihood 驗收。輸出仍因果，不回頭修改舊 pose。

下列 full 成績是使用者回報的 Track2 Kaggle Public Score；使用者目標值另列供版本目標追蹤，不代表已取得該分數。Public Score 不能當逐幀 ground truth，也不能保證單一改動的效果。

| 來源 | Public Score／目標值 | 狀態 |
| --- | ---: | --- |
| v5 full | 0.53683 | 3299 幀 full；map update 為 0 |
| v5.1 full `track02_full_2610291400` | 0.51617 | full 已完成；改動同時含地圖配準 |
| v5.2 full `track02_full_2610311445` | 0.46434 | 使用者回報；比 v5.1 低約 10.04% |
| v5.3 + perf1 full `track02_full_2610311612` | 0.80300 | 使用者回報；3299 幀已完成，來源 SHA 與執行時 live source 相同 |
| v5.3 fix1 + perf1 full `track02_full_2610311843` | 0.46899 | 使用者回報；3299 幀，該次未使用 perf2 |
| v5.4 + perf2 full `track02_full_2611021544` | 0.59002 | 使用者回報；3299 幀已完成，來源核對保存在本輪診斷 |
| v5.5 + perf2 full `track02_full_2611021746` | 0.52708 | 使用者本輪回報；3299 幀已完成 |
| v5.6 + perf2 `track02_full_2611022032` | 0.45455 | 使用者回報，3299 幀，1570.36 秒完成；六個 live/source SHA 一致 |
| v5.7 + perf2 `track02_full_2611031204_0002` | 0.44296 | 使用者回報，3299 幀完成；僅 4 次移動影像速度量測通過 |
| v5.8 + perf2 | 0.45341 | 使用者回報；3299 幀 full，定位1650.03秒 |
| v5.9 + perf2 `track02_full_2611031629` | 0.44917 | 使用者回報；3299 幀 full 已完成 |
| v5.10 + perf2 [`track02_full_2611031941`](../outputs/run/track02_full_2611031941/test_info.json) | 0.42699 | 使用者回報；3299 幀 full 已完成；v5.11 基準 |
| v5.11 + perf2 | — | 首版 full 由使用者停止於 351 幀；score null；不算完整成績 |
| v5.11.1 + perf2 | — | GPU suite 214 passed；生成 handoff 機制檢查通過；無真實定位或新成績 |
| 使用者目標值 | 0.30488 | 目標 CSV `doc/analysis doc/result_2_0.30488.csv` 僅作 hash／provenance 紀錄；核對見 [target reference](../outputs/run/track00_test_2611021349/target_reference.json) |

所有 full 分數都由使用者回報，來自不同版本，不應歸因於 perf2。使用者目標值為 `0.30488`；參照 CSV 用於同索引差異診斷，不作逐幀真值或回修 pose 的依據。no-accel 前五幀真實 A/B 與五個生成場景輸出一致，但不能推論整段相同。

參照 CSV 是非因果輸出，只作同索引差異診斷；不從榜分反推隱藏真值、不擬合軌跡轉換或時間差，也不保證任何版本能達到目標 `0.30488`。五個 full 共用相同 input SHA 和 frame-manifest SHA，詳細結果／輸出 hash／分數來源見[版本證據](../outputs/run/track00_test_2611021839/version_evidence.json)。

v5.6 的 map／heading 基線與方向記憶見 [82](../doc/82_TRACK2_V56_V52_BASELINE_20261102.md)；v5.7 在此基礎上新增前視影像地面速度，見 [83](../doc/83_TRACK2_V57_IMAGE_GROUND_SPEED_20261102.md)。v5.8、v5.9 為後續實驗版本；v5.10 恢復到 v5.7 影像速度、尺度配準與 visual update，並加入 course／heading 更新檢查。v5.11 以 v5.10 完整版為基準增加 startup 候選與 map proposal data-cost 驗收；v5.11.1 延續這些機制並修正 prior-only 啟動後首筆 map 更新的有限 gain handoff，細節見 [87](../doc/87_TRACK2_V511_STARTUP_VALIDATION_20261103.md)。下方 fix1／fix2 章節只記錄模型沿革。

### fix1 的三項模型修正

1. GNSS XY CV filter innovation 繼續推進當幀 sensor prior 與 diagnostics，但不再作 calibration-reference delta 去抵消舊 map memory；校正重定錨只保留 heading calibration 及其 lever-arm 影響。
2. correction-memory 用 prior-Cholesky 白化座標下的 PSD 方向年齡矩陣管理 partial-rank map 記憶；只有實際觀測 row-space 刷新相應模式，其他弱方向超過原 hold 後按原 tau 衰退。
3. map-only likelihood center 除原 bounds 外，須在同一組 final planes 的實際非線性 data loss 不增加才接受；尺度 nuisance 仍做 Schur marginalization，條件 log-scale step 保留，拒絕不靠 blend prior 修補。

## 啟動

完整定位由使用者手動啟動。固定命令如下：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
./scripts/run_track2.sh --full
```

v5.11.1 不新增 full 專用旗標；完整定位仍由使用者手動啟動。驗證範圍與限制見 [87](../doc/87_TRACK2_V511_STARTUP_VALIDATION_20261103.md)。

入口會啟動 Track2 RViz（`/track2/*`，TF `map → track2_camera`），設定在 [`config/track2.rviz`](../config/track2.rviz)。Track1／2／3 共用 GPU lock；另一 Track 正執行時，runner 會拒絕並行配準。Full 輸出在新的 `outputs/run/track02_full_YYMMDDHHmm/`。

## v5.3 fix2：避免高度先驗被重複融合

舊流程以高度先驗求 registration MAP candidate，又把 candidate height 當獨立量測更新高度 Kalman filter。fix2 從同一組 point-to-plane data Hessian／gradient 建立 map-only 高度 likelihood，避免把已用於 solver 的 prior 再當成新資料融合。

令平面 pose nuisance `u=[x,y,yaw]`、log-scale 為 `s`。消去可觀測平面模式後：

\[
I=H_{ss}-H_{su}H_{uu}^{\dagger}H_{us},\qquad
\gamma=g_s-H_{su}H_{uu}^{\dagger}g_u.
\]

若 `I` 太小，該幀沒有可用的高度量測；不以偽逆把弱方向製造成零方差量測。其餘幀的 data-only 高度中心與方差為

\[
h_{\mathrm{data}}=h_c(1-\gamma/I),\qquad
R=(h_c^2/I+0.05^2)\,q_{\mathrm{quality}},
\]

data-only 高度量測的 `0.05 m` error floor 與既有品質膨脹保留。另有一個分開的舊設定：原 registration objective 將 `HEIGHT_OBSERVATION_FLOOR_M=0.05 m` 平方加到高度 prior variance；高度 filter 的 learned std 即使低於 1 cm，solver 使用的 prior 有效 std 仍至少 5 cm。fix2 原先在 pose accepted 後評估高度；v5.4 在 geometry 及原 scale-std 條件通過後先預覽高度，待 pose／height proposal 一起驗收通過才提交；仍須通過原 physical/search bounds、同一 final planes 的非線性 data-loss gate、1 DoF NIS soft／hard gate，再以既有 Joseph covariance 更新。當幀尺度以 posterior height 除以原幾何係數 `d_s` 輸出；若 height measurement 被拒絕，沿用預測／記憶高度及當幀 ground relation。詳細推導、history、時間與外參限制、生成圖見 [79](../doc/79_TRACK2_CAMERA_HEIGHT_POSITION_20261102.md)。

GNSS／IMU 時間和外參仍非完整標定：header 對齊不證明 GNSS heading 的物理 epoch 無延遲；目前 lever mean 0、std 0.5 m，旋轉外參維持未知。高度與 pose filters 不是完整 joint estimator，未補足跨狀態 covariance。

## GNSS 航向模型（v5.2 引入，v5.3 保留）

先前 Track2 讀取 GNSS XY，但沒有使用 `PoseStamped` 裡的 quaternion。v5.2 讀取有效 GNSS quaternion，和相同 measurement time 的 IMU yaw 配對；v5.3 保留這個狀態：

\[
b = [b_{\mathrm{IMU}}, b_{\mathrm{GNSS}}]^T,\quad P_b=\operatorname{Cov}(b)
\]

其中 `地圖 yaw ≈ IMU yaw + b_IMU ≈ GNSS yaw + b_GNSS`。GNSS–IMU 航向差提供 `[1,-1]` 方向的資訊；從過去 GNSS XY 估出的 course 與 IMU yaw 差提供 `[1,0]` 方向的資訊。只看相對航向差無法知道兩個感測器共同偏了多少，因此濾波器保留完整 covariance 與交叉項。v5.2 保留當時的 map solver；v5.3 在該 solver 後新增 map-only likelihood，避免把 solver 已使用的 sensor prior 再融合一次。雙 bias 與誤差預算見 [72](../doc/72_TRACK2_GNSS_ORIENTATION_CAUSAL_20261029.md)；reference rebase、地圖 likelihood 與限制見 [74](../doc/74_TRACK2_TIME_REFERENCE_LIKELIHOOD_20261031.md)。

只使用已到達的感測器資料，且 measurement timestamp 不晚於當前 camera frame。GNSS–IMU pair 在 gyro norm 達到 `2 deg/s` 時拒絕；相同 held GNSS yaw 不重複當成獨立量測。GNSS XY course 使用已完成的過去視窗，在 1–8 秒範圍內累積；只有方向標準差不高於 `3°` 且未觸發轉向條件時才更新。完整模型、時間界線與限制見 [72｜GNSS 航向因果融合](../doc/72_TRACK2_GNSS_ORIENTATION_CAUSAL_20261029.md)。

## 參數位置

直接編輯 [`parameters.py`](parameters.py)。以下初始 bias 與隨機漫步值是沿用的未知誤差工程預算；GNSS heading observation `3°` 也未校準。沒有用非因果 CSV 擬合這些參數。

| 要調整的內容 | 參數 | 目前設定 |
| --- | --- | ---: |
| 開啟／關閉 GNSS quaternion 航向配對 | `GNSS_HEADING_ENABLED` | `True` |
| IMU 初始 heading bias 標準差 | `HEADING_INITIAL_BIAS_STD_DEG` | `10°` |
| GNSS 初始 heading bias 標準差 | `GNSS_HEADING_BIAS_STD_DEG` | `10°` |
| GNSS quaternion 觀測標準差（另加 IMU covariance） | `GNSS_HEADING_OBSERVATION_STD_DEG` | `3°` |
| 兩種 heading bias random walk | `HEADING_BIAS_RW_DEG_SQRT_S`、`GNSS_HEADING_BIAS_RW_DEG_SQRT_S` | 各 `0.12°/√s` |
| 過去 GNSS course 視窗與方向不確定度門檻 | `HEADING_MIN_WINDOW_S`、`HEADING_MAX_WINDOW_S`、`HEADING_MAX_COURSE_STD_DEG` | `1–8 s`、`3°` |
| v5.10 course 半段一致性門檻 | `HEADING_MIN_SAMPLES`、`CORRECTION_NIS_REJECT_BY_DOF[0]` | 每半至少 `5` 筆；1 DoF NIS `10.828` |
| v5.10 同影格 heading bias 更新預算 | `HEADING_MAX_UPDATE_DEG` | 合計最多 `0.5°`；首幀 bootstrap 除外 |
| 保留 v5.10 course／heading 保護 | `V510_ENABLED` | `True` |
| v5.11.1 開關與比較基準 | `V511_ENABLED`、`BASELINE_RUN_ID` | `True`；`track02_full_2611031941` |
| v5.11 startup 高度 seeds | `STARTUP_HEIGHT_SEED_SIGMAS` | `(-1, 0, 1)` 個 prior 標準差 |
| v5.11 yaw seeds | `STARTUP_INCLUDE_GIVEN_YAW`、`STARTUP_INCLUDE_MAP_AXIS` | `True`；`True` |
| v5.11 tracking alternate retry | `TRACKING_RETRY_ON_GEOMETRY_FAILURE` | `True` |
| v5.11 map proposal data-cost 驗收 | `VALIDATE_EMITTED_MAP_PROPOSAL` | `True` |
| v5.11.1 prior-only 後首筆 map gain 限制 | `STARTUP_BOUND_AFTER_FIRST_FRAME` | `True` |
| 航向 pair 轉向拒絕條件與最大差異 | `HEADING_MAX_TURN_RATE_DEGPS`、`GNSS_HEADING_MAX_PAIR_DIFFERENCE_DEG` | `2°/s`、`30°` |
| GNSS frame 契約 | `GNSS_EXPECTED_FRAME` | `''`（frame 仍需依資料座標定義核定） |

初始 10°、random walk 0.12°/√s 及 observation 3° 都不是感測器精度保證或外參標定結果。GNSS quaternion 的座標語意、GNSS／IMU／PLY 外參與 lever arm 仍未獨立標定。`V510_ENABLED=False` 只供生成資料核對 v5.7 基線等價；v5.11.1 的 height/yaw 候選也只改求解初始值，MAP height objective 與原 search bound 不變，PCA 軸沒有正負方向語意。`0.5°` 是 bias calibration 更新限額，不是車身轉動限制或每幀 pose yaw 限制。course split 只作整筆觀測的品質閘門，不會額外生成第二筆 heading measurement。v5.11.1 的 startup、有限 gain handoff 與 data-cost gate 見 [87](../doc/87_TRACK2_V511_STARTUP_VALIDATION_20261103.md)；v5.10 航向模型見 [86](../doc/86_TRACK2_V510_V57_HEADING_20261103.md)，v5.2 雙 bias 模型見 [72](../doc/72_TRACK2_GNSS_ORIENTATION_CAUSAL_20261029.md)。

## fix1 生成驗證與歷史 regression

fix1 CPU core 檢查 **47 passed**；`checks_summary.json` 記錄 **83 個 unique cases 通過、0 unresolved failures**。結果跨兩次執行彙整：首次整組 81 passed、2 failed，因 C++ oracle 缺 `TRACK2_REFERENCE_LIBRARY`；補入既有 0154 oracle 後，重跑受影響的 5 個 case 全通過。no-accel 是 78 passed 的歷史 runtime A/B。該次 fix1 開發檢查沒有真實定位影格；其後 fix1 full 由使用者回報 Public Score `0.46899`，請見上方。

- [fix1 run](../outputs/run/track00_test_2610311800/test_info.json)、[live 套用收據](../outputs/run/track00_test_2610311800/apply_receipt.json)、[最終 checks summary](../outputs/run/track00_test_2610311800/checks_summary.json)、[47 項 CPU 核心檢查](../outputs/run/track00_test_2610311800/pytest_cpu_02.xml)、[首輪整組結果](../outputs/run/track00_test_2610311800/pytest_final.xml)、[oracle／CUDA／gate 重試](../outputs/run/track00_test_2610311800/pytest_oracle_and_gate.xml)
- [old v5.3 generated baseline](../outputs/run/track00_test_2610311800/analysis/baseline.json)與[fix1 generated 結果](../outputs/run/track00_test_2610311800/analysis/fix1.json)
- [行駛場景 PNG](../outputs/run/track00_test_2610311800/analysis/01_generated_motion.png)／[PDF](../outputs/run/track00_test_2610311800/analysis/01_generated_motion.pdf)、[方向性記憶 PNG](../outputs/run/track00_test_2610311800/analysis/02_directional_hold.png)／[PDF](../outputs/run/track00_test_2610311800/analysis/02_directional_hold.pdf)
- [fix1 模型與生成驗證](../doc/77_TRACK2_V53_FIX1_DIRECTIONAL_MEMORY_20261031.md)
- [v5.3 perf1 恢復 SHA 收據](../outputs/run/track00_test_2610311735/apply_receipt.json)
- [no-accel 78 項測試結果](../outputs/run/track00_test_2610311713/pytest.xml)
- [no-accel 五個 generated scenes 比較](../outputs/run/track00_test_2610311713/comparison.json)
- [no-accel 範圍稽核](../outputs/run/track00_test_2610311713/rollback_scope_audit.json)
- [真實前五幀加速／no-accel A/B](../outputs/run/track00_test_2610311724/analysis/README.md)
- [v5.3 regression analysis 與記帳結果](../doc/76_TRACK2_V53_REGRESSION_AUDIT_20261031.md)
- [1735 regression 整段摘要](../outputs/run/track00_test_2610311735/analysis/regression_summary.json)
- [1735 mean attribution](../outputs/run/track00_test_2610311735/analysis/mean_attribution.json)
- [1735 partial-rank null rebase 分析](../outputs/run/track00_test_2610311735/analysis/null_rebase.json)
- [1735 generated counterexample 摘要](../outputs/run/track00_test_2610311735/analysis/generated_degenerate_summary.json)
- [1735 regression 圖表與分析索引](../outputs/run/track00_test_2610311735/analysis/README.md)（建立中）
- [v5.2 full 全段唯讀分析索引](../outputs/run/track00_test_2610311554/analysis/README.md)
- [1527 分析報告索引](../outputs/run/track00_test_2610291527/analysis/README.md)
- [1602 驗證圖表索引](../outputs/run/track00_test_2610291602/analysis/README.md)
- [完整 v5.2 算法與限制說明](../doc/72_TRACK2_GNSS_ORIENTATION_CAUSAL_20261029.md)
- [v5.3 時間參考與地圖似然](../doc/74_TRACK2_TIME_REFERENCE_LIKELIHOOD_20261031.md)
- [v5.3 no-accel 改動與驗證](../doc/75_TRACK2_V53_NO_ACCEL_20261031.md)

doc76 的資料是原 v5.3（fix1 前）baseline，提出的 rebase／directional expiry／likelihood acceptance 風險已對應到 live fix1 的三項調整。兩個生成場景各有 322 state steps：加速場景的 moving-x RMSE 變化是「GNSS XY innovation 不再抵消舊記憶」與「方向性衰退」合併效果；另一個只觀測 y/yaw 的場景直接檢查方向性 expiry，未觀測 x correction 由 `0.376470588 m` 衰退至 `0.060190022 m`。這些是生成機制結果，不是 Track2 精度或 `.80300` 的因果拆解。full 仍由使用者手動啟動。

<details>
<summary>歷史附錄：v5／v5.1 地圖配準機制</summary>

### v5 的 full 與首 30 幀診斷

v5 full `track02_full_2610290851` 已完整輸出 3299 幀，map update 為 0，原始 source 與結果保留。唯讀凍結的[前 30 幀診斷](../outputs/run/track00_test_2610290857/v5_first30_observations.jsonl)顯示可靠 point-to-plane coverage 約 5–12%，30 幀皆未接受 map geometry；v5 的 `MIN_PLANE_COVERAGE=0.35` 全影像比例門檻因此無法通過。各幀也記錄 `line_search_failed`，所以不能把所有拒絕只歸因於 coverage。[首 30 幀圖表與分析](../outputs/run/track00_test_2610290857/analysis/README.md)只分析 v5，不是 v5.1 或 v5.2 replay。

### v5.1 稀疏可靠平面與 line search 修正

1. 不再以全影像 plane coverage 比例一刀切。改用 `angular_support = Σ(source angular weight × reliable match)`，最低支持量為 `PLANE_MIN_SUPPORT_PER_VARIABLE × (observable_rank + 1)`。
2. 移除 `R` 的 `1 / global_coverage²` 膨脹。data information `H` 只累加可靠、帶 angular weight 的平面匹配；匹配減少時資訊自然下降，再以全圖 coverage 膨脹會重複扣資訊。error floor 與 residual quality inflation 保留。
3. 每次 Gauss–Newton line search 固定同一組 point-to-plane associations；接受步驟後才在下一輪 ICP 重新配對。candidate 與 seed 用同一組 final planes 比較，收斂依實際採用的 step 判斷。

### v5／v5.1 的因果 map update 基礎

map 點雲表面通常只在法向方向提供可靠距離。v5 使用 CUDA 局部平面法向殘差，只把資料觀測到的 XY／yaw 方向交給濾波器。每幀 map `[x,y,yaw]` prior 攜帶完整 covariance；point-to-plane 資料資訊轉到世界座標、將 log-scale 作未知量邊際化，再依當幀 covariance whitening。只有資訊量足夠的 eigen-direction 形成部分觀測量；弱方向不會由地圖虛構切向資訊。

v5.1 另保留 v4 correction-memory：短暫失敗時 hold，長時間失敗後逐漸 fade；NIS、motion、registration quality gate 與每幀限幅共同控制更新。當幀地面尺度與傾斜仍用原本幾何估計，不對 raw scale 盲目低通，也不回改舊 pose。完整可觀測性模型見 [doc68](../doc/68_TRACK2_OBSERVABLE_PLANE_20261029.md)，稀疏平面與 line-search 歷史見 [doc69](../doc/69_TRACK2_PLANE_SUBSET_ICP_20261029.md)，v4 correction-memory 見 [doc67](../doc/67_TRACK2_CAUSAL_CORRECTION_MEMORY_20261029.md)。

</details>

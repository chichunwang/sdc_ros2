# Track2：因果 GNSS 航向融合 v5.2

效能版 `v5_2_perf1` 省去內迭代未使用的診斷與 Jacobian，遮罩點雲留在 GPU，RViz 逐點轉換使用 CUDA；定位參數不變。2026-10-31 僅完成 49 項 CPU 檢查，CUDA 比較與套用狀態以 run 的收據為準，操作見 [73｜效能優化](../doc/73_TRACK2_PERFORMANCE_20261031.md)。

## 現況

本快照版本是 `track2_causal_dual_heading_bias_v5_2_perf1`；已套用與否請看效能測試 run 的 `apply_receipt.json`。v5.2 保留 v5.1 的公尺尺度地圖配準與 correction-memory，只增加 GNSS `PoseStamped` orientation 的因果使用，並以兩個航向 bias 和完整 2×2 covariance 表示 GNSS 與 IMU 的相對不確定度。已輸出的 pose 不會回頭修改。

使用者回報的 full／CSV 數值如下；v5 與 v5.1 都已有 full，v5.2 尚無真實 full。三個分數已由使用者確認為同一 Track2 競賽的 Kaggle Public Score；這些回報仍不能替代定位真值或保證 v5.2 成績。

| 來源 | 使用者回報 | 狀態 |
| --- | ---: | --- |
| v5 full | 0.53683 | 3299 幀 full；map update 為 0 |
| v5.1 full `track02_full_2610291400` | 0.51617 | full 已完成；改動同時含地圖配準 |
| `result_2_submitnoncasual.csv` | 0.30488 | 非因果輸出，只供唯讀對照 |

Public Score 由 v5 的 0.53683 降至 v5.1 的 0.51617，下降 0.02066（約 3.85%）；距離 0.30488 尚差 0.21129。[分數確認與比較紀錄](../outputs/run/track00_test_2610292245/analysis/README.md)。

非因果 CSV 不作 ground truth，不用來調 bias、offset、時間差或任何參數；v5.2 不保證能達到 0.30488。

## 啟動

完整定位由使用者手動啟動。在 Ubuntu 主機執行：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
./scripts/run_track2.sh --full
```

入口會啟動 Track2 RViz（`/track2/*`，TF `map → track2_camera`），設定在 [`config/track2.rviz`](../config/track2.rviz)。Track1／2／3 共用 GPU lock；另一 Track 正執行時，runner 會拒絕並行配準。Full 輸出在新的 `outputs/run/track02_full_YYMMDDHHmm/`。

## v5.2 如何使用 GNSS 航向

先前 Track2 讀取 GNSS XY，但沒有使用 `PoseStamped` 裡的 quaternion。v5.2 讀取有效 GNSS quaternion，和相同 measurement time 的 IMU yaw 配對。狀態為

\[
b = [b_{\mathrm{IMU}}, b_{\mathrm{GNSS}}]^T,\quad P_b=\operatorname{Cov}(b)
\]

其中 `地圖 yaw ≈ IMU yaw + b_IMU ≈ GNSS yaw + b_GNSS`。GNSS–IMU 航向差提供 `[1,-1]` 方向的資訊；從過去 GNSS XY 估出的 course 與 IMU yaw 差提供 `[1,0]` 方向的資訊。只看相對航向差無法知道兩個感測器共同偏了多少，因此濾波器保留完整 covariance 與交叉項。Map prior 沿用舊 scalar 介面讀取 `b_IMU` 及其邊際 variance；v5.1 map registration 沒有改版。

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
| 航向 pair 轉向拒絕條件與最大差異 | `HEADING_MAX_TURN_RATE_DEGPS`、`GNSS_HEADING_MAX_PAIR_DIFFERENCE_DEG` | `2°/s`、`30°` |
| GNSS frame 契約 | `GNSS_EXPECTED_FRAME` | `''`（frame 仍需依資料座標定義核定） |

初始 10°、random walk 0.12°/√s 及 observation 3° 都不是感測器精度保證或外參標定結果。GNSS quaternion 的座標語意、GNSS／IMU／PLY 外參與 lever arm 仍未獨立標定。參數意義與 covariance 更新方式見 doc72。

## 驗證與證據

v5.2 已套用 live；套用前確認沒有活動中的定位程序，並比對修改檔案 SHA。Track2 生成資料、CPU／CUDA／ROS 與 runner 檢查 **58 passed、0 failed**，pytest 摘要 14.40 秒、XML suite time 14.402 秒；僅有既有 Matplotlib `Axes3D` warning，不影響 2D 圖。這不構成 v5.2 真實定位或精度證明。目前尚無 v5.2 full 或分數。

- [套用收據與 source SHA](../outputs/run/track00_test_2610291602/apply_receipt.json)
- [58 項檢查結果](../outputs/run/track00_test_2610291602/pytest.xml)
- [1527 分析報告索引](../outputs/run/track00_test_2610291527/analysis/README.md)
- [1602 驗證圖表索引](../outputs/run/track00_test_2610291602/analysis/README.md)
- [完整 v5.2 算法與限制說明](../doc/72_TRACK2_GNSS_ORIENTATION_CAUSAL_20261029.md)

分析報告與 PNG／PDF 圖表已完成；測試和唯讀分析不會自行啟動 full。下一次真實定位由使用者依上方命令手動啟動。

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

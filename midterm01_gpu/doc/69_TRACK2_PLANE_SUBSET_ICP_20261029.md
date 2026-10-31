# Track2 v5.1：稀疏可靠平面與穩定 ICP line search

本文說明現行的 `track2_causal_observable_plane_v5_1`。正式 v5 full `track02_full_2610290851` 已自然完成 3299 幀，`map_accepted_frames=0` 且 `analysis_status=completed`；舊 source 與輸出完整保留。v5.1 的 [50 項合成 CPU／CUDA／runner 檢查](../outputs/run/track00_test_2610290947/pytest.xml)全數通過（pytest 摘要 14.47 秒；XML suite time 14.478 秒），最新[分析報告索引](../outputs/run/track00_test_2610290947/analysis/README.md)已建立。v5.1 已核對 SHA 並套用工作區；尚無 v5.1 真實 full、定位精度或分數證據。



## 實跑先暴露什麼

在該 full 前 30 幀的唯讀 freeze 中，[coverage 診斷](../outputs/run/track00_test_2610290857/v5_first30_observations.jsonl)約為 5–12%；30 幀皆沒有接受 map geometry，輸出為 prior-only。相同 freeze 的[診斷圖表索引](../outputs/run/track00_test_2610290857/analysis/README.md)與 [PNG](../outputs/run/track00_test_2610290857/analysis/01_v5_sparse_plane_diagnosis.png)／[PDF](../outputs/run/track00_test_2610290857/analysis/01_v5_sparse_plane_diagnosis.pdf)只分析這前 30 幀，不是 v5.1 replay。凍結 v5 的 [參數](../outputs/run/track02_full_2610290851/source/track2/parameters.py)要求 `MIN_PLANE_COVERAGE = 0.35`，亦即可靠平面至少占全體 angular weight 的 35%。這個比例門檻會拒絕「只有一小部分點有可靠平面，但這些點仍可能提供有效限制」的影格。v5 full 最終完成 3299 幀且 map update 全數未接受；其輸出與完整分析狀態保留，這不是精度證明。

同一批診斷也記錄每幀的三個配準 stage 都出現 `line_search_failed`（共 90 個 stage 記錄）。這是與 coverage gate 同時存在的另一項失敗條件：即使改善 support 判定，line search 仍需有可比較的候選成本。不能把所有拒絕都簡化成單一原因，也不能把前一版的 47 項合成／CUDA／輸出檢查通過說成這次實跑成功；那些測試沒有涵蓋這種稀疏可靠平面與 association 變動同時出現的情況。[前版測試記錄](../outputs/run/track00_test_2610290847/pytest.xml)只證明對應測試案例通過。

## v5.1 如何處理稀疏平面

### 1. 以可靠平面的 angular support 判斷資料量

程式仍記錄全圖 `coverage` 作為診斷，但不再用固定的全圖比例門檻否決 map candidate。改計算可靠匹配點的 angular support：

`A_support = Σᵢ wᵢ · 1(reliable point-to-plane matchᵢ)`

其中 `wᵢ` 是 source angular weight。要求的最低值為：

`A_required = PLANE_MIN_SUPPORT_PER_VARIABLE · (observable_rank + 1)`

目前 staging 預設 `PLANE_MIN_SUPPORT_PER_VARIABLE = 2.0`；`observable_rank` 是 whitened 資訊矩陣保留的 pose 模式數，額外 `+1` 為 scale nuisance variable。這個門檻要求每個可觀測 pose 模式和 scale 至少有兩個 angular-weight 等效支持，並非精度校準值。

這不表示少數點必然足以定位。最低 matches、plane residual、資料資訊的 whitened observable modes、NIS 與 motion gate 仍然生效；沒有可觀測 pose 模式或 support 未達門檻時，仍拒絕 map update，並沿用既有 causal correction 記憶行為。

### 2. 不把同一批稀疏資料重複降權

資料資訊矩陣 `H` 只累加可靠、帶 angular weight 的 point-to-plane 匹配。可靠支持較少時，`H` 自然變弱，保留模式的 covariance 也會隨資訊減少而增大。v5.1 因此移除額外的 `1 / global_coverage²` covariance inflation，避免樣本數已反映在 `H` 後，又用全圖 coverage 對相同資料再扣一次資訊。

量測 error floor 與依 plane residual 計算的 quality inflation 保留。這仍是局部資訊與誤差模型，尚未由獨立真值校準。

### 3. 在 line search 內固定 point-to-plane associations

最近鄰或平面 ID 隨 pose 候選改變時，成本函數會混入「換了比較對象」的跳動。v5.1 的每次 Gauss–Newton line search 固定當前 iteration 的 point-to-plane associations，讓 backtracking trial 都比較同一組點、法向與平面中心；接受 step 後，下一次 ICP/Gauss–Newton iteration 才重新配對。

最後的 map candidate 與原始 seed 也使用同一組 final planes 計算成本，避免因最終比較時重新配對而出現離散的假成本差。停止條件依 line search 實際採用的 step，而不是未縮放的原始提案。這讓比較更一致，但不能保證低重疊或錯誤對應下仍會收斂。

## 保留的 v5 機制與目前界線

v5.1 保留 point-to-plane 可觀測子空間、log-scale 邊際化、當幀 prior covariance、部分觀測 NIS、XY／yaw gain 限幅、短失敗 hold 與長失敗 fade，以及因果 heading initialization／calibration rebase。尺度與地面傾斜仍按既有當幀估計處理。完整基礎模型見 [68｜可觀測平面因果融合](68_TRACK2_OBSERVABLE_PLANE_20261029.md)。

v5.1 source 與入口見 [Track2 README](../track2/README.md)。[50 項合成 CPU／CUDA／runner checks](../outputs/run/track00_test_2610290947/pytest.xml)全數通過（pytest 摘要 14.47 秒；XML suite time 14.478 秒）；最新[分析報告索引](../outputs/run/track00_test_2610290947/analysis/README.md)已建立。這些合成與 runner checks 不等於 v5.1 真實 full 或精度證據。先前 [35 項 CPU-only NoCuda 檢查](../outputs/run/track00_test_2610290859/pytest.xml)與前 30 幀唯讀圖表保留為歷史證據；後者是 v5 freeze，不能當成 v5.1 replay。v5 full 3299 幀已完成並保留，map accepted 為零；v5.1 已套用工作區。第二次 full 尚未授權，也不會自動重跑。


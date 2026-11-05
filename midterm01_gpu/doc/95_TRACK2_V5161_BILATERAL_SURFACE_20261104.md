# Track2 v5.16.1：bounded bilateral surface fallback

## 目前狀態

**後續正式結果：使用者回報 0.44629，比 v5.10 的 0.42699 差 0.01930，停止推薦原樣重複提交。** 以下為當時實作與 full 研究紀錄；正式分數另存 [收據](../outputs/run/track00_test_2611050118/score_receipt.json)，原 full metadata 保持不變。後續原因拆解見 [97](97_TRACK2_V5163_ROAD_PATCH_20261105.md)。

本輪當時 production 是 `track2_v516_bilateral_surface_v5_16_1_perf2`。本次新增的是受界限約束的雙側表面 fallback：原本可靠的配對全部保留，只替換原最近地圖點不可靠、且來源點局部平面可靠的配對。設計目標是讓附近另一個可靠表面有機會提供幾何約束；目前證據尚不能說明它提高了真實定位精度。

本版沿用 v5.16／v5.10 數值融合、40 m feature 範圍、6000 點上限、`3 / 1.5 / 0.75` 配對設定、18 m ground 設定，以及 GNSS、heading、height、scale 和強資訊門檻 `1`。v5.15 的弱方向融合與獨立 likelihood 不啟用。使用者正式目標分數 `0.30488` 尚未達成，當時尚無新正式成績；後續回報見上方。3299 幀 full [`track02_full_2611042153`](../outputs/run/track02_full_2611042153/test_info.json) 已完成；不含 Kaggle 等外部提交。

## 觀察與原因假說

最近的地圖點若沒有可靠平面，原流程會丟棄該來源配對。這能避免把不可靠表面當成約束，但當原最近點失敗、稍遠處其實有可靠表面時，也會失去可用配對。v5.16.1 只在來源與地圖兩側都能支持局部表面幾何時，嘗試有限範圍的替代配對。

替代配對須符合以下界限：

- 搜尋目前半徑內最多 `K=16` 個候選。
- 新地圖點距離不得超過原最近地圖點距離加一個 map voxel，即 `0.25 m`。
- 來源與目標表面法向的絕對內積須至少為 `cos(15°)`，避免配對方向差異過大。
- 配對 confidence 取來源與目標 confidence 的較小值。

來源與目標兩側的平面可靠度沿用原有 PCA 平面度、厚度及點數門檻；沒有用外部 target CSV 擬合平面或修改配對。表面替代只在已有過去 map anchor、且當幀 prior yaw 標準差不大於 `3°` 時啟用。啟動初期或航向仍不確定時沿用原流程。

## 模型與處理步驟

此 fallback 使用既有估計位姿附近的來源點與局部地圖表面，不引入新真值。令來源平面單位法向為 \(n_s\)，候選目標平面法向為 \(n_t\)；方向相容條件是：

\[
|n_s^\mathsf{T} n_t| \geq \cos(15^\circ).
\]

實際流程如下：

1. 保留原本已通過可靠度檢查的最近點配對。
2. 對原最近地圖點不可靠的來源點，先確認來源點局部平面可靠。
3. 在當前搜尋半徑及 `K=16` 上限內尋找替代地圖點，套用 `0.25 m` 距離增量與法向相容限制。
4. 若候選目標平面也可靠，才以來源／目標 confidence 的較小值建立替代配對；沒有合格候選就維持原本丟棄結果。

程式模組為 `surface_correspondence.py`，由 metric registration 的 prepare 與 association 流程呼叫。可在 `parameters.py` 修改 `V5161_ENABLED` 及兩個啟用 gate。處理只使用當幀資料及過去 anchor，不讀取未來影格，也不回修舊 pose。

## 驗證結果

研究 run 為 `outputs/run/track00_test_2611042122/`。19 項 focused CPU/CUDA 合成檢查已通過，耗時 `7.40 s`，涵蓋搜尋界限、法向條件、保留原可靠配對、gradient 與因果 prefix。完整 Track2 CPU/CUDA suite 為 `303 passed`，耗時 `45.76 s`。關閉新功能後，201 個 sensor frames 與 3 個 CUDA registration frames 的數值、狀態、covariance、memory 與 sensitivity 均和 v5.16 baseline 完全相同，確認此開關未改動舊路徑。

另有 13 個固定歷史 pose 的唯讀診斷。垂直線方案只增加 1 個配對且沒有提高 rank，因此未採用。表面 fallback 在 frame 500 增加 17 個配對、rank 由 0 升至 1；frame 1000 增加 132 個、frame 1500 增加 35 個、frame 2865 增加 85 個。frame 2792 的來源平面條件不足，沒有新增配對。這些是固定 pose 下的配對診斷，不是真實定位精度驗證。

## 真實 800 幀對照

候選 run [`track02_test_2611042137`](../outputs/run/track02_test_2611042137/test_info.json) 與 control [`track02_test_2611042037`](../outputs/run/track02_test_2611042037/test_info.json) 各完成前 800 幀；209 個來源 SHA 與 6 個輸出 SHA 核對通過，兩輪 manifest 與 input_sources 相同。新分支 tracking gate 首次成立於 idx 482，fallback 首次使用於 idx 493，CSV 首次出現差異於 idx 494；在 gate 之前兩版本相同。兩版首次 map update 都在 idx 4。

| 800 幀指標 | v5.16 control | v5.16.1 候選 |
| --- | ---: | ---: |
| map accepted | 263 | 340 |
| scale validated | 186 | 262 |
| fallback 新增 surface matches | — | 303 幀，共 29591 個 |
| localization 耗時 | 573.40 s | 680.35 s |

兩版 XY 輸出最大差為 `0.20898 m`（idx 692），p95 為 `0.14413 m`。六個 map 疊圖與 800 幀差異圖已檢視，路徑大致重疊；這些圖只能描述輸出，不能證明精度提升。

在共同原模型的 0–40 m 固定 10 幀成本比較中，候選 5 幀較低、2 幀較高、3 幀相同；candidate-control 平均差為 `+0.0000691523`。因此整體成本沒有一致改善，不能只挑較低的 5 幀推論候選更好。

對照圖位於 `outputs/run/track00_test_2611042122/analysis/`：

- [03｜800 幀輸出差異 PNG](../outputs/run/track00_test_2611042122/analysis/03_actual_800_differences.png)／[PDF](../outputs/run/track00_test_2611042122/analysis/03_actual_800_differences.pdf)
- [04｜surface fallback 對照 PNG](../outputs/run/track00_test_2611042122/analysis/04_actual_surface_support.png)／[PDF](../outputs/run/track00_test_2611042122/analysis/04_actual_surface_support.pdf)
- [05｜原模型共同成本 PNG](../outputs/run/track00_test_2611042122/analysis/05_original_model_common_cost.png)／[PDF](../outputs/run/track00_test_2611042122/analysis/05_original_model_common_cost.pdf)
- [06｜800 幀 map 軌跡 PNG](../outputs/run/track00_test_2611042122/analysis/06_actual_800_map_paths.png)／[PDF](../outputs/run/track00_test_2611042122/analysis/06_actual_800_map_paths.pdf)

## 真實 3299 幀 full

[`track02_full_2611042153`](../outputs/run/track02_full_2611042153/test_info.json) 完成 3299 幀，localization 耗時 `2363.6483 s`。209 個來源 SHA 與 6 個輸出 SHA 核對通過；frame manifest 與 v5.10 full 相同。full 的前 800 幀 CSV 與 poses 和候選 short 完全相同，sensor prior XY 與 v5.10 full 逐位相同。標準分析完成 8 組 PNG/PDF，含 trajectory、pose changes、registration timing、fusion/scale、map correction memory、observability、image ground speed、heading information；詳見 [full analysis README](../outputs/run/track02_full_2611042153/analysis/README.md)。

| Full 對照 | v5.10 full | v5.16.1 production |
| --- | ---: | ---: |
| map 更新 | 1774 | 2049 |
| scale validated | 1479 | 1763 |
| fallback 有作用的幀數／新增配對 | — | 2699 幀／307504 個 |

兩版 XY 輸出差 median `0.047583 m`、p95 `0.144542 m`、最大 `0.237438 m`（idx 2441）；yaw 最大差 `0.285097°`。這些是版本輸出差，不是逐幀真實誤差。

19 個固定幀與 v5.10 full 的共同成本比較中，原模型有 12 幀候選成本較低、6 幀較高、1 幀相同，candidate-control 平均為 `−0.000346570`；surface 模型也是 12 低、6 高、1 同，平均 `−0.000257266`。全段 map 更新主要集中在特定窗口，其他低更新區仍未改變：

| Frame window | v5.10 → v5.16.1 map accepted |
| ---: | ---: |
| 478–506 | 0 → 3 |
| 1450–1546 | 82 → 81 |
| 2643–2671 | 0 → 0 |
| 2773–2864 | 0 → 0 |
| 3154–3211 | 2 → 11 |

整體效果有限，未有正式分數，也不能由更新數、固定幀成本或兩版本輸出差推定精度改善。對照圖由主 agent 檢視：

- [09｜full 版本變化](../outputs/run/track00_test_2611042122/analysis/09_full_version_changes.png)／[PDF](../outputs/run/track00_test_2611042122/analysis/09_full_version_changes.pdf)
- [10｜full 問題窗口](../outputs/run/track00_test_2611042122/analysis/10_full_problem_windows.png)／[PDF](../outputs/run/track00_test_2611042122/analysis/10_full_problem_windows.pdf)
- [11｜full 共同成本](../outputs/run/track00_test_2611042122/analysis/11_full_common_cost.png)／[PDF](../outputs/run/track00_test_2611042122/analysis/11_full_common_cost.pdf)
- [12｜full map/source early](../outputs/run/track00_test_2611042122/analysis/12_full_map_source_early.png)、[late](../outputs/run/track00_test_2611042122/analysis/12_full_map_source_late.png)／[early PDF](../outputs/run/track00_test_2611042122/analysis/12_full_map_source_early.pdf)、[late PDF](../outputs/run/track00_test_2611042122/analysis/12_full_map_source_late.pdf)

## 後續研究：未採用或尚未部署

另行檢視 v5.13 的歷史 iteration trace：758 個未收斂幀中，295 幀最後五次 cost 呈穩定兩步循環，其中 181 幀 rank 大於 0。假說是固定配對下 line search 可以下降，但外層重新配對後 cost 來回循環；這不代表真實定位誤差。圖表：[07｜matching cycles](../outputs/run/track00_test_2611042122/analysis/07_matching_cycles.png)／[PDF](../outputs/run/track00_test_2611042122/analysis/07_matching_cycles.pdf)。

獨立的暫存 `research_cycle_source` 原型在偵測到循環後，檢查重新配對後的 Armijo 下降；沿用原停止解析度 `2 mm / 0.0002 rad / 0.0002 log-scale`，必要時用四軸正負 probe 判斷是否已受座標解析度限制，並保留首個 anchor 流程及原資訊／品質門檻。CPU toy 檢查中，舊版達 40 次迭代上限，新版在 5 次時停止，cost 為 `0.005`（舊版 `0.02`）；smooth 與無 anchor toy 結果相同，搜尋邊界不直接判為 stationary。這些只是合成數值，不是定位精度驗證。

20 個固定歷史 frame 的 CUDA solver 候選重播已完成，但只比較候選求解器，沒有 filter commit、Track CSV 或真值精度比較。geometry accepted 由 `7` 增至 `10`，在 idx `19`、`205`、`813` 救回循環案例，沒有 lost case；候選與 control 最大 XY 差為 `0.0068 m`。共同原模型成本有 7 幀較低、4 幀較高、9 幀相同，平均差 `−4.2683e−5`。這不是連續 Track 定位，cycle guard 未部署到 v5.16.1。圖：[13｜cycle 固定幀重播](../outputs/run/track00_test_2611042122/analysis/13_cycle_fixed_frame_replay.png)／[PDF](../outputs/run/track00_test_2611042122/analysis/13_cycle_fixed_frame_replay.pdf)。

另對歷史 source ground normal 與已到達 orientation 做一致性抽查，角度差 median 為 `3.157°`、p95 為 `5.161°`。這不足以證明外參真值，因此未採用 ground 外參校正。圖表：[08｜ground orientation consistency](../outputs/run/track00_test_2611042122/analysis/08_ground_orientation_consistency.png)／[PDF](../outputs/run/track00_test_2611042122/analysis/08_ground_orientation_consistency.pdf)。

12 個固定 pose 的 normal-yaw 診斷多數沒有牆面配對，或只有 1 個；加權支持最高 `1.155` 且方向分散，沒有採作 yaw 量測。[14｜normal yaw consistency](../outputs/run/track00_test_2611042122/analysis/14_normal_yaw_consistency.png)／[PDF](../outputs/run/track00_test_2611042122/analysis/14_normal_yaw_consistency.pdf)。

兩項固定 pose 幾何假說均未採用。將 source floor 降至 metric `0.1 m / 3 MAD` 後，8 幀的 rank 全部不變；idx 2792 樣本由 `105→118`、matches `31→42`，rank 仍為 0 且沒有垂直結構支持。[15｜low-height structure](../outputs/run/track00_test_2611042122/analysis/15_low_height_structure.png)／[PDF](../outputs/run/track00_test_2611042122/analysis/15_low_height_structure.pdf)。將範圍由 40 m 擴至 60 m，8 幀 rank 同樣全不變，7 幀點數完全相同；只有 idx 650 增至 6000 點上限。[16｜extended-range structure](../outputs/run/track00_test_2611042122/analysis/16_extended_range_structure.png)／[PDF](../outputs/run/track00_test_2611042122/analysis/16_extended_range_structure.pdf)。這些是固定 pose 的結構抽查，不是定位比較。

新的 NHC 運動模型研究另存於 [`track00_test_2611042251`](../outputs/run/track00_test_2611042251/test_info.json)，目前未修改 production，不能視為 v5.16.1 功能。[相關文獻核對紀錄](../outputs/run/track00_test_2611042122/research_literature.md)。

## 限制

fallback 只能利用已有可靠來源與目標表面建立配對；缺少局部平面或有效幾何時不會憑空產生定位資訊。固定 pose 下新增配對或 rank 變化不代表 pose 更接近真實值。800 幀及 full 對照都沒有獨立真值支持精度結論。正式目標分數 `0.30488` 仍未達成；本文件不將候選差異歸因為單一機制，也不宣稱精度提高。循環求解器、normal-yaw、low-height、extended-range 與 NHC 均不是目前 production 功能。

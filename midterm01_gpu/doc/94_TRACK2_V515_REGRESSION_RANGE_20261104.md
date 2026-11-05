# Track2 v5.15：full regression 與有效範圍診斷

## 結論與證據範圍

**目前交付 v5.16 已驗證回復版，恢復 v5.10（40 m），v5.15 分支停用。** 最終完整 Track2 CPU／CUDA 檢查為 **294 passed、47.10 秒、1 項既有 Matplotlib Axes3D warning**。40 m 實跑重現 v5.10 full 的前 800 幀；既有已評分結果可保留，毋須為回復版重跑 full。30 m 已完成對照且未採用；v5.16 尚無新榜分。

v5.15 full [`track02_full_2611041916`](../outputs/run/track02_full_2611041916/test_info.json) 已完成 3299 幀；使用者回報 Public Score `0.51729`。v5.10 full [`track02_full_2611031941`](../outputs/run/track02_full_2611031941/test_info.json) 使用者回報 `0.42699`。兩輪 input 與 frame manifest 相同，v5.15 有 204 個 source 與 6 個 output SHA 核對完整。這些分數和軌跡輸出只能比較版本結果，不能指出單一機制造成分數差，也不是逐幀真實誤差。

使用者正式目標分數為 `0.30488`。參照檔 [`result_2_0.30488.csv`](<analysis doc/result_2_0.30488.csv>) 的 SHA256 為 `5b04d260123a8d87f0f0c317ddbce5b59c3ee3068cf07e020a370034a9728dbb`。此 CSV 僅供記錄目標與唯讀索引對照，不作逐幀真值，也不據此回修 pose。

## Full 對照：首筆 map 更新與全段輸出

v5.15 保留弱資訊方向，並按 retained rank 計算 `support_required`。在 idx 4，所需角度支持為 `8`，實際為 `6.13066`；v5.10 同幀只要求 `4`，因此 v5.10 在 idx 4 首次接受 map 更新，v5.15 延至 idx 188。v5.15 首筆 map correction 為 `[+0.21705, −0.04381, −0.011553 rad]`。這說明支持條件改變了啟動時機，不等於指出其精度好壞。

| Full 診斷項目 | v5.10 | v5.15 |
| --- | ---: | ---: |
| map 更新次數 | 1774 | 1286 |
| posterior 雙成本拒絕 | — | 526 |
| backtracking 後通過的更新 | — | 297 |

原因假說是局部模型與真實資料之間的落差：生成二次模型能驗證「先驗只計一次」的公式，卻沒有重現錯誤配對、相關重建誤差與啟動期稀疏支持。實際弱方向的 data-only center 可外推數公尺，雙成本檢查又可能拒絕更新。因此要分別檢查支持門檻、局部模型有效範圍與輸出，不能用合成公式正確或成本通過當作精度結論。

全段 v5.15 與 v5.10 的 XY 輸出差異 median 為 `0.12857 m`、p95 為 `0.38199 m`、最大 `0.79910 m`（idx 3005）。它是兩套輸出的差，不是定位誤差；map 更新數、拒絕數及回溯通過數也不是精度指標。

完整逐幀彙整見 [`full_regression_summary.json`](../outputs/run/track00_test_2611042012/full_regression_summary.json)；圖表均為估計器輸出／診斷對照：

- [全段輸出差異](../outputs/run/track00_test_2611042012/analysis/01_full_differences.png)／[PDF](../outputs/run/track00_test_2611042012/analysis/01_full_differences.pdf)
- [啟動期支持不足](../outputs/run/track00_test_2611042012/analysis/02_startup_support_failure.png)／[PDF](../outputs/run/track00_test_2611042012/analysis/02_startup_support_failure.pdf)
- [likelihood 局部診斷](../outputs/run/track00_test_2611042012/analysis/03_likelihood_locality.png)／[PDF](../outputs/run/track00_test_2611042012/analysis/03_likelihood_locality.pdf)
- [實際 map 更新路徑](../outputs/run/track00_test_2611042012/analysis/04_actual_map_paths.png)／[PDF](../outputs/run/track00_test_2611042012/analysis/04_actual_map_paths.pdf)

## 已授權短段與 feature range 抽查

已完成的真實短段各取前 300 幀：40 m 版本為 [`track02_test_2611042015`](../outputs/run/track02_test_2611042015/test_info.json)，30 m 版本為 [`track02_test_2611042015_0002`](../outputs/run/track02_test_2611042015_0002/test_info.json)。兩個 run 的輸出 CSV 相同，各有 84 次 map 更新，首次更新都在 idx 4；40 m 設定逐數值重現 v5.10 full 的前 300 幀。因此本次短段沒有證據支持「30 m 提高精度」的說法。

目前正式設定為 feature max `40 m`、submap half extent `(60, 60, 20)`、matching settings `3 / 1.5 / 0.75`、ground setting `18`。只讀點數抽查如下；括號為該幀抽樣點總數：

| Frame | feature 點距離大於 30 m | 抽樣總數 | 訊號 |
| ---: | ---: | ---: | --- |
| 500 | 222 | 1679 | 有遠距點，但不表示它們形成可靠配準 |
| 704 | 625 | 5448 | 遠距點較多；30–40 m plane residual 中位數較大 |
| 2000 | 43 | 2501 | 遠距 reliable-plane coverage 為 0 |
| 2496 | 19 | 1460 | 遠距 reliable-plane coverage 為 0 |

idx 0 與 idx 299 的最大 feature range 分別為 `22.456 m`、`23.248 m`。idx 704 的 0–20 m plane residual 中位數為 `0.08678 m`，30–40 m 為 `0.23190 m`。這些是取樣幾何與殘差摘要，不證明遠距點提升或降低定位精度；遠距可靠平面 coverage 為 0 的區段也不能由弱方向 likelihood 補出缺失的幾何資訊。

## 追加 800 幀對照與採用決定

追加授權已執行完畢：40 m／30 m 各前 800 幀，加上先前各 300 幀，累計 **2200 幀，沒有 full**。來源快照與輸出 SHA、相同 input／frame manifest 已核對。40 m control 的 CSV、poses、frame manifest 逐位元組重現 v5.10 full 前 800 幀。

| 項目 | 40 m control | 30 m candidate |
| --- | ---: | ---: |
| Map 更新 | 263 | 236 |
| 尺度通過驗證 | 186 | 158 |
| 第 650 幀 source sample | 3958 | 450 |
| 第 650 幀可觀測 pose rank | 1 | 0 |
| 第 650 幀 map 更新 | 通過 | 拒絕 |

第 393 幀是點數與 CSV 首次不同之處；兩組最大 XY 輸出差異 0.38204 m（idx 708），不是精度誤差。第 650 幀自己的 feature cost 雖從 0.16304 降到 0.15166，但使用的 sample 已大幅減少，不能據此判斷變好。在相同 40 m 原始 sample、相同權重及 unmatched penalty 下，抽查 10 幀的 30 m 輸出有 **6 幀成本較高、3 幀較低、1 幀相同**；平均差 `+0.00007972`，近距 0–20 m 的平均差也略高。成本差很小，不能推成精度變差，但配合結構／更新流失，沒有採用 30 m 的充分證據。

因此 **保留 40 m**。縮短距離會移除部分殘差較大的遠點，也可能移除提供 yaw／scale 約束的主要結構；本次真實對照顯示，不能只看遠點殘差或自己的成本下降就裁掉遠點。

- [800 幀輸出差異](../outputs/run/track00_test_2611042012/analysis/08_range_800_differences.png)
- [800 幀支持與各自成本](../outputs/run/track00_test_2611042012/analysis/09_range_800_support.png)
- [相同 sample 的共同評分](../outputs/run/track00_test_2611042012/analysis/10_range_800_common_cost.png)
- [800 幀實際地圖／道路對照](../outputs/run/track00_test_2611042012/analysis/11_range_800_map_paths.png)
- [對照數值](../outputs/run/track00_test_2611042012/range_comparison_800_summary.json)／[共同評分](../outputs/run/track00_test_2611042012/fixed_800_common_quality.json)

## 限制與交付狀態

目前版本 `track2_v510_verified_recovery_v5_16_perf2` 是撤回 v5.15 失敗改動的回復修正；147 個共同非 metadata 參數值與 v5.10 相同。`FEATURE_MAX_RANGE_M=40` 使用當幀 ground scale seed 估計三維距離，地圖裁切半範圍仍為 `(60,60,20) m`、對應距離為 `3／1.5／0.75 m`、地面水平半徑為 `18 m`。

最終目標仍是 `0.30488`，尚未達成；本輪沒有新 Public Score，也不將殘差、接受率、輸出差異或圖上的接近程度當成真實精度。v5.16 不推薦為相同數值策略再跑 full 或重複提交，可沿用已評分 v5.10 原始結果及其來源證據。後續精度改進仍需另外提出有依據、可驗證的模型改動。

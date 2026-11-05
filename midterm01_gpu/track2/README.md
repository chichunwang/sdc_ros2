# Track2：v5.16.4 目前最佳已知，已恢復最佳基線

最終目標：正式評分達到使用者指定參照 [`result_2_0.30488.csv`](<../doc/analysis doc/result_2_0.30488.csv>) 的 **0.30488**；目前尚未達成。參照檔作目標紀錄與唯讀對照，不作量測真值或軌跡回修依據。

**目前最佳已知：[v5.16.4 CSV](../outputs/run/track02_full_2611050123/result_2.csv)**，使用者回報 Public Score **0.42660**。CSV SHA 已核對，見[分數收據](../outputs/run/track00_test_2611051137/score_receipt.json)。比 v5.10 的 `0.42699` 低 `0.00039`；差距小，不作顯著性聲稱。完整 3299 幀、來源／輸出／格式與 9 組 full 圖核對完成。[full 圖表](../outputs/run/track02_full_2611050123/analysis/README.md) · [98｜模型對照](../doc/98_TRACK2_V5164_ASSOCIATION_ABLATION_20261105.md)。

v5.16.5 使用者回報 Public Score **0.43213**，比 v5.16.4 高 **0.00553**；停止推薦原樣重算或重複提交。[分數收據](../outputs/run/track00_test_2611051137/score_receipt.json)。它的 full CSV、模型與診斷保留為候選歷史，不作目前推薦版本：[full CSV](../outputs/run/track02_full_2611050200/result_2.csv) · [full 圖表](../outputs/run/track02_full_2611050200/analysis/README.md) · [99｜候選模型與檢查](../doc/99_TRACK2_V5165_DATA_HEIGHT_20261105.md)。

live `VERSION` 已還原為 `track2_v510_turn_patch_v5_16_4_perf2`；`V5165_ENABLED=False`，v5.16.2／v5.16.3 分支啟用、v5.16.1 分支停用，**40 m 與原配對不變**。恢復檢查已完成：完整 suite **329 passed／52.72 秒／1 項既有 Axes3D warning**；另 4 幀生成 CUDA 與 v5.16.4 frozen core 的 pose／scale／height／covariance 數值精確相同（1.82 秒），來源 SHA guard 通過。首輪 324 passed／1 failed 是 metadata fixture 漏考慮高度開關；將高度開關兩值和原 profiles 交叉檢查後通過，數值核心未改。詳見[101｜v5.16.4 最佳分數與高度回退](../doc/101_TRACK2_V5164_BEST_HEIGHT_REGRESSION_20261105.md)。

使用者目標 **0.30488 尚未達成**。最新研究見 [100｜高度與 pose 資訊](../doc/100_TRACK2_HEIGHT_POSE_INFORMATION_20261105.md)：在獨立高度假設下可保留部分弱約束，但真實歷史高度與 correction 相關，必須先建立聯合 covariance；目前沒有套用新版本。

## v5.16.1／3 正式分數與後續對照

使用者回報 **v5.16.1 0.44629、v5.16.3 0.44657**，比 v5.10 0.42699 分別高 0.01930／0.01958，均退步。兩版完整來源、格式與圖表核對通過，仍沒有預測正式退步；停止推薦重複提交。[分數及 CSV SHA](../outputs/run/track00_test_2611050118/score_receipt.json) · [97｜正式結果與原因假說](../doc/97_TRACK2_V5163_ROAD_PATCH_20261105.md)。

歷史 v5.16.3 full [`track02_full_2611050025`](../outputs/run/track02_full_2611050025/test_info.json) 完成 3299 幀，定位 2331.83 秒，9 組 full 圖已檢視；完整 suite 317 passed。自己的影像回饋下下一筆 GNSS forecast XY RMSE 0.0869043→0.0804418 m，正式卻退步；該數字是預測輸入 GNSS，不是 pose 真值。固定 19 幀共同地圖成本也有升有降。[完整圖表](../outputs/run/track02_full_2611050025/analysis/README.md)。

v5.16.4 只去除兩版共用的表面 fallback，測試它對因果轉彎與融合的影響；相近榜分本身不能證明這個分支有錯。不使用目標 CSV 擬合、未來影格或離線回修。

## v5.16.2 full 實際前綴（歷史）

兩版首筆 map update 都在 idx 4；control → candidate map 更新數為 800 幀 `340→360`、1600 幀 `971→975`。候選 rest gyro-bias 更新 17 次。idx 616 候選額外接受 road-PnP image speed `5.237764 m/s`（該筆前 sensor speed `7.799148 m/s`），XY update 即時拉回約 `0.234 m`，候選與 control 最大 XY 差為 `0.370638 m`。這是 map／scale validation 改變後讓 legacy road-PnP 多接受一筆，不是 coordinated-turn 固定 prior replay 本身出錯；固定 image replay 無法預測這條分支。

| Full prefix | XY next-GNSS forecast RMSE control → candidate (m) |
| ---: | ---: |
| 800 | 0.0849645 → 0.1076721 |
| 1600 | 0.0920080 → 0.0986620 |

1600 幀 left RMSE 為 `0.0405674→0.0280047 m`，forward 為 `0.0825819→0.0946041 m`。這些是下一筆 GNSS 觀測預測，不是 pose 真值；800 幀 XY 與 1600 幀 XY／forward 顯示退步，不代表整體改善。doc96 的舊四窗表為 sensor-only replay，不能當成這次含每版 image／height feedback 的實際前綴結果。預測數據與因果稽核見 [96](../doc/96_TRACK2_V5162_CAUSAL_TURN_20261104.md)。

idx 616 的 branch 稽核與逐版本 prior replay 顯示差異源於 map／scale validation 後多接受一筆 legacy road-PnP 影像速度；候選與 control 的 XY 最大差為 `0.370638 m`。實際前綴 forecast 與原因證據見 [800 幀 forecast](../outputs/run/track00_test_2611042251/actual_forecast_800.json)、[1600 幀 forecast](../outputs/run/track00_test_2611042251/actual_forecast_1600.json)、[prefix audit log](../outputs/run/track00_test_2611042251/audit_actual_prior_prefix.log) 及 [逐版本 prior replay](../outputs/run/track00_test_2611042251/replay_actual_full_prior.py)。後續相同特徵的 free-PnP／gyro fixed／joint solver 已完成，速度仍約 5.24–5.38 m/s，未解決分歧，因此沒有套用換 solver；路面視窗證據見 [97](../doc/97_TRACK2_V5163_ROAD_PATCH_20261105.md)。

v5.16.1 full `track02_full_2611042153` 已完成 3299 幀，沒有新榜分；完整比較與後續未採用研究見 [95](../doc/95_TRACK2_V5161_BILATERAL_SURFACE_20261104.md)。v5.15 full `track02_full_2611041916` 的使用者回報分數為 `0.51729`；[94](../doc/94_TRACK2_V515_REGRESSION_RANGE_20261104.md) 保留該版本的實際診斷。

## v5.14 full 與影片診斷（歷史）

v5.14 full `track02_full_2611041506` 完成 3299 幀，使用者回報 Public Score `0.43937`。此分數比 v5.13 `0.42933` 高 `0.01004`、比 v5.10 `0.42699` 高 `0.01238`；在當時 v5.10 是最低已知完整成績，最新最佳已知成績見上方。來源 129 個 Python SHA 與本次 run 快照相符，map、輸入與 frame manifest 和 v5.10／v5.13 相同。

本次使用者影片 [`2026-11-04 15-06-59.mov`](</home/ricky/2026-11-04 15-06-59.mov>) 的 RViz 標題對應 v5.14 full；灰線是使用者確認的兩條 map 道路線，橘線是估計軌跡。錄影長 2 小時 35 分 1.77 秒，有效定位約前 34 分 45 秒，之後為掛機跳過。細節分析見[91｜量測契約與 full 摘要](../doc/91_TRACK2_V514_RAY_SPEED_20261104.md)、[92｜影片偏離診斷](../doc/92_TRACK2_V514_VIDEO_DEVIATION_20261104.md)及[診斷索引](../outputs/run/track00_test_2611041808/analysis/README.md)。舊 Downloads 影片是 v5.12.1 full，請與本次影片區分。

v5.14 完整 CUDA suite 為 **279 passed、1 項既有 Axes3D warning、46.32 秒**；新增 9 個案例，含 CUDA depth→feature-ray→relative-fit→scalar-fusion 整合。測試圖表與機制限制見[測試分析](../outputs/run/track00_test_2611041445/analysis/README.md)。suite 驗證機制，不代表真實定位精度。

v5.13 候選凍結時的 **270 passed、41.93 秒、1 項既有 Axes3D warning** 是候選階段紀錄；v5.13 後來才完成 full。當時機制說明仍見 [90｜v5.13 影像相對幾何](../doc/90_TRACK2_V513_IMAGE_RELATIVE_GEOMETRY_20261104.md)，候選分析圖見 [track00_test_2611041306](../outputs/run/track00_test_2611041306/analysis/README.md)。

**v5.12.1 歷史驗證：**最終安全 CPU subset 有 17 個新機制案例及原有 8 個 directional-memory 測試，25 passed（2.02 秒）；完整 CUDA suite 247 passed、1 項既有 Axes3D warning（32.34 秒）。凍結 v5.10 generated baseline 核對也通過；123 個 Python source SHA 已凍結。先前 GPU 測試攔到的前向加速回歸已修正且原測試保留。初始化共用 2D power 以避免按正交軸依序消耗時改變 innovation 方向，完整吸收後 tracking 才分軸更新。生成的 stationary cue 測試顯示可抑制首筆 GNSS 大跳，但純 GNSS、沒有獨立 image-rest cue 時仍可能慢收斂或小幅 overshoot。此次限幅沒有新增定位觀測，軌跡變平順不能當成精度證明；沒有獨立真值。逐幀舊輸出、機制修正與合成限制見[89｜v5.12.1 GNSS／map handoff 側向預算](../doc/89_TRACK2_V5121_V510_HANDOFF_LATERAL_20261103.md)、[v5.10 逐幀 motion 診斷](../outputs/run/track00_test_2611032253/analysis/track02_full_2611031941_motion.json)與[本輪 full 分析](../outputs/run/track02_full_2611032346/analysis/README.md)。v5.12.1 標記路段的逐幀診斷見[前輪分析](../outputs/run/track00_test_2611041207/analysis/README.md)。

frame 2 的歷史輸出說明 v5.11.1 handoff 限制為何沒消除 Y 字：v5.10 frame 2 總步長 `0.65119 m`、GNSS 貢獻側向 `−0.23007 m`，map update 為 0；v5.11.1 對首筆 map update 的 frame 1 限制已落在約 `0.01861 m` budget 內，但 frame 2 首筆 GNSS 仍是 `0.65251 m`，側向 `−0.23142 m`。被停止的 v5.12 prefix 也出現 `0.65248 m`／`−0.23127 m`。其他舊輸出還顯示 rebase 在 377／418、map 在 2154／2865 有側向步長超出各自記錄預算，以及 1432／2343 GNSS update 在 map update 為 0 時仍移動。這些是 estimator movement attribution，不能當成真實側向誤差。

## 歷史版本與分數

v5.11.1 的機制在使用者啟動的 full `track02_full_2611032114` 前 300 幀實際運行：startup 前兩幀各完成 12 個候選比較；tracking alternate retry 出現在 212 幀；map proposal 檢查 90 次、拒絕 5 次；186 個凍結來源 SHA 已核對。該 full 隨後自然完成 3299 幀，定位耗時 2522.718837578 秒。精確標記窗的 readonly 機制稽核顯示，13 個異常窗的 alternate geometry rescue 均為 0；穩定對照 800–1000 有一幀（997）被救回，1485–1515 為 0。這只表示舊 retry 沒在異常窗 rescue，不能推斷原因一定是 seed 太近。首筆 map handoff 實際步長 0.0186075116455 m，與當時預算 0.0186075116456 m 相符。v5.11.1 GPU suite 為 214 passed、44.80 秒；完整機制與限制見 [88](../doc/88_TRACK2_V512_MAP_LIKELIHOOD_20261103.md)，先前 suite 的紀錄見 [2103 run](../outputs/run/track00_test_2611032103/analysis/README.md)。

v5.11 full `track02_full_2611032051` 由使用者停止於 351 幀（idx 0–350），`score=null`；run metadata 的 `running` 狀態與 185 個 source SHA 均原樣保留（[run metadata](../outputs/run/track02_full_2611032051/test_info.json)）。v5.11.1 full `track02_full_2611032114` 已自然完成 3299 幀，定位耗時 2522.718837578 秒；來源 186 SHA 與最後 8 個輸出 SHA 已核對，未取得新分數。當時 v5.10 使用者回報 `0.42699` 是最低已知完整分數；最新最佳已知成績見上方。v5.13 後來回報 `0.42933`，分數較高、表現退步。這些版本差異不能單獨證明算法因果。

v5.16.2 full 與 v5.16.3 full 都已完成。現有 CSV 可直接測試，不需原樣重算。固定命令列於下方「執行狀態與命令」。參數位於 [`parameters.py`](parameters.py)；v5.11 startup 多候選、tracking alternate retry、有限 gain 驗收與 prior-only 啟動後首筆 map update 限制的歷史見[87](../doc/87_TRACK2_V511_STARTUP_VALIDATION_20261103.md)；v5.10 的 course／heading 預算見[86](../doc/86_TRACK2_V510_V57_HEADING_20261103.md)，v5.12 map likelihood 歷史見[88](../doc/88_TRACK2_V512_MAP_LIKELIHOOD_20261103.md)。

下方 v5.3–v5.5 參數和機制保留作歷史說明。v5.11.1 從 v5.10 延續 v5.7 的 EPNP 影像速度、固定 40 m feature range、MAP height 與 visual update；不帶入 v5.8／v5.9 的 IPPE／逐影格 LK、多 range、data-only height 或 gyro 約束路面平移。

## perf2 歷史資料處理驗證

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
| v5.10 + perf2 [`track02_full_2611031941`](../outputs/run/track02_full_2611031941/test_info.json) | 0.42699 | 使用者回報；3299 幀 full 已完成；v5.11 基準，v5.12.1 回到此凍結來源 |
| v5.11 + perf2 | — | 首版 full 由使用者停止於 351 幀；score null；不算完整成績 |
| v5.11.1 + perf2 [`track02_full_2611032114`](../outputs/run/track02_full_2611032114/test_info.json) | 0.43685 | 使用者回報；3299 幀 full 完成；比 v5.10 差 |
| v5.12 + perf2 | — | 已套用；GPU suite 230 passed、1 項既有 warning／46.37 秒／121 Python SHA；生成 ablation 通過，未執行真實定位 |
| v5.12 full [`track02_full_2611032240`](../outputs/run/track02_full_2611032240/test_info.json) | — | 使用者停止時已有 395 幀輸出（idx 0–394）；沒有分數；metadata 原樣保存，程序已核對停止 |
| v5.12.1 + perf2 full [`track02_full_2611032346`](../outputs/run/track02_full_2611032346/test_info.json) | 0.42788 | 使用者回報；3299 幀完成，source manifest 193/193 SHA 核對通過；run metadata 的 score=null 原樣保留；比 v5.10 `0.42699` 高 `0.00089`（約 0.21%），略差；[full analysis](../outputs/run/track02_full_2611032346/analysis/README.md) · [source manifest](../outputs/run/track02_full_2611032346/source_manifest.json) |
| v5.13 + perf2 full [`track02_full_2611041345`](../outputs/run/track02_full_2611041345/test_info.json) | 0.42933 | 使用者回報；3299 幀完成，比 v5.10／v5.12.1 退步；[影像診斷](../outputs/run/track02_full_2611041345/analysis/README.md) |
| v5.14 + perf2 full [`track02_full_2611041506`](../outputs/run/track02_full_2611041506/test_info.json) | 0.43937 | 使用者回報；3299 幀完成；比 v5.13 `0.42933` 高 `0.01004`、比 v5.10 `0.42699` 高 `0.01238`；目前逐段檢查影片與 estimator 診斷，不能由分數單獨歸因；[doc91](../doc/91_TRACK2_V514_RAY_SPEED_20261104.md) · [doc92](../doc/92_TRACK2_V514_VIDEO_DEVIATION_20261104.md) · [診斷索引](../outputs/run/track00_test_2611041808/analysis/README.md) |
| v5.15 + perf2 full [`track02_full_2611041916`](../outputs/run/track02_full_2611041916/test_info.json) | 0.51729 | 使用者回報；3299 幀完成，已確認退步並回退 v5.10；292 項生成檢查曾通過但未預測退步；[實際診斷與距離實測](../doc/94_TRACK2_V515_REGRESSION_RANGE_20261104.md) |
| v5.16 已驗證回復版 | — | 恢復 v5.10 數值模型；40 m control 前800幀重現原結果；294項生成檢查通過；30 m 對照未採用，沒有新榜分 |
| v5.16.1 full [`track02_full_2611042153`](../outputs/run/track02_full_2611042153/test_info.json) | 0.44629 | 使用者回報；比 v5.10 差，停止推薦重複提交。[95](../doc/95_TRACK2_V5161_BILATERAL_SURFACE_20261104.md) |
| v5.16.2 full [`track02_full_2611042335`](../outputs/run/track02_full_2611042335/test_info.json) | — | 已完成3299幀；211/211 frozen source SHA核對通過；實際XY forecast略差，完整結果見96，無新榜分；模型說明見 [96](../doc/96_TRACK2_V5162_CAUSAL_TURN_20261104.md) |
| v5.16.3 full [`track02_full_2611050025`](../outputs/run/track02_full_2611050025/test_info.json) | 0.44657 | 使用者回報；比 v5.10 差。3299 幀與 SHA／圖表核對通過，停止推薦重複提交。[97](../doc/97_TRACK2_V5163_ROAD_PATCH_20261105.md) |
| v5.16.4 full [`track02_full_2611050123`](../outputs/run/track02_full_2611050123/test_info.json) | 0.42660 | 使用者回報；CSV SHA 核對完成；比 v5.10 低 0.00039，差距小，不作顯著性聲稱；目前最佳已知。[98](../doc/98_TRACK2_V5164_ASSOCIATION_ABLATION_20261105.md) · [score receipt](../outputs/run/track00_test_2611051137/score_receipt.json) |
| v5.16.5 full [`track02_full_2611050200`](../outputs/run/track02_full_2611050200/test_info.json) | 0.43213 | 使用者回報；CSV SHA 核對完成；比 v5.16.4 高 0.00553，停止推薦原樣重算／重複提交。[99](../doc/99_TRACK2_V5165_DATA_HEIGHT_20261105.md) · [score receipt](../outputs/run/track00_test_2611051137/score_receipt.json) |
| 使用者目標值 | 0.30488 | 目標 CSV `doc/analysis doc/result_2_0.30488.csv` 僅作 hash／provenance 紀錄；核對見 [target reference](../outputs/run/track00_test_2611021349/target_reference.json) |

所有 full 分數都由使用者回報，來自不同版本，不應歸因於 perf2。使用者目標值為 `0.30488`；參照 CSV 用於同索引差異診斷，不作逐幀真值或回修 pose 的依據。no-accel 前五幀真實 A/B 與五個生成場景輸出一致，但不能推論整段相同。

參照 CSV 是非因果輸出，只作同索引差異診斷；不從榜分反推隱藏真值、不擬合軌跡轉換或時間差，也不保證任何版本能達到目標 `0.30488`。五個 full 共用相同 input SHA 和 frame-manifest SHA，詳細結果／輸出 hash／分數來源見[版本證據](../outputs/run/track00_test_2611021839/version_evidence.json)。

v5.6 的 map／heading 基線與方向記憶見 [82](../doc/82_TRACK2_V56_V52_BASELINE_20261102.md)；v5.7 在此基礎上新增前視影像地面速度，見 [83](../doc/83_TRACK2_V57_IMAGE_GROUND_SPEED_20261102.md)。v5.8、v5.9 為後續實驗版本；v5.10 恢復到 v5.7 影像速度、尺度配準與 visual update，並加入 course／heading 更新檢查。v5.11／v5.11.1 及 v5.12 的程式歷史與合成測試仍保留；歷史 v5.12.1 回到 v5.10 基線，停用 v5.11／v5.12 分支，改處理 GNSS delayed-position、側向 map budget 與 reference sensitivity 的有限 gain。v5.13–v5.15 的 full 與診斷保留作歷史。歷史 v5.16.2 在 v5.16.1 表面 fallback 上增加因果 coordinated-turn motion prior，詳見 [95](../doc/95_TRACK2_V5161_BILATERAL_SURFACE_20261104.md) 與 [96](../doc/96_TRACK2_V5162_CAUSAL_TURN_20261104.md)。前序模型說明見 [86](../doc/86_TRACK2_V510_V57_HEADING_20261103.md)、[87](../doc/87_TRACK2_V511_STARTUP_VALIDATION_20261103.md)、[88](../doc/88_TRACK2_V512_MAP_LIKELIHOOD_20261103.md)、[89](../doc/89_TRACK2_V5121_V510_HANDOFF_LATERAL_20261103.md)及[90](../doc/90_TRACK2_V513_IMAGE_RELATIVE_GEOMETRY_20261104.md)。下方 fix1／fix2 章節只記錄更早的模型沿革。

### fix1 的三項模型修正

1. GNSS XY CV filter innovation 繼續推進當幀 sensor prior 與 diagnostics，但不再作 calibration-reference delta 去抵消舊 map memory；校正重定錨只保留 heading calibration 及其 lever-arm 影響。
2. correction-memory 用 prior-Cholesky 白化座標下的 PSD 方向年齡矩陣管理 partial-rank map 記憶；只有實際觀測 row-space 刷新相應模式，其他弱方向超過原 hold 後按原 tau 衰退。
3. map-only likelihood center 除原 bounds 外，須在同一組 final planes 的實際非線性 data loss 不增加才接受；尺度 nuisance 仍做 Schur marginalization，條件 log-scale step 保留，拒絕不靠 blend prior 修補。

## 執行狀態與命令

live source 已還原至 v5.16.4 版本；v5.16.2／3 分支啟用，v5.16.1／5 分支停用，40 m 與原配對不變。生成／CUDA 恢復檢查已完成，摘要和差異見上方 [doc101](../doc/101_TRACK2_V5164_BEST_HEIGHT_REGRESSION_20261105.md)。下列入口保留給使用者需要建立新 full 時使用；實際執行狀態以新 run metadata 為準。Kaggle／E3 由使用者提交。

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
./scripts/run_track2.sh --full
```

此為固定操作入口。執行參數可直接在 [`parameters.py`](parameters.py) 調整；目前 v5.16.3 完整結果見 [97](../doc/97_TRACK2_V5163_ROAD_PATCH_20261105.md)。v5.16.2 模型與實際回饋診斷見 [96](../doc/96_TRACK2_V5162_CAUSAL_TURN_20261104.md)，v5.16.1 的 full 對照與機制限制見 [95](../doc/95_TRACK2_V5161_BILATERAL_SURFACE_20261104.md)。

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

直接編輯 [`parameters.py`](parameters.py)。目前為 v5.16.5 高度量測一次融合，保留 v5.16.2 coordinated-turn prior、v5.16.3 路面視窗與 **40 m** feature range；`V5161_ENABLED=False`、`V5162_ENABLED=True`、`V5163_ENABLED=True`、`V5165_ENABLED=True`。gyro rest-noise 與 random-walk engineering budgets 見 [96](../doc/96_TRACK2_V5162_CAUSAL_TURN_20261104.md)，路面視窗修正見 [97](../doc/97_TRACK2_V5163_ROAD_PATCH_20261105.md)。下表保留舊分支設定供追溯，不表示所有列出的歷史旗標都是現行啟用值；目前 branch selectors 的實際值以 `parameters.py` 為準。GNSS heading observation `3°` 未校準，沒有用非因果 CSV 擬合參數。

| 要調整的內容 | 參數 | 目前設定 |
| --- | --- | ---: |
| 現行模型組成與範圍 | `FEATURE_MAX_RANGE_M` | `40 m`；v5.10 原配對、v5.16.5 高度量測、v5.16.3 路面視窗與 v5.16.2 coordinated-turn prior |
| v5.13 歷史影像高度／深度 anchor 記憶 | `IMAGE_ANCHOR_MEMORY_ENABLED` | 保留 `True`，總開關關閉，當前不生效 |
| v5.13 歷史相對平移與相對 yaw channel | `IMAGE_RELATIVE_TRANSLATION_ENABLED`、`IMAGE_RELATIVE_YAW_ENABLED` | 保留 `True`、`True`，當前不生效 |
| v5.13 歷史相對平移 solver | `IMAGE_RELATIVE_SOLVER` | 保留 `gyro_translation`，當前不生效 |
| v5.13 歷史角度分層 map sample cap | `MAP_ANGULAR_STRATIFIED_CAP` | 保留 `True`，當前不生效；v5.15 沿用 v5.10 原抽樣及 6000 點上限 |
| v5.13 歷史 map 法向與 solver 診斷 | `MAP_GEOMETRY_DIAGNOSTICS` | 保留 `True`，當前不生效 |
| v5.12.1 歷史版本開關 | `V511_ENABLED`、`V512_ENABLED`、`V5121_ENABLED` | 當時為 `False`、`False`、`True` |
| v5.12.1 歷史 GNSS position 分幀 likelihood 預算 | `GNSS_POSITION_INFORMATION_BUDGET` | 當時為 `True`；v5.13 總開關關閉 |
| v5.12.1 歷史 map body-lateral 預算 | `MAP_BODY_LATERAL_BUDGET` | 當時為 `True`；v5.13 總開關關閉 |
| v5.12.1 歷史 reference sensitivity 限制 | `MAP_BOUNDED_REFERENCE_RESPONSE` | 當時為 `True`；v5.13 總開關關閉 |
| 開啟／關閉 GNSS quaternion 航向配對 | `GNSS_HEADING_ENABLED` | `True` |
| IMU 初始 heading bias 標準差 | `HEADING_INITIAL_BIAS_STD_DEG` | `10°` |
| GNSS 初始 heading bias 標準差 | `GNSS_HEADING_BIAS_STD_DEG` | `10°` |
| GNSS quaternion 觀測標準差（另加 IMU covariance） | `GNSS_HEADING_OBSERVATION_STD_DEG` | `3°` |
| 兩種 heading bias random walk | `HEADING_BIAS_RW_DEG_SQRT_S`、`GNSS_HEADING_BIAS_RW_DEG_SQRT_S` | 各 `0.12°/√s` |
| 過去 GNSS course 視窗與方向不確定度門檻 | `HEADING_MIN_WINDOW_S`、`HEADING_MAX_WINDOW_S`、`HEADING_MAX_COURSE_STD_DEG` | `1–8 s`、`3°` |
| v5.10 course 半段一致性門檻 | `HEADING_MIN_SAMPLES`、`CORRECTION_NIS_REJECT_BY_DOF[0]` | 每半至少 `5` 筆；1 DoF NIS `10.828` |
| v5.10 同影格 heading bias 更新預算 | `HEADING_MAX_UPDATE_DEG` | 合計最多 `0.5°`；首幀 bootstrap 除外 |
| 保留 v5.10 course／heading 保護 | `V510_ENABLED` | `True` |
| v5.11 historical 開關與比較基準 | `V511_ENABLED`、`BASELINE_RUN_ID` | 候選關閉；歷史 run `track02_full_2611031941` |
| v5.11 startup 高度 seeds | `STARTUP_HEIGHT_SEED_SIGMAS` | `(-1, 0, 1)` 個 prior 標準差 |
| v5.11 yaw seeds | `STARTUP_INCLUDE_GIVEN_YAW`、`STARTUP_INCLUDE_MAP_AXIS` | `True`；`True` |
| v5.11 tracking alternate retry | `TRACKING_RETRY_ON_GEOMETRY_FAILURE` | `True` |
| v5.11 map proposal data-cost 驗收 | `VALIDATE_EMITTED_MAP_PROPOSAL` | `True` |
| v5.11.1 prior-only 後首筆 map gain 限制 | `STARTUP_BOUND_AFTER_FIRST_FRAME` | `True` |
| v5.12 歷史 data-only map likelihood 分支 | `MAP_DATA_LIKELIHOOD_ENABLED` | 目前停用；失敗的 v5.15 歷史分支曾啟用此機制 |
| v5.12 歷史 fixed-plane 與 rematched posterior 分支 | `MAP_REMATCHED_POSTERIOR_CHECK` | 目前停用；失敗的 v5.15 歷史分支曾啟用雙重驗收及 gain 回溯 |
| v5.12 版本開關（依賴 v5.11／v5.2 fusion） | `V512_ENABLED` | 候選 `False`（historical v5.12 為 `True`） |
| v5.12 historical 幾何支持下的非收斂 yaw recovery | `TRACKING_RETRY_AFTER_NONCONVERGENCE` | 候選停用 |
| v5.12 historical 非收斂 yaw seed 偏移來源 | `MAP_STARTUP_YAW_SEED_SIGMAS` | v5.12 歷史 seed 設定保留；候選總開關停用 |
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

doc76 的資料是原 v5.3（fix1 前）baseline，提出的 rebase／directional expiry／likelihood acceptance 風險已對應到 live fix1 的三項調整。兩個生成場景各有 322 state steps：加速場景的 moving-x RMSE 變化是「GNSS XY innovation 不再抵消舊記憶」與「方向性衰退」合併效果；另一個只觀測 y/yaw 的場景直接檢查方向性 expiry，未觀測 x correction 由 `0.376470588 m` 衰退至 `0.060190022 m`。這些是生成機制結果，不是 Track2 精度或 `.80300` 的因果拆解。該歷史 full 當時由使用者啟動。

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

# Track2 v5.16.5：高度量測只融合一次

**下一步可測：[v5.16.5 full CSV](../outputs/run/track02_full_2611050200/result_2.csv)**。本版 full 已正常完成 3299 幀（exit 0）；正式分數未知，使用者指定的 **0.30488** 目標尚未達成。v5.16.4 的 [CSV](../outputs/run/track02_full_2611050123/result_2.csv) 也保留供測試，尚無正式分數。[v5.16.5 full 圖表](../outputs/run/track02_full_2611050200/analysis/README.md)。

目前程式是 **v5.16.5**：保留 v5.10 原配對、因果轉彎、路面視窗與既有半徑設定，只把過去高度 anchor 改為由純地圖 H／g 得到高度 likelihood，再融合一次。這是避免重複使用 prior 的資料模型修正，不是精度改善結論。完整 suite **325 passed／49.78 秒**，metadata 後續修正另有 4 個生成端到端 profile 通過；22 個固定實際影格 H／g 重建、6 筆可用高度非線性驗收和 4 幀關閉分支精確 CUDA 對照通過。細節與限制見下文。

本版在 v5.16.4 full 結束並核對執行來源後才套用：前一版結束後曾核對 136 個 live Python SHA，GPU 釋放後才更新來源；137 個新 live Python 與最後暫存來源相符。這是套用順序的歷史記錄，不改寫既有 run 證據。

## 觀察與原因假說

v5.10 原高度更新先在配準 objective 中使用高度 prior，求出 MAP（最大後驗）高度，再把該高度視為獨立量測融合回同一個 prior。這重複使用了 prior；生成檢查通過及地圖殘差降低都無法保證它統計正確，更不能當正式精度。

對保存的 H／g 做局部診斷：v5.16.1 的 prior 到 MAP 高度差中位數 0.00925 m，但純地圖資料的高度中心差為 0.02220 m；v5.16.3 對應 0.00930／0.02260 m。這表示兩個量測中心不等價；單步候選 anchor 差最大約 0.01766 m。此診斷尚未通過原點雲上的非線性驗收，也沒有定位真值。

[H／g 診斷](../outputs/run/track00_test_2611050118/height_reuse_audit.json) · [圖表](../outputs/run/track00_test_2611050118/analysis/03_height_prior_reuse.png)。歷史 data-only 高度曾和其他變更一起出現在舊模型；本輪只隔離原 v5.10 配對下的高度更新，不重跑舊失敗整版。

## 為什麼重複 prior 會改變答案

用一個生成示例說明，不代表實際誤差：prior 高度 1.50 m、標準差 0.05 m；獨立地圖量測 1.65 m、標準差 0.10 m。正確的一次融合 gain 為 0.2，高度為 **1.530 m**。若先得到 1.530 m 的 MAP 高度，再把它當成地圖量測與原 prior 融合，高度只到 **1.506 m**，仍回報縮小的 covariance。這不是新增量測，而是把 prior 再放進中心一次。

同一來源平面距離下，尺度等於高度除以 raw 平面距離；高度偏移也會改變整個點雲的公尺幾何。這說明值得單獨驗證此修正，不能據示例預測正式分數。

## 模型與輸入步驟

配準 pose、配對、40 m、原品質門檻、因果轉彎與路面視窗沿用 v5.16.4。只改「過去高度 anchor」的更新，使用當幀地圖資料的 Hessian（局部資訊矩陣）和 gradient（殘差梯度），不包含 solver 加入的 sensor／height prior。

令 \(u=[x,y,yaw]\)、\(s=\log scale\)，在當幀 candidate 邊際化 pose：

\[
I_s=H_{ss}-H_{su}H_{uu}^{\dagger}H_{us},\qquad
\gamma_s=g_s-H_{su}H_{uu}^{\dagger}g_u.
\]

只在有正資訊時得到局部公尺高度量測：

\[
z_h=h_c(1-\gamma_s/I_s),\qquad
R_h=h_c^2/I_s+\sigma_{floor}^2.
\]

先按既有物理／搜尋界限，在原始同一組 final planes 上驗證非線性 data loss 沒有增加，再用既有高度 filter 的 innovation 檢查及 Joseph covariance 一次融合。不放寬門檻、不加入新調參。沒有可用的高度量測就保留 prior 及其不確定性。

當前輸出 pose／scale 的選擇規則維持原設定；新高度 anchor 影響後續影格。`scale_validated` 與 `height_update.accepted` 分開記錄，前者不代表新高度量測通過。本版僅在原地圖配對 profile 啟用，`V5161_ENABLED=False`。

## 檢查與狀態

獨立物理高度線性模型已 **6 項 CPU 檢查通過，1.94 秒**：改變 prior mean／variance，從純資料 H／g 恢復的高度量測不變，一次融合的均值及方差與解析 Bayesian 解相符。此證明只適用生成模型與數值契約，不代表真實精度。

當時排程的檢查已完成：序列執行暫存版完整 CPU／CUDA suite、4 幀生成 CUDA 關閉分支精確對照，以及固定實際影格的原 H／g 重建和同平面非線性高度驗收。沒有並行爭用 GPU，也沒有更換執行中來源。其後已完成 v5.16.5 full；結果摘要與密集對照數值列於文末附錄。

[本輪紀錄](../outputs/run/track00_test_2611050139/test_info.json) · [暫存來源 SHA](../outputs/run/track00_test_2611050139/staged_files.json) · [背景檢查狀態](../outputs/run/track00_test_2611050139/stage_checks_status.json)。檢查完成後才記錄已實作／真實定位／正式分數，不預填成功結論。

## 最後機制與來源檢查

完整 Track2 CPU／CUDA suite **325 passed、49.78 秒**，1 項既有 Axes3D warning。另 4 個生成端到端 profile 檢查（4.77 秒）確認 metadata 正確描述高度啟用／停用與 excluded 項目；後續只改了 runner metadata 及其 assertion，數值核心沒有改動。

固定 **22 個實際影格**重建原 v5.16.4 H／g 並核對原紀錄，其中 6 筆符合原 pose／scale 更新條件，**6 筆全通過**同平面非線性高度驗收。4 幀生成 CUDA 在關閉新分支後，pose／scale／高度／covariance 與凍結 v5.16.4 精確相同，舊診斷欄位集合也保持相同。這些是機制檢查，不是額外真實定位幀或精度證明。

首輪 staged suite 因覆蓋 PYTHONPATH 遺失 ROS 路徑，collection 失敗；保留 inherited ROS 路徑後，324 passed／1 failed 攔到停用分支多出空 height 診斷欄位。改為只在啟用時輸出新欄位後，最後整組通過，失敗來源與 log 保留。部分 host exec bridge 回傳143，但完整 pytest XML／log 已結束；最後檢查改由容器內 supervisor 持 shared GPU lock 到所有 child 結束，三項最後工作 exit 0。

[最後 suite](../outputs/run/track00_test_2611050139/pytest_stage_final.xml) · [最後 CUDA 高度驗收](../outputs/run/track00_test_2611050139/height_fixed_cuda_final.json) · [關閉分支精確對照](../outputs/run/track00_test_2611050139/disabled_height_cuda_final.json) · [套用收據](../outputs/run/track00_test_2611050139/apply_receipt.json) · [metadata 後續收據](../outputs/run/track00_test_2611050139/metadata_followup_apply_receipt.json)。

## Full 輸出與對照數值附錄

v5.16.5 full [`track02_full_2611050200`](../outputs/run/track02_full_2611050200/test_info.json) 正常完成 3299 幀，exit 0；定位耗時 **1544.039190204 秒**、registration 耗時 **1448.730864716 秒**、map 接受 **1614 幀**。來源 **217/217**、live **137/137**、輸出 **6/6**、analysis 來源 **7/7** 核對均無失配。CSV SHA256 為 `1799ca2707ea71342a5b40daea0582220188fd318826720b3d98821b9dd9d847`。full 的 **9 組圖**與研究用的 **12 組圖**已由主 agent 檢視；圖表清單與來源見 [run analysis](../outputs/run/track02_full_2611050200/analysis/README.md)。正式分數未知，不能宣稱已達成 `0.30488`。

本版只改純地圖 H／g 高度 likelihood 的一次融合。full 中 **1321/1321** 筆高度更新通過同平面非線性驗收；control 有 **1455** 次舊 MAP 高度更新。兩版高度 anchor 最大差 **0.1714814668 m**，這是模型輸出差異，不是誤差或精度證明。

和 v5.16.4 比較時，共同的 **19 個固定 raw samples** 上，本版 data cost 有 14 幀較低、5 幀較高，記錄的平均差為 **−0.00071844695**；surface cost 有 12 幀較低、7 幀較高，平均差為 **−0.00046812585**。固定成本升降不一致，不能用來推斷正式精度。所有 XY prior 相同；兩版 473 筆對應 GNSS forecast 逐筆相同，整體 XY RMSE 均為 **0.08044178925 m**。

兩版輸出 XY 的中位差為 **0.04428049 m**、p95 **0.20191809 m**、最大 **0.40394451 m**（idx 916）；最大 yaw 差 **0.501871°**。這些是版本間輸出差異，不是定位誤差。idx **2643–2671** 與 **2773–2864** 兩段的 map update 仍為 0。

來源範圍研究結果保存在 [ground support probe](../outputs/run/track00_test_2611050213/ground_support_probe.json)。14 個固定幀中，將 source ground 限制於 **18 m** 時 cost 為 5 幀下降、7 幀上升、2 幀相同；在 source-ground 18 m 的基礎上，再把 map ground 限制於相機可見區時，4 幀下降、10 幀上升。兩者都沒有一致優勢，因此沒有採用新裁切。現行 **40 m** 是非地面 feature 半徑；map ground 使用水平 **18 m** 範圍；source ground 目前不做距離裁切。

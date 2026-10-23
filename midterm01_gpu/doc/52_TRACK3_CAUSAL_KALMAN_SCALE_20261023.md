# Track3 因果 Kalman 與逐幀尺度配準（2026-10-23）

## v2 更新：合成檢查通過，真實定位尚未驗證

因 v1 full 的 map 更新採用 0 幀、速度為零且位置固定初值，入口已更新為初始化修正版 v2。v2 通過 59 項不同的 CPU／CUDA 機制檢查；帶噪傾斜合成場景與已知尺度測試、合成 3／5 幀 byte-prefix 一致及 zero-accepted 明確失敗均通過，詳見 [53｜初始化修正與驗證](53_TRACK3_INITIALIZATION_REPAIR_20261023.md)。這些是合成測試，不是真實 Track 定位結果。尚無真實 v2 定位或新分數，亦無精度改善證據；score <1.29 是目標而非保證。

目前完整命令由使用者手動執行，會跑現行 v2；雖與 v1 使用相同 CLI 入口文字，程式來源已變，不能用來重現 v1。v1 source snapshot 保留在舊 run 的 `pipeline_sources/`。只有 full 提交產生正式分數；短段可選作故障檢查，並非 full 資格門檻：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
bash scripts/dev.sh up
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 outputs/ml_runtime/venv/bin/python scripts/run_track3_causal.py --full --scale-mode estimate --initial-scale 3
```

`--initial-scale 3` 仍是未校準 seed；完整座標／物理尺度標定未知，合成 prefix 只證明序列化因果一致性。

## 先回答：配準是程式自動做，scale 是點雲尺寸倍率

你不需要逐幀手動配準、量出真實角度當標籤，或手動挑 map 對應點；之前手算的合成角度只是學習例子。程式會自動調整影像重建 PLY 的尺寸、旋轉和平移，讓 PLY 中的牆、地面、柱子等幾何盡量貼合 LiDAR map；這個程序就是配準。IMU gyro 幫忙預測短時間內轉了多少，不能單獨告訴車往前走了多遠，也不能取代 map 幾何定位。Kalman 把運動預測與通過檢查的 map 觀測合起來，map 也不是無誤差真值。初始速度設為零；若 map 觀測一直被拒絕，gyro 無法提供前進距離，因此這次輸出位置停在初始點。

scale (s) 是 PLY 三軸 XYZ 共用的尺寸倍率。若原始點為 `p_raw`，程式拿去配準的是 `s·p_raw`，再以旋轉 `R` 和平移 `t` 放進 map：`p_map = R(s·p_raw) + t`。例如 scale 3 表示把 PLY 原始座標的 2 個長度單位當成 6 個 map 長度單位；這不是已知真值，也不表示車的 global XY 或 map 要乘 3，更不是 IMU 係數或相機內參。舊 legacy 命令把 3 固定不變；新分支的 `--initial-scale 3 --scale-mode estimate` 只把 3 當起始值，程式嘗試估尺度，但必須通過幾何與 Kalman 檢查才會採用。

新因果分支已做一次 full；使用者文字回報 Team 3 score **82.17**，未獨立核對上傳 bytes。本機 [full run](../outputs/run/track03_full_2610231521/) 有 3142/3142 幀、輸出有限且 validation errors 為空，但追蹤失敗：map 最終採用 0 幀，3085 幀被標記 `tracking_lost`；速度全程為零，輸出 XYZ 每幀都等於初始位置 `(660.30826416, -222.547286987, 10.384306145)`。幾何檢查只通過一個候選 idx 715；該候選的 scale 約 7.96438，雖然 NIS 2.835 小於上限 24.322，仍因物理旋轉跳動 `physical_rotation_jump` 被 Kalman 拒絕。正式輸出 scale 全為 `3.0000000000000004`（顯示六位小數為 `3.000000`），因為沒有 map 更新被採用；這沒有證明 scale 3 正確或估尺度成功，也不能據 idx 715 改用 7.964。`optimizer_converged` 是 3141 次拒絕原因之一；每筆可同時有多個拒絕原因，單獨放寬門檻不能當成定位改善。CSV SHA256：`fa60b9da48a21320a42ec15a845f9f439eefa5f9de89bc69eba7686c5635ca5e`。

完整計算寫完輸出，不等於建立了有效追蹤。這個已知失敗的 v1 不建議原樣重跑；現行入口已更新為 v2，但還沒有真實定位結果。只有 full 提交會產生正式 Kaggle 分數；短段只用來檢查機制與故障，不是 full 資格門檻。新算法版本可用 full 成績比較；完全相同的已知失敗版本則沒有重跑價值。完整 run 仍由使用者手動執行。

## 狀態與證據範圍

這是依使用者授權新增的因果 Track3 分支，包含前向濾波、尺度／幾何配準，以及資料與 GPU 執行路徑。已有一次真實資料 full 嘗試與使用者回報分數；結果與界限見本文件開頭。**尚無已驗證的精度改善，也未取得正式因果合規認證。**原 legacy run 及其使用者回報分數 106.72 屬舊算法歷史，不能套用到此分支。

證據包括 [full run](../outputs/run/track03_full_2610231521/)、其唯讀 [診斷結果](../outputs/run/track03_test_2610231541/diagnosis.json)、主要研究紀錄 [track03_test_2610231450](../outputs/run/track03_test_2610231450/) 與真實輸入 CPU audit [track03_test_2610231503](../outputs/run/track03_test_2610231503/)。唯讀診斷核對來源 SHA，未執行定位、GPU 工作或軌跡修正。CPU audit 在前 100 幀末時間的 cutoff 前取到 358 筆 IMU，其中首幀可用 2 筆；初始矩陣按 source→map 慣例通過 proper rotation 檢查，重力方向 sanity 的垂直 cosine 中位數為 0.9924。這只驗證數值慣例與初始重力方向檢查，沒有驗證物理座標標定或定位精度。

先前版本的 39 項測試是 v1 階段紀錄，不能代表目前入口的 v2 驗證。更新的 59 項不同檢查與合成證據見 [53｜初始化修正與驗證](53_TRACK3_INITIALIZATION_REPAIR_20261023.md)；其中不包含真實 Track 定位。此處保留原 v1 設計、full 失敗與當時證據，不把 v2 狀態回填成 v1 成功。程式入口與模組見 [run_track3_causal.py](../scripts/run_track3_causal.py)、[causal_track3_filter.py](../scripts/causal_track3_filter.py)、[causal_scale_registration.py](../scripts/causal_scale_registration.py)、[export_track3_imu.py](../scripts/export_track3_imu.py)、[causal_map_cache.py](../scripts/causal_map_cache.py)。

## 狀態、觀測與假設

濾波使用 11 維誤差狀態：

| 區塊 | 維度 | 意義 |
| --- | ---: | --- |
| `p` | 3 | map 座標位置 |
| `v` | 3 | 常速度模型的速度 |
| δθ | 3 | 左乘的小旋轉誤差 |
| `b_g` | 1 | 垂直 gyro bias |
| `λ` | 1 | log 尺度 |

相鄰影格之間以 `p⁻ = p + v·dt` 前向預測；旋轉依方向四元數提供的 IMU 世界 +Z 軸 `u = R_qᵀ e_z` 投影 gyro，再扣除垂直 bias，按 `R⁻ = R_z((uᵀω − b_g)dt)R` 更新。這假設 orientation world +Z 與 map +Z 平行，且只採平面轉動近似。沒有假設已知相機／IMU 外參，不使用 IMU 絕對 yaw，也不把加速度雙重積分成位置；加速度只用在初始化重力 sanity check。

初始矩陣沿既有 source→map 慣例直接使用，不求逆、不另作軸轉換。這不等於座標完整標定：raw PLY 上游軸向、物理 pose 參考點及官方 sample 欄位語意仍未確認。IMU covariance 全零表示未知，而非零誤差；濾波器中的噪聲假設、速度 40 m/s、角速度 90°/s、map correction 3 m（初始化 5 m）、相鄰平移 slack 0.5 m、旋轉 slack 3°及尺度每幀 log step 0.15，都是明示的工程安全域，沒有實測校準，也不是按榜分挑選的最佳參數。

map 幾何配對的 point-to-plane 殘差為 `r = nᵀ(R(s·p_raw) + t − q)`，每個配對對 6-DOF pose 與 log scale 提供最多 7 個局部未知量。濾波的 map 殘差為位置差、`Log(R_obs Rᵀ)` 及 `log(s_obs/s)`。更新使用 Joseph covariance 形式。NIS、物理跳動限制、尺度域、幾何退化及 held-out 檢查任一拒絕時，保留當前向前預測並照常輸出該幀；若速度初始化為零且始終沒有 map 更新，位置預測仍會停在起點。超過 2 秒沒有 map 支持會標示 `tracking_lost`；尚未實作 global relocalization。

## 三項實作改動

1. **核對初始化／座標，建立 availability-gated IMU 與前向 Kalman。** 核對 PLY 檔名與前相機 bag 的整數時間前綴、初始矩陣及重力方向 sanity。IMU 只可在 bag availability timestamp 不晚於當前影格時使用，不作未來插值；物理外參缺項與平面近似明確記錄。
2. **加入因果運動預測與配準拒絕。** 位置採常速度預測，gyro 約束平面旋轉增量與 bias；先檢查 map 候選的物理跳動、NIS 與尺度創新，再檢查 Kalman 後驗，避免相關 gain 產生超界修正。拒絕 map 更新後仍發佈本幀預測並增長不確定度；持續沒有 map 支持會標示失追。
3. **加入當前影格的受約束尺度與 robust point-to-plane map 觀測。** 地面／立面分層，並在空間與 normal 支持上均衡配對；以訓練配對作受限 line search，held-out patch 不參與挑 seed 或更新步。檢查 XY／yaw 與尺度可觀測性、幾何退化及 held-out 殘差後才接受觀測。啟動尺度只在明示的工程數值範圍 0.5–16 內，用 7 個 log-spaced seed 加未校準初值 3 比較訓練 cost；這不是尺度校準或榜分搜尋。之後 logscale 每幀最多變動 ±0.15；遇退化時不更新尺度／pose。

CUDA 重用既有 kNN、PCA／normal 及配對路徑，PCA `eigh` 每批維持 256。原 map 只讀 XYZ（沒有 timestamp）；ASCII 以單次串流存成 float32 XYZ cache，約 0.69 GB，不減少點數。每個 tile 由當前預測附近 100 m half-cube 建立，GPU 以 0.3 m voxel 保留原始點中的首點；不宣稱與舊 PCL centroid 等價。SQLite／ASCII／磁碟 I/O、小型最多 7 變數 solver、11 state Kalman，以及觸發時的 kNN 距離同值／voxel 邊界裁決留在 CPU。CUDA 失敗不回退 CPU。map 準備、CUDA 暖機與逐幀時間分開記錄；本次真實 full 的 map 準備合計 46.562 秒，CUDA 暖機 0.195 秒另計，沒有把 map 準備總時間當成單獨 ASCII 解析耗時。新分支固定／逐幀尺度回放皆未整合，因此標示 `replay_supported=false`。

全流程不使用 RTS、batch graph、未來資料或舊 pose 回寫。固定逐幀 `Run:`／`Frame i/N`、9 欄 `timings.csv`、`Graph: executed=false, applied=false, time=0.00 ms` 及 `Finished`／`CSV`／`Validation` 格式保留；準備與事後圖表分開計時。新 full 成功後的唯讀圖表 hook 只寫入該 run 自己的 `analysis/`。

## 手動短段與 full 比較

以下三行是可選的前 100 幀機制檢查命令，不是正式計分，也不是 full 的先決資格。定位由使用者手動啟動：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
bash scripts/dev.sh up
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 outputs/ml_runtime/venv/bin/python scripts/run_track3_causal.py --frames 100 --scale-mode estimate --initial-scale 3
```

固定尺度消融由使用者另行手動執行相同前綴、將 `--scale-mode estimate` 改為 `--scale-mode fixed`。dry-run 只列印計畫；audit-only 執行輸入／座標檢查，不產生 pose。新算法或參數版本可以直接用 full Kaggle 分數比較；同一已知失敗版本不原樣重跑。拒絕率、殘差和平順度都不是精度證明。完整定位與提交仍由使用者手動啟動。

目前結果、欄位與限制以各 run 內的 `validation.json`、`coordinate_audit.json`、`observations.jsonl`、source SHA 和參數紀錄為準；不修改原 CSV、既有 legacy run 或歷史 score。

# 25｜三相機瓶頸分析與相同算法加速（2026-10-18）

後續已依使用者要求完成 [顯存換速度短測](26_GPU_MEMORY_RESULTS_20261018.md)：batch 2／3 與保留快取未測得明顯收益；PCA chunk 1024 的取樣約少 13.8 ms，選點相同，整幀收益估計約 2%。沒有更改執行中的完整設定，Track2 已恢復繼續。注意短測使用 native allocator，與完整 wrapper 的 expandable-segments 設定不同，報告保留此限制。

已完成獨立加速入口，不更換模型、解析度、深度尺度、同步方式或幾何取樣規則。四個真實場景的取樣階段暖機快 **6.5–7.6 倍**；另用真正的 MapAnything 推論重建 Track1 2483–2486，暖機後三幀平均 **0.780 s/frame**，同位置原版紀錄 **1.990 s/frame**，四幀 PLY／labels／取樣／相機 pose 完全相同。這是小段實測，尚無新完整序列耗時或 Kaggle 分數。

使用者已停止 `outputs/run/track00_full_2610181155`，Track1 已提交 **2487／3011 幀**，剩 **524** 幀。原 run 保留；已將核對過的幀複製到 **`outputs/run/track00_full_2610181301`**，可用加速入口接續。三相機尺度候選仍為 0.67398047，跨場景驗證未通過，效能改進不能代替精度驗證。

## 一幀 1.x 秒花在哪裡

讀取原 run 當時最近 200 個 `frame.json`（最後 idx 2098），重建平均 **1.526 s**，median **1.559 s**、P90 **1.715 s**。這是 raw 三相機重建階段，後續 ROS 定位尚未開始，不能把這個時間當作 ICP 耗時。

對真實 cached depth/confidence 反投影，四個場景共有 14.5–21.1 萬候選點；0.3 m voxel 後仍有 6574–21805 點需要幾何分類。原版對每一點在 Python loop 做 16-neighbor 查詢結果篩選、局部 mean／covariance、一次 3×3 `np.linalg.eigh`、camera/normal/cell 分組。Track1 idx 1505 的 cProfile 記錄超過一萬次小矩陣 eigensolve；大量小函式呼叫與分組是主要 CPU 瓶頸，GPU 在等待後處理。

cProfile 本身增加耗時，以下速度表使用另外的**無 profiler**重複量測，不使用 profiler 總時間算倍率。每場景每 backend 三次；reference／CPU 為三次中位數，CUDA 為後兩次平均，首輪含初始化另保存。

| 真實場景 | 原 scalar CPU 取樣 | 批次 CPU | 批次 CUDA | 取樣倍率 | 選點索引／report |
| --- | ---: | ---: | ---: | ---: | --- |
| Track1 idx 1505 | 0.797 s | 0.230 s | 0.114 s | 6.99× | 完全相同 |
| Track1 idx 1823 | 1.160 s | 0.261 s | 0.153 s | 7.58× | 完全相同 |
| Track2 idx 1397 | 0.564 s | 0.154 s | 0.079 s | 7.18× | 完全相同 |
| Track2 idx 1661 | 0.430 s | 0.159 s | 0.066 s | 6.48× | 完全相同 |

這些取樣微量測與當時使用者完整重建並行，GPU／CPU 有其他負載；它們證明重構移除了大量 Python 開銷，不是隔離硬體的精準 throughput 測試。批次 CPU 已經顯著改善，CUDA 排序再降低 voxel 成本；不能把全部改善都歸因於 GPU eigensolve。

## 原程序停止後的真正推論實測

以相同原始 JPG、模型、prior、候選尺度及 map cache，單獨執行新入口四幀，未使用 replay、未開 `--verify-sampling`。第一幀 **1.239 s** 含 CUDA／校正快取暖機；後三幀為 **0.783／0.776／0.781 s**。

| 暖機後重建階段 | 三幀平均 | 執行位置 |
| --- | ---: | --- |
| 模型推論（含原 infer 內部處理） | 584.5 ms | 原 GPU 模型／原 bf16 forward |
| voxel＋exact kNN＋PCA＋均衡取樣 | 123.1 ms | GPU＋CPU |
| JPG 讀取／解碼／校正 remap | 18.8 ms | CPU，快取 calibration map |
| depth 反投影＋座標轉換 | 15.6 ms | CPU float64 |
| 模型輸入前處理 | 3.0 ms | 原流程 |
| PLY＋labels 寫入 | 1.7 ms | CPU／磁碟 |
| 其他：valid mask、資料搬移、hash 等 | 約 33 ms | 混合 |
| **重建總計** | **779.8 ms** | 不含啟動與後續定位 |

這段另跑原 ROS 定位，`timings.csv.total_ms` 四幀平均 **27.0 ms**；這個小段的主要成本是重建。歷史最佳前相機完整 run 的定位平均約 Track1 **86 ms**／Track2 **45 ms**，其中 PLY 載入約 70／37 ms；不同輸入點數與場景不可直接算同算法加速倍率。

同位置舊紀錄與新版暖機平均為 1.990 → 0.780 s，觀察到約 **2.55×／耗時減少 60.8%**。原版是先前量測，新版在原程序停止後執行，沒有交錯 A/B，機器負載可能不同。小段逐幀重建時間不含模型載入、checkpoint／原始資料初始核對、幀末 JSON／`empty_cache()`，也不含 ROS 啟動／讀圖與定位；完整一輪仍須看實際總 wall time。新版現在的主要瓶頸已變為模型推論，約占這段逐幀時間 **75%**；不更換模型的前提下，之後可改善的空間比這一輪小。

## 改動與數學對應

實作在 `scripts/rig_geometry_accel.py`，獨立 adapter 為 `scripts/run_rig_fast.py`。原 `run_rig_full.py`／`calibrated_rig_geometry.py`／`three_camera_source.py` 及 C++／CUDA 配準程式都沒有因本輪加速而修改。

- **Voxel representative**：沿用 CPU float64 `floor(point / 0.3)`，將整數 xyz 可逆打包為 int64 key，在 GPU stable sort；每 voxel 仍選原順序第一點，最後回原索引排序。超出 int64 安全範圍直接失敗。
- **局部 PCA**：同一份 exact CPU `cKDTree.query(k=16)` 結果，距離門檻仍 1.5 m，至少 6 點。批次計算同一個 covariance：`Σ = Σ_j (p_j−μ)(p_j−μ)ᵀ / n`，再以 float64 CUDA `eigh` 求最小特徵值的 normal。原 eigenvalue／quality 門檻不變。接近門檻或 normal 軸並列時，用原 scalar CPU 運算決定，避免浮點誤差改變分類。
- **均衡選點**：相同 camera×normal bins、2 m spatial cells、confidence 降序與原 idx tie-break；以陣列排序實作相同的逐 cell round-robin，budget 仍 6000，選出索引排序不變。
- **影像校正快取**：每個 calibration／解析度／mask 只建立一次 fisheye remap。仍由同一 OpenCV 解碼、線性插值及 mask 最近鄰插值，六個真實 view 的冷／暖快取 RGB、K、mask 逐元素相同。

排序使用 [PyTorch stable sort](https://docs.pytorch.org/docs/2.8/generated/torch.sort.html) 保留等 key 順序；PCA 使用 [batched `torch.linalg.eigh`](https://docs.pytorch.org/docs/2.8/generated/torch.linalg.eigh.html)。CUDA 小矩陣 eigensolver 有額外 workspace；首次嘗試 chunk 8192 要求約 4.14 GiB workspace，在原模型並行時失敗，已保存失敗紀錄。最終採 **chunk 256**，後續實際 CUDA 測試通過，不在錯誤時靜默切 CPU。

未變更 bf16／AMP 設定、模型 checkpoint、三視角 448×294 解析度、confidence mask、near/far range、深度尺度、baseline／安裝高度、時間運動補償、heading_rts prior 或 directional 配準目標。未用降低點數／少看一個相機／減少迭代來換取速度。

## 為何仍保留部分 CPU

目前把成本高且能在相同規則下驗證的 PCA／voxel sorting 搬到 GPU。exact KD-tree 在上述場景約 **14–49 ms**；以現有 PyTorch 改成全距離 GPU 矩陣會增加平方級記憶體，換近似搜尋則會改鄰居及分類，所以此輪保留。反投影／座標轉換約 16 ms，保留 float64 可維持 PLY 位元一致；改 GPU 插值也需重新驗證每個 pixel／mask。

定位原本已有 CUDA 最近鄰；CPU 解的是 **4×4 正則化線性系統**，搬這個小系統需額外傳輸與同步，本次沒有把它列為瓶頸。磁碟 hash／檔案寫入也不能只靠搬 GPU 消除。模型內部及每幀 `empty_cache()` 維持原設定，尚未量測安全性與顯存峰值前不移除。

## 一致性與續跑

cached real predictions 的 Track1 1505–1506、Track2 1661–1662，原版與加速版各自產生 PLY／labels、跑原 ROS solver；所有檔案 SHA256、取樣 report、per-camera pose 與定位 CSV 相同：

| 診斷片段 | 相同 CSV SHA256 |
| --- | --- |
| Track1 1505–1506 | `df9ab5a481ebc15c75d92feed8d46e4968a0cd7d668fedbe1917c242296bb63a` |
| Track2 1661–1662 | `317e22e15dcf11e302fd7448d71badbbcae4be4dfb06710a13a5831c7bf7e4a3` |

真正重新推論的 Track1 2483–2486 也與原完整 run 已保存幀的 PLY／labels SHA256 相同。這支持切換實作、保留已完成資料；仍不是全序列浮點逐幀一致或精度泛化證明。ROS 環境共 36 項測試，35 項通過、無 Torch 的 CUDA 項跳過 1 項；主機 ML venv 的五項加速測試包含真 CUDA，全數通過。測試涵蓋 diagonal normal ties、confidence ties、voxel 邊界、退化點、少量點、真實校正快取。完成後的加速診斷 run 續跑有重用 localization，`performance.json` 保留不變。

`migrate_rig_fast.py <原 run root>` 先取得舊 Track lock，核對原 identity、程式／模型／套件／標定／index／prior、每幀原始影像及 PLY／labels hash；只複製已原子提交的完整幀，忽略未提交 temporary dirs，不改舊目錄。新 identity 加入 execution 與 `imported_reference.json` SHA256，保留原 identity 與逐幀 metadata hash。新版不直接接受原版 identity。匯入的舊幀沒有新 stage timers，只用後續新增幀衡量加速效能。

## 本次直接接續

本次已匯入完成，使用者直接執行下列命令；不必再次匯入，也不必重編 C++：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
bash scripts/run_three_camera_full_fast.sh --resume \
  outputs/run/track00_full_2610181301
```

預期先核對已完成 2487 幀，重建 Track1 剩餘 524 幀，定位 Track1，再重建／定位 Track2。初始 hash 核對與模型載入仍需時間。後續中斷使用同一命令。

下一輪從頭開始的入口：

```bash
bash scripts/run_three_camera_full_fast.sh --dry-run
bash scripts/run_three_camera_full_fast.sh
```

若需匯入其他已停止原版 run，執行 `outputs/ml_runtime/venv/bin/python scripts/migrate_rig_fast.py <原 run root>` 一次，再把工具印出的新 Run 交給加速 wrapper 的 `--resume`。wrapper 拒絕同時啟動另一個重建或 localization；結果在該 run 的 `result_1.csv`／`result_2.csv`，不自動提交。

完整證據目錄為 `outputs/run/track00_test_2610181247/`：`live_run_observation.json`、`reference_profile.json`／pstats、`accelerator_components.json`、`performance_summary.json`、`output_equivalence.json`、`real_inference_comparison.json`／`real_inference_summary.json`、逐組 command／log／source snapshot、`migration.log`／`migration_validation.json`。資料與 outputs 被 Git 忽略，跨機需另外保存。新幀的 `stage_seconds`／`sampling_stage_details` 與 `performance.json` 可用來確認下一輪真正瓶頸；`--verify-sampling` 會額外重跑 CPU reference，僅供短段核對，不作速度量測。

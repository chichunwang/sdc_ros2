**2026-10-23 現行規則提示：**因果定位適用所有 Track；不得以未來資訊回修 pose、使用 RTS／backward smoothing，或在定位後離線修正軌跡。Track1／2 GNSS 仍可作輸入，但 RTS 不因此合規。本文相關算法、命令與成績均為歷史紀錄，不作現行正式提交推薦；規則與程式影響見 [48](48_RULES_AND_CODE_AUDIT_20261023.md)。

# 44｜CUDA 加速 map 尺度配準（2026-10-20）

## 範圍與狀態

本輪把 `run_map_smoother.py` 中 map 尺度配準的部分密集計算移到 CUDA，目標是縮短逐幀 registration 時間，同時保留既有 CPU 結果與尺度／接受判定。這是實作加速，不是新的定位精度結論；沒有新 Kaggle 分數，也沒有宣稱 GPU 版本的完整 Track 更快。

**2026-10-21 現況入口：**Track2 `track02_full_2610212056` 在原始 idx 653 的 PCA 對 9070 點一次建立完整 float64 covariance batch 時因 CUDA 記憶體不足失敗，此前已輸出 653 幀。已將 PCA eigendecomposition 改為 256 點分塊，kNN、點數、float64、參數及判定門檻不變；GPU 機制與 map／graph 回歸 18 項通過。修正後 `track02_test_2610212107` 的 idx 640–663 共 24 幀完成 validation，idx 653 成功輸出；該小段不等於 full 成功或精度提升。根因、記憶體量測與短段限制見 [46｜CUDA PCA 記憶體界限](46_CUDA_PCA_MEMORY_BOUND_20261021.md)。本文件下列 GPU voxel／96 幀短測及 51 項檢查是 2026-10-20 歷史結果。沒有新 full 完成紀錄或 Kaggle 分數。

新增入口為 `scripts/cuda_scale_geometry.py`、`scripts/cuda_scale_input.py` 與 `scripts/run_map_smoother_gpu.py`。GPU voxel input 已整合；2026-10-20 的 51 項機制／回歸與兩段共 96 幀、從 raw bag／map 開始的端到端短測通過。2026-10-21 PCA 分塊修正後的 24 幀短段另見 [46](46_CUDA_PCA_MEMORY_BOUND_20261021.md)，兩批結果不可混作同一驗證。

## GPU／CPU 工作邊界

GPU 環境沿用現有 PyTorch 2.8 + CUDA 12.8、RTX 5060 8 GB。自訂 CUDA kernel 由 Python 呼叫 NVRTC 12.8 並透過 `ctypes` 載入；不需要 host `nvcc`、`sudo`、新增系統套件或環境安裝。編譯 binary 與 cache 僅放在當次 `outputs/run/<run>/cuda_cache/`。

已搬到 GPU 的部分包括 float64 bounded k=8／k=24 spatial-grid nearest-neighbor、PCA／covariance／eigh、ground／non-ground 配對、deterministic patch balance／hash winner 與體素選點。仍由 CPU 執行 ASCII PLY 解析、小型 7 未知數 least-squares、界限／旋轉檢查與稀疏 graph。特定有限 tie 由 reference CPU cKDTree 裁決，voxel 近整數邊界以原 NumPy floor 裁決；這是明確的精度規則，不是 CUDA error 時的 CPU fallback。CUDA 執行錯誤會直接失敗。

grid kNN 用逐層 cell 搜尋、cell AABB 下界與嚴格距離界做剪枝，保留半徑內的 exact k 個鄰點；半徑外只回無效候選，等同原 fit 的距離篩除。CUDA kernels 以 float64 並關閉 FMA contraction，PCA 與矩陣也用 float64。密集格記憶體、int64 patch key 溢位會明確失敗。初版 GPU 曾因小 kernel／同步開銷而未加速；最終版本融合配對及 patch winner kernels，原 fit／接受程式碼保持相同。

原 `map_scale_icp` 計算路徑保持不變。透過 per-instance code object clone 替換依賴，重用原有 fit／accept 程式碼，不修改全域 module。`ScaleICPConfig`、`SlowMapConfig`、seed（6、8.7、11）、迭代次數及 gates 都維持原設定。

## 已核對的配對結果

`outputs/run/track02_test_2610202028/paired_validation.json` 記錄 Track2 八個分散原始 frame：0、47、604、1600、1647、2600、3251、3298。CPU 與 GPU 的 map accepted 狀態及拒絕理由相同；pose 最大差為 `2.27e-13`，scale 最大差為 `4.97e-14`。在這組配對樣本上，registration 總時間比為 2.8653555，單幀約 1.75–5.01 倍。這只描述該 run 的 registration 部分與樣本，不能外推整體 full 時間、其他硬體或精度提升。

`track00_test_2610202034/verification.json` 與 `check_1.log` 記錄七項 GPU 機制及 44 項既有回歸，全 51 項通過。GPU 檢查含 exact 鄰點及 crop 外查詢、tie 與原 CPU 選擇一致、有效 PCA 法向正負號不變性、room 接受／ground 退化拒絕、CUDA 不可用明確失敗、非有限資料拒絕、ASCII voxel 邊界原 XYZ／選點一致。

`track00_test_2610202031/verification.json` 的六組合成 voxel 邊界／限點及三個真實 frame（0、1600、3251）與原 loader 的 XYZ 陣列逐值完全相同。讀 PLY 及內部 voxel 的時間仍記為既有 `load_ms`，不改欄位格式。

另以 GPU 端到端中段的相同 map／prior 做 48 幀 CPU 控制，保存於 `track02_test_2610202032/comparison.json`：accepted 與拒絕理由全部一致、pose 最大差 `4.55e-13`，該組 registration 時間總比約 **2.43 倍**。這和前述八個分散影格的 2.87 倍是不同樣本，均不是完整 Track 效能或精度結論。

| GPU run／Track2 原始影格 | 幀數 | 幾何接受 | Graph | pipeline／逐幀總秒數 |
| --- | ---: | ---: | --- | --- |
| `track02_test_2610202031`／1600–1647 | 48 | 15 | 無慢 anchor，保留本輪感測初始化 | 27.783／12.623 |
| `track02_test_2610202032_0002`／0–47 | 48 | 48 | 實際求解並套用，graph 0.663 s | 31.764／14.495 |

兩段均 returncode=0、輸出完整有限、CSV／矩陣／timing 一致。`0–47` 起始片段與歷史 CPU `track02_test_2610200236` 的 prior／fine map bytes 完全相同；48 個 map 觀測 accepted／理由一致、pose 最大差 `4.55e-13`。最終 graph 輸出的 XY RMS 版本差約 `7.02e-7 m`，CSV bytes 不同；不承諾逐 byte 相同或新分數。詳見 `track00_test_2610202034/historical_same_input_head_comparison.json`。歷史 CPU 與本輪 GPU 的主機負載不同，不把歷史 pipeline 秒數當等條件加速實驗。

## 接受率診斷的解讀

`outputs/run/track02_test_2610202016/acceptance_audit.json` 記錄 1693 幀 snapshot，其中 1485 幀 accepted（87.7141%）、208 幀 rejected；主要拒絕原因為 non-ground（204 幀，可能同時有其他原因）。最長拒絕連續區段為 51 幀，約 1.79994 秒。前五個完整 12 秒區塊都有 XY 支持的慢 anchor；最後一個不完整區塊不足以作結論。

此 audit 使用 prior yaw 作參考，只能說明 XY 支持；不能據此宣稱最終 yaw anchor、graph 已執行／套用，或定位精度提升。`map_accepted=false` 表示該幀沒有新增 map measurement，不代表丟棄該 pose；有支持的區塊仍可能在 graph 約束該 pose。若整段沒有 anchor，輸出保留感測器初始化，並非原 C++ 最佳結果；長時間缺少 anchor 會降低慢偏差校正能力。不可為提高 accepted 比例而放寬門檻。

## 手動完整命令

GPU voxel input 與端到端小測已完成。full 命令從原始 bag／map 建立新 run，不依賴歷史 run；CPU／GPU 配對測試讀取同一份既有 map cache，這不代表 full 命令依賴該 cache。CPU full 的 96 個既有來源 SHA 本輪最後核對皆不變；新 GPU 功能全部新增檔案，不更換執行中的來源。

目前手動 Track2 命令如下；Track1 使用同一算法與參數，只將 `--track 2` 改為 `--track 1`。不得由文件觸發完整定位或 Kaggle 提交。

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
bash scripts/dev.sh up
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 outputs/ml_runtime/venv/bin/python scripts/run_map_smoother_gpu.py --track 2 --full
```

執行輸出遵守 [08 固定輸出格式](08_MANUAL_WORKFLOW.md#固定輸出格式使用者指定-2026-10-20)，包括 `frame_timing_v1`、既定逐幀計時與 `timings.csv` 欄序。`track02_full_2610201936` 是 2026-10-20 的 CPU 歷史 run；截至 2026-10-21 已無執行中程序，僅有 2581 筆 observations，沒有完整 validation 或完整 CSV，因此不得記為完成。2026-10-21 的 GPU `track02_full_2610212056` 亦因上述 PCA CUDA OOM 在 idx 653 中止，不能當作完整 Track 成功。此處命令仍是使用者手動啟動的入口，不代表本文件執行了 full。

## 證據來源

- 配對數值與 registration 計時：`outputs/run/track02_test_2610202028/paired_validation.json`。
- 接受／拒絕與 anchor 支持診斷：`outputs/run/track02_test_2610202016/acceptance_audit.json`。
- mechanism tests／端到端摘要與程式 SHA：`outputs/run/track00_test_2610202034/`。
- voxel input 原始 XYZ 等值：`outputs/run/track00_test_2610202031/verification.json`。
- 48 幀相同 map／prior CPU 控制：`outputs/run/track02_test_2610202032/comparison.json`。
- NVIDIA runtime compilation 介面依 [NVRTC 官方文件](https://docs.nvidia.com/cuda/nvrtc/index.html)；同步計時依 [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)。這些是 API 來源，不是本機加速或精度證據。

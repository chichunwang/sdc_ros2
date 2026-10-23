# 46｜CUDA PCA 記憶體界限與分塊修正（2026-10-21）

## 事件與根因

使用者執行的 GPU full `outputs/run/track02_full_2610212056` 已產生原始 idx 0–652 共 653 筆 timings 與 observations。下一筆 idx 653、timestamp `1727241204788076715` 的 PLY 有 41043 點；loader 載入 12000 點，range filter 後有 9070 點進入 PCA。當時 `torch.linalg.eigh` 一次處理形狀 `(9070, 3, 3)` 的 float64 covariance batch，單次額外 peak 記憶體量測為 4,925,410,816 bytes（約 4.59 GiB），CUDA 回報該請求約 4.59 GiB、當下可用約 2.46 GiB，故在此幀失敗。

量測來自 `outputs/run/track00_test_2610212101/workspace_probe.json`，環境為 PyTorch 2.8.0+cu128、CUDA 12.8、RTX 5060 8 GB。文件中的「額外 peak」是這個 probe 對指定 PCA 工作區的觀測，不能直接換算為任何 GPU 的固定最低容量。當時 allocator 顯示約 4.38 GiB reserved-but-unallocated；因此「請求與 free 的差約 2.13 GiB」也不能解讀成精確的升級顯存需求，或據此把 allocator 環境變數當成主要修正。

## 修正與驗證範圍

主 agent 將 `cuda_scale_geometry.py` 的 PCA eigen decomposition 限制為每批 256 點；source 與 map normals 共用該 helper。對同一 `(9070, 3, 3)` probe，batch 256 的額外 peak 為 139,019,776 bytes（約 132.58 MiB）。這是工作區記憶體量測，不是定位精度或完整輸出等價證據。

此修正只切分 `eigh` 的批次，未改 kNN、輸入／配對點數、float64 精度、尺度或其他參數、gates、迭代次數。CUDA 錯誤仍直接失敗，沒有 CPU fallback。runner 新增 `processed_frames`、`failed_original_idx` 與 CUDA peak allocated／reserved diagnostics；既有 console 行及 `timings.csv` 欄位不變。

`outputs/run/track00_test_2610212106/verification.json` 記錄 8 項 GPU 機制與 10 項 map／graph 回歸，共 18 項通過。後續修正後端到端短段 `outputs/run/track02_test_2610212107/` 以 idx 640–663 跑完 24 幀，validation returncode 為 0、`processed_frames=24`、`output_complete_and_finite=true`；9 幀 accepted、15 幀 rejected，原失敗位置 idx 653 成功且為 accepted。Graph 因 `no_supported_slow_map_anchors` 未執行。pipeline 時間為 24.6363824 s，其中 frame 累計 9.058342309 s、graph 計時 0.01028161 s。

這段短測由新 24 幀 raw 輸入生成 prior 與 map crop，和中止 full 使用的 prior／crop 不同；因此不能把兩者 pose 變化直接歸因於 PCA 分塊，也不能據此聲稱數值配對等價或精度改善。此處只確認端到端短段完整性及 idx 653 不再觸發原 OOM。

同一短段整個 process 的 PyTorch allocator peak allocated 為 184,632,832 bytes（176.08 MiB），peak reserved 為 213,909,504 bytes（204 MiB）。這是 PyTorch allocator 統計，不涵蓋全部 CUDA driver 與桌面 GPU 使用量；不能把 204 MiB 當成整張 GPU 的總佔用。

### 與中止 full 同輸入配對及重複 PCA

後續 `outputs/run/track02_test_2610212110/paired_validation.json` 使用中止 full 的同一 fine map、sensor prior 與四個原始 PLY（original idx 642、652、653、654）做 CPU／GPU 配對。四幀 accepted 狀態與拒絕理由完全一致，皆因 `nonground_support` 拒絕；pose 最大絕對差為 `4.547473508864641e-13`，scale 最大絕對差為 `7.815970093361102e-14`。idx 642／652 與失敗 full 原 GPU observation 的 pose、scale 完全相同，accepted 狀態與理由亦相同。

因此，同一 full 輸入下 idx 653 這次是正常完成 PCA 後因 `nonground_support` 被拒絕；先前 24 幀短段的 idx 653 為 accepted，因該短段使用新生成的 prior／map crop。兩次結果條件不同，不能混為同一門檻結果，也不能把 `nonground_support` 當成 OOM。此次配對的 6 個 failed-full input/output 檔案 SHA 前後相同。

`track02_test_2610212110/verification.json` 亦記錄在原 failed-full GPU map/cache 常駐時，對 idx 653 的 9070 個真實 PCA 點，以 16 種不同點數重複 PCA 共 80 次；每次 eigen decomposition 的批次上限仍為 256，品質／法向結果逐次一致，過程未再 OOM。整個 process 的 PyTorch peak allocated 為 361,998,848 bytes（345.23 MiB），peak reserved 為 574,619,648 bytes（548 MiB）。這仍是 allocator 統計，不是整張 GPU 的總佔用或最低顯存需求。

先前 `track02_test_2610212109/validation.json` 記錄四幀配對已通過，但後續 stress setup 因診斷腳本誤用 `config.scale_max`／`config.range_m` 欄名而產生 `AttributeError`，壓力段未執行；`pipeline_error=false`。該 run 保留原 validation.returncode=1，不代表 solver 配對失敗或 CUDA PCA OOM。修正診斷欄名後另建 `track02_test_2610212110`，配對及 80 次重複 PCA 均完成。此為研究腳本準備錯誤的負面紀錄，與定位 solver 失敗分開記錄。

## 尚未證實的事項

- 後續使用者已完成 `track02_full_2610212125` 的 3299 幀，輸出完整、graph 求解且套用，未再遇到本次 OOM；回報顯示分數 0.77，較原最佳約 0.30 差，不能將運行完成視為精度提升。完整實作盤點與分數紀錄見 [47](47_IMPLEMENTATION_STATUS_AND_TRACK2_SCORE_20261021.md)。
- 24 幀短段只驗證指定區間的端到端運行與輸出檢查，不能替代 full。
- `map_accepted=false` 是 map 觀測未被採用的定位結果，不能當成 OOM 證據；OOM 是 CUDA 記憶體例外及失敗位置診斷所記錄的執行錯誤。
- Track1／Track2 原最佳 CSV 未改；已知最佳分數仍為 1.15008／0.30488。

## 使用者手動入口

執行順序維持專案固定格式：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
bash scripts/dev.sh up
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 outputs/ml_runtime/venv/bin/python scripts/run_map_smoother_gpu.py --track 2 --full
```

完整定位與外部提交由使用者手動執行；後續已完成的 full 與回報分數另依 [47](47_IMPLEMENTATION_STATUS_AND_TRACK2_SCORE_20261021.md) 記錄，不由命令本身推定上傳的 CSV bytes。

## 證據

- 首次 full 中止位置與 653 筆部分輸出：`outputs/run/track02_full_2610212056/`。
- PCA workspace／批次 peak 量測：`outputs/run/track00_test_2610212101/workspace_probe.json`。
- 18 項修正後 GPU 與 map／graph 回歸：`outputs/run/track00_test_2610212106/verification.json`。
- 修正後端到端 24 幀短段：`outputs/run/track02_test_2610212107/`。
- 中止 full 同輸入 CPU／GPU 四幀配對及 PCA 重複檢查：`outputs/run/track02_test_2610212110/verification.json`、`outputs/run/track02_test_2610212110/paired_validation.json`。
- 前一次配對後壓力 setup 未執行的診斷腳本錯誤紀錄：`outputs/run/track02_test_2610212109/validation.json`。

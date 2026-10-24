# 51｜Track3 性能與顯存驗證紀錄（2026-10-24）

## 部署版本與結果範圍

部署版本為 `track3_cuda_v4_perf_v1`，基於 `c324774`，只更改計算實作與增加 CuPy pool 記錄／管理；配準模型、門檻、迭代數、採樣、座標、FP64 與因果 pose 更新規則均未調低。

本次 GPU 驗證使用生成資料，沒有執行真實 Track localization。`track00_test_2610242232` 的 GPU 測試 55 項通過，耗時 93.61 s，另有一項既有 Matplotlib `Axes3D` warning；comparison checks 全數通過。詳細結果見 [`gpu_pytest.xml`](../outputs/run/track00_test_2610242232/gpu_pytest.xml) 與 [`comparison.json`](../outputs/run/track00_test_2610242232/comparison.json)。

原 full `track03_full_2610242203` 已以 v4 baseline 自然完成 3142 幀，2926 幀接受 map 更新，`elapsed_seconds=6275.19`。三組標準唯讀圖表及來源 manifest 成功產生，原始輸出與 SHA 保存在該 run 的 [`test_info.json`](../outputs/run/track03_full_2610242203/test_info.json) 和 [`analysis/`](../outputs/run/track03_full_2610242203/analysis/README.md)。這不是 `v4_perf_v1` full，因此無法由此得出新版本 full 速度或精度。

## 實作改動

| 熱點 | 改動 | 保留的演算法定義 |
| --- | --- | --- |
| Backtracking 與 solver iteration | line search 接受候選時，`backtrack()` 已計算候選姿態的 `(H, g, cost, coverage)`；下一次 iteration 沿用該組資料，不對相同 `T`、`scale` 重複呼叫 `linearize()`。 | 仍完整評估每個 backtracking 候選；原有步長、cost、coverage、接受條件與停止條件不變。 |
| NDT Gaussian voxel 對應 | CUDA kernel 搜尋 ±2 格共 125 個 voxel key，使用排序 key 二分查找，再套用 `1.5 × leaf` 半徑 gate；NDT 以 Mahalanobis 距離選取。Kernel 使用 `--fmad=false`。 | 保留 FP64、signed-21-bit 範圍、候選半徑及既有 tie 順序；不建立 query × 125 × 3 × 3 大型暫存張量。 |
| 相機 pixel 到 3D 對應 | 每個 tracked pixel 使用 128-thread block 搜尋最近投影點並 block-reduce；PnP 只取得通過 4 px gate 的選取 3D 點。 | 保留 Euclidean pixel 距離、原始索引 tie 順序、PnP 門檻；不建立 query × point 距離矩陣。 |

GPU 對照中 pixel index 完全相同，距離通過 `rtol=1e-13`、`atol=1e-10`；NDT／ICP 的 validity mask 和有效 index 完全相同；生成 registration 的 `T`、scale 相同，接受狀態及 gate 原因相同。這些是合成案例的等價性證據，並非真實定位精度驗證。

## 生成資料 GPU 微基準

測試環境為 RTX 5060。每個元件先暖機 2 次，再量測 5 次取同步 wall-time median；合成 registration 暖機 2 次，再量測 3 次。pool reserved 是 benchmark checkpoint 抽樣，不是整卡 VRAM 或配置瞬時 peak。

| 案例 | 輸入 | Baseline median | `v4_perf_v1` median | 加速比 | Pool reserved（baseline → 新版） |
| --- | --- | ---: | ---: | ---: | ---: |
| Pixel projection search | 1200 queries、50,000 targets | 806.644 ms | 3.433 ms | 234.98× | 342.56 → 0.803 MiB |
| NDT voxel search | 12,000 queries、561 cells | 44.879 ms | 0.590 ms | 76.04× | 361.20 → 6.432 MiB |
| ICP voxel search | 12,000 queries、561 cells | 27.582 ms | 0.466 ms | 59.17× | 120.83 → 6.432 MiB |
| 合成 registration | 2 m NDT 最多 30 iterations + 1 m ICP 最多 40 iterations | 124.589 ms | 56.115 ms | 2.22× | — |

合成 registration 關閉 preprocessing 與 initializer，不包含影像前端、點雲去噪或首幀初始化。2 次暖機加 3 次量測，共 5 次 solver 執行的 `linearize()` 呼叫數由 baseline 100 降至新版 60。元件和這組限定合成 solver 的倍數不能外推成完整 Track 加速倍數。

## 顯存政策與量測

| 設定 | 值 | 定義 |
| --- | ---: | --- |
| `pool_limit_mib` | 4096 MiB | CuPy allocator pool 上限參數，不是整卡顯存上限 |
| `trim_above_mib` | 3072 MiB | pool reserved 壓力門檻 |
| `reserve_mib` | 1024 MiB | 啟動時計算 pool cap 的 reserve 預算 |
| `min_trim_mib` | 256 MiB | 嘗試釋放 cache 所需的最小 cached 量 |

啟動時 pool limit 取設定上限、`device_free + pool_reserved - reserve` 及已存在的較小 pool cap 中最小值。若 budget 低於 pool 已使用量，會報錯；不降精度，也不改用 CPU。要求 trim 的 map、initializer、submap 與 frame-end checkpoint，只有 cached 至少 256 MiB 且（reserved 達 3072 MiB 或 device free 低於 1024 MiB）時，才呼叫 GPU／pinned pool 的 `free_all_blocks()`。它只釋放未使用的 cache，仍在使用的配置不會釋放；分裂區塊也可能令實際釋放量為零。

pool cap 不涵蓋 CUDA context、libraries、pool 外配置或其他程序，也不保證全程保留 1024 MiB 整卡 free。`gpu_memory.jsonl` 與 `test_info.json` 的值是檢查點樣本，不是瞬時峰值。CuPy pool 快取行為見 [CuPy memory management](https://docs.cupy.dev/en/v13.6.0/user_guide/memory.html)。

allocator shape stress 共 25 種生成輸入；雙方均完成相同的 25-case prefix。pool reserved checkpoint 最大值由 baseline 4.25823 GiB 降至新版 17.073 MiB。這是該壓力測試中的 CuPy pool 抽樣，不是實際 full 的整卡使用量或 VRAM peak。成功 full 且具有 3142 筆 frame-end 記錄時，部署版會在保留三組標準圖表的基礎上，額外產生 `04_gpu_memory.png/pdf`。

## 實際資料與限制

baseline full 可讀取的前 816 幀中，排除首幀的 idx 1–815 平均 load 為 988.5 ms、registration 為 911.3 ms、total frame 為 1987.4 ms、preprocessing 為 170.0 ms。首幀 registration 128138.4 ms、total 129756.1 ms，不代表穩態。同一 run 的 35 次、每秒一次 `nvidia-smi` 樣本記錄程序 5476–5478 MiB、整卡已用 6553–6557 MiB、最低 free 1149 MiB；這是低頻抽樣，不是 peak，也不能據此推論顯存洩漏。

目前沒有 `v4_perf_v1` 的真實 Track full 速度或獨立真值精度。接受率、synthetic 等價性、元件加速比與 pool stress 都不能替代實際定位誤差。benchmark、pytest 和比較結果保存在 `outputs/run/track00_test_2610242232/`；`outputs/` 受 `.gitignore` 管理，不會自動同步到其他電腦。本次 benchmark 的[圖表與 PDF 報告](../outputs/run/track00_test_2610242232/analysis/README.md)已存於該 run 的 `analysis/`。

## 實作與結果來源

- [`track3/gpu.py`](../track3/gpu.py)：accepted backtrack state reuse、voxel model。
- [`track3/search.py`](../track3/search.py)：pixel 與 voxel CUDA search kernels。
- [`track3/memory.py`](../track3/memory.py)：pool cap、checkpoint、cache trim 與 summary。
- [`track3/visual_prior.py`](../track3/visual_prior.py)：pixel 對應和 PnP 輸入。
- [`scripts/run_track3.py`](../scripts/run_track3.py)：記憶體記錄及 observation 輸出。
- [`scripts/analyze_run.py`](../scripts/analyze_run.py)：full-run memory plot。
- [`config/track3.json`](../config/track3.json)：版本與 pool 參數。
- [`benchmark_baseline.json`](../outputs/run/track00_test_2610242232/benchmark_baseline.json)、[`benchmark_optimized.json`](../outputs/run/track00_test_2610242232/benchmark_optimized.json)：量測原始資料。

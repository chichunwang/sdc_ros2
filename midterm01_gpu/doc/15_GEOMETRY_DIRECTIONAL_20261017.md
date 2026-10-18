# 15｜幾何可信度診斷與方向性配準權重

更新：2026-10-17。此版本已完成 GPU workspace Release 建置與數值測試；使用者後續完成兩條完整 Track／Kaggle 提交，回報 **Track1 1.15275、Track2 0.30576**，見 [完整成績與 CSV](17_GEOMETRY_RESULTS_20261017.md)。Track2 有 252 幀退回當幀先驗，仍有完整有限輸出；第 410 幀警告及後續診斷修正見 [調查紀錄](18_GEOMETRY_FALLBACK_20261017.md)。後續完整測試與提交仍由使用者手動執行。

前一版已評分對照為 `heading_rts + geometry off`：Track1 1.16951、Track2 0.31967，見 [歷史成績與 CSV](14_HEADING_RTS_RESULTS_20261017.md)。本次沿用 heading_rts，明確選 `--geometry-mode directional`，總分再降低 1.43%／4.35%；這不是 diagnostic 模式或 motion_rts 的分數。

## 三種模式

| benchmark 選項 | 行為 | 目前證據 |
| --- | --- | --- |
| `--geometry-mode off`（預設） | 原已評分 point-to-point robust ICP | 原有完整 Track 與分數 |
| `--geometry-mode diagnostic` | 保留原目標與輸出，只增加幾何診斷 | 兩條前 50 幀均與已評分完整 run 的前 50 行逐 byte 相同 |
| `--geometry-mode directional` | 依可靠表面的法向／切向分配地圖殘差權重 | 使用者完整 3011／3299 幀通過，Kaggle 1.15275／0.30576 |

新模式只支援 `sensor_guided/cuda_robust_icp`。省略 geometry 仍為 off，省略 fusion 仍為 fixed，省略 profile 仍為 legacy/cuda_ndt。程式預設沒有更換。

## 改動目的與資料流

原 point-to-point 殘差把 XYZ 距離等向處理。道路或平牆上的相鄰地圖點可能產生沿表面的拉力，但一塊平面實際上主要提供法向約束。本次保留感測器先驗、四自由度與修正上限，讓可靠表面較強地約束法向，切向則降低權重，繼續由感測器先驗提供約束。

1. **CUDA 配準最近鄰保留**，依先驗裁切及搜尋候選地圖點。
2. 對實際用到的地圖點，CPU 建立／快取局部鄰域 PCA。每幀只在該 target index 第一次被使用時估計，不在每次迭代重算同一個點。
3. 由 covariance eigenvalues 分辨平面、線狀與厚／非平面鄰域，估計法向 `n` 與品質 `q`。
4. directional 模式以各點的方向性 metric 計算 robust 配準目標與導數；CPU 求解原四自由度方程、回溯線搜尋。
5. 依舊檢查當幀先驗目標、修正上限及時間平滑。可信幾何不足時輸出**當下先驗**，不沿用上一幀靜止 pose。
6. 另存 geometry.csv：地圖項自身的幾何指標與弱方向，避免先驗把 Hessian 撐成滿秩，掩蓋幾何不足。

CPU kNN 在這裡用於**表面鄰域 PCA**，不是 CUDA 配準最近鄰失敗時的 fallback；CUDA 錯誤仍直接失敗。

## 局部表面可信度

對每個地圖點取最多 16 個近鄰，保留距離 2.5 m 內的點；至少需要 6 點。將座標先移到鄰域局部原點，再以 double 計算 covariance，避免大座標的消去誤差。

令特徵值由小到大為 `λ0 ≤ λ1 ≤ λ2`，條件如下：

- `λ1 ≥ 0.04 m²`：避免線狀或極小鄰域假裝成平面。
- `λ0 / λ1 ≤ 0.25`：排除太厚、法向不明確的鄰域。
- `planarity = (λ1 − λ0) / λ2 ≥ 0.2`。
- 最小特徵值的特徵向量為 `n`，品質 `q = clamp(planarity, 0, 1)`。法向正負號不影響 `n nᵀ`。

這是幾何形狀的可靠性指標，不能排除動態物體、重複結構、單目尺度誤差或錯誤的 frame 契約，也不是定位正確機率。

## 有方向性的目標函數

令對應點殘差為 `r`，建立：

```text
M = n nᵀ + τ (I − n nᵀ)
d = sqrt(rᵀ M r)
```

預設 `τ=0.05`，表示法向係數為 1、切向係數為 0.05；仍保留少量切向項。這是帶局部表面方向的 anisotropic point-to-point residual，並非完整 GICP，也沒有把後端改名為 NDT。

沿用 Huber 門檻 0.5 m 與 **Euclidean 2 m 對應截斷**。令 `ρ` 為 Huber loss、`C=ρ(2)`、normal gain 為 `g`，每點地圖 cost 為：

```text
可靠且 Euclidean 距離 ≤ 2 m：e = g [C − q (C − ρ(d))]
無可靠幾何或超過 2 m：      e = g C

E(δ) = 0.5 δᵀ P δ + sum(e) / (N × 0.3²)
```

預設 `g=2`。固定未匹配 cost 的設計，避免單靠失去對應或轉向較低品質表面就降低目標。對應與局部權重固定時使用 Gauss–Newton 導數，每個候選步再重新查最近鄰並核對實際目標。

先驗 `P` 仍使用既有 XY 0.5 m、高度 1 m、yaw 3° 的正則化尺度；沒有把 heading smoother 尚未校準的標準差直接當成 map covariance。source 數量仍以 N 正規化，不靠增加重複點提高總地圖權重。

至少需要 10 個可靠 source 對應與 6 個不同 target 點，才接受 directional 更新。XY 1.5 m、Z 每幀 1.5 m、yaw 8° 上限、alpha=0.25 correction 平滑及最終目標檢查沿用原版。directional 與原模式的 cost 定義不同，`prior_cost/final_cost` **只在同幀同模式內比較**；Euclidean clipped RMSE 另存作診斷，不是 Kaggle 誤差。

## geometry.csv 怎麼看

每幀保存：

- Euclidean inliers、可靠幾何對應數、不同 target 數與 CPU PCA 查詢數。
- 平面品質平均、法向殘差 RMS、支持點在 map XYZ 的範圍。
- 法向分布的 3 個特徵值。
- **僅含地圖法向資訊**的 4×4 matrix、4 個特徵值與特徵向量。
- effective rank、condition 指標及是否接受 directional 更新。

診斷使用法向 Jacobian `nᵀJ`，排除感測器先驗、切向下限及 normal gain。先以 `S=diag(1 m,1 m,1 m,3°)` 正規化狀態：`δ=S u`，再分析 `H_scaled`；不直接混合公尺與弧度比較特徵值。特徵向量欄的第 c 欄對應第 c 個特徵值，從小到大排列，列為 scaled X/Y/Z/yaw；要解讀成實際修正方向需乘回 S。

effective rank 採門檻 `max(1e-4, 0.05 × 最大特徵值)`。它受狀態尺度與門檻影響，**不是嚴格的可觀測性證明**。例如地面可提供高度資訊，但平移 XY 的幾何資訊仍可能很弱；帶完整先驗的求解器可正常解出結果，不代表地圖本身四方向都有約束。

這些 matrix／eigenvalues 是局部模型指標，包含固定對應近似、相關點與重複結構等限制，不能把 inverse 直接宣稱為準確 pose covariance。本版方向性權重來自每點局部 `M/q`；eigensystem 用於診斷，沒有另外偷偷依 eigenvalue 改感測器權重。

`geometry_config.json` 由 C++ 寫出實際設定、模型版本 1、單位及 rank 門檻。benchmark 核對幀數／整數 timestamp／idx、有限值、支持數關係、eigenvector 正交性、matrix 重建、rank 與模式旗標，再寫入 `validation.json.geometry_diagnostics_not_accuracy`。

## 本次驗證與實測範圍

已使用 `scripts/dev.sh build-workspace --packages-select localization` 完成 Release 建置。2 個 CTest 契約、兩者另以真實 CUDA NN 執行的版本，以及 13 個 Python 測試全部通過。新測試涵蓋：

- 大座標地面 PCA、法向快取、線狀／稀疏幾何拒絕。
- diagnostic 與原目標／輸出完全相同。
- 純地面只有高度資訊；足夠寬的牆面提供 X／yaw 資訊，不虛構切向約束。
- 零切向權重時，弱方向由感測器保持，可靠法向仍能改善已知幾何。
- 幾何不足／無重疊回當下先驗，不帶入舊修正。
- Python 報告核對奈秒 timestamp、支持數、matrix／eigensystem，以及拒絕損毀資料。

最初開發驗證各跑前 50 幀，當時沒有啟動完整 Track；後續使用者的完整 Track2 證據另記於上方調查紀錄：

| Track／模式 | map 接受／幀數 | 前處理＋ROS（s） | NN＋CPU 細修／幾何（s） | 可靠幾何比例平均 | rank 分布 |
| --- | ---: | ---: | ---: | ---: | --- |
| Track1 diagnostic | 50／50 | 6.871 | 0.634 | 8.46% | 1：28、2：22 |
| Track1 directional | 50／50 | 7.029 | 0.646 | 10.82% | 1：29、2：21 |
| Track2 diagnostic | 50／50 | 8.580 | 0.206 | 20.04% | 1：48、2：2 |
| Track2 directional | 50／50 | 8.569 | 0.210 | 24.38% | 1：50 |

四輪均退出碼 0、空 validation_errors、50 次 map 接受。**兩條 diagnostic 結果各自與已評分完整 run 的前 50 行逐 byte 相同**。Track1 兩模式均明確命中同一個已核對 SHA256 的 1 m map cache，Track2 兩模式均停用 cache；不把跨 Track 的時間當加速倍率。第一次三輪之後修正了「完全沒有候選 map 點時」的 directional 常數 cost 倍率，最終 Release／CUDA 契約均已重驗；此改動未經過那三輪全部接受的更新路徑，來源快照與最終版本另存。

這些短測顯示模式可運作與診斷能揭露弱方向，**不證明新候選分數更好**。directional 在前 50 幀的 XY 修正 RMS 比原版更小（Track1 約 0.00564 m、Track2 約 0.00050 m），反映切向拉力被壓低；修正較小本身也不是精度改善。後段、轉彎與整條 Track 的取捨需依使用者完整測試及 Kaggle 分數。

證據位於 [本次開發驗證目錄](../outputs/runs/geometry_dev_07284ae0)、[四輪短測摘要](../outputs/runs/geometry_dev_07284ae0/smoke_summary.json)。各 run 的 params、frame、source／binary SHA256、來源快照、geometry CSV、耗時與 validation 均保留，原最佳 CSV 不變。outputs 被 Git 忽略，交接需另備份證據。

## 完整測試與上傳：由使用者執行

本機已完成建置，現在可直接執行下列命令，兩條依序完成。移機或再次改 C++ 時，先依操作手冊重做 Release 建置。

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu

python3 scripts/benchmark.py --track 1 --frames 3011 --profile sensor_guided --sensor-fusion heading_rts --geometry-mode directional
python3 scripts/benchmark.py --track 2 --frames 3299 --profile sensor_guided --sensor-fusion heading_rts --geometry-mode directional
```

新的 run 名稱會包含 `geometry_directional`。先確認 `validation.json` 的退出碼 0、`output_complete_and_finite=true`、空錯誤、正確筆數與 geometry_mode=directional；若有 sensor-only 幀，讀新 `guidance.csv.update_status` 及候選支持數，再用 geometry.csv 檢視輸出 pose 的幾何。舊幾何紀錄只有回退後的數據，不能當作候選拒絕原因。CUDA runtime 錯誤會直接失敗，不能當作正常的 geometry 拒絕。

可上傳候選仍是每個 run 的 **`diagnostic.csv`**（`idx,x,y,yaw`，yaw 弧度），兩條分別應有 3011／3299 筆。`geometry.csv` 只供診斷，不是提交檔。保留目前最佳答案與這次 run，依實際分數決定接受哪個版本。

可貼上的流程見 [16_run_geometry_directional.txt](16_run_geometry_directional.txt)。改為 `--geometry-mode diagnostic` 可保存幾何但保持原權重；改為 `--geometry-mode off` 可回到已評分權重。獨立比較時，heading_rts、尺度、voxel、cache 策略與其他權重保持一致。

## 方法與程式參考

- [`geometry_weighting.cpp`](../workspace/src/localization/src/geometry_weighting.cpp)：局部 PCA、品質 metric、data-only eigensystem 與輸出。
- [`sensor_guided.cpp`](../workspace/src/localization/src/sensor_guided.cpp)：directional 目標、導數、可靠支持門檻與先驗回退。
- [`geometry_report.py`](../scripts/accuracy/geometry_report.py)：benchmark 幾何契約與摘要。
- [PCL 官方法向估計教學](https://pointclouds.org/documentation/tutorials/normal_estimation.html)：以鄰域 covariance PCA 估法向；本次採自訂品質門檻。
- [Open3D robust kernel 教學](https://open3d.org/docs/latest/tutorial/pipelines/robust_kernels.html)：robust loss 與加權 Jacobian 的方法參考；本次沒有新增 Open3D 依賴。

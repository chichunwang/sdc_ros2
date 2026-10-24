# 50｜首段尺度、雜訊與初始化修正（2026-10-24）

本輪針對「一開始停住、相機點雲太小、無法沿道路前進」修改 v4：`track3_cuda_v4_denoise_geometry_visual_prior`。沒有使用 IMU、GNSS、未來影格、隱藏真值或離線軌跡修正。

## 觀察與假說

使用者要求停止的 v3 full `track03_full_2610242038` 在 254 幀只有 2 次 map 更新，最後一次 idx22；frame0–12 持續輸出同一個 initial prior。首幀候選 scale 2.902 被拒絕，輸出尺度仍是未標定的 3。程式把拒絕候選丟棄的機制正確，但初始化與細配準沒有完成，因而停住。

[官方整合規則第6頁](../../midterm01/docs/Competition%20Regulations/00_SDC_2026_Midterm01_Localization_%E6%95%B4%E5%90%88%E6%96%B0%E7%89%88_20261022.pdf#page=6) 把 Scale calibration／雜訊處理放在 scan matching 前，未指定 SOR 或參數。[PCL 的 SOR 說明](https://pointclouds.org/documentation/tutorials/statistical_outlier.html) 說明稀疏離群點會影響局部幾何估計；本次採用的是依相機深度正規化、median/MAD 門檻的 CUDA 版本，不冒稱與 PCL 的 global mean/std 版本相同。

假說分開處理：來源點有可清理的車頭／離群點；初始尺度與傾角不合；Euclidean correspondence 搭配 point-to-plane loss 的細階目標不一致，會造成 line search 困難。去噪單獨測試已顯示前者不能獨立解決全部問題。

## 變更與依據

- 官方前視 ego mask 經 fisheye rectify 後，移除車頭及 FOV 外點；來源 PLY 不回寫。
- CUDA exact kNN（k16）計算平均距離，除以前向深度 X，使用 median + 3×1.4826×MAD；保留遠處自然稀疏的幾何。整數索引與參數完整記錄。
- 當前 source/map 地面法向改善 roll/pitch；以 initial scale 及 init 高度／source plane offset 產生尺度候選。地面本身不能獨立標定 scale，候選還要用本幀非地面 XYZ 配準。prior yaw 與無向道路 PCA 提供 yaw seeds，沒有把畫面「上方」硬寫成位移。
- 8/4/2 m Gaussian NDT 以 Mahalanobis 選 cell；細階改為 exact nearest map-point 的 point-to-point Cauchy ICP，對應與殘差度量一致。surface-normal Hessian 仍獨立拒絕退化幾何，沒有放寬 coverage／收斂閘門。
- 在接受首筆 map anchor 後，以前後 rectified JPG、前一幀 PLY 深度與其 scale，做 KLT 往返檢查及 PnP RANSAC，取得 current-frame visual prior。失敗時保留 prior；沒有指定固定車速或強制前進。
- RViz 綠色顯示 map accepted，橘色顯示 current prior-only cloud 與原因，失敗幀也可檢查重疊。

各模型、Jacobian、約束與數值界限見 [ALGORITHM](ALGORITHM.md)。點雲對應與配準累加使用 CUDA；小型 4×4／7×7 系統與影像 PnP pose solve 使用 CPU。

## 真實對照與授權用量

總授權為10次 frame 處理，包含失敗：

| Run | 執行與結果 |
| --- | --- |
| `track03_test_2610242100` | 舊 solver + 去噪，1幀；移除1040/33989點，仍 `backtracking_failed`，候選scale2.907 |
| `track03_test_2610242110` | 新初始化首次1幀嘗試，因CuPy lexsort API呼叫錯誤中止，仍計入額度；失敗來源保留 |
| `track03_test_2610242114` | 修正後frame0通過；initializer scale6.7233，最終scale6.7935 |
| `track03_test_2610242119` | idx0–6共7幀；idx0–5接受map，idx6達到ICP迭代上限，保留有效visual prior |

[額度紀錄](../outputs/run/track03_test_2610242048/authorization_usage.json) 記錄使用10、剩餘0；本輪沒有啟動新 full。已停止 full 的來源 SHA 前後相符；原始資料、歷史輸出、先前 pose_trace 與使用者 `analysis_pose.py` 保留。

首幀 ego/FOV 移除843點，SOR再移除977點，保留32169點。初始化的scale3候選cost約0.09175／0.14018，選出的6.723候選cost約0.07372；它不是只靠人手指定較大scale。後續scale約6.7935，相當於3的2.264倍。camera ground height約1.783m是該幾何候選的推算量，不是獨立量測。

以同一組清理後來源點與同一XYZ地圖，只評估兩組已保存pose，不做重新配準：

| 指標 | v3 emitted initial prior | v4 frame0 emitted map pose |
| --- | ---: | ---: |
| scale | 3 | 6.7935 |
| 地面法向差 | 18.32° | 0.48° |
| 最近map點距離中位數 | 0.837m | 0.182m |
| 距離P90 | 1.699m | 0.523m |
| 3m截斷RMSE | 1.110m | 0.376m |

7幀的六次相鄰輸出，在上一幀估計相機+X方向的投影分別約0.234、0.369、0.423、0.286、0.422、0.300m，全部為正；visual PnP有276–288內點，median reprojection約1.31–1.49px。這是短段機制與幾何一致性；沒有ground truth，不能當作完整Track的pose誤差或尺度標定真值。第一個timestamp間隔11.7ms，速度欄可能因此達25.4m/s；沒有據此改寫原時間戳或軌跡。

## 驗證與圖表

最終 [35項合成／數值／CUDA／ROS檢查](../outputs/run/track00_test_2610242124/test_info.json) 全數通過，涵蓋已知scale7恢復約7.013、PnP的RDF/FLU方向與world composition、CUDA kNN、mask、退化、拒絕時保留prior、CSV/ns契約與full圖表來源保護。Matplotlib仍有既有Axes3D warning，這些2D圖可正常產生。

最終streaming kNN和真實prefix使用的matrix版，在frame0選出的點完全相同。暖機後mean-kNN函數約0.087s，matrix版約0.608s；這是該函數／點雲的比較，不是整段定位加速比。首次初始化測得約138s，包含四組幾何候選與首次模型／CUDA準備，不代表每幀都花相同時間。

[本輪分析](../outputs/run/track03_test_2610242119/analysis/README.md) 含三張PNG、三頁PDF、metrics與來源SHA；前後比較只套用已保存的pose，沒有修改定位輸出。idx6失敗保留，不因圖表較順而改寫或宣稱map成功。完整Track仍由使用者執行 [README](../README.md) 的 `./scripts/run.sh --full`。

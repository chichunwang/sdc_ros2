# Track3：v4 CUDA 點雲配準與因果 visual prior

目前程式與設定版本為 `track3_cuda_v4_denoise_geometry_visual_prior`。輸入是官方前視相機已重建 PLY；以每幀估計的正尺度 `s` 和 map pose `T=[R,t]` 將相機點轉到 LiDAR map：

\[
q_i=sRp_i+t.
\]

PLY 的 FLU 軸向有首段影像重投影幾何證據，但這不等於已確認完整車體外參、前進方向或 Kaggle 評分座標；背景分析見[首段資料稽核](49_DATA_FIRST_FRAME_AUDIT_20261024.md)。Runner 不重建影像點雲，不使用 IMU／bag、GNSS、未來影格、backward smoothing 或定位後軌跡修正。配準與點雲映射在 CUDA，小型線性系統在 CPU 求解。

## 點雲遮罩與雜訊篩除

讀取 PLY 後先以相機標定將官方前視 ego mask rectify 到影像投影座標，再移除車頭遮罩內、FOV 外、非有限或非前向（FLU `X<=0`）點。之後以 CUDA 做 depth-normalized kNN statistical outlier removal（SOR）：計算每點到 `k=16` 個近鄰的平均距離 `d`，用前向深度正規化為 `d/X`，保留不高於 `median + 3 × 1.4826 × MAD` 的點。最終使用逐查詢維持 k 個最小距離的 CUDA kernel，避免每個 block 對完整距離矩陣排序；定義仍是 exact kNN，不是 CPU 近似回退。

影像點雲隨深度變稀；若直接用固定公尺距離做 SOR，容易把遠處立面也當成離群點。除以 `X` 是依據這個透視採樣特性正規化，median/MAD 降低少數極端點對門檻的影響。官方規則 PDF 第 6 頁提到 scale calibration 與 noise handling，但沒有指定 SOR、`k=16` 或 3 倍 MAD；這些是根據點雲深度與採樣幾何採用的實作選擇，不是教師給定的參數。[官方整合規則 PDF 第 6 頁](../../midterm01/docs/Competition%20Regulations/00_SDC_2026_Midterm01_Localization_%E6%95%B4%E5%90%88%E6%96%B0%E7%89%88_20261022.pdf#page=6)

先前 v3 舊 solver 的 denoising-only frame 0 對照，點數由 33,989 降至 32,949（移除 1,040，約 3.1%），但仍以 `backtracking_failed` 拒絕，候選 scale 約 2.907；未篩選 baseline 候選約 2.902。這只能說明該次去噪沒有單獨解決舊 solver 的首幀失敗；不能據此判定 v4 的去噪效果、尺度或定位精度。

## 首段幾何初始化與尺度

在尚未接受任何 map registration 時，`GroundInitializer` 以 `init_guess.txt` 作初始姿態中心。它在局部 map 中找地面候選，使用 18 m 半徑與 0.6 m XY 網格，每格取較低的地圖點來擬合 map ground plane；同時由當前 PLY 擬合 source ground plane。RANSAC 有 256 組固定隨機種子提案索引，索引表在 CPU 產生；取樣點、平面殘差、支持數與 PCA 計算在 CUDA。map/source 平面內點門檻分別是 0.15 m 與 0.015 raw PLY units。

兩個平面基底用來建立 roll/pitch 候選及相機離地高度關係。scale seeds 為設定初值 `3` 和 `init_guess` 相對 map ground 的高度除以 source ground plane offset（高度不正時退回 3）；地面關係可產生比例候選，但因相機參考點、初值姿態及平面假設都可能有誤，不能單獨視為公尺尺度標定。yaw seeds 為 prior yaw 與附近 map ground XY 的 PCA 主軸；PCA 軸沒有正負方向，程式選與 prior yaw 較近的一側。它不能單憑道路形狀判斷車頭應往哪個方向前進。

接著只用本幀非地面幾何，做 4 自由度 `[x, y, yaw, log(scale)]` exact-nearest-point robust alignment。source 點以 0.035 raw-unit voxel 做特徵降採樣，並按方位／仰角分區降低近處密集相機射線的支配；候選 map 點取 ground plane 以上的局部幾何。每組 scale/yaw seed 都記錄 cost、coverage、迭代階段和收斂狀態；目標使用 Cauchy 距離，距離搜尋半徑依序為 3／1.5／0.75 m。這是給後續 NDT／ICP 的幾何 seed，不是獨立的最終 pose 接受判定，也不代表使用前進速度先驗。initializer 發生支持不足時，當幀會拒絕並保留 prior；seed 產生後仍由完整配準品質閘門決定是否採用。

## 逐幀影像運動 prior

設定預設 `prediction: visual_pnp`。使用前視 rectified JPG 與嚴格相等的整數 timestamp 配對；前一影像以 `goodFeaturesToTrack` 取最多 1200 點，KLT 光流追蹤至目前影像，再反向追蹤，往返像素誤差需小於 1 px。把前一幀已保存 PLY 投影到影像，GPU 為 tracked pixels 找 4 px 範圍內的 3D 深度對應；使用前一幀 PLY 點與其 pose state scale，在 CPU 執行 `solvePnPRansac` 並以內點 refine。PnP 的 RDF 結果轉回 FLU，合成方向檢查確認轉換與相對 pose 方向。

視覺 prior 至少需要 30 個內點、內點比例 0.4、median reprojection error 不高於 3 px，並受 50 m/s 平移速度與 2 rad/s 旋轉率上限檢查；這些是排除明顯錯誤匹配的 sanity bounds，不是相機運動標定或真值。只有首筆 map registration 被接受後，下一幀才啟用視覺運動 prior。遇到影像特徵不足、PnP 失敗或門檻不符，改用上一幀完整輸出 pose。沒有硬編碼的「往前」位移。

## CUDA NDT 與 exact point ICP

每幀以 prior 附近的 XYZ map 建立 local submap；移動超過 15 m 才重建 voxel models。粗配準依序使用 8／4／2 m Gaussian voxel NDT。每格至少需要 5 個地圖點且模型至少 7 格；協方差特徵值設下限避免平面方向反矩陣過大。查找相鄰 ±2 格（125 個 cell），在 1.5 voxel leaf 截斷半徑內，以 Mahalanobis 距離選 Gaussian cell，並最小化 Cauchy 加權 Mahalanobis residual。這是專案自行實作的 Gaussian voxel 最小平方方法，不是 PCL NDT likelihood 或 optimizer 的逐行移植。

1 m 細階使用 `GpuNearest` 對已降採樣的 local map 支持點做 exact nearest-point 搜尋，對應選擇與最佳化目標一致，最小化 3D point-to-point Cauchy residual；不再用 voxel normal 的 point-to-plane residual。surface normals 仍由 voxel model 獨立估計，供可觀測性閘門使用，不是細階配準殘差。每個階段先按 source voxel leaf 約 `leaf/3`（至少 0.15 m）降採樣，來源上限 12,000 點。

求解增量是

\[
\Delta=[\delta t,\;\delta\omega,\;\delta\log s],
\]

平移在 map 座標加上，旋轉左乘，scale 以正值 log increment 更新。點 Jacobian 使用 `[I, -[sRp]×, sRp]`。點對應、殘差與 normal-equation 累加在 CUDA，CPU 解 7×7 系統並執行受限步長與 backtracking。scale regularizer 以當前 seed 或前一個被接受的 scale 為參考；初始化 sigma 為 1.0，後續為 0.15，bounds 為 0.2–30。這些是數值先驗與界限，不是公尺比例的標定值。

## 接受條件與因果 pose 更新

map registration 要求最後一個 ICP 階段收斂、有效對應 coverage 至少 0.3、pose 有效、scale 有限且落在 bounds 內。可觀測性另由 fine-stage surface-normal Hessian 檢查：加上 scale prior 後消去 scale 自由度，以 normalized 6D Hessian 最小特徵值大於 `1e-6` 作退化門檻。因此早期 NDT stage 記為 `backtracking_failed`，後續 fine ICP 仍可能完成收斂；各 stage 結果都保留在 diagnostics，最後 gate 不等於每一個 stage 都必須各自收斂。

通過時直接採用候選 `(R,t,s)`。拒絕時整組候選平移、旋轉與 scale 都不寫入 pose，保留該幀開始時的完整 prior 和 prior scale；若此 prior 由有效 PnP 產生，輸出會沿用該 visual prior。舊 pose 不回寫、沒有 Kalman/NIS 第二次融合，也不做離線軌跡平移或旋轉修正。Hessian 可觀測、coverage 高或接受地圖更新，均不能單獨證明定位精度或 scale 正確。

## 診斷、輸出與驗證

`observations.jsonl` 記錄各 stage 的 cost、coverage、line-search、preprocessing、camera mask、initializer 候選與 visual PnP；`pose_trace.jsonl` 記錄 `predicted_T`、`registered_T`、`final_T`、尺度及相對 prior 的 Δ。每幀 pose 存 row-major 3×4 共 12 個數值；CSV 為 `idx,x,y,yaw`，idx 從 0 起，yaw 為 `atan2(R10,R00)` 弧度。輸出格式與 pose 可讀不代表 Kaggle 的 frame、欄位、座標或 yaw 單位契約已驗證。

真實 frame 0 短段 run [`track03_test_2610242114`](../outputs/run/track03_test_2610242114/test_info.json) 的 map registration 通過品質閘門，`map_accepted=true`。ego mask/FOV 先移除 33,989 點中的 843 點；depth-normalized SOR 再由 33,146 點移除 977 點，保留 32,169 點。4 組幾何初始化候選中，scale seed 3 的兩個候選收斂至約 2.7686／3.2688（cost 約 0.09175／0.14018）；較低 cost 候選來自 ground-height 比例 seed，初始化 scale 約 6.7233、cost 0.07372、coverage 0.8631，狀態為 stationary。後續 NDT／point ICP 通過，輸出 scale 約 6.7935；該幀記錄的 camera ground height 約 1.7833 m，並非獨立量測真值。2 m NDT stage 曾以 `backtracking_failed` 結束；最後 1 m ICP 收斂，故整幀仍通過 final gate。registration 用時約 137.78 s，含 9.91 s 去噪、四組初始化、模型準備及首次 CUDA 編譯等工作。詳細欄位見該 run 的 [observation](../outputs/run/track03_test_2610242114/observations.jsonl) 與 [pose trace](../outputs/run/track03_test_2610242114/pose_trace.jsonl)。

另有 synthetic initializer 以已知 scale 7 測試，兩個候選恢復為約 7.0132／7.0134；這支持合成幾何機制，不是實際場景準確度證明。v3 full `track03_full_2610242038` 已停止，輸出 254 幀、2 幀接受地圖更新，最後一次接受在 `idx=22`；停止前後來源 SHA 相符。這不是完成的 full run。尚無獨立真值、Kaggle 成績或完整 v4 Track3 精度驗證。成功完成 3142 幀 full 後才在 run `analysis/` 產生三組唯讀 PNG/PDF 與來源 SHA manifest；圖表不更改 pose 或 CSV。v1/v2 的歷史證據保留於 `outputs/run/`，更早首段幾何與 v2 失敗分析見[資料稽核](49_DATA_FIRST_FRAME_AUDIT_20261024.md)。

7 幀真實 prefix `track03_test_2610242119` 在 idx 0–5 接受 map，idx 6 以 `not_converged:max_iterations` 拒絕候選並保留有效 PnP prior；6 次相鄰輸出的前向投影為正（約 0.234–0.423 m），尺度由 6.7935 至 6.7551。PnP 的內點數為 276–288，median reprojection 約 1.31–1.49 px。第一個 sensor timestamp 間隔只有 11.7 ms，因此相對速度診斷可達 25.4 m/s，不能把速度欄位當成獨立車速真值，也沒有據此改寫 timestamp 或 pose。此為只有 7 幀的短段檢查，不是 full 精度驗證。

最終全套 35 項檢查通過。最終 streaming exact-kNN 與 prefix 版本在真實 frame0 選出的 32,169 點完全相同；暖機後 mean-kNN 計算約 0.087 s（matrix 版約 0.608 s），這只是該函數、該點雲的時間，不是完整 Track 加速比。對固定 pose 的唯讀比較見[本輪診斷](50_STARTUP_SCALE_NOISE_FIX_20261024.md)。

# 36｜配準與融合瓶頸：GICP 實作、跨幀模型與驗證

日期：2026-10-19（Asia/Taipei）。使用者詢問 NDT／ICP、GNSS／IMU 是否能升級，接著要求「總結出新方向後就馬上執行測試和改進」。本輪已實作兩個可隔離比較的模型，完成數學／CUDA 檢查與真實資料短測；**沒有取得新的正式精度改善證據，原最佳 1.15008／0.30488 保留**。完整 Track 與提交仍由使用者操作。

固定研究入口：[outputs/run/track00_test_2610192006/README.md](../outputs/run/track00_test_2610192006/README.md)。原目錄為 `outputs/run/track00_test_2610192006/`；這個研究總目錄包含 GICP 與最小融合原型，不需要逐一找歷史 run。

## 1. 已核對的瓶頸，與尚不能下的結論

讀取 [35 的最佳完整結果](35_MOTION_RTS_AND_GNSS_HEADING_20261019.md)，沒有重算全序列：

| 既有完整輸出的內部診斷 | Track1 | Track2 |
| --- | ---: | ---: |
| 地圖 XY 修正 RMS | 1.58 cm | 1.05 cm |
| 地圖 XY 最大修正 | 10.94 cm | 5.79 cm |
| yaw 修正 RMS | 0.0671° | 0.0409° |
| yaw 最大修正 | 0.4977° | 0.3113° |
| 到達 95% XY／yaw 上限的幀數 | 0／0 | 0／0 |
| 依現有門檻只有一個強 normal-information 方向的幀比例 | 69.18% | 91.85% |
| rank=1 幀最強 scaled eigenvector 的 Z 分量平方中位數 | 98.76% | 99.89% |
| reliable source matches／unique target 中位數，分別計算 | 475／50 | 322／31 |

實際上限為 XY 1.5 m／yaw 8°，多數幀只求解 1–4 次。因此沒有證據支持把「放寬界線、增加迭代或 GPU 算力」當作目前主要精度方向。

rank 使用既有 scaled-state、最大特徵值 5% 與絕對 1e−4 門檻，排除 prior、tangent floor、normal gain；它不是精確的不可觀性定理，也不是 covariance。資料顯示約束方向很不均衡，另外實際檢查最強 eigenvector，在 rank=1 幀其 Z 分量平方中位數約 98.76%／99.89%；這提供「目前 map 約束多數偏向高度」的直接內部證據，仍不能把它當成 score 的誤差分解。總分尚無 XY／yaw 分項，地圖修正量也不是真值誤差。

目前 `motion_rts` 分開處理 XY 常速度 KF／RTS 與 yaw bias KF／RTS，再逐幀配準；沒有用 raw gyro／acceleration 的完整慣性狀態模型，也不是 GNSS／IMU／map 聯合估計。ICP 的 prior precision 固定，對應 XY std=0.5 m、yaw std=3°；沒有沿用已校準的動態感測器 covariance。map leaf=1 m、source leaf=0.3 m，局部幾何主要由 target PCA 提供。這些是可改進的模型選擇，尚不能證明其中某一項就是全部排行榜差距的原因。

## 2. NDT、GICP、VGICP 應怎麼選

| 方法 | 改變什麼 | 本專案的判斷 |
| --- | --- | --- |
| NDT／多尺度 NDT | 用 voxel Gaussian 匹配，改變目標與搜尋地形 | 可做相同 prior／資料／範圍的對照；不能用舊 legacy 與新 guided 的差距判定 NDT 本身較差。解析度與每格支持必須匹配資料 |
| GICP | 同時建模 source／target 局部結構 | 本輪已實作有界 4-DOF 版本，直接檢查兩邊表面模型是否有用 |
| VGICP | 以 voxel 聚合局部 distributions，改善匹配計算效率 | 原理值得採用；目前 solver 本身已是數毫秒到數十毫秒，GPU 加速不是增加定位資訊的方法 |
| 細 map surfel＋空間／法線均衡 | 保存細部結構，減少同一小塊地面被大量重複計數 | 更貼近目前水平幾何不足的假說；必須先證明立面／路側目標可靠、分散且跨幀一致，不能放大少量壞對應 |

[NDT 的官方說明](https://pointclouds.org/documentation/tutorials/normal_distributions_transform.html) 描述 voxel 分布與尺度依賴；[GICP 原論文](https://www.robots.ox.ac.uk/~avsegal/resources/papers/Generalized_ICP.pdf) 使用兩邊局部平面結構；[VGICP 原論文](https://staff.aist.go.jp/k.koide/assets/pdf/icra2021_02.pdf) 說明 distribution 聚合與平行化。這些方法的 LiDAR 實驗不等於已驗證能提升本題相機重建雲的分數。

我們有固定 LiDAR 地圖與相機重建點雲，已核對的 bag 沒有連續 LiDAR scan，不能直接照搬 FAST-LIO。可借鑑其聯合估計與迭代更新觀念，但觀測模型要重建；[FAST-LIO 原作者程式](https://github.com/hku-mars/FAST_LIO) 明確使用 LiDAR scan／IMU 與相關同步、外參。

## 3. 已實作：有界四自由度局部平面 GICP

新選項為 `--geometry-mode gicp`，仍是 `sensor_guided/cuda_robust_icp` 後端，CUDA 最近鄰＋CPU 小型求解。不是 PCL 完整六自由度 GICP，也不是 NDT／VGICP；原 off／diagnostic／directional 路徑與程式預設保留。

對相同配對誤差 `e=Rp+t−q`，使用：

$$
W_i=\left[\tfrac12(C^M_i+RC^S_iR^\top)\right]^{-1},\qquad
E_{map}=\sum_i w_i\,\rho\left(\sqrt{e_i^\top W_i e_i}\right).
$$

兩邊以相同 PCA 品質條件估計局部法線，平面法向／切向的相對 variance 使用原 directional 的 `tangent_weight=0.05`。每邊各占一半 variance，使完全一致的法線退化回原 directional metric；沒有靠額外放大 gain 取得更大修正。confidence 取兩邊較小者，來源無可靠平面時保留 constant unmatched loss，沒有偷偷退回 directional。這是**局部幾何 regularization，不是校準好的相機深度噪聲**。

source covariance 會隨 yaw 旋轉，已包含 `dW/dyaw` 對 robust objective 的 gradient 項；Hessian 使用正定 GN 近似及既有有界 line search。保留原 XY／yaw／Z 界線、prior、Huber、carry、固定 IMU roll／pitch。GICP 需要正的 tangent weight，零值會明確拒絕。

新增 `gicp.csv` 保存每幀 source 平面支持、候選 paired points／unique targets；geometry.csv 仍是 mutually supported pairs 的 target-normal 診斷，不冒充完整 GICP Hessian／covariance。

### 數學、建置與真實短段

- GPU workspace 已重新 **Release** 建置；CPU 環境不動。建置前另存與已評分 SHA 相同的舊 solver binary 到本研究 `before_build/`。
- 四個 CTest 通過；新 metric 的一致法線退化、normal sign、正定性、旋轉 covariance／完整 Huber yaw gradient 有 finite-difference 檢查；已知三平面姿態擾動、界線與來源線狀退化也通過。
- 新 GICP 與既有 directional geometry 的 CUDA 機制對照通過。這些是算法／契約測試，不是實際精度分數。
- 沿用 34 事前選好的八個 48 幀場景，8×2×48＝**768 poses** 全完整；兩組相同 prior bytes、binary、raw PLY、非 geometry 參數，只有 cost model 不同。
- 八組原 directional 的矩陣／CSV／geometry／height 與舊控制組逐 byte 相同；兩條各 48 幀通用 raw-bag GICP 入口也與凍結候選逐 byte 相同，合計 **864 筆真實短段 pose**。

八段全部每方法各接受 384／384 幀，沒有新的拒絕；但 GICP 的共同 all-source clipped Euclidean map residual 只有 **1／8** 段較低，其餘較高。兩版 output XY 差 RMS 約 0.02–1.54 cm、yaw 差約 0.0002–0.0679°。它主要改變如何使用既有配對，沒有補出獨立水平量測。

plane RMSE 的配對集合不同，不能直接把它的降低當作勝出；共同 map residual 也參與了配準，不能當真值。**不升為最佳，也不按這八段調 gain／tangent weight。**

## 4. 已實作並測試：最小 GNSS／gyro 跨幀融合原型

`kinematic_graph.py` 在原 GNSS 時間建立 `[x,y,vx,vy,yaw,b_eff]` 聯合狀態，使用 sparse nonlinear least-squares，同時處理：

1. 每筆 raw GNSS XY 一次，radial Huber loss。
2. 原常速度模型的白加速度 PSD=4 m²/s³；沒有把 GPS 差分當成獨立 odometry。
3. raw gyro 積分的相對 yaw，用 quaternion roll／pitch 將 body gyro 轉成 Euler yaw rate，並估計慢變有效 yaw-rate bias。
4. 軟性的非完整運動約束：車體側向速度接近零；這是車輛模型，不是新增感測器讀值。
5. 一個弱初始 yaw prior 與 rate-bias prior；沒有再加入 course heading 因子或逐筆 quaternion yaw 因子。

`b_eff` 不是完整三軸 IMU bias 校準。這個原型**還沒接入 production fusion／map factors**，也沒有積分加速度；兩條 raw acceleration norm median 約 9.80 m/s²，必須處理重力、bias 與座標後才可拿來預測位置。quaternion 假設 body→world 時，旋轉後平均約 `[−0.09,0.04,9.80]`／`[0.06,−0.08,9.80]`；這只支持一個 convention 假說，不能代替外參校正。

新增 gyro density=0.3°/√s、effective-bias RW=0.02°/s/√s、側向速度 constraint density=0.5 (m/s)√s，兩條共用；都是本輪單一實驗模型假設，非 covariance 校準結果。初始 yaw std=30°、rate-bias std=1°/s；沒有在 holdout 上搜尋。未知 lever arm、前向軸與 sideslip 會影響非完整約束是否適用。

四個 Python 測試通過：analytic sparse Jacobian（含 robust GNSS）、非零 roll／pitch 的 body gyro→yaw-rate、已知轉彎／gyro bias、任意改動留出位置不影響 initialization／residual／fit。這些是機制驗證，不能取代真實資料。

### 原固定 GNSS 留段，沒有一致改善

沿用 34 的兩條各 12 個固定時間區塊，width=0.75 s、兩側各 0.25 s guard。模型只看保留 GNSS；測試區塊時保留 raw gyro／quaternion 作感測器輸入，沒有用被留出的 GNSS 位置決定 gate 或初始化。input timestamps／XYZ／quaternion 與原 motion_rts 研究完全相同。

| 6 個保留區塊、18 筆 noisy GNSS 的 prediction RMSE | motion_rts | 新聯合原型 | 較低區塊 |
| --- | ---: | ---: | ---: |
| Track1 | **0.13076 m** | 0.14025 m | 3／6 |
| Track2 | **0.05159 m** | 0.06147 m | 3／6 |

兩條完整 sensor-fit 與 24 次留段 fit 都回報 optimizer success；solver 按 cost／step tolerance 終止，未宣稱所有 fit 都滿足 gradient tolerance。有效 rate-bias 約 −0.24 至 −0.30°/s，是此平面模型的估計，不是實體 gyro 已校準真值。

Track2 design pooled RMSE 由 0.07376 降至 0.06860 m，holdout 卻變差；沒有只選 design 的正面結果。幾毫米到一公分的輸入差異、少量區塊，也不能用來判定真實 ATE 必定較差。**目前未通過升級門檻，不接入正式流程，也不按保留段調 noise 來追結果。**

## 5. 接下來最有價值的升級順序

**第一，查出並驗證座標／時間／量測契約，與可靠水平 map 結構同步進行。** GNSS 輸出代表天線、IMU、車體還是其他 origin？camera source→body 是否需要 lever arm？GNSS orientation 是否由位置或 IMU 產生？目前沒有足夠來源資訊。即使標頭與 bag 相差約 1 ms，也不能據此排除感測器內部延遲。不能把初始 guess 或 Kaggle 分數當標定真值。

對 map，以局部細解析度 surfel、靜態立面／路側結構、空間／法線均衡與 unique target 支持驗證，確認增加的是**獨立且可靠的水平限制**。保持原 source／prior，先做一個可解釋的粗細 map 消融；細 map cache 需新 identity、來源 SHA／leaf／crop 範圍核對，不能把既有 height crop cache 當完整定位 map。單純增加點數或放大僅有的壞立面也可能更差。這部分仍是下一階段，**本輪沒有宣稱已完成細 map 升級**。

**第二，建 GNSS＋IMU＋可靠 map factors 的聯合軌跡估計。** 本題可用較晚資料，優先考慮 batch／滑動視窗 smoother；狀態逐步升為 `[p,v,R,b_g,b_a]`，用 IMU preintegration 連接 keyframes，GNSS 及 map 殘差分別只進入一次。不要把已融合 GNSS 的 pose、其差分、course 再當三份獨立觀測，也不要同時強約束同源 gyro 和 quaternion。參考 [GTSAM 的 IMU factor](https://borglab.github.io/gtsam/imufactor/) 與 [原始 preintegration 論文](https://arxiv.org/abs/1512.02363)。

ESKF／IEKF 適合需要即時輸出的聯合狀態更新；本題離線條件下，smoother 可在視窗內重新線性化。兩者仍取決於量測品質、模型與噪聲，改 optimizer 名稱不會憑空增加資訊。現行 RTS 本來就使用未來資料，新的優點應是**共同求解狀態與跨幀觀測**，不是宣稱首次使用未來資料。[GTSAM 的 factor graph 教學](https://gtsam.org/tutorials/intro.html) 說明這種連接方式；[幾何退化研究](https://www.cs.cmu.edu/~kaess/pub/Zhang16icra.html) 說明應分方向處理弱約束。

**第三，取得獨立絕對參考後，再研究慢變 GNSS bias。** 單獨使用 `z_GNSS=p+b`，同時把 p 加常數、b 減相同常數，GNSS 與相對慣性觀測仍可不變。因此不能承諾單靠 IMU／GNSS 平滑就消除共同水平偏差；需要可靠地圖水平約束或其他絕對參考，並檢查可辨識性。

## 6. 證據與手動入口

本輪研究目錄包含 `bottleneck_audit_v2.json`（最初版保留為 bottleneck_audit.json）、`summary.json`、`ablation_runs.json`、`generic_equivalence.json`、`cpp_tests.log`、`python_tests.log`、`kinematic_tests.log`、`kinematic_validation.json`、原始 fit／state 與來源快照。沒有新完整 Track／Kaggle 成績；已評分來源與分數現記於各自full目錄的test_info.json，不另設最佳分類。

僅重現 GICP 短段，可從原 bag 通用執行，無須舊 run（新建 output）：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
python3 scripts/benchmark.py --track 1 --start-frame 905 --frames 48 \
  --profile sensor_guided --sensor-fusion motion_rts \
  --geometry-mode gicp --height-mode carry
```

完整序列仍由使用者操作；本輪證據不支持直接推薦完整 GICP 或把融合原型接入正式流程。`kinematic_graph.py`／`validate_kinematic_graph.py` 是研究工具，尚無 production `--sensor-fusion` 選項，避免把未通過的模型誤當新最佳。

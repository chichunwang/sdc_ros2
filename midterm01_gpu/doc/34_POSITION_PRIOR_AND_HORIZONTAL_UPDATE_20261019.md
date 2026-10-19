# 34｜位置先驗消融與水平觀測更新

日期：2026-10-19（Asia/Taipei）。使用者明確要求接續 [33 的完整評分退步](33_SURFACE_HEIGHT_REFERENCE_20261019.md)，先比較位置先驗，再檢查獨立水平觀測，觀測可靠後才進入聯合估計。

**本輪結論：`motion_rts` 位置候選已完成局部驗證，可交由使用者手動完整評分；沒有取代原最佳。視覺／水平地圖觀測未通過本輪融合條件，尚未實作或接入新的 joint solver。** 原最佳 1.15275／0.30576、原前相機、scale=3、原姿態、`carry`、`directional` 及 C++ solver／CUDA 都保留。未啟動完整定位或上傳。

**後續評分更新：** 使用者已手動完成兩條 motion_rts，截圖回報 **1.15008／0.30488**，升為已評分最佳。下面保留當時短測與「尚未完整」的歷史語境；最新保存與 GNSS orientation 消融見 [35](35_MOTION_RTS_AND_GNSS_HEADING_20261019.md)。

## 找本輪結果

固定入口：**`midterm01_gpu/outputs/run/track00_test_2610191627/README.md`**。

原始研究目錄：`outputs/run/track00_test_2610191627/`。只有這一個研究總目錄；短段 run 的捷徑與 log 集中在其 `short_runs/`。`outputs/run/track01_full_2610192232`／`track2` 仍指使用者上次完整測試，沒有被本輪短測更新。

- `summary.json`：留段、短段配對、水平觀測與是否可進入手動完整候選的結論。
- `prior_ablation.json`：按原輸入運動選的片段、位置先驗差異、同片段的 GNSS 留段結果。
- `fixed_time_block_validation.json`：兩條各 12 個固定時間區塊，交錯 design／holdout，各有 6 個保留區塊。
- `ablation_runs.json`：8 組配對的實際 run、來源與原始 idx。
- `horizontal_probes_verified/report.json`：8 個三視角群組、24 對影像與水平 map 的觀測檢查；`sources/`／`runtime.json` 保存開始時的程式、版本、命令與品質門檻。
- `frame_contract_audit.json`：實際 topics／frame_id／標定檔雜湊；`generic_smoke.json`／`ros_entry_equivalence.json`：通用入口與凍結候選的等價檢查。

研究證據與 raw sensor export 在被 Git 忽略的 outputs；跨機需另備份。raw export 只讀共用 bag，沒有改原資料或 ROS Python 環境。

## 1. 只換位置先驗

對照是原最佳的 `heading_rts`：位置為 5 點二次 Savitzky–Golay＋線性插值，朝向為 heading-bias KF／RTS。候選只把 XY 換成既有的常速度狀態 `[x,y,vx,vy]` KF／RTS，每筆原始 GNSS 只更新一次，影像時間只有 prediction。

兩條原 bag 的實際 sensor_samples SHA256 與原最佳 metadata 一致。候選 `sensor_prior.csv` 的 timestamp、Z、quaternion **直接複製原字串**，只有 XY 欄位替換。沒有重新重建、加外參、改 source scale、改 map 目標或高度模式。完整輸入 prior 的 XY 版本差 RMS 約 Track1 0.03919 m／Track2 0.01852 m，這是改動量，不是真值誤差。

設定沿用 `FusionConfig`：GNSS std=0.3 m、白雜訊加速度 PSD=4 m²/s³，沒有本輪參數搜尋或按 Track 改權重。這是既有模型假設，不是新量得的感測器 covariance；RTS 不能自行辨識共同慢變 GNSS bias。

### 留段驗證與防洩漏

每個驗證區塊寬 0.75 s，兩側各再移除 0.25 s guard；訓練時完全不讀該區塊的 GNSS 值，從前後觀測預測被留出的讀值。測試已驗證任意改動留出值不會改變兩個模型的 prediction，整數奈秒先相減再換秒；禁止端點無支持外推。

最初直接在「移除區塊後的測量順序」套 SG，會破壞它的等時間間隔假設，造成不公平的跨缺口 stencil。已改為**各原始連續訓練段分開 SG，再跨留段插值**。舊診斷保存為 `prior_ablation_retained_order_sg_diagnostic.json`，不拿其較大差距當改善證據。

| 固定時間區塊：6 個保留段的 pooled GNSS prediction RMSE | heading_rts 位置模型 | motion_rts |
| --- | ---: | ---: |
| Track1 | 0.16226 m | **0.13076 m** |
| Track2 | 0.15127 m | **0.05159 m** |

Track1 保留段 4／6 較低，Track2 6／6 較低；不是所有區塊都勝出。另在按運動類型選的片段，Track1 的保留直行段反而由 0.14781 升至 0.17298 m。沒有隱藏這個負面結果，也沒有因此為它調參。

這些是對**有噪聲 GNSS 讀值**的區塊預測，不是 ground truth ATE，也不是預估 Kaggle 分數。兩條路線上的時間區塊仍非獨立外部資料集，無法驗證共同 GNSS bias、未知 lever arm 或正式 body frame。

### 分散短段定位

依相同 sensor prior 的速度／yaw rate 選片段，先選後算兩個方法的殘差。每段 48 幀、至少 80% 幀滿足類別：低速 ≤2 m/s；直行 ≥5 m/s 且 yaw rate <2°/s；轉彎 >2 m/s 且 yaw rate ≥2°/s。最多每類兩段，片段不重疊、同類中心至少相隔 3 s，使用原時間分布的 25／75% 位置，不按評分或地圖殘差挑 idx。

| Track | 類型／split | 原始 idx |
| --- | --- | --- |
| 1 | 直行 design／holdout | 905–952／2286–2333 |
| 1 | 轉彎 design | 1808–1855 |
| 2 | 低速 design／holdout | 96–143／224–271 |
| 2 | 直行 design／holdout | 1045–1092／2568–2615 |
| 2 | 轉彎 design | 1379–1426 |

Track1 沒有符合本輪長度／時間邊界的低速段，兩條的轉彎均只找到一個合格分散段；保留樣本缺口，不改定義補出案例。

8 組×2×48＝**768 poses** 均完整有限、退出 0、validation 無錯；逐組核對相同 binary、相同 raw PLY SHA、相同非 prior 參數，以及相同 timestamp／Z／quaternion 字串。短窗 carry 都重新從 nominal 高度起跑，不能冒充原完整序列的高度歷史或精度勝出。

## 2. 獨立水平約束：已做觀測檢查，沒有強行融合

每個片段取相隔 14 幀的三張 raw 前相機 JPG，核對 bag／header 奈秒時間；用既有 fisheye 標定／遮罩轉 pinhole。獨立 SIFT mutual-ratio matching 不使用 GNSS／IMU／map／深度作匹配或 motion fit。

第一輪 448×294 支持不足；後續在相同場景以 896×588 增加特徵解析度，**沒有放寬品質門檻**。5-point RANSAC 的 proposal 再以 fit-inliers 的 robust epipolar least-squares 細修；已知有雜訊的三維場景測試驗證旋轉誤差 <0.2°，避免僅靠最小樣本 proposal 的不穩定解。空間保留的 feature cells 不參與 fit，只作 epipolar 檢查與獨立旋轉一致性；三對旋轉另做 cycle 檢查。

本輪門檻包括至少 60 匹配、40 cheirality inliers、20 保留匹配、4×3 grid 至少 6 個 fit 支持 cell、median rotation-compensated parallax ≥1°、保留 Sampson median／P90 ≤1／3 px、獨立旋轉與 cycle 差 ≤1°。它們是保守診斷條件，尚非已校準 covariance；未通過不表示場景在任何算法下都不可能定位。

**最終 24 對影像僅 1 對通過全部單對門檻，8 個三視角群組為 0／8 通過。** 低速例子即使有上千匹配，也缺乏可靠平移視差／cheirality；部分行駛場景有小 cycle error，仍因展布、保留誤差或長間隔支持不足而拒絕。沒有把「很多匹配」當成可用的 metric odometry。

單目 essential 的 t 只有方向，沒有公尺尺度；未用 GNSS 距離乘回去後再冒充獨立 metric 量測。未來可把方向／bearing 與未知尺度正確放入 joint model，而非一定要先有單獨 metric odometry；仍須可靠靜態對應與 frame／噪聲模型。[OpenCV 官方 API](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) 說明 essential／recoverPose 契約，robust 細修使用 [SciPy least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)。

同一批場景另用原 raw PLY（最多 6000 代表點、scale=3）、原最佳完整 pose、已核對的 1 m map／PCA，檢查 `|normal_z|≤0.5` 的可靠立面；每個 target voxel 只保留一個 source 對應，消去 nuisance Z 後檢查 XY／yaw information。每場景只有 **0–23 個 unique vertical targets**，8／8 均未通過預定支持／conditioning 檢查。資訊量是診斷，不是 covariance；不能靠放大權重補出缺少的水平資訊。保存的 map pools 尚未驗證靜態語意、深度誤差與量測獨立性，沒有直接餵回 solver。

### 為何 joint 尚未接入

兩條 bag 實際 topics 都沒有 `/tf_static`；IMU frame 為 `imu_link`，GNSS frame_id 為空。現有 YAML 是 camera-to-camera rig，尚不能核對 camera／IMU／body 的完整 transform。時間比對與 camera optical→FLU 契約有保存，沒有自行推定 lever arm 或再套一次未知外參。

因此本輪水平觀測還不符合融合前提，**沒有實作／啟用新的跨幀 joint solver**。下一階段需先取得或校準 camera–IMU／body 契約，改善靜態 feature tracks 的展布／視差及獨立保留驗證，再以正確尺度、噪聲與相關性建立 GNSS／IMU／視覺／map factors。[GTSAM 教學](https://gtsam.org/tutorials/intro.html) 可作 factor 建模依據；不能把先驗的差分再當獨立 odometry，或把含 GNSS course 的 heading prior 當成完全獨立 GNSS heading 量測。

## 3. 通用入口、檢查與下一輪完整評分

新增工具 `prior_ablation.py`（raw export／位置凍結消融／GNSS 留段）、`run_prior_ablation.py`（最多十二個 48 幀片段）、`horizontal_constraints.py`（獨立影像與 map probes）、`summarize_prior_ablation.py`（只讀彙整）。研究工具不啟動完整序列，沒有改 benchmark 的 legacy／fixed／off／carry 預設。

另修正 `sensor_fusion.py` 的單軸批次 Euler 輸入形狀為 `(N,1)`，相容 ROS 舊 SciPy 與隔離分析環境；不是算法／權重變更。ROS 兩條完整 prior 的 heading／motion quaternion、Z 與原字串解析值**完全相同**，XY 與凍結候選只差浮點量級（最大約 9.1e−13 m）。

通用入口不指定 prior-run、直接從 bag 產生 motion_rts，另在兩條各跑 48 幀：**result.txt／diagnostic.csv／geometry.csv／height.csv 與凍結 motion 候選逐 byte 相同**。因此以下完整命令與本輪被驗證的位置候選一致，不需要歷史 run 路徑。

14 項相關 Python 測試通過（含既有 sensor fusion／prior、新 GNSS 防洩漏／整數時間／片段選取、已知視覺 motion／純旋轉退化／fit-holdout 隔離／地面無水平約束）；ROS 的 5 項 sensor fusion 測試也通過。8 組配對加 2 組通用入口，共 **864 筆真實短段 poses**；C++／CUDA 未改，無須重編。原最佳 CSV SHA 再核對、不覆寫。這些都不等於正式 ATE／Kaggle 成績。

**下一個可手動完整評分的候選只啟用 motion_rts，保留 carry，不啟用 surface_tilt 或新水平 factors。** 在 Ubuntu 主機依序執行：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
bash scripts/dev.sh up

python3 scripts/benchmark.py --track 1 --full \
  --profile sensor_guided --sensor-fusion motion_rts \
  --geometry-mode directional --height-mode carry

# Track1 成功後再執行 Track2。
python3 scripts/benchmark.py --track 2 --full \
  --profile sensor_guided --sensor-fusion motion_rts \
  --geometry-mode directional --height-mode carry
```

各自另建唯一 run、更新 `outputs/run/track01_full_2610192232`／`track2`；成功完整且 validation 無錯後，提交檔仍為 `diagnostic.csv`。完整定位、上傳與分數回報由使用者執行，本輪沒有代為啟動。若新總分不勝原最佳，就保存負面結果，不以留段 GNSS 改善替代正式判斷。

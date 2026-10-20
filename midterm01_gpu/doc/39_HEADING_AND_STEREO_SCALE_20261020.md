# 39｜向 Track1 0.6 推進：航向模型與原前相機 stereo 尺度

日期：2026-10-20（Asia/Taipei）。本輪目標是降低 Track1 的正式評分至 **0.6 或以下**。這是開發目標，**尚無本輪 Kaggle 分數，不能宣稱已達成**。目前已評分最佳仍為 motion_rts＋directional＋carry、source scale=3，Track1 **1.15008**／Track2 **0.30488**。

**後續已評分更新**：使用者完成 `track01_full_2610200142`，回報Track1 **1.15167**，比最佳增加0.00159／約0.138%，未升為最佳；3011筆本機核對通過，沒有獨立核對Kaggle上傳SHA。下文保留這輪開發時的歷史紀錄，後續分析與現在的full指令請看 [40｜航向來源單項測試](40_SCORED_SCALE_FAILURE_AND_POSE_YAW_20261020.md)，勿把本文件原full命令當成最新推薦。

依使用者最新約定，有可解釋的改動及必要程式／輸出檢查後即提供 full 指令，讓使用者實際評分；本地尚無真值不阻擋完整實驗。完整定位及 Kaggle 提交仍由使用者手動執行。本輪已把新航向模式接入一般 benchmark；來源尺度沿用現有 `--ply-scale` 介面，確實進入定位 solver，並非只產生診斷報告。

## 1. 這輪改什麼，為什麼

目前 map XY／yaw 更新很小，直接換 ICP cost 的歷史短段也未一致勝出。[38](38_SOURCE_RIGIDITY_AND_SCALE_20261020.md) 的跨幀檢查提示來源公尺尺度不一致，但先前只靠 GNSS motion 估尺度，無法區分來源形變與相對 pose／外參問題。本輪並行處理兩件可分開驗證的事：

| 改動 | 要檢查的假說 | 實際介面 |
| --- | --- | --- |
| 已知 rig baseline 直接核對原前相機 PLY | scale=3 讓來源物體距離偏近，限制水平地圖結構的對應 | `--cloud-source front --ply-scale 7.404260939740923` |
| 轉彎／加速相容的航向 block model | GNSS 直線回歸方向與平均 IMU yaw 的時間權重不同，可能混入模型偏差 | `--sensor-fusion turn_heading_rts` |
| 直接置換 yaw 的來源消融 | 來源未確認的 GNSS pose yaw 是否比原 IMU＋慢偏差模型適合本題 | `--sensor-fusion pose_yaw` |

這不是用排行榜挑倍率；scale 只由預先固定 fit 場景的 sparse stereo 決定。`pose_yaw` 不把 GNSS yaw 當成第二份獨立因子乘進 IMU 信心，而是直接更換 yaw 來源，保留原始 IMU tilt 與同一份 motion_rts XYZ。

## 2. 原 PLY 的 stereo 尺度：與三相機模型倍率分開

新增 `scripts/audit_front_stereo.py`，對原始 PLY 的 RGB 找 virtual→rectified front 紋理映射，再把 front／side 影像配對以 YAML 的已知 baseline 三角化。紋理映射以固定 4×3 空間 cells 留出；只查找可見的原 vertex index，不插值或生成三維深度。stereo 只保留正深度、Sampson／reprojection≤2px、視差角≥2° 的幾何對應。

16 個固定分散場景，10 個通過 RGB projection 留出；動態 header 補償後有 **110 個**原 vertex anchors，static 則為 77。兩相機同時看見同一原 vertex 最多一票；場景內取 log ratio median，再對 fit 場景等權取 median，避免多點場景支配倍率。

四個足夠支持的 fit 場景為 Track1 idx925／1054／2709、Track2 idx1154；三個 holdout 為 Track1 idx2306／1956、Track2 idx330。fit／holdout 由各 Track 事前清單的 ordinal 交錯決定，不依深度結果選擇。

估計原始座標到 metric 的共用倍率為 **7.404260939740923**。三個 holdout 場景的相對尺度偏差 median／P90 為 **11.18%／14.26%**。這個值直接取代 `3`，不是再乘上 3，也沒有移植之前三相機 MapAnything 的倍率。

限制仍然重要：足夠支持的 fit 場景只有 4 個，且只有一個來自 Track2；anchors 僅支持約 2–20m，沒有 20–60m 支持。近／中距離的描述統計不同，仍有深度形變，不能把單一倍率說成完整校準。

已知 baseline 提供獨立物理長度，但 motion compensation 仍使用 GNSS／IMU prior 的 header 間相對運動，camera→body lever arm 尚未驗證。因此本輪是**有 stereo 依據的固定尺度實驗**，不是完全獨立於 GNSS 的 metric calibration。

static／motion 比較必須使用相同 anchors。例如 Track1 idx925 的共同 3 點 range ratio median 為 0.9462；不能把兩個不同保留集合的尺度中位數 9.965→7.035 全歸因於同步。

完整來源／方法／原點對應及 SHA 見 `outputs/run/track00_test_2610200131/front_stereo/`；三角化檢查保留錯配、移動物體與場景覆蓋限制，沒有讀 map posterior 或 neural depth。

## 3. 新航向模型與來源置換

### turn_heading_rts

在每個原始 2 秒 GNSS block，直接擬合：

```text
p(t) = c + R(b) ∫ (v + a·(s − t_center)) h(ψ_IMU(s)) ds
h(ψ) = [cos ψ, sin ψ]
```

`c` 是區塊起點平移，`v,a` 是車速及加速度，`b` 是該區塊的 map-Z 航向修正。積分用原始 IMU 時間，GNSS 每筆只進擬合一次；不積分 accelerometer。這讓方向與位置使用同一時間模型，也能在適用範圍內利用轉彎資料。

`v,a,c` 作 nuisance，從完整局部 Jacobian 消去後取得 bias 的 model uncertainty，再沿用原 `bias_rts`。這組 factors **取代** 原 course，不與同源 course 重複計數。兩條 Track 共用同一組設定：沿用 GNSS std=0.3m、2s block、最小速度5m/s、bias RW=0.12°/√s；新增局部轉彎上限10°/s、span20°、殘差3σ及2m lever-arm uncertainty allowance。

2m 是未量測外參下的一個實驗性不確定度假設，以約 `lever_arm_std × angular_rate / speed` 放大轉彎的不確定度；沒有估出或套用某個實體 lever arm，metadata 明確記錄數值與限制。

可用 blocks 從原47／48增加為49／51。兩條完整**先驗準備**的 yaw 相對最佳改動 RMS 僅 **0.00376°／0.02853°**，24 個固定留段對 noisy course／pose yaw 的預測幾乎相同。這個模型修正了時間權重與轉彎適用性，但**沒有證據支持單靠它把1.15降至0.6**。

### pose_yaw

直接讀 `/localizer_node/gnss_pose` quaternion 的 yaw，unwrap 後在原 GNSS 時間上插值；端點明確限制≤0.3s角速度外推。保留當幀 IMU SLERP 的 roll／pitch、motion_rts XY 與原 Z，不沿用原 heading bias smoother，也不宣稱新的 heading covariance。

兩條完整先驗的 yaw 改動 RMS 為 **2.11719°／1.82886°**。來源與是否由位置差分／上游融合產生仍未知，尤其 Track2 低速初段可能受影響。因此把它當成來源消融，而不是宣稱更可靠的獨立 heading。

IMU 插值仍採用 [SciPy 的 Slerp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Slerp.html)。`PoseStamped` 與帶 covariance 的 pose 是不同訊息型別，不能從 orientation 有值推論已提供校準不確定度，見 [ROS geometry_msgs](https://index.ros.org/p/geometry_msgs/)。

## 4. 為何不只靠平滑追 0.6

本輪另外檢查固定原 IMU heading、訓練 GNSS slow bias 的條件 CV＋側向速度模型；原24個留段中，Track1／2 holdout noisy-GNSS RMSE 分別由 **0.13076→0.13950m／0.05159→0.06183m**。沒有接到正式 fusion；這個負面結果保留於主研究 run 的 `directional_motion_audit.json`。

對實際輸出而言，若分數為 `RMS_XY + 0.2 RMS_yaw_deg`，反三角不等式給出：

```text
|S_new − S_old| ≤ RMS(ΔXY_output) + 0.2 RMS(Δyaw_output_deg)
```

這只適用實際 output。先驗變動可以作規模估計，不能代替 solver 後輸出，也不能保證分數下界。它提示：幾百分之一度的 prior 改動不足以解釋0.55的分數差距；需要有效地圖水平觀測、來源幾何改善，或有依據的較大航向改動。

## 5. 機制、短段及完整入口

本輪沒有改 C++／CUDA solver，無須重編；新 Python 模式透過原通用 benchmark 從 bag 生成 prior，依然使用 `sensor_guided/cuda_robust_icp`、directional、carry。

已保存以下驗證，短段數值與原控制的詳細比對見主研究 run 的 `short_summary.json` 及後續補充：

- 新 integrated-heading 9 項機制測試，主機／ROS 各通過。
- 新 direct-yaw 3 項機制／整合測試，主機／ROS 各通過：branch cut、明確端點外推、無效 quaternion／時間覆蓋拒絕、XYZ與 tilt 保留、prefix相容。
- 原 sensor fusion／GNSS heading／prior 12 項回歸測試，主機／ROS 各通過。
- 原前相機 stereo 7 項機制測試通過：已知baseline、錯配／低視差、可見vertex、深度邊界、空間與場景holdout隔離。

合計 **主機31項／ROS24項**；這是機制與回歸測試數，並非31種精度比較。

### 已完成的真實短段

使用原八個48幀 motion windows，固定同 binary、同原始 PLY SHA、同 XYZ／timestamp 字串及非消融 solver 參數：

| 方法 | source scale | 短段／poses | 相對本輪原設定，各段最大 XY／yaw RMS差 |
| --- | ---: | ---: | --- |
| motion_rts控制 | 3 | 8／384 | 0／0，並與歷史矩陣／CSV／geometry／height逐byte相同 |
| turn_heading_rts | 3 | 8／384 | 1.03cm／0.0744° |
| pose_yaw | 3 | 8／384 | 2.07cm／4.2126° |
| motion_rts、只換stereo尺度 | 7.40426 | 8／384 | 5.28cm／0.4362° |
| turn_heading_rts＋stereo尺度 | 7.40426 | 8／384 | 5.41cm／0.4507° |

共 **40組／1920筆定位** 全部完整、有限、退出0；32組改動版本的CSV皆與原控制不同。沒有把先驗全序列準備算成新增定位。初始摘要工具因直接比較 doc35 舊 binary SHA 而停止，原因是歷史 binary 早於既有 GICP 建置；本輪控制與所有改動版本實際使用同一個現行 binary。已改為本輪同 binary 配對，同時另外要求控制四項輸出與歷史逐byte相同。最初失敗log保留，沒有放寬結果回歸。

單換尺度使原資料幾何支持增加。例如 Track1 905起48幀，原rank1有45幀，新尺度rank2／rank3為17／31幀；不同target中位數50→199.5。轉彎1808起，原rank1／2為31／17，新尺度rank2／3為10／38，不同target127.5→392。rank依原data-only scaled-information門檻，仍不是已校準covariance或ATE；新的點數／對應集合也不同，不能從這些數字宣稱分數改善。

另在兩條各48幀，以不指定歷史prior／cache的通用入口，實際從原bag與原map執行新主命令，共 **96筆**，完整性通過。與配對入口比較，Track2的TXT／CSV／geometry／height逐byte相同；Track1 XY完全相同，最大矩陣元素差5.9×10⁻⁸、yaw差2.68×10⁻⁶°，height bytes相同，其餘文字bytes不同，保留為數值／序列化差異，不宣稱全部逐byte相同。**本輪新增短段定位合計2016筆**。

新尺度後另查八組三幀、24個水平surfel觀測；20個完整覆蓋舊研究fine-cache搜尋範圍，4個覆蓋不足者另行標記。完整覆蓋者依原品質條件仍未接受，8組均未通過temporal gate。這只是既有cache下的診斷，不把未覆蓋者當作完整map驗證；沒有為此放寬門檻或接入新的joint因素。

這輪 full 主命令將固定 stereo 尺度與 turn-aware heading 一起測試；已另做兩項單因素短段對照以協助解釋，不需要先宣稱它一定勝過最佳。命令從原 bag／map 建新 output，不依賴歷史 prior／研究 cache：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
bash scripts/dev.sh up

python3 scripts/benchmark.py --track 1 --full \
  --profile sensor_guided --sensor-fusion turn_heading_rts \
  --cloud-source front --ply-scale 7.404260939740923 \
  --geometry-mode directional --height-mode carry

# Track1完成並核對後，再執行Track2。
python3 scripts/benchmark.py --track 2 --full \
  --profile sensor_guided --sensor-fusion turn_heading_rts \
  --cloud-source front --ply-scale 7.404260939740923 \
  --geometry-mode directional --height-mode carry
```

結果會存入新的 `outputs/run/track01_full_YYMMDDHHmm`／`track02_full_YYMMDDHHmm`，同分鐘自動加碰撞序號。看 `validation.json` 的退出碼、完整有限值、錯誤及3011／3299筆後，再提交該 run 的 `diagnostic.csv`。CSV 內容是否不同與正式分數是否改善分開記錄。

若要單獨判斷尺度的影響，上面 fusion 改回 `motion_rts`，其餘保留。若要測 GNSS pose yaw 來源本身，使用 `--sensor-fusion pose_yaw --ply-scale 3`；不要把這三種來源一起視為三份獨立航向量測。

暫不在新尺度命令加 `--horizontal-refinement surfel_joint`：該分支目前明確限制原 scale3／motion_rts 契約。本輪先讓 source 尺度實際影響原 C++ 配準，不放寬 map factor 的品質檢查。

原已評分最佳重現仍為 `--sensor-fusion motion_rts --ply-scale 3`；預設 profile／fusion／geometry 仍是 legacy／fixed／off，沒有偷偷改成新實驗。

## 6. 本輪證據位置

主研究：`outputs/run/track00_test_2610200127/`，含參數、raw priors、短段命令／log、控制組與版本差異、失敗方向模型、機制／回歸測試及原最佳 SHA 前後核對。

航向積分模型與24留段：`outputs/run/track00_test_2610200130/`。原前相機 stereo 最終來源／深度錨點：`outputs/run/track00_test_2610200131/front_stereo/`。早期 stereo 的0129／0130_0002保留原始過程；最終原始圖片／bag／PLY／標定／index／程式與最佳五檔 SHA 均已重讀核對。

研究 run 裡的先驗全序列準備不等於完整定位；本輪沒有自行啟動完整 Track 或提交。更新 Kaggle 成績時，另記實際 full run、參數與 CSV SHA；0.6 只在正式結果支持後才記為達成。

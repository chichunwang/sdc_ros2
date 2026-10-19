# 33｜用來源／LiDAR 平面建立獨立高度參考

日期：2026-10-19（Asia/Taipei）。接續 [32｜回授與 nominal anchor](32_HEIGHT_DRIFT_ANALYSIS_20261019.md)。使用者確認 LiDAR 固定、相機雲上下浮動。固定 nominal GNSS 高度雖能抑制漂移，卻未一致改善貼合，因此本階段改用幾何參考，仍保留原 carry 最佳。

**後續完整評分更新：Track1 `surface_tilt` 為 1.15519，較原最佳 1.15275 增加 0.00244（0.2117%），沒有升為最佳。** 兩條完整定位已由使用者手動完成，本輪只讀既有輸出；Track2 尚未回報此候選分數。原短段章節保留當時實測脈絡，最新完整核對與後續方向見下節。

## 完整評分失敗與下一步（使用者後續回報）

分數來自使用者 2026-10-19 提供的 Kaggle submission-history 截圖，同圖另有三相機 1.15346、原最佳 1.15275。local candidate 依本輪執行上下文對應至 `track1_...height_surface_tilt_...2610191555`；截圖沒有上傳 bytes 的 SHA256，未獨立下載核對，不能把本機 hash 宣稱為 Kaggle 已驗證 hash。

兩條 candidate 分別位於 `outputs/run/track01_full_2610192232`（`...2610191555`）與 `track2`（`...2610191607`）。本輪完整 CSV／矩陣／rotation／idx／整數時間與原最佳逐筆檢查通過，3011／3299 筆、退出 0、validation 無錯；**兩條 frames.csv 與 sensor_prior.csv 均與各自原最佳 bytes 完全相同**。這支持比較候選高度／姿態模型的影響，不把 prior 的變動當成原因。

| Track1 全序列指標 | 原最佳 carry | surface_tilt |
| --- | ---: | ---: |
| Kaggle（越低越好） | **1.15275** | 1.15519 |
| pose Z 範圍（m） | 4.860832 | **0.915039** |
| 最大相鄰 Z 改變（m） | 0.325699 | **0.175606** |
| XY 相對相同 sensor prior 的 RMS（m） | 0.016618 | 0.016074 |
| yaw 相對相同 sensor prior 的 RMS（°） | 0.070289 | 0.077359 |

上述 Z 範圍／相鄰差是軌跡形狀診斷，**不是相對真值的高度誤差**。新舊版本 XY 差 RMS=0.019449 m、yaw 差 RMS=0.090457°、Z 差 RMS=0.663216 m。候選同時估幾何 roll／pitch，並非「只改 Z」；最大 tilt 約 14.04°。能說這整套候選在這次評分退步，不能從單一總分分離 Z、tilt、XY 與 yaw 各自的貢獻，更不能斷言退步全來自某個 idx。

提交只有 `idx,x,y,yaw`。分析程式另在記憶體只交換保存矩陣的 Z，證實所有提交值完全相同；它沒有建立新提交檔、沒有評分。**純粹讓 RViz Z 穩定不會改變提交內容**，只有經配準耦合改變 XY／yaw 才可能影響分數。局部表面配準較好也不能取代正式精度驗證。

兩版完整 geometry 的 normal information rank=1 分別為 2116／2122 幀，rank=2 為 799／813 幀；整體仍以少數方向為主。消去 nuisance Z 後，XY／yaw 的 normal-information／prior 對角中位數分別約 `[0.00484,0.00858,0.00579]` 與 `[0.00591,0.01068,0.00744]`。此量不含 tangent floor／normal gain，也不等於實際 solver Hessian 或 covariance；不能將它解讀為精度比例。它與兩版實際只有厘米級 XY prior correction 一起，支持「高度改動沒有帶來很強的獨立水平約束」的判斷。

下一輪應保留原前相機／已評分姿態設定，按下列順序隔離問題：

1. 核對時間與 source／body／GNSS／IMU frame。既有 bag/header 實測差小於 4.8 ms，不無據假設大延遲或以排行榜擬合固定 offset；未知 lever arm 需要標定／官方依據。
2. **先做位置先驗模型的單一消融**：原始 GNSS 的常速度 KF／RTS（既有 `motion_rts`）對照目前 Savitzky–Golay＋線性插值；heading、來源、尺度、姿態／carry 及 map 模型保持相同。噪聲／process model 需用 GNSS innovation、獨立留出的時間區塊、轉彎／低速與邊界行為檢查，不用地圖殘差或總分盲選參數。GNSS holdout 仍不是定位真值，也不能辨識 GNSS 的共同慢變偏差。這是低成本候選，不保證改善。
3. 若要增加獨立資訊，優先驗證跨幀靜態特徵的相對運動、可靠立面／路側結構的 XY／yaw 約束，再做含 GNSS／IMU／相對運動／map factors 的聯合估計。先驗重複作差不能冒充獨立 odometry，同一張雲或高度平面也不能當成另一筆獨立真值；重建尺度與深度形變須一併檢查。只在可觀測方向採用可靠資訊的原理見 [Zhang 等人的退化分析](https://www.cs.cmu.edu/~kaess/pub/Zhang16icra.html)，軌跡與觀測的 factor 建模見 [GTSAM 官方教學](https://gtsam.org/tutorials/intro.html)。這些是後續設計，沒有因此宣稱已實作或精度勝出。

固定分散片段與未用來設計模型的場景，涵蓋直行、轉彎、低速與弱幾何；一次只比較一項改動。局部品質檢查通過後，由使用者手動完整定位／評分。此輪沒有改 solver／prior／預設、沒有產生另一份候選或啟動新定位，原最佳不變。

證據集中在 [track00_test_2610191615](../outputs/run/track00_test_2610191615/README.md)，包含 [Track1 全序列配對](../outputs/run/track00_test_2610191615/track1.json)、[Track2 本機配對](../outputs/run/track00_test_2610191615/track2.json)、[分數來源與綁定限制](../outputs/run/track00_test_2610191615/reported_score.json)。本機 candidate CSV SHA256 分別為 `06f4d0fcb6a1f3213e607f4da748a64bbde0b43d8dff70bbde8ca0500f843bac`／`53f49828a8fdbc8ad922b3a53c14330c67c37eaae64be62aa50f296283f9b953`。分析工具為 `scripts/analyze_height_score.py`，依指定 baseline／candidate 只讀現有完整結果，輸出使用 exclusive create 保留既有報告。

## 證據與假說

前階段的 Track1 原始 idx 0–255，pose Z 範圍 4.458 m；9 個固定 probes 的 source 下方平面相對原點截距只變約 0.213 m，同一視野的 LiDAR 下方平面截距約 107.921–108.261 m，範圍約 0.340 m。這不支持將 pose 的全部變動解釋成道路起伏；也解釋為何直接拉回常數 GNSS 高度會留下高度差。

假設兩邊選到的是同一個局部路面，可寫成：

```text
source（只乘當幀 IMU R，不加 t）：z = a_s f + b_s l + c_s
map（以當幀 sensor XY 為原點）： z = a_m f + b_m l + c_m
高度參考 h_t = c_m − c_s
```

`f,l` 是依當幀 yaw 定義的前向／側向水平座標；map 的 `c_m` 使用全局 Z。未知的 source 原點離地高度已存在 `c_s` 中，沒有另填相機安裝高度，也沒有縮放 baseline／深度或改 roll／pitch。它估計的是 source→map 平移，不是已確認的車體離地高度。

這是局部平面的截距模型，不是跨距離的完整形變修正。兩邊法向不同時，只平移 Z 無法同時對齊整張平面；模型會限制法向差，仍需保留傾斜／深度的不確定性。

## 實作與品質保護

新增 `--height-mode surface`，僅支援原 front PLY、sensor_guided／cuda_robust_icp。初期短段對照明確提供定位用的 `--map-cache` 與分開的 0.3 m `--height-map-cache`；後續已補通用入口：不指定舊快取時，定位自行讀原始 map 並做 1 m voxel，高度 map 則從本輪 prior 範圍在新 run 內建立。使用既有隔離 ML venv 的 numpy／scipy，ROS Python 不變；不是重跑 MapAnything。

`prepare_surface_height.py` 在定位前逐幀建立 `height_reference.csv`：

- 同原 prior XY／rotation／整數時間，核對來源 PLY、map／cache 身分。sensor prior 本身不改，fusion 診斷與 XY／yaw 都保留。
- 高度 map 先取可靠 local PCA 表面，再進行下方平面 RANSAC；不將大量牆面／混合 voxel 當成路面支持率的分母。PCA 使用既有 SurfaceMap 的鄰域、品質／展布規則，法向坡度上限 0.35。
- 前方 1–15 m、左右 ±5 m；來源相對 Z −4 至 −0.3 m，map 相對 nominal Z −6 至 −0.3 m。200 次固定 seed RANSAC，垂直殘差 0.15 m、坡度範數 ≤0.35、平面空間展布檢查。
- 兩邊各至少 20 個 inliers、支持率 ≥35%、二維空間展布及原點截距外推條件檢查，法向差 ≤10°（surface_tilt 則另估有界姿態）。若存在截距差 ≥0.4 m、支持數達主要平面 80% 的競爭平面，拒絕作 reference。
- 缺少幾何的時間缺口最多 1 s、首尾最多 0.3 s；超出就失敗並保存診斷，不靜默改用 GNSS。先插值，再以真實 timestamp 的 ±0.25 s 對稱 median 抑制單幀高度尖峰，屬離線處理。
- C++ 以這份 `h_t` 作絕對高度 reference，使用 32 的總 offset 正則化／±1.5 m 邊界與不重複沿用 dz。每幀 `height.csv` 同時保存 nominal GNSS Z、reference Z、prior／output Z 與兩種 offset。

上述門檻是候選模型的幾何範圍與保護，沒有宣稱已校準成真實感測器噪聲。RANSAC 的垂直殘差不是法向殘差；取樣／cache 及來源形狀會影響估計。對稱 median 會抑制很短的真實高度特徵，不能當線上即時方法。

**下方平面仍不是語意路面真值**：屋頂、不同道路層、動態車輛、遮擋及錯誤重建仍可能通過部分幾何檢查，競爭平面檢查也不能保證發現所有另一層。來源／車體／GNSS 外參與正式 frame 契約仍未由本輪完成校準。

## 獨立的細地圖與已遇到的失敗

定位仍用原 1 m map cache。先前直接在此 cache 做 reference，Track1 前段僅 39／256 幀通過；加上可靠 PCA 後僅 14／256，首尾支援缺口超過上限，兩次都明確失敗，沒有產生假定高度或定位輸出。失敗證據保留在 `track00_test_2610191453`／`track00_test_2610191456` 及各自 run。

接著從原 map 建獨立 0.3 m 高度快取，與來源 input voxel 的尺度相同，以減少 1 m voxel 混合不同表面。直接全圖下採樣時，實際捕捉到 PCL 的 integer grid overflow 警告：工具退出 0 卻原樣輸出全部點。因此新增 `map_modified` 的可選 `crop_min/crop_max` 與 overflow 明確失敗檢查，避免把未下採樣結果當細快取。

新快取取完整 sensor XY 路線包圍盒 ±25 m，以及 nominal Z 的 −6 至 −0.3 m 高度帶；這正覆蓋模型搜尋範圍，並非調整原定位 map。來源 SHA256、範圍、leaf 與重現 params 保存於 `outputs/run/track00_test_2610191459/`；preprocessor 拒絕範圍不足的 cache。

| 快取建置 | Track1 | Track2 |
| --- | ---: | ---: |
| 原始點數 | 59,531,846 | 50,881,350 |
| 明確範圍 crop 後 | 13,912,850 | 19,070,279 |
| 0.3 m voxel 後 | **872,530** | **825,583** |

未成功下採樣的舊檔標記於 `track00_test_2610191458/not_a_voxel_cache.json`，沒有建立合法 identity 或拿來定位；本輪產生的兩個大型無效中間檔已清理，header／大小／SHA256 保存在同目錄 `rejected_outputs.json`，沒有刪除原始 map。新工具的 overflow 拒絕、已知 crop 保留兩點、空 crop 拒絕三個檢查都通過。

## 實測與結果

最終 `surface_tilt v2` 在六個預定片段全部完成；下表列出改善與退步，不以實作完成宣稱正式定位精度勝出。

### 高度候選揭露了法向問題

0.3 m 細 map 上，純 `surface` 在 Track1 前 256 幀有 218 幀直接幾何支持，短缺口通過原限度補間，定位完整。Z 範圍由原 4.458 m 減至約 0.357 m；共同 probe NN RMSE 由 0.9629 降至 0.9000 m，但來源平面擬合區域外的 probe RMSE 由 1.0866 增至 1.1107 m，不能說所有幾何都改善。

同模型在原 idx 1003–1098，僅 4／96 幀通過，正常停止而未定位。細 map 的平面支持已達 90% 左右，但其側向坡度約 −0.175；來源平面的側向坡度約 +0.03～+0.05，前向坡度約 −0.07～−0.08，法向不一致超過純 Z 模型允許範圍。這是「單一 Z 無法修正傾斜」的實際反例，沒有放寬純高度模式的門檻強行通過。

### `surface_tilt`：固定 yaw 的有界幾何姿態候選

另外明確增加 `--height-mode surface_tilt`，使用同一來源／map 平面、品質／競爭層保護、細 map 與時間處理。此模式用平面估計 roll／pitch，保留當幀 sensor yaw，不將道路轉成水平：

```text
n_raw = R_imuᵀ n_source_rotated
固定 yaw，求最接近原姿態的 roll／pitch，使 R_reference n_raw = n_map
完整 rotation 相對原 sensor prior 的角差 ≤15°
d_source = 原來源平面的有號原點距離（旋轉不變）
c_source_corrected = d_source / (n_map)_z
h_t = c_map − c_source_corrected
```

候選 reference 的 roll／pitch 也採 ±0.25 s 對稱 median，之後再次核對角度上限。平滑後不保證每幀法向完全重合。C++ 讀取額外 quaternion，核對單位長度、相對 sensor prior 的 ≤15° 角差、yaw 不變；4-DOF 細修保留這份 reference 的 roll／pitch，仍可小幅修正 XY／Z／yaw。

15° 是本候選允許修正的模型範圍；初始矩陣差約 10°、局部法向差約 12–13°支持測試這種機制，沒有宣稱它是已校準外參誤差或感測器噪聲。兩條使用相同設定。這是在 source→map 配準中估計姿態，不是將某 Track 的初始 roll 差直接套成所有幀的固定外參。

評估另用**依原 sensor rotation 固定選定、未參與來源下方平面擬合**的相同 probe 點，檢查改善是否只限於地面。它們仍來自同場景／同 map，不是獨立定位真值或跨資料集泛化證明。

### 原點截距的可觀測性保護

第一版 surface_tilt 的 Track2 idx 2199–2294 因後段前向展布不足 3 m 而拒絕，僅 82／96 有支持，尾端缺口超過 0.3 s。該段仍有約 50 個近處平面點，固定跨度沒有直接描述 source 原點截距是否可觀測，因此 v2 以平面擬合的幾何條件取代固定 3／1 m 門檻：

```text
A_i = [forward_i, lateral_i, 1]
N_eff = min(N,20)
origin_proxy = 0.15m × sqrt( [(AᵀA)^−1]_cc × N/N_eff )
origin_proxy ≤ 0.3m
```

0.15 m 取自平面 inlier fitting scale；0.3 m 是現有 map 殘差尺度。20 點有效支持上限避免高密度相關點被當成任意多獨立量測。這是**保守的幾何條件代理值，不是已校準標準差／covariance**；先前二維 covariance 展布與至少 20 點的保護仍保留。

同樣只有 2 m 前向跨度，位於原點前方 1–3 m 的平面能合理估計截距；12–14 m 的窄帶往原點外推則可能很差。新增已知平面測試驗證前者可接受、後者被拒，而不是僅為通過某個 Track 調低跨度門檻。v2 對兩條所有原定短段重新驗證，另存唯一 run；先前拒絕及 v1 結果保留。

### 最終 v2 配對結果與限制

兩條各三段，六組配對共 **1792 poses**（carry 896＋v2 候選 896）。carry 重用上一階段已保存的同 binary 對照片段；分析工具逐組核對 binary bytes、原始來源 manifest、sensor prior 值及其餘參數相同。候選全部退出 0、有限完整、validation 無錯誤；**859／896** 幀直接得到合格平面，其餘 37 幀在原時間上限內補間，沒有放寬 gap 門檻。

每段取 9 個事先固定的時間分位，共 **54 個共同 probe 場景**。以下 NN 距離仍 capped at 2 m，使用原定位／RViz 的 **1 m 全表面 map cache**，不是只在新的 0.3 m 地面帶上評估。來源擬合區域外的點也用相同原 sensor rotation 定義，避免不同候選挑到不同點。

| Track／原始 idx | Z 範圍，短段 carry→v2（m） | 全 probe NN RMSE：短段 carry／v2／原完整最佳（m） | 擬合區域外 NN RMSE：v2／原完整最佳（m） |
| --- | ---: | ---: | ---: |
| 1／0–255 | 4.458 → **0.613** | 0.9629／**0.7807**／0.9629 | **0.9528**／1.0866 |
| 1／1003–1098 | 0.874 → 0.620 | 1.1288／0.6923／**0.6608** | 0.9058／**0.7228** |
| 1／2007–2102 | 1.604 → **0.232** | 0.7903／**0.5827**／0.8005 | **0.7318**／0.8523 |
| 2／0–255 | **0.347** → 0.851 | **0.5706**／0.6384／**0.5706** | 0.6289／**0.6285** |
| 2／1099–1194 | 1.162 → 0.450 | 0.6489／0.6119／**0.6072** | 0.9397／**0.9003** |
| 2／2199–2294 | 0.880 → 0.213 | 0.5184／0.4923／0.4929 | 0.5885／**0.5811** |

相對同起點的短段 carry，5／6 段的整體與擬合區域外 NN 殘差都降低；**Track2 開頭退步**，Z 範圍及最大相鄰 ΔZ 也增加（0.0419→0.1982 m）。相對保存的**原完整最佳**，只有 3／6 段整體 NN 較低、2／6 段擬合區域外較低。中途短段的初始化歷史不同，不能拿其 carry 的較差初始化當成超越完整最佳的證據。

Track1 前段最大相鄰 ΔZ 由 0.2324 降至 0.1412 m，其他兩段由 0.1366／0.1949 降至 0.1080／0.0709 m。兩條各段的幾何 tilt 修正上限分別約 10.18／14.04／12.43°，2.97／2.56／3.84°，不是一組全 Track 固定外參。

上述都是內部幾何／版本差異，沒有真值，不能稱為 ATE 改善。此候選證實「改變高度與傾斜模型可以改善部分浮動／貼合」，也保留「未對所有場景勝出」的限制。**預設仍 carry，原最佳 CSV／分數不變；沒有自動完整 Track 或新 Kaggle 分數。**

![同原始雲與全局基準的側視對照](../outputs/run/track00_test_2610191515/cloud_overlay.png)

側視圖用同一 raw PLY／scale、1 m LiDAR cache、固定 sensor XY 與全局 Z 基準，沒有每張圖個別補高度。Track1 255 是第一個固定 window 的尾幀，1051 與 Track2 128 為各 window 中點；圖亦顯示仍有未對齊結構。第二列左側是重新起跑的短段 carry，不能冒充原完整最佳。

- [最終逐 probe 報告](../outputs/run/track00_test_2610191515/comparison/report.json)
- [六組來源／對照 run](../outputs/run/track00_test_2610191515/pairs.json)
- [高度曲線](../outputs/run/track00_test_2610191515/comparison/height_comparison.png)
- [側視繪圖程式](../outputs/run/track00_test_2610191515/plot_overlay.py)

此處的 evidence 在被 Git 忽略的 outputs，需另行備份。高度 reference／四元數是實驗觀測模型，尚未證明為正確 body pose；frame 契約、深度尺度／形變，以及不可靠路面時如何融合其他高度觀測仍待研究。

## 操作與證據

本候選仍須明確指定；省略 height-mode 是 carry，不自動改最佳或啟動完整序列。

### 通用完整定位：從原始資料另存新 output

2026-10-19 依使用者回饋，完整操作改用以下入口，不依賴特定歷史 run。這些指令由使用者手動啟動，Track1 成功後再執行 Track2：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
bash scripts/dev.sh up

python3 scripts/benchmark.py --track 1 --full \
  --profile sensor_guided --sensor-fusion heading_rts \
  --geometry-mode directional --height-mode surface_tilt

python3 scripts/benchmark.py --track 2 --full \
  --profile sensor_guided --sensor-fusion heading_rts \
  --geometry-mode directional --height-mode surface_tilt
```

`--full` 依當前 Track 第一層 timestamped PLY 選取全部幀；不可與 `--frames` 或非零 `--start-frame` 混用，省略它仍只跑預設 50 幀。正式提交的完整性仍依 3011／3299 筆檢查，不以「已選完現有檔案」當成資料齊全的證明。

每條新建 `outputs/run/trackN_...height_surface_tilt_YYMMDDHHmm/`，同分鐘遞增尾碼。流程依序是原始 PLY manifest → 原 bag 的 GNSS／IMU prior → 本輪範圍的 0.3 m 高度 map → 高度／姿態 reference → 原 map 的 1 m 定位 → `result.txt`／`diagnostic.csv`／`validation.json`。既有 ML venv 與 GPU Release 環境仍須存在；不需要舊最佳的 pose／CSV／prior／地圖快取。

目前所有完整／短段測試統一在outputs/run，以trackNN_full/test_YYMMDDHHmm命名；每次直接使用終端印出的Run目錄，不建立固定捷徑。既有測試已集中搬移，原CSV／矩陣／manifest內容及SHA保持。

`height_map_params.json/yaml`、`height_map_command.json`、`height_map_preparation.log` 保存新高度 map 的建置條件；只有工具成功且 binary XYZ 點數／payload 檢查通過後，才發布正式 cache 與含來源／輸出 SHA256 的 sidecar。失敗只保留本輪診斷或 partial，不拿它當有效 map。`height_map_input.json` 記錄本輪建置或明確指定的重用模式；`benchmark_command.json` 保存主機入口命令。

`height_map_preparation_seconds`、`height_preparation_seconds`、`sensor_preparation_seconds` 分開記錄並計入 pipeline 時間。省略 `--map-cache` 仍停用定位快取；本次只為明確選用 surface 家族的模式自動建立必要高度 map，沒有改 legacy／carry 的預設。

輸出以原始點雲＋每幀 pose 保存，RViz 再套用完整 R／t；沒有重建一套新 PLY，也不重算原始 MapAnything。若 reference 支持不足仍會中止並保存 log，不放寬幾何／時間保護。

### 短段對照與既有候選回放

以下保留固定 prior／快取的短段重現命令。`--prior-run` 用於控制比較條件，`--map-cache`／`--height-map-cache` 用於明確重用；均非完整操作的必要歷史依賴：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
python3 scripts/benchmark.py --track 1 --frames 256 \
  --profile sensor_guided --sensor-fusion heading_rts --geometry-mode directional \
  --height-mode surface_tilt \
  --prior-run outputs/run/track01_full_2610172233 \
  --map-cache outputs/run/track00_test_2610171550/cache_track1_leaf1.pcd \
  --height-map-cache outputs/run/track00_test_2610191459/track1_leaf03.pcd
```

每輪另建唯一 run，保存 `height_reference.frames.json`、reference CSV／report／來源 hash、preparation 命令／log、模型程式快照、矩陣與各項診斷。preparation 耗時另計入 `height_preparation_seconds`，不冒充 CUDA 配準時間。

先在目前回放終端按 `q` 關閉舊回放，再直接看**已算好的** Track1 前段候選：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
python3 scripts/replay.py --track 1 --paused \
  --run outputs/run/track01_test_2610191515
```

`--check-only` 已核對此候選 256 幀的來源、整數時間、rotation 與原顯示 map。回放套用保存的完整 R／t，不重新計算定位；省略 run 仍開原最佳，不會偷偷切到候選。

## 最終檢查

GPU workspace Release 已建置；3 項 CTest 通過，高度機制另用真實 CUDA NN 通過。6 項 Python surface-height 測試涵蓋已知坡面／雜訊點、競爭高度層拒絕、固定 yaw 的已知 tilt、15° 保護、近／遠短平面的原點外推條件、整數時間／缺口及 median 尖峰。既有 4 項 geometry-report 測試也通過。

C++ 另核對 reference quaternion／sensor yaw／15°；benchmark 核對結果第三列 rotation 與 reference 一致、Z 參考的 timestamp／hash／邊界。兩條原最佳前 256 幀的 carry 矩陣／提交 CSV／geometry 維持 byte 相同；原最佳 CSV SHA256 再次核對。新增 map utility 的三個 crop／overflow 檢查及 Python 語法／diff 格式檢查通過。

通用入口另外完成 **Track1／Track2 各 24 幀**的獨立實測：沒有指定 prior-run、map-cache 或 height-map-cache；由各自原 bag 產生 prior、原 map 建立本輪高度 map，再執行真實 CUDA 定位。兩條均退出 0、24 筆有限完整、validation 無錯誤；新高度 map 分別 80,427／32,200 點，identity／輸出 SHA256／prior SHA256／binary payload 均核對。這是入口驗證，不是新的完整 Track 或精度勝出證據。逐 run 路徑與檢查結果見 [獨立流程證據](../outputs/run/track00_test_2610191551/verification.json)。

此入口新增 3 項 cache 測試（搜尋範圍、截斷輸出、失敗不發布有效 cache），以及 1 項全幀選取測試（假 Track，於 manifest 截止，不啟動定位）；連同原 6 項 surface-height／4 項 geometry-report，共 14 項通過。`--full` 與顯式 frames／非零 start-frame 的兩種衝突均在啟動前拒絕。沒有執行任何真實完整序列，C++ 未改、無須為通用入口重編。

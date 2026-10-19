# 38｜Track1 停滯後的來源剛體一致性與尺度實驗

日期：2026-10-20（Asia/Taipei）。本輪依使用者「根據最新最佳與最近改進，持續優化 Track1」的要求，完成新的來源觀測工具與短段研究。**最佳仍為 Track1 1.15008／Track2 0.30488；本輪沒有新 Kaggle 分數，也沒有證明定位精度提升。**

使用者後續更新合作方式：Kaggle 只取最佳成績，不另設「完整候選」資格；後續可解釋的新改動完成必要程式／輸出檢查後，即可提供 full 指令實際評分，不要求本地先證明精度提升。本報告的未通過檢查與不推薦事項保留當時研究判斷，並不構成禁止完整實驗的權限規則。完整計算與 Kaggle 提交仍由使用者手動執行。

## 1. 先固定已知成績與問題

使用者本輪截圖顯示 Team 3 在 Track1 **第 6 名、1.15008**；Team 12 為 **0.50997**。這是收到的截圖快照，並非即時查榜；不能從排行榜反推他隊使用的算法。

| 方法／證據 | Track1 | Track2 | 判斷 |
| --- | ---: | ---: | --- |
| heading_rts、geometry off | 1.16951 | 0.31967 | 歷史已評分 |
| heading_rts、directional、carry | 1.15275 | 0.30576 | 歷史已評分 |
| motion_rts、directional、carry | **1.15008** | **0.30488** | 目前已評分最佳 |
| 校正三相機完整候選 | 1.15346 | 0.31064 | 未勝出 |
| surface_tilt 完整候選 | 1.15519 | 尚無回報 | 高度較穩定未帶來 Track1 精度改善 |
| GICP／kinematic graph | 無新分數 | 無新分數 | 短段／感測器留出沒有一致勝出 |
| surfel_joint | 與最佳 CSV 相同 | 與最佳 CSV 相同 | 零可靠 factor，未執行 joint；Track1 再提交仍 1.15008 |

已評分最佳來源為 `outputs/run/track01_full_2610191750`／`track02_full_2610191757`。使用者開啟的 `track01_full_2610192358/diagnostic.csv` 與 Track1 最佳相同；沒有因最新目錄較新就當作新算法成果。截圖未提供 Kaggle 上傳 SHA，仍只核對本機 bytes。

[37](37_HORIZONTAL_SURFEL_JOINT_20261019.md) 已確認 162 個 Track1 surfel 觀測全拒絕；84 個進入擬合，其中 71 個碰界、57 個擬合／留出不一致。這使「來源幾何是否符合剛體模型」比繼續增加 solver 迭代更值得先驗證。

## 2. 新工具：先不看地圖，檢查來源自己是否一致

新增 `scripts/source_rigidity.py` 與 `scripts/audit_source_rigidity.py`：

1. 讀取原前相機 PLY 的 XYZRGB，使用沿用的 scale=3。
2. 以虛擬 pinhole 把 FLU 點投影成紋理圖，z-buffer 保留可見原始點。2 px 補洞只供描述子取樣，**不製造或插值三維深度**。
3. SIFT 雙向 ratio matching；每個原始 vertex／4 px cell 只計一次。特徵旁需有至少三個原始投影點，近鄰深度比不得超過 1.1，避免深度邊緣混層。
4. 依固定 4×3 影像網格劃分擬合／留出；擬合只讀訓練 cells，留出不按三維殘差刪除。
5. 比較兩個模型：

   - 剛體：`b = R a + t`。
   - 相似變換：`b = s R a + t`。

   用訓練資料 RANSAC 起始，再以徑向 pseudo-Huber 做 robust refinement；三維誤差保持旋轉不變性。額外記錄點對距離比，其不受剛體平移／旋轉影響。
6. 完成上述來源擬合後才讀保存 pose 作差異診斷，GNSS／map posterior 都沒有進入來源模型估計。

虛擬相機只作紋理查找，並非宣稱找到了真實 intrinsics 或 camera→IMU 外參。演算法沒有把相對平移當成獨立公尺 odometry，也沒有發布 pose/factor。

SIFT ratio／mutual matching 的依據可參考 [OpenCV 原作者文件](https://docs.opencv.org/4.8.0/d5/d6f/tutorial_feature_flann_matcher.html)；描述子一致仍可能是移動物體或錯配，不等於已知靜態真值。

### 初步投影核對的範圍

Track1 第 929 幀的 exploratory RGB 投影檢查：虛擬雲與校正後影像有 99 對描述子配對、76 對 homography inliers，全部配對的重投影誤差 median／P90 約 0.52／1.54 px；與未校正 raw image 為 59／30 對，median／P90 約 1.40／117.09 px。這是單幀、同資料擬合的檢查，**不能當完整投影校準或泛化證明**；它只不支持目前直接對 PLY 再套一次魚眼校正。腳本、影像與配對預覽保存於第一個研究 run。

## 3. 兩種時間間隔，共 96 對真實資料

沿用原八個 motion windows，再加兩條各四個固定路線比例位置，共 16 組、每組三幀、三對配對。長間隔先做；發現多數移動片段支持不足後，另做短間隔。這是有紀錄的實驗修訂，沒有以降低支持門檻增加通過數。

| 間隔 | Track1 可擬合／總對數 | Track2 可擬合／總對數 | 剛體留出殘差中位數的中位數 |
| --- | ---: | ---: | --- |
| 相鄰取樣隔 14 幀，約 0.50 s；首尾約 1.01 s | 2／21 | 7／27 | 0.3915／0.0596 m |
| 相鄰取樣隔 2 幀，約 0.072 s；首尾約 0.144 s | 17／21 | 10／27 | **0.0881／0.0353 m** |

長短間隔的可擬合子集不同，不能把表格直接當算法精度前後比較。Track2 的足夠支持多在低速片段，不能推成高速段也具有同樣可靠性。

短間隔相似變換的對應統計為 0.0916／0.0363 m，沒有一致勝過剛體；估出的相對尺度中位數為 0.9969／0.9976。因此本輪**不支持把主要問題簡化為「每幀都需要另外縮放」**。但短距離剛體一致也不證明共同絕對尺度正確。

例如 Track1 923→925、925→927：來源剛體平移約 **0.294／0.301 m**，保存軌跡相對位移約 **0.723／0.733 m**，留出殘差約 **0.040／0.036 m**。這提示來源公尺尺度與軌跡位移不一致；保存軌跡不是 ground truth，尚不能排除外參、深度形變或相對運動模型造成的差異。

## 4. 共用尺度候選：有局部作用，未通過升級

新增 `scripts/source_scale_lab.py`。來源點仍以目前 `3×raw` 表示，另估一個兩條共用的倍率 α：

```text
d = R_bᵀ R_a a − b
t = R_bᵀ (p_a − p_b)
min_α Σ robust(‖α d + t‖)
```

`R,p` 在這個實驗來自 **sensor_prior.csv**，不是 map posterior。每個時間 pair 等權，避免較密的影像支配結果；只用原 straight/design windows 的訓練 cells 擬合。其他場景與原擬合對的空間留出都不參與共用倍率估計。α 設寬鬆正值數值界線 [0.25,8]，不是噪聲校準範圍，候選沒有碰界。

實際足夠支持只來自 Track1 一個設計場景的兩對重疊觀測，Track2 straight/design 不足；**跨 Track 訓練支持未通過**，也不能把兩對重疊觀測當成兩個獨立場景。候選為 α=2.02171694，即 **ply_scale=6.06515082**；只作診斷，不提供為新完整測試推薦值。

其他具位移支持的 17 對驗證資料，14 對時間殘差中位數較低；各對中位數再取中位數，由 **0.5676 降為 0.3449 m**。但各場景獨立訓練子集所估倍率，相對共用候選的差異 median／P90 達 **28.8%／87.6%**。這些逐場景倍率僅作診斷，未回填 PLY，也未用來重新調共用值。

**因此，一個較大的固定倍率能解釋部分差異，尚不足以證明單一全域尺度可用。** GNSS prior 位移、未知 lever arm／相機與 IMU 旋轉、RGB 錯配、相對深度偏差仍可能混在一起；本輪不是獨立 metric calibration。

### 與既有水平 surfel 的單因素對照

16 個場景中心固定同一份原始 source vertices、保存 pose、surfel 參數與接受門檻，僅比較 scale=3／6.06515 的 source。這是 **32 次來源／map 診斷**，不是 32 筆新定位輸出，也沒有重跑完整 Track。

舊 fine-map cache 的高度範圍對部分放大點雲不足。本輪新增逐點搜尋範圍檢查，保留原先未完整覆蓋的結果並明確標記；最後從原始 map 在新 run 擴大 crop，供兩個版本共同使用。leaf 仍 0.3 m、normal／fit／holdout 等門檻不變，沒有把不完整快取當成公平完整對照。

最終 32 個版本／場景全部通過 source 搜尋範圍覆蓋檢查；原尺度與候選各 **0／16** 個可靠 factor。Track1 的 map patch 支持中位數由 **34→59**，Track2 **0→8**，但仍未解決空間不足、碰界或留出解不一致。完整流程（含兩條原始 map 讀取／crop）約 98.8 秒，沒有啟動 localization。

來源尺度改善空間支持也不等於觀測已可靠。保存 pose 的 Z／tilt 沒有為新尺度重新估計，因此這個對照只回答目前 factor 能否直接使用，不代表所有 scale＋height 聯合模型都失敗。

## 5. 驗證與本輪交付

- 12 項新增機制測試通過：剛體／相似變換與逆變換、共線拒絕、RGB outlier robust fit、留出不可影響訓練解、尺度與非剛體差異、可見原始 vertex 查找、稀疏／背向來源拒絕、已知共同尺度、全域 map origin 不變性及零尺度資訊拒絕。
- 既有 9 項 surfel／joint 機制測試通過。沒有改 C++／CUDA solver，不需重編定位執行檔。
- 已評分兩條來源 CSV／矩陣／prior／manifest／score metadata 的 SHA 在研究前後相同。
- 所有研究在 `outputs/run/track00_test_*`，保留未通過結果、程式快照、參數與 SHA。沒有新提交、完整定位或正式精度改善宣告。

本輪證據：

| 目錄 | 內容 |
| --- | --- |
| `outputs/run/track00_test_2610200050` | 開工 diff／commit、單幀投影、14 幀間隔 48 對、測試 log |
| `outputs/run/track00_test_2610200054` | 2 幀間隔 48 對、原始 vertex 對應 NPZ、剛體／相似模型與留出 |
| `outputs/run/track00_test_2610200057` | 初始共用尺度／16 中心 map 對照，後續發現 cache 搜尋範圍未完整覆蓋，不能當最終公平比較 |
| `outputs/run/track00_test_2610200058` | 加入 cache 範圍檢查的對照，明確拒絕未覆蓋者 |
| `outputs/run/track00_test_2610200100` | 原始 map 擴大 crop 後的最終對照、來源／build 證據與驗證紀錄 |

這是單次報告內的來源紀錄；沒有另建測試索引或最近／最佳分類。

## 6. 重現與後續順序

只重現跨幀來源檢查，不啟動定位或完整重建：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
outputs/ml_runtime/venv/bin/python scripts/audit_source_rigidity.py --stride 2
# --stride 14 重現較長時間間隔；每次另建唯一 run。
```

重現本輪共用尺度研究與 map 診斷：

```bash
bash scripts/dev.sh up
outputs/ml_runtime/venv/bin/python scripts/source_scale_lab.py \
  --rigidity-run outputs/run/track00_test_2610200054 \
  --fine-map-track1 outputs/run/track01_full_2610192358/horizontal/fine_map/fine_track1.pcd \
  --fine-map-track2 outputs/run/track02_full_2610192240/horizontal/fine_map/fine_track2.pcd \
  --expand-map-cache
```

以上是研究重現入口，需要恢復本機被 Git 忽略的證據與 ML venv；不是新的 production/full 命令。

下一步優先取得**跨場景、獨立於 GNSS 的深度／尺度參考**：以已知相機 baseline 的幾何錨點直接核對原前相機 PLY，保留真實像素到 vertex 的關係，再區分共用尺度、距離相關形變與 camera/body 契約問題。既有三相機 MapAnything 尺度實驗沒有直接標定原前相機 PLY，不能把它的倍率搬過來。

在這一步通過前，保持已評分最佳與接受門檻；不依榜分搜尋倍率、不把 α 寫進主流程、不再重跑零 factor 的相同完整 surfel 候選。來源可靠後，才值得讓新的水平觀測進入已有的 joint graph。

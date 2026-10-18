# 22｜三相機尺度、同步與短段對照實作

後續已實作 raw-image 校正三相機、固定 rig 高度、逐相機運動補償、stereo 深度尺度及側面取樣，見 [校正重建與公平對照](23_CALIBRATED_RIG_RESULTS_20261018.md)。本文件原有 merged PLY／分數分析保留為前一階段證據；新方法尚無完整 Track 或 Kaggle 成績。

日期：**2026-10-18（Asia/Taipei）**。依使用者指示開始實作；本輪完成尺度估計器、時間契約核對、已知 per-view 點雲的運動補償函式，以及 benchmark 的三相機來源／尺度／短段／先驗重用介面。**目前是實驗候選，沒有新 Kaggle 分數，也沒有執行新的完整 Track。**

現行已評分前相機最佳仍為 Track1 **1.15275**、Track2 **0.30576**。先前只讀評估保留於 [三相機初步評估](21_THREE_CAMERA_ASSESSMENT_20261018.md)；本輪的新證據修正了其中尚未查清的時間來源。

## 完成了什麼

- [`three_camera_source.py`](../scripts/three_camera_source.py)：三相機 index／FLU／來源雜湊契約、CDR Header 整數時間解析、bag／header 精確對應與歧義拒絕、選取既有完整 run 的相同感測器先驗。
- [`three_camera_calibration.py`](../scripts/three_camera_calibration.py)：PLY／已核對 map cache 讀取、局部 PCA、帶每幀四自由度 nuisance 的共享尺度估計、排除 pose 先驗後的尺度資訊診斷、已知 per-view 點雲的剛體時間補償函式。
- [`three_camera_lab.py`](../scripts/three_camera_lab.py)：完整 index 對 bag/header 的核對、逐幀同步 CSV、實際感測器運動敏感度、尺度估計與不參與估計的保留影格檢查、來源快照與 hash。
- [`benchmark.py`](../scripts/accuracy/benchmark.py)：新增 `--cloud-source`、明確 `--ply-scale`、`--source-report`、`--start-frame`、`--prior-run`；三相機強制附相符來源報告，不會默認沿用尺度 3。預設仍為前相機，legacy／fixed／geometry off 的預設均未改。

沒有修改定位 C++／CUDA 目標與參數，因此本輪不需重建 workspace。資料保持唯讀、以 GPU outputs 內的容器 symlink 選取短段，沒有複製原始點雲或改動 CPU 環境。symlink 的 `/root/data/...` 目標是在 GPU 容器內解析，主機直接檢視時可能顯示為失效連結；來源 hash 與原位置另存於 `source_input.json`。

## 尺度怎麼估計、怎麼確認

使用模型：

```text
p_map = Rz(delta_yaw) × R_prior × (s × p_raw) + t_prior + delta_xyz
```

`s` 是作用在 source 點座標的單一尺度；輸出的 R 仍是合法旋轉矩陣，沒有把 scale 塞進 pose 的旋轉區塊。GNSS／IMU 只提供初值與原有先驗約束，不當成真值。

1. 用有來源 Track／原 map SHA256／1 m leaf sidecar 的 LiDAR map cache 建局部平面，核對 cache identity，避免拿錯地圖或快取。
2. 對三相機點雲做固定規則的代表點取樣，在 map 上找對應，使用有可靠性權重的點到平面 Huber loss。超出 2 m 或無可靠表面的點保留常數 cost，不能靠丟失對應降低目標。
3. 一起估 `log(s)` 與每幀小量 XYZ／yaw 修正。XYZ／yaw 沿用既有正則化尺度與修正上限；不以逐幀倍率配答案。數值初值為 1，診斷範圍 `[0.5,2]`，每步限制 scale 與 pose 變動並檢查目標降低。
4. 由僅含 map 的局部資訊矩陣，消去自由 pose 方向後檢查剩餘尺度資訊。**只有一塊水平地面時，scale 可以與高度互相補償，不能據此宣布尺度已確認。** 該退化案例已納入測試；這些資訊仍是局部模型診斷，不是已校準 covariance 或全域可觀測性保證。
5. 每條 Track 的估計影格固定為 idx `500,1000,1600,2200`，保留影格為 `750,1300,1900,2600`。分開估計後，再以兩條共 8 個估計影格求**同一個共享尺度**，另在共 8 個保留影格檢查；沒有用 Kaggle 分數決定倍率。

| 尺度檢查 | 結果 |
| --- | ---: |
| Track1 單獨估計 | 0.88918650 |
| Track2 單獨估計 | 0.87123483 |
| 兩條共同估計，初值 1.0 | **0.87124954** |
| 兩條共同估計，初值 1.2 | 0.87082845 |
| 兩次共同估計差異 | 約 0.048% |

兩個初值用來檢查局部解的穩定性，不是大量參數搜尋；仍不能證明已找到全域正確尺度。[帶 scale 的相似變換估計](https://www.open3d.org/docs/release/cpp_api/pipelines_2registration_2_transformation_estimation_8h_source.html) 是一般配準模型參考，本案實作的是帶先驗及四自由度 nuisance 的 robust 點到平面版本。

另對尚未正式確認的高度參考作敏感度檢查：只將所有估計影格的高度先驗減／加 1 m，來源點與 XY／yaw 不變，估得尺度為 **0.86984907／0.87468679**。這支持候選在本輪局部模型下相對穩定，並不是拿高度偏移作新的定位設定；沒有把任何測試偏移套到 benchmark。

| 保留影格，同一來源／同一目標的總 cost | scale=1 | 共享 scale=0.87124954 | 降低 |
| --- | ---: | ---: | ---: |
| Track1 | 72.07162 | 71.95086 | 約 0.17% |
| Track2 | 64.84256 | 63.04094 | 約 2.78% |

此 cost 不是公尺或正式定位誤差，不能與不同來源的 cost 比較。Track1 的支持很弱，部分保留影格的 NN／平面殘差仍變差；Track2 的多數保留影格有改善。**因此 0.87124954 的定位是「有跨影格證據、可做對照的尺度候選」，尚不能說物理尺度已精確校準，更不能宣稱一定提高 Kaggle 精度。** 尺度也可能吸收重建形變、frame、時間或標定誤差，後續應檢查更多獨立場景與分項誤差。

## 同步怎麼核對：找到混合時間來源

完整巡查兩條 index，逐筆在對應相機 bag topic 中尋找**完全相同整數 timestamp**，同時檢查 bag 與 Header 兩種可能，找不到或出現歧義直接失敗：

| index 欄位 | Track1 | Track2 |
| --- | --- | --- |
| `f_ts` | 3011 筆均為前相機 bag timestamp | 3299 筆均為前相機 bag timestamp |
| `fl_ts` | 3011 筆均為左前相機 header timestamp | 3299 筆均為左前相機 header timestamp |
| `fr_ts` | 3011 筆均為右前相機 header timestamp | 3299 筆均為右前相機 header timestamp |

CDR parser 另與 ROS `deserialize_message(sensor_msgs/msg/Image)` 核對兩條 Track 各三個相機 topic 的首筆，六筆一致。時間始終以整數奈秒保存；先相減才轉秒。

**先前直接用 `fl_ts − f_ts`／`fr_ts − f_ts` 混合了不同時間來源，不能當作拍攝差。** 本輪先取回每筆訊息的 bag 與 header，再統一用 header 比較：

| 同一 header 時基，相對前相機，中位數 | Track1 | Track2 |
| --- | ---: | ---: |
| 左前 − 前 | +6.952 ms | −2.395 ms |
| 右前 − 前 | +21.189 ms | +20.704 ms |

以實際 GNSS 平滑位置及 IMU yaw 內插估算，相對前相機 header 的右前相機移動量中位數約 **0.216／0.244 m**，最大約 **0.307／0.323 m**。這是由感測器估計的時間敏感度，不是點雲或定位真值誤差。少數端點不在感測器覆蓋內，報告保留 edge clamp 計數，不把它們當作已確認的運動。

定位輸出仍以原 **前相機 bag timestamp** 對應 idx，感測器先驗逐筆與對照相同；沒有直接將整條輸出切到 header 時間。Header 是否代表真實曝光、各相機硬體時鐘是否已同步，現有資料仍不足以獨立證明。

## 哪部分同步已實作，哪部分還不能套到 merged PLY

已知 per-view 點雲與其 capture time 時，可以使用：

```text
T_front_ref_camera_i = inverse(T_map_front(t_ref))
                       × T_map_front(t_i) × T_front_camera_i
```

這會把各相機拍攝時刻的點，轉到共同 reference time。`camera_to_reference()` 已實作並以靜態世界／車輛移動的已知案例驗證；輸入矩陣必須具有已確認且一致的 frame／單位。

**現成 MapAnything 檔已用預測相對姿態合併，只有 XYZRGB，沒有每點 camera_id／原始 per-view pose。** 所以本輪能確認時間來源與差異、提供補償函式，但**尚未對真實 merged PLY 做逐相機 motion compensation**，也無法確認原製作者是否已處理這些差異。不能把整份 merged PLY 用單一時間平移來冒充完成三路補償。

要完成真正逐相機時間補償，需要取得對應 per-view 輸出／cache，或另行以已有原始影像、相機校正及感測器運動重建。這與「核對 index、讓先驗對齊共同 reference」是不同層次。本輪未啟動整包重建，也沒有重套三相機外參或 optical-to-FLU 軸轉換。

## 短段對照與驗證範圍

對照組為目前前相機 scale=3、三相機 scale=1、三相機共享 scale=0.87124954。每條取移動直行與轉彎各 24 幀；片段依感測器速度／yaw rate 選取，與尺度估計／保留影格分開，不依分數挑選。每組重用相同完整 run 的 sensor prior，保留同一 map、directional 參數與接受門檻。

12 組共 **288 幀**均退出碼 0、完整有限輸出、空 validation_errors，且 288 次地圖更新均接受。三種來源各 96 幀，逐窗口 sensor_prior.csv 的內容完全相同。

| 片段 | 原始 idx，24 幀 | 平均速度 | 平均絕對 yaw rate |
| --- | --- | ---: | ---: |
| Track1 直行 | 1493–1516 | 12.15 m/s | 0.164°/s |
| Track1 轉彎 | 1811–1834 | 11.78 m/s | 3.074°/s |
| Track2 直行 | 1649–1672 | 15.12 m/s | 0.107°/s |
| Track2 轉彎 | 1385–1408 | 14.06 m/s | 3.289°/s |

| 每組 96 幀的局部 effective rank 分布 | rank 1 | rank 2 | rank 3 | rank 4 |
| --- | ---: | ---: | ---: | ---: |
| 前相機 scale=3 | 85 | 11 | 0 | 0 |
| 三相機 scale=1 | 0 | 50 | 27 | 19 |
| 三相機 scale=0.87124954 | 0 | 31 | 35 | 30 |

三相機增加了模型中的幾何方向，但 rank 仍取決於既定尺度／門檻，且無法驗證重建本身是否正確，不能把滿秩當精度證明。CUDA 搜尋＋CPU 細修／幾何平均每幀約 **8.30／113.41／90.03 ms**，三相機計算量也明顯增加；這是不同輸入的短段成本，不是 GPU 加速倍率。

另以**相同前相機 probe 點與同一 map**比較三種 pose，避免直接拿不同來源的殘差當公平比較。probe 是 raw PLY 以 0.1 原始單位 voxel 選代表點後乘 3，所有方法逐幀使用同一批點，並計算 2 m 截斷 NN RMS；這與 C++ centroid filter 略有不同，但對照 probe 一致。

| 共同前相機 probe，clipped NN RMS（m，非 ATE） | 前相機 | 三相機 scale=1 | 三相機 scale=0.87124954 |
| --- | ---: | ---: | ---: |
| Track1 直行 | 0.80764 | 1.38258 | 1.27502 |
| Track1 轉彎 | 1.24192 | 1.55830 | 1.56277 |
| Track2 直行 | 0.65194 | 1.46645 | 1.34461 |
| Track2 轉彎 | 0.61080 | 1.40366 | 1.29095 |

三相機共享尺度的高度與前相機輸出差異 RMS 分別為 **2.647／2.092／1.841／1.497 m**。為理解機制，另只在診斷中保留前相機的 Z，代入三相機的 XY／yaw；共同 probe RMS 變成 **0.80820／1.24670／0.65157／0.61161 m**，接近前相機對照。只代入三相機 Z 時則接近上表完整三相機的較大殘差。這支持共同 probe 的大差異主要受高度影響，**不代表已證明哪一份高度或 XY／yaw 是真值**；這些混合 pose 僅為機制診斷，沒有另存提交候選。

本任務 CSV 只提交 XY／yaw，因此共同三維 probe 的惡化也不能直接等同正式分數變差。較保守的結論是：三相機提供更多幾何方向、尺度候選有局部支持，但兩種重建來源仍有明顯高度／形狀契約差異，XY／yaw 的改動是否更準尚無證據。**本輪不將三相機設為推薦最佳，不因 rank 較高就啟動完整替換；下一步應釐清來源 frame／高度與 per-view 融合品質。**

來源、完整命令、原始 idx、輸出與驗證全部保存在下方實驗目錄。[尺度／同步／共同 probe 比較圖](../outputs/runs/three_camera_dev_2610181055/scale_sync_comparison.png) 將候選與限制放在同一張圖，沒有把內部 cost 當正式分數。

本輪 Python 測試 21 項通過，涵蓋已知尺度、地面退化、運動補償、時間來源混合／歧義、精確 prior 重用及既有融合／幾何／濾波契約。修改 benchmark 後，另以相同完整先驗重跑 Track2 前 24 幀，`result.txt` 與原已評分完整 run 的前 24 行逐 byte 相同。C++／CUDA 未修改，沒有把這些合成或回歸檢查當成實測精度。

## 重現、操作與證據

唯一實驗目錄：`outputs/runs/three_camera_dev_2610181055/`，不覆寫既有最佳檔。

- [Track1 尺度與同步報告](../outputs/runs/three_camera_dev_2610181055/calibration_track1_3/report.json)、[Track2 報告](../outputs/runs/three_camera_dev_2610181055/calibration_track2/report.json)。Track1 前兩次嘗試的 ROS 環境未載入／誤假設左右 index 為 bag 時間之失敗 log 保留。
- [共享尺度與保留影格檢查](../outputs/runs/three_camera_dev_2610181055/joint_scale.json)、[可重現的共享尺度程式](../outputs/runs/three_camera_dev_2610181055/joint_fit.py)。
- [高度參考敏感度](../outputs/runs/three_camera_dev_2610181055/height_anchor_sensitivity.json)。
- [片段選取](../outputs/runs/three_camera_dev_2610181055/windows.json)、[每輪完整命令與來源](../outputs/runs/three_camera_dev_2610181055/comparison_runs.json)、[比較分析程式](../outputs/runs/three_camera_dev_2610181055/analyze_comparison.py)。
- [12 組短段與共同 probe 摘要](../outputs/runs/three_camera_dev_2610181055/comparison_summary.json)。
- [最終 Python 測試 log](../outputs/runs/three_camera_dev_2610181055/python_tests_final.log)、[完整時間契約終檢](../outputs/runs/three_camera_dev_2610181055/final_time_verification.json)、[原前相機回歸 log](../outputs/runs/three_camera_dev_2610181055/front_regression.log)。

outputs 被 Git 忽略；報告、cache、短段結果與來源快照需另行保存。移機後不能只靠同步 Git 就假設報告路徑存在。日常入口與本輪新選項見 [操作手冊](08_MANUAL_WORKFLOW.md)。

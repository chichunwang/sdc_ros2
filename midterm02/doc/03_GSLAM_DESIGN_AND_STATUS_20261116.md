# 03｜gslam：GPU 視覺引導幾何 SLAM 的設計與現況（2026-11-16）

## 這份文件回答什麼

`vslam/gslam/` 是從頭寫的 SLAM 程式（沒有複製 `midterm01/`、`midterm01_gpu/` 的算法程式）。這裡說明它怎麼運作、哪些部分在 GPU、目前驗證到哪裡，以及你接下來要執行的命令。

**現況一句話：full Track01 已跑完，產生通過格式檢查的 `result_1.csv`（1314 個 pose，零追蹤失敗，22 個閉環）；還沒有 Kaggle 分數，所以精度未知。**

| 階段 | 狀態 |
| --- | --- |
| 已實作 | GPU 前處理、GPU ICP／GICP、視覺前端（Essential＋深度定長、PnP）、視覺先驗融合、品質閘門、keyframe／子圖、視覺閉環＋點雲驗證、pose graph、輸出與圖表 |
| 合成檢查 | 通過：`scripts/dev.sh gpu`（GPU 與 CPU 最近鄰逐點一致，7.0 ms 對 81.7 ms）；`scripts/dev.sh test` 29 項全過 |
| 真實短段 | 300 幀多組消融（見「實測結果」） |
| full | `outputs/run/track01_full_2611160111`（預設參數，約 5 分鐘） |
| 正式分數 | **無**（Kaggle 提交由使用者執行） |

## 每一幀做什麼

```text
PLY ──► GPU 前處理（深度範圍、體素降採樣、孤立點）──────────────┐
影像 ─► SIFT 特徵 ─► GPU 描述子比對 ─► PnP（3D 來自上一幀點雲）─► 視覺初值
                                                                 ▼
                              GPU GICP／point-to-plane ICP 對局部子圖配準
                                                                 ▼
                    品質閘門 ──不通過──► 換初值＋放寬對應距離重試 ──仍不通過──► 沿用預測、標記失敗
                                                                 ▼
                    keyframe？──► 更新局部子圖 ──► 視覺閉環檢索 ──► 影像＋點雲驗證 ──► pose graph
```

### 1. 為什麼要用影像給配準初值

觀察：ICP／GICP 只會收斂到初值附近的局部最佳解。車輛轉彎時，等速模型預測的方向會錯，配準可能卡在錯的解。
做法（兩種模式，依資料自動選）：
- 車輛有移動時用 **Essential matrix**：RANSAC＋`recoverPose` 給旋轉與平移方向。它只用 2D 對應，不受 LoGeR 深度誤差影響（實測內點率約 95%）。平移長度 λ 再用上一幀的 3D 點估計：每個點給一個 λ 的最小平方解，取視差較大的一半再取中位數。
- 幾乎靜止（特徵位移中位數 < 2 px，Essential matrix 退化）時用 **PnP**：3D 來自上一幀點雲。
得到 `T_prev_cur` 後接到上一幀 pose：`T_init = T_w_prev · T_prev_cur`。

為什麼不全用 PnP：車輛移動時，LoGeR 約 5% 的深度誤差會讓重投影誤差超過 2 px，PnP 內點比例掉到 0.3。

### 1b. 視覺位姿不只當初值，也當先驗

觀察：只把視覺位姿當初值時，ICP 會把 pose 拉離它 0.15 m（中位數），單幀位移在 0.07–1.57 m 間亂跳（實際車速約 0.1–0.3 m／幀）。
做法：配準代價多加一項 `n·(w_t·|ξ_v|² + w_r·|ξ_ω|²)`，ξ 是目前 pose 相對視覺 pose 的左乘小量。權重由「點雲殘差標準差／視覺標準差」的平方除以有效獨立點數得到（`params.py` 的 `fusion`）。
**限制**：目前權重下視覺項佔主導（配準結果離視覺 pose 中位數僅 0.006 m），點雲主要透過閉環修正軌跡。這組權重是模型假設，沒有 GT 可校正。

### 2. 像素怎麼對到 3D 點

不假設 PLY 的點順序等於像素順序。實測每份 PLY 有 61,740 點，投影後是規則的 315×196 格點（約每 4.6 px 一點）。`PixelCloud` 在特徵點周圍四個格點之間對深度做雙線性內插，再沿特徵點的視線反投影；四個格點深度差超過 8%（深度不連續）就不給 3D。投影不成規則格點時退回最近點查表。
這個假設（點雲確實在該相機座標、內參正確）已用 `scripts/inspect_data.py` 驗證：投影位置的影像顏色與點顏色的平均絕對差是 0.0003–0.0006，隨機配對是 0.17–0.27。

### 3. GPU 配準的模型

source 點 `p`（相機座標）、target 是世界座標的局部子圖：`q = s·R·p + t`（`s` 預設固定為 1）。

- point-to-plane ICP：殘差 `e = nᵀ(q − m)`，`m` 是最近的 target 點、`n` 是它的法向。
- GICP：殘差 `r = q − m`，權重 `M = (C_m + s²·R·C_p·Rᵀ)⁻¹`，兩邊的共變異數都正則化成「一小片平面」`V·diag(ε,1,1)·Vᵀ`。
- 兩者都加 Cauchy 權重降低錯誤對應的影響，由粗到細分兩個階段（對應距離 1.0 m → 0.4 m）。
- 更新量 `δ = (v, ω, σ)`：`∂q/∂v = I`、`∂q/∂ω = −[q]×`、`∂q/∂σ = q − t`。

### 4. 品質閘門（全部通過才接受）

fitness、inlier RMSE、與初值的平移／旋轉差、資訊矩陣最小特徵值（幾何退化）、以及**配準出的相對旋轉與影像估計的相對旋轉是否一致**（旋轉不受點雲尺度影響，是獨立的檢查）。
不通過時依序重試；全部失敗就沿用預測 pose、標記 `tracking_failure`，運動狀態保留，不會讓 pose 跳回舊值。

### 5. 閉環

1. 檢索：目前 keyframe 的每個描述子在「夠久以前的 keyframe」描述子庫中找最相似者並投票（GPU 矩陣乘法）。
2. 影像驗證：Essential matrix 內點數、PnP 內點數。
3. 點雲驗證：以 PnP 結果為初值，把目前 keyframe 配準到候選 keyframe 附近的子圖；檢查 fitness、RMSE，以及**配準結果與影像估計是否一致**。
4. 全部通過才加入 pose graph（GTSAM，Huber 穩健核）。所有候選（含被拒絕的）都記在 `loop_candidates.csv`。

非 keyframe 的 pose：每幀相對前、後兩個 keyframe 各算一個修正後的候選，再在 SE(3) 上內插，因此經過 keyframe 時連續。

## 哪些真的在 GPU

| 運算 | 裝置 | 位置 |
| --- | --- | --- |
| 體素降採樣、深度／孤立點過濾 | CUDA（CuPy） | `gpu_cloud.py`、`preprocess.py` |
| 最近鄰搜尋、鄰域共變異數 | CUDA（自寫 RawKernel） | `gpu_cloud.py` |
| 法向／共變異數特徵分解、殘差、H 與 g 累加 | CUDA | `gpu_cloud.py`、`gpu_registration.py` |
| 描述子比對、閉環投票 | CUDA | `visual.py` |
| 6×6／7×7 線性解 | CPU | `gpu_registration.py` |
| **SIFT／ORB 特徵抽取** | **CPU**（pip 版 OpenCV 沒有 CUDA 模組） | `visual.py` |
| Essential／PnP RANSAC | CPU（資料量小） | `visual.py` |
| pose graph | CPU（GTSAM） | `pipeline.py` |
| SuperPoint＋LightGlue（選用） | CUDA（PyTorch） | `visual.py` |

沒有 CPU fallback：CuPy 匯入失敗或沒有 CUDA 裝置會直接報錯。驗證方法：`scripts/dev.sh gpu`（逐點比對 GPU 與 CPU 最近鄰、列出耗時與 GPU 記憶體使用）。

## 用 RViz 觀看執行過程

手動執行一律用 `scripts/dev.sh slam`：它會先開 RViz 視窗（若還沒開），再啟動 SLAM 並把狀態即時送到 RViz。

```bash
./scripts/dev.sh slam --track 01
```

```bash
./scripts/dev.sh slam --track 01 --frames 300
```

只開／重開 RViz 視窗用 `./scripts/dev.sh rviz`，關閉用 `./scripts/dev.sh rviz-stop`。關掉視窗後 SLAM 會印出警告並繼續跑完。

為什麼是兩個容器：SLAM 容器是 Python 3.9（官方 starter 指定），和 ROS 2 Humble 的 Python 3.10 不相容。所以 RViz 與 bridge（`scripts/rviz_bridge.py`）在另一個 `osrf/ros:humble-desktop` 容器 `midterm02_rviz`，SLAM 透過 TCP（`vslam/gslam/viz_stream.py`）把資料送過去。視覺化只讀狀態，不影響 pose；有開 RViz 的 run 會在 `test_info.json` 記 `rviz: true`。

RViz 畫面（設定檔 `config/vslam.rviz`）：

| 顯示項目 | 內容 |
| --- | --- |
| Camera (3D overlay) | 原始相機影像，疊上軌跡與 keyframe |
| Features and status | 特徵點（黃）、通過幾何驗證的匹配（綠線）、被剔除的匹配（紅線），左上角是幀號、追蹤狀態、內點數、fitness、RMSE、單幀位移 |
| Map (keyframes, RGB) | 所有 keyframe 依目前 pose 拼成的彩色地圖；閉環後會重新排列，牆面重影代表軌跡有誤 |
| Current frame cloud | 目前幀的點雲（紅） |
| Local submap | 配準目標（預設關閉，可勾選） |
| Path odometry／live／optimized | 灰＝純 odometry；黃＝執行中的 pose；藍＝結束時的最佳化結果 |
| Keyframes、Loop edges + camera frustum | keyframe 位姿（橘箭頭）、閉環邊（綠線）、相機視錐（黃） |
| TF | `map → world_cv → camera` |

SLAM 的世界座標是第一幀的相機軸（y 向下），bridge 用靜態 TF 轉成 RViz 的 z 向上，畫面才不會顛倒。
限制：黃色的即時軌跡不會在閉環後回頭修正舊的部分（keyframe 箭頭與地圖會）；完整的修正結果是結束時出現的藍線。

## 實測結果（Track01，2026-11-16）

沒有 ground truth，以下都是自我一致性指標，**不是精度**。「粗糙度」是相鄰三幀位置的二階差分（越大代表軌跡越鋸齒）。

300 幀短段，依修改順序：

| 版本 | 追蹤失敗 | 用到視覺初值的幀 | 粗糙度 中位數／p90 [m] | 路徑長 [m] |
| --- | --- | --- | --- | --- |
| 最近點查表＋PnP 初值（run `…160036`，200 幀） | 28／200 | 33 | — | — |
| 深度內插＋放寬等速閘門（`…160041`） | 0 | 210 | 0.304／0.734 | 79.3 |
| ＋視覺先驗融合（`…160042`） | 0 | 210 | 0.036／0.538 | 55.1 |
| ＋Essential 模式（`…160054`，含地面尺度） | 0 | 299 | 0.028／0.082 | 36.0 |

路徑長由 79 m 降到 36 m，是因為鋸齒被消除；Essential 與 PnP 兩種獨立的平移長度估計中位數比值為 0.994。

full（1314 幀）：

| 版本 | 閉環數 | 閉環配對處的 odometry 落差 中位數／最大 [m] |
| --- | --- | --- |
| 地面高度尺度正規化（`track01_full_2611160101`） | 20 | 3.92／7.08 |
| 不做尺度正規化（`track01_full_2611160106`、`…160111`，預設） | 22 | 2.21／3.68 |

**被否定的假說**：相機離地高度與影像估到的相鄰幀尺度比有正相關（r≈0.57，前 300 幀），所以我試了「尺度 = 參考高度／該幀高度」。full 上它讓 odometry 漂移變大，因此預設關閉（`scale.mode='none'`），程式保留供實驗（`E10a_ground_scale`）。可能原因：地面擬合受路面坡度與路緣影響；且靜止段裡兩種尺度指標的變化方向相反，表示 LoGeR 的誤差不是單一尺度係數。

可重現性：同參數兩次 full 的位置差最大 1.7×10⁻¹³ m（GPU 浮點加總順序），`result_1.csv` 的 SHA 因此不同，但對分數沒有影響。

其他觀察：
- 逐幀耗時中位數 176 ms，其中 SIFT 抽取（CPU）約 140 ms、GPU 配準 13.5 ms、GPU 比對 0.7 ms。
- 官方說路線約 247 m，我們的軌跡長 194 m；整體尺度不影響評分（Sim(3) 對齊）。
- 最佳化後的軌跡在起點附近（右側直線段）三圈沒有完全重合，且有幾處轉折，是目前最可疑的區域。

## 還沒做的

- 尚未跑過的消融：E2／E3（純運動模型＋ICP／GICP）、E5／E6、E8（無閉環）、E9b（無融合）的 full；這些要有 Kaggle 分數才有比較意義。
- NDT 粗配準（條件式 fallback）：目前的 fallback 是放寬對應距離的多尺度配準。
- IMU：資料讀取已支援有／無 IMU，但尚未使用 IMU。
- 動態物體過濾、Sim(3) pose graph。
- 整條流程的合成序列整合測試（需要合成影像）。
- E0（官方 starter 原樣）：`vslam/main.py` 與 `utils/` 保持未修改，TODO 尚未填。

## 你接下來要做的事

1. 把 `outputs/run/track01_full_2611160111/result_1.csv` 提交到 Kaggle，回報 Public 分數（我會記進該 run 的 `test_info.json`）。
2. 看 `outputs/run/track01_full_2611160111/analysis/` 的四張圖。

重跑與消融（Docker 群組生效後；清單在 `vslam/gslam/params.py` 的 `EXPERIMENTS`）：

```bash
./scripts/dev.sh exec python vslam/run_slam.py --track 01
```

```bash
./scripts/dev.sh exec python vslam/run_slam.py --track 01 --experiment E8_no_loop
```

## 已知風險

- 尺度估計（`estimate_scale`）的可觀測性：合成測試發現只有單側平面（一面地、一面側牆、一面前牆）時，尺度變化可被平移完全補償，配準會收斂到錯的尺度；兩側都有面才定得出尺度。真實街景是否足夠要用資料確認，所以預設關閉。
- 品質閘門、融合權重與閉環門檻都沒有 GT 可校正；目前只確認 Track01 上零追蹤失敗與三圈大致疊合。
- 資料包不含 `image_slam_starter/`，`data/` 的結構是手動放成 `data/data_VSLAM-T1/Track1/data/`；下載的 zip 留在 `data/_download/`。
- SuperPoint＋LightGlue 待 E3 QA 確認前不納入正式結果；starter README 另有「不得用完整 SLAM／SfM 系統產生提交」的規定，本程式是自寫流程，不屬於該類。

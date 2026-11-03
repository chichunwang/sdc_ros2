# Track2 v5.8：讓路面影像速度量測走得更遠

v5.7 的移動影像速度只在 frame 196、231、238、245 被採用；3299 幀中，frame 245 之後沒有新的影像速度或靜止量測通過融合。因此 v5.7 的影像資訊主要集中在開頭。本版改善相鄰影格的角點追蹤、平面 PnP 求解及地面公尺尺度的延續，嘗試讓行駛中的影像也能提供因果速度資訊。

v5.7 full 的 Public Score `0.44296` 是使用者回報。v5.8 是新的候選程式，目前沒有 v5.8 full 或新成績；不宣稱已達使用者目標 `0.3`。下一步由使用者手動啟動 full：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
./scripts/run_track2.sh --full
```

## 從觀察到改動

v5.7 有 471 個影像區間候選，只有 4 次移動速度更新及 17 次靜止更新通過融合。舊紀錄中，202 個候選有地面角點數，median 是 39；99 個有 PnP inlier 記錄，median 是 18。252 個候選找不到已驗證公尺尺度地面，另有 103 個因地面角點不足而未進 PnP，94 個被 PnP outlier 閘門擋下。這些原因指向量測涵蓋率不足，不能只靠提高量測權重解決。

實際照片的唯讀角點追蹤元件檢查顯示，跨整段直接追蹤會丟掉不少路面特徵。例如 frame 259→266 地面角點由 58 增至 103，frame 2051→2058 由 34 增至 81。這只是特徵追蹤數，不是 PnP 或定位結果。

v5.8 只改前視影像量測：

1. 每幀以雙向 Lucas–Kanade 追蹤已選出的角點，仍每隔至少 `0.25 s` 消費一次不重疊量測。逐幀追蹤減少單次大位移造成的失配；追蹤步數也提高影像幾何的不確定度。
2. 路面點近似共平面，先用 homography RANSAC 篩除錯配，再用 IPPE 求平面相機姿態，檢查兩個候選分支並以 iterative PnP 細化。候選仍須符合正深度、gyro 轉動預算、路面法向位移、重投影誤差與速度範圍。
3. 只要過去某幀已通過 feature scale 驗證，後續已消費影像的 keyframe 就可用目前地面平面和這個歷史公尺高度建立路面模型。高度使用定位器保存的因果高度 anchor；沒有通過檢查的當幀配準不會另造尺度量測。
4. 移動路面角點下限由 40 降為 24；這是進入幾何求解的工程下限，不代表量測有足夠資訊。後續仍要求至少 60% RANSAC inlier、至少 24 個 inlier、6 維 Jacobian full rank，並用 covariance、gyro 與 GNSS NIS gate 控制影響。全畫面靜止候選仍需 40 個角點和原有格網條件。

IPPE 適用共平面點，`solvePnPGeneric` 可取回平面問題的多個候選解，介面依 [OpenCV 4.5.5 官方文件](https://docs.opencv.org/4.5.5/d5/d1f/calib3d_solvePnP.html)。逐幀 LK 的像素標準差 floor 依追蹤步數平方根增加，另加入可行分支的速度歧義方差；這是保守的局部模型，未完整估計跨幀相關性。

地圖配準與 v5.6／v5.7 的 `v52_directional_memory` 融合維持相同設定。沒有調 GNSS noise、map 權重、heading bias、外參、scale bounds 或歷史 correction memory；新設定只有影像移動角點下限。非因果參考 CSV 只供定位差異的診斷，不用來回歸 pose、估時間偏移或選出最佳分數參數。

## 參考 CSV 的差異與優先觀察區段

與 `result_2_0.30488.csv` 以相同 idx 直接對照，沒有做旋轉、平移或時間對齊擬合。v5.7 的平均 XY 差異約 `0.1781 m`、P95 約 `0.4325 m`；最大差異約 `0.733 m`，在 frame 1390。這是兩份輸出的差異，不是真值誤差。地圖可觀測性 rank 1 在全序列 3299 幀中有 1878 幀，表示很多幀只從 map 得到部分方向的資訊；rank 本身也不能判定哪條軌跡更準。

| 優先區段 | 證據與檢查方向 |
| --- | --- |
| 1300–1499，尤其 1390 | XY 差距集中最大；觀察影像車速是否能通過幾何與 NIS gate，並保留 map rank 診斷。 |
| 300–399，尤其 314、376 | 參照 yaw 差異較明顯。首次 GNSS course heading 更新在 frame 314，約 0.486 m/s、方向標準差約 2.98°；它是模型估計，不是已知航向真值。 |
| 900–999、1700–1799、2400–2599、3200–3298 | 其他較大的區段差異；檢視實際影像量測覆蓋、地圖觀測 rank 和 NIS，不把接受率當精度。 |

frame 1390 附近的 real-photo component 有 64 個跨區間地面 tracks、71 個逐幀追蹤後存留；1456 為 113→150。這能支持新追蹤方式值得驗證，仍不能保證其 PnP 通過或降低 Public Score。

## 本輪檢查與限制

生成幾何對照使用 20 個固定 seed × 4／12／20 m/s，共 60 個相同輸入案例，加入 0.2 px 像素雜訊和 20% 錯配：v5.7 EPNP 通過 36/60，v5.8 平面 IPPE 通過 59/60；兩者在通過案例的 median absolute speed error 分別約 0.00126 與 0.00105 m/s。最大誤差分別約 0.00276 與 0.00596 m/s，因此這個生成比較只支持新求解器提高了該合成模型下的接受覆蓋，不表示精度全面變好。

凍結完整 source snapshot 的 regression 為 **196 passed、1 項既有 Matplotlib `Axes3D` warning、24.91 秒**；結果見 [pytest_verified.xml](../outputs/run/track00_test_2611031306/pytest_verified.xml)。這是程式回歸，不是實際定位精度測試。實際照片檢查只做角點和可見地面 mask 的唯讀 component 計數；沒有對真實 Track2 影像執行 PnP，沒有新 pose 或 map localization，也沒有跑 v5.8 full。v5.8 的實際接受數、地圖融合影響與 score 都要等使用者啟動 full 後才能判讀。

圖表與輸入紀錄保存在 [本輪診斷 run](../outputs/run/track00_test_2611031257/)：

- [v5.7 與參照 CSV 的逐幀差異圖](../outputs/run/track00_test_2611031257/analysis/01_reference_difference.png)／[PDF](../outputs/run/track00_test_2611031257/analysis/01_reference_difference.pdf)
- [軌跡對照圖](../outputs/run/track00_test_2611031257/analysis/02_reference_trajectory.png)／[PDF](../outputs/run/track00_test_2611031257/analysis/02_reference_trajectory.pdf)
- [實際影像逐幀角點追蹤與地面支持統計](../outputs/run/track00_test_2611031257/analysis/image_component_metrics.json)：沒有 PnP 或新 pose
- [影像來源與追蹤圖 1](../outputs/run/track00_test_2611031257/analysis/03_adjacent_image_tracks_1.png)／[PDF](../outputs/run/track00_test_2611031257/analysis/03_adjacent_image_tracks_1.pdf)、[圖 2](../outputs/run/track00_test_2611031257/analysis/03_adjacent_image_tracks_2.png)／[PDF](../outputs/run/track00_test_2611031257/analysis/03_adjacent_image_tracks_2.pdf)、[圖 3](../outputs/run/track00_test_2611031257/analysis/03_adjacent_image_tracks_3.png)／[PDF](../outputs/run/track00_test_2611031257/analysis/03_adjacent_image_tracks_3.pdf)、[圖 4](../outputs/run/track00_test_2611031257/analysis/03_adjacent_image_tracks_4.png)／[PDF](../outputs/run/track00_test_2611031257/analysis/03_adjacent_image_tracks_4.pdf)
- [生成平面求解比較圖](../outputs/run/track00_test_2611031257/analysis/04_generated_planar_solver.png)／[PDF](../outputs/run/track00_test_2611031257/analysis/04_generated_planar_solver.pdf)

參考差異、同索引對照、component 數據與照片圖由 [run analysis](../outputs/run/track00_test_2611031257/) 保留。照片圖中的地面 mask 是點雲平面與可見性幾何，不是語意道路分割。

<details>
<summary>回歸過程中的環境錯誤與重試</summary>

這些是分開的歷史輪次，不相加為測試總數。`track00_test_2611031257` 曾誤執行全 repository 測試，因 external Track3 author tests 缺少 `external_source` 而出現 2 個環境失敗；該次執行中止，當時記錄為 124 passed、2 failed、127.19 秒。`track00_test_2611031306` 首輪 194 passed、2 failed、24.89 秒，原因是漏設既有 C++ oracle 的 `TRACK2_REFERENCE_LIBRARY`。之後在同一完整 Python source snapshot 編譯既有 `liboriginal_guided.so`，單輪 196 項全通過；產品程式沒有因這項環境修正而變更。影像 component 的 16 項檢查是較早的元件輪次，也不與完整回歸相加。

</details>

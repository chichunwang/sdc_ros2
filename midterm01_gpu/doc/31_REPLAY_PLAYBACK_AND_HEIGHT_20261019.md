# 31｜回放播放丟雲修正與高度診斷

日期：2026-10-19（Asia/Taipei）。使用者以 RViz 肉眼觀察已保存的前相機最佳結果，回報「地圖高度」與「播放時 pose 繼續移動、影像點雲不更新，暫停後復位」。本輪修正回放顯示並讀取既有幾何，沒有重算定位、改 solver／尺度、修改原結果或啟動完整 Track。

## 播放問題：找到實際日誌，修正回放端

原視窗為 `track00_test_2610191054`，來源是 Track1 前相機 `heading_rts + directional`，分數回報 1.15275 的原完整 run。原 RViz log 持續出現：

```text
Message Filter dropping message: frame 'replay_source' ... discarding message because the queue is full
```

初版把點雲保留在移動的 `replay_source`，由另一條 ROS topic 提供動態 TF。RViz PointCloud2 的 MessageFilter 會等待時間吻合的 transform；pose／文字訊息已到達時，點雲仍可能卡在 TF 等待佇列。暫停後反覆發布同一 pose，讓畫面追上，所以只測暫停狀態不足以驗證連續播放。

對照來源：[RViz MessageFilterDisplay](https://github.com/ros2/rviz/blob/humble/rviz_common/include/rviz_common/message_filter_display.hpp)、[PointCloudCommon 的 transformCloud](https://github.com/ros2/rviz/blob/humble/rviz_default_plugins/src/rviz_default_plugins/displays/pointcloud/point_cloud_common.cpp)。本機實際丟棄日誌另存於本輪 evidence。

修正版對每幀執行：

```text
讀取原 PLY → 原尺度一次 → 原 input voxel
→ 用保存的完整 R、t 算 p_map = R × p_scaled + t
→ 發布 /replay/source，frame_id = map
```

因此點雲與 LiDAR 地圖已在同一固定座標，不需經動態 TF 等待才能顯示。`map → replay_source` TF 仍發布原 pose，用於視角追蹤與方向參考；XYZ／RPY 文字仍是原矩陣，不是點雲包圍盒或地面高度。每次共同發布的訊息使用相同回放時間，原 timestamp_ns 保留在 CSV／HUD。

旋轉及平移運算使用 double；輸出的 PCL XYZ 仍為 float，數公里座標的表示誤差約為毫米以下。反轉保存 pose 後，25 個 voxel centroid 與原 PLY 的獨立比對通過（容許 0.8 mm）。這不是定位精度測試。

## 64 幀真實 RViz 前後對照

相同 Track1 原 idx **150–213**、同一 map／尺度／input voxel、1×、software OpenGL、同一 RViz 設定，短段循環觀察。舊版以修改前 C++ 快照及相同 Release flags 建置獨立 executable，沒有覆寫定位結果。GUI 與 publisher 仍在同一暫存容器。

診斷過程使用暫時 ROS domain 141；另外的靜態契約核對使用 142，與使用者 domain 2 視窗隔離，未修改專案預設 domain。完成後只刪除本輪測試容器／程序。

| 指標（10 秒訂閱觀察） | 舊版 | 修正版 |
| --- | ---: | ---: |
| 接收點雲訊息 | 102 | 100 |
| 不同 XYZ 幾何 payload | 63 | 59 |
| 收到時間完全相同的 pose／cloud pairs | 102 | 100 |
| 點雲 frame | replay_source | map |
| RViz Message Filter dropping 日誌行數 | **4** | **0** |
| 最長點雲接收間隔 | 0.257 s | 0.247 s |

日誌有節流，4 行不等於只丟 4 幀。兩版的 ROS 訂閱端都能收到持續更新的點雲，舊版仍在 RViz transform filter 丟棄，支持問題位於顯示路徑。不同 topic 的渲染與軟體 OpenGL 處理仍需時間；此測試沒有證明任意硬體／任意速度永不掉幀。

核對工具為 `scripts/tests/check_replay_playback.py`；原始證據：

- [原使用者視窗日誌](../outputs/run/track00_test_2610191104/original_rviz.log)
- [同一短段的前後比較](../outputs/run/track00_test_2610191104/playback_geometry/comparison.json)
- [修正版 ROS 幾何／TF／跳幀契約](../outputs/run/track00_test_2610191107/ros_check.json)

重開回放即可使用已建置的新 executable；已啟動的舊 executable 不會因重建自動替換。原有 RViz 視角／顏色調整由使用者自行保存，修正不覆寫原設定檔。

## 高度：區分 map Z、source 原點與實際表面

畫面 XYZ 的 Z 是 **source 原點在 map 的座標值**。例如 frame 171 的 Z=108.459221 m，不是相機離地 108 m。LiDAR map 也使用相同全局 Z 基準，附近包含路面、牆面、上方結構及多個高度層，不能直接把附近所有 map 點的平均／中位 Z 當作路面。

從原 PLY 及保存矩陣讀取 frame **0／166／171／906**，保持尺度 3、原 R／t 不變，繪出局部側視：

[側視高度／形狀對照圖](../outputs/run/track00_test_2610191104/height/height_profiles.png)

圖的橫軸是沿保存 yaw 的水平距離；縱軸是 `map Z − 當幀 source 原點 Z`，以原點高度作共同參考，不改兩片點雲的相對高度。取左右各 4 m 的窄帶、前方 0–25 m、相對 Z −4 至 +6 m，保留道路附近與上方結構。藍色為影像重建，灰色為 LiDAR。可見不同位置／距離仍有高度與形狀不一致，這個現象在保存的原始幾何中就存在。

另外只做診斷：對相近 map 表面的可靠 PCA 法向，選 `|n_z|≥0.95` 的水平表面候選，最近鄰距離≤3 m；比較 `n·(p_source−p_map)/n_z`，以每個不同 map target 的 source 殘差中位數聚合，避免大量 source 點對同一 map 點重複加權。

| 原 idx | 不同水平 map targets | 有號垂直差 median | P10／P90 |
| --- | ---: | ---: | ---: |
| 0 | 7 | +0.393 m | −0.300／+1.912 m |
| 166 | 23 | +0.196 m | −0.307／+2.332 m |
| 171 | 23 | +0.151 m | −0.392／+2.328 m |
| 906 | 46 | +0.305 m | −0.500／+0.985 m |

正值代表 source 高於所選局部平面。這些候選不是人工確認的同一道路，也可能含屋頂／其他層；target 數量有限、最近鄰可能對錯面。因此數字**不能稱為真實離地高度、定位誤差，也不能推導一個固定 Z 補償**。分距離結果已留在 JSON，不把整體 median 當成整片幾何貼合。

可解釋的後續假說：固定平移只會造成各處共同高度偏移；pitch 會帶來隨距離變化的垂直差，尺度／深度形變也會改變結構尺寸與路面形狀。現有對照只支持繼續核對 source frame、深度尺度與跨時間靜態幾何，尚不能辨識單一原因。本輪沒有改動這些定位／重建設定，原有 Kaggle 分數與最佳來源不變。

分析入口使用原先隔離的 ML venv（numpy／scipy／matplotlib），不安裝進 ROS Python：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
outputs/ml_runtime/venv/bin/python scripts/diagnose_replay_height.py \
  outputs/run/track00_test_2610191054 --indices 0,166,171,906 \
  --output outputs/run/<新的唯一分析目錄>
```

[完整高度診斷與限制](../outputs/run/track00_test_2610191104/height/height_report.json)。evidence 位於被 Git 忽略的 outputs，需另行備份。

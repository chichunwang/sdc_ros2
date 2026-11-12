# 01｜Midterm02 Visual SLAM 官方規則整理（2026-11-13）

來源：課程投影片《SDC Midterm II Visual SLAM》（25 頁）。Track02 資訊官方尚未公布（TBA），公布後補在本檔末「待更新」。

## 一句話

只用前鏡頭影像（與官方預先算好的每幀 3D 點雲、內參、Track01 的 IMU），估計**每一張影像**的相機位置與方向，輸出 camera-to-world 4×4 pose 的 CSV 上傳 Kaggle。沒有 LiDAR、沒有地圖。

## 官方提供什麼

| 檔案 | 內容 | 注意 |
| --- | --- | --- |
| `images/<timestamp>.jpg` | 前鏡頭影像 1440×928 | 檔名＝影像時間戳（奈秒，整數） |
| `raw_ply/<timestamp>.ply` | 每張影像一份 LoGeR 預測點雲，相機座標、OpenCV 軸（x 右、y 下、z 前），x y z＋RGB | 只「大約」是公尺，尺度不精確且逐幀略有差異；**包含天空** |
| `camera_info.json` | 相機內參、IMU→相機旋轉 `R_cam_imu` | |
| `imu.csv` | 100 Hz，時間戳是 bag 接收時間 | 只有 Track01 |
| `sample_submission.csv` | 提交範本 | 從它取 timestamp 列表 |
| `image_slam_starter/` | 範例流程：`main.py`、`utils/ICP.py`（TODO 1 掃描匹配）、`utils/loopclosure.py`（TODO 2 閉環）、keyframe／submap／posegraph／io／viz、`process/build_map.py` | 在 Kaggle data bundle 內；環境 Python 3.9＋numpy、open3d、gtsam、small_gicp、opencv-python |

| Track | 影像數 | 頻率 | 路線 | IMU | Public／Private |
| --- | --- | --- | --- | --- | --- |
| Track01 | 1314 | 10 Hz | 約 247 m，同方向繞 3 圈 | 有（100 Hz） | 132／1182 |
| Track02 | TBA | TBA | TBA | 無 | TBA |

## 要找出什麼

每張影像的 camera-to-world pose。官方建議流程（starter code 也是這個順序）：

```text
前端（每張）：影像 → LoGeR 點雲 → 前處理（降採樣、距離裁切、去雜訊/去天空）→ 掃描匹配 ICP/GICP/NDT → 里程計
後端（keyframe）：每約 1 m 一個 keyframe＋局部 submap → 閉環偵測與驗證 → pose graph 全域最佳化 → 每張影像的 pose → submission.csv
```

## 提交格式

```text
timestamp,m00,m01,m02,m03,m10,m11,m12,m13,m20,m21,m22,m23,m30,m31,m32,m33
1741574911912996024,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1
```

- 一張影像一列，非 keyframe 也要有；缺列、重複、非數值會被拒收。
- 相機軸為 OpenCV。只評位置 `m03,m13,m23`，但旋轉必須是合法 pose。

## 評分：Sim(3) 對齊後的 ATE

1. 先把你的整條軌跡做一次 Sim(3) 對齊（Umeyama：平移 t、旋轉 R、尺度 s，共 7 自由度，整條一組）。
2. 每張 `e_i = ‖s·R·p_i + t − g_i‖`，`ATE = sqrt(Σ e_i² / N)`，單位公尺，越低越好。GT 是 LiDAR pose 轉到相機位置，看不到。
3. **免費的**：起點、軸向、整體尺度。**會被罰的**：累積漂移、轉角錯、路段被拉長或壓縮、繞圈沒閉合。平方使少數大誤差影響很大。
4. Public（約 10%，同一小段路線）與 Private（約 90%）各自對齊、各自計分。短段很好擬合，Public 好不代表 Private 好；最終只看 Private。

## 怎樣得高分

| 項目 | 配分 |
| --- | --- |
| 競賽排名（兩 Track 各 r=0.5 相加） | 最高 56 |
| 報告（TA） | 30 |
| 簡報（4–6 分鐘，11/26 課堂） | 20 |

| Private ATE 名次（只計優於 baseline 的隊伍） | 每 Track 分數 |
| --- | --- |
| 第 1／2／3 名 | 56r／52r／48r |
| 前 50%／前 75% | 42r／36r |
| 其他優於 baseline | 30r |
| 遲交（至 12/03 23:59）且優於 baseline | 22r |
| 未優於 baseline | 0 |

技術上最有效的方向（依評分公式推得，尚未驗證）：

- **閉環最重要**：Track01 同方向繞 3 圈，閉環＋pose graph 能直接消除累積漂移與「圈不重合」。
- **尺度一致性**：Sim(3) 只能修正一個全域尺度；LoGeR 逐幀尺度差會造成路段伸縮，需在配準或圖最佳化中處理（例如 Sim(3) 邊或逐幀尺度估計）。
- **前處理**：去天空、距離裁切、降採樣，避免天空與遠處噪點拖壞 ICP。
- **IMU（Track01）**：提供旋轉先驗與尺度觀測的可能來源，注意時間戳是接收時間。
- **避免離群大誤差**：RMSE 平方懲罰，失敗幀（跳躍）比整體小偏差更傷。
- 用 `build_map.py` 看整條路線地圖（重影、彎牆）判斷，不要追 Public 分數。

報告／排名的「貢獻」條件：可用開源系統（ORB-SLAM3、MASt3R-SLAM、VGGT-SLAM、DROID-SLAM）或預訓練模型（LoGeR、MapAnything、VGGT、特徵匹配器）當起點，但**必須加入自己的想法**並列出每項修改前後分數；只跑原版、只調參數、多跑挑最好 → 貢獻 0 分、不排名（以遲交計分）。

報告需含：Pipeline、Contributions、問題與解法（含實驗）、使用的系統／模型／函式庫與版本、從現成系統出發時的分數演進、組員工作量比例。

## 允許與禁止

- 允許：Kaggle 頁提供的所有資料、任何公開預訓練模型與開源函式庫、開源系統作為起點、AI 程式工具。
- 禁止：其他感測器、外部地圖、GPS、衛星影像、往年資料；手改 pose；跨隊分享程式或提交。
- 不確定是否允許：先到 E3 Competition2 QA 討論區詢問。

## 時程與行政

| 事項 | 期限 |
| --- | --- |
| 分組表（最多 2 人，不能與 Competition 1 組員同組） | 11/13 15:00 |
| Kaggle 截止、簡報 email（pj.ee14@nycu.edu.tw） | 11/26 09:00（UTC+8） |
| 課堂簡報 | 11/26 |
| 遲交 Kaggle 與 E3 上傳 | 12/03 23:59 |

- 每人一個 Kaggle 帳號；隊名 `<學號1>_<學號2>`；**先建隊並邀請隊友再提交**（合併時若兩人提交數合計超過每日上限會被拒）。
- 每日最多 10 次提交（08:00 台灣時間重置）；最多選 2 份最終提交，取較好者，兩份都必須能用 E3 上的程式重現。
- E3：`<ID1>_<ID2>_vslam.zip`，含整份程式（不含原始資料）、`result_1.csv`、`result_2.csv`、`vslam_report_<ID1>_<ID2>.pdf`（可附簡報影片連結）、說明如何重現的 README。

## 待更新

- Track02 影像數、頻率、路線、Public／Private 比例與 Kaggle 連結：官方 TBA。
- baseline ATE 數值：見各 Kaggle leaderboard 的 baseline 列，取得後記錄於此。

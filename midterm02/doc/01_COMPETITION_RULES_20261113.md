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
2. 每張 `e_i = ‖s·R·p_i + t − g_i‖`，`ATE = sqrt(Σ e_i² / N)`，單位公尺，越低越好。GT＝影像時刻的 LiDAR pose × LiDAR-to-camera 外參，學生看不到。等同 `evo_ape --align --correct_scale`。往年 Track1 範例 ATE＝0.64 m（不同資料，僅供感覺量級）。
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
- 同分取較高名次；名次只計贏過 baseline 的隊伍。

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

## 投影片逐頁要點（25 頁）

上面是依主題整理；這裡依投影片順序列出每頁要點，方便對照原檔。Kaggle 頁面上的規則與資料說明原文見 [doc/02](02_KAGGLE_RULES_TRACK01_20261113.md)。

| 頁 | 標題 | 要點 |
| --- | --- | --- |
| 1 | 封面 | SDC Midterm02 Visual SLAM，2026/11/12 – 11/26 |
| 2 | Demo | 左：相機影像逐幀播放；右：逐幀建出的俯視地圖，畫出估計軌跡與目前相機位置。這是期望的成果樣貌 |
| 3 | Task | 估計每張前鏡頭影像的相機 pose（位置＋方向），用相機、可選 IMU。輸入：影像、每張一份 LoGeR 點雲、內參、IMU（僅 Track01）。要自己做：掃描匹配（odometry）、keyframe、loop closure、pose graph 最佳化。輸出：每張一個 4×4 camera-to-world，CSV 上傳 Kaggle。**不提供 LiDAR** |
| 4 | Flow | 前端（每張）：2D 影像 → LoGeR 3D 點雲（官方給）→ 前處理（降採樣、距離裁切、去雜訊）→ 掃描匹配（ICP、GICP、NDT）得 odometry。後端（keyframe）：keyframe（例如每 1 m 一個，附局部 submap）→ 閉環（找回訪地點、估計轉換）→ pose graph（一起最佳化所有 keyframe pose）→ 每張影像的 pose → `submission.csv`。starter 程式照這個流程寫 |
| 5 | 章節頁 | Dataset & Code |
| 6 | Dataset | 兩條 Track，各自一個 Kaggle 競賽。Track01：1314 張、10 Hz、約 247 m、同方向 3 圈、IMU 100 Hz；Track02：TBA、無 IMU。影像 1440×928。world frame 與軌跡尺度可任意 |
| 7 | Dataset · files | `images/`、`raw_ply/`、`camera_info.json`、`imu.csv`（僅 Track01）、`sample_submission.csv`；檔名＝奈秒時間戳。raw_ply：相機座標、OpenCV 軸、xyz＋RGB；**只大約是公尺、尺度不精確且逐幀略異、含天空**。`camera_info.json` 含內參與 `R_cam_imu`；`imu.csv` 時間戳是 bag 接收時間。細節見 starter README |
| 8 | Provided code | `main.py`：整條流程 → `submission.csv`；`utils/ICP.py`＝TODO 1（當前點雲對局部 submap 配準）；`utils/loopclosure.py`＝TODO 2（找候選、驗證、估計閉環轉換）；另有 keyframe、submap、posegraph、io、viz；`process/build_map.py` 用你的 pose 拼接點雲。環境 Python 3.9＋numpy、open3d、gtsam、small_gicp、opencv-python。**用 build_map.py 看地圖：軌跡錯誤會出現重影的牆或彎曲的牆** |
| 9 | 章節頁 | Competition |
| 10 | Competition | 團隊賽，最多 2 人。設計自己的 visual SLAM：可改任何範本、可用其他套件／公開預訓練模型／開源系統當起點，也可從零寫。**必須在 Private 榜贏過 baseline 才有分**；贏過 baseline 得分，名次越好分越多。baseline 是 Kaggle 榜上的 baseline benchmark 列 |
| 11 | Allowed / Not allowed | 允許：Kaggle 頁提供的所有資料；公開預訓練模型（LoGeR、MapAnything、VGGT、特徵匹配器）與開源函式庫；開源系統當起點（ORB-SLAM3、MASt3R-SLAM、VGGT-SLAM、DROID-SLAM）；AI 程式工具。禁止：其他感測器、外部地圖、GPS、衛星影像、往年資料；手改 pose；跨隊分享程式或提交。只用現成系統沒有自己的想法：貢獻 0 分、不排名。**要呈現改了什麼、每項改動讓分數進步多少**。助教可能重跑，無法重現的結果不算 |
| 12 | Join | Track01 Kaggle 連結；Track02 TBA。每人一個 Kaggle 帳號，隊名 `<學號1>_<學號2>`；先建隊、邀隊友再提交。每日最多 10 次（台灣 08:00 重置）。最多選 2 份最終提交，取較好者，兩份都要能用 E3 程式重現。截止 2026/11/26 09:00（UTC+8） |
| 13 | Submission file | 每列：timestamp（影像檔名去 `.jpg`）＋ 4×4 camera-to-world 按列攤平。OpenCV 相機軸。每張都要有 pose（不管是不是 keyframe），缺列／重複／非數值會被拒。只評位置 `m03,m13,m23`，旋轉仍須是合法 pose。從 `sample_submission.csv` 開始改 |
| 14 | 章節頁 | Evaluation：Sim(3) 對齊的 ATE |
| 15 | 提交與比較 | 你提交每張影像的相機位置與朝向，連起來就是車子走的路線。GT 是用 LiDAR 量的真實路線，轉換到每張影像時刻的相機位置（**GT＝影像時刻的 LiDAR pose × LiDAR-to-camera 外參**），學生看不到。分數衡量路線「形狀」多接近，單位公尺，越低越好，只看位置 |
| 16 | 為何不能直接相減 | 影像本身無法得知三件事：① 起點（你把第一張放原點，GT 在 LiDAR 地圖座標）→ 平移 t；② 朝向（world 軸任意）→ 旋轉 R；③ 大小（影像分不出真實街道或 10 倍縮小模型，點雲也只大約是公尺）→ 尺度 s。所以先把整條路線一起平移、旋轉、縮放去貼 GT：Sim(3)，3＋3＋1＝7 自由度 |
| 17 | 步驟 ①–③（往年 Track1 範例） | ① 原始兩條路線：學生的較小、轉了約 120°、位置不同、有漂移。② 兩條各減去平均位置，中心都移到原點。③ 用一次 SVD 求讓兩個形狀最重合的旋轉 R，方向對了但仍太小。方法：Umeyama (1991)，封閉解，每次答案相同 |
| 18 | 步驟 ④–⑤ | ④ 最佳尺度 s＝兩形狀重疊程度 ÷ 學生形狀大小，範例 s＝2.12。⑤ 平移 t 把學生中心放到 GT 中心，量每張的位置誤差。`e_i = ‖s·R·p_i + t − g_i‖`，`ATE = sqrt(Σe_i²/N)`；平方使少數離很遠的影像代價很大。範例（往年資料）ATE＝0.64 m。s、R、t 整條路線只算一次，不是每張各算 |
| 19 | 對齊會／不會消除什麼 | 免費消除：world 起點、軸向、整體尺度，只動這 7 個數。仍會扣分：沿路累積的漂移、轉角角度錯、路段被拉長或壓縮、繞圈沒閉合。對每隊都是最佳擬合，所以公平：評的是路線形狀 |
| 20 | Public／Private | Public 約 10%，全部來自路線的某一段，比賽中可見，僅供參考；Private 約 90%，截止後公布，**期末評分用 Private**。兩部分**各自對齊**（各自的 s、R、t）、各自計分。短段容易擬合，Public 好不代表 Private 好。用 build_map.py 看整條路線來判斷方法好壞。Track01：132 Public／1182 Private |
| 21 | Grading | 競賽排名最高 56、報告（TA 評）30、簡報 20。每 Track 依 Private ATE：第 1／2／3 名 56r／52r／48r；前 50% 42r；前 75% 36r；其他贏 baseline 30r；遲交且贏 baseline 22r；未贏 baseline 0。每 Track r＝0.5，兩 Track 相加；名次只計贏 baseline 的隊伍。遲交至 12/03 23:59；**同分取較高名次** |
| 22 | 使用現成系統的評分 | 有自己的想法 → 排名：可從任何開源系統或預訓練模型出發，找出它在本資料哪裡失敗並修正；依 Private 排名；報告寫改了什麼、為什麼、每項改動前後分數。沒有自己的東西 → 不排名：原樣執行、只調參數／config、多跑挑最好；貢獻 0 分；該 Track 以遲交計（贏 baseline 22r，否則 0）；後面的隊伍名次往前遞補。選定提交必須能用 E3 上的程式重現，助教會重跑核對分數。不確定是否算自己的想法，到 E3 QA 問 |
| 23 | 報告與簡報 | 報告：Pipeline（程式怎麼運作）、Contributions（原創想法與改進）、問題與解法（附實驗證明什麼有效）、使用的系統／模型／函式庫與版本、若從現成系統出發則列原始分數 → 每項改動後分數、Workload（每位組員工作與比例）。簡報：4–6 分鐘、11/26 課堂、介紹成果與貢獻、11/26 09:00 前寄簡報或連結到 pj.ee14@nycu.edu.tw。貢獻依原創性與解決問題能力評分 |
| 24 | E3 submission | 一個檔案 `<學號1>_<學號2>_vslam.zip`：整份程式（不含原始資料）、各 Track 最佳提交 `result_1.csv`、`result_2.csv`、報告 `vslam_report_<學號1>_<學號2>.pdf`（可附簡報影片連結）、說明如何執行並重現兩份 CSV 的 README。截止 12/03 23:59 |
| 25 | 分組與提問 | 每組最多 2 人，填分組表；需要隨機組員就填表右側；不可與 Competition 1 組員同組。分組截止 11/13 15:00。問題到 E3 Competition2 QA 討論區；不確定模型或工具是否允許，**使用前**先問 |

## 待更新

- Track02 影像數、頻率、路線、Public／Private 比例與 Kaggle 連結：官方 TBA。
- baseline ATE 數值：見各 Kaggle leaderboard 的 baseline 列，取得後記錄於此。

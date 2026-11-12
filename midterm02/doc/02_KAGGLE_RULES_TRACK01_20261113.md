# 02｜Kaggle Track01 規則與說明原文紀錄（2026-11-13）

來源：使用者於 2026-11-13 貼上的 Kaggle Track01 競賽頁「Rules」與「Description」。先列中文重點與相較投影片的新資訊，原文完整保存在文末附錄，以原文為準。

## 相較投影片（doc/01）新增或更精確的資訊

| 項目 | 內容 |
| --- | --- |
| 每日提交重置 | 00:00 UTC（＝台灣 08:00），每日最多 10 次 |
| 未選最終提交 | Kaggle 會自動挑 Public 分數最好的提交當最終提交 → **務必自己選**，因 Public 只代表一小段路線 |
| 評分等價工具 | 與 `evo_ape --align --correct_scale` 相同（Umeyama Sim(3) 對齊後位置 RMSE，單位公尺） |
| 禁止：真值與額外資料 | 不得使用 GT 或任何「測試序列」的非官方提供資料 |
| 禁止：帳號 | 不得一人多帳號，不得透過其他隊伍帳號提交 |
| 違規後果 | **整場比賽 0 分** |
| 資料授權 | 資料僅供本課程比賽使用，未經助教許可不得散布或作他用（含 bundle 內的 starter code） |
| 報告須寫明 | 使用的系統、模型、程式碼及版本 |
| LiDAR 用途 | 只用來建立隱藏的 GT 軌跡，不提供給學生 |

## 中文重點整理

1. **隊伍**：最多 2 人，不可與 Competition 1 組員同隊；每人一個 Kaggle 帳號，隊名 `<學號1>_<學號2>`；先合併隊伍再提交。
2. **提交**：每日 10 次；最多選 2 份最終提交，取 Private 較好者；選定的提交必須能用 E3 上的程式重現，助教會重跑核對分數；截止 2026/11/26 09:00（UTC+8）；Public 約 10%、同一段路線，僅供參考。
3. **允許**：公開預訓練模型（LoGeR、MapAnything、VGGT 等）、開源函式庫、開源 SLAM／SfM 系統當起點（ORB-SLAM3、MASt3R-SLAM、VGGT-SLAM、DROID-SLAM 等）、AI 程式工具。
4. **沒有自己想法**（原樣執行、只調參數／設定檔、多跑挑最好）：報告貢獻 0 分、不排名，該 Track 以遲交計（優於 baseline 22r，否則 0）。不確定時到 E3 QA 問。
5. **禁止**：手改或手標預測／pose；使用 GT 或非官方提供的測試序列資料；跨隊分享程式或提交；多帳號或借用他隊帳號。違規整場 0 分。
6. **資料**：僅供本課程比賽，不得散布或他用。
7. **程式與報告**：依投影片上傳 E3；無法由繳交程式重現的結果不計分。
8. **評分表**（r=0.5／Track）：第 1／2／3 名 56r／52r／48r；前 50%／75% 42r／36r；其他優於 baseline 30r；遲交（至 12/03 23:59）且優於 baseline 22r；未優於 baseline 0。

## 對本專案的影響

- `data/`（含資料包內的官方 starter 原檔）維持在 `.gitignore`，不得推到公開遠端或分享。從 starter 複製到 `vslam/` 的程式若要推到遠端，遠端須為私人倉庫。
- 本機 Sim(3) 比較工具應與 `evo_ape --align --correct_scale` 一致，可用 evo 交叉驗證。
- 最終兩份提交由使用者在 Kaggle 手動選定，不可留給 Kaggle 自動選。

## 資料包結構（Kaggle Data 頁）

```text
data_VSLAM-T1/
├── Track1/data/
│   ├── raw_ply/<timestamp>.ply     LoGeR 點雲：相機座標、OpenCV 軸、約略公尺、含天空；x y z + RGB（取自影像）
│   ├── images/<timestamp>.jpg      前鏡頭 1440×928
│   ├── camera_info.json            內參、R_cam_imu
│   ├── imu.csv                     100 Hz（bag 接收時間）
│   └── sample_submission.csv       提交格式（全為單位矩陣）
└── image_slam_starter/             main.py、utils/、process/build_map.py、README.md
```

環境：Python 3.9；套件見 `image_slam_starter/README.md`（numpy、open3d、gtsam、small_gicp、opencv-python）。

本專案放置方式：整包原樣解壓到 `midterm02/data/`，得到 `midterm02/data/data_VSLAM-T1/…`；starter 原檔就留在資料包內（唯讀、被 Git 忽略），修改版另複製到 `midterm02/vslam/`。

## 附錄：原文（使用者貼上，2026-11-13）

### Rules

1. Teams

* At most 2 people per team. You may not team up with anyone you were grouped with in Competition 1.
* One Kaggle account per person. Set your team name to `<student ID1>_<student ID2>`.
* Merge your team before submitting: Kaggle refuses a merge when the members' combined submissions exceed the daily limit.

2. Submissions

* At most 10 submissions per day (the day resets at 00:00 UTC = 08:00 Taiwan time).
* Select up to 2 final submissions; the better of them counts on the Private leaderboard. If you select none, Kaggle picks your best Public ones.
* Your selected submissions must be reproducible with the code you upload to E3. The TAs will rerun your code and check that the score matches.
* Deadline: 2026/11/26 09:00 (UTC+8).
* The Public leaderboard uses about 10% of the images, all from one part of the route, and is for reference only. Final grading uses the Private leaderboard.

3. Allowed and not allowed

* Allowed: any public pretrained model (e.g. LoGeR, MapAnything, VGGT), any open-source library, and open-source SLAM / SfM systems as a starting point (e.g. ORB-SLAM3, MASt3R-SLAM, VGGT-SLAM, DROID-SLAM). AI coding tools (e.g. ChatGPT, Copilot) are also allowed. Say in your report which systems, models and code you used, with versions.
* Existing systems without your own ideas: if you run an existing system without your own ideas (for example running it unchanged, or only tuning its parameters / config files, or running it many times and submitting the best run), you get 0 points for the contribution part of the report and no ranking points: that track is graded like a late submission (22r if better than the baseline, otherwise 0), and the teams behind you move up in the ranking. Not sure whether something counts as your own idea? Ask on the E3 QA forum; the TAs and the professor will decide.
* Not allowed:
   * hand-editing or hand-labelling predictions or poses;
   * using ground truth or any data about the test sequences other than what is provided on this page;
   * sharing code or submissions between teams;
   * using more than one account, or submitting through another team's account.
* Violations give 0 points for this competition.

4. Data

The data is provided only for this course competition. Do not redistribute it or use it for other purposes without the TAs' permission.

5. Code and report

Upload your code, best submission files and report to E3 as described in the course slides. The TAs may rerun your code: results that cannot be reproduced from the submitted code do not count.

6. Questions

Use the Competition2 QA forum on E3.

### Description

SDC 2026 Fall, Midterm Competition II: Visual SLAM, Track01.

You get the front-camera images of one drive, a 3D point cloud predicted from each image (LoGeR), the camera intrinsics and the IMU. No LiDAR data is given; LiDAR is used only to build the hidden ground-truth trajectory. Build a SLAM pipeline (scan matching, keyframes, loop closure, pose graph) and estimate the camera pose of every image. The starter code (image_slam_starter/ in the data) has the pipeline skeleton; scan matching (utils/ICP.py) and loop closure (utils/loopclosure.py) are left as TODOs.

Track01: 1314 images at 10 Hz, about 247 m, three laps in the same direction.

Any public pretrained model, any library, and open-source SLAM / SfM systems (for example ORB-SLAM3, MASt3R-SLAM, VGGT-SLAM, DROID-SLAM) may be used as a starting point. Your own ideas on top of them are what count: a team that runs an existing system without its own ideas or changes gets no ranking points (see the Rules).

Grading: graded on the Private leaderboard. Lower ATE is better. Only teams better than the baseline benchmark on the leaderboard get ranking points, and ranks count only those teams (r = 0.5 for each track):

| Your Private ATE | Points |
| --- | --- |
| 1st / 2nd / 3rd | 56r / 52r / 48r |
| top 50% / top 75% | 42r / 36r |
| other teams better than the baseline | 30r |
| late submission (until 12/03 23:59), better than the baseline | 22r |
| not better than the baseline | 0 |

The Public leaderboard uses about 10% of the images, all from one part of the route; the Private leaderboard uses the other 90%.

Evaluation: submissions are scored with the ATE (absolute trajectory error): the RMSE, in metres, of the camera positions (m03, m13, m23) after the best similarity alignment (rotation, translation and scale; Umeyama) of your trajectory to the ground truth, the same as evo_ape --align --correct_scale. Lower is better. Your world frame and the scale of your trajectory can be anything.

Submission File: one row per image. timestamp = the image file name (without .jpg), followed by the 4×4 camera pose (camera → world, OpenCV camera axes), row by row:

```text
timestamp,m00,m01,m02,m03,m10,m11,m12,m13,m20,m21,m22,m23,m30,m31,m32,m33
1741574911912996024,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1
...
```

Every image needs a pose. Missing, duplicated or non-numeric rows are rejected.

### Dataset Description

Files

* data_VSLAM-T1/Track1/data/
   * `raw_ply/<timestamp>.ply`: point cloud predicted from each image by LoGeR (camera frame, OpenCV axes, only roughly in metres, sky included; x y z + RGB colour from the image), file name = image timestamp in ns
   * `images/<timestamp>.jpg`: the front-camera images (1440 × 928)
   * `camera_info.json`: camera intrinsics and the IMU-to-camera rotation `R_cam_imu`
   * `imu.csv`: IMU at 100 Hz (bag receive time)
   * `sample_submission.csv`: the submission format (all identity poses)
* data_VSLAM-T1/image_slam_starter/: starter code (`main.py`, `utils/`, `process/build_map.py`, `README.md`)

Environment

Python 3.9; see `image_slam_starter/README.md` (numpy, open3d, gtsam, small_gicp, opencv-python).

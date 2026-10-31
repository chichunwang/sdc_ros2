# HW4｜Application of EKF

這份作業用 Extended Kalman Filter（延伸卡爾曼濾波器，EKF）融合 radar odometry 的平面運動與 GPS 的平面位置，在 RViz2 顯示估計軌跡。`/gt_odom` 只作結果檢查，不可參與狀態初始化或濾波更新。

程式位於 `hw4/colcon_ws/src/fusion/scripts/`，沿用 HW2 的 ROS 2 Humble Docker。原 `EKF.py` 核心算法未在本輪更改；加入了 run 來源快照、實際 covariance 紀錄和 GT 時間對齊評估。Ubuntu/HW2 ROS 2 Humble 實測與 RViz 錄影已完成，YouTube 影片為 <https://youtu.be/nmfKO3pA5y8>；PDF 已依使用者紅線刪除兩段說明並完成打包，`output/112611054_hw4.zip` 內四檔已核對；E3 尚未上傳。學號已設定為 `112611054`。

### Ubuntu 實測紀錄（2026-10-29）

- 容器 `ros2_container` 已以目前 worktree 掛載重新建立；舊 image `104f7d...` 已恢復 `sdc_hw2_ros2` 標籤。相依安裝成功，colcon build 成功（1.06 s），ROS Humble 容器內 17 項測試通過（1.360 s）。
- 官方 README bag 解壓至 `hw4/colcon_ws/data/sdc_hw5_ros2/`，保留原名稱。ZIP 為 1,223,510,833 bytes，SHA256 `5b20e6c453812d848f1e71ce4282f0079c41a1c993cd947b498dc7264acf8488`；DB3 為 1,790,959,616 bytes。
- 完整 80 秒 run：`hw4/colcon_ws/result/run_261029_231926_g9rH/`。Radar 973、GPS 64、GPS update 64、late GPS 0、invalid 0、non-increasing radar 0、estimate 1037、GT 實收 7544；原 bag 有 7714 筆 GT，best effort 接收少 170 筆，故不表示全部收到。GPS fallback 0、radar fallback 1。radar／GPS／GT 的 header.frame_id 均為 `origin`，bag 有 `/clock`。
- GPS 原始 covariance 對角為 `(3, 3) m²`。radar 實際選用 covariance 對角統計最小值全為 0，最大值為 `[9.380944161956033e-05, 6.072459584673981e-05, 0.0006405714870086814]`。結果比較依各 topic 實際時間戳處理；GT 原始時間有 1526 筆非遞增（包含重複），評估對整數 timestamp 去重並保留最後值，再線性插值、不外推。
- 記錄的 RMSE：radar 34.30278999808334 m、GPS 0.46098894599053886 m、estimate 1.0322084968142409 m。三者時間戳樣本不完全相同，不能解讀為嚴格同樣本比較，也不能據此宣稱 EKF 比 GPS 更準；GT 僅用於結果評估。
- 聚合證據在 `../midterm01_gpu/outputs/run/track00_test_2610292317/`；較早驗證輪的 Ubuntu 截圖在 `hw4/output/evidence/ubuntu_2610292317/`。該輪完成後曾停在 RViz 等待錄影；錄影已於下方獨立一輪完成。

### 錄影與 PDF 輪（2026-10-29）

- 審閱用 PDF 對應新 run `hw4/colcon_ws/result/run_261029_232220_oYEV/`，與上方先前實測 run 分開保存。影片：https://youtu.be/nmfKO3pA5y8；終端身分截圖為該 run 的 `06_terminal_recording_identity.png`。
- 新 run 計數：radar 961、GPS 63、estimate 1024、GT 6648；bag 原有 radar 973、GPS 64、GT 7714，因此分別少 12、1、1066 筆。GPS update 63、late GPS 0、non-increasing radar 0、invalid 0、pending GPS 0。雷達首末時間與 bag 跨度 79.615616084 s 吻合；fusion 收到 Ctrl+C 後 clean exit 0，player shell exit 0，但 log 沒有明確自然播畢訊息，不能宣稱完整收到所有訊息或自然播畢。
- RMSE：radar 34.346932618594955 m、GPS 0.4634891799869087 m、estimate 1.035346041665754 m；estimate P95 為 1.665499907427702 m。GT timestamp 去重後 5340 筆，最大線性插值跨距 0.14985 s；各資料時間戳樣本不完全相同，不作嚴格同樣本比較，也不據此宣稱 EKF 比 GPS 更準。GT 僅作評估。
- GPS covariance 對角為 `(3, 3) m²`、GPS fallback 0 次、radar fallback 1 次。run source manifest SHA 核對相符。PDF 共 5 頁，已依使用者標記刪除兩段文字、重新完成視覺 QA，並依授權打包。ZIP 內 PDF 與最終報告 bytes 相同，結果圖與兩份程式的 SHA 保持原 run 版本。

## 在 Ubuntu 執行

以下指令在 Ubuntu 主機的專案 checkout 執行。若資料夾位於使用者原路徑，從 repo 根目錄開始：

```bash
cd /home/ricky/sdc_ros2
git status --short --branch
cd hw4
```

如果程式還沒同步到 Ubuntu，先切換該 checkout 的 `hw4` 分支，並同步 `hw4/` 和 `hw2/docker-compose.yml`。Windows 的 `Code Base/` 目錄不是 Ubuntu 容器內同一份檔案；bag 資料也不隨 Git 同步。確認 HW2 沒有正在執行的工作後，再套用本專案的掛載：

```bash
bash scripts/dev.sh recreate
bash scripts/dev.sh deps
bash scripts/dev.sh build
bash scripts/dev.sh test
```

`recreate` 只重建 HW2 的 `ros2_dev` 容器執行個體，以套用 `../hw4/colcon_ws:/ros_ws/hw4` 與 `../hw4:/ros_ws/hw4_tools` 兩個掛載；後者讓同一個容器能讀取交付設定、證據和報告程式。不重建 image，也不會操作 Midterm 容器。若 Docker 需要管理員權限，使用 `sudo bash scripts/dev.sh ...`。此流程沿用現有 ROS 2 Humble 環境，不另裝 ROS、WSL 或 Docker。

### 安裝題目 rosbag

手動下載[老師提供的 bag ZIP](https://drive.google.com/file/d/1Ei_dFm24jZSidt-asCZctTm_572b9kkw/view)，解壓至 `hw4/colcon_ws/data/`。保留壓縮檔內的 bag 資料夾、資料庫分片及原始名稱；講義在不同段落分別使用 `sdc_hw4_ros2` 和 `sdc_hw5_ros2`，程式以 `metadata.yaml` 找到 bag，不需改名。

```bash
mkdir -p colcon_ws/data
unzip ~/Downloads/sdc_hw4_rosbag.zip -d colcon_ws/data
find colcon_ws/data -name metadata.yaml -print
bash scripts/dev.sh info
bash scripts/dev.sh inspect
```

若資料夾中有多個 bag，對 `info`、`inspect`、`play` 明確傳入含 `metadata.yaml` 的主機端目錄：

```bash
bash scripts/dev.sh inspect "$PWD/colcon_ws/data/sdc_hw5_ros2"
```

`inspect` 會讀取 bag 的 `/radar_odometry`、`/gps`、`/gt_odom`，確認訊息型別、frame、時間與 covariance。若 frame 與 launch 的 `fixed_frame`（預設 `origin`）不同或空白，先核對座標契約，不能直接把不同座標當作相同座標融合。請用實際 inspect 數字回答 Discussion 第二題，不要把程式的 fallback 假設寫成 bag 提供的數值。

### 播放與保存

開兩個 Ubuntu 終端機，兩者都在 `.../sdc_ros2/hw4`。終端機 1 啟動 EKF 和 RViz2：

```bash
printf 'SDC HW4 | Student ID: 112611054\n'
bash scripts/dev.sh launch
```

看到 `EKF ready` 後，在終端機 2 播放一次 bag：

```bash
bash scripts/dev.sh play
```

若 bag 資料夾名稱不同，使用剛才查到的路徑，例如 `bash scripts/dev.sh play "$PWD/colcon_ws/data/sdc_hw4_ros2"`。bag 已含 `/clock`，播放命令加 `--delay 5`，**不要加 `--clock`**。不要在同一個 fusion 程序重播；要再跑一次，先在終端機 1 按 `Ctrl+C` 保存結果，然後重啟 fusion。

錄製 RViz2 畫面，並截取能辨識本人與學號的執行終端機。若 RViz 的 Fixed Frame 或追蹤的 `car` TF 不存在，先用 bag 資料和 RViz Views 檢查；不要悄悄變更融合 frame。需要暫時開放 GUI 時，Ubuntu 桌面終端可執行 `xhost +si:localuser:root`；完成後用 `xhost -si:localuser:root` 撤回。沒有桌面畫面的 SSH 工作階段使用 `bash scripts/dev.sh launch use_rviz:=false`。

bag 播放結束後，在終端機 1 按 `Ctrl+C`。每輪會保存在不同的 `colcon_ws/result/run_<台北時間>_<唯一尾碼>/`，不覆蓋上一輪。除了 `result.png`、感測器 CSV 和 `summary.json`，run 會存下來源碼與 manifest、covariance 統計、GT 時間對齊分析及誤差圖，供報告產生器確認本輪來源。檢查非零 GPS 更新與估計樣本數、frame／時間、covariance 統計及 `analysis.json`；圖表存在只表示有產生圖檔，不能單獨證明融合精度。

## EKF 設計與實測判讀

狀態為平面位置與航向 $[x,y,\psi]^T$。radar 給連續的 pose 增量；GPS 只量測 $(x,y)$。範圍、矩陣與 Jacobian 推導見[設計說明](doc/EKF_DESIGN.md)。實測 covariance、時間對齊的結果檢查和報告交付見[交付手冊](doc/SUBMISSION.md)。

## 提交

官方 HackMD 頁仍顯示 `10/28 23:59`；E3 作業頁截圖顯示 **2026-10-29 23:59**，請以 E3 實際提交頁為準。壓縮檔內需有四個檔案：`112611054_hw4.pdf`、實測 `result.png`、`fusion.py` 和 `EKF.py`；PDF 要回答兩題 Discussion、放實測圖、本人終端身分截圖與 RViz2 YouTube 連結。錄影、影片 URL 與 PDF 已完成；使用者指定刪文並授權打包後，已建立四檔正式 ZIP。E3 尚未上傳。

報告設定檔、PDF 審閱與四檔 ZIP 流程見[交付手冊](doc/SUBMISSION.md)。最終 5 頁 PDF 與四檔 ZIP 已完成；ZIP 內 PDF bytes 與刪文後檢查的版本一致。檔案為 `output/pdf/112611054_hw4.pdf`、`output/112611054_hw4.zip`，核對紀錄為 `output/submission_manifest.json`。E3 尚未上傳。

## 來源與驗證紀錄

題目：[SDC Assignment 4](https://hackmd.io/muS7dwNbQjemnah98PLNRA)，本輪於 2026-10-29 重新讀取。程式基底：[老師的 sdc_hw4.zip](https://drive.google.com/file/d/1HUce6pQXLxd4QsVbJcQ9A70JSxe-JyQG/view)，原始 ZIP SHA256 為 `e01c33c59557081cdc668ad92eff33121cc8b7342955a29a62f0acb6d27507e0`。

離線檢查與合成輸出保存於分支的 `midterm01_gpu/outputs/run/`，不隨 Git 同步；其合成圖表不供作業繳交。原有 10 項數學／時序檢查保留，新測試核對 GT 隔離、整數 epoch 時間對齊、來源 SHA、拒絕覆寫與 ZIP 四檔契約。

較早階段的本機離線檢查有 17 項通過（10 項數學／時序、7 項 ROS stub／輸出／交付機制），unittest 計時 3.106 秒。紀錄：[unittest.log](../midterm01_gpu/outputs/run/track00_test_2610291901/unittest.log)；合成圖表與打包 fixture：[analysis](../midterm01_gpu/outputs/run/track00_test_2610291901/analysis/)。該階段未執行 Ubuntu colcon、ROS node 或真實 bag；其後已另完成上方所列 Ubuntu build、ROS 測試及真實 bag run。Python 語法、package XML、Compose YAML、掛載及 shell 語法檢查屬較早的靜態／離線檢查。

較早交接階段曾產生 `output/112611054_hw4_source_update.zip` 供同步程式（忽略輸出不隨 Git 同步，本 Ubuntu checkout 未提供此包）：它包含 `hw4/` 的程式／文件和 `hw2/docker-compose.yml`，沒有 bag 或實測結果。若 Ubuntu checkout 仍在 `/home/ricky/sdc_ros2`，先確認工作樹已備份且沒有 HW2 程序，再將程式更新包解壓到該 checkout 根目錄。這是來源更新包，作業交付仍依上面的四檔規則產生。

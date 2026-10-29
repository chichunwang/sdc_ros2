# HW4｜Application of EKF

這份作業用 Extended Kalman Filter（延伸卡爾曼濾波器，EKF）融合 radar odometry 的平面運動與 GPS 的平面位置，在 RViz2 顯示估計軌跡。`/gt_odom` 只作結果檢查，不可參與狀態初始化或濾波更新。

程式位於 `hw4/colcon_ws/src/fusion/scripts/`，沿用 HW2 的 ROS 2 Humble Docker。原 `EKF.py` 核心算法未在本輪更改；加入了 run 來源快照、實際 covariance 紀錄和 GT 時間對齊評估。本機離線合成檢查涵蓋 EKF 數學／時序、ROS stub 輸出整合及四檔 ZIP，詳見文末驗證紀錄；這不是 ROS 或真實 bag 驗證。目前沒有真實 bag 執行結果、RViz 截圖或錄影，也尚未確認實測精度。學號已設定為 `112611054`。

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

官方 HackMD 頁仍顯示 `10/28 23:59`；你提供的 E3 作業頁截圖顯示 **2026-10-29 23:59**，本專案文件依截圖標記最新 E3 時間。請以 E3 實際提交頁為準。壓縮檔內需有四個檔案：`112611054_hw4.pdf`、實測 `result.png`、`fusion.py` 和 `EKF.py`；PDF 要回答兩題 Discussion、放實測圖、本人終端身分截圖與 RViz2 YouTube 連結。錄影、E3 上傳由本人完成。

在 Ubuntu 主機的 `hw4` 目錄執行 `bash scripts/dev.sh report` 產生報告；設定檔、證據和四檔 ZIP 流程見[交付手冊](doc/SUBMISSION.md)。目前沒有 bag 實測結果或錄影，因此此工作區還沒有可提交的完整 PDF／ZIP。

## 來源與驗證紀錄

題目：[SDC Assignment 4](https://hackmd.io/muS7dwNbQjemnah98PLNRA)，本輪於 2026-10-29 重新讀取。程式基底：[老師的 sdc_hw4.zip](https://drive.google.com/file/d/1HUce6pQXLxd4QsVbJcQ9A70JSxe-JyQG/view)，原始 ZIP SHA256 為 `e01c33c59557081cdc668ad92eff33121cc8b7342955a29a62f0acb6d27507e0`。

離線檢查與合成輸出保存於分支的 `midterm01_gpu/outputs/run/`，不隨 Git 同步；其合成圖表不供作業繳交。原有 10 項數學／時序檢查保留，新測試核對 GT 隔離、整數 epoch 時間對齊、來源 SHA、拒絕覆寫與 ZIP 四檔契約。

本輪最終 17 項離線測試通過（10 項數學／時序、7 項 ROS stub／輸出／交付機制），unittest 計時 3.106 秒。紀錄：[unittest.log](../midterm01_gpu/outputs/run/track00_test_2610291901/unittest.log)；合成圖表與打包 fixture：[analysis](../midterm01_gpu/outputs/run/track00_test_2610291901/analysis/)。Python 語法、package XML、Compose YAML、兩個掛載及 shell 語法檢查通過。未執行 Ubuntu colcon、ROS node 或真實 bag。

本機提供 `output/112611054_hw4_source_update.zip` 供同步程式：它包含 `hw4/` 的程式／文件和 `hw2/docker-compose.yml`，沒有 bag 或實測結果。若 Ubuntu checkout 仍在 `/home/ricky/sdc_ros2`，先確認工作樹已備份且沒有 HW2 程序，再將程式更新包解壓到該 checkout 根目錄。這是來源更新包，作業交付仍依上面的四檔規則產生。

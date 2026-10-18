# HW4：Application of EKF

題目來源：[SDC Assignment 4](https://hackmd.io/muS7dwNbQjemnah98PLNRA)（2026-10-18 讀取）。
使用現有 **HW2 ROS 2 Humble Docker**，完成老師的 `fusion.py`、`EKF.py`。
`/radar_odometry` 提供運動增量，`/gps` 提供位置校正，`/gt_odom` **僅存檔／畫圖，不進入濾波**。

程式已放好，以下建置、播放、錄影指令由使用者在 **Ubuntu 主機** 手動執行。
目前已通過 10 項純 NumPy 合成數學／時序測試；尚未用使用者的 Ubuntu ROS 或真實 bag 驗證，不能據此宣稱定位精度。

## 1. 程式與資料位置

先在 Ubuntu 確認目前 checkout；既有專案位置為 `/home/ricky/sdc_ros2`：

```bash
cd /home/ricky/sdc_ros2
pwd
realpath hw4
cd hw4
mkdir -p colcon_ws/data
```

若 Ubuntu 是另一份 checkout，先把本次 `hw4/` 程式及 `hw2/docker-compose.yml` 修改同步過去；資料不隨 Git 同步。
這裡的 `/hw4` 指專案內的 `sdc_ros2/hw4/`，不是檔案系統根目錄 `/hw4`。

只需手動下載 [老師的 rosbag ZIP](https://drive.google.com/file/d/1Ei_dFm24jZSidt-asCZctTm_572b9kkw/view)，在 Ubuntu 解壓到 `hw4/colcon_ws/data/`，**保留 bag 整個目錄及原始檔名**，例如：

```text
/home/ricky/sdc_ros2/hw4/
├── scripts/dev.sh
└── colcon_ws/
    ├── src/fusion/
    │   ├── scripts/fusion.py
    │   ├── scripts/EKF.py
    │   ├── launch/EKF_fusion.launch.py
    │   ├── rviz_config/rviz_config.rviz
    │   ├── CMakeLists.txt
    │   └── package.xml
    ├── data/
    │   └── sdc_hw5_ros2/        # 以實際解壓名稱為準，也可能叫 sdc_hw4_ros2
    │       ├── metadata.yaml
    │       └── sdc_hw5_ros2.db3 # 多個分片時全部保留
    └── result/
```

例如下載檔位於 Ubuntu Downloads：

```bash
unzip ~/Downloads/sdc_hw4_rosbag.zip -d colcon_ws/data
find colcon_ws/data -name metadata.yaml -print
```

講義前段稱 `sdc_hw4_ros2`，目錄樹／播放段稱 `sdc_hw5_ros2`；工具依 `metadata.yaml` 找 bag，**不用改名**。
若 ZIP 多包一層目錄也能找到。只有一個 bag 時自動選擇；多個 bag 時請傳入含 `metadata.yaml` 的 Ubuntu 主機目錄，例如 `bash scripts/dev.sh play "$PWD/colcon_ws/data/sdc_hw5_ros2"`。
原始 data、build、install、log、結果都排除 Git；不必重新下載程式模板，也不必搬動 Midterm 資料。

## 2. 套用 HW2 容器掛載（首次）

已依本 repo 的 `hw2/docker-compose.yml` 加上：

```yaml
- ../hw4/colcon_ws:/ros_ws/hw4
```

沿用原 `ros2_dev` service、`ros2_container` 容器、`sdc_hw2_ros2` image、GUI 掛載與 ROS domain 0，保留所有 HW2 volumes。
以下 `recreate` 會重建 **HW2 這一個容器的執行個體**來套用掛載，正在其中執行的 HW2 程序會中斷；請在沒有 HW2 工作執行時做。它不重建 image、不操作 Midterm project。

```bash
cd /home/ricky/sdc_ros2/hw4
bash scripts/dev.sh recreate
bash scripts/dev.sh deps
bash scripts/dev.sh build
bash scripts/dev.sh test
```

`deps` 在現有容器安裝 NumPy、Matplotlib、colcon；沒有另外安裝 ROS／WSL／Docker。
重新 recreate 容器後，如原 image 沒有 Matplotlib，需再執行 `deps`。
若 Docker 要求 sudo，用 `sudo bash scripts/dev.sh ...` 執行對應指令即可。
若出現找不到 `sdc_hw2_ros2` image，先確認是否沿用正確的 HW2 環境；`recreate` 刻意不自動拉取或重建它。

## 3. 先確認 bag

```bash
bash scripts/dev.sh info
bash scripts/dev.sh inspect
```

`info` 顯示 topic／數量；`inspect` 只讀三個感測器 topic，列出實際訊息型別、frame、頻率、第一個 6×6 covariance、對角線範圍、全零 covariance 筆數及時間倒序情況。
預期 `/gps` 為 `geometry_msgs/msg/PoseWithCovarianceStamped`，另兩者為 `nav_msgs/msg/Odometry`；bag 內應有 `/clock`。
模板的共同 frame 是 `origin`。如果實際 frame 不同或空白，請先把 inspect 結果貼回來確認座標契約；程式不會把不同 frame 當成相同座標直接融合。
`inspect` 預設讀 sqlite3 bag，若資料格式與題目不同應先確認。

## 4. 執行：兩個 Ubuntu 終端機

**終端機 1：啟動 EKF 與 RViz2。**

```bash
cd /home/ricky/sdc_ros2/hw4
bash scripts/dev.sh launch
```

看到 `EKF ready` 後，**終端機 2：播放 bag。**

```bash
cd /home/ricky/sdc_ros2/hw4
bash scripts/dev.sh play
```

播放命令使用 `--delay 5`；bag 已含 `/clock`，**不加 `--clock`**。請每輪只播放一次，重播前重啟 fusion，避免時間倒退混入同一輪。
在播放期間錄製 RViz2 畫面，並截圖包含本人學號／身分的終端機。學號請使用自己的實際資料，不以容器 root 名稱替代。
RViz2 可見 `/radar_odometry`（綠）、`/gt_odom`（紅）、`/pred`（藍）與 bag 中的 LiDAR。
視角沿用老師模板追蹤 `car`；若 bag 沒有該 TF，可在 Views 將 Target Frame 改成 `origin` 再用滑鼠調整視角。

播放完成後回到 **終端機 1 按 Ctrl+C**，程式會印出主機結果位置並儲存：

```text
hw4/colcon_ws/result/run_<台北時間>_<唯一尾碼>/
├── result.png
├── estimate.csv   # stamp_ns,x,y,yaw；yaw 弧度
├── radar.csv
├── gps.csv
├── gt.csv
└── summary.json
```

每次執行各自保存，不覆寫上一輪。`summary.json` 中應有非零 `gps_updates` 及 estimate/gt 筆數；檢查 `invalid_messages`、`late_gps`、`nonincreasing_radar`、fallback 計數。
晚到 GPS／第一個 radar 前的 GPS／最後 radar 後未能對齊的 GPS 分開計數；大量丟棄或沒有更新時先查時間與 frame，不把一張圖存在視為成功。
目前只畫軌跡，未計算 time-aligned RMSE；不同頻率資料不能直接逐列相減算精度。

不開 GUI 的診斷模式：`bash scripts/dev.sh launch use_rviz:=false`。
若 RViz 顯示授權失敗，在 Ubuntu 桌面終端執行 `xhost +si:localuser:root` 後再啟動；結束可用 `xhost -si:localuser:root` 撤回。需使用桌面的 DISPLAY，SSH 無桌面時先使用不開 GUI 的模式。

## 5. 修改與學習

- 修改 Python 後重新 `bash scripts/dev.sh build` 再 launch。
- 進容器：`bash scripts/dev.sh shell`，workspace 為 `/ros_ws/hw4`。
- 手動原生命令也可用：source `/opt/ros/humble/setup.bash`，在 `/ros_ws/hw4` 執行 `colcon build --packages-select fusion`，source `install/setup.bash`，再執行 `ros2 launch fusion EKF_fusion.launch.py`。
- 直接呼叫原生 launch 時，預設結果位置為 `colcon_ws/result/result.png`，會覆寫同名檔；建議使用上方 wrapper 保留各輪結果。
- EKF 方程、covariance 意義與目前近似見 [算法說明](doc/EKF_DESIGN.md)。實際數值須依 inspect 讀到的 bag 回答，不能把 fallback 當成老師提供的 covariance。

## 6. 作業交付

依連結公告，截止 **10/28 23:59**；最新日期以老師公告為準。使用者單人完成，不預填學號或編造實測結果。
壓縮檔 `<student_id>_hw4.zip` 內共四個檔案：`<student_id>_hw4.pdf`、實測 `result.png`、`fusion.py`、`EKF.py`。
PDF 放本人終端機識別截圖、result.png、Discussion 兩題答案，以及 RViz2 錄影的 YouTube 連結；影片本身不用放 ZIP。
YouTube 上傳及作業繳交由使用者操作。

程式基底為題目提供的 [sdc_hw4.zip](https://drive.google.com/file/d/1HUce6pQXLxd4QsVbJcQ9A70JSxe-JyQG/view)。
下載原始 ZIP 的 SHA256：`e01c33c59557081cdc668ad92eff33121cc8b7342955a29a62f0acb6d27507e0`。

# HW4｜實測與交付

本手冊說明如何在 Ubuntu 的 HW2 ROS 2 Humble Docker 執行 HW4，並整理 HackMD 要求的四個交付檔。你提供的 E3 作業頁截圖標示截止時間為 **2026-10-29 23:59**；HackMD 文字仍顯示 10/28，實際繳交時間以 E3 頁為準。

## 一次實測

先依 [HW4 README](../README.md) 同步 `hw4/` 程式及 `hw2/docker-compose.yml` 到 Ubuntu，手動下載並解壓老師的 bag ZIP 到 `hw4/colcon_ws/data/`。不要改 bag 資料夾或分片名稱：文件有 `sdc_hw4_ros2`、`sdc_hw5_ros2` 兩種叫法，依 `metadata.yaml` 確認實際路徑。所有下列命令從 Ubuntu 主機執行：

```bash
cd /home/ricky/sdc_ros2/hw4
bash scripts/dev.sh recreate
bash scripts/dev.sh deps
bash scripts/dev.sh build
bash scripts/dev.sh test
bash scripts/dev.sh info
bash scripts/dev.sh inspect
```

`recreate` 會重建現有 HW2 `ros2_dev` 容器執行個體以套用 HW4 掛載，包括保留既有 `/ros_ws/hw4` colcon workspace，以及供報告程式使用的 `/ros_ws/hw4_tools` 專案根目錄；執行前確認容器內沒有別的 HW2 工作。它不建立新 ROS/Docker 環境，也不操作 Midterm container。若 Docker 需要 sudo，為上述命令加 `sudo` 即可。

先看 `inspect` 的 topic type、frame、covariance、時間範圍。不同 frame 或缺少 `/clock` 時先停止並查明 bag 版本，不直接融合。開兩個 Ubuntu 主機終端機，兩邊都進入 `hw4`：

```bash
# 終端機 1：啟動 EKF 與 RViz2，記下列印的唯一結果路徑
bash scripts/dev.sh launch

# 終端機 2：等終端機 1 出現 EKF ready 再播放一次
bash scripts/dev.sh play
```

若 bag 放在其他子目錄且不只一個，對 `play` 明確傳入主機路徑，例如：

```bash
bash scripts/dev.sh play "$PWD/colcon_ws/data/sdc_hw4_ros2"
```

bag 已含 `/clock`，使用 wrapper 的 `--delay 5` 播放，不要加 `--clock`。播放結束後回終端機 1 按 `Ctrl+C`；fusion node 會在 Ubuntu 主機的 `colcon_ws/result/run_.../` 保存 `result.png`、estimate/radar/GPS/GT CSV、`summary.json`、來源快照與 covariance／時間對齊分析。每輪都會用新的 run 目錄。

### 應保存的實測證據

播放時錄製 RViz2，預設顯示 `/radar_odometry`、`/pred`、`/gt_odom` 與 LiDAR；GPS 在 `result.png` 另外疊圖。截取能辨識本人與學號 `112611054` 的執行中終端機畫面，並將影片上傳 YouTube。可在啟動前執行 `printf 'SDC HW4 | Student ID: 112611054\n'`，並在截圖中保留學號與實際程式輸出；不能拿 Docker `root` 代替身分。若 RViz 畫面需要授權，在 Ubuntu 桌面終端使用 `xhost +si:localuser:root`，實驗完用 `xhost -si:localuser:root` 撤回。沒有桌面時可用 `bash scripts/dev.sh launch use_rviz:=false` 做功能診斷，但無法取得要求的 RViz 影片。

停止 fusion 後檢查 `summary.json`：estimate、radar、GPS 和 GT 樣本數，`gps_updates`、fallback covariance 數、`late_gps`、`nonincreasing_radar`、`invalid_messages`。確認 `covariance_statistics.json` 與 inspect 結果一致，確認 analysis 確實在共同時間區間比較 estimate 和 GT。不要用不同頻率資料逐列計 RMSE，也不要把圖檔已存在當成精度證明。

## 設定報告並建立交付物

修改 `hw4/submission_settings.py`，填寫：

- `STUDENT_ID`：已設為 `112611054`；確認無誤後保留。
- `RUN_DIR`：剛才實測的 `hw4/colcon_ws/result/run_...` 路徑。優先用相對於 `hw4` 的路徑，例如 `colcon_ws/result/run_261029_123456_abcd`。
- `TERMINAL_SCREENSHOT`：能辨識本人及學號的 Ubuntu 終端機截圖路徑。優先用相對路徑，例如 `output/evidence/terminal.png`，並將截圖存進 HW4 的忽略輸出目錄，確保同一容器可讀。
- `VIDEO_URL`：已上傳的 RViz2 YouTube URL。

在 Ubuntu 主機進入 `hw4`，使用固定入口；它會在 HW2 同一個 ROS 2 Humble 容器讀取 HW4 根目錄的設定、截圖與本輪 run：

```bash
bash scripts/dev.sh report
```

`deps` 會在 Ubuntu 主機的 HW2 ROS 2 Humble Docker 容器中安裝 NumPy、Matplotlib、ReportLab（PDF）、Arphic 中文字型和 colcon extensions。不另裝一套 ROS；若尚未安裝，在 Ubuntu 主機執行：

```bash
bash scripts/dev.sh deps
```

若中文字型不可用，PDF 會使用 CID 中文字型 fallback。結果或身分證據尚未填妥時，入口只輸出 `hw4/output/pdf/112611054_hw4_DRAFT.pdf`，**不建立 ZIP**；草稿不可繳交。RUN_DIR 需包含同一次執行的 `result.png`、`summary.json`、`analysis.json`、`covariance_statistics.json`、`source_manifest.json`、`source/fusion.py` 和 `source/EKF.py`。產生器會依來源 manifest 比對這兩個 run 快照的 SHA256，報告結果和 covariance 從該 run 讀取。

另外，實測 run 根目錄會保留 `commit.txt`、`git_status.txt`、`source_changes.patch`；診斷輸出有 `02_position_error.png` 和三份 `*_position_errors.csv`。這些提供追溯和判讀，**不額外塞進繳交 ZIP**。確保截圖、影片連結與 RUN_DIR 都指向同一輪完成的 Ubuntu 實測，再執行報告命令。

證據齊全後，入口產生 `hw4/output/pdf/112611054_hw4.pdf` 和 `hw4/output/112611054_hw4.zip`。ZIP 內恰有四個檔案：

```text
112611054_hw4.pdf
result.png
fusion.py
EKF.py
```

PDF 包含 Discussion 答案、該輪實測結果圖、實際 bag covariance 數字與 fallback 說明、終端機身分截圖及 YouTube 影片連結。`scripts/prepare_submission.py` 也可直接在已安裝 ReportLab 的 Python 環境執行；一般 Ubuntu 操作使用上面的 `bash scripts/dev.sh report` 入口即可。檢查 PDF 和 ZIP 清單，再自行到 E3 上傳；助理不代傳影片或作業。不要手動把 bag、Docker 資料、其他 run、暫存圖片或本地套件塞進 ZIP。

## 限制

目前沒有 bag 實測結果、錄影或身分截圖，因此不能提供實測 covariance、精度數字、最終 PDF 或 ZIP。程式中的 GPS/radar fallback 是僅在所選 covariance 全零時使用的假設值。合成測試通過不能代替真實 bag 結果。

本機離線合成檢查涵蓋原有 10 項 EKF 數學／時序與新增 ROS stub 輸出整合、來源核對和 ZIP 打包；這不代表執行了 ROS 或讀取真實 bag。最終數量與日誌見 [README 驗證紀錄](../README.md#來源與驗證紀錄)。原 EKF 核心算法在本輪沒有改動。

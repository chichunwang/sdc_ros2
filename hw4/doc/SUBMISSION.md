# HW4｜實測與交付

本手冊說明如何在 Ubuntu 的 HW2 ROS 2 Humble Docker 執行 HW4，並整理 HackMD 要求的四個交付檔。你提供的 E3 作業頁截圖標示截止時間為 **2026-10-29 23:59**；HackMD 文字仍顯示 10/28，實際繳交時間以 E3 頁為準。

## 一次實測

以下是已完成的 Ubuntu/HW2 ROS 2 Humble 實測紀錄；當時從目前 worktree 執行。bag 解壓在 `hw4/colcon_ws/data/sdc_hw5_ros2/`，保留官方名稱。ZIP SHA256 為 `5b20e6c453812d848f1e71ce4282f0079c41a1c993cd947b498dc7264acf8488`（1,223,510,833 bytes），DB3 為 1,790,959,616 bytes。HW2 的 `ros2_container` 已按目前掛載重建；舊 image `104f7d...` 恢復 `sdc_hw2_ros2` 標籤，deps 成功，colcon build 成功（1.06 s），ROS Humble 容器內 17 項測試通過（1.360 s）。以下列出的原命令僅作操作記錄，不需為了本文件重跑。

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

該輪檢查確認radar／GPS／GT 的 header.frame_id 均為 `origin`，bag 有 `/clock`。完整 80 秒 run 保存在 `hw4/colcon_ws/result/run_261029_231926_g9rH/`。統計為 radar 973、GPS 64、GPS update 64、late GPS 0、invalid 0、non-increasing radar 0、estimate 1037、GT 實收 7544；bag 原有 7714 筆 GT，best effort 接收少 170 筆。GPS fallback 0、radar fallback 1。Ubuntu 截圖在 `hw4/output/evidence/ubuntu_2610292317/`。

#### 錄影輪與審閱用 PDF

錄影輪使用獨立 run `hw4/colcon_ws/result/run_261029_232220_oYEV/`，未覆蓋前述驗證 run。bag 計數為 radar 973、GPS 64、GT 7714；run 收到 radar 961、GPS 63、GT 6648，故分別少 12、1、1066 筆。GPS update 63、late GPS 0、non-increasing radar 0、invalid 0、pending GPS 0。首末雷達時間與 bag 時間跨度 79.615616084 s 相符；fusion 收到 Ctrl+C 後 clean exit 0，player shell exit 0，但 log 無明確自然播畢行，不能宣稱所有訊息均收到或 player 自然播畢。來源 manifest SHA 相符。

GPS covariance 對角為 `(3, 3) m²`；GPS fallback 0 次、radar fallback 1 次。RMSE 為 radar 34.346932618594955 m、GPS 0.4634891799869087 m、estimate 1.035346041665754 m，estimate P95 為 1.665499907427702 m。GT timestamp 去重後 5340 筆，最大線性插值跨距 0.14985 s。這些來源的實際時間戳樣本不完全相同，不能作嚴格同樣本比較或宣稱 EKF 比 GPS 更準；GT 僅作結果評估。

RViz 影片為 <https://youtu.be/nmfKO3pA5y8>；同輪終端身分截圖為 `06_terminal_recording_identity.png`。使用者檢視 5 頁 PDF 後，指定刪除第 1、5 頁的兩段來源說明並授權打包。刪文後已重新渲染檢查全部頁面，建立 `output/112611054_hw4.zip`；四檔清單、CRC、PDF 及來源 bytes 均已核對。E3 尚未上傳。

該次操作以兩個 Ubuntu 主機終端機啟動與播放 bag：

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

停止 fusion 後已保存 `summary.json`、`covariance_statistics.json` 及 analysis。GPS 原 covariance 對角為 `(3, 3) m²`；radar 實際選用 covariance 對角統計 min 全為 0、max 為 `[9.380944161956033e-05, 6.072459584673981e-05, 0.0006405714870086814]`。RMSE 為 radar 34.30278999808334 m、GPS 0.46098894599053886 m、estimate 1.0322084968142409 m；各資料的時間戳樣本並不完全相同，不能視為嚴格同樣本比較或宣稱 EKF 比 GPS 更準。GT 原始時間有 1526 筆非遞增（包含重複）；評估依整數 timestamp 去重保留最後值，線性插值且不外推。GT 僅作評估，未參與 EKF。

## 設定報告並建立交付物

修改 `hw4/submission_settings.py`，填寫：

- `STUDENT_ID`：已設為 `112611054`；確認無誤後保留。
- `RUN_DIR`：剛才實測的 `hw4/colcon_ws/result/run_...` 路徑。優先用相對於 `hw4` 的路徑，例如 `colcon_ws/result/run_261029_123456_abcd`。
- `TERMINAL_SCREENSHOT`：能辨識本人及學號的 Ubuntu 終端機截圖路徑。優先用相對路徑，例如 `output/evidence/terminal.png`，並將截圖存進 HW4 的忽略輸出目錄，確保同一容器可讀。
- `VIDEO_URL`：已上傳的 RViz2 YouTube URL。

RViz 錄影與 YouTube URL 已完成，`submission_settings.py` 已填入同輪 RUN_DIR、`06_terminal_recording_identity.png` 與影片 URL。PDF 已經使用者檢視，依指定刪文後完成正式 ZIP。工具現支援 `python3 scripts/prepare_submission.py --pdf-only`（在容器的 `/ros_ws/hw4_tools` 內執行）只產 PDF；本輪 PDF 與 ZIP 已完成，此命令僅供後續需要重新產生審閱稿時參考。本次打包直接使用刪文後完成視覺檢查的 PDF bytes，未在打包時再次生成報告；E3 尚未上傳。

```bash
python3 scripts/prepare_submission.py --pdf-only
```

上面是審閱稿的 PDF-only 產生器入口（在容器的 `/ros_ws/hw4_tools` 內執行），本輪 PDF 已由主 agent 產生，不要為了文件更新而重跑。此處不提供自動 ZIP 的 `report` 命令，避免使用者審閱前意外打包。

`deps` 會在 Ubuntu 主機的 HW2 ROS 2 Humble Docker 容器中安裝 NumPy、Matplotlib、ReportLab（PDF）、Arphic 中文字型、Liberation 英文字型和 colcon extensions。不另裝一套 ROS；若尚未安裝，在 Ubuntu 主機執行：

```bash
bash scripts/dev.sh deps
```

PDF 依使用者同意的近似字型採中文 AR PL UKai TW 楷體與英文 Liberation Serif（Times 風格），只保留右下頁碼；缺字型時會明確提示安裝。結果或身分證據尚未填妥時，入口只輸出 `hw4/output/pdf/112611054_hw4_DRAFT.pdf`，**不建立 ZIP**；草稿不可繳交。RUN_DIR 需包含同一次執行的 `result.png`、`summary.json`、`analysis.json`、`covariance_statistics.json`、`source_manifest.json`、`source/fusion.py` 和 `source/EKF.py`。產生器會依來源 manifest 比對這兩個 run 快照的 SHA256，報告結果和 covariance 從該 run 讀取。

另外，實測 run 根目錄會保留 `commit.txt`、`git_status.txt`、`source_changes.patch`；診斷輸出有 `02_position_error.png` 和三份 `*_position_errors.csv`。這些提供追溯和判讀，**不額外塞進繳交 ZIP**。確保截圖、影片連結與 RUN_DIR 都指向同一輪完成的 Ubuntu 實測，再執行報告命令。

使用者核可 PDF 後，才建立 `hw4/output/112611054_hw4.zip`，並確認 PDF bytes 與審閱版本相同。ZIP 內恰有四個檔案：

```text
112611054_hw4.pdf
result.png
fusion.py
EKF.py
```

PDF 包含 Discussion 答案、該輪實測結果圖、實際 bag covariance 數字與 fallback 說明、終端機身分截圖及 YouTube 影片連結。使用者核可後才以已審閱 PDF 原 bytes 建立 ZIP，並檢查 PDF bytes 與四檔 ZIP 清單；再由使用者自行到 E3 上傳。不要重新產生 PDF，也不要把 bag、Docker 資料、其他 run、暫存圖片或本地套件塞進 ZIP。

## 目前交付狀態與限制

真實 bag、Ubuntu build／ROS 測試、兩次 run、終端身分截圖、RViz 錄影與 YouTube URL 均已完成；5 頁 PDF 已依使用者紅線刪文，正式四檔 ZIP 已依授權完成。E3 尚未上傳。程式中的 GPS/radar fallback 僅在所選 covariance 全零時使用；錄影輪 GPS fallback 0 次、radar fallback 1 次。該輪 bag 訊息未全部收到，且播放器 log 沒有明確自然播畢訊息；RMSE 依不同來源可用時間戳計算，不能宣稱同樣本比較，也不能宣稱 EKF 優於 GPS。

較早階段的本機離線合成檢查涵蓋原有 10 項 EKF 數學／時序與新增 ROS stub 輸出整合、來源核對和 ZIP 打包；它不是 ROS 或真實 bag 驗證。該階段日誌見 [README 驗證紀錄](../README.md#來源與驗證紀錄)。原 EKF 核心算法在本輪沒有改動。

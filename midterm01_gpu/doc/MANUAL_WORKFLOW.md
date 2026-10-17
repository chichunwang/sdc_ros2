# Midterm01 GPU 日常操作與結果讀取

更新：2026-10-17。以下 Docker／benchmark／檔案指令都在 **Ubuntu 主機** 執行。後續使用 `midterm01_gpu`，CPU 程式／結果保留；完整 Track 測試由使用者手動執行，不因整理文件而重跑。

開發以使用者理解與應用知識為目的：先觀察、提出假說、討論適合的模型／算法，再實作驗證；非必要不以特定 Track 調參刷分為主。必要調參需有可說明的依據，詳見 [學習與算法改進原則](DEVELOPMENT_PRINCIPLES.md)。

感測器融合 `heading_rts`／`motion_rts` 已加入 benchmark，省略選項仍為舊 `fixed`。先前 `heading_rts + geometry off` 已評分 1.16951／0.31967，歷史版本見 [融合進步紀錄](HEADING_RTS_RESULTS_20261017.md)；`motion_rts` 仍只有短測。

目前最佳為 **heading_rts＋directional，1.15275／0.30576**，使用者已完成兩條完整測試／提交，見 [成績與最佳 CSV](GEOMETRY_RESULTS_20261017.md)。此版本有修改 C++，本機已完成 Release 建置；明確加 `--geometry-mode directional` 才啟用，預設仍為 off。完整手動命令見 [run_geometry_directional.txt](run_geometry_directional.txt)，設計見 [幾何說明](GEOMETRY_DIRECTIONAL_20261017.md)。每輪上傳檔仍是 `diagnostic.csv`，另產生的 `geometry.csv` 是診斷資料。

## 工作位置與編輯檔案

VS Code 開啟 `/home/ricky/sdc_ros2`。主機修改透過 bind mount 直接出現在 GPU 容器，C++／CUDA 修改後需重建，資料與舊 CPU 程式不用複製。

| 工作 | 主機位置（相對專案根目錄） | GPU 容器位置 |
| --- | --- | --- |
| 定位流程／profile／ROS 參數 | `midterm01_gpu/workspace/src/localization/src/localization.cpp` | `/root/midterm/src/localization/src/localization.cpp` |
| robust 四自由度細修 | `midterm01_gpu/workspace/src/localization/src/sensor_guided.cpp` | `/root/midterm/src/localization/src/sensor_guided.cpp` |
| CUDA 最近鄰 | `midterm01_gpu/workspace/src/cuda_ndt_backend/src/nearest_neighbor.cu` | `/root/midterm/src/cuda_ndt_backend/src/nearest_neighbor.cu` |
| 標頭／launch／編譯依賴 | `midterm01_gpu/workspace/src/` 下對應 package | `/root/midterm/src/` |
| GNSS／IMU 前處理 | `midterm01_gpu/scripts/prepare_sensor_prior.py` | `/root/scripts/prepare_sensor_prior.py` |
| benchmark 實作／主機入口 | `midterm01_gpu/scripts/accuracy/benchmark.py`／`scripts/benchmark.py` | benchmark 從主機執行 |
| 原始資料 | `midterm01/data/` | `/root/data/`（唯讀） |
| GPU 輸出 | `midterm01_gpu/outputs/` | `/root/outputs/` |

新版是 CUDA 最近鄰＋CPU robust 最佳化，並非全部計算都在 GPU。架構與原有／新增貢獻見 [演算法教學](ALGORITHM_WALKTHROUGH_20261017.md)。

## 啟動、建置與指定幀數

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
bash scripts/dev.sh up
bash scripts/dev.sh cuda-check

# 修改 C++／CUDA 後執行；只換 Track／幀數不用重編。
bash scripts/dev.sh build-workspace --packages-select cuda_ndt_backend localization

# 每條命令完成後再執行下一條。
python3 scripts/benchmark.py --track 1 --frames 50 --profile sensor_guided --sensor-fusion heading_rts --geometry-mode directional
python3 scripts/benchmark.py --track 2 --frames 50 --profile sensor_guided --sensor-fusion heading_rts --geometry-mode directional
```

完整測試由使用者手動執行：

```bash
python3 scripts/benchmark.py --track 1 --frames 3011 --profile sensor_guided --sensor-fusion heading_rts --geometry-mode directional
python3 scripts/benchmark.py --track 2 --frames 3299 --profile sensor_guided --sensor-fusion heading_rts --geometry-mode directional
```

`--frames 1` 是單幀，`--frames 100` 是前 100 幀；benchmark 預設 50，只接受正整數。原 ROS 節點的 `max_frames=0` 與 benchmark 介面不同，不要在這裡用 `--frames 0`。省略 profile 會執行舊 `legacy`／`cuda_ndt`；新版務必帶 `--profile sensor_guided`，自動選 `cuda_robust_icp`。

benchmark 從唯一 run 的 `frames.csv` 產生感測器先驗，source／map 尺度、濾波與輸出參數另存 `params.json`。只改幀數不必重建映像；修改系統依賴才需 `dev.sh build`。GPU 容器管理使用本目錄 dev.sh，service 是 `midterm01_gpu`、預設容器名是 `sdc_midterm01_gpu`。

基本資料檢查若要保存報告：

```bash
check_run="$(mktemp -d outputs/runs/data_check_XXXXXXXX)"
bash scripts/dev.sh check-data --strict-count \
  --output "/root/outputs/runs/$(basename "$check_run")/data_check.json"
```

這只檢查必要檔、非空與第一層 PLY 數量；不證明時間、frame、幾何或精度。原始資料保持唯讀。

## 在 VS Code 找結果

終端印出的 `Run: /home/ricky/sdc_ros2/midterm01_gpu/outputs/runs/...` 是主機路徑，可直接在 VS Code 開啟。

benchmark 新 run 保留 Track／profile／後端名稱，尾碼改用建立 run 當下的台北時間（`Asia/Taipei`），格式為 `YYMMDDHHmm`，不再附幀數或隨機字串。例如 2026-10-17 22:50 開始的 directional 測試：

```text
track1_sensor_guided_heading_rts_geometry_directional_cuda_robust_icp_d2d_2610172250
```

同一分鐘若同名已存在，依序加 `_2`、`_3`，以免覆寫結果。幀數仍記在 `params.json.max_frames`、`validation.json.expected_frames`；既有 run 名稱保留。

容器路徑要依容器換成主機路徑：

| 容器 | 容器內位置 | 主機 VS Code 對應位置 |
| --- | --- | --- |
| GPU `sdc_midterm01_gpu` | `/root/outputs/runs/<run>` | `/home/ricky/sdc_ros2/midterm01_gpu/outputs/runs/<run>` |
| CPU `sdc_midterm01` | `/root/outputs/runs/<run>` | `/home/ricky/sdc_ros2/midterm01/outputs/runs/<run>` |

相同 `/root/outputs` 不代表同一個目錄；在 GPU 容器找 CPU run 會找不到。主機 `/root` 也不是任一容器的 `/root`。掛載不確定時在主機查看：

```bash
docker inspect sdc_midterm01 sdc_midterm01_gpu \
  --format '{{.Name}}{{range .Mounts}} {{.Source}} -> {{.Destination}};{{end}}'
```

新 run 先看這些檔案：

| 檔案 | 用途 |
| --- | --- |
| `validation.json` | 預期／實際列數、退出碼、有限值、錯誤、耗時與診斷 |
| `result.txt` | 每幀 3×4 矩陣的 12 個值，無 timestamp |
| `diagnostic.csv` | `idx,x,y,yaw`，idx 從 0 起、yaw 弧度 |
| `frames.csv` | idx、整數 timestamp、原 PLY 檔名 |
| `sensor_prior.csv/json` | 新版同步後感測器輸入、heading bias 與設定 |
| `guidance.csv` | 每幀地圖修正、接受狀態與正則化目標 |
| `params.json`、`command.sh` | 實際參數與重現命令 |
| `run.log`、`timings.csv`、`gpu_samples.csv` | 程序紀錄、分段耗時、GPU 取樣 |

提交前檢查 `returncode=0`、`output_complete_and_finite=true`、`validation_errors=[]`，且 Track1／Track2 分別 3011／3299 筆；不只看結果檔存在。新版 `failed_alignments` 代表地圖細修未接受，該幀仍輸出當下感測器先驗；不是 CUDA kernel 錯誤數。`cuda_converged_frames=0` 是新版未使用 D2D 候選的正常現象。

幾何支持不足現在顯示 `INFO Map update skipped ... insufficient_unique_targets` 或 `insufficient_reliable_geometry`，並印出候選可靠對應數與不同地圖點數；其他地圖拒絕仍為 WARN，真正 runtime 錯誤仍為 ERROR 並中止。`guidance.csv` 新增 `update_status,candidate_inliers,candidate_reliable_points,candidate_unique_targets`，`validation.json.guidance_not_accuracy` 彙整原因。`geometry.csv` 描述最終輸出的 pose；拒絕時這是先驗，不能反推被拒候選的精確原因。舊 run 沒有新欄位，保持原樣；第 410 幀實例見 [調查紀錄](GEOMETRY_FALLBACK_20261017.md)。

`pipeline_elapsed_seconds` 包含感測器前處理與 ROS 執行；`elapsed_seconds` 不含前處理。`timings.csv.ndt_ms` 新版記 CUDA 搜尋＋CPU 細修，不是 NDT。快取預設停用，要明確指定 GPU outputs 下的 `--map-cache` 並保留 SHA256／leaf sidecar；比較時間須固定快取與 Release 條件。

目前最新已評分的 heading_rts＋directional 結果為 **Track1 1.15275／Track2 0.30576**，見 [完整進步紀錄與 CSV](GEOMETRY_RESULTS_20261017.md)。來源是使用者手動提交與本輪截圖，兩份本機 CSV 已核對 SHA256 並另外保存。前一版仍在 [heading_rts 歷史紀錄](HEADING_RTS_RESULTS_20261017.md)；地圖殘差或收斂不能代替 Kaggle 成績。

## CPU 結果看不到與 chmod

使用者指定的 CPU run **有內容**。修復前資料夾是 root:root／`700`（`drwx------`），主機帳號 ricky 無法列出；檔案本身已是 `644`。2026-10-17 已限於該 run 執行以下修復，之後主機可讀，結果 SHA256 未改：

```bash
# 在 Ubuntu 主機執行；sdc_midterm01 是 CPU 容器。
docker exec --user root sdc_midterm01 chmod -R a+rX \
  /root/outputs/runs/first_test_9vBOV7le/track1_n0_xLOSykY4

ls -lah /home/ricky/sdc_ros2/midterm01/outputs/runs/first_test_9vBOV7le/track1_n0_xLOSykY4
wc -l /home/ricky/sdc_ros2/midterm01/outputs/runs/first_test_9vBOV7le/track1_n0_xLOSykY4/result.txt
```

`a+rX` 給使用者／群組／其他帳號讀取權限；大寫 `X` 只給目錄及原本可執行的檔案執行權限。目錄的 `r` 用來列出名稱、`x` 用來進入／存取內容；本次目錄從 700 變為 755，普通 TXT 仍是 644。這只提供讀取，檔案擁有者仍為 root，不會讓主機直接改寫原始結果。

若 CPU 容器沒有啟動，對應的主機方式如下；與上面二擇一即可，不必為讀檔啟動 CPU 定位：

```bash
sudo chmod -R a+rX \
  /home/ricky/sdc_ros2/midterm01/outputs/runs/first_test_9vBOV7le/track1_n0_xLOSykY4
```

不要把範圍擴到 `/root`、原始 data 或全專案，也不需要 `chmod -R 777`。換 run 時先查 `namei -l <主機完整路徑>`，確認是路徑錯誤、父目錄無法進入或該 run 權限；GPU run 要使用 GPU 容器／主機 GPU 路徑。主機以 ricky 執行 benchmark 會建立自己擁有的 run，通常不需同樣修復。

若 VS Code Explorer 還顯示舊狀態，按重新整理，確認開的是上面的 `midterm01/outputs/...` 主機路徑；使用 Dev Containers 時則需附加 CPU 容器才能看它的 `/root/outputs`。完成紀錄與檔案雜湊見 [CPU Track1 紀錄](../../midterm01/docs/CPU_TRACK1_COMPLETED_20261017.md)。

## 保留 CPU TXT，另存 CSV

指定完整 CPU run 有 `result.txt`、`run.log`、`command.sh`、`exit_code.txt`，**尚無 CSV**。若要比較，可在主機以 GPU 版標準函式庫轉檔工具，將衍生 CSV 存到新的 GPU run，保留 CPU 原檔：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
cpu_txt=/home/ricky/sdc_ros2/midterm01/outputs/runs/first_test_9vBOV7le/track1_n0_xLOSykY4/result.txt
csv_run="$(mktemp -d outputs/runs/cpu_track1_csv_XXXXXXXX)"
python3 workspace/src/localization/processing_codes/txt_to_idx_x_y_yaw.py \
  "$cpu_txt" "$csv_run/cpu_track1.csv"
wc -l "$csv_run/cpu_track1.csv"
```

應有 3012 行（表頭＋3011 筆）。這是 CPU 參考輸出，不是新版 sensor_guided 候選；轉檔沒有重新計算 pose，也不代表正式 frame／單位契約已確認。

## 視覺化與歷史重現

目前 GPU benchmark 不開 RViz；GPU dev.sh **沒有 `--gui`**，不要直接套用舊 CPU GUI 指令。現有 GPU Compose 未掛載 DISPLAY／X11，GUI 需另行配置並驗證。現有軌跡與疊圖可先看 [新版報告](../SENSOR_GUIDED_20261017.md) 連結的 PNG。舊 `visualization.cpp` 是 Radar／PCD 介面，不是本次 RGB PLY 回放器。

只有比較舊方法時才使用以下前 50 幀命令：

```bash
python3 scripts/benchmark.py --track 1 --frames 50 --profile legacy --backend pcl_ndt
python3 scripts/benchmark.py --track 1 --frames 50 --profile legacy --backend cuda_ndt --cuda-mode D2D
```

CPU／D2D 是不同配準實作，前一估計不是 ground truth。完整新版測試與提交由使用者執行；寄信與 E3 上傳也需使用者操作或另行明確授權。

# 08｜Midterm01 GPU 日常操作與結果讀取

更新：2026-10-18。以下 Docker／benchmark／檔案指令都在 **Ubuntu 主機** 執行。後續使用 `midterm01_gpu`，CPU 程式／結果保留；完整 Track 測試由使用者手動執行，不因整理文件而重跑。

第一次理解程式請先讀 [01｜架構圖與逐步流程](01_PROGRAM_FLOW.md)；歷史、日期與舊名對照見 [28｜文件目錄](28_DOCUMENTATION_INDEX.md)。

開發以使用者理解與應用知識為目的：先觀察、提出假說、討論適合的模型／算法，再實作驗證；非必要不以特定 Track 調參刷分為主。必要調參需有可說明的依據，詳見 [學習與算法改進原則](19_DEVELOPMENT_PRINCIPLES.md)。

感測器融合 `heading_rts`／`motion_rts` 已加入 benchmark，省略選項仍為舊 `fixed`。先前 `heading_rts + geometry off` 已評分 1.16951／0.31967，歷史版本見 [融合進步紀錄](14_HEADING_RTS_RESULTS_20261017.md)；`motion_rts` 仍只有短測。

目前最佳為 **heading_rts＋directional，1.15275／0.30576**，使用者已完成兩條完整測試／提交，見 [成績與最佳 CSV](17_GEOMETRY_RESULTS_20261017.md)。此版本有修改 C++，本機已完成 Release 建置；明確加 `--geometry-mode directional` 才啟用，預設仍為 off。完整手動命令見 [16_run_geometry_directional.txt](16_run_geometry_directional.txt)，設計見 [幾何說明](15_GEOMETRY_DIRECTIONAL_20261017.md)。每輪上傳檔仍是 `diagnostic.csv`，另產生的 `geometry.csv` 是診斷資料。

2026-10-18 新截圖：使用者 Team 3 為 Track1 第 4／Track2 第 2 名，最佳分數未變；非即時查榜。[現行算法與改進分析](20_CURRENT_ALGORITHM_AND_ACCURACY_20261018.md) 記錄排名差距、地圖弱方向與候選方法。最初的文件整理沒有更改算法；同日後續實作見下段。

同日後續依使用者要求已完成三相機尺度／同步與短段對照，見 [實作與確認方法](22_THREE_CAMERA_IMPLEMENTATION_20261018.md)。Python 新介面不需重編 C++；三相機仍為實驗來源，沒有新完整測試或分數，日常推薦保持前相機。

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

新版是 CUDA 最近鄰＋CPU robust 最佳化，並非全部計算都在 GPU。架構與原有／新增貢獻見 [演算法教學](07_ALGORITHM_WALKTHROUGH_20261017.md)。

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

## 三相機實驗、短段與相同先驗

新選項不改程式預設：

| 選項 | 意義 |
| --- | --- |
| `--cloud-source front` | 預設前相機 raw_ply；省略 scale 仍為 3 |
| `--cloud-source mapanything` | 現成三相機 merged PLY；只支援 sensor_guided／cuda_robust_icp，須明確提供 scale 與來源報告 |
| `--ply-scale` | 作用在點座標；三相機不默認沿用前相機尺度 |
| `--source-report` | 本輪 lab 的 report.json；核對 Track、index／report hash 與 frame 契約，記錄實驗性質，不代表已校準精度 |
| `--start-frame` | 原始 0-based 起點；預設 0，與 frames 合起來選連續短段；只支援 sensor_guided |
| `--prior-run` | 從同 Track／fusion 的既有 GPU run 依 timestamp 複製相同感測器先驗；不重用 map posterior，fusion CLI 參數此時不重新套用 |

本輪共享尺度候選為 **0.8712495367580543**。完整 6310 組同步核對確認 `f_ts` 為 bag 時間、`fl/fr_ts` 為 header 時間；比較時已轉成同 header 時基，但定位／idx 仍以原前相機 bag 時間為參考。merged XYZRGB 沒有每點相機身分，**尚未做真實逐相機時間補償**。

以下只重現一段已測過的三相機實驗，執行會另建唯一 run，不是完整提交檔：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
python3 scripts/benchmark.py --track 1 --frames 24 --start-frame 1493 \
  --profile sensor_guided --sensor-fusion heading_rts --geometry-mode directional \
  --cloud-source mapanything --ply-scale 0.8712495367580543 \
  --source-report outputs/runs/three_camera_dev_2610181055/calibration_track1_3/report.json \
  --prior-run outputs/runs/track1_sensor_guided_heading_rts_geometry_directional_cuda_robust_icp_d2d_3011f_70rv9cj4 \
  --map-cache outputs/runs/cuda_ndt_20261017_nlvdk4j5/cache_track1_leaf1.pcd
```

改成 `--cloud-source front --ply-scale 3` 並移除 `--source-report`，可用同一段、同一先驗作前相機對照。實驗條件、來源／尺度、原始 idx 及每個 PLY SHA256 見 `source_input.json`；`source_calibration.json` 保存當輪來源報告；`sensor_prior.json.reused_prior` 記錄先驗來源。

短段的 frames.csv／diagnostic.csv 仍由局部 idx=0 起，原始 idx 另存於 source_input.json；`validation.json.submission_frame_set_complete=false`。移機若沒有被 Git 忽略的報告／cache／prior，先恢復對應證據，不能任意用另一份來源報告替代。

本輪三相機雖增加局部幾何方向，共同三維 probe 的高度／形狀差異仍大；XY／yaw 是否更準尚無證據。先保留前相機最佳，不把這段命令當作已驗證較佳的完整流程。尺度估計、同步核對的容器命令與原始參數見 lab report 的 `command`、本輪 `joint_fit.py` 及 [實作紀錄](22_THREE_CAMERA_IMPLEMENTATION_20261018.md)。

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

幾何支持不足現在顯示 `INFO Map update skipped ... insufficient_unique_targets` 或 `insufficient_reliable_geometry`，並印出候選可靠對應數與不同地圖點數；其他地圖拒絕仍為 WARN，真正 runtime 錯誤仍為 ERROR 並中止。`guidance.csv` 新增 `update_status,candidate_inliers,candidate_reliable_points,candidate_unique_targets`，`validation.json.guidance_not_accuracy` 彙整原因。`geometry.csv` 描述最終輸出的 pose；拒絕時這是先驗，不能反推被拒候選的精確原因。舊 run 沒有新欄位，保持原樣；第 410 幀實例見 [調查紀錄](18_GEOMETRY_FALLBACK_20261017.md)。

`pipeline_elapsed_seconds` 包含感測器前處理與 ROS 執行；`elapsed_seconds` 不含前處理。`timings.csv.ndt_ms` 新版記 CUDA 搜尋＋CPU 細修，不是 NDT。快取預設停用，要明確指定 GPU outputs 下的 `--map-cache` 並保留 SHA256／leaf sidecar；比較時間須固定快取與 Release 條件。

目前最新已評分的 heading_rts＋directional 結果為 **Track1 1.15275／Track2 0.30576**，見 [完整進步紀錄與 CSV](17_GEOMETRY_RESULTS_20261017.md)。來源是使用者手動提交與本輪截圖，兩份本機 CSV 已核對 SHA256 並另外保存。前一版仍在 [heading_rts 歷史紀錄](14_HEADING_RTS_RESULTS_20261017.md)；地圖殘差或收斂不能代替 Kaggle 成績。

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

目前 GPU benchmark 不開 RViz；GPU dev.sh **沒有 `--gui`**，不要直接套用舊 CPU GUI 指令。現有 GPU Compose 未掛載 DISPLAY／X11，GUI 需另行配置並驗證。現有軌跡與疊圖可先看 [新版報告](06_SENSOR_GUIDED_20261017.md) 連結的 PNG。舊 `visualization.cpp` 是 Radar／PCD 介面，不是本次 RGB PLY 回放器。

只有比較舊方法時才使用以下前 50 幀命令：

```bash
python3 scripts/benchmark.py --track 1 --frames 50 --profile legacy --backend pcl_ndt
python3 scripts/benchmark.py --track 1 --frames 50 --profile legacy --backend cuda_ndt --cuda-mode D2D
```

CPU／D2D 是不同配準實作，前一估計不是 ground truth。完整新版測試與提交由使用者執行；寄信與 E3 上傳也需使用者操作或另行明確授權。


## 原始影像的校正三相機小規模流程

本機已在主機 `outputs/ml_runtime/venv` 安裝 MapAnything／PyTorch 2.8.0+cu128，模型在 `outputs/ml_runtime/checkpoint`，upstream 在 `outputs/ml_runtime/map-anything`。ROS 容器 Python 未變；重建在主機 ML venv，定位仍在 GPU 容器。方法、固定高度、尺度限制與實測見 [校正三相機報告](23_CALIBRATED_RIG_RESULTS_20261018.md)。這一階段只產生少量診斷 PLY／pose，不是完整提交檔。

下列從 GPU 目錄執行；以目前已核對的原始 prior／map cache 公平對照。不同 checkout 的 outputs 不會由 Git 同步，缺檔時先確認來源，不能拿其他 Track 的 cache／prior 代替。

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
RIG_RUN="$(python3 -c 'import sys; from pathlib import Path; sys.path.insert(0,"scripts/accuracy"); from benchmark import create_run_directory; print(create_run_directory(Path("outputs/runs"),"calibrated_rig_manual"))')"
RIG_P1=outputs/runs/track1_sensor_guided_heading_rts_geometry_directional_cuda_robust_icp_d2d_3011f_70rv9cj4
RIG_P2=outputs/runs/track2_sensor_guided_heading_rts_geometry_directional_cuda_robust_icp_d2d_3299f_7o_atg6_
RIG_M1=outputs/runs/cuda_ndt_20261017_nlvdk4j5/cache_track1_leaf1.pcd
RIG_M2=outputs/runs/three_camera_dev_2610181055/cache_track2_leaf1.pcd
RIG_PY=outputs/ml_runtime/venv/bin/python

# 預設 8 groups、24 張原始影像；最多接受 24 groups，不會啟動整條 Track。
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$RIG_PY" scripts/reconstruct_calibrated_rig.py \
  --data-root ../midterm01/data --checkpoint outputs/ml_runtime/checkpoint \
  --model-repo outputs/ml_runtime/map-anything \
  --prior-track1 "$RIG_P1" --prior-track2 "$RIG_P2" \
  --output "$RIG_RUN/reconstruction"

# 與 depth／map 無關的影像配對檢查；然後用 fit groups 的 stereo 錨點估共享尺度。
"$RIG_PY" scripts/validate_rig_reconstruction.py "$RIG_RUN/reconstruction"
"$RIG_PY" scripts/anchor_rig_scale.py \
  --reconstruction "$RIG_RUN/reconstruction" --output "$RIG_RUN/anchored"
"$RIG_PY" scripts/anchor_rig_scale.py \
  --reconstruction "$RIG_RUN/reconstruction" --output "$RIG_RUN/anchored_density600" --budget 600

# 各比較依序執行，沒有另一個 localization 時才啟動下一組。
RIG_COMPARE=(--prior-track1 "$RIG_P1" --prior-track2 "$RIG_P2" --map-cache-track1 "$RIG_M1" --map-cache-track2 "$RIG_M2")
python3 scripts/compare_calibrated_rig.py "${RIG_COMPARE[@]}" \
  --reconstruction "$RIG_RUN/reconstruction" --output "$RIG_RUN/comparison"
python3 scripts/compare_calibrated_rig.py "${RIG_COMPARE[@]}" \
  --reconstruction "$RIG_RUN/anchored" --output "$RIG_RUN/comparison_anchored" \
  --labels anchored_front,anchored_static,anchored,anchored_uniform_matched,anchored_balanced
python3 scripts/compare_calibrated_rig.py "${RIG_COMPARE[@]}" \
  --reconstruction "$RIG_RUN/anchored_density600" --output "$RIG_RUN/comparison_density600" \
  --labels anchored_front,anchored_uniform_matched,anchored_balanced --label-prefix small_

"$RIG_PY" scripts/analyze_calibrated_rig.py \
  --comparisons "$RIG_RUN/comparison/comparison_runs.json" "$RIG_RUN/comparison_anchored/comparison_runs.json" "$RIG_RUN/comparison_density600/comparison_runs.json" \
  --reconstruction "$RIG_RUN/reconstruction" --anchored "$RIG_RUN/anchored" \
  --data-root ../midterm01/data --map-cache-track1 "$RIG_M1" --map-cache-track2 "$RIG_M2" \
  --output "$RIG_RUN/comparison_summary.json"
```

單獨定位某個已重建的小片段，可用下列介面；`--start-frame` 是**原序列**索引，report 中必須實際含有所選的每一幀，不能把稀疏重建誤當完整序列：

```bash
python3 scripts/benchmark.py --track 1 --frames 2 --start-frame 1505 \
  --profile sensor_guided --sensor-fusion heading_rts --geometry-mode directional \
  --cloud-source reconstruction --ply-scale 1 \
  --source-report "$RIG_RUN/anchored/anchored_balanced_track1/report.json" \
  --prior-run "$RIG_P1" --map-cache "$RIG_M1"
```

確認尺度看 `scale_anchor.json` 的 fit／holdout depth ratios、視差角及錨點數，不能只看 fit 接近 1。確認同步看 per-view 的整數 header／bag 對應、`applied_camera_pose`、`image_geometry_validation.json` 的 static／motion 極線誤差；沒有硬體時鐘或曝光真值時，仍只能確認資料契約與幾何一致性。側面效果看 same-depth front-only／all-view、同點數 uniform／balanced 與消去 Z 後的 XY／yaw data information；map residual 不是正式定位精度。

若另建機器上的 ML 環境，先建獨立 venv，再依官方 PyTorch cu128 wheel 安裝 **torch 2.8.0、torchvision 0.23.0、torchaudio 2.8.0**，安裝 upstream 指定 commit 的 MapAnything 與 scipy／pyyaml；勿將 numpy 2 或這套 ML 依賴裝進 ROS Humble Python。依本次 `uv_freeze_final.txt` 與 checkpoint SHA256 重現；模型／venv／原始資料均不加入 Git。


## 三相機偏差驗證後的完整一輪實驗

2026-10-18 跨場景驗證**未通過**：去重後 stereo 共用深度尺度候選 0.67398047，足夠錨點的 holdout groups median／P90 差異仍為 18.2%／30.6%；獨立 temporal 尺度也不穩定。這不是定位誤差。推薦最佳仍為前相機；完整三相機僅作凍結候選的實際評分實驗。方法、來源、恢復限制見 [偏差驗證與完整入口](24_BIAS_VALIDATION_20261018.md)。

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
bash scripts/dev.sh up
bash scripts/run_three_camera_full.sh --dry-run

# 使用者手動啟動兩條完整 Track：raw 三相機重建 → 原 heading_rts／directional 定位。
bash scripts/run_three_camera_full.sh
```

終端印出唯一 `Run:`。完成後 `<Run>/result_1.csv`／`result_2.csv` 分別應有表頭加 3011／3299 筆；不是舊 merged PLY，也不會自動提交。中斷後續跑同一個 run：

```bash
bash scripts/run_three_camera_full.sh --resume outputs/runs/three_camera_full_YYMMDDHHmm
```

預設不存 dense depth／rectified JPG；保留逐幀選取點雲、camera labels/confidence、固定外參／運動 pose、時間與來源 hash。會核對模型／程式／套件／校正／原始影像與已完成檔案，拒絕混用不同設定。完整開頭／結尾的 header 比原 prior 範圍多出約 10–14 ms，以明確上限 30 ms 的等速度／等角速度外推並記錄，不靜默 clamp；原定位 prior 不變。

## 相同算法的三相機加速入口

加速版保留上節算法及數值門檻，使用 GPU 批次 PCA／整數 voxel 排序、向量化取樣與校正 map 快取。瓶頸、實測範圍及輸出一致性見 [效能報告](25_RIG_PERFORMANCE_20261018.md)。主機 ML venv 執行重建；ROS／C++ 未改，不需重編。

**歷史續跑命令：** 此 run 當時匯入 2487 幀後接續，現在兩條重建／定位皆已完成。下列保留供理解續跑入口，不需為本次文件整理再次執行。實際輸出追蹤見 [01 第 10 節](01_PROGRAM_FLOW.md#10-你目前查看的三相機-result_2csv-從哪裡來)，後續成績與診斷見 [29](29_THREE_CAMERA_SCORED_ANALYSIS_20261018.md)。

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
bash scripts/run_three_camera_full_fast.sh --resume \
  outputs/runs/three_camera_full_fast_imported_2610181301
```

下一輪從頭開始的手動入口：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
bash scripts/run_three_camera_full_fast.sh --dry-run
bash scripts/run_three_camera_full_fast.sh
```

若要接續已停止的**原版** run，先執行一次匯入工具；它只複製經核對的已提交幀，不啟動完整重建。使用者 2026-10-18 的舊 run 為 `outputs/runs/three_camera_full_2610181155`，停止時 Track1 已保存 2487 幀。

```bash
outputs/ml_runtime/venv/bin/python scripts/migrate_rig_fast.py \
  outputs/runs/three_camera_full_2610181155

# 將下列目錄換成匯入工具印出的新 Run；後續中斷也使用同一指令。
bash scripts/run_three_camera_full_fast.sh --resume \
  outputs/runs/three_camera_full_fast_imported_YYMMDDHHmm
```

不能直接把原版目錄交給加速版 `--resume`：新版另外核對 execution 的程式 hash、backend、精度、chunk 及匯入紀錄。原版已完成幀不重算；尚未完成幀用加速版，之後照原流程定位 Track1，再重建／定位 Track2。結果仍是實驗候選，尺度驗證未通過，沒有新增精度或 Kaggle 分數證據。

新幀 `frame.json.stage_seconds` 及 `performance.json` 記影像讀取校正、模型前處理／推論、反投影／座標轉換、取樣、PLY／labels 寫入；取樣另拆 voxel、exact CPU kNN、PCA 與 spatial round-robin。匯入的舊幀只有原先的總耗時，不能混入新版各階段平均。`--verify-sampling` 會額外重跑 scalar CPU 原版以檢查索引，僅供短段診斷，不能開著量正式加速速度。

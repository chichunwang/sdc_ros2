# Ubuntu 接手與首跑實測（2026-10-17）

> 本文件只記 Ubuntu 接手時的 CPU 單幀／前 50 幀實測。後續指定 CPU Track1 完整結果見 [完成紀錄](CPU_TRACK1_COMPLETED_20261017.md)；目前開發主入口為 [GPU 工作區](../../midterm01_gpu/README.md)，新版成績與完整兩條 Track 見 [sensor_guided 實測](../../midterm01_gpu/SENSOR_GUIDED_20261017.md)。下文「尚未完整測試／未取得分數」均指首跑當時。

本次直接在 `/home/ricky/sdc_ros2` 執行，起始 Git 工作樹乾淨；保留原始資料、HW2 容器與舊結果。完整證據位於 `outputs/runs/handoff_20261017/`（Git 忽略，不會隨 Git 同步）。原始 commit、修改前後 diff、Docker image ID、Compose 掛載與命令分別留存。

## 環境與建置

- Ubuntu 主機 Docker Engine 29.9.0 可用，帳號 ricky 已有 docker 群組權限。根分割區檢查時可用 53 GiB，RAM 31 GiB。
- 沿用現有 `sdc_hw2_ros2:latest` 建置獨立 `sdc_midterm01_ros2:latest`，啟動 `sdc_midterm01`。未啟動或重建 HW2 容器。
- ROS／colcon 均在 Midterm01 容器內執行。`colcon build --symlink-install --parallel-workers 2` 成功，colcon 摘要耗時 23.5 秒。既有 visualization.cpp 有 PCL deprecated API 警告，另有 CMake 相容性警告，未阻礙建置。
- `data_check_host.json` 與 `data_check_container.json` 的 strict-count 都通過，資料仍唯讀掛載。

本次主機命令（相對 repo 根目錄）：

```bash
python3 midterm01/scripts/check_data.py --strict-count \
  --output midterm01/outputs/runs/handoff_20261017/data_check_host.json
bash midterm01/scripts/dev.sh init
bash midterm01/scripts/dev.sh config
bash midterm01/scripts/dev.sh build
bash midterm01/scripts/dev.sh up
bash midterm01/scripts/dev.sh check-data --strict-count \
  --output /root/outputs/runs/handoff_20261017/data_check_container.json
bash midterm01/scripts/dev.sh build-workspace --parallel-workers 2
```

Docker 建置紀錄為 `docker_build.log`，建置總牆鐘時間未獨立量測；不可與 colcon 摘要或測試執行時間混為一談。

## 資料與 frame 核對

| 項目 | Track1 | Track2 |
| --- | ---: | ---: |
| 前相機 PLY | 3011 | 3299 |
| MapAnything PLY／index 列 | 3011 | 3299 |
| 空 PLY／無法解析時間／重複時間 | 0／0／0 | 0／0／0 |
| timestamp 排序對應檔名 idx | 0–3010 連續一致 | 0–3298 連續一致 |
| MapAnything index 引用缺檔 | 0 | 0 |
| 初始旋轉 det | 1.000000000685 | 1.000000000817 |
| 初始旋轉正交最大誤差 | 8.11e-10 | 6.08e-10 |
| SQLite quick_check | ok | ok |
| 原始地圖點數（header） | 59,531,846 | 50,881,350 |
| 地圖格式 | ASCII | binary |

`audit.json` 保存必要檔大小、地圖 SHA256、初始 4×4 矩陣、四組 PLY manifest、抽樣 header、topic 型態及數量。四組 PLY 的檔名均符合目前 C++ 可解析的正整數奈秒格式；每條 MapAnything 的 `f_ts`、`index` 逐列等於前相機時間序列與編號，引用路徑唯一且存在。

已找到 `TrackN/TrackN_images/camera_info/`，包含三相機內參、前鏡頭 mask 與 `lucid_camera_extrinsics.yaml`。外參明訂 `p_target = T_target_source × p_source`，平移單位 meter，提供相機彼此轉換，沒有提供車體或地圖外參。MapAnything report 宣告 `model-predicted front-camera frame`、`fusion_uses_calibrated_tf=false`，不能再直接套一次三相機校正融合。

`front_geometry.json` 抽查兩條 Track 的第 0、49、末幀：頂點分別為 Track1 41058／42740／36073，Track2 21878／21276／16046，均為有限 XYZ。前相機樣本 x 主要為正、z 較小，不能僅因名稱 camera frame 就擅自套 optical-to-body 軸轉換。現有 `ply_scale=3.0` 是未校準的 R0 參數，沒有由幾何抽樣證明公尺尺度正確。

官方 PDF 全 24 頁已擷取，並檢視資料頁；官方 p.8 確認 LoGeR／MapAnything 來源、幀數與 noisy initial pose。PDF 並未提供完整提交 frame／yaw 單位契約。本地 data 搜尋沒有找到 sample submission，因此 idx 僅驗證到資料檔名／影像／bag／MapAnything index 的對應，尚未驗證 Kaggle 完整 ID 集合。取得官方 sample 或明確補充規格後，才能確認提交格式與車體 pose 轉換。

目前計算定義為 `p_map = T_map_source × (3.0 × p_raw)`，第一幀初始矩陣僅作 NDT 初值，後續使用上一個接受 pose；沒有額外套車體外參。程式輸出的矩陣不能在外參未確認時宣稱就是評分要求的車體參考點。

## 首跑修正

- launch 新增 `use_rviz`，預設 false；headless 不需 DISPLAY。
- launch 新增 `map_cache_file`，預設空字串，避免向唯讀 data 寫快取。本次四次測試都停用快取，未重用舊 cache。如另啟快取，需明確指定 outputs 中以 Track、voxel、地圖版本區分的絕對路徑。
- launch 結果預設移至 outputs 並以啟動時間（含微秒）建立獨立 run 路徑；正式實驗仍建議明確指定具實驗意義的新 run 目錄。
- TXT 轉 CSV 移除未使用的 pandas、NumPy、SciPy imports，實際轉換僅需標準函式庫。
- 未改動既有 NDT、尺度、下採樣、局部地圖或收斂判定，保留前相機 R0 行為。

## 測試與證據

測試驅動程式 `run_tests.py`、唯讀資料核對 `audit.py`、bag header 檢查 `bag_audit.py` 均保存在本次輸出目錄。驅動程式先依序完成兩條 Track 單幀，再做兩條 Track 前 50 幀；各 run 使用新目錄並拒絕覆用既有目錄。

每個測試保存 `command.sh`、`params.json`／`params.yaml`、`run.log`、`result.txt`、`frames.csv`、`validation.json` 和診斷用 `diagnostic.csv`。總表為 `tests.json`；時間包含 ros2 程式啟動、完整地圖讀取／下採樣與定位，不包含 Docker／colcon 建置。所有測試不啟用 bag、不開 GUI、不使用 cache、`keep_alive=false`。

固定參數：map voxel 1.0、input voxel 0.3、submap 半寬 100／半高 30、NDT resolution 8.0／step 0.8／epsilon 0.05／30 iterations、ply_scale 3.0、min_input_points 10（既有程式尚未用於點數判斷）、publish_global_map=false。

診斷 CSV 的 yaw 仍為 atan2 輸出的弧度，idx 由 0 起；短序列 CSV 不具正式提交列數，也未上傳 Kaggle。PCL fitness 和收斂數量均不是正式 ATE。尚未執行兩條完整 R0、未取得 Kaggle 分數，不能宣稱超過 baseline。

`bag_audit.json` 另以 ROS 反序列化檢查所有訊息可讀，逐條 Track 核對所有前相機 PLY 時間均存在於對應前相機的 bag timestamp，並逐幀確認同名原始影像存在。這些時間並非 header timestamp：前鏡頭首末訊息的 bag-header 差約為 Track1 7.19／9.46 ms、Track2 9.93／6.93 ms。IMU header frame 為 `imu_link`，相機為 `gige_100_*_hdr`，GNSS header frame 空字串；因此不能由 GNSS frame 名稱推定 map／body 定義。bag 原始前相機訊息數 Track1 3056、Track2 3299，Track1 的 PLY 是對應子集，不應將 bag 序號直接當作提交 idx。

`header_audit.json` 已巡查全部 12620 個 PLY header，皆有正頂點數與 float XYZ。前相機均為 ASCII；MapAnything 均為 binary_little_endian，逐檔確認固定欄位 payload 長度與 header 頂點數相符，無錯誤。此檢查未宣稱已逐點掃描所有前相機 ASCII payload，也未證明重建幾何正確；實際點座標解析範圍為前述樣本與執行幀。

`calibration_audit.json` 已檢查六份內參為有限值且焦距為正，並核對兩條 Track 的三組相機外參正反轉換；正反乘積最大誤差低於 1e-9，旋轉行列式接近 1。這不包含車體外參或影像重投影校正品質驗證。

診斷擷取程式初版未正確處理 ROS PointCloud2 的 NumPy structured array，已修正並成功擷取；原始錯誤保留在 `capture_initial_error.log`，不影響定位程式與結果。`captured_clouds.json` 記錄擷取時間、frame_id 和點數；`track1_overlap.png`／`track2_overlap.png` 與 `overlap_diagnostics.json` 比較初始矩陣和接受的 NDT pose。此處最近鄰距離來自發布的 RGB 下採樣點雲，與 NDT 使用的 XYZ 下採樣點雲／fitness 不是同一計算。

## 實際結果

| 測試 | 結果列數 | NDT fallback | 總耗時（秒） |
| --- | ---: | ---: | ---: |
| track1_01frames | 1 | 0 | 97.643 |
| track2_01frames | 1 | 0 | 14.868 |
| track1_50frames | 50 | 0 | 322.519 |
| track2_50frames | 50 | 0 | 118.509 |

四次程序 return code 均為 0；每行恰為 12 個有限值，診斷 CSV 表頭、從 0 起 ID、列數與有限值已核對。兩條 50 幀的第一個 pose 與各自單幀結果逐字一致。完整列序／檔名／timestamp 位於各 run 的 `frames.csv`。

數值旋轉檢查最大正交誤差小於 2.3e-7，det 接近 1。但「收斂 50 次」不代表姿態穩定或正確：Track1 最大相鄰平移 0.796 m，依影像時間換算約 22.18 m/s；Track2 最大相鄰平移 0.262 m，約 7.17 m/s，且首末 yaw 從 -0.229 變成 -0.736 rad。這些是待分析的短序列現象，尚無 ground truth 可判定。不得將零 fallback 寫成定位品質通過。

疊圖診斷最近鄰中位數：Track1 第 2 幀初始／NDT 為 0.601／0.527 m；Track2 第 0 幀為 0.652／0.455 m。這些殘差僅表示局部最近鄰關係，不能用來確立尺度或車體參考點。

後續依序取得官方 sample／車體 frame／yaw 單位說明，確認前相機尺度與幾何，分析上述突跳，再保存完整兩條 Track R0；本次只執行 1／50 幀測試，沒有完整定位結果或 Kaggle 成績。

Headless launch 實跑（Track2 單幀）也通過，耗時 14.700 秒，return code 0，無 RViz、無 cache 讀寫，輸出與直接 ros2 run 的單幀 pose 完全一致。證據在 `launch_validation/`；命令刻意不指定 use_rviz／map_cache_file，以驗證安全預設值。

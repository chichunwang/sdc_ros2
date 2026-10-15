# sdc_ros2 專案說明與協作約定

## 專案與目前任務

這是 SDC 自駕車課程工作區，包含 ROS 2 感測器處理、點雲／Radar、機率模型及車輛定位作業。主要工作為 **Midterm01：將影像重建的 3D 點雲配準到 LiDAR 地圖，估計兩條 Track 每幀的 pose**。使用者單人完成一組，個人貢獻依實際工作記為 100%，不安排找組員或雙人分工。

2026-10-16 交接：後續由 **Ubuntu 裡的 Codex** 繼續。使用者已確認資料位於 `~/sdc_ros2/midterm01/data`；下一步是檢查既有資料、建置與保存目前 NDT 參考結果。不要重新以「尚未下載模板／資料」作為開工前提。

## 目錄與閱讀順序

| 位置 | 用途 |
| --- | --- |
| `hw2/` | 原有 ROS 2 Humble Docker、感測器分析工具與工作區，保留既有用途 |
| `hw3/` | Markov chain、模擬與穩態分布練習 |
| `docs/` | 通用作業／工具說明 |
| `midterm01/` | 本次定位專案、獨立 Docker、程式、資料與實驗／報告目錄 |
| `tmp/` | 忽略的暫存、PDF 擷取與預覽，不作為交接必要依賴 |

Midterm01 開工先讀：

1. `midterm01/README.md`：狀態、三層路徑、首跑與已知問題。
2. `midterm01/docs/SDC 2026 Midterm01 Competition -  Localization.pdf`：官方題目，共 24 頁。注意檔名在 `-` 後有兩個空白。
3. `midterm01/docs/README.md`：題目分析、評分、期限、改善與交付安排。
4. `midterm01/docs/KAGGLE_DATASET_SPEC.md`：使用者提供的 Kaggle 原文、topics、CSV 欄位及筆數。
5. `midterm01/data/README.md`：使用者已建立的資料樹、Ubuntu 檢查方式與 MapAnything 對應。
6. `midterm01/workspace/src/localization/` 的 `CMakeLists.txt`、`package.xml`、`include/localization/localization.h`、`src/localization.cpp`、`launch/localization.launch.py`、`processing_codes/txt_to_idx_x_y_yaw.py`。

資源索引在 `midterm01/docs/DOWNLOADS.md`。參考檔已移入 `midterm01/docs/`，不要依過期的 `localreference/` 路徑重建副本。`docs/UBUNTU_DATA_DOWNLOAD.md` 與 `scripts/download_kaggle.py` 目前均不存在，不恢復舊流程，也不引用為現有入口。正式要求以官方 PDF、Kaggle 及後續公告為準；分析建議不是課程強制演算法。

## 已知環境與證據範圍

| 層級 | 位置／設定 | 確認範圍 |
| --- | --- | --- |
| Windows | `C:\Users\ricky\Downloads\Project\sdc_ros2` | 本次可直接讀寫的 checkout；本地 Track1／Track2 是空目錄 |
| Ubuntu 主機 | 使用者確認 `~/sdc_ros2`；資料在 `~/sdc_ros2/midterm01/data` | 使用者文字確認與截圖；本次未 SSH 實測 |
| Midterm01 容器 | Compose service `midterm01`、project `sdc_midterm01`、預設容器 `sdc_midterm01` | 由目前設定推得；實際容器狀態待 Ubuntu 查驗 |
| HW2 容器 | service `ros2_dev`、映像 `sdc_hw2_ros2:latest`、容器 `ros2_container`、工作目錄 `/ros_ws` | 原有 HW2 設定，與 Midterm01 容器分開 |

Ubuntu 可直接登入或使用 VS Code Remote SSH；不必依賴 SSH 才能執行指令。先在 Ubuntu 用 `pwd`／`realpath` 確認絕對路徑，不把 `~` 預先展開成猜測的 `/home/...`。Windows 缺少 Docker／ROS／資料不代表 Ubuntu 缺少，不要求改裝 Windows Docker、WSL 或重下載 Ubuntu 已有內容。

現有 Midterm01 Compose 的主機來源均相對於 `midterm01/` 解析：

| Ubuntu 主機來源 | 容器目標 | 權限 |
| --- | --- | --- |
| `./workspace` | `/root/midterm` | 讀寫 |
| `.env` 的 `MIDTERM_DATA_DIR`，預設 `./data` | `/root/data` | 唯讀 |
| `./config` | `/root/config` | 唯讀 |
| `./scripts` | `/root/scripts` | 唯讀 |
| `./outputs` | `/root/outputs` | 讀寫 |

Dockerfile 延伸 HW2 基底映像；`dev.sh build` 缺少基底時只傳入 `hw2/Dockerfile` 和 `hw2/ros_entrypoint.sh` 建置。Docker 配置來源為使用者指定的 [HW2 講義](https://hackmd.io/@26f-sdc/rkh6TD-czg)，其 `sdc_hw2/` 對應本專案 `hw2/`，不移動現有目錄來迎合講義名稱。HW2 的 `guanfu_road_clipped_ros2` bag 已有存在紀錄，不能當作 Midterm01 Track 資料。

Windows／Ubuntu 是兩份 checkout。Git 同步程式；忽略的資料、`.env`、建置產物與輸出須留在各自環境。同步前檢查兩端 Git 狀態，保留既有修改；正式 PDF 與新 Markdown 目前仍需納入實際同步範圍，不能假設未追蹤檔會隨 Git 傳到 Ubuntu。

## 資料已建立的狀態

使用者於 2026-10-16 確認下列內容在 Ubuntu `midterm01/data/`：

- `Track1/`、`Track2/`：各有 `raw_ply/`、`Track1_images/` 或 `Track2_images/`、`init_guess.txt`、`map.pcd`、`metadata.yaml`、`Track1.db3` 或 `Track2.db3`。
- `kaggle_track01/Track1_Mapanything_ply/` 與 `kaggle_track02/Track2_Mapanything_ply/`：各有 `ply/`、`index.csv`、`report.json`。

截圖證明目錄與檔案名稱，未證明 PLY 筆數、非空、幾何／時間正確或下載完整。`camera_info/` 與 official sample submission 在截圖未見，先搜尋實際位置再判定是否缺少。MapAnything 的 `index.csv` 是待檢查的點雲索引，不直接視為 Kaggle submission。

歷史上 Track01 整包下載曾 404，但單檔與清單成功；使用者現已建立資料，不再以舊 404 當作當前阻塞，也不重設已可用的 token。除非確認缺檔，才補下載。

## 程式現況與必須先處理的落差

已存在完整 `localization` package 與基礎 NDT。以目前原始碼為準，不再宣稱「只有空架構」「尚無定位演算法」或要求先重寫 ICP：

- PLY 只掃描指定資料夾的第一層；檔名必須含 `_ts_<正整數奈秒>`，依解析出的時間排序，逐幀載入。無法解析的檔名會略過，現有檢查工具不會發現此差異。
- `LoadInitialPose()` 讀取 `init_guess.txt` 前四列的 4×4 矩陣，只用作第一幀；後續使用上一幀接受的 pose。
- 預設 `ply_scale=3.0`、map voxel 1.0、input voxel 0.3、submap 半寬 100／半高 30、NDT resolution 8.0／step 0.8／epsilon 0.05／30 iterations。這些是程式預設，尚非最佳或已校準設定。
- bag 預設停用；傳入 `bag_path` 才讀 GNSS／IMU，依 **bag timestamp** 保留不晚於 PLY 的最近訊息。目前 NDT 未利用它們修正 pose；資料過期、header timestamp、座標與外參仍需查驗。
- NDT 成功判定只有收斂與有限值；失敗保留上一個 pose。`min_input_points` 已宣告，但目前未用於點數判斷。壞 PLY、空 submap 等會提早結束，不能以「有結果檔／正常退出」推定輸出完整。
- 每幀結果是 3×4 矩陣按列攤平的 12 個數值，沒有 timestamp。轉換腳本依行號產生從 0 起的 `idx`，yaw 為 `atan2(R10,R00)` 的弧度；正式 ID、座標與單位須再對照 sample。

Ubuntu 首跑優先事項：

1. 驗證資料結構／數量及可解析的 PLY 時間，核對初始矩陣、尺度、座標系及車體外參；MapAnything 另外核對 `index.csv`、`report.json`，不要直接套前相機尺度或改名。
2. `localization.launch.py` 無條件開 RViz，沒有 `use_rviz` 參數；無 GUI 首跑用 README 的 `ros2 run localization localization`。如要改 launch，新增明確開關再測試，不捏造尚不存在的參數。
3. launch 固定 `map_cache_file=map_downsampled_1m.pcd`，會在唯讀資料目錄讀寫快取。首跑 `ros2 run` 省略該參數以停用快取；後續將快取改到 outputs 並區分 Track、voxel 與地圖版本。既有快取會略過下採樣，不能靜默沿用不符設定的檔案。
4. Python 轉換腳本 import `pandas`、`numpy`、`scipy`，目前實際轉換僅用標準函式庫。Dockerfile 未明確安裝 pandas／scipy；先檢查，決定補依賴或移除未使用 import 後驗證。
5. launch 的預設結果路徑在 package 內，執行時明確改到 `/root/outputs/runs/<run>/`，避免覆寫既有結果。`keep_alive` 預設 true，批次執行設 false。
6. `visualization.cpp` 仍是 Radar／PCD 舊介面且含占位路徑，不直接當成影像 PLY 回放工具；優先使用定位節點的點雲／TF／path。

## 建置、驗證與工作順序

Ubuntu 主機入口在 `midterm01/scripts/dev.sh`。完整首跑指令見 `midterm01/README.md`，依序：

1. 檢查 Git、實際資料與既有容器；`init` 只建立缺少的目錄和 `.env`，確認資料根目錄設定。
2. 可先用主機 Python 執行 `scripts/check_data.py --strict-count`；再做 `config`、`build`、`up`、`status`、容器 `check-data` 與 `build-workspace`。
3. 容器 source ROS 與 workspace；無 GUI、無 bag、無快取跑單幀，再跑固定短序列，保存參數／版本／輸出及耗時。
4. 保存兩條 Track 的目前 NDT 參考版 R0，再依失敗案例做單因素改善；ICP、多尺度、三相機與感測器融合均是候選實驗，不是課程指定順序。
5. 確認 CSV 內容、取得實際 Kaggle 評分後，才宣稱超過 baseline；保存最佳 `result_1.csv`／`result_2.csv`、報告及重現方法。

目前沒有整個 repo 的統一測試入口。`build-workspace` 沒有 package 會失敗。`check_data.py` 只查必要檔存在、非空、第一層 PLY 數量，不查 bag、校正、時間、索引、CSV 或幾何。`config/tracks.json` 只供該工具使用，不會自動更改 launch／C++。修改 C++ 後重新建置；對真實資料執行的檢查需保存報告，不能拿模擬或 Windows 空目錄的結果代替。

## 正式要求與協作規則

- 使用繁體中文說明與撰寫文件，保留必要英文技術名詞；限定目前任務範圍，保留使用者既有修改。
- 兩條 Track 都需完成並超過助教 baseline。CSV 表頭順序 `idx,x,y,yaw`，資料列分別恰為 3011／3299，不含表頭；完整 ID／順序及車輛 pose 規格另核對官方 sample。
- 分清程式內部弧度、提交單位與評分旋轉誤差的度。PDF p.19 是「平移 RMSE + 0.2 × 旋轉 RMSE」；不能把 PCL fitness 或 GNSS 當正式 ATE。
- 不重複套外參或初始矩陣。明確定義 source 到 map 的變換，再確認是否需轉為車體 pose。
- 保存 commit／未提交差異、參數、資料來源、frame 對應、輸出、耗時、失敗與兩條 Track 分數。原有下採樣／局部地圖／NDT 不宣稱為自己的新增貢獻。
- 大型資料、bag、建置產物與執行輸出遵守 `.gitignore`，不強制加入 Git，也不取消現有忽略規則。
- PDF 公告 Kaggle 截止與簡報寄送為 **2026-10-29 09:00**，當天課堂簡報 **4–6 分鐘**；E3 程式與報告為 **2026-11-05 23:59**。時區、網站最新限制、baseline 與單人檔名仍需官方確認。寄信／提交是待使用者執行或另行授權的事項，不因 PDF 提到就自動操作。
- 本次只完成 Windows 靜態核對與文件更新；未連線 Ubuntu、未編譯 C++、未跑實際點雲、未取得 Kaggle 分數。後續依實測更新「已完成／待確認」，不得把計畫寫成成功結果。

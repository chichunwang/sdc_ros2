# 需要自己下載的資料與程式

先用你的 Kaggle 帳號加入兩場競賽並接受規則。官方加入邀請在題目 PDF 第 16 頁。以下連結與資料內容取自該 PDF 第 8–12 頁；Kaggle 實際檔案、登入權限及規格以下載頁為準。

## 第一版必須取得

| 項目 | 下載來源 | 你要做的事 | 遠端 Ubuntu 建議位置 |
| --- | --- | --- | --- |
| 老師的 Localization 程式模板 | [Google Drive：Localization](https://drive.google.com/file/d/1dAnE_-QwlmxVQIdNu13cNBjqEZSL6vCp/view?usp=sharing) | 下載、解壓，找出完整 `localization/` ROS package | `~/sdc_ros2/midterm01/workspace/src/localization/` |
| Track01 官方資料 | [Kaggle：Track01 data](https://www.kaggle.com/competitions/sdc-2026-midterm-01-track-01/data?select=Track1) | 下載並解壓 Track1，保留官方檔案結構 | `~/sdc_ros2/midterm01/data/Track1/` |
| Track02 官方資料 | [Kaggle：Track02 data](https://www.kaggle.com/competitions/sdc-2026-localization-track-02/data?select=Track2) | 下載並解壓 Track2，保留官方檔案結構 | `~/sdc_ros2/midterm01/data/Track2/` |
| Track01 sample submission | 同上 Track01 的 Kaggle Data 頁 | 取得官方提交範例，確認 ID、欄位與順序 | 建議放在 `data/Track1/sample_submission.csv` |
| Track02 sample submission | 同上 Track02 的 Kaggle Data 頁 | 取得該 Track 的提交範例，分開保存 | 建議放在 `data/Track2/sample_submission.csv` |

若完整資料包已包含 sample submission，不需要再下載一份。上表的 CSV 名稱是建議的本地儲存名稱，不是對官方下載檔名的宣稱。

每條 Track 第一版會用到 `map.pcd`、`init_guess.txt` 與 `raw_ply/`。校正資料 `camera_info/` 也應一併保留，用於確認尺度與座標。官方完整包如包含影像、bag 及 `metadata.yaml`，請保留原始結構；開始配準時可先直接使用已重建的 `raw_ply/`。

### 模板怎麼放？

找到老師壓縮檔裡這一層：

```text
localization/
├── package.xml
├── CMakeLists.txt
├── include/
├── src/
├── launch/
└── 其他原有內容（含後處理工具）
```

將整個 package 放到 `workspace/src/localization/`。保留其中的標頭檔、launch、RViz、設定與後處理腳本；範本工作區的 `build/`、`install/`、`log/` 留待自己的環境重新產生。模板附帶的 Docker 設定先保留在解壓來源，取得後再比較依賴；本架構的容器設定位於 `midterm01/`。

官方轉換腳本 `txt_to_idx_x_y_yaw.py` 應隨 Localization 模板取得。PDF 的目錄文字與截圖有 `process`／`processing_codes` 差異，下載後搜尋實際檔案，不需要另外找不明來源的同名程式。

### 資料已在 Ubuntu 其他位置時

可以直接使用現有資料。例如實際資料根目錄包含 `Track1/`、`Track2/`，在 `.env` 填入該根目錄的絕對 Linux 路徑：

```dotenv
MIDTERM_DATA_DIR=/你的實際資料根目錄
```

此行是格式示例，請換成真實路徑。Docker 會把它掛至 `/root/data/`。兩條 Track 的資料夾名稱若不同，調整 `config/tracks.json`；老師模板內的資料路徑也要在接入時同步核對。

## 後續改善才需要

| 項目 | 下載來源 | 何時需要 |
| --- | --- | --- |
| Track01 三相機點雲 | [Track01 3cameras_ply](https://www.kaggle.com/competitions/sdc-2026-midterm-01-track-01/data?select=3cameras_ply) | 前相機流程完成後，比較多視角點雲 |
| Track02 三相機點雲 | [Track02 3cameras_ply](https://www.kaggle.com/competitions/sdc-2026-localization-track-02/data?select=3cameras_ply) | 同上，使用該 Track 的對應點雲 |
| LoGeR 程式 | [Google Drive：LoGeR，選用](https://drive.google.com/file/d/1ZHeWxVrf86GDtHtFY7JyXY7w1vge3Fsa/view?usp=sharing) | 要自行修改影像重建流程時再下載 |

第一版已直接使用官方重建點雲，因此暫時不需要自己下載或訓練 LoGeR／MapAnything 模型。IMU、GNSS 與原始影像優先使用官方 Track 包內的資料；只有在決定研究這些改善時才解析。

## 已有環境與本架構提供的內容

- HW2 的 Ubuntu 資料已存在，但它是 `guanfu_road_clipped_ros2`，不能代替本次兩條 Track。
- ROS 2 Humble 基底沿用專案 `hw2/` 的 Docker 配置；Midterm01 Dockerfile 會透過 apt 安裝 PCL 等編譯依賴，無須手動下載 PCL 原始碼。
- 本架構已提供掛載、建置入口、資料結構檢查與實驗／報告目錄；目前尚未包含官方模板或定位演算法。
- 在 Windows 下載後，仍需將大檔放到遠端 Ubuntu 的實際資料目錄。被 Git 忽略的資料不會隨一般程式同步傳過去。
- 本次沒有代你下載資料、加入競賽、連線遠端 Ubuntu 或提交任何結果。

## 下載完成檢查清單

- [ ] 兩場 Kaggle 都已加入，能讀取各自 Data 頁。
- [ ] `workspace/src/localization/package.xml` 與 `CMakeLists.txt` 已放妥。
- [ ] 兩條 Track 各有自己的 `map.pcd`、`init_guess.txt`、`raw_ply/` 與校正資料。
- [ ] 兩條 Track 的官方 sample submission 已分別保存。
- [ ] `.env` 指向遠端 Ubuntu 上正確的 Midterm01 資料根目錄。
- [ ] `bash scripts/dev.sh check-data --strict-count` 通過；此工具只驗證檔案結構與數量，座標及 CSV 內容另行核對。


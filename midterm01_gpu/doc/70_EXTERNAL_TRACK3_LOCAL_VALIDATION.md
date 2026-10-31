# 作者 Track3 程式：本機驗證與操作

這個入口用本機 Track3 資料執行 `midterm01/others code` 的 CPU 因果版流程，讓你查看其輸出、尺度與配準診斷。它是獨立研究對照，不取代現行 GPU 定位器。

目前 Track3 資料尚未準備，因此這輪只完成環境與生成資料檢查，沒有真實定位結果。完整 Track 仍由你親自啟動。

## 準備資料與執行

把**已有的官方 Track3 原始資料**放回以下位置，不要用 Track1／Track2 改名代替：

```text
/home/ricky/sdc_ros2/midterm01/data/Track3/
├── raw_ply/          # 3142 個原始 PLY，保留原名與整數奈秒時間戳
├── map.pcd           # 作者轉檔器要求 ASCII PCD
├── init_guess.txt
└── Track3.db3        # bag 根目錄只應有這一個 db3
```

資料需要放在工作區內，才能由既有容器的唯讀 `/workspace` 掛載讀到。若你放在工作區內其他位置，直接改 [`scripts/external_parameters.py`](../scripts/external_parameters.py) 的 `TRACKS[3]["data_relative"]`；同檔 `PROCESSES = 2` 控制 CPU 配準程序數。其餘參數保持作者入口設定。Track1／2 的新增入口與參數差異見 [71｜Track1／2 本機驗證](71_EXTERNAL_TRACK12_LOCAL_VALIDATION.md)；Track3 的各階段命令仍與作者 shell 逐項對照。

在 Ubuntu 主機執行：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
./scripts/dev.sh up
./scripts/run_external_track3.sh --check
```

`--check` 執行作者的 22 項合成單元測試及包裝器契約檢查，也確認資料檔名、幀數與地圖格式；不執行真實定位。請查看印出的 run 路徑內 `data_check.json`，`ready: true` 才表示輸入前置檢查已通過。這不是定位精度檢查。

資料就緒後，**由你親自啟動完整執行**：

```bash
./scripts/run_external_track3.sh --full
```

每次建立新的 `outputs/run/track03_full_YYMMDDHHmm/`，同分鐘碰撞自動加序號。這支入口與三個 Track 共用 lock；其他 Track 正在執行時直接拒絕，不排隊、不停止既有程序。原始資料、作者資料夾和歷史輸出不改寫。

主機的系統 Python 不需要另裝 numpy 等套件。入口使用既有 `midterm01_track3` 容器，每輪只把 `polars==1.44.2` 裝進 run 的 `deps/`，不更動共用容器套件。這需要首次安裝時能連線套件來源；其餘實際套件版本存於 `environment.json`／`environment_freeze.txt`。

執行包含地圖快取、整段 PLY 快取與四輪 ICP，請預留原始資料之外的磁碟與記憶體。作者 README 的「約 35 分鐘」是作者環境描述，本機尚未測得完整耗時。停止時先在原本終端按 `Ctrl+C`，再查看 run 的狀態與 log；不要重用同一個 run。

## 結果放在哪裡、怎麼看

| 檔案／目錄 | 用途 |
| --- | --- |
| `result_3.csv` | 作者本輪最終 CSV 的原樣複製，3142 列 `idx,x,y,yaw`，yaw 為弧度 |
| `work/track3_cam_causal.csv` | 作者流程直接產生的最終 CSV，兩檔 SHA 必須一致 |
| `work/*.npz`、`work/*.npy` | 尺度、循序配準、四種範圍結果、混合配準與 EKF state |
| `contract_audit.json` | 索引、筆數、有限數值、時間戳與各階段 frame 順序檢查 |
| `stage_timings.csv` | 各階段實測 wall time，不是逐幀耗時 |
| `observations.jsonl` | 各階段 `elapsed_seconds` 與 return code |
| `commands.json`、各階段 `.log` | 本輪實際命令與原始 console 診斷 |
| `external_original/` | 作者原始碼的未修改備份 |
| `external_source/` | 本輪實際執行副本，只做下述兩處 I/O 相容修正 |
| `external_compatibility.patch` | 原始備份到執行副本的確切差異 |
| `source/`、各 manifest | 包裝器快照、Git commit／差異、原始與執行副本 SHA |
| `frame_manifest.csv`、`input_sources.json` | 整數奈秒時間戳、PLY／map／init／bag 來源 SHA |
| `test_info.json` | 狀態、版本、參數假設、耗時、錯誤與輸出 SHA |

成功 full 後自動生成 `analysis/`：

1. `01_trajectory.png/pdf`：地圖座標下的 XY 軌跡。
2. `02_pose_changes.png/pdf`：相鄰位移、yaw 與相鄰 yaw 變化。
3. `03_registration_timing.png/pdf`：**各階段**耗時，圖上明示無逐幀計時。
4. `04_registration_and_scale.png/pdf`：尺度、非地面貼合比例、XY 資訊量與採用距離。
5. `README.md`、`manifest.json`：判讀限制與繪圖來源 SHA。

圖表失敗寫入 `analysis_status.json`，不會把定位成功改成失敗。原始程式沒有逐幀耗時，也沒有最後覆換 yaw 之後的完整 3D pose，因此不補造 `timings.csv` 或 `poses.txt`。這份對照的 03 圖保留固定檔名，但內容明確是階段耗時。

## 相容修正與驗證範圍

本機首次檢查為 20 passed、2 failed，兩個失敗都出現在 ASCII PLY 讀取。生成資料實驗確認：Python buffered reader 的邏輯位置停在 header 後，底層 file descriptor 已預讀到檔尾；此環境的 Polars 從底層位置讀取，因而得到空 CSV。將檔案以 `buffering=0` 開啟後，兩個位置一致，生成資料的數值完整讀回。

只對**執行副本**的兩個檔案套用相同修正：

- `src/sdcloc/ply.py`：`path.open('rb', buffering=0)`。
- `scripts/convert_map.py`：`src.open('rb', buffering=0)`。

這兩處改的是讀取介面，沒有改 ICP、尺度、濾波、yaw 算法或參數。包裝器另外以假 Python 記錄 argv，核對每一階段命令與作者 `run_track3_causal.sh` 相同；這個檢查不讀真實資料或執行定位。

作者的「因果版」仍保留 `gyro-dt=0.14`、從全段找第一個有效配準，以及最後另外產生 yaw 等行為，尚未證明符合本專案競賽規則；這輪是研究驗證入口，不代表可正式提交。作者附帶的單元測試也涵蓋離線 smoother 函式，但 full 入口只呼叫 `run_track3_causal.sh` 對應階段，不呼叫離線 `build_nognss.py`。

沒有獨立真值，也沒有作者已評分 CSV 可供 bytes 比對。之後 full 的格式檢查通過，只能證明本輪流程和資料契約通過，不能宣稱作者 Public 分數已重現。

<details>
<summary>本次檢查證據</summary>

- 首輪失敗：[`track00_test_2610291339`](../outputs/run/track00_test_2610291339/test_info.json)，保留原樣。
- buffered I/O 最小生成實驗：[`buffered_io_diagnosis.json`](../outputs/run/track00_test_2610291340/buffered_io_diagnosis.json)。
- I/O 修正後 22 項作者測試與 3 項初版包裝器檢查通過：[`track00_test_2610291341`](../outputs/run/track00_test_2610291341/test_info.json)。圖表是環境檢查及生成資料，不是真實軌跡。
- 最終檢查：[`track00_test_2610291342`](../outputs/run/track00_test_2610291342/test_info.json)，作者 22 項測試與包裝器 4 項檢查全數通過；作者測試 3.40 秒、包裝器檢查 2.58 秒。包裝器檢查包含生成 ASCII 地圖轉檔、完整格式圖表與來源不變、錯序／NaN 拒絕、命令逐項對照。作者既有 smoother 測試有 4 個 empty-slice／invalid-divide warning，Matplotlib 有既有 Axes3D warning；本次 2D 圖成功。
- [環境檢查圖](../outputs/run/track00_test_2610291342/analysis/01_environment_checks.png)與[圖表說明](../outputs/run/track00_test_2610291342/analysis/README.md)。來源原檔 SHA 與快照原始備份相同，analysis manifest SHA 核對通過；沒有真實定位。

</details>

# sdc_ros2 協作規則

## 專案與目錄

這是 SDC 自駕車課程工作區，主要任務是 Midterm01：將影像重建的 3D 點雲配準到 LiDAR 地圖，估計各 Track 每幀 pose；目前共有三條 Track。使用者單人完成一組，個人貢獻依實際工作記為 100%，不安排組員或雙人分工；AI 分工不算組員分工。

| 位置 | 用途 |
| --- | --- |
| `midterm01/` | CPU 參考版本、共用 data／config、官方題目與交付文件 |
| `hw2/` | 原有 ROS 2 Humble Docker、感測器分析工具，保留既有用途 |
| `hw3/` | Markov chain、模擬與穩態分布練習 |
| `hw4/` | Radar odometry／GPS EKF；沿用 HW2 Humble 容器，資料與執行由使用者在 Ubuntu 手動處理 |
| `docs/` | 通用作業／工具說明 |
| `tmp/` | 忽略的暫存／擷取／預覽，不作為交接依賴 |

## 工作範圍與資料保護

- 使用繁體中文說明與記錄，保留英文技術名詞及使用者既有修改。
- Midterm01 的開發、建置與測試統一使用 `midterm01_gpu/`；開工先讀其 [README](midterm01_gpu/README.md) 與 [AGENTS.md](midterm01_gpu/AGENTS.md)。`midterm01/` 保留 CPU 參考程式、既有結果與共用原始資料；`hw2/`、`hw3/`、`hw4/` 保留各自用途。
- 不搬移、覆寫或重算既有 CPU 結果，不改原始資料、bag、既有 CSV／矩陣／manifest。舊 metadata 路徑只在讀取時解析，不回寫證據檔；確認缺檔後才補下載。
- Docker 指令在 Ubuntu 主機執行，ROS／colcon 在對應容器執行。GPU 啟停使用自己的 `scripts/dev.sh`，不重啟 Docker daemon 或停止／重建 CPU project；容器內同名路徑不可當成同一份主機資料。
- 資料、`.env`、build／install／log、outputs 遵守 `.gitignore`，不強制加入 Git，也不假設忽略檔案會跨電腦同步。權限修復限指定 run，不改 `/root` 或 data，不用 `chmod -R 777`。

## 執行與授權

- **完整 full Track 定位由使用者親自啟動。**助理完成已授權的開發與必要檢查、提供可直接執行的命令；不得自行啟動或重跑 full。
- 短段定位與研究測試依當次明確授權範圍執行；未授權的定位交由使用者手動啟動。有限幀數授權不得延伸成 full 授權；讀取既有輸出或整理文件不需重跑定位。
- 完成有依據的算法改動及必要的機制、來源、數值與輸出檢查後，即可提供手動 full 命令；不以尚未證明本機精度改善為由新增「full 資格」關卡。已知失敗或輸出 bytes 相同的版本，不推薦原樣重算／重複提交。
- 修改程式前先查是否有使用者啟動的 full。保護執行中的程序、原檔與來源 SHA，不停止、重啟或中途換檔；更新先暫存合規 run，等程序退出並核對原 SHA 後才套用。定位／GPU 工作不並行爭用同一張 GPU。
- Kaggle 提交、E3 上傳與寄信由使用者執行，或依另行明確授權操作；文件提到交付要求不代表已授權執行。已授權的必要工作持續完成，不增加例行確認關卡。

## 輸出路徑、命名與保存

所有完整、短段、研究及環境測試只放在 **`midterm01_gpu/outputs/run/`（單數）**，每次建立新的唯一目錄：

| 類型 | 命名 |
| --- | --- |
| Track1 | `track01_full_YYMMDDHHmm`／`track01_test_YYMMDDHHmm` |
| Track2 | `track02_full_YYMMDDHHmm`／`track02_test_YYMMDDHHmm` |
| Track3 | `track03_full_YYMMDDHHmm`／`track03_test_YYMMDDHHmm` |
| 跨 Track／環境測試 | `track00_test_YYMMDDHHmm` |

- 同分鐘重名加 `_0002` 等碰撞序號；不重建 `current/`、`best/`、`runs/`、`submissions/` 分類或測試索引。
- 每次 run 保存算法／版本、參數、commit／未提交差異、來源 SHA256、frame manifest、結果、耗時與失敗資訊；分數及來源、舊路徑記在該 run 的 `test_info.json` 等紀錄中。時間戳保持整數精度。
- 既有結果、已評分 CSV 與來源 SHA 原樣保留；新算法／診斷另存，不覆寫歷史證據。

## 結果檔案與 full 圖表

- 新 runner 的 `timings.csv` 欄位順序固定為：

```text
idx,original_idx,timestamp_ns,backend,load_ms,registration_ms,total_ms,map_accepted,scale
```

- `idx` 是本輪 local 0-based 索引；console 的 `idx` 對應 `original_idx`；`timestamp_ns` 是整數奈秒。保留 observations 的 `elapsed_seconds`；準備、暖機、配準、graph 與畫圖時間分開記錄。
- 每幀 TXT 為按列攤平的 3×4 矩陣，共 12 個數值、不含 timestamp。結果 CSV 表頭固定 `idx,x,y,yaw`，idx 從 0 起；現有轉檔 yaw 為 `atan2(R10,R00)` 弧度。Track1／2／3 各為 3011／3299／3142 筆資料，不含表頭；格式可讀或上傳成功不代表 frame／單位契約已驗證。
- 每次成功完成 full 後，以 Python／Matplotlib 產生唯讀圖表，放在該 run 的 `analysis/`：`01_trajectory.png/pdf`、`02_pose_changes.png/pdf`、`03_registration_timing.png/pdf`、`manifest.json` 與 `README.md`。manifest 保存來源與 SHA；圖表不修改 pose、CSV、參數或定位結果。
- 圖表失敗另記 `analysis_status.json`／log，保留定位成功狀態與原始輸出，不宣稱圖表完成。短段、失敗或未完成 run 預設不產生完整比較圖；歷史 run 不搬移、不補圖，不為畫圖重跑定位。
- 固定教學素材保留於 `midterm01_gpu/doc/analysis/`。輸出細節與新 full 圖表重試入口見 [08｜操作手冊](midterm01_gpu/doc/08_MANUAL_WORKFLOW.md)。

## 學習目的與算法改進原則

- 以使用者能理解、推導、解釋並延伸算法為目標：觀察證據 → 提出假說 → 說明模型、變數、殘差與約束 → 實作與驗證。非必要不調參；必要調整交代物理／統計／標定或數值依據與驗證方法，不盲目掃榜分。
- 所有 Track 遵守因果定位：不使用未來資訊回修舊 pose，不使用 RTS／backward／non-causal smoothing 或定位後離線軌跡修正。定位地圖只用 XYZ，不利用時間欄位還原軌跡；外部預測及 GNSS 不當真值，不依多次榜分反推隱藏真值。
- 分清計畫、實作、合成檢查、真實短段與正式 full 提交成績。接受率、殘差、收斂、GPU 使用率及輸出完整不能當作精度證明；如實保存成功／失敗、分數來源與顯示精度，不捏造缺少的診斷資料。
- GPU 實作與環境限制遵守 [GPU AGENTS.md](midterm01_gpu/AGENTS.md)；詳細開發原則見 [19](midterm01_gpu/doc/19_DEVELOPMENT_PRINCIPLES.md)，現行官方規則與程式盤點見 [48](midterm01_gpu/doc/48_RULES_AND_CODE_AUDIT_20261023.md)。

## Subagent 分工

範圍明確、可獨立完成且委派有益的簡單工作主動使用對應角色；一兩個指令或緊密相依的工作由主 agent 處理。主 agent 模型保持使用者選擇，最多同時兩個 subagent。

| 角色 | 模型／reasoning | 工作 |
| --- | --- | --- |
| `quick_scan` | `gpt-6-luna`／`mid` | 唯讀找檔、log、CSV／SHA 核對 |
| `docs_helper` | `gpt-6-luna`／`max` | 依確認事實整理文件與命令 |
| `small_fix` | `gpt-6.1-sol`／`medium` | 已確認原因與規格的局部小修及輕量檢查 |

- 算法、CUDA／solver／數值、frame／時間／尺度契約、跨模組設計及最終驗證由主 agent 負責；建置、重建與 GPU 工作統一協調。
- 委派須交代背景、確切讀寫範圍、驗收與禁止動作；遵守根與目標目錄 AGENTS，不覆寫他人修改，不同 agent 不同時改同檔，subagent 不再向下委派。委派不擴大執行授權，主 agent 檢查差異與證據。
- 設定見 `.codex/config.toml` 與 `.codex/agents/`；依工具實際能力選角色／模型，不把 TOML 語法通過當成模型已執行，不宣稱未成功的模型切換。

## 文件入口與維護

- 測試完的都放在 `/midterm01/output/run/該次full test track/` 。
- 測試結果都要給圖表讓我做分析，圖表要放在 `/midterm01/output/run/該次full test track/analysis/` 。

# Midterm02 Visual SLAM 工作區規則

本目錄是 SDC Midterm02（Visual SLAM）的開發、測試與交付入口，位於 `midterm02/main` 分支（worktree 目錄名仍為 `midterm01-kalman-filter`）。開工先讀本檔、[README](README.md)、[投影片規則整理](doc/01_COMPETITION_RULES_20261113.md) 與 [Kaggle 規則／資料原文](doc/02_KAGGLE_RULES_TRACK01_20261113.md)。

根目錄 [AGENTS.md](../AGENTS.md) 的語言、資料保護、授權、subagent 與文件規則繼續適用；**下列條目與 Midterm01 規則衝突時，以本檔為準**（Midterm02 的任務本質不同）。

## 與 Midterm01 規則的差異（必讀）

| 項目 | Midterm01 | Midterm02（本檔） |
| --- | --- | --- |
| 任務 | 影像點雲配準到給定 LiDAR 地圖 | 無 LiDAR、無地圖；從影像／LoGeR 點雲自建軌跡（SLAM） |
| 因果性 | 只能因果定位，禁止回修舊 pose | **允許且需要** loop closure 與 pose graph 全域最佳化回修所有 pose（官方流程本身要求）；但禁止手改 pose 與依榜分反推真值 |
| 座標／尺度 | 對齊官方 map frame | world frame 與整體尺度任意（評分做 Sim(3) 對齊）；只需「形狀」正確 |
| 結果格式 | `idx,x,y,yaw` | 官方 `timestamp,m00..m33`（camera-to-world 4×4，OpenCV 軸） |
| 感測器 | GNSS／IMU 依 Track | 只能用 Kaggle 提供的資料：影像、raw_ply、內參、IMU（僅 Track01）；**禁止 GPS、外部地圖、衛星圖、往年資料** |

## 競賽硬性規則（違反即失分）

- 每張影像都要有一列 pose，含非 keyframe；缺列、重複、非數值會被拒收。旋轉部分必須是合法旋轉矩陣（正交、det=+1），最後一列 `0,0,0,1`。
- 只評分位置 `m03,m13,m23`：Sim(3)（Umeyama）對齊後的 RMSE ATE，越低越好。Public 約 10%（Track01 為 132 張，同一段路線）、Private 約 90%（1182 張），兩者各自對齊；**最終成績只看 Private**，不得只追 Public 分數。
- 必須優於 Private 榜的 baseline 才有分。只跑現成系統、只調參數／config、或多次執行挑最佳 → 貢獻 0 分且不排名。所有使用的現成系統、模型、套件要記錄版本，並記錄「原始分數 → 每項修改後分數」。
- 最終選定的兩份提交都必須能用上傳到 E3 的程式**重現**（TA 會重跑）。因此所有正式結果必須來自固定程式、固定參數、可重現的亂數種子；非確定性元件需固定 seed 或記錄變異範圍。
- 不手改或手標任何 pose、不使用 GT 或非官方提供的測試序列資料、不跨隊分享程式或提交、不使用其他感測器或外部資料、不多帳號或借用他隊帳號。**違規整場 0 分**。
- 評分等同 `evo_ape --align --correct_scale`。最終兩份提交必須由使用者手動選定；未選時 Kaggle 會自動挑 Public 最好的，可能不是 Private 最好的。官方規則已明確允許的公開預訓練模型（含 SuperPoint／LightGlue 等學習式特徵匹配器、VGGT、MapAnything、MASt3R）與開源 SLAM／SfM 系統，可直接用於正式結果（使用者 2026-11-16 決定，取消「待 E3 QA 確認」）；報告須列出名稱、版本、授權，並說明自己的改動。只有官方規則未涵蓋的工具才先到 E3 Competition2 QA 詢問。
- Kaggle 每日最多 10 次提交（台灣時間 08:00 重置），最多選 2 份最終提交。Kaggle 提交、選定最終提交、E3 上傳、寄送簡報 email 一律由使用者執行；助理只準備檔案與檢查。

## 執行與授權

- 完整 Track SLAM（full）由使用者親自啟動；助理完成開發與必要檢查後提供可直接執行的命令。短段（限定幀數）測試須有明確幀數授權，不得延伸成 full 授權。
- 修改程式前檢查是否有使用者啟動中的 full；不停止、不重啟、不在執行中換檔。GPU 可以同時跑多個工作（使用者 2026-11-16 決定，取代根目錄 AGENTS.md「GPU 工作不並行」的規定；96 GB 顯示記憶體足夠）。
- **算法程式從頭重寫（使用者 2026-11-13 決定）**：不搬移、不 import `midterm01/`、`midterm01_gpu/` 的算法程式（配準、濾波、尺度、IMU 等），以官方 starter 為起點自行實作。只沿用 midterm01_gpu 的 run 目錄／輸出格式與證據保存規則。
- Kaggle 資料包整包原樣解壓到 `data/`（得到 `data/data_VSLAM-T1/Track1/data/…` 與 `data/data_VSLAM-T1/image_slam_starter/`），唯讀，不改名、不重寫、不覆寫。starter 原檔留在資料包內；修改一律在 `vslam/` 的副本進行，並在 `vslam/origin.json` 記錄來源檔與 SHA256。
- **資料授權**：資料（含 bundle 內 starter code）只供本課程比賽，未經助教許可不得散布。`data/` 永不進 Git；含 starter 衍生程式的 `vslam/` 只能推到私人遠端。
- **執行平台：Ubuntu（使用者 2026-11-13 決定）**。所有 SLAM 執行、測試與 full 都在 Ubuntu 主機的專用 Docker（Compose project `midterm02_vslam`，用 `scripts/dev.sh` 管理）內進行；Windows 只用於編輯與文件。不改 Docker daemon，不動 midterm01 的容器。
- 環境以官方 README 的 Python 3.9＋numpy、open3d、gtsam、small_gicp、opencv-python 為基準；新增套件／模型時記錄名稱、版本、來源與授權狀態到 `doc/` 與 run 紀錄。

## 輸出路徑、命名與保存（沿用 midterm01_gpu 格式）

所有 full、短段、研究及環境測試只放在 **`midterm02/outputs/run/`（單數）**，每次建立新的唯一目錄，用 `scripts/create_run.py` 建立：

| 類型 | 命名 |
| --- | --- |
| Track01 | `track01_full_YYMMDDHHmm`／`track01_test_YYMMDDHHmm` |
| Track02 | `track02_full_YYMMDDHHmm`／`track02_test_YYMMDDHHmm` |
| 跨 Track／環境測試 | `track00_test_YYMMDDHHmm` |

- 時間為 Asia/Taipei；同分鐘重名加 `_0002` 等碰撞序號。不建立 `current/`、`best/`、`submissions/` 等分類。
- 每個 run 目錄固定內容：

| 檔案 | 內容 |
| --- | --- |
| `source/`、`source_manifest.json` | 執行前凍結的程式與設定快照、每檔 SHA256、commit |
| `git_status.txt`、`uncommitted.patch` | 執行當下 Git 狀態與未提交差異 |
| `test_info.json` | 版本、參數、seed、輸入 SHA、frame 數、各階段耗時、狀態（success／failed／stopped）、失敗訊息、分數與分數來源 |
| `frame_manifest.csv` | `idx,timestamp_ns,image,ply`，依 timestamp 排序；idx 0-based |
| `timings.csv` | 欄位固定：`idx,timestamp_ns,backend,load_ms,preprocess_ms,registration_ms,total_ms,is_keyframe,fitness,inlier_rmse` |
| `keyframes.csv` | `kf_id,idx,timestamp_ns` |
| `loop_closures.csv` | `query_kf,match_kf,accepted,fitness,inlier_rmse,reason` |
| `poses_odometry.csv`、`poses_optimized.csv` | 最佳化前後的官方格式 pose，便於比較 loop closure 效果 |
| `result_1.csv`／`result_2.csv` | 可直接上傳 Kaggle 的官方格式（Track01／Track02），名稱同 E3 交付要求 |
| `analysis/` | 成功 full 的唯讀圖表（見下） |

- `timestamp_ns` 一律以整數字串處理（影像檔名去掉 `.jpg`），不得轉成 float。
- 每份 `result_*.csv` 產生後必須通過 `scripts/check_submission.py`（列數、表頭、timestamp 與 `sample_submission.csv` 完全一致、數值、旋轉合法）。通過檢查不代表精度已驗證。
- 既有 run、已評分 CSV 與來源 SHA 原樣保留；新算法另開 run，不覆寫歷史證據。使用者回報的 Kaggle 分數記在對應 run 的 `test_info.json`（`public_score`、`reported_by`、`reported_at`）。

## 結果圖表

- 每次成功 full 後，以 Python／Matplotlib 產生唯讀圖表於 `analysis/`：`01_trajectory.png/pdf`（俯視軌跡，odometry vs optimized，標出 keyframe 與 loop edge）、`02_pose_changes.png/pdf`（逐幀位移／轉角）、`03_registration_timing.png/pdf`（配準耗時與 fitness）、`04_loop_closures.png/pdf`、`manifest.json`（來源檔 SHA）與 `README.md`。
- 地圖檢查用 `build_map.py` 輸出的點雲／截圖也放在 `analysis/`，重影或彎牆即軌跡錯誤的證據。
- 圖表失敗另記 `analysis_status.json`，不改定位結果、不宣稱圖表完成；短段或失敗 run 不產生完整比較圖。

## 學習目的與算法改進原則

- 依「觀察證據 → 假說 → 模型（變數、殘差、約束）→ 實作 → 驗證」推進；每項改動單獨驗證，記錄改動前後分數，作為報告的「貢獻」與消融實驗（ablation）依據。
- 非必要不調參；必要調整說明物理／統計依據。LoGeR 點雲尺度只「大約是公尺」且逐幀略有差異、含天空點，尺度一致性與前處理是主要風險。
- 本機無 GT：以閉環誤差、`build_map.py` 地圖一致性、odometry 與 optimized 差異等自我一致性指標判斷，並明確區分「已實作、合成檢查、真實短段、full、正式分數」。Public 分數只是一小段路線，不得當成整體精度證明。

## 交付物對照（E3：`<ID1>_<ID2>_vslam.zip`）

整份程式（不含原始資料）＋`result_1.csv`＋`result_2.csv`＋`vslam_report_<ID1>_<ID2>.pdf`＋重現用 README。交付草稿放 `deliverables/`；打包與上傳由使用者執行。

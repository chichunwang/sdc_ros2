# GPU 工作區輸出

目前 Midterm01 的新實驗統一放本目錄。GPU 容器 `sdc_midterm01_gpu` 的 `/root/outputs/` 對應主機 `/home/ricky/sdc_ros2/midterm01_gpu/outputs/`；原 CPU 容器同名路徑則對應 `midterm01/outputs/`，不能混用。

- `runs/`：每次唯一 run 的參數、版本／來源、frame、TXT／CSV、sensor prior、guidance、耗時、GPU 取樣與驗證。
- `submissions/`：完整候選 `result_1.csv`／`result_2.csv`、來源與 SHA256；Kaggle 分數另記，CSV 存在不代表已提交。
- `figures/`：衍生軌跡與點雲圖。

最新已評分最佳為 heading_rts＋directional，Track1 **1.15275**／Track2 **0.30576**。`latest_geometry_submission.txt` 指向這輪獨立保存的 CSV／provenance；成績與雜湊見 [紀錄](../doc/GEOMETRY_RESULTS_20261017.md)。前一版 `latest_heading_rts_submission.txt` 與既有 CSV 保持不變。

主機執行 `scripts/benchmark.py --profile sensor_guided` 自動建立唯一 run 並印主機 `Run:` 路徑；查看 `validation.json` 後再使用 `diagnostic.csv`。完整 Track1／Track2 為 3011／3299 資料列，不含表頭。詳細欄位、計時、快取與 chmod 見 [GPU 操作手冊](../doc/MANUAL_WORKFLOW.md)。

benchmark 新 run 尾碼為建立時的台北時間 `YYMMDDHHmm`，例如 `track1_sensor_guided_heading_rts_geometry_directional_cuda_robust_icp_d2d_2610172250`；不再附幀數或隨機字串。同分鐘同名依序加 `_2`、`_3`，既有目錄保留原名。實際幀數查看 `params.json.max_frames` 或 `validation.json.expected_frames`。

現有完整新版候選來源見 [SENSOR_GUIDED_20261017.md](../SENSOR_GUIDED_20261017.md)，本機 `latest_improvement_run.txt` 與 `latest_improved_submission.txt` 是該批驗證／候選的位置紀錄；不代表每次手動 benchmark 都會更新這些指標。

所有實際結果由 Git 忽略，另行備份；不覆寫舊結果，不把忽略的輸出當成能由 Git 還原。map cache 是 outputs 中可選的衍生資料，須保留 Track／原 map SHA256／leaf sidecar。原始資料唯一在 `../../midterm01/data`，唯讀共用，不複製到這裡。

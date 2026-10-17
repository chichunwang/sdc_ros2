# Midterm01 GPU 工作約定

- 遵守 repo 根目錄 AGENTS.md；繁體中文記錄，保留 CPU 原環境及既有修改。2026-10-17 使用者指定 Midterm01 後續工作統一使用本 GPU 目錄；CPU 僅保留參考、既有輸出與共用資料。
- 2026-10-17 使用者指定以學習、自主思考及真實世界應用能力為首要目的；優先分析模型／算法、解釋原理並讓使用者參與推導，非必要不以特定 Track 刷分調參或盲目搜尋為改進主線。必要調參須有物理／統計／標定或數值依據及驗證假說；消融用於理解機制。詳見根 AGENTS 的「學習目的與算法改進原則」及 [開發原則](doc/DEVELOPMENT_PRINCIPLES.md)。不新增已授權工作的例行確認關卡。
- 資料只放在 `../midterm01/data`，以唯讀 bind mount 讀取；不複製資料，不建立本目錄的 data。
- 本目錄的 `workspace/src` 是獨立快照；只在本 workspace 建置，避免改動原環境的 build／install。
- 2026-10-17 使用者要求：本次已啟動的改善驗證完成後，後續完整 Track 測試由使用者手動執行並回報結果；沒有新的明確指示時，不自行啟動下一輪完整序列。
- GPU Compose project／預設 container 為 `sdc_midterm01_gpu`，service 為 `midterm01_gpu`，ROS domain 2；不要重啟 Docker daemon 或啟停／重建原 CPU project。依使用者要求讀取 CPU 結果／修復指定 run 讀取權限時，只限該 run。
- `scripts/dev.sh up` 使用原生 NVIDIA CDI；`--cpu up` 僅供無 GPU 的建置與診斷，不可宣稱 GPU 已啟用。
- 讀取 README.md 與 SETUP_20261017.md 的實測範圍；CUDA kernel 未通過前不宣稱 GPU runtime 成功。
- 日常主流程明確使用 `--profile sensor_guided`，自動選 `cuda_robust_icp`。程式省略 profile 時仍預設 `legacy`／`cuda_ndt`，供歷史重現；`pcl_ndt` 為明確指定的 CPU 比較模式。不得把文件推薦寫成程式預設已改，也不得在 CUDA 錯誤時默默改用 CPU。
- `benchmark.py --profile sensor_guided` 明確選用新版 `cuda_robust_icp`，使用 GNSS／IMU 先驗、CUDA 最近鄰與 CPU 正則化 4-DOF 細修；舊 `legacy` 行為保留。新版證據與限制見 SENSOR_GUIDED_20261017.md。GNSS 不是真值，D2D 完整序列曾發生 CUDA 記憶體錯誤，不將失敗 run 當作完整候選。
- 最新已評分版本為 `--profile sensor_guided --sensor-fusion heading_rts --geometry-mode directional`；省略 fusion 仍是 fixed、geometry 仍是 off。使用者完整測試／提交截圖回報 Track1 1.15275／baseline 1.79075、Track2 0.30576／baseline 1.92587，截圖當下第 3／第 1 名；較 geometry off 的 1.16951／0.31967 降低 1.43%／4.35%，變更與 SHA256 見 `doc/GEOMETRY_RESULTS_20261017.md`。上一輪分數保留於 `doc/HEADING_RTS_RESULTS_20261017.md`；motion_rts 僅短測，不沿用分數。CUDA_NDT_20261017.md 與 ACCURACY_COMPARISON_20261017.md 僅記舊後端；以 benchmark 保存參數、frame、GPU 取樣及分段耗時，不將收斂或 GPU 使用率當精度證據。Track2 舊文字 0.56063 與歷史截圖前最佳 0.56083 的差異保留，不改寫舊提交 provenance。
- 使用 `scripts/dev.sh build-workspace` 執行 Release 建置；輸出用 `outputs/runs/` 中新的唯一目錄。
- benchmark 新 run 保留 Track／演算法前綴，尾碼使用建立當下台北時間 `YYMMDDHHmm`（如 `2610172250`），不再附幀數／隨機字串；同分鐘同名加 `_2`、`_3` 避免覆寫，既有 run 不改名。
- 幾何方向性配準使用 `--geometry-mode directional`；兩條完整已評分，預設 off 保留前一版 point-to-point 權重。diagnostic 只記幾何，不沿用 directional 分數。directional Track1 3011 次接受更新，Track2 3047 次接受／252 次返回先驗。第 410 幀為幾何支持保護，診斷修正見 `doc/GEOMETRY_FALLBACK_20261017.md`；已評分 CSV 來自診斷修改前的原完整 run，不把成績進步歸功於後續 log 修改。新 guidance 記候選拒絕原因，geometry.csv 只描述返回 pose，不反推歷史候選原因。幾何指標排除先驗，不能把 information inverse 當已校準 covariance；設計／手動完整命令見 `doc/GEOMETRY_DIRECTIONAL_20261017.md` 與 `doc/run_geometry_directional.txt`。GPU Release 已建置，CPU 參考環境不動。
- 主機 Toolkit 安裝需 sudo；已提供 `scripts/setup_host_gpu.sh`，不藉其他容器繞過主機 sudo 密碼要求。
- 日常命令／主機與容器路徑／chmod 見 `doc/MANUAL_WORKFLOW.md`。CPU `/root/outputs` 對應 `../midterm01/outputs`，GPU 同名路徑對應本目錄 outputs；不 chmod 主機 `/root` 或原始資料，不用 777。`doc/command_paste.txt` 保留使用者紀錄，常用流程另放 `doc/run_sensor_guided.txt`。

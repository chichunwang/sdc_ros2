# Midterm01 GPU 工作約定

- 遵守 repo 根目錄 AGENTS.md；繁體中文記錄，保留 CPU 原環境及既有修改。
- 資料只放在 `../midterm01/data`，以唯讀 bind mount 讀取；不複製資料，不建立本目錄的 data。
- 本目錄的 `workspace/src` 是獨立快照；只在本 workspace 建置，避免改動原環境正在使用的 build／install。
- 2026-10-17 使用者要求：本次已啟動的改善驗證完成後，後續完整 Track 測試由使用者手動執行並回報結果；沒有新的明確指示時，不自行啟動下一輪完整序列。
- GPU Compose project／service／container 為 `sdc_midterm01_gpu`，ROS domain 2；不要重啟 Docker daemon 或操作原 CPU project。
- `scripts/dev.sh up` 使用原生 NVIDIA CDI；`--cpu up` 僅供無 GPU 的建置與診斷，不可宣稱 GPU 已啟用。
- 讀取 README.md 與 SETUP_20261017.md 的實測範圍；CUDA kernel 未通過前不宣稱 GPU runtime 成功。
- 現在預設 `registration_backend=cuda_ndt`，保留明確指定的 `pcl_ndt` 比較模式；不得在 CUDA 錯誤時默默改用 CPU。初建快照尚未整合的敘述為歷史紀錄。
- `benchmark.py --profile sensor_guided` 明確選用新版 `cuda_robust_icp`，使用 GNSS／IMU 先驗、CUDA 最近鄰與 CPU 正則化 4-DOF 細修；舊 `legacy` 行為保留。新版證據與限制見 SENSOR_GUIDED_20261017.md。GNSS 不是真值，D2D 完整序列曾發生 CUDA 記憶體錯誤，不將失敗 run 當作完整候選。
- 實測範圍以 CUDA_NDT_20261017.md 為準；以 `scripts/benchmark.py` 跑固定前 N 幀並保存參數、frame、GPU 取樣及分段耗時，不將收斂或 GPU 使用率當作精度證據。
- 使用 `scripts/dev.sh build-workspace` 執行 Release 建置；輸出用 `outputs/runs/` 中新的唯一目錄。
- 主機 Toolkit 安裝需 sudo；已提供 `scripts/setup_host_gpu.sh`，不藉其他容器繞過主機 sudo 密碼要求。


# Midterm01 實驗索引

更新：2026-10-17。後續開發／測試統一在 [GPU 工作區](../../midterm01_gpu/doc/27_PROJECT_OVERVIEW.md)，完整測試由使用者手動執行並回報。此處保存跨版本索引，實際 run 依版本分別在 CPU／GPU outputs，不搬移舊結果。

| 版本／實驗 | 已核對範圍 | Track1 Kaggle | Track2 Kaggle | 證據 |
| --- | --- | --- | --- | --- |
| 原 CPU PCL NDT R0 | 兩條單幀／前 50 幀，0 fallback；50 幀 322.519／118.509 秒 | 未取得 | 未取得 | [首跑](../docs/UBUNTU_FIRST_RUN_20261017.md) |
| 使用者 CPU 完整 Track1 | 3011 筆有限矩陣、退出碼 0、log 0 次失敗，14813.444 秒 | 未取得對應成績 | 指定 run 未見完整 Track2 | [CPU 完成紀錄](../docs/CPU_TRACK1_COMPLETED_20261017.md) |
| GPU workspace 內 CPU Release／CUDA D2D | 兩條前 50 幀，CPU Release 與原 R0 逐 byte 相同；D2D 有較大 pose 跳動 | 不由短序列推算 | 不由短序列推算 | [速度](../../midterm01_gpu/doc/03_CUDA_NDT_20261017.md)、[幾何比較](../../midterm01_gpu/doc/05_ACCURACY_COMPARISON_20261017.md) |
| 舊 CUDA D2D 完整 Track1 | 雖 3011 筆完整，路線停滯／姿態翻轉 | 使用者截圖 **573.36983** | 未回報 | [失效分析](../../midterm01_gpu/doc/06_SENSOR_GUIDED_20261017.md) |
| 目前 sensor_guided／cuda_robust_icp | 兩條完整 3011／3299 筆，本機契約與修正限制檢查通過 | 使用者截圖 **1.22369**，同圖 baseline **1.79075** | 未回報 | [新版方法、完整結果與候選 SHA256](../../midterm01_gpu/doc/06_SENSOR_GUIDED_20261017.md) |

自己的 CPU R0 不等於助教 baseline；舊 D2D 分數不能寫成原 CPU 分數。Track1 新版成績來自使用者手動提交／截圖，未另行查榜，不能推及 Track2 或各元件單獨貢獻。

新版完整測試的同步＋ROS 時間為 Track1 230.92 秒（命中 map cache）、Track2 144.34 秒（cache 停用）；不與未快取 CPU／舊 CUDA 總耗時直接計算同條件 GPU 加速倍率。Release、演算法、快取與 I/O 都可能影響耗時。

每個新 run 保存問題／單因素改動、commit／未提交差異、來源 SHA256、參數、frame manifest、TXT／CSV、有限值／列數、退出碼、失敗／先驗回退、軌跡／殘差、分段與完整時間、實際 Kaggle 分數及 CSV SHA256。benchmark 可保存上述執行診斷；外部 Kaggle 分數需由使用者另記。收斂、fitness、GNSS 與地圖貼合都不能替代正式 ATE。

原有下採樣／submap／基本 NDT 不算新增貢獻；新版感測器同步、heading 校正、robust 四自由度細修、CUDA 最近鄰、門檻與平滑的設計／限制見 [架構教學](../../midterm01_gpu/doc/07_ALGORITHM_WALKTHROUGH_20261017.md)。切換 MapAnything 前另核對時間、index、尺度與 frame，不重套外參。

所有資料／輸出遵守 `.gitignore`，Git 不備份最佳 CSV；正式報告與簡報見 [reports/README](../reports/README.md)。

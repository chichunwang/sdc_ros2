# 執行輸出

容器 `/root/outputs/` 對應 Ubuntu `~/sdc_ros2/midterm01/outputs/`。初始化腳本建立：

- `runs/`：每次實驗的 TXT、版本／參數、資料與輸出檢查、診斷。
- `submissions/`：各候選 CSV；最終選定 `result_1.csv`、`result_2.csv`。
- `figures/`：點雲疊圖、軌跡圖、報告圖片。

目前定位程式／launch 的預設結果在 package 內，需執行時明確設定 `save_path`，不會自動使用這裡。參考命令見 [專案 README](../README.md)。程式會覆寫同名 TXT，使用新 run 名稱，例如 `runs/R0_ndt_20261016/track1/`。

map cache 也應寫到此處的獨立目錄，區分 Track、地圖版本與 voxel；目前 launch 尚未完成此調整，首跑停用快取。此目錄的執行結果被 Git 忽略，不能只靠 Git 同步或還原最佳 CSV。

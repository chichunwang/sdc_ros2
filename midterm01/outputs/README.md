# CPU 參考版執行輸出

本目錄保留既有 CPU 結果。後續新開發／測試統一使用 [GPU 工作區](../../midterm01_gpu/doc/27_PROJECT_OVERVIEW.md) 的 `outputs/`，不搬移或覆寫 CPU 證據。

CPU 容器 `sdc_midterm01` 的 `/root/outputs/` 對應主機 `/home/ricky/sdc_ros2/midterm01/outputs/`。GPU 容器同名 `/root/outputs/` 對應 `midterm01_gpu/outputs/`。VS Code 在主機應查看主機專案內的目錄。

- `runs/handoff_20261017/`：兩條 Track 單幀／前 50 幀首跑證據，見 [首跑報告](../docs/UBUNTU_FIRST_RUN_20261017.md)。
- `runs/first_test_9vBOV7le/track1_n0_xLOSykY4/`：使用者 CPU Track1 完整 3011 筆 TXT、log、命令與退出碼；尚未產生 CSV。原 root:root／700 已修正可讀，見 [完成與權限紀錄](../docs/CPU_TRACK1_COMPLETED_20261017.md)。
- `submissions/`／`figures/`：CPU 衍生候選與圖片；不與 GPU 版本成績混用。

若遇到看似空目錄，先核對容器／主機掛載，再檢查目錄的讀取與進入權限；指定 run 的 chmod 與另存 CSV 方法見 [GPU 操作手冊](../../midterm01_gpu/doc/08_MANUAL_WORKFLOW.md#cpu-結果看不到與-chmod)。不 chmod 原始資料或主機 `/root`。

CPU launch 已預設 outputs 下唯一時間路徑，RViz 預設關閉、map cache 空字串停用；舊文所述固定 package 輸出／固定 data cache 已過時。手動重現仍明確指定新 run，避免覆寫結果。結果與快取遵守 Git 忽略規則，最佳 CSV 與證據需另外備份。

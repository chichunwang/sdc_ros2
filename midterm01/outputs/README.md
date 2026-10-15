# 執行輸出

容器內 `/root/outputs/` 對應遠端 Ubuntu 的 `midterm01/outputs/`。初始化腳本會建立：

- `runs/`：每次實驗的 `result.txt`、參數快照、資料檢查與逐幀診斷。
- `submissions/`：轉換後的 Kaggle CSV；交付時選定 `result_1.csv` 與 `result_2.csv`。
- `figures/`：點雲疊圖、軌跡圖及報告用圖片。

執行輸出被 Git 忽略。每次實驗使用獨立名稱，例如 `runs/R0_track01/`；取得模板後，將實際輸出位置設定到這裡。若模板仍使用自身的 `result/` 路徑，先核對再調整，不假設它已自動寫入本目錄。


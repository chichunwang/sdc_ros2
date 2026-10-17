# 報告與簡報

目前方法與重現入口為 [GPU 工作區](../../midterm01_gpu/README.md)，後續測試統一使用 `sensor_guided`／`cuda_robust_icp`。原 CPU／CUDA D2D 為歷史對照；原有 NDT／下採樣／submap 不列為新增貢獻。三階段設計見 [架構教學](../../midterm01_gpu/doc/ALGORITHM_WALKTHROUGH_20261017.md)。

本專案由本人單人完成，貢獻比例 100%。依實際完成情況記錄資料處理、演算法、實驗及文件工作。

報告草稿依序整理：

1. 任務、兩條 Track 與資料特性。
2. 定位流程、座標轉換與初始姿態。
3. 自己發現的問題、設計的改善與解決過程。
4. 對照實驗、Kaggle 成績、耗時與失敗分析。
5. 重現步驟、依賴、最佳設定與個人工作說明。

課堂簡報為 4–6 分鐘。題目公告 Kaggle 及簡報寄送期限為 2026-10-29 09:00，課堂簡報在 10/29，E3 程式與報告截止為 11/05 23:59；正式期限與時區仍以課程及 Kaggle 為準。

E3 應包含 GPU `workspace/src/localization/`、本版本必要的 `workspace/src/cuda_ndt_backend/`、Docker／Compose、sensor prior／benchmark 等自寫腳本、依賴與固定來源／授權、執行 README、最佳 `result_1.csv`／`result_2.csv` 與報告 PDF，排除原始資料與建置產物。單人報告與壓縮檔的學號命名方式需依課程規則確認。


成績引用目前只到使用者截圖回報的新版 Track1 **1.22369**（同圖 baseline **1.79075**）；Track2 尚無回報。兩條完整本機驗證、候選 SHA256 與計時條件見 [新版實測](../../midterm01_gpu/SENSOR_GUIDED_20261017.md)。指定 CPU Track1 完整輸出另見 [CPU 紀錄](../docs/CPU_TRACK1_COMPLETED_20261017.md)，不混用版本成績；尚未完成正式最終報告／簡報／E3 交付。

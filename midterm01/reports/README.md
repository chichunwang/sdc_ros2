**2026-10-23 現行規則：**使用 [00 整合新版](../../midterm01_gpu/doc/Competition%20Regulations/00_SDC_2026_Midterm01_Localization_%E6%95%B4%E5%90%88%E6%96%B0%E7%89%88_20261022.pdf)、[01 因果定位細則](../../midterm01_gpu/doc/Competition%20Regulations/01_Mid01_Updates_Rules.pdf) 與 [48 稽核](../../midterm01_gpu/doc/48_RULES_AND_CODE_AUDIT_20261023.md)。所有 Track 禁止 RTS／backward smoothing、未來資訊回修及定位後離線軌跡修正；歷史結果不代表現行合規。

# 報告與簡報

現行競賽規則先讀 [2026-10-21 補充規則 PDF](../../midterm01_gpu/doc/Competition%20Regulations/91_%E8%88%8A%E7%89%88_%E5%B7%B2%E5%8F%96%E4%BB%A3_%E5%8F%AF%E5%88%AA%E9%99%A4_Mid01_Updates_Adjustments_20261021.pdf)及[45｜規則解讀與專案影響](../../midterm01_gpu/doc/45_COMPETITION_UPDATES_20261021.md)；原始 24 頁規則移至 [000](../../midterm01_gpu/doc/Competition%20Regulations/90_%E8%88%8A%E7%89%88_%E5%B7%B2%E5%8F%96%E4%BB%A3_%E5%8F%AF%E5%88%AA%E9%99%A4_SDC_2026_Midterm01_Localization_24%E9%A0%81.pdf)。補充公告新增 Track3，資料未發布前不猜測其幀數、格式、sample 或 E3 檔名。既有報告草稿中的雙 Track 範圍屬歷史，需按補充規則更新。

目前方法與重現入口為 [GPU 工作區](../../midterm01_gpu/doc/27_PROJECT_OVERVIEW.md)，後續測試統一使用 `sensor_guided`／`cuda_robust_icp`。原 CPU／CUDA D2D 為歷史對照；原有 NDT／下採樣／submap 不列為新增貢獻。三階段設計見 [架構教學](../../midterm01_gpu/doc/07_ALGORITHM_WALKTHROUGH_20261017.md)。

本專案由本人單人完成，貢獻比例 100%。依實際完成情況記錄資料處理、演算法、實驗及文件工作。

報告草稿依序整理：

1. 任務、各 Track 與資料特性（Track3 資料契約公布後補充）。
2. 定位流程、座標轉換與初始姿態。
3. 自己發現的問題、設計的改善與解決過程。
4. 對照實驗、Kaggle 成績、耗時與失敗分析。
5. 重現步驟、依賴、最佳設定與個人工作說明。

課堂簡報為 4–6 分鐘。補充公告沒有更改既有期限：Kaggle 及簡報寄送為 2026-10-29 09:00，課堂簡報在 10/29，E3 程式與報告截止為 11/05 23:59；正式期限與時區仍以課程及 Kaggle 為準。排名基數與 `r=(0.3,0.3,0.4)` 權重依補充公告，且禁止利用榜分反推隱藏真值；詳見[45](../../midterm01_gpu/doc/45_COMPETITION_UPDATES_20261021.md)。

原 Track1／2 的 E3 交付曾記載最佳 `result_1.csv`／`result_2.csv`；補充公告未給 Track3 的 E3 檔名、幀數、格式或 IMU 契約，因此不將 `result_3.csv` 寫成官方要求。其餘交付與命名方式依[45](../../midterm01_gpu/doc/45_COMPETITION_UPDATES_20261021.md)及後續公告確認。


歷史紀錄：2026-10-17 使用者截圖回報新版 Track1 **1.22369**（同圖 baseline **1.79075**）；Track2 當時尚無分數回報。該輪完整本機驗證、候選 SHA256 與計時條件見 [新版實測](../../midterm01_gpu/doc/06_SENSOR_GUIDED_20261017.md)，不代表現行最佳。指定 CPU Track1 完整輸出另見 [CPU 紀錄](../docs/CPU_TRACK1_COMPLETED_20261017.md)，不混用版本成績；尚未完成正式最終報告／簡報／E3 交付。

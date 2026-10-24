# Midterm01 規則與工作約定

## 競賽規則

- 以[整合規則（2026-10-22）](docs/Competition%20Regulations/00_SDC_2026_Midterm01_Localization_%E6%95%B4%E5%90%88%E6%96%B0%E7%89%88_20261022.pdf)與[10/22 更新規則](docs/Competition%20Regulations/01_Mid01_Updates_Rules.pdf)為準；內容不一致時，採較新的明文更新。官方目標是將影像重建點雲配準到預建 LiDAR 地圖，估計每個影格的 pose。可使用其他軟體套件及官方提供的點雲；LoGeR 程式屬 optional，規則也允許直接使用已提供的前視相機 PLY。
- 定位必須遵守 causal／online 原則：估計當前影格只使用當前和過去資料。禁止用未來資料回改舊 pose、backward／non-causal smoothing、定位後平移或旋轉整條軌跡、人工改 pose，及在定位流程外做其他軌跡修正。一般演算法比較、實驗和參數調整允許；不得以 Kaggle Public Leaderboard 分數推測隱藏真值或擬合最終預測。
- Track03 有前、左前、右前三相機影像、前視相機重建 PLY、LiDAR 地圖與加入雜訊的初始 pose，共 **3142 幀**；**不提供 GNSS pose**。Public 分數只評估軌跡前段（規則寫 5–40%），Private 評估其餘部分並用於最終評分；Track03 基本分門檻為 Private Score ≤ **1.40**。分數不是定位精度真值，也不得用來回修軌跡。
- Kaggle 每日最多提交 **15 次**；須於 **2026-10-29 09:00** 前選定合規的最終提交。助教會重跑程式，檢查最終 CSV 是否可由提交的程式、設定與資料流程合理重現；違規或無法重現可能導致該 Track 0 分。E3 繳交期限為 **2026-11-05 23:59**：壓縮檔含程式（不含原始資料）、自寫程式、`result_1.csv`／`result_2.csv`／`result_3.csv`、報告及完整執行 README。報告應交代 pipeline、原創貢獻、問題與解法、工作分配；本組為個人完成，貢獻記 **100%**。另需準備4–6分鐘簡報，10/29課堂報告，當日09:00前寄投影片或連結至老師指定信箱；文件要求不代表助理獲授權寄信或上傳。
- 現有轉檔工具使用 `idx,x,y,yaw` 欄名，並以 `atan2(R10,R00)` 計算弧度 yaw；這是本專案目前的轉檔契約，**不是 PDF 明定的 Kaggle 欄位或單位**。官方規則要求每影格 pose，但未明定評分 frame 對應、欄位、角度單位與慣例；提交前須核對 Kaggle 官方模板及 Track03 實際索引，不能只憑 CSV 可讀或上傳成功推定正確。

## 專案操作

- 依根目錄 [AGENTS.md](../AGENTS.md)，Midterm01 GPU 開發、建置、測試使用 `../midterm01_gpu/`；既有 CPU 程式、原始資料與歷史結果留在 `midterm01/`，不得搬移、覆寫或重算。Track03 原始資料在 `data/Track3/`；未確認來源缺檔前不補抓，既有 metadata 僅讀取、不回寫。
- Docker 指令於 Ubuntu 主機執行，ROS／colcon 指令於指定容器執行。完整 full 定位由使用者親自啟動；短段、研究或環境定位必須有當次明確授權。不得因有限幀數授權延伸執行 full。定位／GPU 工作前查核執行程序，不停止或中途換檔。
- 所有新定位、研究和環境測試輸出一律建立唯一目錄於 `../midterm01_gpu/outputs/run/`，依根 AGENTS 的 `track03_full_YYMMDDHHmm`／`track03_test_YYMMDDHHmm` 等命名與 run 紀錄要求保存；不覆寫歷史結果。根 AGENTS 文件末段的舊 `midterm01/output/run/` 文字與其前文及多次明訂路徑衝突，執行時依本專案詳細路徑規則使用 `midterm01_gpu/outputs/run/`。

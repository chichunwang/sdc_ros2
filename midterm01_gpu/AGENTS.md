# GPU 工作區規則

遵守根目錄 AGENTS.md 及 midterm01/AGENTS.md 的競賽規則。
- 完整 Track 定位必須由使用者手動啟動；未獲明確幀數授權不得執行真實短段定位。
- 修改前檢查執行中 full，保護其來源及 SHA。不得並行爭用 GPU。
- 不改 `../midterm01/data`、CPU 程式或歷史結果。資料在容器掛載為唯讀。
- 所有環境、合成及定位測試放 `outputs/run/` 唯一目錄，保存來源 SHA、版本、差異、設定、耗時及失敗。
- GPU 計算不得靜默降級到 CPU。小矩陣濾波、I/O 可在 CPU，點雲對應及配準累加在 CUDA 執行。
- 僅因果定位，無 GNSS、無回頭修正 pose、無榜分擬合。尺度及座標假設須明確記錄。
- 成功 full 自動生成 run/analysis 內 PNG/PDF 與來源 manifest；圖表不改定位結果。
- 使用 scripts/dev.sh 管理自己的 Compose project `midterm01_track3`，不改 Docker daemon 或 CPU 容器。
- 所有委派禁止再委派；不同 agent 不可同改一檔。算法、frame/時間/尺度與最終驗證由主 agent 負責。

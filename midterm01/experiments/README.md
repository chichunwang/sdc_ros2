# 實驗紀錄

R0 使用目前程式已有的前相機 **基礎 NDT**，包含清理、下採樣、局部地圖與失敗保留前一 pose。先保存完整參考結果，再以同資料／片段逐項比較改善。R0 不是助教 baseline，也不是已完成的實驗。

| 實驗 | 版本／commit | 點雲來源 | 主要改動 | Track01 分數 | Track02 分數 | 耗時 | 結果位置 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | 待記錄 | raw_ply | 目前基礎 NDT 參考版 | 待測 | 待測 | 待測 | 待產生 |

每次記錄：

- 要驗證的問題、固定資料片段／frame manifest、資料來源與版本。
- commit 及未提交差異、完整命令與參數；包含尺度、voxel、submap、NDT、cache、bag 是否啟用。
- 輸出原始 TXT／CSV、行數／ID 檢查、完整耗時與環境。
- 成功／失敗 frame、點數、收斂／fitness、姿態變化、fallback 與疊圖；目前節點尚未輸出完整逐幀診斷表，依需求補上。
- 兩條 Track 的實際 Kaggle 分數、官方 baseline 數字及比較結論。

結果放 `outputs/runs/<run>/track1`、`track2`，提交候選放 `outputs/submissions/<run>/`，避免覆寫。結果被 Git 忽略，需另外保存與交接；本目錄追蹤實驗說明與必要設定。

未實跑不填估計分數，不把收斂／fitness 或 GNSS 當成正式 ATE。切換 MapAnything 時先核對 index、時間、尺度及座標；不要同時換全部參數後宣稱單一因素有效。

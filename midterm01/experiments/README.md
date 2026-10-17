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

## 2026-10-17 首跑實測

兩條 Track 均已完成既有前相機 NDT 單幀與前 50 幀。輸出列數及有限值通過，fallback 都是 0；單幀耗時為 97.6／14.9 秒，50 幀為 322.5／118.5 秒（Track1／Track2，含讀取地圖）。保持 ply_scale=3.0 與原始 NDT 預設，未更動演算法。Track1 有 0.796 m 相鄰步長、Track2 短序列 yaw 變化約 0.507 rad，精度與穩定性仍待查。完整參數、frame manifest、疊圖及限制見 [實測報告](../docs/UBUNTU_FIRST_RUN_20261017.md)，完整兩條 Track R0 與 Kaggle 評分尚未執行。

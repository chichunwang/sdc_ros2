# GPU 輸出

所有完整與短段測試都在 `run/`：

```text
run/track01_full_2610192232/
run/track02_full_2610192240/
run/track01_test_YYMMDDHHmm/
run/track00_test_YYMMDDHHmm/
```

`full` 是完整序列嘗試，`test` 是短段／研究；Track00 是跨 Track／環境測試。目錄內 `test_info.json` 保存時間、測試種類、版本及已回報分數；`params.json` 是實際參數。`diagnostic.csv` 為提交檔、`result.txt` 為回放矩陣、`validation.json` 為完整性檢查。資料夾名不含算法版本，不另建最近／最佳分類或索引。

時間為台北 `YYMMDDHHmm`，同分鐘碰撞加序號。`ml_runtime/` 為模型與Python環境，不是測試結果。舊結果已搬入run，原始證據內容不改；詳見 [操作手冊](../doc/08_MANUAL_WORKFLOW.md)。

# 09｜GPU 工作區輸出

所有測試統一放在 `midterm01_gpu/outputs/run/`。容器對應 `/root/outputs/run/`；CPU環境與原資料不搬移。

```text
outputs/run/
  track01_full_2610192232/
  track02_full_2610192240/
  track01_test_YYMMDDHHmm/
  track02_test_YYMMDDHHmm/
  track00_test_YYMMDDHHmm/
```

`full` 表示要求完整Track，成功與否仍看validation；`test` 包含短段、消融與研究。Track00為跨Track或環境測試。台北時間格式`YYMMDDHHmm`，重名加`_0002`等碰撞序號。版本不放目錄名，不建立其他分類或測試索引。

- `test_info.json`：測試種類、時間、版本／來源，已評分來源可有reported_score。
- `params.json`／`pipeline_source_sha256.json`／`git_commit.txt`：參數、來源版本與未提交差異。
- `diagnostic.csv`：提交CSV；完整Track1／2應有3011／3299資料列。
- `result.txt`：每幀3×4矩陣，回放使用。
- `validation.json`：完整性檢查、退出碼及錯誤；通過不代表精度提升。

benchmark會印出本次Run實際路徑。回放省略`--run`時選該Track最近通過完整性檢查的full目錄，也可指定任何一次已保存run。

既有結果已集中搬移，原始CSV、矩陣、manifest及SHA不改。舊路徑記在各次test_info.json，工具僅在讀取時解析；原始證據檔不重寫。

`outputs/ml_runtime/`是模型與隔離Python環境，維持原位置。大型輸出由Git忽略，需另行備份。日常命令及欄位說明見[08](08_MANUAL_WORKFLOW.md)。

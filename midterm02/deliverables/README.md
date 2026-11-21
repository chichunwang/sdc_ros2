# 交付草稿

| 交付 | 期限 | 執行者 |
| --- | --- | --- |
| 簡報（4–6 分鐘）email 至課程信箱 | 11/26 09:00 | 使用者 |
| Kaggle 最終 2 份提交選定 | 11/26 09:00（遲交至 12/03 23:59） | 使用者 |
| E3 `<ID1>_<ID2>_vslam.zip` | 12/03 23:59 | 使用者 |

zip 內容：整份程式（不含 data）、`result_1.csv`、`result_2.csv`、`vslam_report_<ID1>_<ID2>.pdf`、重現用 README。`result_*.csv` 必須取自 `outputs/run/` 中可追溯的 run，並記錄其 run ID 與 SHA256。

## 目前的最終提交候選（2026-11-21）

最終兩份提交由使用者在 Kaggle 選定（每個 Track 各兩份，只看 Private）。

| Track | run（`outputs/run/`） | 版本 | Public | `result_*.csv` SHA256（前 16 碼） |
| --- | --- | --- | --- | --- |
| 1 | `track01_full_2611161954` | D2 | **0.068** | `450f0d338c2d3093` |
| 1 | `track01_full_2611161129` | gslam 0.3.0 基本 BA | 0.085 | `c582160f6ad2ffee` |
| 2 | `track02_full_2611172344` | F1（droid_ba-1） | **0.254** | `a1b76d30c28ce3d8` |
| 2 | `track02_full_2611191713` | F1 反向邊 σ ×0.03（droid_ba-5） | 0.255 | `98caadad075addc3` |
| 2 | `track02_full_2611200139` | F1＋`--weak-odom`（droid_ba-7） | 未提交 | `df35ea845c6d7118` |

各候選的優缺點見 [doc/03](../doc/03_GSLAM_DESIGN_AND_STATUS_20261116.md) 文末「你接下來要做的事」；重現命令見 [doc/05](../doc/05_MANUAL_RUNS_20261118.md)。
交付前要做：commit 目前的程式（E3 的程式要能重現選定的 run）；報告附上 DROID 在 GPU 上的浮點變異範圍（doc/03「可重現性驗證」）。

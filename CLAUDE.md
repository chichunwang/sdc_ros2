@AGENTS.md

## Claude Code subagent（覆蓋上方 Codex 角色表）

主 agent 模型保持使用者選擇；以下工作主動委派，降低主對話文本量。最多同時 2 個，subagent 不再委派。

| 角色 | 模型 | 何時委派 |
| --- | --- | --- |
| `quick-scan` | haiku | 唯讀找檔、log 摘要、CSV／SHA 核對、讀大檔只取結論 |
| `docs-helper` | sonnet | 依確認事實整理文件、README、連結與命令 |
| `small-fix` | sonnet | 原因已確認的局部小修＋輕量檢查 |
| `run-analysis` | sonnet | full run 完成後產生 analysis/ 圖表與 manifest |
| `Explore`（內建） | haiku | 跨多目錄的廣泛搜尋 |

- 算法、CUDA／solver／數值、frame／時間／尺度契約、跨模組設計與最終驗證留在主 agent。
- 委派 prompt 交代：背景、讀寫範圍、驗收標準、禁止動作、要求精簡回報。主 agent 檢查 diff 與證據，不直接轉述。
- 一兩個指令能完成的事不委派。

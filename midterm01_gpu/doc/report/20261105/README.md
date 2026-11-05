# 王祈淳 112611054｜Midterm01 報告與結果

報告分為 Track1／2 的 GPU 因果定位方法，以及 Track3 的指定作者程式流程。PDF 與 Markdown 內容相同；`assets/` 保存報告使用的五張圖。

## 檔案

| 檔案 | 用途 |
| --- | --- |
| `localization_report_112611054.pdf` | 可閱讀的最終報告 |
| `localization_report_112611054.md` | 可直接修改的報告原稿 |
| `result_1.csv`、`result_2.csv`、`result_3.csv` | 使用者指定最終 CSV 的原樣副本 |
| `manifest.json` | 原始檔／交付檔 SHA256、筆數、分數來源與限制 |
| `assets/` | 軌跡、Track2 尺度／耗時、分數比較與兩張流程圖 |

三份 CSV 的原始來源皆為 `midterm01_gpu/doc/analysis doc/trackN.csv`，只複製與更名，沒有改數值或換行。

## 本次確認的成績與更正

| Track | Public Score | Private Score | 來源 |
| --- | ---: | ---: | --- |
| 1 | 0.56188 | 未提供 | 使用者本次確認 |
| 2 | 0.42660 | 未提供 | 使用者本次確認、既有分數收據 |
| 3 | 0.57380 | 0.86831 | 使用者本次確認 |

Track2 先前上傳的 `0.30488` 使用全段平滑。依使用者說明，該上傳紀錄不能代表因果定位成果；本報告採用 v5.16.4 的 `0.42660`。Track1 沒有這項平滑更正。

## 環境與資料

Track1／2 使用 Ubuntu 主機、NVIDIA driver、NVIDIA Container Toolkit 與專案 CUDA／ROS 2 Humble 容器。容器建置方式見 `midterm01_gpu/README.md`；主機使用 `scripts/dev.sh` 管理本專案容器。原始資料保留在 `midterm01/data/Track1`、`Track2`，不可收入 E3 壓縮檔。

Track3 指定程式位於 `midterm01/others code track3`。作者方法是 Python CPU 流程，來源內有 `requirements.txt`；包裝器使用獨立執行副本並處理 Polars I/O 相容性。資料需包含 `raw_ply/`、ASCII `map.pcd`、`init_guess.txt`、`Track3.db3`。

## 現有入口

完整定位一律由使用者手動啟動；本次製作報告沒有執行定位。

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
./scripts/dev.sh up
./scripts/run_track1.sh --full
```

等 Track1 結束後，才執行 Track2：

```bash
./scripts/run_track2.sh --full
```

Track3 包裝入口：

```bash
./scripts/run_external_track3.sh --check
./scripts/run_external_track3.sh --full
```

`--check` 是環境、合成與前置資料檢查。`--full` 的每次執行建立新的 `midterm01_gpu/outputs/run/trackNN_full_YYMMDDHHmm/`；三條 Track 共用 lock，不同時占用 GPU。

## 精確重現的已知界線

- Track2 `result_2.csv` 與 `outputs/run/track02_full_2611050123/result_2.csv` bytes 相同，對應 `track2_v510_turn_patch_v5_16_4_perf2`。應保留該 run 的 `source/`、`source_manifest.json`、`parameters.json`、輸入與 frame manifest。live 目前回復 v5.16.4 數值分支，但不以 live 版本號替代凍結來源證據。
- Track1 最終 CSV 與 `doc/analysis doc/result_1_0.56188.csv` 相同。使用者指定 Track1 採 v5.16.4 作報告方案；現有獨立 `run_track1.sh` 仍是 v4，未找到最終 CSV 到 v5.16.4 的來源對照。因此不能宣稱這條命令已能重現所附 Track1 結果；這項限制不表示判定 Track1 使用全段平滑。
- Track3 最終 CSV 為使用者提供，成績也是使用者回報。指定作者流程的 `gyro-dt=0.14`、全段搜尋首個 good 初始化與獨立 yaw 輸出仍待因果 prefix 驗證；本機沒有該 CSV 的完整 bytes 重現收據。詳見報告第 9 章及 `doc/70_EXTERNAL_TRACK3_LOCAL_VALIDATION.md`。

上述限制原樣揭露，沒有為報告修改歷史 CSV、程式、分數或原始資料。

## E3 整理

官方要求壓縮檔包含完整 `localization/` 程式（不含原始 dataset）、自寫程式、三份結果、報告與執行 README。本目錄只有報告與結果，尚不是完整的程式繳交壓縮檔。

整理完整程式時請保留 `midterm01/src/localization`、`midterm01_gpu` 所需 source／config／環境檔與指定 Track3 作者原碼；另附已選版本來源快照與必要 manifest。排除原始資料、bag、`.env`、build／install／log 與大型 cache，不要只打包本目錄便視為完整程式交付。

單人檔名採 `localization_report_112611054.pdf`；壓縮檔可命名 `112611054_localization.zip`。實際 Kaggle 提交、E3 上傳與寄信由使用者操作。

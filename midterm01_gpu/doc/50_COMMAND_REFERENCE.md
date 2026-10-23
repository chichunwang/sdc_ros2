# 指令表

更新日期：2026-10-23。以下命令在 Ubuntu 主機執行；一次只執行一個定位程序，上一輪結束後再選擇下一輪命令。

本文整理 Track3 本機操作入口。現行 v2 的初始化修正與合成驗證見 [53](53_TRACK3_INITIALIZATION_REPAIR_20261023.md)，模型／尺度與 v1 full 失敗證據見 [52](52_TRACK3_CAUSAL_KALMAN_SCALE_20261023.md)；既有 legacy baseline 與規則限制見 [49](49_TRACK3_CAUSAL_BASELINE_20261023.md)、[48](48_RULES_AND_CODE_AUDIT_20261023.md)，通用輸出格式見 [08](08_MANUAL_WORKFLOW.md)。所有定位由使用者手動啟動；目前沒有真實 v2 定位或精度改善證據。

## 常用命令

| 用途 | 命令／行為 | 是否定位 | 說明 |
| --- | --- | --- | --- |
| Track3 因果 v2 full | 見下方三步命令 | 是 | 正式分數需 full 提交；由使用者手動執行。score <1.29 是目標，不保證達成。 |
| Track3 因果 v2 前 100 幀 | 可選；將 `--full` 換成 `--frames 100` | 是 | 機制／故障檢查，不是 full 資格門檻，也不能替代正式評分。 |
| Track3 舊 legacy 前 50 幀／full | 見「歷史 legacy baseline」 | 是 | 保留重現舊 run 用；固定 scale 3 未校準，不作現行推薦。 |
| GPU 容器環境狀態 | `bash scripts/dev.sh status` | 否 | 查看開發容器狀態。 |
| GPU 環境診斷 | `bash scripts/dev.sh doctor` | 否 | 執行環境診斷。 |
| Track3 因果分支 dry-run | `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 outputs/ml_runtime/venv/bin/python scripts/run_track3_causal.py --frames 100 --scale-mode estimate --initial-scale 3 --dry-run` | 否 | 列印設定計畫；不定位、不建立新 run。 |
| Track3 因果分支 audit-only | 將上述命令的 `--dry-run` 換成 `--audit-only` | 否 | CPU 輸入／座標檢查，不產 pose、不呼叫 GPU；會建立合規 test run 保存 audit。 |
| Track3 資料盤點 | `python3 scripts/check_competition_rules.py --track 3 --inventory-only` | 否 | exit 0 僅表示資料盤點通過，不是定位支援或提交合規認證。 |
| 新 full run 圖表重試 | 見下方指令 | 否 | 僅讀取指定的新 full run；不重跑定位。請將範例路徑替換成實際新 run 的絕對路徑。 |

環境命令從專案目錄執行：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
bash scripts/dev.sh status
```

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
bash scripts/dev.sh doctor
```

dry-run 與資料盤點均在主機唯讀檢查，不需啟動容器：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
python3 scripts/benchmark.py --track 3 --frames 50 --profile legacy --backend cuda_ndt --cuda-mode D2D --ply-scale 3 --dry-run
```

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
python3 scripts/check_competition_rules.py --track 3 --inventory-only
```

## Track3 因果分支：目前 v2 full 命令

現行入口執行的是 v2。雖然 CLI 與 v1 相同，程式來源已改，不能用現行命令重現 v1；v1 source snapshot 保存在舊 run 的 `pipeline_sources/`。完整模型與假設見 [52](52_TRACK3_CAUSAL_KALMAN_SCALE_20261023.md)，初始化修正和 59 項合成／機制測試見 [53](53_TRACK3_INITIALIZATION_REPAIR_20261023.md)。這些檢查尚未證明真實定位有效。只有 full 提交會產生正式分數；短段是可選診斷，不設額外資格門檻。`--initial-scale 3` 是未校準 seed，score <1.29 只是一個目標。完整定位與提交由使用者手動執行：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
bash scripts/dev.sh up
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 outputs/ml_runtime/venv/bin/python scripts/run_track3_causal.py --full --scale-mode estimate --initial-scale 3
```

命令使用 Python runner，不需重編 C++。可選短段時將 `--full` 改為 `--frames 100`；固定尺度消融則另行手動把 `--scale-mode estimate` 改成 `--scale-mode fixed`。v1 使用者回報 82.17，本機輸出 XYZ 保持初始位置且 map 更新採用 0 幀；上傳 bytes 未核對。此已知失敗版本不應原樣重跑，舊 source snapshot 供檢視。舊 legacy 的 106.72 是另一個歷史結果，也未核對上傳 bytes。

CPU-only dry-run：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 outputs/ml_runtime/venv/bin/python scripts/run_track3_causal.py --frames 100 --scale-mode estimate --initial-scale 3 --dry-run
```

CPU audit-only（輸入／座標檢查，不產生定位 pose，也不呼叫 GPU）：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 outputs/ml_runtime/venv/bin/python scripts/run_track3_causal.py --frames 100 --scale-mode estimate --initial-scale 3 --audit-only
```

## 歷史 legacy baseline 命令

以下命令只保留用於歷史 legacy baseline 重現，並非新 Track3 方法推薦。短段輸出目錄使用 `outputs/run/track03_test_YYMMDDHHmm`；full 使用 `outputs/run/track03_full_YYMMDDHHmm`。若同分鐘目錄名稱碰撞，使用下一個序號（例如 `_0002`）。程式會列印實際 `Run: <絕對路徑>`，後續檢查應使用該路徑。

Track3 legacy 50 幀短段：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
bash scripts/dev.sh up
python3 scripts/benchmark.py --track 3 --frames 50 --profile legacy --backend cuda_ndt --cuda-mode D2D --ply-scale 3
```

Track3 legacy 3142 幀 full（歷史命令）：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
bash scripts/dev.sh up
python3 scripts/benchmark.py --track 3 --full --profile legacy --backend cuda_ndt --cuda-mode D2D --ply-scale 3
```

`--frames 50` 與 `--full` 互斥。歷史 legacy 命令只執行 Track3，不會連跑 Track1、Track2。尺度 3 尚未校準；該入口是 `legacy`／`front` map-only，不讀 IMU／GNSS，也不使用 RTS／graph。這些是路徑審查，不構成官方認證。官方 sample、idx／yaw 語意與本機完整 frame 契約仍待核對。不要把舊 legacy score 當作新因果分支結果。此流程不需重編 C++。

## 定位輸出與檢查

每次定位以主控台 `Run:` 所列的新目錄為準。主要檔案用途如下：

| 檔案 | 內容與檢查方式 |
| --- | --- |
| `result.txt` | 每列 12 個矩陣值；供本機 pose 回放與檢查。 |
| `diagnostic.csv` | 欄位 `idx,x,y,yaw`；full 應有 3142 筆資料列（不含表頭）。公告欄序相同，但 idx／yaw 的正式語意契約仍待確認。 |
| `frames.csv` | 本輪逐幀輸出與來源 frame 記錄。 |
| `timings.csv` | 逐幀計時與定位診斷資料。 |
| `validation.json` | 至少檢查 `returncode == 0`、`output_complete_and_finite == true`、`validation_errors` 為空；full 另檢查 `submission_frame_set_complete == true` 且 frame 數為 3142。 |

上述條件只代表輸出完整性與有限值檢查通過。配準收斂、接受率、殘差或矩陣有限都不代表精度、真值一致或提交規格已獲認證。

## Full run 事後圖表

自 2026-10-23 起，成功完成的 full run 圖表放在該 run 的 `analysis/` 子目錄，包含 `01_trajectory.png`／`.pdf`、`02_pose_changes.png`／`.pdf`、`03_registration_timing.png`／`.pdf`、`manifest.json` 與 `README.md`。圖表失敗狀態另記在 run 根目錄的 `analysis_status.json` 與 `analysis.log`，不改寫定位成功狀態。短段、失敗或未完成 run 預設不產生完整比較圖；歷史 run 不搬移、不補圖。圖表是唯讀診斷，不證明精度或真值，也不回修軌跡。

若某個新 full run 的圖表需要單獨重試，先把下列範例路徑替換成該次 run 的實際絕對路徑，再執行：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 outputs/ml_runtime/venv/bin/python scripts/plot_run_analysis.py --run "/absolute/path/to/new_full_run"
```

此命令只讀指定 run；`/absolute/path/to/new_full_run` 是佔位文字，必須替換，不要原樣執行。重試僅適用新 full run，不用來補歷史圖或短段圖。

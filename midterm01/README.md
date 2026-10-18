# Midterm01：CPU 參考版與共用資料

2026-10-17 起，使用者指定 **Midterm01 後續開發、建置與測試統一使用 [midterm01_gpu](../midterm01_gpu/doc/27_PROJECT_OVERVIEW.md)**。本目錄保留原 CPU NDT 程式、既有輸出、唯一原始資料、官方題目與交付說明；GPU 以唯讀掛載共用 data／config。

日常操作請看 [GPU 操作手冊](../midterm01_gpu/doc/08_MANUAL_WORKFLOW.md)，新版命令必須帶 `--profile sensor_guided`。原 [CPU 手冊](docs/MANUAL_WORKFLOW.md) 供歷史重現，不是目前主要入口。

## CPU 已完成的結果

| 測試 | Track1 | Track2 | 證據 |
| --- | --- | --- | --- |
| 單幀／前 50 幀 | 已完成、有限值與筆數通過、0 fallback | 已完成、有限值與筆數通過、0 fallback | [Ubuntu 首跑](docs/UBUNTU_FIRST_RUN_20261017.md) |
| 使用者完整 CPU run | **3011 筆**、退出碼 0、log 0 次失敗、14813.444 秒 | 本次指定 run 未見 Track2 完整輸出 | [CPU Track1 完成紀錄](docs/CPU_TRACK1_COMPLETED_20261017.md) |
| CPU Kaggle 精度 | 未取得對應成績 | 未取得對應成績 | 完整輸出／收斂不能當精度證據 |

指定 CPU run 主機位置：

```text
/home/ricky/sdc_ros2/midterm01/outputs/runs/first_test_9vBOV7le/track1_n0_xLOSykY4
```

該資料夾原為 root:root／700，主機 ricky 無法列出內容；已修正為 755，TXT 內容 SHA256 保持一致。`result.txt`、`run.log`、`command.sh`、`exit_code.txt` 已存在，這個完整 run 尚未產生 CSV。chmod 與另存 CSV 方法見 [GPU 手冊](../midterm01_gpu/doc/08_MANUAL_WORKFLOW.md#cpu-結果看不到與-chmod)。

GPU 新版兩條完整序列已完成本機驗證，使用者曾回報 Track1 Kaggle 1.22369／baseline 1.79075；Track2 尚無回報。這不是 CPU 版本的分數，詳見 [新版實測](../midterm01_gpu/doc/06_SENSOR_GUIDED_20261017.md)。

## 保留目錄與掛載

| 位置 | 用途 |
| --- | --- |
| `data/` | 唯一 Track1／Track2、LoGeR／MapAnything、bag／影像／校正資料，GPU 唯讀共用 |
| `config/tracks.json` | 基本檔案／數量檢查；不會自動改定位參數 |
| `workspace/src/localization/` | 原 CPU PCL NDT 參考程式；不與 GPU workspace 自動同步 |
| `outputs/` | CPU 歷史結果，Git 忽略；避免覆寫 |
| `docs/` | 官方 PDF、Kaggle 規格、首跑與 CPU 完成紀錄 |
| `experiments/`／`reports/` | 實驗索引與正式報告／簡報安排 |

CPU Compose project／預設容器為 `sdc_midterm01`、service `midterm01`、ROS domain 1。本機已核對掛載：

| 主機來源（相對本目錄） | CPU 容器位置 | 權限 |
| --- | --- | --- |
| `./workspace` | `/root/midterm` | 讀寫 |
| `.env` 的 `MIDTERM_DATA_DIR`，預設 `./data` | `/root/data` | 唯讀 |
| `./config`、`./scripts` | `/root/config`、`/root/scripts` | 唯讀 |
| `./outputs` | `/root/outputs` | 讀寫 |

GPU 容器同名 `/root/outputs` 是主機 `midterm01_gpu/outputs`，不包含 CPU run。主機 VS Code 應開專案目錄，依 `midterm01/outputs` 或 `midterm01_gpu/outputs` 分別查看，不用主機 `/root/outputs`。

## 文件與資料契約

閱讀 [官方要求](docs/README.md)、[Kaggle 欄位與筆數](docs/KAGGLE_DATASET_SPEC.md)、[資料檢查](data/README.md)、[下載索引](docs/DOWNLOADS.md)。資料已建立；確認缺檔才補下載，不重設已可用 token 或搬動目錄。

原 CPU 流程為 PLY 時間排序 → 乘尺度 3.0／濾波 → 依上一 pose 裁切 submap → PCL NDT → 接受或保留前一 pose → ROS／TXT。`init_guess.txt` 只初始化第一幀，source→map 最終矩陣不再乘初始矩陣；bag 可讀但 CPU NDT 沒有融合。`min_input_points` 原 CPU 已宣告但尚未用於點數門檻。

目前 CPU launch 已有 `use_rviz=false`、`map_cache_file=''` 與 outputs 下唯一預設路徑；批次需 `keep_alive=false`。TXT 每行 12 個數值，轉檔只需 Python 標準函式庫。舊 Windows 文件中的無 RViz 開關／固定 cache／pandas 依賴已過時。

正式 CSV 為 `idx,x,y,yaw`、資料列 3011／3299；現有轉檔 idx 從 0 起、yaw 弧度，官方完整 frame／ID／單位仍待確認。PCL fitness、GNSS 與地圖殘差都不是官方 ATE，不把原有 NDT／submap／下採樣寫成新增貢獻。

## 新機缺少 GPU 所需基底映像時

GPU Dockerfile 仍延伸 CPU 的 `sdc_midterm01_ros2:latest`。只在新機缺少此基底時建置映像：

```bash
cd /home/ricky/sdc_ros2/midterm01
bash scripts/dev.sh build
```

此入口缺少 HW2 基底時會以 `hw2/Dockerfile` 與 entrypoint 建置。完成後回 GPU 目錄建置／啟動；日常不需啟動 CPU project。CPU 歷史執行命令保留在 [CPU 手冊](docs/MANUAL_WORKFLOW.md)。

原始資料、bag、`.env`、build／install／log、outputs 仍遵守 `.gitignore`；Git 不會備份最佳 CSV。後續完整測試與 Kaggle／E3 提交由使用者執行，期限與交付見 [官方要求](docs/README.md)。

# Midterm02 Visual SLAM

用前鏡頭影像與官方 LoGeR 點雲，估計 Track01／Track02 每張影像的 camera-to-world pose，提交 Kaggle（Sim(3) 對齊後 ATE，越低越好）。規則重點見 [AGENTS.md](AGENTS.md)，官方規則完整整理見 [doc/01](doc/01_COMPETITION_RULES_20261113.md)。

## 目前狀態（2026-11-13）

- 已建立目錄架構、run 建立工具與提交格式檢查工具（含單元測試）。
- **尚未下載資料與 starter code，尚未實作任何 SLAM 算法，沒有任何分數。**
- Track02 官方尚未公布。

## 下一步（使用者）

1. 從 Kaggle Track01 的 Data 頁下載資料包，整包原樣解壓到 `midterm02/data/`，應得到 `midterm02/data/data_VSLAM-T1/Track1/data/` 與 `midterm02/data/data_VSLAM-T1/image_slam_starter/`。
2. 告訴助理，助理會把 starter 複製到 `vslam/`、記錄來源 SHA，並依 starter README 建立環境。

`data/`、`outputs/` 皆被 `.gitignore` 忽略，不進 Git、不會跨電腦同步。資料只供本課程使用，不得散布（見 [doc/02](doc/02_KAGGLE_RULES_TRACK01_20261113.md)）。

## 目錄

| 位置 | 用途 |
| --- | --- |
| `vslam/` | 我們的 SLAM 程式（從 starter 複製後修改）；參數集中在 Python 檔 |
| `Dockerfile`、`docker-compose.yml`、`scripts/dev.sh` | Ubuntu 專用容器 `midterm02_vslam` |
| `scripts/create_run.py` | 建立唯一 run 目錄並凍結程式快照、SHA、Git 差異 |
| `scripts/check_submission.py` | 檢查提交 CSV 格式與 timestamp 是否與 sample 完全一致 |
| `tests/` | 工具與算法的單元測試 |
| `doc/` | 規則、算法說明、實驗紀錄（編號＋日期命名） |
| `deliverables/` | 報告、簡報、E3 交付草稿 |
| `data/`（忽略） | 官方資料包原樣解壓（含 starter 原檔），唯讀 |
| `outputs/run/`（忽略） | 所有 run：`track01_full_YYMMDDHHmm` 等 |

## 環境（Ubuntu＋Docker）

執行一律在 Ubuntu 主機的專用容器 `midterm02_vslam`（Python 3.9，CPU）內進行；Windows 只用於編輯。`requirements.txt` 目前未固定版本，取得 starter README 後要改成固定版本。

```bash
cd ~/sdc_ros2/midterm02          # 依你 Ubuntu 上的 repo 位置調整，需在 midterm02/main 分支
./scripts/dev.sh build
./scripts/dev.sh up
./scripts/dev.sh test             # 單元測試
./scripts/dev.sh shell            # 進入容器
```

## 常用命令（容器內）

```bash
python scripts/create_run.py --kind test --track 01
python scripts/check_submission.py outputs/run/<run>/result_1.csv data/data_VSLAM-T1/Track1/data/sample_submission.csv
python -m unittest discover -s tests
```

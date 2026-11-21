# Midterm02 Visual SLAM

用前鏡頭影像與官方 LoGeR 點雲，估計 Track01／Track02 每張影像的 camera-to-world pose，提交 Kaggle（Sim(3) 對齊後 ATE，越低越好）。規則重點見 [AGENTS.md](AGENTS.md)，官方規則完整整理見 [doc/01](doc/01_COMPETITION_RULES_20261113.md)。

## 目前狀態（2026-11-21）

| Track | 目前最佳（Public） | 檔案 | baseline |
| --- | --- | --- | --- |
| Track1（1314 張，有 IMU） | **D2** 0.068：DROID-SLAM＋我們估出的畸變校正 | `outputs/run/track01_full_2611161954/result_1.csv` | 0.955 |
| Track2（4206 張、約 2.1 km，無 IMU） | **F1** 0.254：DROID＋閉環＋反向重訪約束＋全解析度 SIFT BA | `outputs/run/track02_full_2611172344/result_2.csv` | 0.658 |

- **分數演進**：Track1 0.369 → 0.144 → 0.085（自寫 gslam＋BA）→ 0.068（D2）；Track2 3.318（D2）→ 2.481（D6）→ 1.466（D8）→ 0.457（P1）→ 0.254（F1）。
- **11/18 之後的 Track2 實驗**（F2、MASt3R、P2、學習式特徵、D10、反向邊 σ 交叉驗證、`--weak-odom`、低視差深度先驗）都沒有比 F1 明顯更好。
  其中 σ ×0.03（0.255，交叉驗證較好）與 `--weak-odom`（修掉物理上不可能的位姿，未提交）可當最終提交候選。
- **最終提交候選、SHA256 與所有未解決的問題**：見 [doc/03](doc/03_GSLAM_DESIGN_AND_STATUS_20261116.md) 文末「目前遇到的問題（2026-11-21）」與「你接下來要做的事」。

## 文件導覽

| 文件 | 內容 |
| --- | --- |
| [doc/01](doc/01_COMPETITION_RULES_20261113.md)、[doc/02](doc/02_KAGGLE_RULES_TRACK01_20261113.md) | 官方規則、Kaggle 規則與資料原文 |
| [doc/03](doc/03_GSLAM_DESIGN_AND_STATUS_20261116.md) | gslam 設計、每個版本的做法與實驗數字（依日期）、目前問題、最終候選 |
| [doc/04](doc/04_HANDOFF_PROMPT_20261116.md) | 11/16 的交接說明（歷史；現況以 doc/03 為準） |
| [doc/05](doc/05_MANUAL_RUNS_20261118.md) | 手動跑各版本的命令（E、D、P1、F1 與變體）、RViz 重播、驗證腳本 |
| [doc/06](doc/06_TRACK2_LEARNING_JOURNEY_20261118.md) | Track2 從 3.318 到 0.254 的推理過程，以及之後失敗的嘗試 |
| [deliverables/README.md](deliverables/README.md) | 交付期限與內容 |

## 下一步（使用者）

1. commit 目前的程式與文件（`droid_ba.py` droid_ba-5～8、`bundle.py`、`opposite_pgo.py`、`rviz_replay.py` 等尚未 commit）。
2. 11/26 09:00 前在 Kaggle 選定兩份最終提交、寄出簡報；12/03 前上傳 E3（見 [deliverables/README.md](deliverables/README.md)）。

## 開發歷程（2026-11-15～16）

- 已建立目錄架構、run 建立工具與提交格式檢查工具（含單元測試）。官方資料原樣放在 `data/data_VSLAM-T1/`、`data/data_VSLAM-T2/`；starter 工作副本在 `vslam/`，來源與 SHA256 見 [vslam/origin.json](vslam/origin.json)。
- 自寫的 GPU 視覺引導幾何 SLAM `vslam/gslam/`（入口 `vslam/run_slam.py`），參數集中在 `vslam/gslam/params.py` 的 `EXPERIMENTS`。gslam 第一版 Public 0.369、第二版（密集閉環＋尺度處理）0.144、第三版（BA）0.085。
- 另接 DROID-SLAM（`docker/droid/`、`scripts/run_droid.py`）、MASt3R-SLAM（`docker/mast3r/`）、VGGT（`docker/vggt/`）、深度模型評估（`docker/depth/`）；後處理 `scripts/opposite_pgo.py`（P1）、`scripts/droid_ba.py`（F1）。
- gslam 執行時用 `./scripts/dev.sh slam --track 0X` 會自動開 RViz；跑完的外部系統 run 用 `./scripts/dev.sh replay outputs/run/<run>` 重播（見 doc/03「用 RViz 觀看執行過程」、doc/05）。
- starter README 禁止「完整 SLAM／SfM 系統產生提交」，與 Kaggle 規則／投影片的文字不同；使用者 2026-11-16 依官方規則決定可直接使用公開預訓練模型與開源 SLAM 系統，報告須列出名稱、版本、授權與自己的改動（見 [AGENTS.md](AGENTS.md)）。

`data/`、`outputs/` 皆被 `.gitignore` 忽略，不進 Git、不會跨電腦同步。資料只供本課程使用，不得散布（見 [doc/02](doc/02_KAGGLE_RULES_TRACK01_20261113.md)）。

## 目錄

| 位置 | 用途 |
| --- | --- |
| `vslam/` | 我們的 SLAM 程式（從 starter 複製後修改）；參數集中在 Python 檔 |
| `Dockerfile`、`docker-compose.yml`、`scripts/dev.sh` | Ubuntu 專用容器 `midterm02_vslam` |
| `scripts/create_run.py` | 建立唯一 run 目錄並凍結程式快照、SHA、Git 差異 |
| `scripts/check_submission.py` | 檢查提交 CSV 格式與 timestamp 是否與 sample 完全一致 |
| `scripts/run_droid.py`、`scripts/opposite_pgo.py`、`scripts/droid_ba.py` | D 系列（DROID）、P1（反向重訪約束）、F1（DROID＋SIFT BA）；參數在各檔開頭或 `EXPERIMENTS` |
| `scripts/analyze_external_run.py`、`scripts/rviz_replay.py` | 外部系統 run 的 analysis 圖表；把跑完的 run 重播到 RViz |
| `docker/{droid,mast3r,vggt,depth}/` | 外部模型的一次性容器（`run.sh`） |
| `tests/` | 工具與算法的單元測試 |
| `doc/` | 規則、算法說明、實驗紀錄（編號＋日期命名） |
| `deliverables/` | 報告、簡報、E3 交付草稿 |
| `data/`（忽略） | 官方資料包原樣解壓（含 starter 原檔），唯讀 |
| `outputs/run/`（忽略） | 所有 run：`track01_full_YYMMDDHHmm` 等 |

## 環境（Ubuntu＋Docker）

執行一律在 Ubuntu 主機的專用容器 `midterm02_vslam`（映像 `midterm02_vslam:py39-gpu`，Python 3.9＋CUDA；學習式特徵要用 `./scripts/dev.sh build-learned`）內進行；DROID／MASt3R／VGGT 各有一次性容器（`docker/*/run.sh`）。Windows 只用於編輯。出現 `permission denied … docker.sock` 時，命令前加 `sg docker -c '…'`。

```bash
cd ~/sdc_ros2/midterm02          # 依你 Ubuntu 上的 repo 位置調整，需在 midterm02/main 分支
./scripts/dev.sh build
./scripts/dev.sh up
./scripts/dev.sh test             # 單元測試（tests/，47 項）
./scripts/dev.sh shell            # 進入容器
```

**手動跑各版本（E27、D8、P1、F1 與變體）、RViz 重播與驗證腳本**：見 [doc/05](doc/05_MANUAL_RUNS_20261118.md)。

## 常用命令（容器內）

```bash
python scripts/create_run.py --kind test --track 01
python scripts/check_submission.py outputs/run/<run>/result_1.csv data/data_VSLAM-T1/Track1/data/sample_submission.csv
python -m unittest discover -s tests
```

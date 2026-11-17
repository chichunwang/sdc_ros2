# Midterm02 Visual SLAM

用前鏡頭影像與官方 LoGeR 點雲，估計 Track01／Track02 每張影像的 camera-to-world pose，提交 Kaggle（Sim(3) 對齊後 ATE，越低越好）。規則重點見 [AGENTS.md](AGENTS.md)，官方規則完整整理見 [doc/01](doc/01_COMPETITION_RULES_20261113.md)。

## 目前狀態（2026-11-16）

- 新增自寫的 GPU 視覺引導幾何 SLAM `vslam/gslam/`（入口 `vslam/run_slam.py`）、GPU 版 Dockerfile／compose、`scripts/gpu_check.py`、`scripts/inspect_data.py` 與測試。設計、GPU／CPU 分工、執行順序見 [doc/03](doc/03_GSLAM_DESIGN_AND_STATUS_20261116.md)。
- 容器 `midterm02_vslam:py39-gpu` 已建置；GPU 驗證與 34 項合成測試通過。第一版 full（`track01_full_2611160111`）Kaggle Public **0.369**（baseline 0.955）。第二版（密集閉環＋尺度處理，`track01_full_2611160533`）Public **0.144**。第三版（加上多幀重投影最佳化 bundle adjustment，`track01_full_2611161129`）Public **0.085**。第四版（BA 再加跨幀與跨圈配對，`outputs/run/track01_full_2611161431/result_1.csv`）**尚未提交**。
- **手動執行一律用 `./scripts/dev.sh slam --track 01`**：會自動開 RViz 視窗（相機影像、特徵匹配、點雲地圖、軌跡、keyframe、閉環），說明見 doc/03 的「用 RViz 觀看執行過程」。
- 參數集中在 `vslam/gslam/params.py`；容器映像改名為 `midterm02_vslam:py39-gpu`。

## 先前狀態（2026-11-15）

- 已建立目錄架構、run 建立工具與提交格式檢查工具（含單元測試）。
- 已從 `Project Assets/Midterm02_data/` 複製官方資料到 `data/data_VSLAM-T1/`，保留素材原件；包含 1,314 張影像、1,314 份點雲、相機內參、IMU 與提交範本。逐檔 SHA256 核對紀錄見 `data/import_manifest.json`。
- starter 原檔保留於資料包，工作副本已放入 `vslam/`；每個來源檔的路徑與 SHA256 見 [vslam/origin.json](vslam/origin.json)。尚未修改算法、執行 SLAM 或取得分數。
- Track02 已公布（2026-11-17 教授更新投影片）：4206 張、10 Hz、約 2.1 km、終點回到起點、無 IMU，Public 421／Private 3785。資料已原樣解壓到 `data/data_VSLAM-T2/`（`Track2/data/` 與 `image_slam_starter/`）。Track1 最佳方法 D2 的 Track2 結果：`outputs/run/track02_full_2611171900/result_2.csv`（待提交）；我們自己的 E21 在 Track2 軌跡壞掉，不提交。詳見 [doc/03](doc/03_GSLAM_DESIGN_AND_STATUS_20261116.md)「Track02 第一輪」與「速度改動」。

## 下一步（使用者）

1. 將本機忽略的 `midterm02/data/` 另行複製到 Ubuntu 工作區相同位置；Git 不會同步資料。
2. 依 [starter README](vslam/README.md) 準備 Ubuntu Docker 環境，再於 `vslam/` 實作 TODO。starter 列出的安裝套件另含 pandas、scipy、tqdm；套件版本與容器環境尚待建立及驗證。

starter README 最後一段禁止完整 SLAM／SfM 系統產生提交，與已保存的 Kaggle 規則／投影片允許以現成系統為起點的文字有差異；兩份原文均保留，採用此類系統前需釐清。此次僅匯入資料及未修改的 starter。

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

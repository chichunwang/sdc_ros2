# Midterm01：影像點雲與 LiDAR 地圖定位

單人完成。目標是將兩條 Track 的影像重建點雲配準到各自 LiDAR 地圖，估計每幀車輛 pose，產生可提交的 Kaggle CSV。

目前已建立環境與工作目錄；老師的 `localization` 模板、Track 資料與定位實作尚待接入。先看 [需要自己下載的資料與程式](docs/DOWNLOADS.md)。

## 架構

```text
midterm01/
├── README.md
├── Dockerfile                    # 延伸既有 HW2 的 ROS 2 Humble 映像
├── docker-compose.yml            # 獨立 Midterm01 容器與資料掛載
├── docker-compose.gui.yml        # 選用：遠端 Ubuntu 的 X11 顯示
├── .env.example                  # 遠端主機資料根目錄與容器設定
├── config/tracks.json            # Track 路徑與公告幀數
├── workspace/src/                # 老師的 localization package 放這裡
├── scripts/
│   ├── dev.sh                    # 遠端 Ubuntu：初始化、容器與檢查入口
│   ├── build_workspace.sh        # 容器內：檢查 package 後執行 colcon
│   └── check_data.py             # 檔案完整性與 PLY 數量檢查
├── data/                         # 原始資料，可改掛遠端既有目錄
├── outputs/                      # runs、submissions、figures
├── experiments/README.md         # 版本、設定、分數與問題紀錄
├── reports/README.md              # 單人報告與簡報安排
└── docs/DOWNLOADS.md              # 必需／選用下載清單
```

`data/Track1`、`data/Track2` 與輸出子目錄由初始化腳本建立；大型資料、執行輸出及個人 `.env` 被 Git 忽略。說明文件與程式可正常追蹤。

## 執行環境與路徑

你在 Windows 編輯，實際課程執行端是 VS Code Remote SSH 連接的 Ubuntu，專案為 `~/sdc_ros2`。以下 Docker 指令在**遠端 Ubuntu** 執行。

| 遠端 Ubuntu 主機 | Midterm01 容器 |
| --- | --- |
| `~/sdc_ros2/midterm01/workspace/` | `/root/midterm/` |
| `.env` 中的 `MIDTERM_DATA_DIR`，預設 `./data` | `/root/data/`，唯讀 |
| `~/sdc_ros2/midterm01/config/` | `/root/config/`，唯讀 |
| `~/sdc_ros2/midterm01/scripts/` | `/root/scripts/`，唯讀 |
| `~/sdc_ros2/midterm01/outputs/` | `/root/outputs/` |

獨立容器預設名 `sdc_midterm01`，Compose project 為 `sdc_midterm01`，ROS_DOMAIN_ID 預設為 1。HW2 仍使用自己的 `ros2_container` 與原有掛載。資料掛載使用明確的主機／容器對應，語法參考 [Docker Compose volumes 文件](https://docs.docker.com/reference/compose-file/services/#volumes)。

## 1. 遠端 Ubuntu：初始化

先把這些新檔案同步到遠端專案，再執行：

```bash
cd ~/sdc_ros2/midterm01
bash scripts/dev.sh init
```

這會建立資料與輸出目錄，並在 `.env` 不存在時複製範例設定。若資料已在 Ubuntu 其他位置，將 `.env` 的 `MIDTERM_DATA_DIR` 改成實際資料根目錄的絕對路徑。預設目錄結構是：

```text
資料根目錄/
├── Track1/
│   ├── map.pcd
│   ├── init_guess.txt
│   └── raw_ply/
└── Track2/
    ├── map.pcd
    ├── init_guess.txt
    └── raw_ply/
```

若 Track 目錄或點雲資料夾名稱不同，在 `config/tracks.json` 調整。這份 JSON 目前供資料檢查工具使用；接入老師程式時，仍須同步核對它的 launch 與 C++ 資料路徑。

## 2. 放入需要自己下載的內容

按 [下載清單](docs/DOWNLOADS.md) 取得：

1. 老師的 Localization 模板：完整 package 放入 `workspace/src/localization/`。
2. 官方 Track1 與 Track2：放入資料根目錄的對應子目錄。
3. 各自的 official sample submission：保存原始欄位及 ID，後續轉檔時核對。

HW2 的既有 bag 可留在原位置。第一次先用官方 `raw_ply/`；三相機點雲與 LoGeR 程式列為後續選用。

## 3. 遠端 Ubuntu：建置並開啟容器

```bash
bash scripts/dev.sh config
bash scripts/dev.sh build
bash scripts/dev.sh up
bash scripts/dev.sh status
```

`config` 用來檢查解析後的遠端資料來源與掛載。`build` 會先檢查 `sdc_hw2_ros2:latest` 是否存在；缺少時使用專案 `hw2/` 的 Dockerfile 與 entrypoint 建置基底，再建立 Midterm01 映像並補上 PCL、Eigen 與 colcon 等依賴。建置基底時只傳入這兩個必要檔案，不包含 HW2 大型資料；此方式使用 [Docker 的 tar build context](https://docs.docker.com/build/concepts/context/#local-tarballs)。Docker daemon 權限依你現有 Ubuntu 的設定使用；若平常需 sudo，在需要 Docker 的腳本命令前加 `sudo`，`init` 則以一般帳號執行。

`up` 需要映像已建好。原始資料以唯讀方式掛載，輸出使用 `/root/outputs`。目前配置先支援不開 GUI 的資料處理；需要 RViz 時，確認遠端 Ubuntu 的顯示環境後使用 `bash scripts/dev.sh --gui up` 加入 X11 overlay。顯示授權及 `DISPLAY` 需依你既有的遠端顯示方式設定。

## 4. 遠端 Ubuntu：檢查資料與建置 ROS package

```bash
bash scripts/dev.sh check-data
bash scripts/dev.sh check-data --strict-count --output /root/outputs/runs/data_check.json
bash scripts/dev.sh build-workspace
bash scripts/dev.sh shell
```

資料檢查可用 `--track track01` 或 `--track track02`。它檢查地圖、初始姿態檔與 PLY 是否存在且非空，以及 PLY 數量是否符合公告；數量差異預設為 warning，`--strict-count` 則視為錯誤。這不是點雲內容、座標系、官方 frame ID、sample submission 或定位品質的驗證。

`build-workspace` 在容器內執行 colcon；尚未放入老師 package 時會明確報錯。模板接入後，仍需確認它的完整依賴與資料契約。進入容器後，完成建置才能使用：

```bash
# 以下在容器內執行
cd /root/midterm
source install/setup.bash
```

接著讀取老師的 `localization.launch.py`，確認輸入、輸出與參數後再啟動。此階段尚未建立實際定位啟動腳本，避免使用未確認的模板介面。

純資料結構檢查也可直接在有 Python 3 的主機執行，不依賴 ROS：

```bash
python3 scripts/check_data.py --data-root /實際資料根目錄 --track track01
```

## 接下來的實作順序

1. 接入模板，核對點雲尺度、座標系、外參、初始姿態與結果格式。
2. 完成單幀 ICP，顯示配準前後疊圖。
3. 連跑短片段，再完成兩條 Track，使用官方轉換工具產生 CSV。
4. 加入局部地圖、下採樣、初始姿態預測與失敗回退，逐項記錄改善。
5. 在 `experiments/` 整理實驗，於 `reports/` 整理報告與簡報。

目前的架構不包含已完成的定位演算法或有效 Kaggle 成績。Ubuntu 上的 Docker 建置、ROS 編譯與真實資料測試需在實際執行環境驗證；檔案放妥後依上述步驟接續。

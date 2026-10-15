# sdc_ros2 專案說明與協作約定

## 專案用途

這是 SDC 自駕車課程的作業與專案工作區，包含 ROS 2 感測器資料處理、點雲與 Radar 分析、機率模型練習，以及車輛定位競賽。根目錄彙整各項作業，各子目錄依自身的環境與說明建置、執行。

目前主要工作是 **Midterm01：以影像產生的 3D 點雲，對齊預先建立的 LiDAR 地圖，估計每一幀的車輛位置與朝向**。使用者獨自完成一組，不需要找組員或安排雙人分工。

## 目錄與用途

| 目錄 | 內容 |
| --- | --- |
| `hw2/` | 對應 HW2 講義的 `sdc_hw2/`，含 ROS 2 Humble 的 Docker 環境與感測器分析工具 |
| `hw2/workspace/` | HW2 的 Python 節點、launch 與 RViz 設定 |
| `hw3/` | Markov chain 天氣模型、隨機模擬與穩態分布的 Python 練習 |
| `docs/` | 作業說明、終端機指令與開發工具使用文件 |
| `midterm01/` | Midterm01 的實作與專案文件工作目錄 |
| `localreference/` | 本機參考資料，目前包含 Midterm01 題目 PDF 與分析；已被 Git 忽略 |
| `tmp/` | 產生的預覽、擷取結果及其他暫存檔；已被 Git 忽略 |

Midterm01 開工前，先閱讀：

1. `localreference/midterm01/README.md`：題目分析、單人實作計畫、期限與繳交清單。
2. `localreference/midterm01/SDC 2026 Midterm01 Competition -  Localization.pdf`：課程正式題目，共 24 頁。
3. 取得老師模板後，再讀取模板自身的說明、建置設定、定位程式與結果轉換腳本。
4. Docker 配置來源：[SDC Assignment 2 - ITRI Dataset Visualization](https://hackmd.io/@26f-sdc/rkh6TD-czg)，並對照本機 `hw2/` 的設定與下方路徑說明。

若參考檔被移動，先搜尋目前位置；不要直接依過期路徑重新建立或覆蓋檔案。課程要求以正式題目及後續公告為準，分析文件中的演算法建議不是強制規定。

## Midterm01 的目標與建議方向

- 必須完成兩條路線：Track01 公告為 3011 幀，Track02 公告為 3299 幀；每條路線使用各自的地圖與初始資料。
- 輸入包含前相機點雲 `raw_ply/`、LiDAR 地圖 `map.pcd`、含雜訊的 `init_guess.txt`；另有三相機點雲、影像、校正資料、IMU 與 GNSS 可供改善。
- 核心工作是實作 scan matching，取得每幀車輛 pose，轉成符合官方規格的 Kaggle CSV；兩條 Track 均須超過助教 baseline。
- 第一版建議直接使用已提供的前相機點雲與 ICP，先完成讀取、配準、序列輸出及有效提交，再逐步測試局部地圖、下採樣、初始姿態預測與失敗回退。
- NDT、多尺度配準、三相機及 IMU／GNSS 是後續候選改善，應依實際失敗案例與對照實驗決定是否加入。先建立參考版，再評估效果。
- 題目已提供重建點雲，LoGeR 程式是選用資源；不必先重跑影像重建才能開始定位。
- 最終需整理程式、可重現的 README、兩條 Track 最佳結果 `result_1.csv`／`result_2.csv`、報告與課堂簡報。

截至 2026-10-15，已完成題目分析；本次可直接檢查的 Windows checkout 尚未放入老師的定位模板與 Track 資料，尚無實測定位分數。使用者已確認遠端 Ubuntu 有 ROS 執行環境與資料，截圖也顯示 HW2 bag 存在；Midterm01 模板與 Track1／Track2 在遠端的存放狀態尚未查驗。後續依實際執行端的檔案與結果更新，不以 Windows checkout 推斷遠端資料缺失。

## 實際開發與執行環境

使用者目前在 **Windows** 操作，並透過 **VS Code Remote SSH 連接遠端 Ubuntu**。ROS 課程的執行環境與資料位於遠端 Ubuntu；後續 ROS／Docker 操作優先以這台遠端主機為準，不預設改用 Windows 本機 Docker 或 WSL。

| 層級 | 已知位置與用途 | 證據／確認範圍 |
| --- | --- | --- |
| Windows 本機 | `C:\Users\ricky\Downloads\Project\sdc_ros2`，目前助理可直接讀寫的 checkout | 本次工具實際存取的位置 |
| 遠端 Ubuntu | SSH 畫面顯示 `220.132.18.102`，終端機為 `ricky@rickyubuntu24`，專案為 `~/sdc_ros2` | 使用者說明與截圖；尚未由本次工具連線查驗 |
| Docker 容器 | 課程配置宣告容器名 `ros2_container`，內部工作目錄為 `/ros_ws` | 來自 Windows checkout 的 Docker 設定；遠端容器目前狀態與實際掛載待核對 |

截圖可確認遠端 Ubuntu 有以下 HW2 資料：

```text
~/sdc_ros2/hw2/data/guanfu_road_clipped_ros2/
├── guanfu_road_clipped_ros2.db3
└── metadata.yaml
```

遠端也有 `~/sdc_ros2/hw2/workspace/`、`hw2/docker-compose.yml`、`hw2/Dockerfile` 與 `hw2/ros_entrypoint.sh`。`~/sdc_ros2` 是畫面顯示的 shell 路徑；需要絕對路徑時，先在遠端用 `pwd` 或 `realpath` 確認，不先假設 home 的實際位置。

Windows 與遠端 Ubuntu 是兩份 checkout。Git 可同步已追蹤的程式；`data/`、`localreference/`、`tmp/` 等忽略項目不會因一般 Git 同步而出現在另一台機器。遠端若缺少本地參考文件，先確認文件是否僅放在 Windows，不據此推定遠端缺少 ROS 或資料。

## Docker 配置來源與本機路徑差異

使用者指定的 Docker 配置來源為 [HW2 作業說明](https://hackmd.io/@26f-sdc/rkh6TD-czg)。講義使用獨立的 `sdc_hw2/` 目錄；本專案則將其放在 `sdc_ros2/hw2/`，並將 `midterm01/`、`hw3/` 等目錄並列。後續操作要依本機結構調整，不為了配合講義目錄名稱而移動或重建既有資料夾。

本次已讀取 Windows checkout 的 `hw2/docker-compose.yml` 與 `hw2/Dockerfile`。若遠端使用同一份設定，路徑對應如下；主機側的 `./data`、`./workspace` 應以遠端 Ubuntu 的 Compose 專案目錄解析。

| 講義或 Compose 中的項目 | 遠端 Ubuntu 對應位置 | 容器內位置／狀態 |
| --- | --- | --- |
| 講義的 `sdc_hw2/` | `~/sdc_ros2/hw2/` | Docker build context |
| `./workspace` | `~/sdc_ros2/hw2/workspace/` | 宣告掛載至 `/ros_ws/workspace` |
| `./data` | `~/sdc_ros2/hw2/data/`，截圖可見 HW2 bag | 宣告掛載至 `/ros_ws/data` |
| HW2 bag | `~/sdc_ros2/hw2/data/guanfu_road_clipped_ros2/` | 依上述掛載應為 `/ros_ws/data/guanfu_road_clipped_ros2/` |
| Midterm01 實作目錄 | 規劃使用 `~/sdc_ros2/midterm01/`，遠端目前狀態待確認 | Windows checkout 的 Compose 尚未宣告此掛載 |
| 題目與分析 | 本次可確認在 Windows 的 `localreference/midterm01/`；遠端是否有副本未知 | Windows checkout 的 Compose 尚未宣告此掛載 |

上表的主機檔案存在狀態來自使用者截圖，容器路徑則依設定檔推導；尚未讀取遠端 Compose 的實際內容或查驗執行中容器掛載。Windows checkout 沒有 `hw2/data/` 與截圖所示遠端已有資料並不衝突，不能據此要求使用者重新下載 HW2 資料。截圖中的 HW2 bag 也不等於 Midterm01 的 Track1／Track2。

操作時遵守以下原則：

- 建置前先在遠端 Ubuntu 確認 `hw2/docker-compose.yml` 的 `build`、`volumes` 及資料實際位置。使用講義的 `docker compose` 指令時，在遠端切換至 `~/sdc_ros2/hw2/`；若從其他位置執行，明確指定 Compose 檔與專案目錄。
- 依目前設定，映像名稱為 `sdc_hw2_ros2`、Compose service 為 `ros2_dev`、容器名稱為 `ros2_container`、Dockerfile 的工作目錄為 `/ros_ws`。這些是設定值；執行前仍需查驗容器是否存在及是否沿用此設定。
- 掛載來源是執行 Docker 的遠端 Ubuntu 主機路徑，掛載目標是容器路徑；Windows 的 `C:\...` 不會自動成為遠端主機可用的掛載來源。
- Midterm01 題目中的 `/root/midterm` 與 `/root/data/Track1`、`/root/data/Track2` 是範例配置，Windows checkout 的 HW2 Compose 尚未宣告這些掛載。沿用遠端容器時，依遠端實際程式及資料目錄配置掛載，或同步調整程式／launch 的容器內路徑。
- `midterm01/` 是 `hw2/` 的同層目錄；編寫掛載來源時要考慮此層級。保留既有 HW2 工作區，另行配置 Midterm01 所需路徑，不假設 `hw2/workspace/` 已含 Midterm01 程式。
- 講義的 `xhost`、X11 socket 與 GUI 啟動步驟需對照遠端 Ubuntu 的實際顯示環境。RViz 如何顯示、容器的 `DISPLAY` 與 X11 掛載是否有效，仍須在遠端確認。

## 實作時的重要約定

- 使用繁體中文說明與撰寫專案文件，保留必要的英文技術名稱。
- 先檢查相關檔案與 Git 狀態，保留使用者既有修改；變更限制在目前任務，不順便重構其他作業。
- 明確區分 Windows／PowerShell、遠端 Ubuntu／Bash 與 Docker 容器三層環境；Docker 管理指令以遠端 Ubuntu 為執行端，ROS 2 與 `colcon` 以已配置好的 ROS 環境／容器為執行端。給指令時標明在哪一層執行。
- Windows 上缺少 `docker`、`ros2`、`colcon` 或資料目錄，不代表遠端缺少環境。不要因此先要求在 Windows 重裝 ROS／Docker、改用 WSL 或重新下載遠端已有資料。
- Docker 配置以使用者指定的 HW2 講義與本機設定為依據，依上方路徑對照核對掛載、依賴及 launch 設定；講義目錄名與本機目錄不同。
- 配準前先確認點雲尺度、座標系、相機到車體外參、轉換方向、timestamp 與角度單位；不自行假設模板已處理或尚未處理這些轉換。
- 取得 sample submission 與轉換腳本後，確認正式 CSV 欄名、ID 集合、順序與 pose 規格；不可只憑檔名猜測格式。
- 不將 ICP／NDT 的收斂狀態或 fitness 分數當成正式定位誤差，也不直接把 GNSS topic 當作評分 ground truth。
- 保存候選版本的設定、程式版本、輸出、耗時及兩條 Track 分數，讓報告中的改善可以重現。
- 資料集、ROS bag、建置輸出與本機參考資料依 `.gitignore` 管理；不要為了加入程式而強制追蹤大型資料或取消既有忽略規則。
- 修改程式後執行與變更相關、且目前環境確實支援的驗證；若缺少模板、資料或執行環境，明確記錄尚未驗證的項目。
- 回報時區分「已完成」、「建議」與「待確認」，不宣稱尚未執行的建置、實驗或提交已成功。

## 建置與驗證入口

目前沒有適用於整個根目錄的統一建置或測試指令。Docker 的基礎配置參考使用者指定的 HW2 講義與專案中的設定；實際在遠端 Ubuntu 核對 Compose、容器與掛載。Midterm01 取得模板後，再核對其依賴、ROS 2 package 與執行方式。

Midterm01 題目示範在配置好的容器內執行：

```bash
cd /root/midterm
colcon build
source install/setup.bash
ros2 launch localization localization.launch.py
```

只有程式確實位於容器的 `/root/midterm`，且資料路徑與模板設定一致時，才直接使用上述指令。若採用其他掛載目標，先調整工作目錄及 launch 的輸入／輸出路徑。修改 C++ 後需重新建置。提交前核對兩條 Track 的 ID 完整性、有限數值、座標與角度規格，並保存最佳結果所對應的版本與設定。

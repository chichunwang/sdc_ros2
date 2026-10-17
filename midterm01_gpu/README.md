# Midterm01 GPU 開發環境

新增 GNSS／IMU 約束的定位 profile，處理舊 CUDA D2D 完整 Track1 漂移與姿態翻轉；使用時加 `--profile sensor_guided`，預設使用新 `cuda_robust_icp`（GPU 最近鄰搜尋＋CPU 受約束細修），詳見 [SENSOR_GUIDED_20261017.md](SENSOR_GUIDED_20261017.md)。舊指令預設仍為 `legacy`／`cuda_ndt`，用於重現先前結果。使用者於 2026-10-17 手動提交新版 Track1，截圖回報 **1.22369**，優於同圖 baseline **1.79075**；Track2 分數尚未回報。三階段架構與演算法教學見 [doc/ALGORITHM_WALKTHROUGH_20261017.md](doc/ALGORITHM_WALKTHROUGH_20261017.md)。

此目錄是 `/home/ricky/sdc_ros2/midterm01_gpu` 的獨立 ROS 2 Humble／CUDA 12.8 開發環境，針對 RTX 5060 的 `sm_120` 編譯。原有 `midterm01` 容器與正在執行的 CPU 定位保留。

2026-10-17 後續檢查：主機 CDI 已啟用，新容器已通過 RTX 5060 的真實 CUDA kernel／4096 個結果核對，證據在 `outputs/runs/cuda_runtime_20261017_79psai_k/`。後續已整合 CUDA NDT，現在定位預設使用 `registration_backend=cuda_ndt`；實際定位測試與限制見 `CUDA_NDT_20261017.md`。

兩條 Track 前 50 幀的幾何貼合與軌跡穩定性比較見 [ACCURACY_COMPARISON_20261017.md](ACCURACY_COMPARISON_20261017.md)。目前 CUDA D2D 與原 PCL 演算法不同，實測有較大的 pose 跳動，尚不能視為等精度替代；保留原 PCL 演算法的 CUDA 移植尚未實作。

## 資料與隔離

| 主機來源 | 容器位置 | 用途／權限 |
| --- | --- | --- |
| `../midterm01/data` | `/root/data` | 共用現有資料，唯讀，不建立第二份 |
| `./workspace` | `/root/midterm` | 本環境獨立程式、build、install、log，讀寫 |
| `../midterm01/config` | `/root/config` | 共用 Track 檢查設定，唯讀 |
| `./scripts` | `/root/scripts` | GPU 環境工具，唯讀 |
| `./outputs` | `/root/outputs` | 本環境建置紀錄、測試與定位輸出，讀寫 |

Compose project／service／預設容器均為 `sdc_midterm01_gpu`，映像為 `sdc_midterm01_gpu_ros2:cuda12.8`，`ROS_DOMAIN_ID=2`。原 CPU 環境使用 domain 1。啟停腳本只操作 GPU project。

`workspace/src/localization` 是建立此環境時從原環境複製的程式快照，後續獨立編輯；不會自動同步修改，也不共用原環境的 build 或 install。資料根目錄由 `.env` 的 `MIDTERM_GPU_DATA_DIR` 設定，預設 `../midterm01/data`。Docker 使用 `create_host_path: false`，來源不存在時直接報錯，不建立空資料夾。

## 建置

以下在 Ubuntu 主機執行：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
bash scripts/dev.sh init
bash scripts/dev.sh config
bash scripts/dev.sh build
```

Dockerfile 延伸已存在的 `sdc_midterm01_ros2:latest`，安裝 CUDA 12.8 的 nvcc、cudart 開發套件及 CCCL；不在主機安裝 CUDA 或替換 NVIDIA 驅動。建置時會編譯 `sm_120` CUDA 檢查程式。

## 主機啟用 GPU：Docker 原生 CDI

主機需有可用的 NVIDIA 驅動與支援[原生 CDI 的 Docker](https://docs.docker.com/reference/cli/docker/container/run/#cdi-devices)（28.3 以後預設啟用）。此機的 Docker 29.9.0 已回報 `/etc/cdi`、`/var/run/cdi` 搜尋路徑。

若主機尚未安裝 NVIDIA Container Toolkit，需由使用者在主機輸入 sudo 密碼執行一次：

```bash
sudo bash scripts/setup_host_gpu.sh
```

此腳本依 [NVIDIA 官方安裝方式](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) 設定套件來源，安裝 `nvidia-container-toolkit-base`，並依 [CDI 官方說明](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html) 產生 `/var/run/cdi/nvidia.yaml`。只使用原生 CDI，不修改 Docker daemon 設定、不重啟 Docker。Toolkit 的 CDI refresh 機制負責開機／驅動更新後重新產生規格；更換驅動或裝置後若出問題，重新執行生成與檢查。

```bash
bash scripts/dev.sh up
bash scripts/dev.sh cuda-check
bash scripts/dev.sh build-workspace
run_dir="$(mktemp -d outputs/runs/check_XXXXXXXX)"
bash scripts/dev.sh check-data --strict-count \
  --output "/root/outputs/runs/$(basename "$run_dir")/data_check.json"
bash scripts/dev.sh shell
```

`cuda-check` 必須實際通過記憶體配置、CPU→GPU 傳輸、GPU kernel 執行、GPU→CPU 傳輸及 4096 個計算結果核對，才算 GPU runtime 通過；只看到 `nvcc --version` 或 `nvidia-smi` 不足以證明。

尚未啟用主機 CDI 時，可以明確使用無 GPU 診斷模式完成映像／ROS 建置與唯讀資料檢查：

```bash
bash scripts/dev.sh --cpu up
bash scripts/dev.sh build-workspace
bash scripts/dev.sh check-data --strict-count
```

此模式的 `cuda-check` 預期失敗。主機 CDI 完成後，執行一般 `bash scripts/dev.sh up`，只會重建本 GPU project 的容器並加入 GPU，保留 workspace 與 outputs。

## CUDA NDT 與前 50 幀試跑

`cuda_ndt_backend` package 以固定 commit／SHA256 建置 [fast_gicp 的 CUDA NDT](https://github.com/koide3/fast_gicp)。預設 `cuda_ndt`、D2D、DIRECT7；Track1 測試中 P2D 出現較大的姿態跳動，最後選用 D2D，仍不保證等同或優於原 CPU 軌跡。初始化或 CUDA 執行錯誤會直接失敗，不會改成 CPU 後端。一般不收斂時仍沿用原本保留上一個 pose 的規則，並記錄失敗幀。

在主機執行：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
bash scripts/dev.sh up
bash scripts/dev.sh build-workspace
python3 scripts/benchmark.py --track 1 --frames 50
```

幀數直接改指令中的 `--frames 50`，例如 `--frames 100` 就跑從起始開始、依 timestamp 排序的前 100 幀，不用改 C++ 或重新編譯。省略時預設 50；此腳本只接受正整數，不以 0 表示全部。完整預期筆數為 Track1 的 `--frames 3011`、Track2 的 `--frames 3299`，執行前須確認資料與短序列結果。已建置後，日常使用只需 `up` 與執行 benchmark；修改程式後才重新 `build-workspace`。可直接貼上的常用流程見 [doc/command_paste.txt](doc/command_paste.txt)。

Track2 改成 `--track 2`。明確比較同一個 Release 程式內的 CPU PCL 後端：

```bash
python3 scripts/benchmark.py --track 1 --frames 50 --backend pcl_ndt
```

每次自動產生唯一的 `outputs/runs/trackN_<backend>_<mode>_<frames>f_*` 目錄，保存完整參數、命令、版本／來源 SHA256、frame 對應、原始矩陣、診斷 CSV、逐幀分段時間、GPU 使用取樣、程序退出碼與輸出有限值／筆數檢查。`--compare /path/to/result.txt` 可額外記錄軌跡差異；該差異不是 ground truth 誤差。腳本不開 RViz、不讀 bag、不使用 map 快取，預設總時間包含地圖讀取／下採樣，且拒絕在本容器已有 localization 時再啟動另一輪。

重複比較不同設定時，可明確加 `--map-cache outputs/runs/<唯一實驗目錄>/cache_track1_leaf1.pcd`，將約 50 萬點的衍生下採樣地圖存於 outputs；原始 5953 萬點地圖與資料集仍只有 `midterm01/data` 一份。首次產生仍讀取原始 map；後續使用時先核對 Track、原始 map SHA256 與 leaf size 的 sidecar，拒絕不符的快取。SHA256 前置檢查不計入 ROS 程序時間；結果另記是否命中快取。比較完整耗時時兩邊必須使用相同快取策略，不能將命中快取的時間對比原始未快取結果。預設比較仍停用快取。

`timings.csv` 的 `ndt_ms` 包含 input／submap 設定、傳輸、target tree／voxel 建立、配準及 CUDA 同步；`fitness_ms` 是另計的 CPU PCL fitness。`publish_ms` 包含顏色點雲濾波、發布與 pose 寫入。程式啟動時另印地圖初始化秒數。

CUDA 大量點運算包括 voxel 統計、correspondence、cost、gradient 與 Hessian；CPU 仍處理 6×6 LM 解算、讀檔／濾波／裁切、fitness 與 ROS 發布。CUDA 使用 LM，`cuda_translation_epsilon=0.01`、`cuda_rotation_epsilon=0.002`；`ndt_step_size`／`ndt_transformation_epsilon` 只適用於 `pcl_ndt`。兩者最佳化與鄰域行為不同，不能保證同一軌跡，也不能把速度提升視為精度提升。

CUDA target 先平移至 voxel resolution 整數倍的局部原點，避免數千公尺全域座標的 float voxel 共變異數消去誤差；初值平移到同一座標系，最終 pose 再加回原點，voxel 格線相位與 source→map 定義保留。這是數值座標平移，不是另外套外參或重套初始矩陣。無有效 Hessian 的結果會視為未收斂，避免零對應點被誤報成功。

後續執行定位時保持 `p_map = T_map_source × (ply_scale × p_raw)` 定義與單次初值套用規則；沿用原資料的 frame／尺度待確認事項。輸出明確指定新的 `/root/outputs/runs/<唯一run>/`，不覆寫原 CPU 結果，不自動提交 Kaggle。

`build-workspace` 預設最多兩個 colcon worker；原 CPU 工作仍在執行時可在 `.env` 將 `BUILD_WORKERS=1`。該參數控制 package 並行數；本環境另外限制各 package 的編譯工作數以控制負載。

## 狀態與記錄

```bash
bash scripts/dev.sh doctor
bash scripts/dev.sh status
bash scripts/dev.sh stop
```

實際建置／驗證記錄放在 `outputs/runs/`，詳細首建狀態見 `SETUP_20261017.md`。輸出、`.env`、build、install、log 均由 Git 忽略。

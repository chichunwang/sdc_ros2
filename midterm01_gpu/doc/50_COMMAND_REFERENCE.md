# 指令表

更新日期：2026-10-24。以下命令在 Ubuntu 主機執行；一次只執行一個定位程序，上一輪結束後再選擇下一輪命令。

## Track3 rewrite：獨立 C++／CUDA observable ICP v1

新入口 `track3_rewrite/run.py`，method=`track3_cpp_cuda_observable_icp_v1`，原生 package 位於 `workspace/src/track3_rewrite/`。此版從零實作 C++／CUDA 配準核心，不沿用舊 scripts、`track3_clean/` 或舊 CUDA solver；v2 fast 與 clean 分支維持獨立。模型、數值方法、限制與驗證狀態見唯一 [51](51_TRACK3_ARCHITECTURE_AND_DEVELOPMENT_20261023.md)。

`--check` 僅執行隔離 Release build 與 GPU 合成 CTest，不跑真實 cloud；`--full --dry-run` 僅盤點 full 輸入，也不做配準。此兩種檢查都不能記為定位成功。本輪 12項GPU機制測試通過，真實 3142 幀 dry-run 通過；本輪 SHA 與檢查紀錄見 `outputs/run/track03_test_2610240045/`。原生GPU合成3／5幀已完成，TXT／CSV前綴逐byte相同；另9項launcher／I/O檢查與no-support／SIGINT失敗案例通過。沒有執行新真實定位或full，正式精度未知。

完整定位由使用者手動執行：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
bash scripts/dev.sh up
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 outputs/ml_runtime/venv/bin/python track3_rewrite/run.py --full --map-cache outputs/run/track03_full_2610231608_0002/map_cache/map_xyz.npy
```

指定的歷史run只提供經SHA核對的原始map XYZ cache，沒有讀取其pose。也可移除 `--map-cache ...`，由新版獨立解析原始 `Track3/map.pcd` 建立本輪cache；NumPy僅用於此ASCII I/O，不是定位solver依賴。新native程式不使用Torch allocator，請勿混入舊 `--vram-budget-gib` 旗標。

## 保留入口：Track3 clean local-Sim3

`scripts/run_track3_clean.py` 使用獨立 package `track3_clean/`，method=`track3_clean_local_sim3_v1`。此版保留既有數值元件，與本文開頭的 C++／CUDA 核心重寫分開；架構與 pair model 見唯一 [51](51_TRACK3_ARCHITECTURE_AND_DEVELOPMENT_20261023.md)，正式精度仍未知。full 由使用者手動執行，不增加精度資格門檻。已有定位程序運行時不可並行使用 GPU，CLI 會拒絕啟動。clean 當時盤點所見的舊 full `track03_full_2610232327` 後來已依使用者授權中斷，保留為未完成 run。

唯讀 dry-run 只盤點 full 輸入契約，不執行配準：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
bash scripts/dev.sh up
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 outputs/ml_runtime/venv/bin/python scripts/run_track3_clean.py --full --dry-run --vram-budget-gib 4 --map-cache outputs/run/track03_full_2610231608_0002/map_cache/map_xyz.npy
```

完整定位命令（由使用者手動執行）：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
bash scripts/dev.sh up
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 outputs/ml_runtime/venv/bin/python scripts/run_track3_clean.py --full --vram-budget-gib 4 --map-cache outputs/run/track03_full_2610231608_0002/map_cache/map_xyz.npy
```

30 項 CPU 數值／契約／I/O／CLI／故障檢查通過，CPU合成3／5幀engine序列的TXT／CSV前綴逐byte一致。真實3142幀dry-run通過；dry-run本身不初始化CUDA、不驗map payload SHA、不定位。另一次唯讀盤點已驗證map/cache SHA及首／中／尾PLY解析取樣。證據見 [`track03_test_2610240007`](../outputs/run/track03_test_2610240007/README.md)。此 clean 入口尚無真實GPU定位或正式分數。

## 歷史入口：map-only NDT／ICP v2 與先前研究

以下一段保留當時 v2 更新記錄，供閱讀其歷史語境；本輪獨立 C++／CUDA rewrite 入口見本文開頭。

先前 map-only v2 使用前鏡頭與 map XYZ，以相鄰點雲配準提供初值，再作當前 map NDT／ICP、視線檢查與失鎖重捕獲；不接 IMU／GNSS、不讀外部預測，也不回修過去 pose。200 幀機制檢查已完成，但不是精度證明；算法沿革與限制見唯一 [51](51_TRACK3_ARCHITECTURE_AND_DEVELOPMENT_20261023.md)。既有 legacy baseline 與規則限制見 [49](49_TRACK3_CAUSAL_BASELINE_20261023.md)、[48](48_RULES_AND_CODE_AUDIT_20261023.md)，通用輸出格式見 [08](08_MANUAL_WORKFLOW.md)。

## 常用命令

| 用途 | 命令／行為 | 是否定位 | 說明 |
| --- | --- | --- | --- |
| Track3 map-only NDT／ICP full | 見下方 v2 手動命令 | 是 | full 由使用者手動啟動，不設額外資格門檻；沒有正式新分數或精度證明。 |
| Track3 map-only NDT／ICP 200 幀 | 將 `--full` 換成 `--frames 200` | 是 | 可選機制／故障診斷，不是 full 資格門檻；結果見下方。 |
| 舊獨立幀 NDT／ICP 診斷重現 | 見下方「舊研究重現」 | 否 | 只讀舊 run 的 lossless map XYZ cache 並重核 map/cache SHA；不讀舊 pose、不產生軌跡或提交。8/8 multi-start 拒絕僅屬該舊實驗。 |
| 舊核心去 GNSS full | 見「歷史：舊核心去 GNSS full 命令」 | 已完成 | 使用者回報 77.96；上傳 bytes 未核對。僅供歷史追溯，不要重跑。 |
| Track3 舊 legacy 前 50 幀／full | 見「歷史 legacy baseline」 | 是 | 保留重現舊 run 用；固定 scale 3 未校準，不作現行推薦。 |
| GPU 容器環境狀態 | `bash scripts/dev.sh status` | 否 | 查看開發容器狀態。 |
| GPU 環境診斷 | `bash scripts/dev.sh doctor` | 否 | 執行環境診斷。 |
| 舊核心對照 dry-run | 見歷史命令區塊 | 否 | 歷史檢查命令；列印當時 3142 幀及凍結來源設定，不編譯、不定位、不建立新 run。 |
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

## 歷史：Track3 map-only NDT／ICP v2 命令與研究

### 前鏡頭 v2 runner 手動命令

`scripts/run_track3_map_only.py` 使用 GPU 相鄰 PLY 配準提供初值，再以每幀 map NDT＋ICP 和 training-only source normal-axis／spatial 等權固定權重做 map tracking；scale seeds 為 0.95／1／1.05。拒絕後在既有 8 m／30° bootstrap 域嘗試重捕獲。physical map motion 相對最後接受的 map anchor 依時間間隔比較；pair motion rate 40 m/s、90°/s 保留。原殘差、視線與共識門檻不變。runner 不讀 IMU／GNSS／舊 pose，無 RTS／graph／離線修正。`--vram-budget-gib 4` 是 Torch allocator cap，不涵蓋 driver、NVRTC 等全部 VRAM。方法細節見 [51](51_TRACK3_ARCHITECTURE_AND_DEVELOPMENT_20261023.md)。

當時 v2 full 命令（歷史記錄）：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
bash scripts/dev.sh up
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 outputs/ml_runtime/venv/bin/python scripts/run_track3_map_only.py --full --vram-budget-gib 4 --map-cache outputs/run/track03_full_2610231608_0002/map_cache/map_xyz.npy
```

可選 200 幀機制／故障診斷命令：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
bash scripts/dev.sh up
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 outputs/ml_runtime/venv/bin/python scripts/run_track3_map_only.py --frames 200 --vram-budget-gib 4 --map-cache outputs/run/track03_full_2610231608_0002/map_cache/map_xyz.npy
```

`track03_test_2610232248` 200 幀完整（182 map accepted／18 rejected、lost 0、198/199 pair accepted、最長無 map 0.107871264 s）；19 次重捕獲嘗試中 14 次成功，總耗時 338.130684431 s。registration after first median 為 1473.078968 ms。40 項機制／I/O 檢查與 CPU/CUDA weighted-fit parity 通過，5／200 矩陣及 CSV 前綴逐 byte 一致。Torch peak allocated／reserved 為 924777984／1509949440 bytes。這些數字只描述該段完整性、機制與耗時，不證明精度或全程穩定；validation 標示 accuracy unknown，沒有正式新分數。詳細方法與限制見 [51](51_TRACK3_ARCHITECTURE_AND_DEVELOPMENT_20261023.md)。

### 歷史：v1 前向 runner 失鎖命令與結果

`track03_full_2610231912` 使用 v1，僅處理 1040/3142 幀後 return code 1；`output_complete_and_finite=false`、`validation_errors=[]`，map 98 採用／942 拒絕、pair 1034/1039 採用、idx 125 為最後 map 採用、idx 181 起失追共 859 幀，最長無 map 32.893173258 s。[Run validation](../outputs/run/track03_full_2610231912/validation.json) 與[唯讀前綴稽核](../outputs/run/track03_test_2610231924/rejection_audit.json)保存證據。終止原因未記錄，不推定；原 run 保留，不推薦原樣重跑。同一 CLI 現在執行 v2，不能重現 v1；舊 source snapshot 保留。根因分析見唯一 [51](51_TRACK3_ARCHITECTURE_AND_DEVELOPMENT_20261023.md)。

以下命令是 v1 歷史命令，只供記錄，不作當前推薦：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
bash scripts/dev.sh up
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 outputs/ml_runtime/venv/bin/python scripts/run_track3_map_only.py --full --map-cache outputs/run/track03_full_2610231608_0002/map_cache/map_xyz.npy
```

已完成的 100 幀 v1 機制研究保留於 `track03_test_2610231848`；它不證明精度或全程穩定。現行程式即使指定 `--frames 100` 也會執行 v2，不能據此宣稱重現 v1；查閱舊算法須使用該歷史 run 的來源快照。詳見唯一 [51](51_TRACK3_ARCHITECTURE_AND_DEVELOPMENT_20261023.md)。

成功完成 full 才在當次 run 的 `analysis/` 產生標準圖表；短段不產生完整比較圖，見[08](08_MANUAL_WORKFLOW.md#full-run-事後圖表)。

### 舊研究重現：獨立幀診斷

以下 `scripts/track3_registration.py` 與 `scripts/test_track3_map_registration.py` 是先前的獨立幀診斷，不是現行前向 runner。它使用舊 full run `track03_full_2610231608_0002` 保存的 lossless map XYZ cache，執行前重核原 map 與 cache SHA；不讀舊 pose、不生成定位軌跡或提交，輸出在新的 `outputs/run/track03_test_*`。依序執行：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
bash scripts/dev.sh up
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 outputs/ml_runtime/venv/bin/python scripts/test_track3_map_registration.py --map-cache outputs/run/track03_full_2610231608_0002/map_cache/map_xyz.npy
```

舊研究的 21 項 CPU／CUDA 機制檢查通過（`outputs/run/track03_test_2610231802/final_mechanism_tests.log`）；extended 證據位於 `outputs/run/track03_test_2610231814`。該實驗含 5 組 map-derived 已知 Sim3 配 ICP 有／無 NDT 共 10 次；另對 8 個真實獨立幀（0、1、2、4、8、16、24、32），以 2 種姿態初值及 3 個尺度 seed 測試 48 個候選，該實驗的 8 個 multi-start 全部拒絕。首幀尺度 2.71355 與 6.39973 各有局部通過解，最終因歧義拒絕。這些舊實驗結果不證明 map 精度可用，也不能代表現行前向 runner 的接受情形；沒有新 Kaggle 分數。此命令不是定位 full 命令。

## 歷史：舊核心去 GNSS full 命令

以下是已完成版本的歷史命令，不要重跑。使用原 2026-10-19 已評分 run 保存的 C++ 四自由度 ICP 原碼與相同 CUDA 最近鄰 library；不使用 GNSS、RTS、graph、新版 Sim3 或 Kalman。每幀 prior 使用上一個輸出位置及過去 gyro；尺度固定 3。首次 `ground` 只把首幀下平面對齊 map +Z 作近似初始化，沒有估尺度或平移，不是物理標定。使用者完成 `track03_full_2610231748` 後回報 **77.96**；上傳 bytes 未核對，這不是本機真值。11 項 CPU／CUDA 合成檢查不構成精度保證。

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
bash scripts/dev.sh up
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 outputs/ml_runtime/venv/bin/python scripts/run_track3_old_icp.py --full --ply-scale 3 --initial-tilt ground --geometry-mode directional
```

入口當時會在新 run 內自動編譯獨立 bridge，不替換已安裝 binary。保留舊 map／input PCL voxel 1／0.3 m、directional geometry、1.5 m／8°修正域、alpha 0.25；舊 XY／yaw 正則項圍繞運動 prior，並非 GNSS 量測。下列命令及可選短段只作歷史記錄。原始旋轉對照參數為 `--initial-tilt official`；既有定位程序當時會被拒絕並行啟動。

CPU-only dry-run：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 outputs/ml_runtime/venv/bin/python scripts/run_track3_old_icp.py --full --ply-scale 3 --initial-tilt ground --geometry-mode directional --dry-run
```

v2 的 CPU audit-only 仍保留作輸入／座標檢查，不產生定位 pose，也不呼叫 GPU；它不是本輪舊核心 full 命令：

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

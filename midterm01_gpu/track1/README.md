# Track1：因果尺度與地圖修正記憶

Track1 已準備獨立入口，使用獨立複製的 `track1/` 程式；算法沿用 Track2 v4 的因果 GNSS／IMU prior、地面尺度與傾斜估計，以及帶有不確定度的 map XY／yaw correction。參數沒有為 Track1 精度另行調整，因此程式可執行不代表 Track1 定位精度已驗證。

## 手動啟動

等目前 Track2 full 結束後，由使用者在 Ubuntu 主機終端機手動啟動：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
./scripts/run_track1.sh --full
```

入口會自動開啟 RViz，並建立唯一的 `outputs/run/track01_full_YYMMDDHHmm/`。Track1、Track2、Track3 共用 GPU lock；若另一 Track 還在工作，入口會直接拒絕新 run，不會排隊，請等前一輪結束再手動啟動。Track1 程式已在目前 GPU workspace mount 中，不需為這份獨立副本重建 Docker image 或修改 `.dockerignore`。

RViz topics 在 `/track1/*`，TF 是 `map → track1_camera`。配準接受狀態、尺度與 correction 狀態是估計器診斷，不是真值。

## 算法與調整位置

當幀只消費截至該影格的感測器事件。短暫配準失敗會保留 map correction 0.5 秒，之後以 3 秒 time constant 逐漸退回感測器 prior。候選配準會比較本幀與上次吸收的原始候選之運動差異，並用濾波預測的 NIS、幾何品質及每幀 correction 更新上限來降權或拒絕。這只限制 map correction 的變化；相機 pose 仍隨當幀先驗與融合結果前進，不回頭修改舊 pose。

Track1 的參數集中在 [parameters.py](parameters.py)。Track2 v4 的濾波原理補充見 [doc67](../doc/67_TRACK2_CAUSAL_CORRECTION_MEMORY_20261029.md)，其中的舊 run、888 幀候選重播與數值結果都屬於 Track2，**不是 Track1 驗證結果**。Track1 源碼與各檔案對應來源的 SHA256 記在 [origin.json](origin.json)；每輪 run 另凍結 source 並保存 SHA。

## 輸入與尚待驗證的假設

Track1 有 3011 個相機重建 PLY。bag 使用 `/imu/data` 與 `/localizer_node/gnss_pose`；GNSS 沒有訊息的影格不插值，當作沒有新 GNSS 更新。GNSS header frame 被假設為 map 座標；其座標語意、時間差、camera lever arm 和 covariance 尚未標定。IMU 與 PLY 軸向／外參也尚未正式標定，所以 GNSS／IMU 都是帶假設的融合輸入，不是 ground truth。相機校正與 mask 的來源 SHA 會隨 run 記錄。

### CPU 入口檢查與資料時鐘稽核

CPU-only 檢查共 **6 項通過**（`track00_test_2610290353`，5.02 秒）；唯一既有 `Axes3D` warning 不影響 2D 圖。檢查涵蓋 Track1 參數及 5 個核心檔案與 Track2 v4 的來源 SHA 一致、因果 prefix、3011 幀 CLI 範圍與共享 lock 拒絕行為、run snapshot SHA tamper 偵測、RViz namespace／TF，以及 3011 列輸出和 5 組 PNG/PDF 唯讀分析的合成檢查。No-CUDA guard 阻擋 CuPy；`pytest_guard.json` 與 `check_track1_inputs_guard.json` 均記錄 `cupy_imported=false`。

唯讀資料契約稽核同樣通過：3011 幀相機輸入精確配對，收到 GNSS 440 個與 IMU 10995 個事件，late／duplicate 皆為 0；最大 IMU age 為 0.014715103 秒，小於 0.1 秒，mask 尺寸為 928×1440。這次稽核沒有讀取 PLY／地圖 payload，也沒有執行配準；GNSS 空影格的處理與 GNSS frame 等仍未標定，不能視為真值或精度證據。檢查摘要見 [run analysis README](../outputs/run/track00_test_2610290353/analysis/README.md)，圖表：[CPU 入口檢查](../outputs/run/track00_test_2610290353/analysis/01_cpu_entry_checks.png)、[輸入時鐘](../outputs/run/track00_test_2610290353/analysis/02_input_clocks.png)。

以上只驗證入口、來源一致性、資料配對／時鐘契約與合成分析路徑；沒有 Track1 真實配準或定位精度結果。不可把 Track2 的測試、重播或歷史分數當作 Track1 成績。

## Full run 輸出

完成 3011 幀後，run 目錄預期包含：

| 檔案 | 內容 |
| --- | --- |
| `result_1.csv` | `idx,x,y,yaw`，3011 筆資料列，不含表頭 |
| `poses.txt` | 每幀 row-major 3×4 pose，共 12 個數值 |
| `timings.csv` | 每幀輸入、配準和總耗時，以及尺度／接受狀態 |
| `observations.jsonl` | 感測器、配準品質、尺度與 correction filter 診斷 |
| `pose_trace.jsonl` | 當幀 prior、原始配準候選與實際輸出 pose |
| `analysis/` | 成功 full 後唯讀產生 5 組 PNG/PDF 與來源 manifest |

`analysis/` 的五組圖是 `01_trajectory`、`02_pose_changes`、`03_registration_timing`、`04_fusion_and_scale` 和 `05_map_correction_memory`。圖表只呈現該 run 輸出，不修改 pose 或 CSV；只有成功完成 3011 幀並具備逐幀 correction 診斷時，才會產生完整圖表。

# Track2：因果地圖修正記憶 v4

Track2 現行程式為 `track2_causal_metric_correction_memory_v4`。它保留逐幀 GNSS／IMU prior、地面尺度與傾斜解算，再讓 map feature 的 XY／yaw 修正具有記憶和不確定度。短暫配準失敗時保留上一個修正；候選修正突然跳大時，依幾何資訊、配準品質、車速、時間間隔和轉向降低權重或拒絕。詳細原理、診斷欄位與限制見 [67｜因果地圖修正記憶](../doc/67_TRACK2_CAUSAL_CORRECTION_MEMORY_20261029.md)。

## 啟動與調整

Ubuntu 主機上啟動專用容器與 RViz full 定位：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
./scripts/dev.sh up
./scripts/run_track2.sh --full
```

Full 由使用者手動啟動；每輪會凍結程式與參數到新的 `outputs/run/track02_full_YYMMDDHHmm/`。不要在定位進行中更換來源，也不要與 Track3 同時爭用 GPU。要改 RViz 顯示項目，編輯 [config/track2.rviz](../config/track2.rviz)。Topic 在 `/track2/*`，TF 為 `map → track2_camera`。

直接編輯 [parameters.py](parameters.py)。調整衰減時間看 `CORRECTION_HOLD_SECONDS`、`CORRECTION_DECAY_SECONDS`；調整候選突跳降權／拒絕界線看 `CORRECTION_POSITION_GATE_*`、`CORRECTION_YAW_GATE_*`、`CORRECTION_MOTION_REJECT_MULTIPLIER` 與 `CORRECTION_NIS_*`。運動差異比較本幀與上次吸收的原始候選，NIS 比較濾波預測；超過 motion allowance 先依比例降權，大於 3 倍才因運動差異硬拒絕。候選吸收前，XY／yaw correction 實際每幀增量還會限幅。先固定其他設定、每次改一個模型假設。這些值仍是工程設定，尚未以獨立真值校準。

`PLY_SCALE=3.0` 只作地面資料不足時的未驗證 fallback。正常情況尺度仍從當幀地面高度與非地面幾何決定；v4 不會把 raw scale 或 ground tilt 直接套用本文件介紹的 XY／yaw 修正濾波器。

## 本版驗證狀態

現行機制通過 29 項合成／輸出檢查，耗時 10.30 秒（[pytest.xml](../outputs/run/track00_test_2610290341/pytest.xml)、[analysis](../outputs/run/track00_test_2610290341/analysis/README.md)）。888 幀舊候選唯讀重播對照：[frame 428](../outputs/run/track00_test_2610290340_0002/analysis/01_frame_0428_correction_replay.png)、[frame 653](../outputs/run/track00_test_2610290340_0002/analysis/01_frame_0653_correction_replay.png)。這些檢查程式行為，不代表真實定位精度。v3 full 已停止於 1689 幀，舊 metadata 仍顯示 `running`，保留原樣；三幀歷史幾何對照見 [66](../doc/66_TRACK2_METRIC_SCALE_20261029.md)，不是 v4 連續資料或精度驗證。

每個 full run 預期 3299 幀。主要輸出是 `result_2.csv`（`idx,x,y,yaw`）、`poses.txt`（每幀 3×4）、`observations.jsonl`（原始配準候選、融合修正與不確定度）、`pose_trace.jsonl`（`registered` 原始候選、`final` 實際輸出）及 `test_info.json`。成功 full 自動產生 `analysis/` PNG/PDF 與來源 SHA manifest；v4 另產生 `05_map_correction_memory.png/pdf`，舊 run 沒有逐幀 correction diagnostics 時不會補這張歷史圖。未完成 run 不會產生完整分析圖。

**環境備註：**專用容器由先前停止的容器狀態恢復；尚未以 Dockerfile 從零重建驗證。Track3 程式與設定沒有在本次 Track2 修正中更改。

# 作者定位流程：Track1／Track2 本機驗證

這兩個入口讓你以作者的 CPU 配準與融合流程分別跑 Track1、Track2，作為獨立研究對照。它們獨立於本機 GPU v5.8；尚未以這套入口執行本輪真實定位，也未重現作者候選成績。這輪只做環境與合成檢查，完整定位由你手動啟動。

## 現在怎麼跑

先在 Ubuntu 主機確認資料、bag topic 與程式介面：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
./scripts/dev.sh up
./scripts/run_external_track1.sh --check
./scripts/run_external_track2.sh --check
```

`--check` 會檢查資料檔名與幀數、PCD header、bag 內 GNSS／IMU topic 型態及至少兩筆 GNSS 訊息，並執行作者合成測試和 wrapper（入口包裝器）契約檢查。它會保存該次計畫命令，但不執行真實地圖配準或定位；檢查通過不等於精度通過。

完整定位由你在主機上分別手動啟動。等 Track1 完整結束後，再啟動 Track2：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
./scripts/run_external_track1.sh --full
```

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
./scripts/run_external_track2.sh --full
```

每次都建立新的唯一 run，分別位於 `outputs/run/track01_full_YYMMDDHHmm/` 與 `outputs/run/track02_full_YYMMDDHHmm/`；同分鐘撞名時會加序號。兩個入口共用工作 lock，請勿同時啟動。

## 作者來源、資料與參數

Track1／2 都使用 `midterm01/others code track1` 作者包；Track3 另使用已分開的 `midterm01/others code track3`。兩個 Track1／Track3 作者目錄有 27 個相同相對路徑，其中 21 個 SHA 相同、6 個不同，所以不能把兩包說成完全相同。執行副本只含兩處 `buffering=0` I/O 相容調整；作者原檔和既有候選 CSV 都保留、不作定位真值。

每條 Track 使用自己的原始前相機 PLY、地圖、初始姿態與 bag，不使用 `kaggle_track0N/` 的 MapAnything PLY：

| Track | 資料目錄 | `raw_ply/` 幀數 | 地圖處理 |
| --- | --- | ---: | --- |
| 1 | `midterm01/data/Track1/` | 3011 | ASCII PCD，使用作者轉檔流程 |
| 2 | `midterm01/data/Track2/` | 3299 | Binary PCD，透過 XYZ-only adapter |

Track2 地圖欄位為 `tl th x y z intensity`。adapter 僅把 XYZ 複製至作者 Nx4 cache 格式；第 4 欄補零。`tl/th` 僅計算 binary record byte offset，不解碼時間或重建軌跡；intensity 不作定位輸入。原 PCD 唯讀，轉檔檔案寫入本輪 `work/`。

參數集中在 [`scripts/external_parameters.py`](../scripts/external_parameters.py) 的 `TRACKS`。Track1 主流程的 argv 與作者 `run_track1_causal.sh` 由 fake Python wrapper 合成檢查核對，沒有執行定位。Track1／2 都啟用 GNSS tracking prior，並各自把 GNSS bag、`--lever (0.7086, -0.5995)` 與 `camyaw=-0.4455°` 傳給作者流程；GNSS 是流程輸入，不是 ground truth。固定設定包括高度 `1.573 m`、名義尺度 `8.6`、配準距離 `12／15／18／25 m`、`backfill=0`、`gyro_dt=0.0`、兩個 processes。Track1 的 lever arm 與 yaw offset 依作者 Track1 車體參考點定義。

Track2 沿用 Track1 的 lever arm、相機 yaw 與車體參考點假設，但沒有獨立標定，也未核實官方評分所用的 frame 定義；以上固定值都是研究設定，不代表 Track2 最佳參數或已驗證的座標契約。若要調整，直接改 `TRACKS` 中對應項目，不需新增命令列旗標。

## Run 輸出與限制

完整 run 保存作者程式快照、參數、來源及輸入 SHA256、frame manifest、實際命令、階段 log、`observations.jsonl`、`stage_timings.csv`，以及 `result_1.csv` 或 `result_2.csv`。成功完成後，`analysis/` 會提供軌跡、相鄰 pose 變化、階段耗時、尺度／配準診斷等 PNG／PDF、`README.md` 與 `manifest.json`。

作者程式沒有逐幀 timing，也沒有最後 yaw 覆換後的完整 3D pose，因此不會產生虛構的 `timings.csv` 或 `poses.txt`；階段耗時只對應實際記錄到的階段。CSV 格式與幀數檢查不是定位精度證據，也不代表重現作者成績。入口不會以 `compare_csv` 或候選 CSV 篩選定位結果；外部 CSV 不是 ground truth。

另有因果性限制：作者流程可能以全段第一筆有效 registration 初始化先前狀態，並在最後獨立產生 yaw；IMU clock 假設也尚未核實。雖然 Track1 argv 對照作者 shell 名稱含 `causal`，這仍不足以證明整體流程符合競賽因果規則；請把它視為研究對照，而不是已認證的正式提交版本。中止時在原啟動終端按 `Ctrl+C`，保留 run 內 log 和失敗資訊，不覆寫歷史輸出。

## 本輪環境與合成檢查（2026-11-03）

兩條 Track 都完成資料檢查與測試，未執行真實定位：

| Track | Run | 資料檢查 | 作者合成測試 | wrapper 契約測試 | 環境圖 |
| --- | --- | --- | --- | --- | --- |
| 1 | [track01_test_2611031512](../outputs/run/track01_test_2611031512/test_info.json) | `checks_completed`、`data_ready=true`；3011 幀、440 筆 GNSS | 23 passed，3.32 秒；4 項既有 warnings | 11 passed，5.68 秒；1 項既有 Matplotlib `Axes3D` warning | [PNG](../outputs/run/track01_test_2611031512/analysis/01_environment_checks.png)／[PDF](../outputs/run/track01_test_2611031512/analysis/01_environment_checks.pdf)／[manifest](../outputs/run/track01_test_2611031512/analysis/manifest.json) |
| 2 | [track02_test_2611031512](../outputs/run/track02_test_2611031512/test_info.json) | `checks_completed`、`data_ready=true`；3299 幀、475 筆 GNSS | 23 passed，3.37 秒；4 項既有 warnings | 11 passed，5.71 秒；1 項既有 Matplotlib `Axes3D` warning | [PNG](../outputs/run/track02_test_2611031512/analysis/01_environment_checks.png)／[PDF](../outputs/run/track02_test_2611031512/analysis/01_environment_checks.pdf)／[manifest](../outputs/run/track02_test_2611031512/analysis/manifest.json) |

原作者包、執行副本、入口來源快照、測試與圖表來源 SHA 已完成核對，見 [Track1 audit](../outputs/run/track01_test_2611031512/integration_audit.json)／[Track2 audit](../outputs/run/track02_test_2611031512/integration_audit.json)。兩個 run 的 `analysis/manifest.json` 也記錄環境圖表來源。先前中間 run `track01_test_2611031510`、`track02_test_2611031511` 的新 wrapper fixture 少了 `sdcloc` import path，造成各 1 項測試失敗；只修正 fixture 的 import path 後重跑，最終 run 皆為 11/11 通過。失敗 run 保留，測試數不跨 run 加總。這些檢查沒有執行地圖配準或定位，也沒有證明精度或完整競賽因果合規。

<details>
<summary>先前驗證證據（2026-10-29；對應當時來源快照）</summary>

以下紀錄屬於 2026-10-29 當時的入口與來源快照，不代替本次檢查，也不代表本次來源已有相同測試結果。

兩個入口分別完成相同的 22 項作者合成測試及 9 項跨 Track 契約檢查；兩輪輸入檢查均為 `ready: true`，沒有執行真實定位。作者既有 smoother 生成測試有 4 個 empty-slice／invalid-divide warning，Matplotlib 有既有 Axes3D warning，2D 圖表正常產生。

| Track | 紀錄 | 作者測試 | 契約檢查 | 圖表 |
| --- | --- | --- | --- | --- |
| 1 | [track01_test_2610291349](../outputs/run/track01_test_2610291349/test_info.json) | 22 passed，3.43 秒 | 9 passed，5.56 秒 | [環境檢查圖](../outputs/run/track01_test_2610291349/analysis/01_environment_checks.png) |
| 2 | [track02_test_2610291349](../outputs/run/track02_test_2610291349/test_info.json) | 22 passed，3.38 秒 | 9 passed，5.66 秒 | [環境檢查圖](../outputs/run/track02_test_2610291349/analysis/01_environment_checks.png) |

契約檢查包括三種完整幀數的生成 CSV／圖表、繪圖前後來源 SHA 不變、錯序與 NaN 拒絕、Track3 命令與作者原 shell 對照、Track1／2 不帶 Track3 專用 override、ASCII 地圖數值不變，以及 binary 地圖的時間／intensity 數值改變但 XYZ 不變時輸出 bytes 完全相同。這些不是完整實際地圖轉檔或定位精度證據。

</details>

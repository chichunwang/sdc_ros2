# 作者定位流程：Track1／Track2 本機驗證

本說明讓你用 `midterm01/others code` 的 CPU 配準與融合流程，分別跑 Track1、Track2。這兩條是把作者 Track3 流程移植到 Track1／2 的研究對照，不是作者另有的 Track1／2 已評分版本；目前沒有真實 run 或精度結果。

先在 Ubuntu 主機檢查兩條 Track 的資料與執行環境：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
./scripts/dev.sh up
./scripts/run_external_track1.sh --check
./scripts/run_external_track2.sh --check
```

檢查只讀取輸入檔名、幀數、地圖檔頭、bag 的 IMU topic 型態，並執行作者合成測試和包裝器檢查，不會執行真實定位。各 run 的 `data_check.json` 要有 `ready: true` 才能開始完整定位。檢查本身成功不表示定位精度已驗證。

完整定位由你在主機上**分別手動啟動**。等一條完整結束並完成分析，再啟動下一條：

```bash
./scripts/run_external_track1.sh --full
```

```bash
./scripts/run_external_track2.sh --full
```

兩個入口共用 GPU 工作 lock；同一時間只允許一個 Track 工作。run 會分別寫到 `outputs/run/track01_full_YYMMDDHHmm/` 與 `outputs/run/track02_full_YYMMDDHHmm/`，同分鐘目錄重名會自動增加序號。主機磁碟最近盤點約剩 16 GiB；轉檔與 PLY cache 會另佔空間。不要並行跑兩條，也不要為空間刪除原始資料或歷史結果。

每條 Track 用自己的原始前相機資料，不用 `kaggle_track0N/` 下的 MapAnything PLY：

| Track | 資料目錄 | `raw_ply/` 筆數 | 地圖 |
| --- | --- | ---: | --- |
| 1 | `midterm01/data/Track1/` | 3011 | ASCII PCD，沿用作者轉檔器 |
| 2 | `midterm01/data/Track2/` | 3299 | Binary PCD，使用本專案 XYZ 轉檔介面 |

兩條都從各自目錄讀取 `raw_ply/`、`map.pcd`、`init_guess.txt` 和 `TrackN.db3`。定位器不融合 GNSS；作者的 bag reader 可能會讀取 bag 中的 GNSS 訊息，但估計流程不使用 GNSS 數值。

要改每條資料路徑、程序數或目前使用的流程設定，編輯 [`scripts/external_parameters.py`](../scripts/external_parameters.py)。目前 Track1／2 設定沿用作者 CLI 預設 `init_search=False`、`gyro_dt=0.0`、backfill 關閉；Track3 專用大範圍初始化與 `+0.14 s` IMU 位移不套到 Track1／2。固定高度 `1.573 m`、名義尺度 `8.6`、配準距離 `12／15／18／25 m` 與 `xy_min_info=2` 都沿用作者 Track3 設定，尚未依 Track1／2 校正或驗證。這些設定不等於該 Track 的最佳參數。

## Track2 地圖欄位

Track2 PCD 是 binary，欄位包含 `tl th x y z intensity`。作者轉檔器只支援 ASCII PCD，因此 Track2 入口改用 [`scripts/convert_external_map.py`](../scripts/convert_external_map.py)：

- 只將 XYZ 複製到作者 map cache 所需的前三欄；第 4 欄補零，僅為維持 Nx4 cache 格式。
- `tl/th` 欄位只用來計算 binary record 的 byte offset，不解碼、不拿來還原時間或軌跡；intensity 不作定位資料。
- 轉檔逐塊讀取並檢查 XYZ 有限值及 payload 長度。原 PCD 以唯讀方式開啟，轉檔結果另存於本輪 `work/`。

Track1 地圖是 ASCII，使用作者 `convert_map.py`。作者原始資料包放在 run 的 `external_original/`；實際執行副本放在 `external_source/`，其中 PLY／ASCII PCD 讀取器各有一處 `buffering=0` 相容修正。兩份來源與 patch 都記有 SHA，原始 `midterm01/others code/` 不會被修改。

## 輸出與判讀

成功完成完整 Track 後，run 包含 `result_1.csv` 或 `result_2.csv`、作者各階段中間產物、`stage_timings.csv`、`observations.jsonl`、frame manifest、輸入與來源 SHA、source snapshot、實際命令及階段 log。`result_N.csv` 是作者本輪最後產生的 CSV 原樣複製。

`analysis/` 會有四組 PNG／PDF、`README.md` 和 `manifest.json`：軌跡、相鄰 pose 變化、配準階段耗時、尺度與配準診斷。作者程式沒有逐幀時間，也沒有最後 yaw 覆換後的完整 3D pose；03 圖僅呈現階段耗時，不產生虛構的逐幀 timing 或 `poses.txt`。CSV 筆數與格式檢查通過只代表資料契約檢查通過，沒有 ground truth 就不能據此判定定位精度或宣稱重現作者成績。

此程式保留作者流程中的全段第一個有效 registration 初始化，以及最後獨立產生 yaw 的階段；這份研究入口尚未確認符合競賽因果規則，不能直接視為合規的正式提交版本。停止執行時在原啟動終端按 `Ctrl+C`，保留該 run 的 log 與失敗資訊；不要覆寫它或重算既有 CPU 結果。

<details>
<summary>本輪檢查證據（2026-10-29）</summary>

兩個入口分別完成相同的 22 項作者合成測試及 9 項跨 Track 契約檢查；兩輪輸入檢查均為 `ready: true`，沒有執行真實定位。作者既有 smoother 生成測試有 4 個 empty-slice／invalid-divide warning，Matplotlib 有既有 Axes3D warning，2D 圖表正常產生。

| Track | 紀錄 | 作者測試 | 契約檢查 | 圖表 |
| --- | --- | --- | --- | --- |
| 1 | [track01_test_2610291349](../outputs/run/track01_test_2610291349/test_info.json) | 22 passed，3.43 秒 | 9 passed，5.56 秒 | [環境檢查圖](../outputs/run/track01_test_2610291349/analysis/01_environment_checks.png) |
| 2 | [track02_test_2610291349](../outputs/run/track02_test_2610291349/test_info.json) | 22 passed，3.38 秒 | 9 passed，5.66 秒 | [環境檢查圖](../outputs/run/track02_test_2610291349/analysis/01_environment_checks.png) |

契約檢查包括三種完整幀數的生成 CSV／圖表、繪圖前後來源 SHA 不變、錯序與 NaN 拒絕、Track3 命令與作者原 shell 對照、Track1／2 不帶 Track3 專用 override、ASCII 地圖數值不變，以及 binary 地圖的時間／intensity 數值改變但 XYZ 不變時輸出 bytes 完全相同。這些不是完整實際地圖轉檔或定位精度證據。

</details>

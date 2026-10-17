# GNSS／IMU 約束的點雲定位（2026-10-17）

## 問題與實測依據

使用者提供的 Kaggle 截圖顯示，舊版完整 Track1 CUDA D2D 提交分數為 **573.36983**，助教 baseline 是 **1.79075**。後續使用者手動提交本次新版 Track1，2026-10-17 截圖回報分數 **1.22369**，已優於同圖 baseline。這是使用者提供的成績證據，未另行查詢排行榜；Track2 尚無回報分數。對應候選 SHA256 與紀錄見 [成績證據](outputs/submissions/sensor_guided_20261017_pxy83hix/kaggle_track1_score_20261017.json)。

該次 run 為 `outputs/runs/track1_cuda_ndt_d2d_3011f_xvudsutg/`。雖然有 3011 筆有限 pose、只有 8 幀未收斂，但估計 XY 一直留在起點附近約 40 m 的範圍，且 roll／pitch 曾翻轉。同段 GNSS 輸入顯示路線約一公里；舊估計與 GNSS 輸入的 XY 差異 RMS 為 557.895 m。這個差異不是正式定位誤差，卻能揭露長序列追蹤已失效，不能靠「收斂率高」判斷可用。

## 本次新增的演算法

保留獨立 GPU 環境與舊 NDT 比較模式，新增 `sensor_guided` profile，預設使用 **`cuda_robust_icp`** 後端：

1. 由唯讀 bag 讀取 GNSS 位置與 IMU quaternion，以 PLY 的整數 bag timestamp 對應每幀。GNSS 做 5 點二次平滑與線性內插，IMU 使用 SLERP；GNSS 端點僅容許 0.3 s 等速外推，IMU 端點僅容許 0.03 s 延用，缺口過大直接失敗。
2. 在速度 >5 m/s、轉向速率 <2 度/s 的片段，以 GNSS 速度方向與 IMU yaw 的穩健中位差估計固定 heading bias。這只使用允許的感測器輸入，沒有使用真值、提交結果或 Kaggle 標籤。實際 bias 與樣本雜散記在每次 run 的 `sensor_prior.json`。
3. 每幀以當下 GNSS XY／IMU 方向作先驗，裁切地圖。將 source 的可能位置、修正上限與 2 m 對應門檻合併成可證明涵蓋所有有效對應的包圍盒，進一步挑選附近 map 點；沒有任何可能的有效 map 點時明確使用當下先驗。
4. 在固定 IMU roll／pitch 下，對 XYZ 與 yaw 做帶先驗的 robust 點到地圖細修。使用 Huber 距離、2 m 截斷離群點、GNSS XY 0.5 m／高度 1 m／yaw 3 度的正則化尺度，最多 15 次迭代與回溯線搜尋。GPU 以每個 query 一個 block 完整搜尋候選 map 點，搜尋座標移到局部原點，避免公里級座標的 float 消去誤差；CPU 使用 double 計算目標函數、梯度與 4×4 方程。每次 kernel 呼叫核對 CUDA 錯誤，執行錯誤直接失敗，不默默改成 CPU 後端。
5. XY 修正距離最多 1.5 m、yaw 最多 8 度；以 alpha=0.25 平滑修正，接受前再檢查當下目標函數。GNSS bag 的高度是常數，初幀使用 GNSS 高度，後續沿用地圖細修後高度作高度先驗，每幀高度修正上限 1.5 m。
6. 沒有有效地圖對應或修正遭拒時，輸出當下感測器先驗。每個輸入保留一筆 pose，避免沿用上一幀靜止 pose，讓車輛移動誤差持續累積。

本次新增感測器同步、heading 校正、四自由度正則化細修、CUDA 最近鄰核心、修正門檻與時間平滑，沒有把原有 NDT／下採樣說成新增貢獻。這是 CPU／GPU 協作：GPU 做最近鄰搜尋，CPU 做 robust 最佳化與小型方程求解，**目前新版不是 NDT 演算法**。亦未宣稱完成「與 PCL 完全同公式的 CUDA NDT 移植」。

早期實驗仍使用 fast_gicp CUDA D2D 作候選，但 Track1 第 2242 幀、Track2 第 456 幀觸發 CUDA 記憶體執行錯誤並終止；此前候選未被約束細修採用。失敗 run／驗證保留在本次實驗紀錄，因此最終後端直接使用已驗證的 CUDA 最近鄰核心，不再依賴該 D2D 最佳化器。`--backend cuda_ndt --profile sensor_guided` 只供明確的除錯對照，完整序列已知不可靠；不作為推薦候選。

## 使用方式

目前保留 `legacy` 作預設以重現先前結果，**執行新版必須加 `--profile sensor_guided`**：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
bash scripts/dev.sh up
bash scripts/dev.sh build-workspace --packages-select cuda_ndt_backend localization
python3 scripts/benchmark.py --track 1 --frames 3011 --profile sensor_guided
python3 scripts/benchmark.py --track 2 --frames 3299 --profile sensor_guided
```

請依序執行；試跑仍用 `--frames 50`。每次唯一 run 目錄名稱會含 `sensor_guided_cuda_robust_icp`；`diagnostic.csv` 仍是 `idx,x,y,yaw`，yaw 為弧度。`sensor_prior.csv/json` 保存同步輸入與設定，`guidance.csv` 保存細修接受狀態、修正量與目標函數。`failed_alignments` 在此 profile 是地圖細修未能接受的幀數，這些幀仍有當下感測器先驗。`guidance_not_accuracy.cuda_nearest_neighbor=true` 表示新版 GPU 搜尋已選用；`cuda_converged_frames`／`cuda_seed_used_frames` 是舊 D2D 候選專用欄位，新後端不使用 NDT，所以這兩欄為 0 是正常現象。

可直接複製的流程另存於 [doc/run_sensor_guided.txt](doc/run_sensor_guided.txt)；原本 `command_paste.txt` 中使用者貼上的執行紀錄保留。

`elapsed_seconds` 沿用 ROS／地圖／逐幀執行時間定義；`sensor_preparation_seconds` 是前置感測器同步時間，兩者相加為 `pipeline_elapsed_seconds`。若使用明確 map 快取，必須在比較時註明命中情況，不能與未快取的舊總時間混為同條件比較。

為相容既有記錄格式，此 profile 的 `timings.csv.ndt_ms` 欄名保留，但內容是 **CUDA ICP 搜尋與 CPU 正則化細修總時間**，不是 NDT 耗時。`fitness_ms=0`，因為新後端不另外計算 PCL fitness。`score` 記最終 pose 的 2 m 截斷最近鄰距離平方平均，不能直接與 legacy 的未截斷 PCL fitness 或 Kaggle 分數比較；固定共同地圖的另外評估才用於比較幾何貼合。

## 驗證與限制

- 合成已知場景測試涵蓋大座標、正則化目標改善、不合理 CUDA seed 拒絕、修正上限、合法旋轉及無重疊時使用當下先驗。
- 新 CUDA 最近鄰核心在非整數 block 對齊的 1027 點目標、73 個 query 與 30 次重複執行中，共 2190 個查詢結果逐筆等同 CPU 完整搜尋；合成配準契約同時通過 CPU／GPU 搜尋版本。
- 感測器測試涵蓋奈秒時間、yaw 跨越 ±180 度、短端點外推與資料過期拒絕。
- 原 CPU PCL NDT 的 `legacy` 前 50 幀已與原 R0 結果逐 byte 核對相同；沒有改動原 `midterm01` 程式、資料、建置或正在執行的程序。
- 這是離線演算法，平滑、內插與 heading bias 校正會使用同一包資料中較晚的感測器觀測，不能宣稱為因果即時系統。
- GNSS 的 frame_id 是空字串。程式假設其座標與 map 相容，且 IMU 軸與原 source pose 的 forward／up 相容；初始位置與地圖貼合提供合理性證據，但尚未取得正式完整外參契約。沒有另套車體／相機外參或重複初始矩陣。
- GNSS 只是可用輸入，不是評分真值；地圖貼合、平滑度與正則化目標改善都不能代替實際 Kaggle 分數。正式提交單位、frame 與最佳設定仍應依課程規格及真實提交結果確認。

## 本次完成結果

兩條完整序列已完成，均通過輸出筆數、有限值、旋轉矩陣、修正上限與逐幀正則化目標檢查；沒有 CUDA 執行錯誤或遭拒的地圖更新。此處的成功只表示本機契約與診斷通過，不是定位真值精度保證。

| 項目 | Track1 | Track2 |
| --- | ---: | ---: |
| 完整資料列數 | 3011 | 3299 |
| 感測器同步＋ROS 執行（秒） | 230.92 | 144.34 |
| ROS 執行 `elapsed_seconds`（秒） | 229.65 | 143.04 |
| GPU 搜尋＋正則化細修（秒） | 6.53 | 3.71 |
| map cache | 命中既有 1 m 衍生快取 | 停用 |
| 最大相鄰平移（m） | 0.589 | 0.638 |
| 最大相鄰 yaw（度） | 0.864 | 0.633 |
| 起終點 XY 距離（m） | 1038.486 | 1204.903 |
| 最大 XY 先驗修正（m） | 0.187 | 0.185 |

Track1 的舊 D2D 最大相鄰平移為 10.823 m；新版未再出現起點附近停滯或姿態翻轉。Track1 本輪命中快取，不能直接將完整耗時與舊版未快取的 354.99 秒計算成同條件速度倍率；快取雜湊檢查及版本記錄不在上述計時內。

在每條完整路線均勻抽選 40 幀，以同一份完整 1 m map、原始 PLY 乘 3、0.3 m voxel、單次 source→map pose，另行計算**未截斷**點到地圖最近鄰 RMS：

| 版本 | Track1 殘差 RMS（m） | Track2 殘差 RMS（m） |
| --- | ---: | ---: |
| 僅同步後的 GNSS／IMU 輸入先驗 | 1.627 | 1.023 |
| 新版正則化配準輸出 | 1.253 | 0.624 |

這是同一輸入先驗經地圖修正前後的幾何貼合對照，**不是與真值的誤差，也不是與舊 D2D 或助教 baseline 的成績比較**。舊 D2D 可在錯誤位置取得低局部殘差，因此不能單靠殘差決定定位精度。

本機候選檔（沿用已被使用者提交接受的 CSV 欄位／ID／弧度輸出方式）：

- [Track1 result_1.csv](outputs/submissions/sensor_guided_20261017_pxy83hix/result_1.csv)
- [Track2 result_2.csv](outputs/submissions/sensor_guided_20261017_pxy83hix/result_2.csv)
- [來源與 SHA256](outputs/submissions/sensor_guided_20261017_pxy83hix/provenance.json)
- [完整驗證報告](outputs/runs/sensor_guided_20261017_pxy83hix/REPORT.md)
- [完整軌跡比較圖](outputs/runs/sensor_guided_20261017_pxy83hix/full_trajectory_comparison.png)

本次完整驗證、來源快照與失敗實驗紀錄位於 `outputs/latest_improvement_run.txt` 指向的唯一實驗目錄；候選檔目錄另記在 `outputs/latest_improved_submission.txt`。所有資料及輸出遵守原 Git 忽略規則。Codex 未操作 Kaggle 提交；使用者已手動提交 Track1 並回報上述成績。

使用者已指定本次測完後，後續完整 Track 測試由使用者手動執行並回報結果；本次完成後不再自行啟動下一輪完整序列。演算法的後續取捨依使用者回報的完整測試與 Kaggle 實際分數調整。

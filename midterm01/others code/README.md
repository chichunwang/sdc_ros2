# Track3 定位程式（無 GNSS）

這包程式從助教給的 Track3 資料跑到 Kaggle CSV，輸出前相機位姿。全部是 Python，CPU 就能跑，不需要 ROS、不需要顯卡。

| 一鍵腳本 | 做法 | public 分數 |
| --- | --- | --- |
| `run_track3_causal.sh` | **因果版（正式成績那份）**：每一幀只用到該幀為止的資料，車上可以即時跑 | **0.57** |
| `run_track3.sh` | 離線版：整段一起平滑，每一幀會參考前後所有幀 | 0.48 |

兩份都用一鍵腳本從頭跑過，和交到 Kaggle 的檔逐位元相同。public 只算起點那段直路（約前 13%），兩版在直路的差別就是「能不能看未來的幀」。

## 安裝

1. Python 3.10 以上（我們用 3.12）。
2. `pip install -r requirements.txt`（我們實際用的版本：numpy 2.5、scipy 1.18、polars 1.44、matplotlib 3.11）。
3. 記憶體建議 16 GB 以上。配準會開 2 個程序（`PROCS=3` 可加速，但要 32 GB 以上，不然會爆記憶體）。

## 一鍵執行

```bash
bash run_track3_causal.sh /path/to/Track3 ./work     # 因果版 → ./work/track3_cam_causal.csv（約 35 分鐘）
bash run_track3.sh /path/to/Track3 ./work            # 離線版 → ./work/track3_cam.csv（約 35 分鐘）
```

- `/path/to/Track3` 要有：`raw_ply/`、`map.pcd`、`init_guess.txt`、`Track3.db3`（就是助教 zip 解開的那個資料夾）。
- 指到同一個輸出資料夾，第二條會跳過已經做好的地圖和 PLY 快取。
- 輸出 3142 列、`idx,x,y,yaw`、yaw 是弧度、CRLF 換行，直接交。
- Windows 用 Git Bash 跑；或把腳本裡的指令一行一行貼到終端機。

## 流程（給報告用的說明）

1. **地圖前處理**（`convert_map.py`、`prepare_map.py`）：ASCII PCD 轉 numpy，0.2 m 方格降取樣，算每個點的法向量給點到面 ICP 用。
2. **點雲尺度**（`scale_from_height.py`）：LoGeR 的點雲不是公尺，而且尺度沿路會變（約 6.2～10.3），車停著時點雲還會一直縮。每幀量點雲裡相機前方的路面高度，用「相機離地 1.573 m ÷ 量到的高度」當這一幀的尺度。因果版只看當幀和之前 6 幀的中位數。
3. **逐幀追蹤**（`track_sequential.py`）：第 0 幀不信 init_guess 的姿態（Track3 的 init_guess 帶 20° 假傾斜、yaw 差 17°、位置差 1.9 m），把 roll/pitch 歸零，在附近 yaw ±45°、位置 ±4 m 的格點逐一試配，挑貼合最好的；之後每幀用前幾幀推算的車速預測位置當 ICP 初值。因果版不做「從第 100 幀往回追」。
4. **單幀配準，四種範圍**（`register_all.py`，12／15／18／25 m）：點到面 ICP、粗細兩階段、地面點只配地圖的地面、非地面只配非地面。遠處的點單目深度誤差大、主要把 yaw 扭掉（直路上 25 m 的 yaw 比 12 m 偏 1.5°），所以直路用 12 m 最好（Kaggle：25 m 0.78、18 m 0.62、15 m 0.53、12 m 0.50）；但 12 m 在路口和轉彎沒東西可配（位置可觀測性只有 18 m 的 1/7）。
5. **逐幀挑來源**（`merge_regs.py`）：12 m 的配準夠可靠（xy 資訊量 ≥ 2）就用它，不夠依序退到 15／18／25 m。Track3 直路 2534 幀用 12 m，轉彎段四種範圍各約 70 幀。
6. **因果版濾波**（`build_causal.py`）：前向 EKF，狀態是位置、航向、gyro 零偏。預測用 gyro 積分和最近 20 幀的車速；更新用當幀配準（Cauchy 穩健權重）；IMU 判定靜止時位置、航向不動，gyro 的平均當零偏的量測。零偏一定要當狀態估：Track3 一開始車就在動，拿「開頭幾秒當靜止」算零偏會差 3°/s。
   離線版用 `build_nognss.py`：整段一起解的平滑（等速、不橫移、gyro 相對航向、停車零速、地圖量測 Cauchy），gyro 零偏用三次多項式整段擬合。
7. **yaw**（`yaw_from_motion.py`）：配準的 yaw 有場景偏差（進站那段 12 m 配準的 yaw 掉 2.8°，gyro 說根本沒轉），所以 yaw 的形狀完全信 gyro（扣零偏），只用直路上的行進方向定絕對水平（Track1 排行榜證實 GT 的 yaw 就是行進方向，差 0.09° 以內）。轉彎時位移方向落後車頭、停車後起步是從路邊併回車道，這兩種樣本都不拿來定水平；停車時 yaw 不動。因果版的位移方向取過去 2 秒、水平偏移取過去 20 秒的中位數。
8. **輸出**：前相機位姿，不加車體 lever、yaw 不扣偏角（助教範本程式輸出的也是點雲原點的位姿；Track3 加 Track1 的 lever 會從 0.83 變 1.02）。

Kaggle 上比過、最後沒用的：疊合前後幀配準（越疊越差：疊 2 幀 0.83、疊 4 幀 0.86，都是 25 m）、地圖 Sim3 自估尺度（0.96）、依深度加權（Track1 對假真值一律較差）、Huber 門檻改小（沒差）。

## 遇到問題

- `MemoryError`、`分頁檔太小`或 `can't start new thread`：配準時記憶體不夠，`PROCS=1 bash run_track3_causal.sh ...`，而且不要同時跑別的重活。
- Windows 自己一行一行跑時出現 `UnicodeEncodeError`：先設 `PYTHONUTF8=1`（PowerShell：`$env:PYTHONUTF8=1`），或改用 `python -X utf8 ...`。
- `convert_map.py` 和 `cache_plys.py` 沒有 `--help`，用法就是「輸入路徑 輸出路徑」兩個參數。
- 追蹤那步印出 `init search: fit ...`：fit 是非地面點貼合的比例，0.85 以上算正常（我們是 0.90）。
- 每支輸出腳本都會在 CSV 旁邊寫一個 `.diag.json`（估出來的零偏、靜止幀數、各範圍用了幾幀等）。
- `tests/` 是函式庫的單元測試：`pip install pytest` 後在這個資料夾下跑 `python -m pytest -q`（22 個）。

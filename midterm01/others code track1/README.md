# Track1 定位程式（因果版）

這包和 Track3 那包是同一套程式（`scripts/`、`src/` 完全相同），多了 Track1 的一鍵腳本和 Track1 的候選檔。
全部是 Python，CPU 就能跑，不需要 ROS、不需要顯卡。輸出車體參考點的位姿（Track1 的答案是車體參考點，不是相機；Track3 才是相機）。

| 一鍵腳本 | 做法 | 對假真值 I_ns |
| --- | --- | --- |
| `run_track1_causal.sh` | **因果版**：每一幀只用到該幀為止的資料，車上可以即時跑；和 Track3 正式成績那份同九步 | xy 0.549 m、yaw 0.73°、分數式 0.696 |

Track1 的 Kaggle 已經截止，這份沒有交過榜；數字是對 `candidates/track1_I_ns.csv`（排行榜驗過、public 0.59 的那份）算的。

## candidates/ 裡的檔

| 檔 | 是什麼 |
| --- | --- |
| `track1_causal_hybrid_yawmotion.csv` | **因果版（主候選）**，這包腳本的輸出；`.diag.json` 裡有逐段數字 |
| `track1_causal_r25_yawmotion.csv` | 對照：同流程，但四種範圍不挑、全用 25 m 單幀配準 |
| `track1_Q_ns.csv` | Kaggle 最佳提交（public 0.56）：離線、用 GNSS 位置的 smoother，不是這套流程 |
| `track1_I_ns.csv` | 排行榜驗過的版本（public 0.59），這裡當假真值用。它的位置與 yaw 來自 25 m 配準，**不能拿來評「配準範圍 12 m vs 25 m 哪個好」** |
| `track1_T3c_gyro.csv` | 離線、不用 GNSS 的版本（Track3 的離線流程跑在 Track1 上），對 I_ns 0.398 m／0.20° |

## 安裝、執行

1. Python 3.10 以上（我們用 3.12），`pip install -r requirements.txt`（numpy 2.5、scipy 1.18、polars 1.44、matplotlib 3.11）。
2. 記憶體建議 16 GB 以上；配準開 2 個程序（`PROCS=3` 可加速，但要 32 GB 以上）。

```bash
bash run_track1_causal.sh /path/to/Track1 ./work_track1     # 約 50 分鐘 → ./work_track1/track1_causal_hybrid_yawmotion.csv
```

- `/path/to/Track1` 要有 `raw_ply/`、`map.pcd`、`init_guess.txt`、`Track1.db3`（GNSS 與 IMU 都在 bag 裡）。
- 輸出 3011 列、`idx,x,y,yaw`、yaw 是弧度、CRLF 換行。
- Windows 用 Git Bash 跑；或把腳本裡的指令一行一行貼到終端機。
- 腳本是把 10/31 逐步跑出候選檔的指令照抄（同旗標），沒有再用腳本從頭整條重跑過；若哪一步出錯請回報。

## 流程（給報告用的說明）

九步和 Track3 那包 README 的「流程」完全相同：地圖前處理 → 逐幀尺度（因果）→ 逐幀追蹤 → 單幀配準 12／15／18／25 m → 逐幀挑來源 → 前向 EKF → yaw 用 gyro 形狀＋行進方向定水平。**沒有任何一步用到未來的幀**。只有兩個地方不一樣：

1. **追蹤（第 5 步）加 `--gnss`**。Track1 有 GNSS（4 Hz，在 bag 裡），Track3 沒有。追蹤器預測下一幀位置時，車速改用「幀時間之前最後兩筆 GNSS 定位點」差分，GNSS 位置外插、扣掉 lever 推出相機位置，當 σ 1 m 的弱先驗，和原本的運動預測先驗合成一個；一樣只用該幀之前的資料。
   為什麼要加：Track1 第 150–450 幀的加速段（2 → 9 m/s）沿車向沒有東西可配，ICP 的沿車向極小值跟著預測走、貼合率照樣 0.8，只靠歷史車速的追蹤器速度估計會在 0～9 m/s 之間隨機漫步，位置跑掉幾十公尺卻每幀都判成功。Track3 沒 GNSS，就用歷史位姿估速度（Track3 全段 0 失敗，沒有這個問題）。GNSS 本身因果外插對 I_ns 的誤差 RMS 0.93 m、最大 2.5 m，所以只當弱先驗，位置還是由地圖配準決定。
2. **輸出（第 8 步）加 `--lever 0.7086,-0.5995 --cam-yaw-offset-deg -0.4455`**：配準出來的相機位姿換成車體參考點再進 EKF，之後第 9 步的「行進方向」就是車體航向。lever 是整段「GNSS − 相機」的穩健平均，偏角是相機 yaw 對 GNSS 行進方向的差，都是從資料估的，不是從排行榜反推。相機對 IMU 的時間差給 0（Track1 量到 16 ms）。

## 數字（對 `track1_I_ns.csv`；分數式 = xy RMSE + 0.2 × yaw RMSE 度）

| | 因果版 hybrid | 對照：全 25 m | 離線無 GNSS（T3c） |
| --- | --- | --- | --- |
| 全段 xy RMSE / yaw RMSE / 分數式 | 0.549 m / 0.73° / 0.696 | 0.487 / 0.76° / 0.639 | 0.398 / 0.20° / 0.437 |
| 前 30% | 0.463 / 0.97° / 0.658 | 0.419 / 0.97° / 0.612 | — |
| 0–499 幀 | 0.556 / 1.15° | 0.438 / 1.17° | |
| 500–999 | 0.327 / 0.67° | 0.383 / 0.63° | |
| 1000–1499 | 0.637 / 0.77° | 0.486 / 0.87° | |
| 1500–1999 | 0.458 / 0.31° | 0.424 / 0.30° | |
| 2000–2499 | 0.514 / 0.44° | 0.376 / 0.45° | |
| 2500–2999 | 0.707 / 0.77° | 0.727 / 0.83° | |

- 逐幀挑來源的結果：12 m 1612 幀、15 m 1018、18 m 202、25 m 179（直路用 12 m，路口、彎道退到更遠）。
- 追蹤器：3011/3011 幀成功、0 失敗；對 I_ns xy 0.543 m（中位 0.42、最大 1.96）、yaw 1.19°。EKF 更新 2997 幀、0 拒絕。
- 因果的代價：比離線無 GNSS 版（0.437）差 0.26 分，xy 多 0.15 m、yaw 多 0.53°。yaw 對 I_ns 分段偏 −0.9～+0.7°：I_ns 的 yaw 來自 25 m 配準，這套的 yaw 用行進方向定水平，兩者分段不一致，誰對不知道（Track1 排行榜只證實整段平均差 0.09° 以內）。
- 全 25 m 對照看起來比 hybrid 好，是假真值的偏向（I_ns 本身用 25 m 配準做的）；Track3 的排行榜證明直路 12 m 較準（25 m 0.78 → 12 m 0.50），所以主候選仍是 hybrid。

## 遇到問題

- `MemoryError`、`分頁檔太小`或 `can't start new thread`：配準時記憶體不夠，`PROCS=1 bash run_track1_causal.sh ...`，而且不要同時跑別的重活。
- Windows 自己一行一行跑時出現 `UnicodeEncodeError`：先設 `PYTHONUTF8=1`（PowerShell：`$env:PYTHONUTF8=1`），或改用 `python -X utf8 ...`。
- 追蹤那步每 100 幀印一行 `fails`、`v`（車速）；v 應該平順地從 2 m/s 升到 10 m/s 左右，若變負或亂跳就是追丟了（只看 fails 看不出來）。
- 每支輸出腳本都會在 CSV 旁邊寫 `.diag.json`；`scripts/compare_csv.py 候選.csv --ref candidates/track1_I_ns.csv` 可以算對假真值的數字（全段、前 30%、每 500 幀）。
- `tests/` 是函式庫的單元測試：`pip install pytest` 後在這個資料夾下跑 `python -m pytest -q`（23 個）。

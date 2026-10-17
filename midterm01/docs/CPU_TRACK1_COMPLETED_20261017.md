# CPU Track1 完整結果與讀取權限（2026-10-17）

本次讀取既有 CPU 結果並修復單一 run 的讀取權限，沒有重跑定位、修改資料、停止容器或改動 TXT 內容。後續開發／測試改用 [GPU 主工作區](../../midterm01_gpu/README.md)。

## 既有結果核對

| 項目 | 實測 |
| --- | --- |
| CPU 容器路徑 | `/root/outputs/runs/first_test_9vBOV7le/track1_n0_xLOSykY4` |
| 主機路徑 | `/home/ricky/sdc_ros2/midterm01/outputs/runs/first_test_9vBOV7le/track1_n0_xLOSykY4` |
| 程式與參數 | CPU PCL NDT，`max_frames=0`、`ply_scale=3.0`、RViz／keep_alive／全域 map 發布關閉；完整命令在 run 的 `command.sh` |
| 結果 | `result.txt` **3011 行**，每行恰為 12 個有限數值 |
| 程序狀態 | `exit_code.txt=0`；log：`Finished: 3011 poses, 0 failed alignments.` |
| 耗時 | log `elapsed_seconds=14813.444`（約 4 小時 6 分 53 秒） |
| 現有檔案 | `command.sh`、`exit_code.txt`、`result.txt`、`run.log` |
| CSV／frame manifest | 該完整 run 未見 CSV 或逐幀 manifest；旁邊單幀 run 有 `frames.csv`，不能冒充本完整 run 的 manifest |
| CPU Track2 | 本次 `first_test_9vBOV7le` 下未見完整 Track2 run；這不代表其他位置不存在 |

TXT SHA256（修復前後一致）：

```text
58b5fcd07e948ac278a501d609c0ce6013eff13e43034eb0246c612104e36bd6
```

此次只核對輸出筆數、有限值、命令、退出碼、log 與檔案雜湊，未重新驗證全部 PLY／timestamp 順序、矩陣正交性或定位真值精度。0 次配準失敗不能推論已超過助教 baseline；本次未取得 CPU 對應 Kaggle 成績。GPU 舊 CUDA D2D 與新版成績也不能記成此 CPU run 成績。

## 空目錄的原因與修復

Docker inspect 確認 CPU `/root/outputs` 掛載主機 `midterm01/outputs`，GPU 同名路徑則掛載 `midterm01_gpu/outputs`。在錯誤容器／主機 `/root` 找結果，與主機帳號無權限，是不同問題。

修復前，父目錄 `first_test_9vBOV7le` 是 ricky:ricky／700，指定子目錄是 root:root／700。ricky 能進父目錄，無法列出 root 子目錄；主機 `ls` 實際回報 Permission denied，容器 root 能列出 4 個檔案，並非真的沒有輸出。

在 Ubuntu 主機執行：

```bash
docker exec --user root sdc_midterm01 chmod -R a+rX \
  /root/outputs/runs/first_test_9vBOV7le/track1_n0_xLOSykY4
```

修復後子目錄 root:root／755、TXT 等檔案仍為 644。主機帳號 ricky 已可 `ls`、讀取並核對 3011 行；沒有加寫入權限、改擁有者或開放主機 `/root`。詳細用法與 CPU TXT 另存 CSV 見 [GPU 操作手冊](../../midterm01_gpu/doc/MANUAL_WORKFLOW.md#cpu-結果看不到與-chmod)。

本次檢查證據保存在 `midterm01_gpu/outputs/runs/docs_gpu_handoff_20261017_hj0e8kf9/`：`inspection_before.json`、`permission_after.json`、版本狀態與文件驗證。這些輸出遵守 Git 忽略規則，不會隨 Git 同步。

# 24｜三相機跨場景偏差驗證與完整實驗入口（2026-10-18）

**驗證完成，尚未通過。** 單一共享尺度不能穩定解釋不同場景；目前不能把三相機候選升為推薦最佳。已補齊可由使用者手動執行的完整重建／定位與中斷續跑，指令見下方。完整序列與 Kaggle 未由 agent 執行。

既有前相機最佳仍為 `sensor_guided + heading_rts + directional`，已評分 Track1 **1.15275**、Track2 **0.30576**。兩份最佳 CSV SHA256 保持不變。上一輪相鄰影格的小測見 [校正三相機紀錄](23_CALIBRATED_RIG_RESULTS_20261018.md)；新結果不回填改寫舊尺度 0.64738877 或 merged PLY 的 0.87124954。

## 要驗證的假說與設計

假說：若主要問題是固定的深度倍率，跨兩條 Track 共用一個尺度，應該也能讓相互分離的保留場景與 metric 三角化一致。若只有少數點或移動車輛支持這個倍率，或不同幾何參考給出不同尺度，就不能稱為通用校準。

- 沿原 GNSS／IMU prior 的路線累積距離，在 10–90% 的八個預定分位各取一個移動場景，每條 Track 八個，共 **16 groups／48 張重建影像**。只依運動與距離選擇，避開上一輪 groups 超過 80 幀，不依影像品質、map 殘差或分數挑選。
- 依分位順序交替 fit／holdout，各八個；holdout 距最近 fit 場景沿路線約 **90–143 m**，不再使用相鄰影格當泛化證據。這是沿路線距離，不保證迴圈道路的空間距離相同。
- 每個場景另讀取 **8 幀後的前相機**（約 0.3 s），共 16 張額外原始影像，檢查靜態剛體投影與較長 baseline 的尺度。運動來自相同 sensor prior，不拿 GNSS 當真值。
- 仍使用同 checkpoint、448×294、bf16、固定 rig／高度與相同 raw fisheye 校正。保留原單前相機／預測三相機／給定外參三相機推論，共 48 次模型推論。

工具：[verify_rig_bias.py](../scripts/verify_rig_bias.py)。SIFT mutual ratio 0.8，stereo 錨點沿用至少 2° 視差、Sampson ≤2 px、兩 view 深度比相差 ≤25%；新增同位置去重，避免 SIFT 的多個方向被誤當獨立幾何點。這項 audit 對共享尺度影響很小（0.67463→0.67398），但錨點數更可信。

每個 fit group 至少三個錨點，先取其 log-depth-ratio 中位數，再對 group 等權估一個共享尺度。holdout 不參與估計。預定工程檢查為 fit／holdout 各至少六個支持 group、每 Track 各至少三個，以及 holdout group 中位數相對差異的 median≤10%、P90≤20%；這些是試驗一致性門檻，並非正式精度規格或校準後 covariance。

## 結果

| 診斷 | Stereo，跨相機 baseline | Temporal，前相機跨時間 baseline |
| --- | ---: | ---: |
| 共享深度尺度候選 | **0.67398047** | **0.79228787** |
| 有至少三錨點的 fit groups | 7／8 | 5／8 |
| 有至少三錨點的 holdout groups | 5／8 | 5／8 |
| 支持 holdout groups 的相對差異 median | **18.2%** | **15.5%** |
| 支持 holdout groups 的相對差異 P90 | **30.6%** | **49.2%** |
| group bootstrap 尺度 95% 區間 | 0.657–0.770 | 0.662–1.105 |
| coverage／一致性檢查 | 未通過 | 未通過 |

百分比描述三角化深度的一致性差異，**不是定位誤差或 Kaggle 分數**。統計只包含足夠錨點的 holdout groups，不能代表全部場景。bootstrap 只描述這批 group 的離散程度，不包含所有標定／GNSS 系統誤差。

跨相機 stereo 錨點投影到八幀後，沒有足夠點通過 ≤4 px 的靜態一致性檢查，無法估出可靠的 static-stereo 共享尺度。**不能因此斷言所有點都在移動**：錯配、stereo 校正偏差、運動 prior／body→camera 契約及深度不確定度也會導致失敗；4 px 門檻沒有為了增加通過數而放寬。

為釐清運動模型，另以全部前相機 temporal 配對檢查已知 motion 的極線幾何及 image-only essential rotation。大多數場景的 Sampson median 約 0.2–1.3 px，一個約 2.3 px；visual／sensor 相對 rotation 差約 0.1–0.6°。因此 sensor motion 並非全面失效，但也不足以認證 body lever arm、曝光時刻或 rig 外參已正確。前向運動、動態物體與少量視差都會影響 image-only 結果。

兩種尺度參考相差約 **17.6%**，並且跨場景離散大。這輪的可解釋結論是：**不能只把舊 0.647 改成新 0.674，就宣稱偏差已修好。** 0.674 僅凍結為 fit-only 的完整實驗候選，`candidate_final.json` 明確記錄 `validation_passed=false`；0.792 不作為另一個排行榜搜尋候選。

若後續要改善，優先確認相機／車體 frame 與外參、使用跨時間一致的靜態特徵，並研究跨幀共同尺度／深度不確定度。不要依每個 Track 或每幀的分數分別乘一個倍率。

![偏差驗證圖](../outputs/runs/rig_bias_validation_2610181142/bias_validation.png)

## 完整一輪的指令

**下列會重做兩條 Track 的三相機影像重建，再定位，不是已證明更準的版本。** 若目的是保留目前最佳精度，仍使用操作手冊原有的前相機完整命令。

Ubuntu 主機執行：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
bash scripts/dev.sh up

# 只核對 3011／3299 幀的設定、來源與模型，不推論、不定位、不建立完整 run。
bash scripts/run_three_camera_full.sh --dry-run

# 手動啟動一輪：Track1 → Track2，均使用相同凍結的實驗尺度。
bash scripts/run_three_camera_full.sh
```

終端會先印出 `Run: .../outputs/runs/three_camera_full_YYMMDDHHmm`。完成後：

```text
<Run>/result_1.csv      # 表頭 + 3011 筆 idx,x,y,yaw
<Run>/result_2.csv      # 表頭 + 3299 筆 idx,x,y,yaw
<Run>/track1/          # 逐幀 PLY、camera_id/confidence、來源／時間／pose／hash、定位索引
<Run>/track2/
```

中斷後，使用終端印出的**同一個** run：

```bash
bash scripts/run_three_camera_full.sh --resume outputs/runs/three_camera_full_YYMMDDHHmm
```

首次建立 run 時凍結 candidate；續跑核對模型、程式、Python 套件版本、校正、index、prior、原始影像及已完成輸出的 hash。每幀先寫 temporary directory，再原子提交；已完成的幀不重新推論。完成的定位／CSV 也會重用；定位失敗保留 log，可在重建完成狀態重新啟動定位。資料或設定不一致會直接失敗，不靜默混合結果，也不在 CUDA 錯誤時切換 CPU。

預設只保留選取後的 PLY、camera labels/confidence 與完整來源／相機 pose，不保存每幀 dense depth／rectified JPG，以控制磁碟用量。低階 [run_rig_full.py](../scripts/run_rig_full.py) 可明確加 `--save-depth`，但會增加大量儲存量。小測重建每組約 1.2–1.9 s，完整 6310 groups 屬數小時等級，定位另計；不同場景點數會影響 CPU 取樣耗時，不能只用 0.6 s 的神經網路推論時間估整輪。

## 全序列邊界與測試範圍

原 motion trajectory 只有前相機 bag 時間的 prior。第一個 header 比 prior 起點早約 9.6／12.0 ms，最後右相機 header 比終點晚約 11.8／13.9 ms。完整重建以**明確上限 30 ms 的等速度／等角速度外推**處理這些邊界，每個受影響相機記錄外推秒數，超界直接失敗；沒有靜默夾到端點。原 `PriorTrajectory` 預設仍不允許外推，只有 batch 入口明確啟用；定位使用的原 prior CSV 不變。

已實測序列開頭兩幀、Track2 結尾兩幀、完成後續跑及部分完成的恢復，並驗證有限／完整 pose 與短段 CSV。部分完成狀態是從私有測試輸出保留第一個已提交 frame 人工建立，用於測試恢復分支，不冒充真實斷電事件。兩幀診斷 CSV 名為 `diagnostic_subset.csv`；只有完整 frame set 才匯出 `result_N.csv`。測試還檢查原始影像／輸出損壞、尺度／模型／設定不符的拒絕。**沒有在本次自動跑完整序列。**

本機證據：`outputs/runs/rig_bias_validation_2610181142/`，包含 `plan.json`、`reconstruction/`、`bias_final/bias_report.json`、`candidate_final.json`、`full_command_dry_run.log`、各 smoke／resume log、測試紀錄與來源快照。早期未去重／較少 temporal 診斷的結果也保留，正式判斷以 `bias_final` 為準。outputs、模型及資料不進 Git，交接須另行保存。

# 12｜GNSS／IMU 融合改進與手動完整測試

2026-10-18 現況補充：本文件記錄 heading_rts＋geometry off 的融合階段。最新已評分主流程在其後加入 directional，分數 **1.15275／0.30576**，使用者 Team 3 新截圖排名第 4／第 2 名；見 [現行整合算法與改進方向](20_CURRENT_ALGORITHM_AND_ACCURACY_20261018.md)。以下保留融合階段原始分數、測試與命令，不將後續 directional 成績歸到 motion_rts。

更新：2026-10-17。新增的是**定位前的感測器先驗融合**，後面仍接既有 CUDA 最近鄰＋CPU robust 四自由度配準。使用者已手動完成 `heading_rts` 兩條完整 Track 並提供 Kaggle 截圖：**Track1 1.16951、Track2 0.31967**，截圖當下分別第 3／第 1 名。完整改動、來源 SHA256 與進步幅度見 [本輪成績紀錄](14_HEADING_RTS_RESULTS_20261017.md)。`motion_rts` 仍只有短測，沒有其 Kaggle 成績。

## 評估與選擇

| 方法 | 適合目前專案的程度 | 本次決定 |
| --- | --- | --- |
| 一階 IIR／一般 FIR | 可以平滑小修正，直接濾最終位置會有延遲 | 保留既有 correction IIR，不再疊加 |
| Kalman + RTS | 離線可用前後觀測估計慢變狀態，符合目前資料處理方式 | 實作有品質門檻的 heading bias smoother；另提供位置／速度 smoother |
| EKF／ESKF | 適合姿態、速度、IMU bias 的非線性即時融合 | 原始加速度的重力／軸／外參契約未確認，暫不直接積分 |
| UKF | 能處理非線性傳播，但不會自動解決時間、外參或相關誤差 | 此次狀態模型是線性的，沒有必要增加複雜度 |
| Factor graph／batch optimization | 後續整合原始感測器、相對運動與地圖因子的方向 | 尚未實作；先完成較小、可單獨比較的融合改進 |

RTS 是 Rauch–Tung–Striebel 離線後向平滑：Kalman forward pass 先累積資訊，再用較晚觀測修正較早狀態。它不代表能保證提升分數；運動模型與 Q/R 設定不合適時仍會扭曲答案。

新增三個明確選項，benchmark 與前處理的預設仍保留舊行為：

| benchmark 選項 | XY 先驗 | heading 校正 |
| --- | --- | --- |
| `--sensor-fusion fixed`（預設） | 舊版 5 點 GNSS Savitzky–Golay＋線性內插 | 舊版整包固定 bias |
| `--sensor-fusion heading_rts`（本輪優先候選） | 與 fixed 完全相同 | 品質篩選、robust Kalman＋RTS 時變 bias |
| `--sensor-fusion motion_rts`（額外消融） | 原始 GNSS 的位置／速度 Kalman＋RTS | 與 heading_rts 相同 |

省略 `--profile` 仍是 `legacy/cuda_ndt`；新融合必須明確搭配 `--profile sensor_guided`。新功能沒有更換 legacy profile 或既有 sensor_guided 預設，因此歷史命令仍可重現。目前推薦明確選 `heading_rts`，它是本輪已有兩條 Track 分數的版本；尚未以完整消融比較證明它優於所有其他模型／設定。

## heading_rts 的實際資料流

1. 依整數 bag timestamp 取得原始 GNSS XY、IMU quaternion。以第一筆 GNSS 作時間原點，固定區塊邊界，不因這次只測前 50 幀而改變 heading 模型。
2. 將 GNSS 分成**互不重疊的 2 秒區塊**。用 robust 加權線性迴歸估 XY 速度／course，避免用相鄰位置微分放大雜訊。每區塊至少 5 筆、時間跨度至少 1 秒，且 IMU 必須涵蓋。
3. 只使用速度 >5 m/s、區塊內 IMU yaw rate 最大值 <2°/s、yaw 跨度 <4°、course 與 IMU 方向差 <15° 的觀測。多筆 GNSS 大離群點的區塊也拒絕；倒車的大方向差不當作校正。
4. 觀測為 `z = wrap(course - IMU_yaw)`，狀態為慢變校正 `b`：`b[k+1] = b[k] + w`。以 GNSS 區塊幾何、殘差與標準差下限估觀測 R；過大的 innovation 增大 R、降低拉力，不把每個區塊當等可信。
5. Kalman forward pass＋RTS backward pass 後，限制 bias 在 ±10°、變化不超過 0.2°/s；將 bias 內插到影像 timestamp。只限制校正狀態，不限制 IMU 觀測到的真實轉彎速度。
6. `R_fused = Rz(b(t)) × R_imu_slerp(t)`，保留 IMU roll／pitch 與合法旋轉。只在可靠 heading 區塊更新 bias；低速／轉彎時仍保留 IMU 姿態與模型估出的校正，不把 GNSS course 強行當車頭方向。
7. 少於 3 個可靠區塊時，**明確退回 fixed bias**，原因記在 JSON。可靠區塊範圍外保持邊界 bias，模型不確定度隨距離增長，不無限制外推漂移。

這是在既有 GNSS/map/source frame 相容假設下融合方向資訊。course 是位移方向，不是車體 yaw 真值；側滑、GNSS 系統誤差或天線參考點仍可能影響結果。兩種新模式皆不是完整慣性導航系統，沒有積分原始 gyro／加速度，也沒有估計完整 IMU bias 或外參。

## motion_rts 的位置模型

狀態為 `[x,y,vx,vy]`，使用實際 dt 的 constant-velocity 模型與 white acceleration process noise。原始 GNSS 每筆只更新一次；額外影像 timestamp 只有預測、沒有虛構的量測更新。計算移到局部原點以避免大座標消去誤差；Joseph covariance update＋RTS 反向平滑，GNSS innovation 太大時降權。

此選項替換 GNSS XY 的 SG／線性內插，Z 與姿態流程保持既有契約。它能避免直接位置 IIR 的等速滯後，但模型設定過強仍可能影響加減速或轉彎，因此先當額外消融候選。GNSS course 與 XY 來自相同資料，不宣稱兩者的誤差獨立；目前沒有將該共變異數餵給 map 配準，也沒有把已融合的 map pose 再當獨立 GNSS 融合。

## 參數與可追查輸出

| benchmark 參數 | 預設 | 意義 |
| --- | ---: | --- |
| `--fusion-gnss-std` | 0.3 m | course 品質下限；motion_rts 的 GNSS 觀測標準差 |
| `--fusion-xy-accel-psd` | 4 m²/s³ | motion_rts 的 white acceleration PSD |
| `--fusion-bias-rw-deg` | 0.12 °/√s | bias random walk 標準差，較大時更容易隨時間變化 |
| `--fusion-course-window` | 2 s | course 的不重疊時間窗，CLI 至少 1 秒 |

以上為通用的初始實驗設定，**不是由真值或 Kaggle 校準出的共變異數**。完整品質門檻也會寫入每輪 `sensor_prior.json` 的 `sensor_fusion.config`；模型版本為 1。

每輪仍用新的唯一 `outputs/runs/`，新增檔案如下：

| 檔案 | 用途 |
| --- | --- |
| `sensor_preparation_params.json` | 實際 fusion mode 與 CLI 設定，不混入未知 ROS 參數 |
| `sensor_prior.json` | 模式、所有設定、接受區塊數、降權數、fallback、bias 範圍與來源雜湊 |
| `sensor_prior.fusion.csv` | 逐幀 bias、模型標準差、fixed bias；motion_rts 另有速度、XY 模型標準差 |
| `sensor_prior.heading_blocks.csv` | 每個感測器區塊的樣本數、course 速度、方向差、觀測標準差及接受／拒絕原因；time_s 相對第一筆 GNSS |
| `pipeline_sources/` | 本次 benchmark／先驗前處理／fusion 模組的來源快照，包含尚未 commit 的新檔 |
| `pipeline_source_sha256.json` | 上述程式的 SHA256 |
| `validation.json` | 新增 sensor_fusion 與其診斷，核對 fusion 逐幀 timestamp／筆數／有限值 |

原本 `sensor_prior.csv` 的八欄契約不變，C++／CUDA 程式不需修改或重編。CUDA runtime 出錯仍直接失敗，不轉 CPU。既有的每幀 map 目標檢查與 correction 上限繼續適用於新先驗；其通過只表示內部契約通過，不是精度證明。

為保留比較依據，JSON 舊欄位 `heading_bias_deg` 在新模式中仍記 **fixed 參考值**；實際時變校正看 `sensor_prior.fusion.csv` 的 `bias_deg`，範圍看 `sensor_fusion.bias_range_deg`。`heading_bias_deg_role` 會明示此差別。

## 本次已完成的驗證

數值測試共 11 項（含既有 prior 與濾波工具測試）通過。新增測試涵蓋：已知慢變 bias＋方向離群值、bias 範圍與變化率限制、低速／轉彎／反向拒絕、yaw 跨 ±180°、roll/pitch/Z 保留、奈秒時間、短／長查詢的 heading 一致性、已知等速運動無整段位置滯後、GNSS spike 降權，以及插入 frame timestamp 不增加 GNSS 量測次數。兩條 Track 各前 50 個 prior 的 fixed 模式與修改前程式逐值完全相同。

開發階段的真實資料短測如下；當時 Codex 只執行這些 50 幀測試。使用者之後手動完成的 3011／3299 幀結果另見 [完整成績紀錄](14_HEADING_RTS_RESULTS_20261017.md)：

| Track／模式 | 幀數 | 退出碼／驗證錯誤 | 接受 map 更新 | 感測器前處理＋ROS（s） | map cache |
| --- | ---: | --- | ---: | ---: | --- |
| Track1 heading_rts | 50 | 0／空 | 50 | 6.388 | 明確指定、SHA256 核對通過並命中既有快取 |
| Track2 heading_rts | 50 | 0／空 | 50 | 8.366 | 停用 |
| Track2 motion_rts | 50 | 0／空 | 50 | 8.521 | 停用 |

兩條完整 bag 的 heading 區塊分別接受 47／48 個；這是離線感測器讀取，不代表處理了整條影像序列。另在全路線分散的 13 個 timestamp 作**感測器先驗診斷**，Track2 bias 約由 +3.644° 變到 −0.098°；沒有進行該 13 幀影像配準或全序列定位，也不能把這當誤差改善。

證據：[本次驗證目錄](../outputs/runs/sensor_fusion_dev_4g2k458n)、[短測 JSON](../outputs/runs/sensor_fusion_dev_4g2k458n/smoke_summary.json)、[輸入診斷圖](../outputs/runs/sensor_fusion_dev_4g2k458n/heading_bias_samples.png)。各短測原始 run 路徑、參數、frame、來源快照、時間與完整 validation 由 JSON 連到原 run。Track1／Track2 快取條件不同，不作速度倍率比較；合成測試也不是競賽精度測試。

改進前的文字回報為 Track1 1.22369、Track2 0.56063；新截圖的 Track2 前最佳顯示 0.56083，差異與來源均保留在 [成績紀錄](14_HEADING_RTS_RESULTS_20261017.md)。本輪新最佳為 **1.16951／0.31967**，已另存新的 CSV／provenance，沒有覆寫舊結果。

## 完整測試：由使用者在 Ubuntu 主機手動執行

先跑 heading_rts。兩條命令依序完成；Track1 是 **3011** 張，Track2 是 **3299** 張，不使用 3000 或 `--frames 0`。這次只改 Python，現有 GPU workspace 已建置時不需要重編 C++／Docker。

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu

# GPU 容器已啟動可略過；只操作 GPU project。
bash scripts/dev.sh up

python3 scripts/benchmark.py --track 1 --frames 3011 --profile sensor_guided --sensor-fusion heading_rts
python3 scripts/benchmark.py --track 2 --frames 3299 --profile sensor_guided --sensor-fusion heading_rts
```

每次終端印出 `Run: /home/ricky/sdc_ros2/midterm01_gpu/outputs/runs/...`。先檢查該 run 的 `validation.json`：`returncode=0`、`output_complete_and_finite=true`、`validation_errors=[]`、正確幀數與 `sensor_fusion=heading_rts`。完整候選是同目錄的 **`diagnostic.csv`**，欄位 `idx,x,y,yaw`，yaw 為弧度；保留各自 run，不覆寫目前最佳答案。這兩份完整測試與 Kaggle 提交由使用者執行。

欲獨立比較 GNSS 位置模型，再手動以相同命令改成 `--sensor-fusion motion_rts`；回舊版則用 `--sensor-fusion fixed`。不要在同一輪同時改其他配準權重；以實際完整結果、Kaggle 分數與對應 CSV SHA256 決定是否接受新版本。

可直接貼上的命令另見 [13_run_sensor_fusion.txt](13_run_sensor_fusion.txt)。若要重做短測，幀數改 50；若只做數值契約測試：

```bash
docker exec sdc_midterm01_gpu python3 -m unittest discover -s /root/scripts/tests -v
```

## 後續建議與方法參考

本輪分數已回報，後續保留已評分的 heading_rts 作對照，再依誤差類型判斷要調 heading 模型、位置模型，或回頭確認 frame／時間／尺度契約。更完整的下一步可研究原始 GNSS＋IMU＋獨立視覺相對運動＋map 的 factor graph，並處理 course／GNSS 的相關性；不直接把現有 posterior 當成一筆新的獨立觀測。

- [GTSAM 官方 Factor Graph 教學](https://gtsam.org/tutorials/intro.html)：不同感測器及相鄰狀態關係可表為因子，共同最佳化；本次沒有安裝或整合 GTSAM。
- [SciPy SLERP 文件](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Slerp.html)：以 quaternion 做旋轉內插；內插本身不等於量測濾波。
- [前一輪濾波診斷](11_FILTER_ANALYSIS_20261017.md)：直接濾最終位置造成滯後、慢變 heading 差值及實驗依據。


## 2026-10-18 三相機的先驗重用

[原始三相機校正流程](23_CALIBRATED_RIG_RESULTS_20261018.md) 直接重用原完整 heading_rts 的 sensor prior，以相機各自的 header 時間插值 XYZ／Slerp 旋轉，再轉至前相機 bag reference。定位比較逐位元核對相同 prior CSV；沒有使用原 map posterior，亦未改動 fusion 模型。body→camera lever arm、header 曝光／硬體時鐘語義仍未完整標定。

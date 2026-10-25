# 54｜用外參預覽 IMU 各軸訊號

`scripts/preview_imu_extrinsic.py` 讓你修改 IMU 安裝外參，檢視轉換到前視 PLY 座標後的加速度與角速度。它只分析 bag 訊號，不執行 Track 定位。

## 修改參數

在 Python 檔案最上方設定：

```python
IMU_RPY_DEG = (0.0, 0.0, 0.0)  # 固定軸 X/Y/Z：roll、pitch、yaw，單位度
IMU_XYZ_M = (0.0, 0.0, 0.0)   # IMU 原點在 PLY FLU 的 x/y/z，單位公尺
COMPENSATE_LEVER_ARM = True
ALPHA_WINDOW_S = 0.10
```

轉換方向是 `p_ply = R_ply_imu p_imu + t_ply_imu`，其中 `R = Rz(yaw) Ry(pitch) Rx(roll)`，PLY FLU 為 X 前、Y 左、Z 上。`IMU_XYZ_M` 是 **IMU 原點的位置**；從 IMU 原點指向 PLY 原點的向量是 `-IMU_XYZ_M`。

角速度依 `R_ply_imu` 轉到 PLY 座標；平移不改變角速度。加速度先旋轉，再視設定將 IMU 原點的 specific force 換算至 PLY 原點：

```text
omega_ply = R_ply_imu omega_imu
f_imu_in_ply = R_ply_imu f_imu
f_ply = f_imu_in_ply
        - alpha_ply × IMU_XYZ_M
        - omega_ply × (omega_ply × IMU_XYZ_M)
```

最後兩項是角加速度與向心加速度造成的 lever-arm 修正，只在 `COMPENSATE_LEVER_ARM=True` 時使用。剛體上不同原點的加速度關係可參考 [Drake SpatialAcceleration](https://drake.mit.edu/doxygen_cxx/classdrake_1_1multibody_1_1_spatial_acceleration.html) 的 spatial acceleration shift。`alpha_ply` 由目前與過去的 gyro 樣本，以 `ALPHA_WINDOW_S` 秒視窗線性回歸估計斜率；不使用未來樣本。時間窗內至少需要兩筆 gyro 樣本才能估計角加速度；時間窗內樣本不足時，若平移非零且啟用 lever-arm 修正，該筆換算後加速度留白並標記無效。設 `COMPENSATE_LEVER_ARM=False` 可只看旋轉後的三軸訊號。

加速度維持 ROS IMU 慣例的 specific force，包含重力效應；工具不代扣重力，也不會自動估 bias 或旋轉角。alpha 視窗是數值微分設定，不是 IMU 規格或校正結果。gyro 噪聲會影響微分後的角加速度，因此比較平移外參時也要看 lever-arm 分量，不要把微小差異當成可靠標定。

## 執行一次試驗

先修改上方參數，再於 Ubuntu 主機執行：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
python3 scripts/preview_imu_extrinsic.py
```

主機端入口會自動建立新的 `track03_test_YYMMDDHHmm` run、凍結程式與外參，並使用既有容器執行快照。要試下一組數值，先修改這個 Python 檔案頂端的參數，再執行相同命令；每組結果會留在新的 run。此工具不需啟停容器，也不會修改定位所用的 `track3/imu_parameters.py`。

## 讀取輸出

run 內的 `imu_transformed.csv` 同列保存原始訊號、旋轉後訊號、目標原點訊號、角加速度與 lever-arm 項，並記錄每筆加速度有效旗標。訊息按 bag arrival timestamp 排序，程式確認 arrival 與 header timestamp 都嚴格遞增；不會依 header 重新排序。header timestamp 用於角加速度回歸。時間戳保持整數奈秒；圖表 t=0 對齊第一張 PLY。CSV 保留 bag 中 11,307 筆 `/imu/data` 樣本，最後兩筆晚於最後一張 PLY，僅作訊號預覽。

- `analysis/00_comparison.png/pdf`：六軸前後對照；加速度依設定疊上原始、旋轉後與目標原點訊號，角速度疊上原始與轉換後訊號。
- `analysis/01_acceleration.png/pdf`：加速度各軸原始、旋轉後與目標原點訊號對照；各 lever-arm 補償分量可在 CSV 查看。
- `analysis/02_angular_velocity.png/pdf`：角速度各軸轉換前後比較。
- `parameters.json`：本輪外參與轉換設定。`analysis/metrics.json` 保存數值摘要；`analysis/manifest.json` 保存來源與輸出 SHA256。

## 判讀界線

IMU 自身的 XYZ 訊號轉到假設的 PLY 軸後，可供檢視軸交換與方向是否符合安裝假設。靜止時 specific force 方向最多提供重力傾角線索，不能判定 yaw；未確認靜止與 bias 前，也不要把原始 gyro 均值當成 gyro bias。單獨的 IMU 訊號不能唯一求得六自由度 IMU→相機外參；比較圖呈現的是你輸入的假設所導出的訊號，不是外參真值或定位精度。

原始量測來自唯讀 `Track3.db3` 的 `/imu/data`，frame 為 `imu_link`。時序按 bag arrival timestamp 排列；header timestamp 只用來估計角加速度，且也會檢查為嚴格遞增。不使用 orientation 或 GNSS。外參與轉換設定記在 `parameters.json`，來源與輸出 SHA 記在 `analysis/manifest.json`。

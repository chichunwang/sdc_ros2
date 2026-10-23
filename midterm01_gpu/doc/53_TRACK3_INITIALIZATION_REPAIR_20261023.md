# Track3 初始化失敗診斷與因果 v2 修正設計（2026-10-23）

## 先看結論

v1 full `track03_full_2610231521` 由使用者回報 82.17。3142 幀檔案雖完整，map 最終採用 0 幀、速度全程為零、輸出 XYZ 停在初始位置；這是定位啟動失敗，不是成功追蹤。細節見 [v1 full 唯讀診斷](../outputs/run/track03_test_2610231541/diagnosis.json) 與 [52｜因果 Kalman 與尺度配準](52_TRACK3_CAUSAL_KALMAN_SCALE_20261023.md)。

現在有一個與失敗相符、但尚未證明為唯一根因的幾何線索：前三幀 source 地面在官方初始姿態下，與局部 map 地面方向差約 18.5–18.9°，超過 v1 初始化旋轉限制 15°。靜態檢查只做 PCA，沒有求新 pose。v2 已完成合成機制測試，但尚未做真實 Track 定位；不能據此稱定位已修復或精度改善。

## 幾何診斷顯示什麼

[首三幀靜態幾何紀錄](../outputs/run/track03_test_2610231550/static_geometry.json) 記錄官方初始矩陣的 RPY 約為 roll 14.15°、pitch 14.92°、yaw 145.68°。局部 map 道路法向接近 map +Z；在官方初始姿態下，source 地面法向相對 map 地面方向差為：

| PLY idx | PCA 評估尺度 | 有效 source normal 點數 | 地面方向差 |
| ---: | ---: | ---: | ---: |
| 0 | 3 | 2,029／8,835 | 18.61° |
| 0 | 9 | 8,096／8,835 | 18.69° |
| 1 | 3 | 2,037／9,418 | 18.54° |
| 1 | 9 | 8,628／9,418 | 18.73° |
| 2 | 3 | 2,024／9,509 | 18.74° |
| 2 | 9 | 8,761／9,509 | 18.89° |

尺度會改變 PCA 鄰域中點的實際公尺間距，因此在不同尺度下得到的有效法向點數不同。這個對照說明尺度會影響幾何估計品質；`9` 只是診斷時的一個評估尺度，**不是校準值、新預設或建議參數**。它也不表示把全域位置或 map 乘 9。

這些角度差使「初始姿態搜索範圍不足」成為合理待驗假說，但靜態地面 PCA 不等於 map 配準：source 的物理座標軸、相機／IMU 外參與 pose 參考點尚未完整標定，地面也不能保證處處水平。因此不能單憑這份診斷宣稱初始矩陣錯了或已找到唯一根因。

## v2 修正方向

以下是 v2 的修正設計與已核對實作行為；真實定位效果仍待使用者手動 full 評分。

1. **先擴大首次姿態初始化範圍。** 以初始旋轉標準差 8° 的三倍，即 24°，作初始化搜尋範圍。從 map 與 source 的訓練地面平面方向產生姿態 seed，並用初始高度提出尺度 seed。這些是首次 map anchor 的起始候選，不把未知車高當真值；首次 anchor 仍須通過 5 m 平移、24° 旋轉、NIS 與 Kalman 後驗限制。
2. **每個尺度候選都按它自己的實際公尺尺度重算幾何。** source 點乘候選尺度後再計 PCA normals／quality。CUDA cache 保留原始點、normal、ground 與 quality 四種物件的 identity 及 GPU tensors；不是額外儲存尺度欄位。
3. **避免只靠一小片重疊牆面取得看似很低的誤差。** v2 加入固定 source 全點到 training map 最近距離平方的覆蓋代價；距離先封頂 2.5 m，再平方，因此代價上限為 6.25 m²。失去對應的 source 點仍付封頂代價。這不改原目標門檻。
4. **首個可信 anchor 後再使用相鄰運動限制。** 首次 anchor 不套用相鄰運動 gate，但仍受 5 m 平移、24° 旋轉、NIS 與後驗限制；anchor 建立後，後續幀恢復運動 gate。收斂接受條件有兩條：原 `small_step`，或實際接受的小步加 cost 連續三次穩定；不是每個候選都必須連續穩定三次。碰到參數邊界的解不接受。若全序列 map 採用數為零，明確丟出 RuntimeError 且不印成功的 `Finished`。

GPU 仍使用 float64 幾何與 PCA，PCA eigen decomposition 每批最多 256；小型 solver 留在 CPU。上述改動是初始化與幾何判定的修正方向，不是精度改善證明。若一個 run 的 map 更新採用數為零，runner 應明確回報定位失敗，不輸出代表成功的 `Finished`。

## 驗證狀態與使用方式

截至本紀錄，59 項不同檢查通過：CPU 45 項（[cpu_tests.json](../outputs/run/track03_test_2610231550/cpu_tests.json)、[cpu_tests.log](../outputs/run/track03_test_2610231550/cpu_tests.log)）；既有 CUDA 幾何 8 項、新 source-cache 1 項（[cuda_tests.log](../outputs/run/track03_test_2610231550/cuda_tests.log)）；最終因果 CUDA 5 項（[cuda_runner_tests.json](../outputs/run/track03_test_2610231550/cuda_runner_tests.json)、[cuda_runner_tests.log](../outputs/run/track03_test_2610231550/cuda_runner_tests.log)）。混合 log 的 4 項因果 CUDA 測試也包含在最終 5 項中，合計只計一次；兩份 log 皆保留。測試均為機制或合成輸入，不是真實 Track 定位。

新增合成測試在已知尺度 8.7、帶噪且傾斜 20° 的虛構環境中，CUDA 結果與 CPU 對照最大差在 1e-5 內；合成 3／5 幀輸出前綴逐 byte 相同；zero-accepted case 明確失敗且不印 `Finished`。證據見 [合成 bootstrap 與 prefix](../outputs/run/track00_test_2610231601/)。[full dry-run](../outputs/run/track03_test_2610231550/full_dry_run.json) 顯示 v2、3142 幀、24° 初始化範圍及每層最多 40 次迭代；dry-run 不會定位。沒有真實 v2 短段或 full、新分數或精度改善證據。score <1.29 是目標，不保證達成。

短段用來查機制與故障，不是 full 的資格門檻；只有 full 提交會產生正式 Kaggle 分數。以下命令透過現行入口執行 v2，不重現 v1；完整定位由使用者手動啟動：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
bash scripts/dev.sh up
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 outputs/ml_runtime/venv/bin/python scripts/run_track3_causal.py --full --scale-mode estimate --initial-scale 3
```

`--initial-scale 3` 仍是起始數值，不是已校準尺度。完整定位與 Kaggle 提交由使用者手動執行；不依 82.17 反推尺度、不手動修改 CSV，也不對已產生 pose 做離線回修。

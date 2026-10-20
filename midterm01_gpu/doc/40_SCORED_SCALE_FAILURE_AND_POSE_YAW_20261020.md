# 40｜尺度與 pose_yaw 完整評分結果

日期：2026-10-20（Asia/Taipei）。使用者回報上一輪 Track1 **1.15167**，較最佳 **1.15008** 增加 **0.00159／約0.138%**，沒有升為最佳。使用者進一步指定：有想驗證的版本就直接提供 full 指令，由使用者計算並提交；不再等本地先證明勝出。

## 最新結果：pose_yaw 已評分且退步

使用者回報 Track1 `track01_full_2610200155`（`pose_yaw`＋front scale=3＋directional＋carry）分數為 **1.30507**，較 motion_rts 最佳 **1.15008** 增加 **0.15499／13.47645%**。這是已評分的負面結果；**停止推薦直接以 pose_yaw 取代原航向，也不再將本文件後述的 pose_yaw full 命令視為待執行方案**。原有 mode、機制檢查及命令只作實驗歷史紀錄，未據此更改程式、門檻或資料。

該 full run returncode=0，3011 筆輸出完整。`outputs/run/track00_test_2610200207/scored_comparison.json` 與 `evidence_sha256_unchanged.json` 記錄：frame manifest 與最佳相同、prior XYZ 數值完全相同、tilt 最大差 **4.44e-16 rad**；實際輸出相對最佳 XY 差 RMS **0.01213518857 m**、yaw 差 RMS **2.118744811°**。這些是版本差異，不是逐幀真值誤差。分數與 run 的關聯依使用者回報上下文；未獨立核對 Kaggle 上傳 SHA。單一 aggregate score 也不能分解分數變化中的平移與旋轉部分，且 yaw 變化可連帶改變 map 的 XY／Z 修正。

## 1. 這次分數與本機證據

本機來源為 `outputs/run/track01_full_2610200142`，實際 command／prior 為 `turn_heading_rts`、front scale=7.404260939740923、directional、carry。退出0、validation無錯、TXT／CSV／frame manifest／prior均3011筆；CSV表頭為`idx,x,y,yaw`，每列數值與保存矩陣相符。CSV實際存在，提交檔名是 **diagnostic.csv**。

| 項目 | 原最佳 | 本次已評分版本 |
| --- | --- | --- |
| run | `track01_full_2610191750` | `track01_full_2610200142` |
| 使用者回報分數 | **1.15008** | **1.15167** |
| fusion | motion_rts | turn_heading_rts |
| source scale | 3 | 7.404260939740923 |
| CSV SHA256 | `781db1124f4a9a3e59a1d4adfc7ac7e3c14ff18e09b4157901f2cd807cb4466a` | `75f96243d764d32eaaf71bcbc26976b99eb0830ffa28902481e3f7f81e71b8a7` |

分數來源是使用者截圖，run關聯依使用者IDE與上一輪full命令；截圖沒有上傳SHA，不能宣稱獨立核對Kaggle上傳bytes。新分數已記在0142的test_info.json，原CSV／TXT／manifest／prior沒有改寫。

兩版 frames.csv 逐byte相同，prior XYZ **數值完全相同**。實際輸出的版本差：XY向量RMS **3.707cm**、最大16.598cm；yaw差RMS **0.2758°**、最大1.5264°。Z差RMS約2.454m，但提交CSV沒有Z，不能把高度變化直接解讀成分數變差原因。

新資料幾何rank0／1／2／3／4為0／156／1428／1419／8；原最佳為0／2083／833／95／0。方向資訊增加仍未改善正式分數，這項反例保留。rank、接受率與map殘差不能代替ATE／yaw真值誤差。

## 2. pose_yaw 實驗動機與結果（歷史紀錄）

前一個已評分版本同時改尺度與航向，單一 aggregate score 無法嚴格分解兩項各自效果。因此當時提出回到已評分最佳 front scale=3、只改 yaw 來源的 pose_yaw 實驗。該實驗已完成；上方記錄其評分結果，原實驗動機不代表目前建議。

當時的已評分主流程用IMU quaternion，再由GNSS行進方向估慢變bias。實驗 `pose_yaw` 直接使用原 `/localizer_node/gnss_pose` 的yaw，保留同motion_rts XY、原Z、IMU roll／pitch、directional、carry；設計目的為單獨檢查heading來源假說，不把原course／IMU heading當成三份獨立量測相乘。完整評分後，此替換版本退步。

`pose_yaw` 當時已接入通用benchmark，主機／ROS的機制檢查、八個分散短段共384筆均已完成。來源有效性未獲獨立驗證；既有完整先驗的Track1 yaw改動RMS約2.117°是版本差，不是改善保證。隨後的 full 評分已顯示此版本退步，故保留以下機制與 run 路徑供回顧，不再以 full 評分判斷該假說。

本輪重用已完成的機制／短段證據，不為文件或同一模式重複執行無必要定位：

- `outputs/run/track00_test_2610200127/short_pose_yaw_runs.json`：8×48幀的實際run與命令。
- 同研究run的`short_summary.json`：frozen XYZ／來源／binary／solver參數核對及yaw來源消融。
- `direct_heading_tests.log`／`direct_heading_ros_tests.log`：branch cut、端點外推、無效quaternion、XYZ／tilt保留與prefix檢查。
- `scripts/direct_heading.py`：明確≤0.3s的角速度端點外推，不宣稱已校準covariance。

## 3. 原 Track1 full 命令（歷史，已執行）

以下是當時提供並已由使用者執行的命令，**不是目前推薦或待執行命令**。該 run 為 `track01_full_2610200155`，已完成評分；分數與核對資料見本文件開頭及 `outputs/run/track00_test_2610200207/`。

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
bash scripts/dev.sh up

python3 scripts/benchmark.py --track 1 --full \
  --profile sensor_guided --sensor-fusion pose_yaw \
  --cloud-source front --ply-scale 3 \
  --geometry-mode directional --height-mode carry
```

當時的輸出檢查要求為 `validation.json` returncode=0、`output_complete_and_finite=true`、`validation_errors=[]`、3011筆；該 run 已符合完整輸出要求並完成評分。

本段原先提及的 Track2 同來源對照命令只代表當時的測試設想；沒有在此記錄 Track2 已執行或已評分。pose_yaw 的 Track1 實測退步，不應將其外推成 Track2 結論。

這項失敗證據不改變已評分最佳，也不提供另一個航向版本的 full 命令。0.6 仍是目標，不能記為已達成。

## 4. 位置模型的已知限制與後續工作

在前述尺度比較的兩個版本（`track01_full_2610191750` 與 `track01_full_2610200142`）中，位置先驗完全相同，實際map水平修正仍只有幾公分。資料normal information消去自由Z後，相對既有prior precision的對角中位數，原版約[0.00492,0.00863,0.00568]，新尺度約[0.02332,0.02850,0.08675]（XY／scaled yaw）。這個診斷排除tangent floor及normal gain，**不是完整solver Hessian或已校準量測權重**；它與實際小修正共同顯示：增加可見幾何還沒有形成足以顯著改變水平軌跡的可靠觀測。

先前CV／側向速度／kinematic graph的留段負面結果也仍保留，不能期待再加強同源平滑就消除GNSS共同位置偏差。後續位置方向應研究能容納來源深度偏差的觀測模型，例如沿camera ray的距離不確定度，以及跨幀一致的獨立水平結構；這些是待驗證方向，**本輪沒有實作或宣稱成功**。

本次主研究證據在 `outputs/run/track00_test_2610200156/`：`score_audit.py/log`、`scored_comparison.json`、`score_context.json`、來源SHA與工作樹。分析重用既有checked_run，確實核對本機CSV；助理沒有自行啟動完整Track或Kaggle提交。

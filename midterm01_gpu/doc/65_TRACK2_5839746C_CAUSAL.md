# Track2：以 5839746c directional 模型製作因果版

> **歷史版本提示（2026-10-29）：**本文記錄先前固定尺度 3.0 的 v2，該版本已停止且不再是 Track2 現行入口。現行公尺尺度與配準流程請見 [66｜Track2 公尺尺度與配準](66_TRACK2_METRIC_SCALE_20261029.md)。本文保留 v2 的算法、輸出與當時證據供追溯；其已接受 map update 不足以證明尺度正確或點雲已對齊。

## 目前做法

目前 Track2 回到 `main` commit `5839746ce1e4fe59ed4980ae07cb5ceb3f27ddeb` 的 sensor-guided directional 配準模型與舊參數，將其離線感測器前處理改為逐幀前向更新。它每到一幀就用當下已取得的 GNSS、IMU 與地圖幾何估計並輸出 pose，不回頭改寫已輸出的 pose。

完整定位由使用者手動啟動：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
./scripts/dev.sh up
./scripts/run_track2.sh --full
```

命令會啟動 RViz。RViz 顯示設定在 [config/track2.rviz](../config/track2.rviz)，發布主題在 `/track2/*`，座標樹為 `map → track2_camera`。參數集中在 [track2/parameters.py](../track2/parameters.py)。本文件不包含 flowchart；這次也沒有製作 flowchart。

## 為什麼不能直接沿用舊結果

使用者指出 `5839746c` 的定位結果較好，但該版使用 RTS 等離線平滑方式。離線平滑會利用後來才到的資訊回修較早 pose，違反本 Track 的因果定位規則，因此舊版分數與輸出不能當作合規新版本的結果或成績。

本次保留舊版幾何模型、求解目標與參數設定，重做 GNSS／IMU prior（前置估計），讓它只使用當下可取得的資料。為保留來源與比較依據，選定的舊 C++ 實作及 SHA256 記錄放在 [track2/reference_5839746c](../track2/reference_5839746c/)；其中 `offline_reference/` 是唯讀參考，不會被定位程式匯入。新程式是 CUDA 向量移植，運算路徑與舊 C++ binary 並非位元組相同。

## 每幀使用哪些資訊

每幀相機時間到達後，prior 只消化已符合雙時間條件的事件：事件的 bag 到達時間不晚於目前 PLY 幀時間，感測器訊息 header 時間也不晚於目前相機 header 時間。較晚才到的事件留待之後處理；已錯過所屬幀的舊採集時間資料不會觸發回頭重算。GNSS 與 IMU 的訊息時間契約來自原始輸入，不以地圖時間欄位推回軌跡。

### GNSS 平面位置

GNSS XY 由前向 constant-velocity Kalman filter（定速卡爾曼濾波器）估計。狀態為位置與速度 `[x, y, vx, vy]`；每個新事件只更新目前狀態，不保存 RTS backward pass（反向平滑）或重播舊 pose。GNSS 高度只用於初始相機高度；後續幀沿用前一個地圖配準 pose 的高度。

### GNSS course 與 IMU heading

GNSS course（行進方向）以過去 2 秒的 GNSS XY 區段估計。區段尚未完成時，不交付 heading observation（航向觀測）；完成後才在區段結束時刻對 heading bias（航向偏差）做一次前向 Kalman update。這樣使用區段中可用的歷史樣本，但不把估計回填到區段中點，也不使用未來的區段來初始化整段 bias。

IMU 姿態只取當前已抵達的 quaternion（四元數）；利用最新角速度最多向前外推 0.1 秒。若該姿態太舊，程式拒絕建立姿態 prior，不用後續 frame 的 IMU 姿態作 frame 未來 SLERP（球面內插）。舊版的置中 Savitzky–Golay、RTS、整段 heading bias 校正及雙向 rate clamp 皆不在新定位路徑中。

以下是本版沿用的 GNSS／heading 工程先驗與門檻。它們來自選定舊版程式，不是透過新 leaderboard 分數搜尋；目前沒有 GNSS／IMU 外參或 covariance 的正式標定證據。

| 參數 | 設定 | 用途 |
| --- | ---: | --- |
| GNSS 水平標準差 | 0.3 m | 位置觀測噪聲先驗 |
| XY 加速度 noise PSD | 4.0 | 定速模型的過程噪聲先驗 |
| robust gate | 3 sigma | GNSS 位置與 heading block 的穩健門檻 |
| course 時窗 | 2 s | GNSS 行進方向區段長度 |
| 最低速度 | 5 m/s | 低速時不採用 course heading |
| 允許轉向率／區段航向跨度 | 2°/s／4° | 排除轉向中的 heading 區段 |
| heading observation 標準差下限 | 0.5° | heading 更新的不確定度下限 |
| 最大觀測偏差／最大 bias | 15°／10° | 拒絕不合理觀測並限制 bias |
| bias 隨機漫步／最大變化率 | 0.12°/√s／0.2°/s | 限制相鄰前向 bias 更新 |
| GNSS 預期 frame | 空字串 | 目前沿用輸入的 frame 欄位，不代表 frame 已標定 |

## 地圖配準與舊模型參數

相機點雲先以固定尺度 3.0 轉成公尺，再以 0.3 m voxel filter 降採樣。每幀只配準當下 prior 周圍半尺寸為 `(100, 100, 30) m` 的地圖子區域；地圖 voxel leaf 為 1.0 m。尺度固定，不在此版每幀擬合尺度。

配準使用 4 個自由度：`x, y, z, yaw`。roll／pitch 固定由姿態 prior 提供，幾何求解只估計有界的平移與 yaw 修正。XY 修正的平面模長上限為 1.5 m，z 修正上限為 1.5 m，yaw 修正上限為 8°。前一幀已接受的 correction 乘 0.8 後可作為下一幀起始候選；它只向前帶入，並以 0.25 做前向低通更新，經成本檢查後輸出，沒有回改舊 pose。

配準步驟如下：

1. 在 GPU 找出半徑 2.5 m 內的 16 個最近地圖點，少於 6 個鄰點時不建立可靠平面。
2. 對鄰點做 PCA（主成分分析），只使用符合平面性條件的地圖幾何：第二小特徵值至少 `0.04 m²`、最小特徵值不大於第二小特徵值的 0.25 倍，且 planarity 至少 0.2。
3. 依估計平面法向量計算方向性距離。法向幾何權重為 2.0，切向殘差權重為 0.05；Huber robust loss（Huber 穩健損失）轉折值是 0.5 m，最近鄰距離超過 2.0 m 不作有效配對。
4. 最多執行 15 次 Newton iteration，單次更新最多做 8 次 line-search 減半。只有成本下降且有效可靠幾何足夠時才接受 map update。

最近鄰搜尋、PCA、殘差及 Hessian／gradient 累加都在 CUDA 執行；CPU 只解小型 `4×4` 線性系統。這是舊 `RefineSensorGuided`／`SurfaceGeometryCache` 目標函數的 CUDA 向量移植，不是舊 binary 的逐位重現。若配準被拒絕，輸出當幀 sensor prior，並清除供下一幀沿用的 correction；不以舊靜態 pose 代替本幀輸出。

`midterm01_gpu/track2/parameters.py` 是本版設定入口。主要值如下：

| 設定 | 固定值 |
| --- | ---: |
| `PLY_SCALE` | 3.0 |
| `MAP_LEAF_M`／`INPUT_VOXEL_LEAF_M` | 1.0 m／0.3 m |
| `SUBMAP_HALF_EXTENT_M` | `(100, 100, 30)` m |
| `SENSOR_MAX_XY_CORRECTION_M` | 1.5 m |
| yaw／z 修正上限 | 8°／1.5 m |
| `SENSOR_SMOOTHING_ALPHA` | 0.25 |
| `GEOMETRY_NORMAL_GAIN`／`GEOMETRY_TANGENT_WEIGHT` | 2.0／0.05 |
| `GEOMETRY_NEIGHBORS`／`GEOMETRY_RADIUS_M` | 16／2.5 m |
| `GEOMETRY_MIN_NEIGHBORS`／`GEOMETRY_MIN_PLANARITY` | 6／0.2 |

## 檢查證據與精度限制

第一輪合成檢查通過了下列機制：因果前綴、完成 heading block 後才更新、過期 IMU 拒絕、未來資料污染、雙時間事件、CUDA PCA 與同定義 CPU 參考、no-overlap 時清除 correction、3 幀 runner 輸出契約，以及 4 項 RViz 檢查。數值對照使用由選定 commit 原始 C++ solver 編譯的本機 shared library；`alpha=0.25` 與 `1.0` 的 pose、correction、cost 對照誤差在 `3e-4` 內。測試證據保存在 [第一輪合成檢查 XML](../outputs/run/track00_test_2610290156/pytest.xml)，oracle 建置產物與 log 在 [oracle 建置 run](../outputs/run/track00_test_2610290154/)。

目前沒有用真實 Track2 資料執行新版短段或 full，也沒有新版定位分數；合成檢查不能證明定位精度。GNSS frame／座標原點、GNSS 到相機的 lever arm、IMU 到 PLY 的旋轉與零 lever 假設都還沒有標定，noise 值也只是工程先驗。配準接受率、殘差、收斂、RViz 畫面或輸出完整度都不能代替獨立 ground truth。

先前使用者已手動停止的 v1 full run 保存於 [舊版 run](../outputs/run/track02_full_2610290132_0002/)；當時停在 168 幀，沒有 map update。其程式和結果維持原樣。新 run 另建於 `outputs/run/track02_full_YYMMDDHHmm/`；成功完成 full 後會自動產生唯讀圖表與來源 SHA manifest，短段與未完成 run 不產生完整比較圖。

目前專用 Docker image 是由既有停止容器保存狀態恢復，`scripts/dev.sh up` 可用；尚未以 Dockerfile 從零重建驗證。

## 最終檢查附錄

[最終檢查 run](../outputs/run/track00_test_2610290201/test_info.json) 的 **14 項檢查通過**；生成資料的 3 幀與 5 幀 runner 輸出在共有前綴的 CSV、TXT、pose trace 完全相同。另以 [唯讀輸入檢查](../outputs/run/track00_test_2610290200/data_contract.json) 核對實際 3299 幀及可交付的 475 筆 GNSS、11868 筆 IMU，未執行真實配準或濾波定位。

兩組 C++ oracle 對照的 correction 最大絕對差約 `1.53e-15`，objective 最大差約 `2.67e-14`；輸出 pose 最大差約 `5.86e-5`，主要是原 C++ Matrix4f 最後輸出的 float32 捨入。這是指定生成場景的數值對照，不是完整 Track2 精度或所有輸入的等價證明。測試採用的接受容差為 `3e-4`，詳細記錄與 [圖表](../outputs/run/track00_test_2610290201/analysis/README.md) 一起保存。

第二輪 `track00_test_2610290200` 為 13 passed、1 failed；唯一失敗是搬入圖表測試時漏帶 `pose()` 測試 helper（NameError），定位與數值測試通過。補齊生成姿態後，最終 14 項全過。各輪原紀錄都保留。另有既有 Matplotlib Axes3D warning，二維圖表與 PDF 均正常產生。

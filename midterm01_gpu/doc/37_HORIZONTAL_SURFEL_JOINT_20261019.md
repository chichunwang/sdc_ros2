# 37｜細 map 水平觀測、座標核對與跨幀聯合估計

2026-10-19，接續 [36｜配準與融合實驗](36_REGISTRATION_AND_FUSION_UPGRADE_20261019.md)。使用者授權立即實作／短測，完整 Track1／Track2 仍由使用者手動啟動。

## 本輪結論與範圍

已把 **0.3 m 局部 map surfel → 空間留出 → 跨時間一致性 → GNSS／gyro／map 聯合求解 → 最終留出驗證** 接成可由 benchmark 執行的實驗分支。選項為 `--horizontal-refinement surfel_joint`，預設仍為 `none`，已評分最佳仍是原前相機 motion_rts＋directional＋carry，1.15008／0.30488。

**開發短測與使用者後續兩條完整測試的可靠 map factors 都是 0，聯合求解跳過，最終 CSV 與原 motion_rts 最佳逐 byte 相同。** 2026-10-20 使用者回報 Track1 仍為 1.15008；這輪沒有帶來精度提升。合成牆面能通過並實際修改輸出，證明接受／聯合求解／發布路徑有運作；32 項測試及 512 筆短段完整性通過不代表實際精度進步。後續完整測試及拒絕原因見文末。

研究證據固定入口：`outputs/run/track00_test_2610192200/`，原始目錄 `outputs/run/track00_test_2610192200/`。所有結果現已統一在run目錄，分數記在各次test_info.json。

## 1. 水平約束怎麼產生

新增 `scripts/horizontal_surfel.py`，保留已評分 C++ coarse 配準，另讀 run-local 的 0.3 m map。細 map 從原 map 依本輪候選點雲的世界範圍加 4 m 建立，原始 map／來源資料唯讀；不拿高度用 cache 當作已核對的水平 map。

每個約 2 秒的區塊取中間幀與前後約 0.45 秒的幀，實際時間差需各至少 0.35 秒；不足 0.9 秒的區塊不建立 factor。前相機維持原尺度 3；來源 deterministic voxel 代表點最多 6000 個，距離 1–30 m。這不是新增的尺度標定。

map 的兩個局部半徑為 0.9／1.8 m，各最多 64 個鄰點。兩尺度均需至少 8 點、足夠第二特徵值、厚度比 ≤0.2、planarity ≥0.2，normal 夾角 ≤12°。保留 `|normal_z|≤0.98` 的斜面／立面，純水平面排除。先以 `|normal_z|≤0.6` 做過 24 個場景檢查，立面只有 0–5 個獨立區塊；才擴充到斜面假說，未按榜分搜尋門檻。

配對半徑 1.5 m；以 1.2 m map 空間區塊和四個 normal 方位 bin 去重，每區塊／bin 取最近代表，每 bin 最多 60 個，沿空間分散抽樣。normal bin 以循環方位量化，避免 ±π／0 邊界把同向法線拆開；同一空間區塊內多個方向共用一份總權重，擬合／留出門檻以唯一空間區塊計數。像素密度不能重複增加同一平面的資訊量。空間 hash 固定留出約 1/4 區塊，不依殘差挑選。

用 point-to-plane 殘差 `r=nᵀ(Rz(dyaw)q+dxyz−target)`，同時估計局部 `dx,dy,dz,dyaw`。位移信任範圍各 ±0.75 m、yaw ±3°；擬合用 quality 權重及 0.5 m Huber，沒有 GNSS 或高度 prior 加到這個 map 擬合中。

**關鍵是從資料中消去自由 Z。** 將加權 Jacobian 的 Z 欄記為 `jz`，其餘為 `Jh=[Jx,Jy,Jscaled_yaw]`，得到：

```text
J_horizontal = (I − jz jzᵀ / (jzᵀjz)) Jh
H_horizontal = J_horizontalᵀ J_horizontal
```

Z 欄為零時直接保留 `Jh`。這相當於對 Z 做 Schur 消去，避免用固定高度替水平地面製造 XY 支持。單一傾斜平面也不能獨立分辨沿其法線的水平位移與高度；機制測試涵蓋這項退化。

yaw 用 3° 的參數尺度表示，相當於在比較距離時以 1 m／3° 縮放。僅保留特徵值 ≥`max(1, 0.02×最大值)` 的實測方向，至少兩個方向才接受。這是實驗性的數值／模型保護，不是已標定 covariance。

擬合區塊至少 12、留出至少 6；須收斂、未碰界線、留出絕對殘差 median≤0.3 m、P90≤0.75 m，且 median 不比原 pose 增加超過 0.02 m。留出獨立解只用於驗證，不能回填訓練 factor；兩個解在有支持方向的距離需 ≤0.3（上述縮放座標）。

中間與兩個鄰近幀均需通過，且支持方向的修正一致；只有中間幀加入 graph。跨區塊重用相同 map patches 時，依最大重用次數降低資訊權重，另用 1/2 information allowance 表達未知深度／模型相關性。這仍不足以把共同深度偏差變成獨立噪聲，未宣稱統計校準。

## 2. 座標、外參與延遲核對

`audit_sensor_contract.py` 從本輪 raw bag export 核對 bag／raw SHA、topic／frame、camera-camera 標定、timestamp 與 heading lag 假說；原 bag 以 SQLite 唯讀開啟。

| 項目 | Track1 | Track2 | 決策 |
| --- | --- | --- | --- |
| GNSS bag−header median | 1.138431 ms | 1.095209 ms | 不是硬體同步或內部濾波延遲的證明 |
| IMU bag−header median | 0.211211 ms | 0.294382 ms | 不額外平移時間 |
| camera-camera rotation／逆矩陣閉合 | 通過 | 通過 | 只證明 rig 內部標定一致 |
| GNSS frame_id | 空 | 空 | origin／datum／antenna 參考點未確認 |
| IMU frame_id | imu_link | imu_link | camera→IMU/body 外參未提供 |
| `/tf_static` | 無 | 無 | 無法由 bag 取得所需外參 |
| GNSS orientation publisher／來源 | 未知 | 未知 | 本輪不作獨立 heading factor |

以交錯 8 秒時間塊擬合 GNSS yaw 與 IMU yaw 的 offset＋lag，並在另一半驗證：Track1 lag 估計碰到 −0.5 s 邊界，留出 RMS 2.02855→2.02543°，幾乎沒有改變；Track2 lag 約 −0.476 s，留出 1.89925→2.02651°，反而變差。這不是可套用的硬體延遲校正；offset／濾波／來源相關性均可能模仿 lag，因此兩條都不套時間移位。

官方 PDF 明寫 init_guess 加了噪聲；本機 YAML 只有三相機間外參。不能從 noisy init_guess 反推正式 camera→IMU/body 安裝角，也不能從貼圖或分數推定 GNSS origin／lever arm。保留原已評分 source→map 座標約定，沒有宣稱輸出原點新近通過 body 契約認證。

完整 3D IMU preintegration 尚未啟用。先需要老師／資料提供者說明 GNSS orientation 來源、GNSS origin／antenna point、camera→IMU/body 外參、IMU quaternion 的世界框架及重力／acc 定義。現有零 covariance 表示未指定，不能當作零誤差。

## 3. map 接入後的聯合模型

新增 `surfel_joint.py`，重用 `kinematic_graph.py` 的 sparse Jacobian。狀態為每個 GNSS／map 時間的 `[x,y,vx,vy,yaw,effective_yaw_rate_bias]`。完整姿態中的 roll／pitch 和 Z 保留 C++ base；這是平面研究模型，尚非完整位置／速度／三軸姿態／三軸 bias 的 inertial estimator。

graph 包含：

- raw GNSS XY 每個 timestamp 一次，Huber gate 3σ；GNSS orientation 與由同份 GNSS 衍生的 course 都不再作因子。
- 常速度位置／速度過程、raw gyro 經 quaternion roll／pitch 換算的 Euler yaw-rate 積分、effective yaw-rate bias random walk。
- map factor 的有支持 XY／scaled-yaw 方向，map 原始時間以整數奈秒合併進節點，沒有先四捨五入到 GNSS 節點。
- 弱初始 IMU yaw／rate-bias prior。禁用原研究模型的側向速度因子，因 body 前向軸與 lever arm 尚未核對。

共用模型設定：GNSS std 0.3 m、acceleration PSD 4 m²/s³、gyro density 0.3°/√s、effective bias RW 0.02°/s/√s、初始 yaw std 30°、初始 rate-bias std 1°/s。兩條相同，均屬未標定假設，未因留段或分數搜尋 noise。

C++ base posterior 只用於初值／map association，不被當成另一個獨立 prior。來源及地圖共同偏差仍可能影響 factor，接受率不等於定位精度。

求解後用位置／速度 Hermite 插值，yaw 使用 gyro 傳播加節點修正，邊界最多 0.3 s 外推。整段修正需仍在原局部信任範圍，並在 map 空間留出上重新驗證真實候選 XY／yaw（Z 再獨立消去）。不通過則保留原 bytes，不把沒有可靠 map 的 sensor-only graph 當作升級結果。

## 4. 實測與機制測試

- 0.3 m 細 map 已實際從原始 map 建立：Track1 1,233,899 點、Track2 1,209,487 點，含 Track／map SHA／crop／leaf／payload SHA sidecar。map_modified 失敗、截斷、來源／範圍／leaf 或 payload 不符均不能發布／重用有效 cache。
- 原 full best pose 的 24 個分散場景，嚴格立面及擴充斜面兩種假說皆無通過的 factor；單幀診斷保留於 `fine_probes.json`／`slope_probes_v2.json`。這是只讀既有 pose 的觀測研究，沒有重算完整序列。
- 原先選定的直行／轉彎／低速、design／holdout 共 8×48＝384 筆新定位完成。原 C++ TXT／CSV、geometry／height／sensor_prior 與前輪控制組逐 byte 相同。24 個 map 觀測、8 個時間群組均拒絕，聯合 graph 跳過，最終 bytes 與 base 相同。Track1 轉彎有方向支持，但碰到擬合界線且留出不一致；Track2 主要只有 0–5 個獨立配對區塊。沒有放寬條件強行加入。
- 原控制組 384 poses 沿用已保存證據，沒有為了增加測試數字重跑相同控制組。短段 carry 會重設高度歷史，因此不把短段 Z 當作原 full trajectory 的同一高度。
- 新機制與回歸測試 32 項通過：plane／graph analytic Jacobian、退化消去、two-scale plane／line 品質、holdout 不影響訓練解、時間一致性／patch 去重、raw GNSS 只用一次、精確 map 節點、插值、已知 bias／map anchor、接受與拒絕的最終發布路徑、cache 失敗／不符、舊 fusion／prior／CLI。
- 合成發布測試使用已知牆面及 GNSS 偏移，確實產生新 CSV／pose 並保持 Z／tilt row；純地面測試則保留原 TXT／CSV bytes。合成結果不代表本題實際誤差。

- 兩條通用入口各 48 幀，未指定歷史 prior／任何 map cache，實際從原始 bag／map 生成新 run 與細 map，最終 TXT／CSV／geometry／height 與控制逐 byte 相同。raw prior 的時間與 quaternion／Z 完全相同；XY 最大數值差 Track1 9.10×10⁻¹³ m、Track2 5.55×10⁻¹¹ m，保留為浮點計算差異，沒有冒稱 prior bytes 相同。早期 generic runner 因要求 prior 完全逐 byte 相同而停止，輸出與數值核對其實通過；後續證據見 `generic_checks.json`。
- 兩條起點與終點各 8 幀，共 32 筆通用 raw-prior 測試通過；短於時間支持門檻時不建立 fine map／factor，完整保留 base bytes。
- 實際新 C++ 短段定位共 **384＋96＋32＝512 poses**，均通過完整性核對，沒有新完整 Track。最終去重／權重修正另對前述 10 組已驗證 base stages 做 **480 poses 的 postprocess 重驗**，仍為 0 map factors、保留原 bytes；這 480 不重複計為新的定位輸出。`final_stage_checks.json` 保存逐組來源及 raw-entry 證據。

開發中曾遇到 rank=0 的診斷輸出 Infinity，已改成 JSON null 並加入零資訊測試；最初在未裝 numpy 的主機 Python 啟動研究 runner 失敗，改由既有隔離 ML venv 執行。失敗 log 保留；日常 benchmark 主機入口仍可用系統 python3，所需數值套件由隔離 venv 處理，沒有修改主機或 ROS 的套件。

## 5. 已執行的完整驗證命令

下列保留作為已執行實驗的重現方式，**目前不建議再跑相同完整測試或重複提交**。兩條全路線都沒有可靠 factor；應先改善真實觀測並通過短段留出驗證，再建立下一個完整候選。命令從原始 bag／map 建立輸出，不指定歷史 run、prior 或 cache。本機 C++ 已有 Release；本輪主要是 Python 擴充，不需要因換參數重編。若移機，需另行恢復既有資料／ML venv，outputs 不隨 Git 同步。

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
bash scripts/dev.sh up

python3 scripts/benchmark.py --track 1 --full \
  --profile sensor_guided --sensor-fusion motion_rts \
  --geometry-mode directional --height-mode carry \
  --horizontal-refinement surfel_joint

python3 scripts/benchmark.py --track 2 --full \
  --profile sensor_guided --sensor-fusion motion_rts \
  --geometry-mode directional --height-mode carry \
  --horizontal-refinement surfel_joint
```

此為 **全路線觀測／候選驗證**，已確認沒有產生新 CSV bytes；不是已改善精度的候選命令。

最少只看：

| 檔案 | 用途 |
| --- | --- |
| `outputs/run/trackNN_full_YYMMDDHHmm/diagnostic.csv` | 本次最終 CSV；完整列數應為 3011／3299 |
| `outputs/run/trackNN_full_YYMMDDHHmm/result.txt` | 本次最終回放矩陣 |
| `outputs/run/trackNN_full_YYMMDDHHmm/validation.json` | 完整性與兩個階段的驗證 |
| `outputs/run/trackNN_full_YYMMDDHHmm/horizontal/report.json` | 看 `accepted_map_factors`、`applied`、`reason` |
| `outputs/run/trackNN_full_YYMMDDHHmm/base/` | 本輪原 C++ 結果／CSV／validation，用來比較，未覆蓋舊 run |

`guidance.csv`／`geometry.csv`／`height.csv`／`timings.csv` 都描述 **base C++ 階段**；最終 joint 檢查另看 horizontal report，不能把 base 殘差當 joint 後殘差。耗時分開保存 C++、sensor preparation 與 horizontal stage。程式異常會把 validation 標成失敗；品質拒絕則明確記錄未修改並保留 bytes。

## 6. 下一個可驗證方向

目前最欠缺的是 **跨時間／map 一致、且有足夠水平支持的來源觀測**。擴大 solver 名稱或強化平滑不會補出這些資訊。優先核對來源 frame／depth 形狀及尺度、不使用 noisy init 作外參；再研究跨時間靜態特徵與公尺尺度可觀性。三相機只有在同步、深度不確定度與外參一致時才能補有效側面結構，沿用 [29](29_THREE_CAMERA_SCORED_ANALYSIS_20261018.md) 的負面證據。

本次模型推導參考 [Generalized-ICP 原論文](https://www.roboticsproceedings.org/rss05/p21.html) 的雙邊局部平面觀點；本輪新增 factor 實際用 map point-to-plane 與資料退化消去。完整 inertial 因子後續可依 [Forster 等人的 on-manifold preintegration](https://arxiv.org/abs/1512.02363) 實作；本文的 planar yaw 積分沒有冒稱已完成該模型。

## 使用者後續完整測試與尋找入口

使用者已手動完成兩條完整 surfel_joint，2026-10-19 只讀核對：

| Track | run 日期尾碼（台北） | 筆數 | validation | map factors／applied | CSV 與原 motion 最佳 |
| --- | --- | ---: | --- | --- | --- |
| 1 | `_2610192232` | 3011 | 通過 | 0／false | SHA256 完全相同 |
| 2 | `_2610192240` | 3299 | 通過 | 0／false | SHA256 完全相同 |

分別評估54／59個時間群組，沒有可靠factor，因此沒有產生新的CSV bytes；當時尚無新Kaggle分數。這是使用者後續手動完整結果，前文512筆為先前開發短測，不混算。

### 2026-10-20：Track1 重複 1.15008 的原因

使用者再執行的 `outputs/run/track01_full_2610192358` 有 3011 筆、完整性通過。它與原已評分 `track01_full_2610191750`、前一個 surfel 完整 `track01_full_2610192232` 的 CSV、TXT、sensor_prior.csv、frames.csv 均逐 byte 相同。三份 CSV 的 SHA256 都是 `781db1124f4a9a3e59a1d4adfc7ac7e3c14ff18e09b4157901f2cd807cb4466a`。這能解釋同分；不只是分數顯示精度相同。

最新 `benchmark_command.json` 與 `sensor_prior.json` 均為 **motion_rts**，沒有啟用 `independent_heading_rts`；目前可核對的本機完整 run 沒有後者。因此這次結果不能用來判定獨立航向融合是否有效。截圖顯示三筆 Track1 1.15008，但沒有上傳 SHA，因此沒有獨立核對每筆 Kaggle 上傳內容。

最新完整 Track1 的拒絕統計如下；同一觀測可能有多個原因：

| 原因 | 觀測數／範圍 |
| --- | ---: |
| 空間支持不足，未進擬合 | 78／162 |
| 已擬合但碰修正邊界 | 71／84 |
| 消去自由 Z 後水平 rank 不足 2 | 53／84 |
| 擬合／留出解不一致 | 57／84 |
| 留出殘差超標 | 45／84 |
| 留出結果比原 pose 更差，超過容許量 | 26／84 |

84 個可擬合觀測的 optimizer 都完成，但可靠性檢查全拒絕；**單幀接受數就是 0，並非只被三幀一致性門檻擋住**。54 個群組沒有 factor，聯合 graph 完全未求解。Track2 原完整有 177 個觀測，其中 169 個空間支持不足，剩餘 8 個也全拒絕。這些是當前配對／模型下的診斷，不能推論資料裡完全不存在可用立面。

碰界線與留出不一致支持優先檢查來源深度／尺度、靜態跨幀一致性及座標約定，但尚無法單靠這些數字區分它們。下一輪先做能區分來源形變與 pose 偏差的小型驗證；不以放大位移範圍、降低 rank 或移除留出檢查來強行增加 factor。可靠觀測出現後，才評估跨幀融合的收益。

已修正 benchmark 完成提示：完整性通過與 CSV 是否改變分開顯示；相同 SHA 明確印出「本輪未產生新的提交內容，無須重複提交」。新 `horizontal/report.json` 另記 `joint_estimation_executed`、`csv_changed_from_base`、`result_changed_from_base` 與可重疊拒絕統計。這是診斷／提示修正，不改 solver、接受門檻或既有結果。合成接受／拒絕發布測試也核對提示與實際 CSV 一致。

使用者最新指定：不新增測試索引。所有完整／短段目錄統一在outputs/run，命名trackNN_full/test_YYMMDDHHmm；算法／版本放內部test_info.json。既有結果已搬移，原證據SHA保持，沒有重跑完整定位。

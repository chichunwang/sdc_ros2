**2026-10-23 現行規則提示：**因果定位適用所有 Track；不得以未來資訊回修 pose、使用 RTS／backward smoothing，或在定位後離線修正軌跡。Track1／2 GNSS 仍可作輸入，但 RTS 不因此合規。本文相關算法、命令與成績均為歷史紀錄，不作現行正式提交推薦；規則與程式影響見 [48](48_RULES_AND_CODE_AUDIT_20261023.md)。

# 35｜motion_rts 新最佳與 GNSS orientation 航向消融

日期：2026-10-19（Asia/Taipei）。先保存使用者回報的新分數，再依「就試試看，當作獨立航向觀測」實作單項實驗。完整 Track 與提交仍由使用者操作；本輪只準備先驗、執行預選短段及讀取既有結果。

## 目前已評分最佳

| Track | 原 heading_rts＋directional | 新 motion_rts＋directional＋carry | 分數降低 |
| --- | ---: | ---: | ---: |
| 1 | 1.15275 | **1.15008** | 0.00267／0.2316% |
| 2 | 0.30576 | **0.30488** | 0.00088／0.2878% |

來源是本次對話的使用者 submission-history 截圖，沒有重新查排行榜。總分不能分解成 XY／yaw 誤差，也不能推斷其他隊的算法。

已評分來源：`outputs/run/track01_full_2610191750`／`track02_full_2610191757`。分數保存於各自test_info.json，沒有另設最佳／最近分類；回放預設選最近通過完整性檢查的full目錄。

保存：`outputs/run/track00_test_2610191808/`，`provenance.json` 記分數、檢查、來源、快照與 hash；舊 geometry／heading／三相機／高度結果都保留。

| Track | 新完整 run 尾碼 | 幀數 | CSV SHA256 |
| --- | --- | ---: | --- |
| 1 | `track01_full_2610191750` | 3011 | `781db1124f4a9a3e59a1d4adfc7ac7e3c14ff18e09b4157901f2cd807cb4466a` |
| 2 | `track02_full_2610191757` | 3299 | `5b04d260123a8d87f0f0c317ddbce5b59c3ee3068cf07e020a370034a9728dbb` |

本機矩陣／CSV／索引／整數時間／有限值／旋轉及完整性核對通過，CSV 與矩陣的 XY／yaw 一致。上傳 bytes 未從 Kaggle 獨立下載核對，截圖與本機 run 依操作上下文對應，不能宣稱上傳 SHA 已驗證。

原前相機、scale=3、carry、directional、先驗 Z／quaternion 不變，只有位置平滑不同。兩版 prior XY 差 RMS 約 3.92／1.85 cm，output XY 差 RMS 約 4.06／1.93 cm；新 output 對 prior 的水平修正 RMS 約 1.58／1.05 cm。這些是改動量，不是真值誤差。平滑可能減少隨機噪聲，不能單獨辨識 GNSS 共同偏差。

## 新觀察與實驗前提

舊前處理只讀 `/localizer_node/gnss_pose` 的 position，未用 orientation。兩條 raw bag 的 quaternion 有效、roll／pitch 為零，yaw span 約 35.06°／52.07°。GNSS orientation 與可靠兩秒直行 course 的差 RMS 約 **2.32°／1.24°**，不是近乎相同的讀值。

GNSS bag−header 時差 median 約 1.14／1.10 ms；IMU 約 0.21／0.29 ms。本輪沿用原 bag 時基，沒有按殘差搜尋時間偏移。GNSS frame_id 為空，bag 無 tf_static，只有 camera–camera rig 標定；完整 body／IMU／camera 外參仍待核對。

使用者確認老師尚未說明 GNSS orientation 來源，接著明確授權把它當作獨立航向觀測實驗。因此程式**實際加入獨立因子**，metadata 仍記 `heading_source_verified=false`、`heading_independence_assumed=true`，不把這項假設改寫成已知事實。

原始感測器 export 與 audit 報告保存於上述 scored archive：`input_trackN.npz/json`、`heading_audit_trackN.json`。舊 audit 的 `accepted_for_fusion=false` 是授權這項實驗之前的判斷，保留原報告；本次實驗另存。IMU covariance 全零代表未提供 covariance，不代表完美精度，見 [ROS Humble Imu 定義](https://github.com/ros2/common_interfaces/blob/humble/sensor_msgs/msg/Imu.msg)。未把 raw gyro Z 與 Euler yaw-rate 的差直接當成已校準 gyro bias。

## 模型與三組消融

保留既有慢變 heading bias：

$$
\psi(t)=\psi_{IMU}(t)+b(t),\qquad b_{k+1}=b_k+w_k.
$$

新增因子觀測：

$$
z^P_k=\operatorname{wrap}(\psi_{GNSS\ pose,k}-\psi_{IMU,k})=b_k+v^P_k.
$$

它直接使用 orientation，不以 GNSS 位置差分產生這一筆觀測；是否與既有 course／IMU 在上游相關仍未知。三組都用相同原始 motion_rts XY、原 IMU roll／pitch、carry、directional、前相機與 scale=3：

| `--sensor-fusion` | heading bias 的觀測 |
| --- | --- |
| `motion_rts` | 原可靠直行 course，已評分對照 |
| `pose_heading_rts` | GNSS pose orientation **取代** course；用來分辨新來源本身的影響 |
| `independent_heading_rts` | 原 course **加上** GNSS pose orientation，假設兩種 bias 因子獨立 |

新增來源每兩秒彙整一個 circular robust offset；在低速／轉彎仍可產生因子，因為它沒有 course≈車頭方向的限制，但這依賴來源真的是可靠 heading。樣本／IMU 覆蓋不足、無效 quaternion、offset≥15° 或區塊 P90 偏離≥4° 都拒絕。

block std 下限固定 **3°**，較大 scatter 時提高；不除以 sqrt(N)，不把高頻內插或同區塊多筆值當成額外獨立資訊。3° 是來源不明時的共用實驗噪聲假設，不是標定結論；兩條相同，沒有 noise sweep 或依分數調參。bias random walk 仍 0.12°/√s、最大 ±10°、校正變化率≤0.2°/s。

同一時間的兩個因子以 precision product 合成一個 bias knot，再做原 KF／RTS，避免重複插值節點。這個乘積與模型 variance 變小都依賴獨立假設，不能把它當成已量測的信心提升。兩種 offset 都減去同一份 IMU yaw；本實驗將該讀值視為條件輸入，尚未建模這項共同 IMU 噪聲，也未解決 course 與 XY 共用 GNSS 的相關性。姿態以 map Z 左乘原 SLERP，保留 roll／pitch；不積分 raw gyro／acceleration，不新增 map／VO joint solver。

實作在 `sensor_fusion.py`、`prepare_sensor_prior.py` 與 benchmark 的新 fusion choices。原 legacy／fixed／geometry off／carry 預設不改，原 motion_rts 數值不變；新來源 quaternion 納入 sensor sample hash。

## 留出時間區塊：沒有一致勝出

每條固定分散選 12 個原兩秒區塊，交錯 design／holdout 各 6。每次同時移除目標區塊與左右鄰塊的 course、GNSS orientation 因子，再由其餘區塊預測偏差；不用全域 fixed bias 作 holdout fallback。IMU 仍是查詢時刻的感測器輸入。測試確認任意改動被留出的 XY／orientation，不會改變該區塊 prediction。

下表是保留區塊對兩種**未使用的輸入參考**的 RMSE，單位度；不是 ATE／真實 yaw 誤差，兩種參考也可能彼此或與 IMU 相關。部分 course 不符合直行門檻，所以區塊數較少。

| Track／保留參考 | 區塊數 | 原 course | pose 取代 | 假設獨立融合 |
| --- | ---: | ---: | ---: | ---: |
| 1／course | 5 | 1.45181 | 1.45538 | **1.37853** |
| 1／pose yaw | 6 | 2.22836 | **1.96419** | 2.14482 |
| 2／course | 6 | **0.42105** | 1.51369 | 0.59237 |
| 2／pose yaw | 6 | 1.37649 | **1.20076** | 1.39227 |

Track2 的 design course 原本由 0.624 降到 0.498°，保留 course 卻變差，正是不能只看設計段的例子。直接替代在 Track2 對 course 明顯較差；獨立融合效果較小，仍未一致改善。沒有看完 holdout 再改 std／Track 設定。

全序列僅先驗準備的 yaw 改動 RMS：pose 取代約 0.807°／1.147°，獨立融合約 **0.202°／0.276°**。這是變動量，沒有為此執行完整配準。

## 分散真實短段

沿用 34 在新航向實驗之前選定的八個 48 幀場景：Track1 直行 idx 905／2286、轉彎 1808；Track2 低速 96／224、直行 1045／2568、轉彎 1379。Track1 沒有合格長度低速段，兩條轉彎沒有第二個分散合格段；沒有為湊案例改定義。

**8×3×48＝1152 筆 pose** 全完整有限、退出 0、validation 無錯；逐組核對相同 binary／cloud SHA／非 prior solver 參數、相同 timestamp／XYZ 字串與原 IMU roll／pitch。新跑的八組 motion 對照 `result.txt`／`diagnostic.csv`／`geometry.csv`／`height.csv` 與 34 的 motion 控制組逐 byte 相同。

獨立融合相對 motion 對照的 output yaw 差 RMS：Track1 直行 0.132°／0.030°、轉彎 0.128°；Track2 低速 **0.750°／0.665°**、直行 0.048°／0.116°、轉彎 0.021°。低速缺乏 course，新增來源的影響尤其大，不能只看全路線平均 0.28° 就忽略。output XY 差最大的一段約 1.67 cm RMS。

三種方法在這八個片段都各接受 384／384 幀，沒有新的拒絕；map clipped residual 有升有降，獨立融合 4／8 段較低。接受率與殘差不是定位真值，因此這仍不能宣稱提升分數。短段 carry 重新起算高度歷史，不能代替全序列。

## 找結果、驗證與下一步

固定研究入口：**[outputs/run/track00_test_2610191818/README.md](../outputs/run/track00_test_2610191818/README.md)**，原始目錄 `outputs/run/track00_test_2610191818/`。

- `heading_ablation.json`：三種完整先驗、設定、yaw 變動與 24 個留段。
- `summary.json`／`short_verification.json`：1152 poses、配對數值、原控制組逐 byte 比對與限制。
- `ablation_runs.json`／`short_runs/`：每個短段的原 idx、實際 output、命令與 log。
- `unit_tests.log`／`ros_tests.log`：主機 20 項、ROS 12 項測試通過，含 branch cut、低速／轉彎、無效 quaternion、precision product、roll／pitch／XYZ 保留與 holdout 防洩漏。
- `generic_equivalence.json`：兩條各 48 幀的通用 raw-bag 入口與凍結候選對照；結果完成後才記錄通過，不以先驗準備冒充定位。

兩條通用入口的 `result.txt`／`diagnostic.csv`／`geometry.csv`／`height.csv` 都與凍結獨立航向候選逐 byte 相同。共 **1248 筆真實短段 pose**，沒有啟動新的完整序列；程式已可從原 bag 通用重建這個候選。最終多記 `used_for_fusion` 欄位以區分替代組沒有使用的 course，僅診斷欄位，不改因子／數值；前後來源快照分別保存。

本輪沒有 C++ 改動，不需重編。`replay.py --check-only` 已核對新最佳兩條 3011／3299 的矩陣、點雲來源與旋轉；沒有啟動 GUI／publisher。

**目前不升為最佳。** 優先釐清 GNSS orientation 的發布來源、靜止時是否有效及共同噪聲／延遲，才能決定是否在低速提供絕對航向；相同來源若重複計數，不能靠較小 model variance 佐證改善。未取得真值前，正式評分仍是這項消融的重要外部證據。

若使用者要完成這個獨立航向假說的正式消融，以下是已接好通用入口的候選；只需改 fusion，不綁舊 run。這是**未評分實驗命令**，不是建議替換目前最佳：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
python3 scripts/benchmark.py --track 1 --full \
  --profile sensor_guided --sensor-fusion independent_heading_rts \
  --geometry-mode directional --height-mode carry
python3 scripts/benchmark.py --track 2 --full \
  --profile sensor_guided --sensor-fusion independent_heading_rts \
  --geometry-mode directional --height-mode carry
```

每次都在outputs/run建立trackNN_full_YYMMDDHHmm新目錄，提交diagnostic.csv、回放result.txt，先看validation.json。實驗不自動更新分數；原已評分來源及證據保留。

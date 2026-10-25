# 53｜v5 點雲尺度、靜止與轉彎位置約束（2026-10-25）

**歷史診斷與版本說明：**本文件描述初版 `track3_cuda_v5_causal_scale_stationary` 的尺度與靜止假設。之後的計時、body-frame 視覺速度、移動時 Sim(3) 尺度政策，以及從 frame 0 的真實 80 幀結果，請以[doc55](55_V5_STARTUP_CLOCK_FIX_20261025.md)為準。以下 frame 950／2400 input-only 診斷與初版假設仍作為歷史證據保留，不能代表後續版本已成功修正這些長段場景。

初版 89 項生成資料合成／CUDA 測試（56.73 s，[GPU 測試 XML](../outputs/run/track00_test_2610250847/gpu_pytest.xml)）與 14 項重點回歸（3.01 s，[回歸測試 XML](../outputs/run/track00_test_2610250847/causal_pytest.xml)）有重疊，不相加計數。它們驗證當時生成資料下的機制，不代表真實定位精度。

## frame 950 附近：場景安靜，pose 和 scale 卻持續變化

只讀輸入稽核在 frame 950 到 1800 之間追蹤到 595 個影像特徵，光流位移中位數為 `0.3207 px`、P90 為 `0.9204 px`。相同畫面格子的 PLY 深度中位數比例為 `0.8508`；這表示 raw 點雲尺寸有縮小線索，但畫面格子不是完整的靜態特徵配對，因此比例值不能當成真實尺度或 pose 真值。舊版輸出在同一段的 scale 從 `7.9440` 降至 `6.2731`，pose 也從 `(401.76, -128.35, 5.84)` 移到 `(396.48, -129.80, 5.20)`。舊 run 在 1000–1400 與 1400–1800 分段分別採用 383/400 與 375/400 個 map 量測，NIS 拒絕數為 0；其他幀包含 registration 迭代失敗等拒絕。這表示單純把漂移歸因於濾波器拒收 map measurement，和這份紀錄不符。

證據來自[唯讀原始影像／深度比較](../outputs/run/track00_test_2610250847/raw_input_audit.json)及[舊 full 前綴 pose trace](../outputs/run/track00_test_2610250847/evidence/track03_full_2610250813/pose_trace.jsonl)。稽核沒有解 pose、做 map registration 或使用 GPU，也沒有獨立真值。

另一份[因果 3D 點對尺度稽核](../outputs/run/track00_test_2610250847/pair_scale_audit.json)用 CUDA 像素／PLY 3D 對應和相同的目前／過去影像特徵距離公式估計 raw-unit 變化。frame 950→1800 的倍率為 `1.2030`（log-MAD `0.0232`，一致比例 `93.2%`）；950→2000 為 `1.2347`。它和畫面格子深度比例的方向一致：raw PLY 變小時，維持公尺大小需要把 scale multiplier 調大，與舊 run scale 往下掉的方向相反。這是輸入點雲的內部一致性檢查，沒有做 PnP、map pose 解算或定位，仍不是公尺真值，也不能證明長段尺度完全正確。

## scale 要維持的是公尺幾何，不是固定數字

PLY 裡的座標是影像重建的 raw units。配準用正尺度 `s` 把它換成公尺：

```text
q_map = s * R * p_raw + t
```

若重建出的 raw 點雲大小穩定，`s` 才會大致固定。若同一組場景在 raw PLY 裡縮小，為了讓它在 map 上仍有相同公尺大小，`s` 就要按相反方向變大。使用者看圖覺得 `7.5` 貼合是一項有用的觀察，但它本身沒有提供公尺真值；演算法不會把 `7.5` 寫成常數。

舊版每幀讓 registration 一起解 6-DoF pose 和 scale，但 Kalman filter 只融合 pose，scale 又整個採用 registration 候選。濾波後的 pose 與未同步融合的 scale 可能不再描述同一組點雲幾何。新快照在第一筆可信 map registration 建立絕對 scale anchor；後續用目前與過去影格的影像特徵，對應兩張 PLY 的 3D 點，取多組點對距離比：

```text
s_current = s_reference * median(distance_reference / distance_current)
```

剛體旋轉和平移不改變點對距離，因此這個比值主要反映 raw reconstruction units 的相對縮放。程式以 robust median 和一致性檢查排除不可靠比例；特徵深度不足、動態點太多或重建變形時，保留上一個有效尺度。初始 map anchor 之後，NDT／ICP 固定 scale，只解 6-DoF pose。這是尺度和 pose 的一致性修正，尚未證明 track 的實際公尺精度。

影像尺度、靜止偵測與 PnP 門檻集中在 `track3/motion_parameters.py`。那些數值是尚未校正的工程界限；遇到 scale 對不上時，先看 run 的 `motion_constraints.scale_diagnostics`，不要直接把尺度鎖到肉眼選的值。

## 靜止時怎麼防止積分慢慢把車推走

只有第一筆 map pose anchor 建立後才啟用靜止判斷。程式同時要求前後影像有足夠且分布夠廣的特徵、單幀與安靜基準影像的光流都夠小、raw gyro 夠小並連續維持 `0.3 s`，而且附近的 3D 特徵與尺度估計可靠。低加速度本身不能證明車停住；例如等速直行時，加速度也可能很小。這項判斷只讀目前與過去資料，不會用未來影格。

判定靜止後，輸出 pose 暫時留在上一個已提交位置，速度設為零但保留有限不確定度，並用靜止約束協助估計 accelerometer／gyroscope bias。舊 pose 不會被移動去配合後來資料。影像流動、gyro 顯示轉向，或 3D／尺度支持不足時，靜止狀態會解除；frame 950 或 frame 2400 沒有寫死在條件中。RViz 會以藍色 `STATIONARY (IMAGE + IMU)` 標示這種 pose hold；該幀 `map_accepted=false` 是此約束的明確記錄，不等同於新 registration 成功。

這些線索可能受動態場景、模糊、低紋理或相機震動影響。若 `observations.jsonl` 顯示 `stationary=true` 但實際仍在動，先查 `image_flow`、`baseline_flow`、`raw_gyro_max_radps` 和 3D 支持；若車停住卻始終無法觸發，則檢查同一組欄位看是哪一項證據不足，再一次只改 `track3/motion_parameters.py` 中對應的偵測門檻。不要用 IMU 外參去補光流偵測錯誤。

靜止偵測需要多種線索的設計背景，可參考 [OpenVINS 零速度更新說明](https://docs.openvins.com/update-zerovelocity.html)。本專案的門檻是工程設定，不是從該文件取得的標定值。

## frame 2400 附近：轉向姿態有變化，位置仍未可靠跟上

IMU gyro 可以量到轉動方向；它不會單獨告訴濾波器車子平移了多少。若 map registration 失敗或 NIS 拒絕，EKF 仍靠加速度積分位置，偏差會累積。舊 run 的轉彎診斷中，frame 2400、2500、2550、2650 的 registration 以迭代上限結束；frame 2450 候選被 NIS `207.04` 拒絕（6-DoF `0.999` gate 約 `22.46`），候選相對 prediction 的 Y 位移差約 `−14.25 m`。frame 2600 曾通過一次 gate，NIS `18.46`。所以 pose 數值不是完全不變，但多次候選失敗／拒絕會讓輸出位置無法穩定跟上轉彎；RViz 看起來像留在原地是合理的故障表現。[轉彎診斷摘要](../outputs/run/track00_test_2610250847/turn_summary.json)只整理既有 run，沒有重跑定位。

新快照在 moving 狀態用前一張與目前影像的 KLT 特徵，將前一張 PLY 的 3D 對應點與目前影像做 PnP，取得相鄰影格的相對位移，再換算成 IMU 原點的間隔平均視覺速度，作為 ESKF 的 3D 速度量測。量測仍經自己的 3D NIS gate；沒有設定固定車速，也不會把轉向方向硬指定成平移方向。當 map registration 暫時失敗，這份視覺速度量測能協助平移 prediction 跟上畫面運動；影像特徵不足或 PnP 品質差時，它也可能被拒絕，不能保證每次都補回 map pose。

同一份 input-only 稽核看到 2400→2450 的 median 光流約 `82.43 px`，但只有 31 組有效 3D 對應，低於尺度更新所需支持數，所以該 pair 的尺度更新被拒絕。這表示尺度 gate 沒有用薄弱配對硬算比例；也不能由此宣稱轉彎位置已修好。[稽核 JSON](../outputs/run/track00_test_2610250847/pair_scale_audit.json)只讀這兩幀的影像／PLY，未解 PnP、pose 或 map registration。

視覺速度的 `1.0 m/s` 標準差是未校正的工程假設。相鄰位移換成的是時間區間平均速度，作為目前速度的近似；量測也重用了前一影格的方向與尺度，和 ESKF state 有相關性。NIS 通過不代表它是獨立真值。檢查時要一起看 `motion_constraints.visual_velocity`、`inertial.prediction.visual_velocity_update`、registration reason、NIS、速度和 scale，不能只看 yaw 或綠色／橘色點雲。

## 用口語線索逐次檢查 IMU 參數

「畫面往左偏」不等於「把 IMU yaw 往右改」。點雲偏移可能來自 pose、scale、registration 或安裝外參；只憑 RViz 畫面無法分辨來源。先從同一 run 的逐幀診斷找問題屬於哪一段，再只動一個有物理理由的值：

| 看見的現象 | 先查哪裡 | 可能相關的參數 |
| --- | --- | --- |
| 停車後 pose 還慢慢滑，或高度／傾斜持續變 | `stationary` 是否觸發、raw gyro、accelerometer／gyro bias、map 重力方向假設 | 靜止偵測證據，或有校正依據的 roll／pitch 與 bias 模型 |
| 車頭轉向看起來合理，點雲位置卻落後；registration 也常失敗 | PnP 速度是否通過、3D NIS、registration reason、scale diagnostics | 特徵／深度對應；若只在旋轉時呈現固定槓桿臂偏差，再檢查實際安裝平移 |
| 每次轉彎都呈現相同的 IMU 相對相機方向差 | IMU 軸向與相機軸向、左右轉的 rotation innovation | `track3/imu_parameters.py` 的 `IMU_RPY_DEG`；依實際安裝校正，不從單次偏移反推角度 |
| 旋轉方向合理，但繞車體轉動時相機位置有系統性偏差 | IMU 與相機光心的實際距離及方向 | `track3/imu_parameters.py` 的 `IMU_XYZ_M`；這是 IMU 原點在 PLY FLU 的公尺座標 |
| PLY raw 點雲尺寸逐漸變，map 上的物體也跟著縮放 | `scale_diagnostics`、特徵深度支持、registration 的 pose／scale 行為 | `track3/motion_parameters.py` 的尺度一致性門檻；不要用 yaw 或外參平移修 scale |

固定座標的外參代表設備安裝關係，通常不會隨 frame 改變。`IMU_RPY_DEG` 和 `IMU_XYZ_M` 目前都是未標定假設；基準 run 與每一個候選都保持相同資料段、noise／gate 和其他外參，每輪只改一項，保存各自 run，再比較多次直線、停車與轉彎中的 innovation 趨勢。左右方向只提供排查線索，不能單獨決定要把參數加正還是加負。先依設備實際安裝方向給候選值，再用不同運動條件交叉檢查；不要同時改 yaw、lever arm、scale 和 noise。

`./scripts/run.sh --max-frames 30` 適合確認啟動與短時資料流程，無法覆蓋 frame 950 的長時間靜止或 frame 2400 的轉彎；外參判讀需包含相應場景。完整 full 仍由使用者手動啟動。指令維持既有入口，參數直接改 Python 檔案。

## 因果定位與驗證界限

- 本版當時的 `/imu/data` arrival-time 規則已由後續 camera-header acquisition 對齊取代；目前 clock 與 arrival 因果契約請參見 doc55。視覺尺度、stationary 判斷和 PnP 速度只用目前與過去影格／點雲。
- 不用未來影格插值，不用 backward／RTS smoothing，不回改已輸出的 pose，也不在定位後平移或旋轉整條軌跡。離線圖表只讀已保存輸出。
- map 仍只用 XYZ；無 GNSS。Registration quality gate 與既有 NIS gate 保留，沒有為提高接受率而放寬。
- 本文件記錄的 v5 舊 full 前綴不是新 clock/body-velocity/map-scale 版的成績。80 幀真實短段與幾何一致性摘要見 doc55；接受率、scale 曲線、NIS、收斂、RViz 重疊及最近 map 點距離都不是獨立 ground truth。

競賽因果限制見 [Midterm01 規則](../../midterm01/AGENTS.md)。IMU 時間、bias、外參與初版 ESKF 假設見 [52｜v5 IMU ESKF](52_V5_IMU_EKF_20261025.md)。

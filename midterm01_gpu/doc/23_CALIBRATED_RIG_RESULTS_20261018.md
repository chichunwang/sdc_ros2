# 23｜原始三相機校正重建、尺度錨點與側面取樣（2026-10-18）

**後續兩條完整序列已由使用者完成／評分：** Track1 **1.15346**、Track2 **0.31064**，均未超過前相機最佳。完整幾何、相機級 probe、Z 反事實與新增消融見 [29｜完整成績與原因分析](29_THREE_CAMERA_SCORED_ANALYSIS_20261018.md)。本文件保留早期 8 groups 的方法與結果，不回填改寫舊尺度。

後續已完成 [跨場景偏差驗證與完整實驗入口](24_BIAS_VALIDATION_20261018.md)：新尺度與 temporal 參考仍未通過一致性檢查。本文件保留上一輪相鄰影格結果，不將 0.64738877 改寫成已通過的校準。

同日另完成 [相同算法的 GPU 加速與瓶頸分析](25_RIG_PERFORMANCE_20261018.md)：真正推論小段暖機約 0.78 s/frame，原場景 PLY／labels 相同；已保留使用者停止時的 2487 幀，提供新 run 接續指令。這是運算實作改善，尺度及精度結論不變。

本次依使用者要求，從少量原始影像實作有校正的三相機重建、保留固定安裝高度、加入側面幾何及公平消融。這是新的 raw-image 流程，沒有重套外參到官方已合併 MapAnything PLY，也不沿用上一輪 merged PLY 的尺度候選 0.87124954。

已評分前相機版本仍為 `sensor_guided + heading_rts + directional`，Team 3 的既有分數為 Track1 **1.15275**、Track2 **0.30576**；本次沒有完整 Track 或 Kaggle 分數。

## 固定高度的意義

固定的是每台相機相對於 rig 的安裝位置，沒有逐幀估計左右相機高度。校正檔使用公尺、`p_target = T_target_source p_source`。將 optical（右、下、前）轉成 FLU（前、左、上）後：

| 相機 | 相對前相機前向 x（m） | 左向 y（m） | 上向 z（m） |
| --- | ---: | ---: | ---: |
| 前 | 0 | 0 | 0 |
| 左 | −0.0600846 | +0.784752 | +0.0783861 |
| 右 | −0.0442703 | −0.760501 | +0.1004340 |

所以左右相機確實與前相機不同位置，但這份校正證據是高 **7.8／10.0 cm**，不能自行加一個「車頂比前相機高 2 m」的偏移。尚無完整 body→camera 標定，不能據此判定車體絕對高度。車輛遇到坡道時，整個 rig 在 map 的高度仍會變化；原本配準的全局 Z nuisance 保留。

## 重建與時間模型

工具：[reconstruct_calibrated_rig.py](../scripts/reconstruct_calibrated_rig.py)、[calibrated_rig_geometry.py](../scripts/calibrated_rig_geometry.py)。

1. 讀取 raw JPG 與三份 fisheye K／D。用 `cv2.fisheye.initUndistortRectifyMap` 轉成 pinhole，明確產生輸出 K、有效像素 mask，前相機另排除官方車身 mask。三台都固定 448×294；不是把 fisheye K 直接當 pinhole K。
2. 同一個 [MapAnything 官方模型](https://github.com/facebookresearch/map-anything) 的 Apache checkpoint，分別推論單前相機、三相機預測相對 pose、三相機給定 rig／運動相對 pose。保存各 view 的 depth、confidence、mask、K、模型 pose、實際使用 pose、camera_id 與整數 header 時間。
3. 網路的 pose 輸入是條件，輸出未必精確遵守標定。因此匯出時以已知 K 反投影 `p_cam = depth_z K⁻¹[u,v,1]ᵀ`，並明確使用固定外參合併；不直接信任模型預測的 ray／rig pose。
4. index 的 f_ts 是 bag、fl/fr_ts 是 header。逐筆 exact lookup 回 bag／header，再在同一 header 時基計算拍攝差；時間始終先做整數奈秒差才轉秒。
5. 固定外參為 `T_frontOpt_camOpt`，C 為 optical→FLU。相機拍攝點轉到前相機 reference 的 optical frame：

   `T_refOpt_camOpt = C⁻¹ T_map_frontFLU(t_ref)⁻¹ T_map_frontFLU(t_cam) C T_frontOpt_camOpt`。

   推論的 reference 使用前相機 header，最後再轉至前相機 **bag timestamp**，維持原定位輸出的時間契約。運動插值用原完整 run 的 GNSS／IMU **先驗**，XYZ 線性、旋轉 Slerp；沒有使用 map posterior，也不靜默 clamp 超界時間。

相對運動仍假設 sensor prior 可以代表 front-camera FLU。body lever arm、header 是否精確等於曝光時間及硬體時鐘偏差未獨立標定；這是可追溯的運動補償實作，不是已證明硬體同步。移動車輛／行人也不會因 ego-motion 補償而變成靜態物體。

## 尺度不能只靠「提供內外參」宣稱成功

工具：[validate_rig_reconstruction.py](../scripts/validate_rig_reconstruction.py)、[anchor_rig_scale.py](../scripts/anchor_rig_scale.py)。

以不依賴 depth／map 的 mutual SIFT + Lowe ratio 0.8 配對，計算已知相機 transform 的 Sampson error，再三角化。尺度錨點要求正深度、2–60 m、視差角至少 2°、Sampson error ≤2 px，且同一特徵在前／側預測深度的相對偏差 ≤25%。這些限制為控制小視差與錯配的數值不穩定，不由排行榜搜尋。

取每個訓練 group 的 `log(predicted_depth / stereo_depth)` 中位數，再讓各 group 等權取中位數，得到跨兩條 Track 共用的深度修正 **0.6473887689302475**。只縮放 **camera-frame depth**，相機 baseline、安裝高度與運動平移都不縮放；不能對已含外參平移的整包 PLY 乘這個數字。

| Track／frame | 用途 | 錨點數 | 原深度／三角化深度中位數 | 修正後 |
| --- | --- | ---: | ---: | ---: |
| 1／1505 | fit | 5 | 1.5586 | 1.0090 |
| 1／1823 | fit | 11 | 1.5043 | 0.9739 |
| 2／1397 | fit | 14 | 1.5309 | 0.9911 |
| 2／1661 | fit | 7 | 1.7557 | 1.1366 |
| 1／1506 | holdout | 5 | 1.4613 | 0.9461 |
| 1／1824 | holdout | 10 | 1.4347 | 0.9288 |
| 2／1398 | holdout | 14 | 1.4533 | 0.9408 |
| 2／1662 | holdout | 4 | 1.1952 | 0.7738 |

這揭露了實際限制：三個 holdout group 修正後接近 0.93–0.95，Track2 直行仍約 0.77，不能認定共享尺度已準確標定。少量視差點、動態物體及校正誤差都可能造成偏差；holdout 又只是相鄰影格，並非獨立路段的泛化驗證。此尺度只能當實驗候選，不能取代前相機 scale=3 或上一輪 merged scale。

## 讓側面發揮作用

原理：路面法向主要約束 Z；側面牆面、立面與路緣的不同法向可增加 XY／yaw 約束。單純加入更多側面點，可能讓某一相機的大量路面或植被支配目標。

新增取樣以 `(camera_id, normal 主方向)` 分組。每台相機先 0.3 m voxel，局部 16 鄰居／1.5 m 半徑 PCA 排除缺乏平面支持的點；各非空組共享點數上限，每組再按 2 m 空間格輪流取點，同格優先 confidence 高者。沒有用 map、Track ID、幀區段或分數決定權重。支援不足的組不以重複路面補滿。

這是來源幾何取樣改動；既有 C++ directional 目標、先驗、限幅、IIR 及 CUDA NN 均不修改。保留每點 camera_id sidecar，避免下次又失去逐相機資訊。PCA 本身無法辨認移動車輛或所有植被，品質檢查仍有此限制。

## 小規模公平對照的設計

原始影像為兩條 Track 各直行／轉彎兩個相鄰 group，共 **8 groups、24 張 JPG、24 次模型推論**。選用上一輪僅依感測器運動挑選的片段中心：Track1 1505–1506／1823–1824，Track2 1661–1662／1397–1398。

所有定位使用相同 Release binary、同一 Track 的驗證 map cache、完全相同的原 sensor prior CSV、`heading_rts + directional`、0.3 m input voxel 及同樣優化參數。每個兩幀 window 都獨立初始化，所以沒有把某一候選的上一個 window posterior 傳給另一候選。

| 對照 | 能隔離的問題 |
| --- | --- |
| 原 front raw_ply | 既有算法參考；其重建來源／原生點數不同，不將差異全歸因於相機數 |
| single／predicted／rig | 同 checkpoint／內參／解析度，檢查多視角及外參條件的影響 |
| rig_front／rig | **同一次三視角推論的前相機深度**，只檢查加入側面資料 |
| rig_static／rig | 同一份 motion-conditioned depth，只改是否使用逐相機運動 transform；不是另跑一個無 motion 條件的網路 |
| uniform_matched／balanced | 同一重建、相同匯出點數，只改來源取樣 |
| anchored／rig | 加入稀疏 stereo 深度尺度後的候選；baseline／高度不縮放 |
| 600 點密度控制 | 再控制新的前相機／三相機點數，避免將大量新增點誤當側面幾何本身的功效 |

6000 是新重建的點數上限；原 raw_ply 自身只有數百點。`uniform_matched` 與 balanced 的匯出點數相同，但 C++ voxel 仍可能再合併跨相機重疊點，必須一併報告實際 `input_points`。不能把這張對照表寫成每一組都恰有相同有效點數。

評估同一組 old-front／新 calibrated-front／新 calibrated-side probe 到相同 1 m map 的 capped NN 殘差，以及 data-only normal information。分析 XY／yaw 前先 Schur 消去 Z，避免把高度約束改善當成水平定位改善。這些均不是正式 ATE、Kaggle 分數或已校準 covariance。

## 證據與環境

本機證據入口：`outputs/runs/calibrated_rig_dev_2610181118/`。

- `reconstruction_ampfix/`：完成的 raw reconstruction、per-view NPZ、PLY、frame／input hashes、runtime、固定 rig、SIFT 檢查。
- `anchored/`、`anchored_density600/`：共享深度尺度與兩種點數上限候選。
- `comparison*/`：逐 run 的指令及輸出索引；每個 benchmark run 另有來源／binary／prior／參數／git 差異快照。
- `reconstruction.log`：第一次三視角 OOM；`reconstruction_nocache.log`：AMP scope 包住後處理導致 bf16 linear solve 失敗。失敗結果未拿來比較。

ML 使用主機 `outputs/ml_runtime/venv`，ROS 容器 Python 未修改。模型權重約 4.91 GB，SHA256 `fa06c0fdccefc5048e072c85935d5789b1e36b307f3859033c17f9dcb9fd5201`；checkpoint revision `00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a`，upstream commit `3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9`。PyTorch 2.8.0+cu128／torchvision 0.23.0+cu128，RTX 5060。停用 AMP 權重快取、只在 forward 啟用 bf16，後處理保留 float32；同解析度三視角峰值 **5.344 GB**，推論約 **0.57–0.61 s/group**，重建總計 **49.34 s**（含模型載入約 18.92 s、CPU 取樣及匯出；不含安裝／下載／後續尺度與定位）。

可重現的小規模指令見 [操作手冊](08_MANUAL_WORKFLOW.md#原始影像的校正三相機小規模流程)。目前 source 介面 `--cloud-source reconstruction --ply-scale 1 --source-report ...` 嚴格核對原始 frame index、整數時間、Track、FLU、單位、PLY SHA256 與 outputs 路徑。預設前相機流程與已評分 CSV 保留。


## 完成的定位結果與判斷

共 **64 runs × 2 frames = 128 poses**，全部退出碼 0、輸出有限／完整、128 次地圖更新接受；相同 binary、原始 prior bytes、frame times 與 solver parameters 已逐組核對。Python 測試 **26 項通過**，包含座標軸、rig 高度、深度縮放不縮 baseline、極線公式、尺度 fit／holdout 隔離、側面取樣及來源 hash／時間拒絕。C++／CUDA 未修改，因此沿用既有 Release binary，沒有為 Python／文件改動重建。

以下每列均為相同 8 個原始影格；信息值為消去 Z 後 XY／yaw 的最小 eigenvalue 中位數，越大代表此局部模型在最弱方向提供更多約束，**不是精度提升比例**。probe 殘差單位 m，越小代表所選 probe 與 map 較一致，仍不是 pose 誤差。

| 方法 | 實際 input 點數中位數 | XY／yaw 最小信息 | old-front probe | calibrated-front probe | calibrated-side probe |
| --- | ---: | ---: | ---: | ---: | ---: |
| `front` | 843 | 0.000633 | 0.9307 | 1.2275 | 1.3411 |
| `rig_front` | 5363 | 0.008923 | 1.1675 | 1.0919 | 1.3310 |
| `rig` | 6000 | 0.022340 | 1.1270 | 1.0941 | 1.3175 |
| `rig_uniform_matched` | 5512 | 0.022267 | 1.1254 | 1.0989 | 1.3192 |
| `balanced` | 5396 | 0.029464 | 1.0919 | 1.1083 | 1.3115 |
| `anchored_front` | 2982 | 0.009315 | 1.0748 | 1.1194 | 1.3130 |
| `anchored` | 6000 | 0.016344 | 1.0597 | 1.1258 | 1.3075 |
| `anchored_uniform_matched` | 4197 | 0.016014 | 1.0598 | 1.1258 | 1.3072 |
| `anchored_balanced` | 4120 | 0.019687 | 1.0524 | 1.1219 | 1.3052 |
| `small_anchored_front` | 600 | 0.009711 | 1.0773 | 1.1149 | 1.3112 |
| `small_anchored_uniform_matched` | 594 | 0.012190 | 1.0632 | 1.1211 | 1.3050 |
| `small_anchored_balanced` | 590 | 0.022706 | 1.0247 | 1.1444 | 1.3099 |

可以確定的機制：

- 同一次推論，`rig_front → rig` 的最弱 XY／yaw 信息由 **0.008923 → 0.022340**，側面確實增加方向約束。
- 相同匯出點數的 `rig_uniform_matched → balanced` 由 **0.022267 → 0.029464**；加 stereo 尺度後由 **0.016014 → 0.019687**。
- 600 點控制下，uniform／balanced 的信息 **0.012190 → 0.022706**，但 calibrated-side probe **1.3050 → 1.3099 m** 反而略增。這是「約束更強不等於更準」的實例；有偏的幾何也能讓 Hessian 看起來很好。
- 原 front 在自己的 common probe 上仍最低；新的三相機候選在 calibrated probe 上較好。兩種重建有不同偏差，沒有真值不能只挑對自己有利的 probe 宣稱勝出。**目前沒有充分證據更換已評分最佳版本。**

同步檢查：16 個 front–side 配對中，運動補償後的 Sampson error 中位數有 **11／16** 下降。右相機 **8／8** 下降；左相機 **3／8** 下降，改善不是全面一致。全局時間偏移未重新估計，極線誤差也受外參／特徵定位／動態物體影響，不能因此宣稱硬體同步完成。`anchored_static → anchored` 的 probe 殘差變動很小，尚無定位精度改善證據。

部分 CPU 圖像／map 分析與短測同時執行，registration timings 僅留作執行觀察，不作嚴格效能排名；模型推論時間另於單獨重建流程量測。

## 怎麼確認尺度與同步

1. **先確認固定安裝幾何**：看 `rig_trackN.json` 的 fixed transforms 與相對高度；確認 optical→FLU 只轉一次、baseline 為公尺、沒有套兩次外參。每點保存 camera_id，方便分別查左右。
2. **確認尺度有獨立證據**：看 `scale_anchor.json` 的 fit／holdout、視差角、錨點數及每個 group 的 depth ratio。只用路面高度無法分清尺度與全局 Z；已知 baseline 的非退化三角化才提供 metric 錨點。本次 holdout 的 0.77 是尚未通過的訊號。
3. **確認時間契約**：逐台看原 index stamp exact 對應到哪筆 bag message／header；相減必須用同一時基。NPZ 保留各 header、實際 motion pose 與 header→bag reference transform。不存在的時間直接失敗。
4. **確認補償有幾何支持**：看 `image_geometry_validation.json` 的 static／motion Sampson error、不同路段及不同相機是否一致改善，另看相同 depth 的 static／motion 定位對照。本次右相機較一致，左相機尚未充分支持；若要宣稱硬體同步，仍需曝光／硬體時鐘或可靠外部標定。
5. **最後才確認定位精度**：在更多相互分離的靜態場景檢查 stereo 尺度／極線殘差，排除動態物體並核對 body lever arm；再由使用者手動跑完整 Track、確認 CSV 契約及評分。本次兩份既有最佳 CSV SHA256 已再次核對不變。

結果圖與完整 JSON 位於 [本機圖表](../outputs/runs/calibrated_rig_dev_2610181118/calibrated_rig_summary.png)、[公平對照 JSON](../outputs/runs/calibrated_rig_dev_2610181118/comparison_summary.json)、[尺度錨點](../outputs/runs/calibrated_rig_dev_2610181118/anchored/scale_anchor.json)。outputs 不進 Git，交接時須另行保留。

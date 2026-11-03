# Track2 v5.4：場景證據與因果修正

v5.4 處理三類問題：航向校正透過地圖記憶放大、窄帶地面支持造成新 tilt／scale，以及弱幾何資訊被一律丟棄。它延續 fix2 的一次高度 likelihood 融合及 perf2 加速。**166 項生成／CUDA regression 通過，尚無 v5.4 真實定位或新分數。**

版本為 `track2_causal_covariance_ground_weak_modes_v5_4_perf2`。來源套用狀態見[收據](/home/ricky/sdc_ros2/midterm01_gpu/outputs/run/track00_test_2611021518/apply_receipt.json)，三組合成 PNG/PDF 見[本輪分析](/home/ricky/sdc_ros2/midterm01_gpu/outputs/run/track00_test_2611021518/analysis/README.md)。使用者下一步仍由固定入口手動執行：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
./scripts/run_track2.sh --full
```

## 觀察與改進範圍

[1419 場景還原](/home/ricky/sdc_ros2/midterm01_gpu/outputs/run/track00_test_2611021419/analysis/README.md)將照片、保存的 pose／scale 與地圖 XYZ 對照，沒有重跑定位。idx 308 的大 rebase 可追到 heading calibration；idx 2547 的 ground inliers 集中在左側窄帶，map geometry 被拒後仍輸出 −14.53° roll 與 scale 4.89。

但大於 0.1 m 的 XY rebase 只有 1 幀，地面法向與 IMU prior 差大於 10° 只有 7 幀。這些稀有事件不能解釋全部榜分差。更廣泛的現象是 1475／1705 個已接受更新只保留一個方向，且最長連續 121 幀、4.32 秒沒有接受地圖更新。因此 v5.4 同時研究弱資訊與穩定的有限更新；接受率、rank 或殘差仍不是精度證明。

## 1. 航向 rebase：限制累積敏感度的異常放大

修正記憶 `c=[dx,dy,dyaw]` 與 calibration sensitivity `S` 在校正參考改變時作 `c <- c - S @ delta`。過去把多次不同方向的 `K @ H` 累積，可能得到具有放大效果的斜投影。

令當幀 sensor-prior covariance `C=L @ L.T`，先到白化座標，再將敏感度投影到對稱正半定 contraction：

```text
W0 = inverse(L) @ S @ L
W  = (W0 + W0.T) / 2 = V @ diag(lambda) @ V.T
S_bounded = L @ V @ diag(clip(lambda, 0, 1)) @ V.T @ inverse(L)
rebase = S_bounded @ calibration_delta
```

依據是簡化的獨立高斯資訊模型：若白化後 map information `Lambda` 正半定，其融合響應 `inverse(I+Lambda) @ Lambda` 的 eigenvalue 在 0 到 1。實作以這個集合約束累積敏感度，並記錄投影前後 eigenvalue、projection norm、實際 rebase 使用矩陣。

**這是保守的響應投影，不是完整 joint EKF，也沒有解出完整 sensor/map cross-covariance。**它限制白化後整體響應，不保證每個 XY 分量都比舊版小。正常的 process noise、短暫 hold 與方向性 fade 保留。

## 2. 地面：空間支持、法向一致性與 fallback

先在 CUDA 計算現有 plane inliers 的 3×3 scatter。兩個平面內 eigenvalue 的小／大比例低於 `GROUND_MIN_INPLANE_RATIO=0.20`，即以窄帶／近線狀支持拒絕；這是工程形狀保護，不是標定後精度門檻。

通過形狀檢查後，以當幀 IMU 旋轉建立預期地面法向，將差異投影到兩維 tangent plane。量測法向 covariance 使用支持分布、angular-cell 等效數、0.10 m 模型厚度及 0.5° floor；prior 使用既有未知 IMU／旋轉外參的 roll/pitch 誤差預算，沒有新增實測外參。NIS 超過 13.816（兩維 Gaussian 的 99.9% 參考界）拒絕。

通過才以兩維 Kalman gain 融合法向，再用**同組 inliers 的 median 重新估計 offset**。只改 tilt 而不更新 offset 會製造 scale／XY 不一致，因此兩者一起處理。

拒絕時沿用當幀 sensor-prior roll/pitch、世界 XY/yaw 修正記憶與前筆 scale，並標 `scale_validated=false`。**舊 scale 不保證符合當幀 raw PLY 單位**；若第一幀就失敗，仍只有未驗證的 seed。這項 fallback 保護輸出，不解決缺少可靠地面時的尺度可觀測性。

## 3. 弱幾何：大 variance 融合，真正 null 不更新

尺度 Schur marginalization、當幀 covariance whitening 與 data-only H/g 保留。相對資訊 1.0 不再作所有量測 row 的硬門檻；只有小於 `max(1e-10, max_eigenvalue*1e-8)` 的數值 null 被移除。

每個正資訊模式仍以 `R = 1/lambda + projected_model_floor` 給出 variance，沿用 residual quality inflation。弱模式的 `R` 大、gain 小；不替牆面切線或 scale gauge 製造假資訊。1.0 門檻留作 strong-rank 診斷、memory age 刷新及候選中心的 motion comparison。弱更新仍受 NIS、當幀實際更新限幅與 proposal loss 檢查，不能把舊記憶永久刷新。

一個弱方向的無 prior likelihood center 可能很遠，但 Bayesian posterior 很小。因此不能僅因 center 離開搜尋界就否決全部弱資訊；也不能將局部二次模型的巨大 yaw center wrap 成另一個看似合理的角度。v5.4 保留線性 center，只對真正輸出的 yaw 作角度處理。

## 4. 驗證真正輸出的有限更新，再一起提交

先預覽 data-only height likelihood 的一次 KF，與有限 gain 的 pose proposal 組成實際待輸出的狀態。在原 search bounds 內，使用相同 raw sample／相同 final planes，比較 proposal 與 prediction 的 data-only loss；只在不增加時提交。

拒絕 proposal 時，不提交該筆 pose measurement、height、covariance、sensitivity、last measurement 或 memory refresh。正常的時間 prediction／process noise／記憶衰退仍保留。高度 fix2 的 prior 去重沒有取消；只是把原本的「先更新 pose、再更新 height」改為先預覽、一起驗收。

這仍是固定 association 上的局部檢查，沒有解決完整跨幀、pose/height 與 GNSS/map 相關性，亦不代表全域最優。

## 驗證與限制

- 最終一輪 **166 passed、1 個既有 Axes3D warning，18.93 秒**。涵蓋因果前綴、真 null／弱方向、covariance／Joseph、拒絕不提交、ground 退化、時序與 perf2 bytes／ROS 路徑。詳見[測試 XML](/home/ricky/sdc_ros2/midterm01_gpu/outputs/run/track00_test_2611021518/pytest05.xml)。
- 合成 changing-mode stress 的白化 sensitivity operator norm：舊版最大 1.4075，v5.4 最大約 1（浮點誤差 1e-15）。它是機制數值界限，不是真實位移誤差。
- 人工生成窄帶錯誤地面，並強制拒絕三筆 map 更新以隔離 fallback：舊版輸出 roll −26.89°／scale 11.38，v5.4 保留 prior roll 1.15°／前值 scale 8；恢復可信地面後可重新更新。這不是重跑真實 idx 2547。
- 只使用當幀及已到達的過去觀測、map XYZ。沒有 RTS、backward smoothing、回修舊 pose 或參照 CSV 擬合；未知 lever／外參均值與預算維持原值。
- perf2 reader、camera、ROS、runner 及 Track1／3 不修改。真實 full 與提交由使用者操作；尚不能宣稱接近或達到 Public Score 0.30488。

前版高度模型見 [79](79_TRACK2_CAMERA_HEIGHT_POSITION_20261102.md)，directional memory 背景見 [77](77_TRACK2_V53_FIX1_DIRECTIONAL_MEMORY_20261031.md)，操作見 [Track2 README](../track2/README.md)。

# Track3：模型、限制與驗證

目前設定版本為 `track3_cuda_voxel_v2_armijo_turn`。v2 已通過合成機制與pipeline測試；尚未用真實點雲驗證定位精度，亦無 Kaggle 成績或相對 baseline 的改善證據。

資料流：官方前視 PLY → CUDA voxel 降採樣 → 過去 IMU 與運動預測 → 局部 LiDAR Gaussian voxel 粗配準 → point-to-plane ICP → 一致性檢查與 Kalman 更新 → 立即寫出當前 pose／CSV。既有 pose 不回寫。

## 配準

估計 `T_map_ply=[R,t]` 與正尺度 `s`，點變換 `q=R(s p)+t`。`init_guess.txt` 是有雜訊的起點。初始尺度 3 只沿用 CPU 參考程式預設，沒有當成已知標定；每幀以正尺度的 log increment 與 Gaussian prior 聯合估計。尺度界限0.2–30只是數值保護，並非從榜分選出。

地圖只取 XYZ，分塊讀 ASCII PCD；CUDA 以體素鍵排序、加總與加權合併降採樣。局部地圖以當前預測裁切，移動15m才重建模型；沒有沿完整軌跡切地圖。每個有至少5點的 voxel 建立均值 μ 與 covariance C；特徵值下限為最大值的1%及(0.05 voxel)^2，避免平面退化造成反矩陣爆炸。

8／4／2m 粗配準最小化 Gaussian cell 的 robust Mahalanobis residual：`ρ((q−μ)ᵀ C⁻¹(q−μ))`，ρ是Cauchy。這是自寫 Gaussian voxel 最小平方法，**不是 PCL NDT likelihood 或 More–Thuente 的逐行CUDA移植**。1m細配準用最小特徵值方向為normal，殘差 `nᵀ(q−μ)`，同樣使用Cauchy權重。對應在相鄰27 voxel 的mean中選最近者；這是受限voxel對應，不是全地圖精確KNN。

未知增量 `[δt, δω, δlog(s)]`，Jacobian為 `[I, −[R s p]×, R s p]`；CUDA累加7×7 normal equations，CPU只解這個小系統。步長限制2m／5°／0.1 log-scale維持數值穩定。Armijo回溯最多二分試算20次，只接受滿足足夠下降條件的候選；收斂檢查使用未乘回溯比例的proposal，避免極小步長讓未完成的更新被誤判為收斂。代價對無對應點加固定penalty，避免只剩少量對應就顯示低殘差。最終須收斂、coverage≥0.3，且消去scale後的6維空間Hessian不可退化；純平面不能被當成完整pose觀測。coverage、NIS及可觀測性門檻未因這次修正放寬。這些都是機制檢查，不是精度證據。

局部模型若因稀疏voxel而支援不足，使用typed sparse-support結果拒絕當前frame，不讓它變成fatal例外；暫存模型建置完成前不替換既有cache。CUDA或其他數值錯誤仍照常回報，不被當成稀疏支援不足吞掉。拒絕該幀表示沒有地圖更新，不保證後續預測仍準確。

## 因果 IMU 融合

狀態 `[p_xyz,v_xyz,ψ,b]`，b為垂直陀螺儀bias。只用因果可用的gyro計算平面轉率 `(ω_vertical−b)`，不注入IMU絕對heading。gyro不能單獨確定行進距離，速度仍需由可信的點雲位置觀測建立。gyro有效時，world平面速度隨yaw同時旋轉，位置用constant-turn運動的圓弧積分推進；`z`仍以constant velocity推進。線性化包含gyro bias對位置／yaw的Jacobian，gyro角速度雜訊的共變異也傳入濾波器。當gyro超過新鮮度期限時切回constant-velocity，避免持續外推舊轉率。未標定加速度不作二次積分。這依賴IMU orientation的world z與地圖重力方向一致、車輛近似平面運動；它不是完整camera–IMU外參標定或15-state慣導。roll/pitch沿用接受的3D配準。

ROS bag唯讀，只選`/imu/data`，不讀GNSS；header與bag timestamp都不晚於當前frame才可使用，延遲量測不回溯。以整數ns維持精度。官方bag查核顯示11307筆有效quaternion，但covariance全部0（未知），因此不偽稱有感測器原始噪聲標定。

地圖觀測 `[p,ψ]` 經4維NIS檢查（18.47，約χ²₄ 99.9%），以Joseph covariance更新，角差wrap。初次初始化不套行進中的jump gate。位置與yaw觀測噪聲0.3m、0.035rad以及其他process noise是明列的初始模型假設，尚未用獨立真值校準；有新證據才調整。地圖觀測拒絕時輸出當前因果預測。零map更新或零usable IMU會記failed；大量拒絕仍應讀observations判斷失追，不能只因寫完CSV就提交。

## RViz 觀測的解讀

每次定位預設先啟動RViz。它發布 `/track3/map`、`/track3/submap`、`/track3/current_cloud`、`/track3/pose`、`/track3/path`，並提供 `map → track3_camera` TF。地圖一次發布；當前點雲先用本幀相同的估計轉換及尺度轉到map frame，再發布；pose與結果CSV亦使用同一 SE(3) 轉換。每輪在run目錄使用 `rviz.runtime.rviz`，避免把RViz互動後的設定寫回程式快照。

俯視LiDAR地圖可幫忙辨認道路走廊、轉彎與大致地圖形狀，但不含逐幀真值或時間軌跡，不能由輪廓反推出某幀的正確位置與朝向。看RViz時要同時看本幀 `map_accepted`：拒絕期間的pose/path是IMU運動模型預測，屬 prediction-only，不是可靠的map pose。run結束後RViz會留在最後收到的狀態；該畫面本身不能證明run成功。

## Frame 與評分契約

提供的LoGeR exporter將camera RDF轉成camera-own FLU（X前、Y左、Z上）；目前沿用官方init_guess與CPU範例，將輸出定義為`T_map_ply`。原始PLY的完整生成manifest、評分參考點及相機到車體translation未被提供，**不能宣稱已核對車體pose**。三相機間外參也不能代替IMU／車體外參。

CSV逐frame輸出`idx,x,y,yaw`，0-based，`yaw=atan2(R10,R00)`弧度；TXT每行12個row-major 3×4值。這與原始轉檔範例一致。規則的評分角誤差以degree描述，並不代表CSV輸入必為degree；尚需Kaggle官方submission模板／evaluation約定核對。不能用榜分反推此契約。

## LoGeR 的角色

下載位置`vendor/LoGeR`，checkpoint SHA256：`68a0ae71408eb5605b8409296e7d5bf43c87853595766195257af97c5e60373c`。外層ZIP CRC通過並刪除；內層`sdc2026.zip`已另解到archive_sources後刪除。包內MANIFEST的`run_pipeline.sh`有一筆不符，其他列示來源通過；保留原manifest與實際SHA，沒有改vendor來假造一致。

預設使用官方`raw_ply`：不必再次跑LoGeR。選`--source loger`時，新的`CausalLoGeR`只持有至當前的2張影像、rectify校正與ego mask、呼叫模型、只匯出最後一幀local_points並轉FLU。過去影像可以重新參與當前重建，但舊PLY／pose不回寫；每幀輸出存run/reconstructed_ply。這與提供的offline W32生成策略不同，也尚無實際精度比較。LoGeR重建與配準依序執行，不並行爭用GPU。8GB顯存採混合bfloat16，TTT lr_fc及明確停用autocast的point/conf/camera heads保留FP32；記憶體不足會明確失敗，不能偷偷CPU回退。

套件依據：[CuPy安裝文件](https://docs.cupy.dev/en/stable/install.html)、[PyTorch CUDA12.8版本配對](https://pytorch.org/get-started/previous-versions/#v280)、[LoGeR原作者repo](https://github.com/Junyi42/LoGeR)。本專案使用老師提供的修改版，以下載內容和來源SHA為準。

## 本輪實際驗證

v2全套pytest：24 passed、1 warning（4.00s）。包含Armijo下降及tiny-alpha收斂判定、constant-turn圓弧積分與有限差分Jacobian、gyro過期退回constant-velocity、稀疏模型cache、ROS生命週期與snapshot import、合成三幀pipeline，以及3142筆合成格式畫圖。warning是未使用 `Axes3D`（系統與venv混合環境），2D圖正常。另有LoGeR checkpoint strict load與1／2張合成隨機RGB推論檢查，峰值GPU tensor allocation約3.24GB；未使用真實影像，v2亦尚未跑真實點雲定位。原包無自編RoPE extension，使用PyTorch CUDA運算版本。早期視窗參數與混合dtype失敗及修正紀錄均保留於環境run；不把它們列為成功結果。

## 真實定位失敗與有限診斷

既有 `track03_full_2610241507` 輸出2776筆後因局部稀疏voxel例外中止。map觀測只接受12幀，最後一次為 `idx=569`；`570–2775` 的2206幀都未接受地圖觀測。來源SHA 31筆相符。有限真實診斷授權上限30幀：第一次新增儀表版本在frame 0後因 NumPy `bool_` 序列化錯誤中止，該幀計入授權；修正後的 run1547 執行 `idx=0–28` 共29筆，未跑 `idx=29`。其中2筆接受地圖更新；25個細ICP `backtracking_failed` 案例在小於 `1/8` 的步長有下降cost候選。診斷run的CSV與失敗full前29列逐欄相同，故這是v1儀表診斷，不是修正後精度改善證據。來源紀錄：`outputs/run/track03_test_2610241535/{failure_audit,diagnostic_summary,authorization_usage}.json`、`outputs/run/track03_test_2610241547/`。v2合成檢查通過不代表已修復長序列失追；新的真實定位仍須由使用者另行啟動。

## 左右魚眼相機的補充查核

使用者指出左右相機的魚眼效果及車後側出現在畫面的位置。本次只讀標定檔：兩側皆1440×928、fisheye模型；原始左主點約(709.92,469.14)px、右主點(716.51,463.97)px，均接近影像中央。三相機外參顯示側光軸相對前視光軸約左71.82°／右73.10°；這不是已標定到車體的絕對heading。畫面內車尾的位置、光軸朝向、主點、鏡頭安裝位置是不同概念，不能據目視把cx移到畫面左側。

目前raw_ply流程不讀側相機；這項資訊留作多相機整合依據，未把它當成本次失追原因。將來整合時，各自用K／D先處理影像投影，再用T_front_front_left／right轉到共用frame，且遵守影像不同時間戳的因果同步；不對已生成的PLY重做二維魚眼校正。原始標定、影像、PLY均未改動。

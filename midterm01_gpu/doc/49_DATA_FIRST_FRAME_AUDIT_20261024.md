# 49｜Track3 首段幾何與 v2 配準失敗分析（2026-10-24）

## 結論

目前證據不支持把錯誤歸因於「XYZ 軸整體交換」。Track3 原始 PLY 的相機座標較符合 **FLU（X 前、Y 左、Z 上）**；但首幀初始姿態與地圖道路候選面的傾角不合，尺度和高度也互相矛盾，v2 配準器從第一段就頻繁未收斂。這足以解釋目前軌跡不可信，尚不足以定出正確姿態、尺度或 Kaggle 車體座標契約。

## 軸向與幾何證據

用提供的相機標定把 PLY 投影回同時間的前視影像，FLU 投影式為 `u=cx−fx·Y/X`、`v=cy−fy·Z/X`。frame 0 有 97.74% 點落在 rectified 影像內，像素 RGB 誤差中位數為 9.67/255；把點解讀成 X 前、Y 右、Z 下時，中位數為 71/255。直接當成 RDF（X 右、Y 下、Z 前）時，只有約 24.5% 點有正深度，rectified 影像內投影比例為 0%。這支持 PLY 使用 FLU，不能據此確認左右相機、完整車體外參或 Kaggle 評分座標；不要盲目交換軸，也不要把 RDF→FLU 再轉一次。

在地圖 init pose 附近的 XYZ 區域，以 0.25 m voxel 抽樣後擬合的道路候選面法向為 `[-0.0065, 0.0114, 0.9999]`，init XY 處高度約 8.091 m。`init_guess` 矩陣原點的 map Z 為 10.384 m，roll/pitch 約 14.15°/14.92°。frame 0 PLY 的主要平面法向在套用該初始姿態後，與地圖候選面差 **18.48°**；frame 3、7 若也固定套用 init 姿態，分別約差 18.64°、18.62°。idx 7 實際估計 pose 雖被接受，對應平面仍差約 9.62°，相機離候選面約 1.64 m。這指出初始 roll/pitch 或資料幾何仍有顯著不一致，但道路面只是幾何候選，並非語意真值。

尺度同樣未標定：`scale=3` 是沿用 CPU 範例的預設值。frame 0 來源平面離 PLY 原點約 0.265 raw units，乘 3 後約 0.795 m；這與 init pose 相對地圖道路候選面的約 2.29 m 高差不相符。由於 init pose 本身有誤差，且相機／車體參考原點仍須確認，不能把這個比例直接換算成新的 scale；單靠該平面也無法同時解出尺度、高度、XY 與 yaw。

## v2 為何第一段就停住

原始地圖約 5,736 萬個 XYZ 點；這次只讀取資料做幾何分析，沒有執行定位或 GPU 配準。配準程式使用 PLY 的 XYZ，並未把清楚的 JPG 紋理當作影像特徵。frame 0 的 33,989 點經最粗 8 m NDT 階段、約 2.67 m source voxel 抽樣後只剩 43 點，細 ICP 階段才有 559 點；可見相片清楚不代表粗配準保留了足夠道路邊緣與建物結構。

`track03_full_2610241610` 現有 217 筆完整 observation（idx 0–216）中只有 idx 7 採用地圖；201 筆配準以 `backtracking_failed` 結束、14 筆達到迭代上限。coverage 多數仍高，故主要問題是未收斂，不是可見點覆蓋門檻。idx 0–6 的輸出 x/y 停在初始位置；首次採用地圖的 idx 7 相對前幀跳約 8.09 m，yaw 也跳約 63°。這段期間沒有地圖更新時，濾波器保持零初速度的位置預測；初值有誤差，因此不能把 idx 7 的跳變解釋成真車移動。另有 idx 31 配準器回報收斂，但 NIS=70.37 而遭融合濾波器拒絕。該次 run 的 `test_info.json` 仍記為 `running`／`frame_count=0`；這些前綴資料不是完整 full 成果，metadata 也未在本次分析中改寫。

程式用 Euclidean 最近 voxel mean 選 correspondence，但粗階以 Mahalanobis 殘差、細階以點到平面殘差最佳化；切換 voxel 也會切換 covariance／normal，可能使實際 cost 不平滑，造成 line search 難以接受步長。idx 0 細階最小試算因子 `1.91×10⁻⁶` 的 cost 仍略高於該步基準值（0.051432 對 0.051263）。這是程式設計上的不一致及合理失敗機制；log 沒保存每個候選 pose 的 correspondence ID，不能宣稱已證明所有失敗幀都由此單一原因造成。24 項合成檢查通過只驗證機制，不能證明真實定位精度。

## 後續修正方向與限制

下一版可先把當前點雲道路法向作為 roll/pitch 初始約束候選，再以同一座標系下的地圖幾何聯合處理尺度與相機高度；讓 correspondence 選擇度量和最佳化殘差一致，並在粗階保留更多非平面結構。這些是待實作與驗證的方向，不是已修好的結果。**目前不建議原樣重跑 v2 full 或提交其輸出。**本分析沒有修改程式、定位輸出或原始資料，也沒有產生精度或 Kaggle 成績。

幾何診斷只比較 frame 0、3、7 與 init 附近局部地圖；重投影假設提供的相機標定正確。它沒有驗證完整車體外參、Kaggle frame/yaw 契約，也沒有提供 XY/yaw 真值。

## 證據

- [唯讀幾何分析數值與輸入 SHA](../outputs/run/track03_test_2610241617/data_analysis_0002/report.json)
- [full 前綴與輸出 pose 對照摘要](../outputs/run/track03_test_2610241617/data_analysis_0002/diagnostic_metrics.json)
- [原始點雲軸向分布圖](../outputs/run/track03_test_2610241617/data_analysis_0002/01_raw_cloud_axes.png)、[影像重投影比較](../outputs/run/track03_test_2610241617/data_analysis_0002/02_image_reprojection.png)、[初始姿態幾何差異](../outputs/run/track03_test_2610241617/data_analysis_0002/03_initial_geometry_mismatch.png)
- [v2 可讀 observation 前綴](../outputs/run/track03_full_2610241610/observations.jsonl) 與 [該次凍結配準程式](../outputs/run/track03_full_2610241610/source/track3/gpu.py)

附帶的走廊方向檢查：init附近18m圓盤內、落在道路候選面的XY點，形狀主軸約166.81°（只有無向軸，另有相差180°的方向）；init yaw145.68°，idx7輸出yaw為−151.04°。主軸受支持區域、道路寬度與分岔影響，只可作幾何方向提示，不能當成車輛heading真值，更沒有據此修改任何pose。

# v5 成績與 heading 差異判讀

## 目前能確認什麼

v5 含 IMU full 與 v4 不含 IMU full 都成功輸出 3142 幀；使用者回報 v5 成績 1.29、v4 成績 1.26。使用者提供的外部 Track3 CSV 回報成績 0.65。三個數字在工作區都沒有可核對的官方成績回執或提交檔對應，因此以下把它們當作使用者回報，不當作已驗證成績。

官方規則使用的位置 RMSE 加上 `0.2 ×` 旋轉 RMSE（度）；Public 評估全軌跡前 5–40% 的 prefix，實際 prefix 長度未知，Private 評估剩餘部分。frame 2200–2600 位於全序列約 70–83%，不屬於這條規則所述的 Public 評估前段。因此，即使這段 v5 看起來改善很多，也不能預期 Public 分數一定反映它；不能從「5–40%」推定具體截止 frame。來源見[官方整合說明 PDF，第 20、28 頁](<../../midterm01/docs/Competition Regulations/00_SDC_2026_Midterm01_Localization_整合新版_20261022.pdf>)。

結果依序回答兩件不同的事：整體俯視路徑接近，不代表每幀 heading 接近；而一個轉彎區改善，也不代表計分使用該區或整個誤差都改善。旋轉 RMSE 在官方公式中單獨有權重，故 heading 差異是「成績可能變差」的一個合理方向；但現有資料沒有官方 GT，不能將它量化為本次 0.03 差額的實際原因。

## 唯讀比較設計

分析使用既有 v4、v5 的 `result_3.csv`，以及使用者提供的[外部參考 CSV](<analysis doc/example_track3_0.65.csv>)。外部檔 SHA256 為 `f0a1b75a40756d0b1cfe7bf8f50a3d24faab1cc10bc209ee9866b06cc0947682`；它有 3142 筆、`idx=0..3141`，欄位為 `idx,x,y,yaw`。原始來源、frame／時間對應與座標單位沒有配套 metadata。分析按相同 idx 配對，假設 XY 是 map 公尺、yaw 是弧度；這些是假設，不是已確認契約。

這份 CSV 是另一組預測，不是 ground truth。分析沒有做座標對齊、偏移估計、旋轉擬合、參數搜尋、平滑或 pose 回修，也沒有計算官方分數。v4 與 v5 定位 run 的 `frame_manifest.csv` SHA256 完全相同（`c26dc7f72c4b3e607a8dbcdf06117c1f27a391a7b4832e4d6f7192f5da51c8d8`）；比較工作只讀取既有輸出。完整圖與逐幀值見[唯讀分析 README](../outputs/run/track00_test_2610251302/analysis/README.md)。

## 數值結果

下表是對外部預測的差異中位數，不是位置／姿態誤差：

| 範圍 | v4 XY 差異 | v4 絕對 yaw 差異 | v5 XY 差異 | v5 絕對 yaw 差異 |
| --- | ---: | ---: | ---: | ---: |
| 起始 frame 0–299 | 0.3637 m | 0.7474° | 0.4572 m | 1.5465° |
| 轉彎 frame 2200–2600 | 6.4664 m | 6.3132° | 0.9544 m | 0.8107° |

這顯示 v5 的轉彎區預測與外部檔案更接近；起始段的 XY 差異相近，但 v5 yaw 差異較大。由於外部檔案不是 GT，不能把「更接近外部預測」改寫成「定位更準」。整體與轉彎圖見[01](../outputs/run/track00_test_2610251302/analysis/01_trajectory_comparison.png)，heading 見[02](../outputs/run/track00_test_2610251302/analysis/02_heading_comparison.png)，差異與 stationary 狀態見[03](../outputs/run/track00_test_2610251302/analysis/03_difference_and_stationary.png)。

v4 和 v5 版本同時改了 fusion、尺度與 stationary 策略，並非只開關 IMU 的單變因實驗。因此，僅憑 1.29 與 1.26 的回報，不能說「IMU 本身造成 0.03 退步」；也沒有依 0.65 外部軌跡擬合參數的依據。

## 長停區 heading 狀態

v5 在 frame 929–2179 的 stationary hold 分成四段。每一段內位置與 heading 固定，但重新進入 hold 時 anchor 已改變：

| Hold 範圍 | 幀數 | Anchor yaw |
| --- | ---: | ---: |
| 929–954 | 26 | 160.9823° |
| 1031–1133 | 103 | 164.9216° |
| 1173–1842 | 670 | 165.8942° |
| 1852–2179 | 328 | 167.2167° |

這也解釋 `max_consecutive_map_rejections=670` 的含義：最長 stationary hold 本身就有 670 幀；該摘要不能被解讀為連續 670 幀 map registration 失敗。

資料記錄了四次離開 hold 的觸發條件：

| 解除 frame | Gyro 最大值 | 相鄰影像 flow P90 | Baseline flow P90 | 越過的 quiet check |
| ---: | ---: | ---: | ---: | --- |
| 955 | 0.0121 rad/s | 0.8278 px | 0.4545 px | adjacent P90 > 0.75 px |
| 1134 | 0.0121 rad/s | 1.5264 px | 0.3653 px | adjacent P90 > 0.75 px |
| 1843 | 0.0118 rad/s | 0.0602 px | 1.2522 px | baseline P90 > 1.25 px |
| 2180 | 0.0100 rad/s | 0.1651 px | 1.2940 px | baseline P90 > 1.25 px |

四個 gyro 值都低於 0.02 rad/s。frame 1843 的 baseline P90 只比 1.25 px 門檻高約 0.0023 px，9 幀後 frame 1852 又進入 hold，並建立下一個 yaw anchor。這支持「模式切換對 P90 邊界敏感」的判讀，但不能判斷 flow 來自車身真實微動、畫面內動態物體或追蹤累積誤差；這些仍待檢驗。

依 v5 自身 pose trace 計算，929–2179 heading 總變化為 +6.2344°，其中 IMU 與視覺速度 prediction 帳列 +0.5207°，map measurement update 帳列 +5.7136°。這是儲存狀態的差分分解，不是真實車輛運動。通過 map update 的幀上，final yaw 與 registration candidate 的絕對差中位數為 0.9053°、P90 為 2.3354°；平順的 final trajectory 不等於較準，因為融合後的 yaw 可能落後於 candidate，也可能抑制錯誤候選，沒有 GT 無法分辨。

## 工作假說與下一步

目前可由證據提出、但尚未證明的兩個待查方向是：

1. 起始段 heading 與外部參考的差距，可能來自初始 heading 或 fusion 的偏差／滯後。
2. 長停中的解除與重新進入 stationary hold 之間，map update 可能改變下一段 anchor；frame 1843 的 P90 僅稍高於門檻，顯示解除判斷值得核對原始 tracks 與事件記錄。

下一步應以本機 IMU、影像 tracks、配準 candidate 和 filter event 逐幀核對這些觸發，不依外部 0.65 CSV 反推外參或濾波參數。這份離線分析沒有用未來資料回改 pose，符合因果定位限制。v5 full 已完成；不建議原樣重跑同一版本期待分數改善。後續參數若要調整，仍應直接修改 Python 參數檔，並根據新增的自有 sensor 證據一次釐清一個因素。

## 證據檔

- [分析輸出與來源 SHA manifest](../outputs/run/track00_test_2610251302/analysis/manifest.json)
- [完整比較指標與方法旗標](../outputs/run/track00_test_2610251302/analysis/comparison.json)
- [逐幀差異 CSV](../outputs/run/track00_test_2610251302/analysis/per_frame_differences.csv)
- [v4 full metadata](../outputs/run/track03_full_2610242203/test_info.json)；[v4 結果 CSV](../outputs/run/track03_full_2610242203/result_3.csv)
- [v5 full metadata](../outputs/run/track03_full_2610251201/test_info.json)；[v5 結果 CSV](../outputs/run/track03_full_2610251201/result_3.csv)

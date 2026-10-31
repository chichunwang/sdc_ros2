# Track2 配準耗時與 GPU 工作

Track2 的點雲對應、局部平面估計、殘差與配準累加已使用 CUDA。正在執行的 `track02_full_2610311445` 前 255 幀平均每幀配準約 0.99 秒，約占定位時間 67%；載入階段（含 sensor prior、點雲讀取與遮罩）平均約 0.17 秒。這表示配準是目前已記錄的主要耗時，但現有紀錄沒有 CUDA kernel、CPU 呼叫或資料傳輸的分項，還無法判斷配準裡哪一段硬體工作最慢。

這次先製作新版 `track2_causal_dual_heading_bias_v5_2_perf1`，定位參數、門檻和迭代上限都不變。修改暫存在本次 `track00_test_2610311449/source/`；執行中的 full 使用自己的快照，因此本次 full 不會載入這些修改。等 full 結束並確認程式 SHA 後，使用者可在 Ubuntu 主機的 `midterm01_gpu/` 執行：

```bash
python3 outputs/run/track00_test_2610311449/finish_update.py
```

此入口會先確認定位程序與 GPU lock 已釋放，再建立唯一的測試 run、執行 CUDA 合成等價檢查和速度比較，成功後才套用新版程式。若 full 仍在執行，它會停止在檢查階段；此時不要修改 live 檔案。它不會啟動真實 Track 定位。

## 省下哪些重複工作

Gauss–Newton（高斯–牛頓）配準會反覆嘗試小幅 pose 更新。每次線搜尋候選只需要比較 cost（目標函數值）；之前的程式也計算未被候選採用的 Jacobian，並在固定對應點時再算一次距離，最終還為每個候選排序殘差以輸出中位數等診斷。新版讓候選略過這些診斷計算，只在需要梯度的步驟建立 Jacobian，並在完整評估時保留原有診斷欄位。配準的門檻、資料權重、線搜尋與最大迭代次數均未調整。參數仍集中在 [`track2/parameters.py`](../track2/parameters.py)，本次僅改 `VERSION`，未改 `METRIC_STAGES` 或觀測雜訊。

相機遮罩原先在 CUDA 投影點雲後，先把保留點搬回 CPU，runner 又把點傳回 GPU 配準。新版將遮罩後點雲留在 GPU，並在 Track2 `FrontMask` 內快取遮罩。RViz 顯示需要的逐點 pose／scale 轉換也移到 CUDA；最後仍須把 float32 點資料送回主機以建立 ROS 訊息。這是顯示路徑變更，不會參與配準或改變 pose。

每筆 observation 另記 `phase_timings_ms`，拆出感測器 prior、PLY 讀取、CUDA 遮罩、來源 SHA、配準和 RViz 顯示。配準內的計時會同步 CUDA stream，分列準備、優化與輸出品質評估的 wall time；數值仍包含 Python 控制與 CUDA 等待，不能直接解讀成純 CPU 或純 GPU kernel 時間。CuPy 工作具有非同步特性，效能比較需先暖機並同步計時；官方文件說明了這項量測限制與 `asnumpy` 的裝置至主機傳輸介面：[CuPy 效能量測](https://docs.cupy.dev/en/v13.6.0/user_guide/performance.html)、[cupy.asnumpy](https://docs.cupy.dev/en/v13.6.0/reference/generated/cupy.asnumpy.html)。

## 為何仍有 CPU 工作

PLY 檔案解碼與 SHA256 是磁碟及主機工作；小型 4×4 線性系統與 sensor filter 也不值得為 GPU launch 和來回傳輸而搬上去。ROS PointCloud2 最後要在主機建立訊息 bytes。大型點雲遮罩、座標轉換、最近鄰搜尋和配準資料累加則留在 CUDA，不使用 CPU fallback。

## 驗證狀態

更新後的 CPU 代數與契約檢查為 49 passed（4.35 秒），涵蓋目標函數／導數等式、causal covariance 和 ROS 顯示介面；它們沒有執行 CUDA，不能證明 GPU 等價、實際加速或真實 Track 精度。CUDA 合成等價、輸出契約及新版對舊版速度比較尚待目前 full 結束後執行。尚無新版真實定位或精度結果。

細節與證據：

- [CPU 檢查紀錄](../outputs/run/track00_test_2610311449/cpu_pytest_0003.xml)
- [前綴計時來源與 SHA 摘要](../outputs/run/track00_test_2610311449/analysis/README.md)
- [新版配準實作](../track2/metric_registration.py)
- [新版 Track2 相機遮罩](../track2/camera.py)
- [新版逐幀計時與 RViz 路徑](../scripts/run_track2.py)

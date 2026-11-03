# Track2 無損 ASCII PLY 與 ROS payload 效能整理

記錄日期：2026-11-02　｜　版本：`track2_causal_time_reference_likelihood_v5_3_fix1_perf2`

本記錄描述 I/O／序列化層的效能修正與驗證。本輪套用項目限於下列兩項改善；最後實際套用的檔案與 SHA256 以 [apply receipt](../outputs/run/track00_test_2610311820/apply_receipt.json) 為準。此版本沒有 perf2 真實定位結果。

## 觀察與原因

Track2 在配準之前要逐值讀取 ASCII PLY，並於 Python 層做資料檢查；發布資料前還要建立 ROS 點雲訊息。profile 顯示主要時間花在 CPU parsing 與 Python 資料檢查。這些工作發生於 I/O 與訊息建構路徑，將它們移至 GPU 不會處理同一個瓶頸。

## 本輪作法

1. **ASCII PLY reader：**用 NumPy structured dtype 讀取數值欄位，先使用 `f4`，再轉成 `f8`。先保留 float32 的原始量化語意，再供下游使用 float64。finite filter 與點順序維持原狀；不支援或未知格式仍走舊 reader，避免擴大格式假設。
2. **ROS Track2 payload：**以 `array('B')` 建立 `PointCloud2.data`，避免 ROS 2 Humble 的 `bytes` setter 逐元素型別檢查。修改只套用於 Track2；shared helper 的 default 保留。
3. **演算法不變：**數值參數、solver、濾波、GNSS 時間邊界與 CUDA 配準均未改。

## 驗證結果

| 檢查 | 結果與範圍 |
| --- | --- |
| Reader parity | 3299/3299 輸入 PLY SHA 已核對；XYZ bytes 3299/3299 相同。見 [parity summary](../outputs/run/track00_test_2610311820/reader_parity_summary.json)。 |
| 元件基準 | 六個真實影格，以已保存 pose／scale 測讀檔、遮罩、顯示轉換、訊息建立與序列化；每 variant 暖機 1 次、實測 3 輪。baseline 465.1383 ms、ROS only 161.8969 ms、PLY only 317.7547 ms、合併 14.9455 ms（皆為六幀 per-frame median 的平均）。見 [benchmark](../outputs/run/track00_test_2610311820/component_benchmark.json) 與[分析圖](../outputs/run/track00_test_2610311820/analysis/README.md)。 |
| 合併版比較 | 此元件範圍內節省 96.79%，31.12×。未做 registration、ROS publish 或 pose estimation，不能外推 full 定位速度。 |
| ROS 等價 | message fields 與 payload bytes 等價；CDR 僅在 normalize alignment padding 後相同，不宣稱 raw CDR bitwise 相同。 |
| 回歸 | [regression XML](../outputs/run/track00_test_2610311820/regression_pytest.xml)：141 passed、2 deselected（舊 C++ oracle）。[CPU pytest log](../outputs/run/track00_test_2610311820/cpu_pytest.log)：71 passed，屬前者子集，不相加。[equivalence XML](../outputs/run/track00_test_2610311820/equivalence_pytest.xml)：1 passed synthetic runner。 |
| 輸出 byte parity | synthetic equivalence fixture 的 `result_2.csv`、`poses.txt`、`pose_trace.jsonl`、`frame_manifest.csv` 在 legacy/candidate 間 bytes 相同；此項不是兩次真實 full 的結果比較。 |

## Full 紀錄與限制

既有 `track02_full_2610311612` 完成於 v5_3；六項輸出及 155 個 source SHA 已核對，見 [audit](../outputs/run/track00_test_2610311820/completed_full_audit.json)。`track02_full_2610311843` 是 v5_3_fix1，3299 幀、localization 3038.67493 秒，score 為 null，且沒有 perf2。既有 full 證據不因本次效能修正而改寫。

截至 2026-11-02，perf2 沒有真實定位，也沒有 full 速度或精度分數。完成主線套用核對後，請使用者在 Ubuntu 主機手動執行：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu && ./scripts/run_track2.sh --full
```

本 run 的 benchmark、parity、pytest、舊 full audit 與兩張圖均保留於 [analysis 記錄](../outputs/run/track00_test_2610311820/analysis/README.md)。本文件的相對連結以 live `midterm01_gpu/doc/` 位置解析；程式快照另存於該 run。

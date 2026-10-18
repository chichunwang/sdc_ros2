# 26｜顯存換速度的實測（2026-10-18）

這輪測試沒有找到值得切換的 dense head batch／快取設定。batch 2／3 多用顯存，但整幀速度差異約在 1% 內，且輸出不再逐位元相同。PCA chunk 256 → 1024 的取樣階段快約 **21%**，四個真實場景選點相同；節省約 **13.8 ms/frame**，以本輪約 0.72 s/frame 換算約 **1.9%** 的整幀改善空間，這個整幀估算尚未實測。

目前使用者完整一輪繼續採用原加速版設定，沒有熱改算法或 execution identity。Track1 已完成 3011 幀重建及定位；短測結束後已恢復同一個 wrapper，Track2 繼續執行。沒有新 Kaggle 分數，三相機尺度候選的驗證仍未通過。

## 方法與測試範圍

- RTX 5060 8GB、同 checkpoint／程式版本、三視角 448×294、float32 weights＋bf16 forward、AMP weight cache 關閉。候選深度尺度 0.67398047、原 heading_rts prior、校正及運動補償不變。
- 真實原始影像四組：Track1 1505–1506、Track2 1661–1662。模型載入一次，每個 phase 先暖機一組，再量四組；兩輪交錯 `baseline → batch2 → baseline → batch3 → baseline → retain_cache → baseline → batch3+retain_cache → baseline`。
- 共 **72 筆實際模型量測**：baseline 40 筆，其餘各 8 筆。沒有模型 OOM。每組保存時間、CUDA peak allocated／reserved、depth/confidence/mask 數值比較、選點與 PLY／labels hash。warmup、模型載入、來源初始 hash 及額外比較統計不計入逐幀時間。
- `retain_cache` 在獨立程序內省略 `torch.cuda.empty_cache()`，包括模型內部與幀末清理，**沒有開啟 AMP weight cache**。每個 phase 結束恢復函式並回收快取；不改 upstream 檔案。逐幀計時與原入口一樣不含幀末清理，故不代表完整 loop wall time。
- PCA 另外在同一個 resident model 下，固定真正 baseline 預測得到的點雲，測 chunk 256／512／1024；各場景四次，第一筆不計入暖機統計，共 12 個場景／chunk 組合、48 次取樣。

**Allocator 範圍差異：** 本次獨立 probe 沒有帶入完整 wrapper 的環境變數，使用預設 native allocator；完整 wrapper 使用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。因此 batch／cache 是同一個 allocator 內的公平對照，但快取結果不能直接當作完整入口 allocator 設定下的結論。實際環境保存於 `allocator_environment.json`。後續 scheduler 已補上與完整入口相同的環境設定，probe 也會記錄 allocator；這個補強只通過語法檢查，尚未重跑模型，不能把本輪數字改稱 expandable-segments 實測。

## 模型 batch 與配置快取

時間為暖機後實測平均。候選變化百分比對照同一 frame／repeat 前後兩個 baseline，避免直接用不同 phase 的平均算倍率；約 1% 內的差異不足以主張穩定加速。

| 設定 | 重建平均 | infer 平均 | peak allocated | peak reserved | 相鄰 baseline 的耗時減少 | 輸出 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| batch 1，原快取策略 | 721.4 ms | 584.3 ms | 5.01 GiB | 5.19 GiB | — | 40 筆皆相同 |
| batch 2 | 723.5 ms | 592.1 ms | 5.23 GiB | 5.67 GiB | −0.92% | depth／mask／選點／PLY 不同 |
| batch 3 | 718.6 ms | 587.0 ms | 5.46 GiB | 6.06 GiB | +0.13% | depth／mask／選點／PLY 不同 |
| batch 1，保留配置快取 | 726.1 ms | 587.8 ms | 5.01 GiB | 5.31 GiB | −0.08% | 全部相同 |
| batch 3，保留配置快取 | 719.0 ms | 583.0 ms | 5.46 GiB | 6.07 GiB | +0.84% | depth／mask／選點／PLY 不同 |

這裡的 GiB 是 bytes／2³⁰。`peak allocated` 是 Torch tensors／受 allocator 管理的配置峰值；`peak reserved` 包含保留的配置空間，不等於 nvidia-smi 的整張卡占用。PyTorch 使用配置快取降低配置成本，`empty_cache()` 釋放未使用快取，不會釋放仍被 tensors 持有的記憶體，見 [PyTorch 2.8 記憶體管理](https://docs.pytorch.org/docs/2.8/notes/cuda.html#memory-management)。

batch 參數只改 dense prediction head 的執行批次，encoder／多視角 decoder 仍做相同工作；這輪 infer 總耗時沒有下降。不能由較高的顯存占用推論整個網路會更快，也不能由這個測試反推每一層的確切耗時占比。

batch 2／3 的 mask 差異在某個 view 最多約 **0.795%／0.744% pixels**。raw depth 最大差值包含 mask 切換後的零值與有效深度切換，**不能解讀為公尺定位誤差**。模型結構與輸入相同，批次改變仍會改變浮點計算結果；小幅輸出差異經 mask、voxel first-point 和 PCA 門檻，可能改變離散選點。沒有證據說它提升或降低正式精度。

## PCA chunk

| chunk | 四場景暖機取樣平均 | PCA／取樣階段 peak allocated | peak reserved | 選點索引與 report |
| --- | ---: | ---: | ---: | --- |
| 256 | 65.61 ms | 4.76 GiB | 4.83 GiB | 四場景全部相同 |
| 512 | 57.13 ms | 4.89 GiB | 5.09 GiB | 四場景全部相同 |
| 1024 | 51.84 ms | 5.15 GiB | 5.61 GiB | 四場景全部相同 |

這個 peak 包含 resident model，但未包含該 phase 之前模型 infer 的暫存峰值，不能把此表當完整推論 peak。1024 比 256 的取樣階段少 13.77 ms；如果其餘時間不變，整幀約 721 → 708 ms，約 1.9%。這是推估，尚未把 1024 用於完整實際模型重建。

增大 chunk 減少小批次呼叫／同步，有明確但有限的收益。暫不修改現在完整 run 的 chunk 256：新版 identity 包含 chunk 與來源程式 hash，執行中切換會破壞續跑契約。未來另開候選時可優先考慮 1024，再確認 production allocator、完整重建 peak 與輸出一致性。

## 小段定位差異

baseline、batch2、batch3 的實際輸出各使用原 ROS solver、相同 prior／map cache 做兩條各兩幀定位，六個短段共 12 筆有限 pose，檢查通過。baseline CSV SHA256 重現之前加速版對照：Track1 `df9ab5a481ebc15c75d92feed8d46e4968a0cd7d668fedbe1917c242296bb63a`；Track2 `317e22e15dcf11e302fd7448d71badbbcae4be4dfb06710a13a5831c7bf7e4a3`。

| 改動 | Track | 最大 XY 版本差異 | 最大 yaw 版本差異 |
| --- | --- | ---: | ---: |
| batch 2 | 1 | 0.546 mm | 0.00252° |
| batch 2 | 2 | 1.002 mm | 0.00358° |
| batch 3 | 1 | 0.693 mm | 0.00711° |
| batch 3 | 2 | 0.659 mm | 0.00550° |

這些是相對原執行設定的差異，**不是 ground-truth 誤差、ATE 或精度改善**。yaw 內部仍是 radians，表格只把差值轉 degrees；差值先 wrap。短段差異小也不能證明完整序列全部相同。

## 執行中的完整 run 與保存

只暫停原 wrapper 在兩條 Track 之間的啟動：Track1 child 繼續重建及定位，完成後才載入 probe 模型；測完在 `finally` 對同一 wrapper `SIGCONT`，沒有重新啟動完整序列。`wrapper_resumed.json` 保存恢復紀錄，Track2 已啟動；原加速版的六個來源檔 SHA256 保持不變。舊最佳 CSV 沒有覆寫。

本次證據在 `outputs/runs/rig_memory_probe_2610181311/`：

- `identity.json`／`sources/`：模型、原始 JPG、prior、校正、程式版本與實際執行的 script snapshot。
- `model_results.json`、`chunk_results.json`、`summary.json`：完整重複測量、失敗欄位、數值差異與彙整。
- `allocator_environment.json`、`scheduling.json`、`wrapper_resumed.json`：實際 allocator 與 Track 啟動時序。
- `localization_runs.json`、`localization_comparison.json`、逐組 command／log／report／PLY／labels：原 solver 的短段對照。
- `memory_performance.png`／`.svg`：速度／顯存圖。outputs 不由 Git 同步，需另外保存。

圖表：![顯存與效能對照](../outputs/runs/rig_memory_probe_2610181311/memory_performance.png)

測試工具為 `scripts/probe_rig_memory.py`、`analyze_rig_memory.py` 與兩 Track 間排程的 `schedule_rig_probe.py`。後續若重測，應等目前完整重建完成、選新的 outputs 目錄，並明確帶與完整入口相同的 allocator：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  outputs/ml_runtime/venv/bin/python scripts/probe_rig_memory.py \
  --full-run outputs/runs/three_camera_full_fast_imported_2610181301 \
  --output outputs/runs/rig_memory_probe_manual_YYMMDDHHmm
python3 scripts/analyze_rig_memory.py outputs/runs/rig_memory_probe_manual_YYMMDDHHmm
```

本輪結論：維持當前完整測試；不因 batch／cache 表中的小差異切換模型設定。PCA 1024 有可重現的取樣改善，可作下一個獨立候選，不需要為約 2% 的預估整幀空間再中斷這輪。

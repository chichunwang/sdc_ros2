# 49｜Track3 因果 baseline 與手動短段入口（2026-10-23）

## 目前能確認到哪裡

這份紀錄分開三個狀態，不能互相替代：

1. **靜態因果路徑審查：**既有 `legacy` 路徑使用初始 pose 作為第一幀初值，之後以前一幀接受的 pose 初始化當前幀，將當前 `raw_ply` 配準到靜態 map。拒絕候選時保留前一 pose，仍逐幀輸出。程式結構沒有使用未來幀回修既有 pose；這是程式碼審查，不是執行結果或正式合規認證。
2. **短段運行：尚待使用者執行。**50 幀命令已對本機資料完成乾跑及參數 guard 檢查，沒有執行定位。下方命令交由使用者手動執行後，才有短段輸出可檢視；乾跑不代表配準成功、精度合格或能穩定完成 3142 幀。
3. **正式提交契約：尚待確認。**公告已確認提交欄位與資料筆數，但官方 sample、`idx` 的正式意義、`yaw` 單位／座標系，以及本機結果如何轉成符合提交要求的 CSV，仍不能由現有乾跑或程式預設推定。本文不宣稱已取得官方認證。

規則解讀與所有 Track 的因果限制見 [48｜規則與程式稽核](48_RULES_AND_CODE_AUDIT_20261023.md) 及現行 [00 整合新版規則](<Competition Regulations/00_SDC_2026_Midterm01_Localization_整合新版_20261022.pdf>)、[01 因果定位細則](<Competition Regulations/01_Mid01_Updates_Rules.pdf>)。所有已發布 pose 必須只依當前及過去資訊產生；不得用 RTS、backward／non-causal smoothing 或定位完成後的離線軌跡修正。

## 資料與尺度狀態

本機唯讀盤點記錄在 `outputs/run/track03_test_2610231407/`。該輪找到 3142 個 `raw_ply`，檔名索引為 0–3141；每個檔案的 timestamp 唯一遞增，且逐項等於 bag 前相機訊息 timestamp。bag topic 計數為 `/imu/data` 11307 筆、三個 H265 相機 topic 各 3142 筆，未見 GNSS。map 是 ASCII XYZ＋intensity，57,358,377 點、約 2.2 GB。`init_guess.txt` 為有限 4×4 矩陣，旋轉正交誤差約 `7.97e-10`、det 約 `1.000000000106`，最後一列為 `0 0 0 1`。這些是本機檔案盤點結果，不補足官方 frame 或尺度契約。

公告確認 Track3 有前／左前／右前 H265 topic、`/imu/data` 的 orientation／acceleration、提交欄序 `idx,x,y,yaw`，以及恰 3142 筆（不含表頭；含表頭 3143 筆紀錄）；原文與來源界限見 [Kaggle 資料規格](../../midterm01/docs/KAGGLE_DATASET_SPEC.md)。公告沒有提供可在本機驗證的 official sample。`idx` 0-based 與 yaw 弧度只是既有本機匯出慣例，不能當成官方確認的 ID 規則、單位或座標系。

本輪明示 `--ply-scale 3`，只是沿用既有 baseline 作為機制起點；Track3 原始 PLY 尺度尚未校準，3 不是標定值、最佳值或精度推薦。資料來源報告為 `outputs/run/track03_test_2610231407/track3_dry_run.json`；其中 50 幀乾跑選到 `000000_ts_1727239790258245070.ply` 至 `000049_ts_1727239791997635276.ply`，並記錄 map、初始矩陣及尺度狀態。該 run 的 `test_info.json` 明確記載 `localization_executed: false`。

## Baseline 的因果模型與限制

本輪允許的入口限於 `legacy`、前相機 `front`、逐幀承接 `carry`、幾何模式 `off`、水平 refinement `none`，且不提供 `prior-run`。概念上每幀計算：

```text
T_i = Register(M, P_i, init_guess if i = 0 else T_(i-1))
```

其中 `M` 是靜態 map，`P_i` 是當前前相機點雲，`T_i` 是當前 pose。若當幀配準被拒絕，輸出沿用上一個 pose，因此仍可保留輸出幀，但長時間失去可靠配準可能造成軌跡停滯或失追。接受／拒絕與 map fitness 只能描述本輪演算法行為，不能當成真值誤差或 Kaggle 精度。

這條既有入口不讀 bag（`bag_path` 為空），不做 GNSS／IMU sensor preprocessing，不使用 RTS、graph 或整段軌跡修正。GPU 使用既有 `cuda_ndt` D2D／DIRECT7 配準；ASCII map／PLY 讀取、PCL voxel／crop、fitness 評估及小型 solver 仍在 CPU，屬 CPU／GPU 混合路徑，不是全 GPU。舊 D2D 與 CPU 路徑曾有顯著差異及 OOM 風險；本輪短段尚未定位，不能據此宣稱精度、穩定性或整段可完成。CUDA 失敗時不得靜默改走 CPU。此入口不改 C++，不需因本輪文件而重編。

## 使用者手動短段命令

依固定順序在 Ubuntu 主機執行。容器目前停止時，先由使用者啟動 GPU 容器，再執行單一 50 幀定位命令：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
bash scripts/dev.sh up
python3 scripts/benchmark.py --track 3 --frames 50 --profile legacy --backend cuda_ndt --cuda-mode D2D --ply-scale 3
```

預期建立新的 `outputs/run/track03_test_YYMMDDHHmm` 目錄（同分鐘碰撞時加序號）。乾跑記錄為 `outputs/run/track03_test_2610231407/track3_dry_run.json`，不會被當成定位 run 重用。短段的 `diagnostic.csv` 是 50 幀診斷資料，不是 3142 筆正式 submission。依既有 runner 格式查看首幀／每 25 幀摘要與逐幀 `timings.csv`；本輪沒有改寫歷史 console／timings 格式，也沒有完成新 runner 的逐幀固定格式。

完成後先檢查該 run 的 `validation.json`：要求 `returncode` 為 0、`output_complete_and_finite` 為 true、`validation_errors` 為空。另檢視失敗配準數與相鄰 pose 跳動，作為需調查的診斷；不能把失敗數少或矩陣有限解讀為精度通過。50 幀通過也只代表這個短段的輸出完整性，不能推出全序列完成或正式提交合約成立。

若日後要由使用者啟動完整序列，將同一命令的 `--frames 50` 換成 `--full`，預期處理 3142 幀；本文不要求現在接著跑 full。完整定位與外部提交一律由使用者手動執行。

可在實際定位後，以同參數比較 25 幀與 50 幀 run 的前 25 幀矩陣，檢查增加後續資料有沒有改變已輸出的前綴。這項前綴測試尚未執行，不能記為通過；即使通過，也只提供因果性的一項實證檢查，不能替代規則與提交格式審查。

## 本輪證據與未解問題

`outputs/run/track03_test_2610231407/` 保存本輪資料／CLI／乾跑／靜態檢查紀錄：`track3_data_inventory.json` 記錄 bag topic／整數時間與初始矩陣盤點、`track3_dry_run.json` 記錄 50 幀 dry-run、`baseline_checks.log` 記錄 help 與無效／違規參數 guard 的結果，`test_info.json` 記錄未執行定位。沒有 `diagnostic.csv` 或 validation 定位結果，因本輪沒有跑定位。真正定位時另保存選取 PLY、原始 map／init、程式及 binary 的 SHA256；原始 map 雜湊屬執行前來源記錄時間，不是逐幀配準時間。這些檢查支持命令與限制的本機準備狀態，不支持精度或正式合規結論。

可唯讀重做資料盤點：`python3 scripts/check_competition_rules.py --track 3 --inventory-only`。此選項 exit 0 只代表資料盤點通過；不加它時仍 exit 2，因完整提交契約及實際因果性尚未驗證。命令加 `--dry-run` 可重做上述 50 幀計畫檢查，不啟動定位、不建立新 run。

正式提交前仍需依官方 sample／完整契約確認：`idx` 與資料列對應方式、yaw 單位與座標系，以及本機 pose 到提交欄位的轉換。欄序 `idx,x,y,yaw`、3142 資料列及 E3 檔名 `result_3.csv` 已公告，不列為未知。尺度、map／相機座標關係和初始 pose 的物理意義也尚未由公告確認。公開 Kaggle 頁面本輪無法讀取；沒有即時網站認證或 TA 重跑證據。

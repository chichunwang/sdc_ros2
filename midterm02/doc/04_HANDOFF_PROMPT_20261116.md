你要接手 SDC Midterm02 Visual SLAM 比賽（Kaggle Track01）的演算法改進工作。請用繁體中文溝通與寫文件。

## 開工先讀

1. `code_base/sdc_ros2/CLAUDE.md`、`code_base/sdc_ros2/AGENTS.md`、`code_base/sdc_ros2/midterm02/AGENTS.md`（硬性規則，midterm02 的 AGENTS 優先）
2. `midterm02/doc/03_GSLAM_DESIGN_AND_STATUS_20261116.md`（完整設計、每個版本的做法、離線數據與 Kaggle 分數）
3. `midterm02/doc/SDC Midterm II Visual SLAM.pdf`（官方規則）

## 必須遵守

- 只能用 Kaggle 提供的資料（影像、raw_ply、camera_info.json、Track01 的 IMU）。不可用 GPS、外部地圖、ground truth、往年資料，不可手改 pose，不可依榜分反推真值。
- 不可只調參數：每項改動要有「觀察 → 假說 → 模型 → 實作 → 驗證」，權重要有量測依據。
- 最終選的兩份提交必須能用上傳的程式重現：每個版本都要有對應的實驗名稱（`vslam/gslam/params.py` 的 `EXPERIMENTS`）和固定 seed。
- 程式只在 Docker 容器 `midterm02_vslam` 裡跑：主機指令要包 `sg docker -c '...'`。
  - 測試：`sg docker -c 'docker exec -i midterm02_vslam python -m unittest discover -s tests'`
  - full（使用者已授權本工作的 full；一定開 RViz）：`sg docker -c './scripts/dev.sh slam --track 01 --experiment <名稱>'`，約 10 分鐘，放背景執行。
- full 執行中不改程式檔（`vslam/`）；GPU 可以同時跑多個工作（使用者 2026-11-16 決定）。
- Kaggle 提交、選定最終提交、E3 上傳、寄信都由使用者做。使用者回報分數後，記到該 run 的 `test_info.json`（`public_score`、`reported_by: user`、`reported_at`、`score_source`），並更新 doc/03 的分數表。
- 未經使用者同意不 commit（使用者自己 commit）。`data/` 唯讀、不進 Git。
- 子代理角色在 `code_base/sdc_ros2/.claude/agents/`（quick-scan、docs-helper、small-fix、run-analysis）；演算法與數值留在主代理。
- 長指令不要內嵌大段 Python：先寫成檔案再執行（auto 模式的檢查遇到長指令容易失敗）。

## 目前狀態（2026-11-16）

最新 commit：`bff9baf feat(midterm02) v1.4.0`。之後的修改（gslam 0.5.0–0.6.0：特徵尺度加權、投影引導補觀測、LoGeR 深度因子、對應測試與文件）**還沒 commit**，37 項測試通過。

Kaggle Public 分數（越低越好；Public 只有 132 幀，最終看 Private 的 1182 幀）：

| run（`outputs/run/track01_full_…`） | 實驗 | 做法 | Public |
| --- | --- | --- | --- |
| 2611160111 | — | 第一版 | 0.369 |
| 2611160533 | `E12_no_bundle` | 密集閉環＋尺度圖，無 BA | 0.144 |
| **2611161129** | 預設（`E13_bundle_basic`） | 基本 BA | **0.085（目前最佳）** |
| 2611161500 | `E14b_bundle_revisit_only` | ＋依位姿跨圈配對 | 0.086 |
| 2611161614 | `E18_depth` | ＋LoGeR 深度因子 | 0.087 |
| 2611161822 | `E20_learned` | ＋SuperPoint／LightGlue 軌跡 | 0.099 |
| 2611161541 | `E16_size_weight` | ＋依 SIFT 尺度加權 | 0.092 |
| 2611161550 | `E17_guided` | ＋投影引導補觀測 | 0.092 |
| 2611161431 | `E14_bundle_more_pairs` | ＋跨幀＋跨圈 | 0.097 |
| 2611161449 | `E14a_bundle_skip_only` | ＋跨幀配對 | 0.122 |
| 2611161514 | `E15_more_features` | 特徵上限加倍（無效，結果與 0.085 相同，未提交） | — |

已知結論：

- 剩下的誤差在「沿行進方向的局部尺度」，不在轉角（IMU 陀螺儀與 BA 轉角吻合到 1.3%，見 doc/03）。
- 跨幀配對（Essential RANSAC 驗證）會引入沿對極線的錯配，讓分數變差。
- Public 對 1–5 cm 的軌跡差異很敏感，0.01 以內的分數差不能當作方法好壞的可靠證據。

## 正在進行（你要先處理）

`E18_depth`（LoGeR 深度約束的 BA，gslam 0.6.0）的 full **已跑完**：`outputs/run/track01_full_2611161614/result_1.csv`，格式檢查通過。
與基本 BA 的差：132 幀視窗中位數 0.122 m、最大 0.253 m，和離線結果 `outputs/cache/dba_p1.csv`（0.123／0.255）一致。
**Public 0.087（使用者回報，已記到 `test_info.json` 與 doc/03）**，與基本 BA 持平；整線與基本 BA 差 0.338 m，Public 分不出、Private 可能差很多。**IMU 加速度檢查（doc/03「IMU 加速度計檢查局部尺度」）顯示 E18 的局部長度與 IMU 不一致（殘差 0.186 對基本 BA 0.133 m/s），不採用，預設維持基本 BA。**

E18 的做法：每幀一個未知的 LoGeR 尺度 s_i（相鄰幀平滑，σ=0.02），相機到 3D 點的距離 ≈ exp(s_i) × LoGeR 距離（相對誤差 σ=0.10，Huber），交替解 s 和 BA（`bundle.frame_log_scale`、`bundle.depth_factors`）。它讓軌跡移動得比其他變體大很多，分數可能明顯變好或變差。

## IMU（2026-11-16 後續）

- IMU 加速度一致性檢查（`outputs/cache/dev19_imu_accel.py`、`vslam/gslam/imu.py` 的 `consistency`）：基本 BA 0.133、E18 0.186 m/s，E18 不採用。
- IMU 平均速度變化因子加進 BA（`E19_imu`，gslam 0.7.0，預設關閉）：離線只讓軌跡移動 4 mm，保留塊驗證幾乎沒改善，**不跑 full、不提交**。詳見 doc/03 最後兩節。
- 41 項測試通過（新增 `tests/test_imu.py`）。程式改動（imu.py、bundle.py、pipeline.py、params.py、run_slam.py 版本）尚未 commit。

## 學習式特徵（2026-11-16 後續）

- 使用者決定：官方允許的預訓練模型可直接用於正式結果（已改 AGENTS.md）。映像改為含 PyTorch 的版本（`scripts/dev.sh build-learned`；舊映像另存 `midterm02_vslam:py39-gpu-base-20261116`）。
- `E20_learned`（gslam 0.8.0）：SuperPoint＋LightGlue 軌跡與 SIFT 一起進 BA。full `outputs/run/track01_full_2611161822/result_1.csv`，與基本 BA 差 0.028 m，IMU 一致性 0.133 → 0.121。**Public 0.099（比 0.085 差），不採用**。
- 下一步候選：學習式特徵的次像素／特徵度量精修（SuperPoint 沒有次像素，重投影 0.7 px 對 SIFT 0.4 px）、ALIKED 版本（`E20a_learned_aliked`）。

## DROID 對照與相機自校正（2026-11-16 後續）

- DROID-SLAM（獨立容器 `docker/droid/`）的軌跡和基本 BA 只差約 4 cm，代表誤差主要是兩者共有的；兩者共用的是 `camera_info.json` 的針孔內參。
- BA 自校正發現殘餘徑向畸變（k1≈−0.056、k2≈+0.087，中段約 4.5 px）與 cy 偏約 9 px；前後半段各自估計一致，陀螺儀與 IMU 也支持。
- `E21_selfcalib`（gslam 0.9.0）full：`outputs/run/track01_full_2611161931/result_1.csv`，與基本 BA 差 5.5 cm。**Kaggle 分數待回報**。詳見 doc/03。

## 待提交候選（2026-11-16 晚；使用者當日提交已滿，12 小時後一次提交 10 個）

完整清單、每個要驗證的事與 IMU 一致性見 doc/03「待提交的候選」：E21、D2、D1、E22、F（E22＋D2）、F（E21＋D2）、E25、E24、F（三方）、M1。
**E25 狀態**：已完成（`outputs/run/track01_full_2611162103`，submission check OK）。IMU 一致性 0.122，比 E22 的 0.114 差；離 DROID 也比 E22 遠。跨圈配對沒有幫助（E24 對 E21 也是如此），只作為成對比較的候選。
**跨圈點雲疊合（E26／E26b，gslam 0.10.0）**：使用者在 RViz 看到牆面重影。檢查結果是 LoGeR 深度不一致，不是位姿錯：點雲修正和影像對極幾何衝突（32 對中 0 對改善）。E26（2611162300）和 E21 幾乎相同；E26b（2611162311，強制疊合）IMU 0.310、明顯變差，留作驗證上傳。詳見 doc/03。RViz 收尾時的地圖已改用最終位姿。
接手時先看 log 是否有 `submission check: OK`，再用 `outputs/cache/cmp.py`、`dev20_check.py` 和 E22／E24 比較，補進 doc/03 候選表第 7 項。
注意：第 1 輪初始重投影誤差中位數 2.71 px、95% 4.51 px，比 E22 高（學習式特徵也配對了 1554 對跨圈影像，跨圈軌跡 16.6 萬條），要確認跨圈錯配沒有把軌跡拉壞。
使用者回報分數後：記到各 run 的 `test_info.json`，更新 doc/03 分數表，依成對比較（E21 vs 0.085、D2 vs D1、E22 vs E21、F vs 來源）決定方向。
其他新增：DROID（`docker/droid`、`scripts/run_droid.py`）、MASt3R-SLAM（`docker/mast3r`、`scripts/run_mast3r.py`）、融合（`scripts/fuse_trajectories.py`）、深度模型評估（`docker/depth`，結論：不取代 LoGeR）。
規則變更：學習式特徵等官方允許模型可直接用、GPU 可同時跑多個工作（midterm02/AGENTS.md）。

## 依 E18 分數決定下一步（已完成：結果「差不多」，經 IMU 檢查後不採用）

- **明顯變好（< 0.08）**：設為預設；接著試把視差 < 1° 被丟掉的軌跡（約 73%）用 LoGeR 深度當初值加回來（離線腳本 `outputs/cache/dev17_depthba.py` 的第二個參數就是視差門檻）。
- **明顯變差**：深度因子可能被過度信任（同一幀的 LoGeR 誤差彼此相關）。改用有依據的較弱權重，例如依每幀有效獨立量測數縮放；離線 σ_d=0.3 版本是 `outputs/cache/dba_w03.csv`（與基本 BA 差 0.081 m）。
- **差不多**：Public 分不出好壞，從整條路線一致性的角度和使用者討論要不要保留。

離線開發的方法：`outputs/run/<run>/debug/bundle_input.npz` 存有 BA 的完整輸入，可以不重跑前端直接試 BA（例如 `outputs/cache/dev13.py`、`dev17_depthba.py`）。無 GT 的診斷工具：`scripts/lap_consistency.py`、`scripts/segment_length_consistency.py`、`outputs/cache/cmp.py`。

## 還沒做的交付物（期限前要完成）

- Track02（尚未公布，沒有 IMU；流程本來就不用 IMU）與 `result_2.csv`
- 報告 `vslam_report_<ID1>_<ID2>.pdf`：流程、貢獻、問題與解法（含實驗）、套件與版本、分數演進（0.955 baseline → 0.369 → 0.147 → 0.144 → 0.085 → …）、工作分配
- 簡報：11/26 上課報告，投影片 11/26 09:00 前寄出
- 重現用 README、E3 壓縮檔（12/03 23:59 前）

## 第一個動作

讀完上面的文件後，向使用者確認 E18（`track01_full_2611161614`）的 Kaggle 分數，記錄後依「依 E18 分數決定下一步」繼續。

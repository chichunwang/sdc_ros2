# 52｜根 AGENTS.md 整理前歷史快照

保存日期：2026-10-24（Asia/Taipei）。本檔完整保存本次整理前的根 `AGENTS.md`，包含原有未提交內容；下方原文未改字、順序或相對路徑。

此快照僅供追查歷史，**不是現行協作規則、執行授權或算法推薦入口**。現行規則見 [根 AGENTS.md](../../AGENTS.md)，算法／限制見 [51](51_TRACK3_ARCHITECTURE_AND_DEVELOPMENT_20261023.md)，命令見 [50](50_COMMAND_REFERENCE.md)。原文中的「最新／目前」保留當時語境，原始相對路徑以 `/home/ricky/sdc_ros2/` 為基準。

- 原始檔：`/home/ricky/sdc_ros2/AGENTS.md`
- 原始長度：58752 bytes
- 原始 SHA256：`4b13c6d2aa32d664f60b1ee20f2114ab8d3661bb5fd27db29f9b231b1164bf3c`
- 下方 `markdown` 區塊中的 UTF-8 原文可依上述 SHA 核對。

## 整理前全文

````markdown
# sdc_ros2 專案說明與協作約定

## 2026-10-23 Track3 結構均衡／重捕獲 v2，使用者授權200幀（最新）

- 使用者明確授權修正及200幀測試、要求繼續；現行`run_track3_map_only.py`的method為`track3_forward_gpu_structural_ndt_icp_reacquisition_v2`。固定前鏡頭、LiDAR map XYZ、不接IMU/GNSS、不讀外部預測、不做未來回修。同CLI已不重現1912的v1，舊source／結果保留。
- 每幀map加入原有NDT粗層＋ICP、三尺度seed(.95,1,1.05)；training-only source PCA法向按map XYZ主方向／無平面類作等總票數，再保留raw空間塊權重，權重在當幀各seed/line search固定。拒絕後進既有8m/30°初始化域重捕獲；一般仍2m/8°。Map physical motion改用上次採用map anchor及同段elapsed，分開累積prior修正；pair原rate gate及所有幾何／視線／共識門檻保留，不以接受率放寬。
- 200幀`track03_test_2610232248`完整：map182/200、pair198/199、lost0、最長無map.107871264s，19次重捕獲14次重新取得支持；總338.131s，後續registration中位1.473s。40項機制/I/O通過；CPU/CUDA權重最大差6.66e-15、pose1.01e-16、scale差0；5/200 TXT/CSV逐byte前綴及131檔source SHA一致，聚合`track03_test_2610232000/`。只驗200幀機制及追蹤，不宣稱full精度或新分數；full仍使用者手動，不新增資格門檻。
- 新預設`--vram-budget-gib 4`只限制Torch allocator，非全部driver/NVRTC/cuSOLVER VRAM；PCA256與CUDA失敗不切CPU保留。200幀Torch peak allocated/reserved約.86/1.41GiB；失敗/中斷與stage/state診斷已補，固定輸出格式不變。
- 使用者提供`outputs/Others_track3_T3g.csv`並回報1.02，3142筆、SHA`bf95936d9db848e458ccfac53df2625da6d70e412d311fa523a1c2ca6ad5f3ee`，未獨立核對上傳bytes。外部預測非真值，不作校準或solver輸入。相同前200幀XY差異RMS：舊77.96版37.145m、1912版3.271m、v2 1.272m，不能當分數／ATE；比較及圖在`track03_test_2610232251/`。算法／限制仍唯一[51](midterm01_gpu/doc/51_TRACK3_ARCHITECTURE_AND_DEVELOPMENT_20261023.md)，命令[50](midterm01_gpu/doc/50_COMMAND_REFERENCE.md)。

## 2026-10-23 Track3 前鏡頭失效根因重查（最新）

- 使用者明確先只用前鏡頭，map可靠後才IMU。手動full `track03_full_2610231912`僅1040/3142幀、returncode1、map98/942、末map idx125、首lost idx181、lost859、最長無map32.893173258s；validation_errors空，終止原因未記錄，不推定OOM／使用者中斷。原結果保留、沒有正式新分數，不推薦原樣重跑。
- 根因研究 `outputs/run/track03_test_2610231928/`：22次獨立當前幀GPU fit、10對RGB檢查，無新軌跡／full／回寫；6個原樣控制pose/scale逐值相同，129個生產source與full快照SHA一致，原run直接檔案SHA不變。idx126兩map解原點差0.651m，source全點差median1.249m；近水平共同平面點沿面差median1.069m、法向僅0.0113m，證明低最近鄰殘差可伴隨沿面歧義，不能以rank7/accepted稱精度。
- 一般map tracking使用ICP(`use_ndt=False`)，僅bootstrap／pair有NDT。加入NDT粗層有單幀幫助，但失鎖後2m/8°域、1.95m interior及單幀motion gate會限制累積prior回正；lost只有標記、沒有重捕獲狀態。原初始化8m/30°域獨立診斷idx130/150找到一致候選，idx140仍歧義，不能當已修復／真值。RGB8對有足夠留出，未支持突然大尺度跳變，也不能認證剛體／絕對尺度。
- 詳解集中唯一[51](midterm01_gpu/doc/51_TRACK3_ARCHITECTURE_AND_DEVELOPMENT_20261023.md)與研究run README；本輪未改生產solver或gate。LiDAR map仍在`midterm01/data/Track3/map.pcd`，57,358,377點，bag無PointCloud topic；raw_ply為相機重建來源。full與提交仍由使用者手動，不新增full資格門檻。

## 2026-10-23 Track3 前向無 IMU 新入口（最新接續）

- 新 `midterm01_gpu/scripts/run_track3_map_only.py`，method=`track3_forward_gpu_ndt_icp_adjacent_cloud_visibility_v1`：GPU相鄰PLY六自由度配準提供初值，當前map bounded Sim3 ICP＋training/held-out視線深度＋局部尺度重啟／運動檢查；首幀從當前training求scale（實測約6.3986，非手動標定常數）。不讀IMU/GNSS或舊pose、無RTS/graph/離線回修。首次地面seed也只用training；map拒絕時保留相鄰點雲的當前運動預測。GPU kernel module跨scene共用、Torch deterministic、PCA每批256；固定逐幀格式及成功full的run內analysis保持。
- 本輪依「先測能測的」要求執行有限3／5／32／100幀研究，沒有full／外部提交。100幀 `track03_test_2610231848` 完整，map74/100、98/99個pair採用、lost0、最長無map0.145396566s，idx66切tile，總179.218s；5／32／100 TXT與CSV前綴逐byte相同（`track03_test_2610231802/real_prefix_5_32_100.json`）。最終32項核心／visibility／I/O檢查通過（`final_verified_32_tests.log`），最後共識防護對100幀裁決唯讀核對不變，最新5幀來源重驗存1856。這些不是精度或全程穩定證明，沒有新分數，不把accepted當真值。
- 使用者可依[50](midterm01_gpu/doc/50_COMMAND_REFERENCE.md)手動full3142幀；乾跑已核對3142，完整計算／提交仍使用者手動，不增full資格門檻、不自動啟動。正式精度仍未知，IMU繼續暫緩。算法及先前獨立幀歧義／失敗實驗仍集中唯一[51](midterm01_gpu/doc/51_TRACK3_ARCHITECTURE_AND_DEVELOPMENT_20261023.md)。下節「只有獨立研究、尚無full runner」是較早階段，現以上述為準；舊77.96版本保留、不推薦原樣重跑。

## 2026-10-23 Track3 重新開發：先 map-only NDT／ICP，IMU 暫緩（最新）

- 使用者回報舊核心 full `track03_full_2610231748` 為 **77.96**；3142 幀全 accepted，但水平僅約6×4m範圍，不代表可用精度。原 run／score null／CSV 等保留，使用者回報及來源 SHA 另存 `outputs/run/track03_test_2610231802/old_full_readonly_audit.json`，未核對 Kaggle 上傳bytes。不推薦重跑同版。
- 依使用者要求先重建配準、GPU 優先，暫不整合 IMU。新 `scripts/track3_registration.py`／`test_track3_map_registration.py` 是**獨立當前幀研究**：GPU P2D NDT surrogate＋robust point-to-point ICP／Sim3，固定source支持及空間權重、held-out檢查、資料幾何rank、多初值歧義拒絕；不是舊D2D、PCL原likelihood或已完成full runner。不讀bag／舊pose、不產生軌跡／提交，不用未來幀回修。GPU做voxel／covariance／kNN／PCA／殘差／Jacobian／normal equations；CPU留I/O、小solver與bounds，保留每批256的eigh界限及CUDA失敗直接失敗。
- 21項CPU/CUDA機制檢查通過（`track03_test_2610231802/final_mechanism_tests.log`）；最終研究 `track03_test_2610231818` 有5組map-derived已知變換×2算法共10次、8個真實獨立幀×6初值共48候選，全部8個multi-start最終拒絕。首幀尺度2.71355及6.39973都局部通過，最終歧義拒絕，不能稱map精度可用；IMU繼續暫緩。單一真實CPU/CUDA配對pose最大差1.78e-15、scale差3.55e-15，checks相同；fit 4.883／1.330s僅該例，非full速度。已知變換的公釐誤差不是相機定位精度。詳細推導／限制仍集中唯一[51](midterm01_gpu/doc/51_TRACK3_ARCHITECTURE_AND_DEVELOPMENT_20261023.md)，研究重現命令見[50](midterm01_gpu/doc/50_COMMAND_REFERENCE.md)。full／外部提交仍使用者手動，沒有新分數。

## 2026-10-23 Track3 舊核心去 GNSS 對照與文件整併（最新）

- 使用者要求原已評分版本拿掉 GNSS 手動跑一次，已新增 `midterm01_gpu/scripts/run_track3_old_icp.py`。直接編譯／呼叫 `workspace/src/track3_old_icp/` 的四個凍結原碼，與 `track01_full_2610191750/localization_sources` 逐byte相同；最近鄰原碼及 CUDA library SHA 亦與歷史一致。保留舊固定scale3、四自由度XYZ/yaw、directional、PCL map/input leaf1/.3m、1.5m/8°修正域、alpha.25。移除GNSS/RTS/graph，不接新Sim3/Kalman；prior為上次輸出位置/RP＋截至當前gyro，無速度外推。舊regularizer保留但意義為motion prior，不能稱GNSS量測。首次ground只提供首PLY地面對map+Z的初始化近似，可明確選official控制，不估scale/位置、不宣稱已標定。
- 獨立bridge與11項CPU/CUDA合成串接檢查通過，3/5幀CSV/矩陣前綴逐byte一致，無重疊拒絕保留prior；研究證據 `midterm01_gpu/outputs/run/track00_test_2610231633/`。沒有執行真實舊核心對照定位，沒有新分數或精度保證。每run隔離編譯，不覆蓋舊binary，完整/短段仍全由使用者手動；其他定位程序在跑時不並行啟動。
- v2 full `track03_full_2610231608_0002` 已完成3142筆，只採用52筆map、2357幀tracking_lost，最長無支持63.0847s，尚無score。全程103幀所選候選未收斂但另有收斂候選，選取邏輯及大量撞界/不收斂已記為問題；未放寬門檻或改寫其輸出。本輪只新增舊核心比較，不宣稱修好v2。使用者另跑legacy1631也完成，沒有代替使用者啟動/停止full。
- 原doc51～53已依使用者要求整併為唯一 [51｜Track3架構與演進](midterm01_gpu/doc/51_TRACK3_ARCHITECTURE_AND_DEVELOPMENT_20261023.md)，舊原件與SHA存於 `outputs/run/track03_test_2610231612/original_documents/`，歷史run不變。手動命令見[50](midterm01_gpu/doc/50_COMMAND_REFERENCE.md)。下段「未執行真實v2」為當時歷史狀態，現以本段為準。

## 2026-10-23 Track3 初始化 v2 接續（優先於下段 v1 狀態）

- 使用者要求完成修正並以次輪 full score <1.29 為目標；這是目標，不宣稱保證。現入口 `midterm01_gpu/scripts/run_track3_causal.py` 已標記 `track3_causal_vertical_imu_kalman_sim3_v2`，手動 full 命令及根據見 [51](midterm01_gpu/doc/51_TRACK3_ARCHITECTURE_AND_DEVELOPMENT_20261023.md)。相同 CLI 現在執行 v2，不能再稱可重現 v1；舊 source 快照與 82.17 原始結果保留。
- 靜態幾何診斷（未求新 pose）發現首三幀 source/map 地面差約18.54–18.89°，超舊15°；第一幀候選尺度3/9的有效法向分別2029/8096，9不是標定值。v2 用當前 training 地面幾何給初始化方向／尺度 seed，初始化旋轉域為既有8°工程不確定度的3倍24°；每候選按自身公尺尺度重算法向，CUDA cache 同時核對raw/normal/ground/quality identities。首次 anchor 分開初始化修正與相鄰運動，已有anchor後的運動gate不放寬；加入固定 source 對 training map 覆蓋代價與收斂診斷，撞bounds不接受。全段0 map更新會明確失敗、不印成功Finished。
- 59項不同CPU/CUDA/合成runner檢查通過，合成3/5幀前綴逐byte相同；證據 `midterm01_gpu/outputs/run/track03_test_2610231550/` 與 `track00_test_2610231601/`。只有真實輸入靜態PCA、合成定位及3142幀乾跑，未執行真實v2短段/full，沒有新分數或真實精度證明。完整與短段定位仍交使用者手動，不加full資格門檻，禁止未來回修／盲調榜分／修改既有CSV。

## 2026-10-23 Track3 前向 Kalman／尺度分支接續

- 使用者已授權依序實作初始化／座標核對與 IMU Kalman、因果運動預測／配準拒絕、受約束逐幀尺度。新入口 `midterm01_gpu/scripts/run_track3_causal.py`；模型、工程參數及手動 100 幀命令見 [51](midterm01_gpu/doc/51_TRACK3_ARCHITECTURE_AND_DEVELOPMENT_20261023.md)，指令入口見 [50](midterm01_gpu/doc/50_COMMAND_REFERENCE.md)。既有 legacy／RTS／graph 與歷史結果保留。
- 39 項合成 CPU／CUDA／I/O 測試、合成 3／5 幀序列化前綴及真實輸入 CPU audit 通過，但不是精度證明。**使用者後續完成 `track03_full_2610231521` 並回報 82.17**：3142 筆輸出完整，但最終 map 採用 0 筆、速度始終為零、XYZ 固定初值、輸出尺度始終約 3，屬啟動／平移追蹤失敗；不推薦原樣重跑。證據 `outputs/run/track03_test_2610231541/diagnosis.json`，原 pose／CSV／validation／manifest 不改；分數記在來源 `test_info.json`，上傳 bytes 未核對。舊 legacy 106.72 保留歷史。
- 初始矩陣只確認程式的 `T_map_source` 慣例與 proper rotation；相機→IMU 外參、raw PLY 上游軸向與官方 pose 參考點未獲完整標定。採明列假設的重力垂直 gyro 平面近似，不用 IMU 絕對 yaw，不雙積分加速度；零 covariance 是未知。`initial-scale 3` 是 PLY XYZ 的未校準啟動倍率，不是已確認尺度，也不要求使用者逐幀手動配準。
- **使用者最新澄清：只有 full 提交才有正式成績。**短段只供機制／故障檢查，不代替精度評分，不另設短段或本機精度證明作為提供 full 命令的資格。完成有依據的算法改動與必要檢查後，full 仍由使用者手動啟動；不得自動定位／提交，也不重跑完全相同的已知失敗版本。

## 2026-10-23 full run 事後圖表規則（最新）

- Python／Matplotlib 圖表自本更新起，在每次成功完成的 full track 後產生，放在該次 run 的 `analysis/` 子目錄；不再把 `midterm01_gpu/doc/analysis/` 作為新 run 圖表的集中輸出位置。短段、失敗或未完成 run 預設不產生完整比較圖。歷史 run 不搬移、不補圖，既有 `doc/analysis/` 教學素材與個人紀錄保留。
- 圖表屬定位完成後的唯讀診斷，不修改 pose、CSV、參數或定位結果，也不允許離線修正軌跡；不把接受率、殘差或圖表解讀當成精度／真值證明。缺少感測器或診斷欄位時不得捏造。
- 標準輸出為 `01_trajectory.png/pdf`、`02_pose_changes.png/pdf`、`03_registration_timing.png/pdf`、`manifest.json` 與 `README.md`。manifest 記錄來源檔案與 SHA；畫圖耗時另計，不混入定位 `timings.csv`。圖表失敗須以獨立 `analysis_status.json`／log 記錄，保留定位成功狀態與原始輸出，不宣稱圖表已完成。
- 成功 full 的自動產圖與新 run 獨立重試入口 `python scripts/plot_run_analysis.py --run <new_full_run>` 依實際實作／狀態記錄；此入口只讀當次新 run。完整定位及外部提交仍由使用者手動啟動。

## 2026-10-23 因果定位規則（優先於 2026-10-21 規則解讀）

- **同日 Track3 本機準備接續：**已唯讀確認 3142 個 PLY 檔名 idx 0–3141 及時間戳與 bag 前相機逐項相符，三相機各 3142 筆、IMU 11307 筆、無 GNSS；map／init 矩陣已盤點。GPU 既有 `benchmark.py` 已限 `legacy/front` 開放 Track3，明示尺度並拒絕 GNSS／RTS／graph 路徑，50 幀乾跑通過但未實際定位。命令與證據見 [49](midterm01_gpu/doc/49_TRACK3_CAUSAL_BASELINE_20261023.md)。scale 3 是未校準基線；官方 sample／frame／yaw 與實際前綴一致性未驗證。使用者要求所有定位仍交由他手動執行；不將靜態因果審查稱為完整提交認證。

- **Track3 資料說明接續更新（使用者 2026-10-23 轉貼）**：官方文字已列前／左前／右前 H265 相機 topic 與 `/imu/data`（orientation、acceleration），提交欄序為 `idx,x,y,yaw`、恰為 3142 筆資料（不含表頭；含表頭 3143 筆紀錄）。完整原文與來源界限見 [Kaggle 資料規格](midterm01/docs/KAGGLE_DATASET_SPEC.md)。不要再把上述欄位、筆數或 IMU topic 是否公告列為未知；官方 sample、idx 起算／ID、yaw 單位／座標系、本機 bag schema／初始 pose／frame／尺度及因果 runner 仍未驗證，不由公告文字宣稱資料已下載或定位已支援。

- 現行官方入口為 [整合新版規則（31 頁）](<midterm01_gpu/doc/Competition Regulations/00_SDC_2026_Midterm01_Localization_整合新版_20261022.pdf>) 與 [因果定位細則（4 頁）](<midterm01_gpu/doc/Competition Regulations/01_Mid01_Updates_Rules.pdf>)；舊 24／10 頁原件保留為歷史，見 [90](<midterm01_gpu/doc/Competition Regulations/90_舊版_已取代_可刪除_SDC_2026_Midterm01_Localization_24頁.pdf>)／[91](<midterm01_gpu/doc/Competition Regulations/91_舊版_已取代_可刪除_Mid01_Updates_Adjustments_20261021.pdf>)。解讀及本機程式盤點見 [48](midterm01_gpu/doc/48_RULES_AND_CODE_AUDIT_20261023.md)。
- 所有 Track 均須因果定位：不得用未來資訊回修較早 pose，不得使用 RTS、backward／non-causal smoothing，也不得在定位完成後離線修正軌跡。Track1／2 的 GNSS 輸入仍合法，但不使 RTS 合規。現有 `*_rts`、`pose_yaw` 位置 RTS、map smoother 的 motion_rts／batch graph 及 `refine_horizontal_joint` 離線回寫均不得用於現行正式提交；歷史結果與分數保留，不宣稱合規最佳。
- 10/29 09:00 前選取合規 Kaggle 提交；TA 重跑可有合理浮點差，違規或無法重現該 Track 得 0 分。Track3 公告為 3142 幀、無 GNSS pose；public 前 5–40%、private 其餘；private baseline <=1.40 才有基本分。三 Track 權重 0.3／0.3／0.4。E3 截止 11/05 23:59，檔名 `result_1/2/3.csv`。
- 本更新只修正規則與文件方向，不授權啟動 full、改 solver 或產生提交；完整定位及外部提交仍由使用者明確手動執行。

## 2026-10-21 學習優先約定（歷史；圖表輸出位置以 2026-10-23 更新為準）

- Midterm01 後續工作的首要目的，是讓使用者學會推導、解釋及判斷算法。先說明要回答的問題、模型與變數、觀測／殘差／約束、可檢驗預測及消融，再依已授權範圍實作；不得把實作完成直接當作精度改善。
- 延續「非必要不調參」原則；不為榜分盲目掃描參數或反覆試分。必要調整需指出物理、統計、實測標定或數值穩定性依據，並說明可重現的驗證方法。歷史 0.6 等分數目標及先前鼓勵直接提供 full 命令的偏好保留為當時紀錄，不再主導後續方法選擇；0.77、0.30488 等分數仍依來源與顯示精度忠實記錄，不反推被隱藏小數或真值。
- Python／Matplotlib 是教學圖解的主要工具；教學程式、入口、小型固定教學圖及其資料統一放在 `midterm01_gpu/doc/analysis/`。圖解以實際 Python 產圖及資料核對為準；舊 `.m` 與既有 MATLAB 證據保留歷史，不刪除、不作主要執行路徑，只作靜態檢視，不宣稱已執行。這是使用者指定的新子目錄例外，不搬動其他文件。合成展示條件須標明，不能改成生產參數或實測結果。
- 預定教學圖入口命令為 `cd /home/ricky/sdc_ros2/midterm01_gpu` 後執行 `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 outputs/ml_runtime/venv/bin/python doc/analysis/plot_learning.py`；這是讀檔教學圖，不是定位，不需 Docker。未核實執行前不稱圖表或程式已驗證。定位／研究測試 run 仍只放 `midterm01_gpu/outputs/run/` 並遵守既有命名；不得為畫圖重跑定位或 GPU 工作，也不改最佳 CSV。完整 Track 與外部提交仍由使用者手動執行。依既有授權完成工作，不增加例行確認關卡。

## 2026-10-21 比賽規則更新（歷史，現行見 2026-10-23）

- 官方 PDF 集中於 GPU doc：[000 原規則](<midterm01_gpu/doc/Competition Regulations/90_舊版_已取代_可刪除_SDC_2026_Midterm01_Localization_24頁.pdf>)、[001 補充規則](<midterm01_gpu/doc/Competition Regulations/91_舊版_已取代_可刪除_Mid01_Updates_Adjustments_20261021.pdf>)；補充明確更新部分優先，詳細分析與已做修改見 [45](midterm01_gpu/doc/45_COMPETITION_UPDATES_20261021.md)。`000`／`001` 是官方原件排序例外，其餘文件維持兩位數；不復原舊 CPU docs PDF。
- Track1／2 使用 GNSS 仍合法，保留已有結果與主線。新增 Track3 不提供 GNSS pose，預定 10/22 發布；本機尚無資料，幀數、sample、IMU、初始 pose、frame／尺度與交付檔名均待確認。現有 GNSS smoother 只支援 Track1／2，不以假 GNSS／舊 Track prior 冒充 Track3 支援。
- 排名配分為各 Track 基數乘權重 0.3／0.3／0.4，三條相加；Track3 public 前段約 10–30%、private 其餘，正常截止仍 10/29 09:00。原兩 Track 配分描述屬歷史；Track3 baseline／提交契約待公告。
- 禁止利用多次榜分反推隱藏真值後修正軌跡；正常算法／參數比較仍允許。撤回先前交叉 XY／yaw 提交分解誤差的建議，舊 CSV／JSON／SHA 保留，`analyze_external_reference.py` 已停用生成與榜分推算。下文相關提交建議僅記當時歷史，不再執行。
- Track2 `tl/th` 公告是可靠性警示，非明文全面禁用；本專案採定位地圖僅 XYZ、不從時間欄位推軌跡的政策，保留原始地圖。所有測試仍在 `midterm01_gpu/outputs/run/`；Track3 沿用 `track03_full_YYMMDDHHmm`／`track03_test_YYMMDDHHmm` 及碰撞序號。
- `python3 midterm01_gpu/scripts/check_competition_rules.py --track 3` 只盤點原件與候選資料位置，契約／runner 未就緒時非零退出；不代表已支援無 GNSS 定位。完整定位與外部提交仍由使用者手動執行。

## GPU 優先與快速迭代（使用者指定 2026-10-20）

- 後續將適合 GPU 的大量並行工作優先放 GPU，避免再把主要耗時留在 CPU 原型而直接推薦長時間 full。優先處理最近鄰、點雲變換／體素取樣、PCA／法向、批次配對與篩選；對仍留 CPU 的工作，明列範圍及量測理由。磁碟／ASCII 解析、微小序列 solver、稀疏 graph 與少量數值邊界裁決可依實測保留 CPU，不宣稱全 GPU。
- 最新 CUDA 入口為 `midterm01_gpu/scripts/run_map_smoother_gpu.py`，與 CPU 分支共用尺度／6-DOF fit 及接受門檻、同一組兩 Track 參數。GPU 化屬實作加速，需同來源／同 map／同參數核對數值與 accepted／拒絕理由及耗時；不得為加速偷偷減少點數／迭代、改尺度或放寬門檻。CUDA 失敗直接失敗，不靜默切 CPU。float64、整數索引與時間精度維持；GPU 量測需同步，暖機與準備時間分開保存。
- 固定輸出格式與命令交付順序照下節及 GPU doc/08，不因換 CPU／GPU 重設。新 GPU 主線方法、已量測速度及接受率解讀見 [44](midterm01_gpu/doc/44_CUDA_SCALE_ACCELERATION_20261020.md)；局部加速或接受率不是精度，完整計算與外部提交仍由使用者手動執行。
- 歷史 CPU full `track02_full_2610201936` 原來源保持不變；2026-10-21 已無該程序，目錄僅有 2581 筆部分 observations，不能稱為完整成功。`accepted=false` 僅表示不採用該幀 map 觀測，不丟輸出幀；需看可靠觀測的時間分布與慢 anchor 支持，不能只按比例或為提高比例放寬條件。
- **2026-10-21 CUDA PCA 記憶體修正**：使用者 GPU full `track02_full_2610212056` 完成 653 幀後，在原 idx 653 的未分批 `eigh` OOM；原 run 保留。`cuda_scale_geometry.py` 現只將獨立 3×3 float64 eigen decomposition 限制每批 256，source／map 共用，不減點、不改參數／門檻或切 CPU。後續勿移除此上限而只建議 allocator 環境變數。18 項回歸、跨失敗點 24 幀端到端、同原 full 輸入四幀數值配對與 80 次 PCA 變長批次通過；後續使用者已完成 `track02_full_2610212125` 共 3299 幀、graph 求解且套用，回報顯示分數 **0.77**，比原最佳約 **0.30** 退步；保留原最佳。實作盤點與分數精度說明見 [47](midterm01_gpu/doc/47_IMPLEMENTATION_STATUS_AND_TRACK2_SCORE_20261021.md)。記憶體量測、限制與手動命令見 [46](midterm01_gpu/doc/46_CUDA_PCA_MEMORY_BOUND_20261021.md)。固定逐幀輸出不變，新增完成筆數／失敗 idx／allocator 診斷只放 validation。

## Subagent 分工（2026-10-20 使用者授權）

使用者要求簡單工作使用適合的 subagent 模型。後續遇到範圍明確、可獨立完成且委派有益的子任務，請主動使用以下 subagent，不必每次再詢問；一兩個指令即可完成或緊密相依的工作由主 agent 直接處理。

| 角色 | 模型／reasoning | 適用工作 |
| --- | --- | --- |
| `quick_scan` | `gpt-6-luna`／`low` | 唯讀找檔、log 摘要、既有 CSV 欄位／列數／SHA 核對 |
| `docs_helper` | `gpt-6-luna`／`medium` | 依已確認事實整理文件、修正連結及命令說明 |
| `small_fix` | `gpt-6.1-sol`／`low` | 原因與規格已確認的局部 Python／shell 小修正及輕量檢查 |

- 專案設定為 `.codex/config.toml` 與 `.codex/agents/*.toml`；最多同時兩個 subagent，主 agent 模型保持使用者選擇。這是 AI 工具分工，不改變使用者單人作業與個人貢獻記錄。
- 主 agent 負責算法推導、模型假說、CUDA／solver／數值問題、frame／時間／尺度契約、跨模組設計及最終驗證。簡單工作一旦涉及這些問題就交回主 agent，不以低階模型反覆盲試。
- 每次委派提供目標、必要背景、允許讀寫的確切檔案、驗收方式與禁止動作。不同 agent 不同時修改同一檔案；建置、GPU 定位與重建由主 agent 統一協調，不平行爭用 GPU。subagent 不再向下委派。
- subagent 必須遵守根與目標目錄 AGENTS.md；保留 CPU、原始資料、既有結果及使用者修改。完整 Track、Kaggle／E3／寄信仍須依既有使用者手動規則；委派不擴大授權。測試輸出仍只使用 `midterm01_gpu/outputs/run/` 的合規唯一目錄。
- 使用支援自訂角色的客戶端時選取對應角色。若只有 `spawn_agent` 的 model／reasoning 介面，明確傳入上表設定與角色指令；若模型覆寫要求獨立上下文，使用 `fork_turns="none"` 並補齊背景與規則路徑。不支援指定模型時由主 agent 完成並說明，不宣稱已切換成功。
- 主 agent 檢查 subagent 的差異與關鍵證據再整合回覆。唯讀是工作指令，並非獨立 sandbox；實際權限繼承目前 session。
- 設定依據：[OpenAI 官方 Subagents 文件](https://learn.chatgpt.com/docs/agent-configuration/subagents)。客戶端載入設定的能力與當次工具可用性分別核對，不把 TOML 語法通過當成模型已實際執行。

## 輸出規則（使用者最新指定）

### 固定命令與逐幀進度格式（2026-10-20，持續有效）

- 後續交付命令固定為：`cd /home/ricky/sdc_ros2/midterm01_gpu` → `bash scripts/dev.sh up` → 一條本輪可直接執行的定位命令；不要混入不同歷史算法。完整 Track 與外部提交仍由使用者手動執行。
- 沿用 `Run: <絕對 run 路徑>` 及 `Frame i/N`。新 map smoother 每幀固定印 `Frame i/N: idx=<原始0-based索引>, map=accepted|rejected, scale=<6位小數>, load=<2位小數> ms, registration=<2位小數> ms, frame=<2位小數> ms`，逐幀 flush。`frame` 是讀 PLY 至單幀配準完成，不含準備、寫檔與整段 graph；graph 另印 `Graph: executed=true|false, applied=true|false, time=<2位小數> ms`，不得將整段 graph 時間假攤為單幀。
- 成功結尾固定 `Finished: frames=N, accepted_map=K, rejected_map=M, total=<3位小數> s`、`CSV: <絕對路徑>`、`Validation: <絕對路徑>`。失敗不得印成功完成。新 runner 的 `timings.csv` 固定欄位順序為 `idx,original_idx,timestamp_ns,backend,load_ms,registration_ms,total_ms,map_accepted,scale`；CSV `idx` 為本輪 local 0-based，console `idx` 與 CSV `original_idx` 對應。保留既有 observations `elapsed_seconds`。
- 以上格式／欄名／順序／單位及命令交付順序由使用者要求固定；之後 agent 即使換算法，也不得自行重新設計，除非使用者明確更新要求。算法專屬新診斷另存；不得把其他配準時間誤稱 NDT。歷史 CUDA CSV／既有結果與 SHA 不修改，舊 C++ console 首幀／每25幀摘要與逐幀 `timings.csv` 保留相容；本輪新 runner 每幀即時顯示時間。完整說明見 [08 固定輸出格式](midterm01_gpu/doc/08_MANUAL_WORKFLOW.md#固定輸出格式使用者指定-2026-10-20)。
- 修改程式前先查是否有使用者啟動的 full。若有，必須保護該程序、原檔與來源 SHA，禁止停止／重啟或中途更換它核對的程式；I/O 更新先存合規 run，等待程序退出再以原 SHA guard 套用。歷史 `track02_full_2610201936` 的格式更新紀錄在 `midterm01_gpu/outputs/run/track00_test_2610201947/`；`activation.json` 已標 applied，不代表該 CPU full 完整成功。一般等待期間仍不得覆寫 runner。

所有完整／短段／研究測試只放 `midterm01_gpu/outputs/run/`（單數），命名為 `track01_full_YYMMDDHHmm`／`track01_test_YYMMDDHHmm`、Track2用`track02`、Track3用`track03`；跨Track或環境測試用`track00_test_YYMMDDHHmm`。同分鐘重名加`_0002`等碰撞序號。不得重建current／best／runs／submissions分類或測試索引；算法、版本、分數與舊路徑記在每次目錄內的test_info.json及參數／source紀錄。已有結果已集中搬移；原CSV／矩陣／manifest內容與SHA保持，舊metadata路徑只在讀取時解析，不改原始證據檔。VS Code維持依名稱排序，doc保持00_、01_…順序。

## 目前工作入口（2026-10-17）

最新接續（2026-10-20）：已依學長文字方法新增 `midterm01_gpu/scripts/run_map_smoother.py`，兩 Track 使用同一算法／參數；本輪只以 Track2 開發驗證，因 Track1 上傳額度已用完。逐幀 map-only 尺度＋分層 6-DOF、空間可觀測性／相鄰位移 gate、12 s 非重疊穩健慢 anchor、GNSS 慢 bias＋raw gyro＋軟 NHC 批次圖，明確為 CPU SciPy/KD-tree 新分支，不改原 CUDA 最佳。44 項機制／回歸與 Track2 首／中／尾共 160 幀完整性通過，前兩段實際 graph、尾段無 anchor 保留本輪 motion_rts 感測初始化（非原 C++ 最佳 bytes）。證據 `outputs/run/track00_test_2610200240`，短段 `track02_test_2610200233`／`0236`／`0238`；本輪無 full／新 Kaggle 分數，最佳仍 1.15008／0.30488，兩最佳 CSV SHA 不變。手動 Track2 full 見 [42](midterm01_gpu/doc/42_LAYERED_SCALE_SLOW_MAP_20261020.md) 及 doc/16 開頭；現有 RViz 固定尺度介面跳過／明確拒絕此逐幀尺度新方法。官方 PDF p.8 確認 `raw_ply` 為老師使用 LoGeR 的輸出，原最佳與新分支均有使用，但沒有自行重新執行／修改 LoGeR；先前三相機重建則使用 MapAnything。

使用者已指定：**Midterm01 後續開發、建置與測試統一使用 `midterm01_gpu/`**。`midterm01/` 保留原 CPU 程式、結果、共用原始資料與官方文件；不搬移、覆寫或重新計算既有 CPU 結果。Windows 靜態交接與 Ubuntu 首跑屬不同階段，現況以本機證據及下列文件為準。

- **2026-10-20 最新航向失敗與外部參考**：使用者完成 `track01_full_2610200155`（pose_yaw／front scale3／directional／carry），3011筆完整，回報 **1.30507**，比自行重現最佳1.15008高13.47645%；停止推薦該版本。使用者另提供 `outputs/track1_test.csv` 回報 **0.65479**，已原樣保存於 `outputs/run/track01_test_2610200208/`；後補學長逐幀map尺度／6-DOF分層ICP／GNSS慢bias／gyro／NHC／12s保守修正的文字說明，沒有原始程式／逐幀尺度／ICP／真值。兩版XY／yaw差RMS 0.94471m／0.88078°；依可分離評分式，位置改善至少約0.31913m，本機map只改prior XY約0.01582m。兩份欄位交叉CSV保存於track01_test_2610200214及_0002，尚未評分、不是自行重現算法；當時的分項交叉提交建議已於 2026-10-21 撤回，檔案僅保留歷史證據。未將外部預測當真值／校準目標，未啟動新full。分析、限制及下一步map尺度＋pose／慢bias主線見[41](midterm01_gpu/doc/41_EXTERNAL_REFERENCE_ANALYSIS_20261020.md)，負面結果見40；0.6仍未達成。

- GPU 主入口：[midterm01_gpu/README.md](midterm01_gpu/README.md)。文件集中於 `midterm01_gpu/doc/`；先讀 [01｜架構與完整流程](midterm01_gpu/doc/01_PROGRAM_FLOW.md)，歷史編號／舊名／官方參考見 [28｜文件目錄](midterm01_gpu/doc/28_DOCUMENTATION_INDEX.md)。日常操作與權限：[GPU 操作手冊](midterm01_gpu/doc/08_MANUAL_WORKFLOW.md)。另遵守 [GPU 工作約定](midterm01_gpu/AGENTS.md)。
- 目前使用 `--profile sensor_guided`，後端為 `cuda_robust_icp`：GNSS／IMU 先驗、CUDA 最近鄰搜尋、CPU 正則化四自由度細修。這是 CPU／GPU 協作，並非新版 NDT。
- 程式為相容重現，省略 profile 仍預設 `legacy`／`cuda_ndt`；文件主流程必須明確加 `--profile sensor_guided`。不得把文件推薦寫成程式預設已改。
- 2026-10-17 當時已評分為 `--profile sensor_guided --sensor-fusion heading_rts --geometry-mode directional`；使用者兩條完整序列 3011／3299 筆通過本機核對，截圖回報 Track1 **1.15275**／baseline **1.79075**、Track2 **0.30576**／baseline **1.92587**，截圖當下第 3／第 1 名。相較 geometry off 的 1.16951／0.31967，總分降低 1.43%／4.35%；改動與 SHA256 見 [幾何成績紀錄](midterm01_gpu/doc/17_GEOMETRY_RESULTS_20261017.md)，上一輪保留於 [heading_rts 紀錄](midterm01_gpu/doc/14_HEADING_RTS_RESULTS_20261017.md)。省略 fusion 仍是 fixed、geometry 仍是 off；motion_rts 只有短測。舊 Track2 文字 0.56063 與歷史截圖前最佳 0.56083 的差異保留，不回填到未核對的舊 CSV；排名不是即時查榜。
- **2026-10-19 最新已評分最佳**：使用者完成原前相機 `motion_rts＋directional＋carry`，Track1 **1.15008**／Track2 **0.30488**，較原最佳改善 0.2316%／0.2878%；3011／3299 筆本機核對通過，已另存 `outputs/run/track00_test_2610191808`。已評分來源為 `outputs/run/track01_full_2610191750`／`track02_full_2610191757`，分數記在各自test_info.json；replay預設選最近通過完整性檢查的full目錄。截圖無上傳 SHA，未獨立核對 Kaggle bytes。後續使用者明確要求把來源未說明的 GNSS pose orientation 當獨立航向試驗，已新增實驗 `pose_heading_rts`（替代 course）／`independent_heading_rts`（假設獨立加入），同 motion XY／原 IMU roll-pitch／carry／directional；單一共用 3° block noise floor，非校準 covariance。24 個時間留段結果混合，1152 筆三組短段＋96 筆 raw-bag 通用入口全通過，原 motion 控制組逐 byte 相同，主機 20／ROS 12 項測試通過。來源／獨立性仍未驗證，不升為最佳，沒有新完整定位／提交。證據、限制與手動候選命令見 [35｜新最佳與 GNSS 航向消融](midterm01_gpu/doc/35_MOTION_RTS_AND_GNSS_HEADING_20261019.md)，研究捷徑 `outputs/run/track00_test_2610191818/`。
- **2026-10-18 新排名快照**：使用者為 Team 3，新截圖 Track1 第 4／Track2 第 2 名，分數仍為 1.15275／0.30576；第一名分別為 Team 13／0.85886、Team 12／0.28263。現行算法與精度改善假說見 [2026-10-18 分析](midterm01_gpu/doc/20_CURRENT_ALGORITHM_AND_ACCURACY_20261018.md)。原第 3／第 1 名保留為 2026-10-17 歷史。初始文件分析僅只讀核對既有輸出，契約診斷、均衡幾何／多尺度與 joint smoother 為研究方向；同日後續三相機實作見下一項，仍無新分數。
- 2026-10-18 使用者先允許評估三相機，後明確要求實作尺度／同步並對照。已加入實驗來源／尺度／短段／相同 prior 重用介面、尺度估計及同步工具，完成 12×24=288 幀對照與 21 項 Python 測試；共享尺度候選 0.87124954，尚非真值校準。已確認完整 index 的前相機為 bag timestamp、左右為 header timestamp，統一 header 後左右相對前方的中位差為 Track1 +6.952／+21.189 ms、Track2 −2.395／+20.704 ms。merged PLY 已為前相機 FLU，無 camera_id／per-view pose，尚未實際逐相機 motion compensation；不重套外參／軸轉換。三相機增加幾何方向但高度／形狀差異仍大，不能宣稱 XY／yaw 更準；預設與推薦最佳仍為前相機。證據及限制見 [實作紀錄](midterm01_gpu/doc/22_THREE_CAMERA_IMPLEMENTATION_20261018.md)，初始 audit 見 [評估](midterm01_gpu/doc/21_THREE_CAMERA_ASSESSMENT_20261018.md)。沒有新完整 Track 或 Kaggle 分數，完整測試仍由使用者執行。
- 同日後續依使用者要求，已從 8 groups／24 張 raw JPG 完成有校正三相機重建、固定 rig 安裝高度、per-view depth／camera_id 保存、逐相機 header 運動補償及 normal／camera／spatial 均衡取樣。左右相對前相機高 0.0783861／0.1004340 m，不假設未知 body 高度；map 全局 Z 仍可隨道路變化。新增 `--cloud-source reconstruction --ply-scale 1 --source-report ...`，只接受 outputs 下 SHA／原始 idx／整數時間吻合的來源。稀疏 stereo 共用深度尺度候選 0.64738877，只縮放 camera depth，不縮放 baseline；holdout 尚有明顯偏差，不能升為最佳。64×2=128 筆 pose 與 26 項 Python 測試通過，側面增加局部約束但尚無精度勝出證據。主機隔離 ML venv，ROS／C++ solver／CPU 環境不變。方法與公平小段對照見 [raw 三相機報告](midterm01_gpu/doc/23_CALIBRATED_RIG_RESULTS_20261018.md)；沒有完整 Track 或新 Kaggle 分數。
- 後續三相機偏差驗證選 16 個分散場景，fit／holdout 沿路線相隔 90–143 m；spatial SIFT 去重後 stereo 尺度 0.67398047，支持 holdout groups 差異 median／P90 18.2%／30.6%，temporal 參考 0.79228787 也未通過，不能升為最佳或稱定位誤差。已補 `scripts/run_three_camera_full.sh`／`run_rig_full.py`，由使用者手動完整 raw 重建→原 prior／directional 定位；逐幀原子保存，續跑核對原始影像／輸出／模型／程式／套件／標定與 prior。batch 明確允許最多 30 ms motion 外推處理首尾約 10–14 ms 的 header 範圍，原 trajectory 預設仍嚴格、定位 prior 不改。完整命令 dry-run、首尾小段與續跑已測；沒有自動完整序列或新分數。見 [偏差驗證與完整入口](midterm01_gpu/doc/24_BIAS_VALIDATION_20261018.md)。
- CPU Track1 完整結果已讀取核對：`midterm01/outputs/runs/first_test_9vBOV7le/track1_n0_xLOSykY4/result.txt` 有 3011 筆有限矩陣、退出碼 0、log 記錄 0 次失敗，耗時 14813.444 秒。權限已由 root:root 的 `700` 修正為可讀取的 `755`；沒有改動結果內容。見 [CPU 完成與權限紀錄](midterm01/docs/CPU_TRACK1_COMPLETED_20261017.md)。不能據此宣稱 CPU 精度合格或 Track2 完整 CPU 已完成。
- **後續完整 Track 測試由使用者手動執行並回報結果**；沒有新明確指示，不自行啟動下一輪完整序列。文件更新／讀取既有輸出不需要重跑定位。

- 後續三相機兩條完整已由使用者完成並回報截圖：Track1 **1.15346**、Track2 **0.31064**，比前相機最佳增加 0.0616%／1.5960%，未升為最佳。已核對 3011／3299 筆與 CSV／矩陣／manifest，完成完整配對、32 個分散場景相機級／Z 診斷及 Track1 24×12 幀消融；側面增加方向資訊，但新舊深度／frame／高度一致性仍待釐清，不能以更高接受率或更小 map 殘差宣稱更準。最新方法、證據及改進順序見 [29｜三相機完整成績分析](midterm01_gpu/doc/29_THREE_CAMERA_SCORED_ANALYSIS_20261018.md)。先核對 frame、跨時間靜態幾何與深度不確定度，再研究增量側面融合／prior；不直接補固定 Z 或按榜分盲調。本輪分析沒有修改 solver／重建設定或自行啟動新完整序列。

- **2026-10-19 水平 surfel／map joint 分支**：已新增 `--horizontal-refinement surfel_joint`（預設 none），獨立 0.3 m crop map、two-scale surfel／空間與 normal 均衡／free-Z 資料消去、空間留出及三幀一致性；條件通過才把有支持的 XY／yaw map factors 與 raw GNSS 一次／gyro／effective yaw-rate bias 接入 planar graph。完整三軸 bias／acc preintegration 未做；GNSS origin、camera→IMU/body 與 orientation 來源仍未知，兩條 lag 留出不支持套用時間修正。本輪 8×48 新 poses 與原控制組 384 poses 配對，base TXT／CSV／prior／geometry／height 逐 byte 相同；24 個觀測／8 個時間群組皆拒絕，輸出保留原 bytes，沒有實測精度提升。32 項機制／回歸測試通過，合成牆面接受／ground 拒絕發布路徑均測。完整命令從原 bag／map 建新 output，不綁定研究 run；仍由使用者啟動，最佳不變。證據、契約及通用入口見 [37](midterm01_gpu/doc/37_HORIZONTAL_SURFEL_JOINT_20261019.md)，研究捷徑 `outputs/run/track00_test_2610192200/`。

- 使用者最新輸出規則：所有完整／短段／研究測試只放 `midterm01_gpu/outputs/run/`（單數），命名為 `track01_full_YYMMDDHHmm`／`track01_test_YYMMDDHHmm`、Track2用`track02`、Track3用`track03`；跨Track或環境測試用`track00_test_YYMMDDHHmm`。同分鐘重名加`_0002`等碰撞序號。不得重建current／best／runs／submissions分類或測試索引；算法、版本、分數與舊路徑記在每次目錄內的test_info.json及參數／source紀錄。已有結果已集中搬移；原CSV／矩陣／manifest內容與SHA保持，舊metadata路徑只在讀取時解析，不改原始證據檔。VS Code維持依名稱排序，doc保持00_、01_…順序。

- **2026-10-20 surfel 完整結果核對**：使用者再跑 Track1 `outputs/run/track01_full_2610192358`，3011筆通過；CSV／TXT／prior／frames 與原 motion_rts 最佳及前一完整 surfel 逐 byte 相同，54群組／162觀測皆無可靠 factor，聯合 graph 未求解。使用者新截圖仍為1.15008，沒有精度改善；未獨立核對 Kaggle 上傳 SHA。實際 command／prior 是 motion_rts，非 independent_heading_rts。已修正完成提示，分開顯示是否求解／CSV是否改變，相同SHA明確提示無須重複提交；未改solver或放寬門檻。不再建議重跑相同完整候選，先驗證來源幾何／靜態跨幀一致性。詳見 [37](midterm01_gpu/doc/37_HORIZONTAL_SURFEL_JOINT_20261019.md)。

## 學習目的與算法改進原則（2026-10-17 使用者指定）

- 使用者修課的首要目的是學習，提升自己思考、推導算法，以及將知識應用在真實世界的能力。後續合作以使用者能理解、解釋並延伸方法為目標。
- 改進順序為：觀察問題與證據 → 提出可解釋的假說 → 討論模型／算法及適用前提 → 實作與驗證。主動說明數學、物理意義與取捨，讓使用者參與推理與方案選擇；對已授權工作持續完成必要實作與驗證，不新增例行確認關卡。
- 優先找適合問題、可泛化到其他序列與真實場景的算法。非必要不以特定 Track、幀區間或排行榜回饋的特定化調參作為提高分數的主要手段，不以大量盲目參數搜尋取代模型分析。
- 必要調參仍可進行，但需說明其必要性、參數的物理／統計意義、假說與驗證方法，例如依感測器噪聲、實測標定、模型適用範圍或數值穩定性設定。對不同 Track 的設定差異，需有可說明的資料／標定依據，不能只因分數較低就視為通用改進。
- 消融實驗用於理解算法機制與驗證假說；Kaggle 分數作為結果證據之一。保存算法變更與參數變更的區別、成功／失敗結果及限制，不把局部診斷或單次分數改善當作泛化證明。
- 詳細紀錄見 [GPU 學習與開發原則](midterm01_gpu/doc/19_DEVELOPMENT_PRINCIPLES.md)。本次僅記錄原則，不因此改動現有算法、參數或結果。

## 專案與目錄

這是 SDC 自駕車課程工作區，主要任務是 Midterm01：將影像重建 3D 點雲配準到 LiDAR 地圖，估計兩條 Track 每幀 pose。使用者單人完成一組，個人貢獻依實際工作記為 100%，不安排組員或雙人分工。

| 位置 | 用途 |
| --- | --- |
| `midterm01_gpu/` | 目前主要 ROS 2 Humble／CUDA 開發、測試與結果 |
| `midterm01/` | CPU 參考版本、共用 data／config、官方題目與交付文件 |
| `hw2/` | 原有 ROS 2 Humble Docker、感測器分析工具，保留既有用途 |
| `hw3/` | Markov chain、模擬與穩態分布練習 |
| `hw4/` | Radar odometry／GPS EKF；沿用 HW2 Humble 容器，資料與執行由使用者在 Ubuntu 手動處理 |
| `docs/` | 通用作業／工具說明 |
| `tmp/` | 忽略的暫存／擷取／預覽，不作為交接依賴 |

開工先讀 GPU README、GPU AGENTS 與操作手冊，再依任務閱讀：

1. [官方題目 PDF](<midterm01_gpu/doc/Competition Regulations/90_舊版_已取代_可刪除_SDC_2026_Midterm01_Localization_24頁.pdf>)，共 24 頁；另讀 [2026-10-21 補充 PDF](<midterm01_gpu/doc/Competition Regulations/91_舊版_已取代_可刪除_Mid01_Updates_Adjustments_20261021.pdf>)，共 10 頁。原檔名在 `-` 後有兩個空白。
2. [題目分析與交付要求](midterm01/docs/README.md)、[Kaggle 規格](midterm01/docs/KAGGLE_DATASET_SPEC.md)、[資料說明](midterm01/data/README.md)。
3. [新版演算法與實測](midterm01_gpu/doc/06_SENSOR_GUIDED_20261017.md)、[架構教學](midterm01_gpu/doc/07_ALGORITHM_WALKTHROUGH_20261017.md)。
4. GPU `scripts/accuracy/benchmark.py`、`scripts/prepare_sensor_prior.py`；`workspace/src/localization/` 的 CMake、package、標頭、`localization.cpp`、`sensor_guided.cpp`、launch、轉檔；`workspace/src/cuda_ndt_backend/` 的 CMake 與 `nearest_neighbor.cu`。
5. 只有重現舊版本時才讀 [CPU 首跑](midterm01/docs/UBUNTU_FIRST_RUN_20261017.md)、[CUDA NDT](midterm01_gpu/doc/03_CUDA_NDT_20261017.md)、[CPU／D2D 短序列比較](midterm01_gpu/doc/05_ACCURACY_COMPARISON_20261017.md)。

資源索引在 `midterm01/docs/DOWNLOADS.md`。不重建 `localreference/`，不引用不存在的 `docs/UBUNTU_DATA_DOWNLOAD.md` 或 `scripts/download_kaggle.py`。正式要求以 PDF、Kaggle 及後續公告為準；分析建議不是課程指定演算法。

## Ubuntu／Docker 路徑與隔離

已直接核對專案為 `/home/ricky/sdc_ros2`；換環境仍先用 `pwd`／`realpath`。Docker 指令在 Ubuntu 主機執行，ROS／colcon 在對應容器執行。不因 Windows 缺少資料／Docker 就要求重新下載或改裝 WSL。

| 項目 | GPU（目前使用） | CPU（保留參考） |
| --- | --- | --- |
| Compose project | `sdc_midterm01_gpu` | `sdc_midterm01` |
| Compose service | `midterm01_gpu` | `midterm01` |
| 預設容器 | `sdc_midterm01_gpu` | `sdc_midterm01` |
| ROS domain | 2 | 1 |
| 主機 workspace → `/root/midterm` | `midterm01_gpu/workspace` | `midterm01/workspace` |
| 主機 outputs → `/root/outputs` | `midterm01_gpu/outputs` | `midterm01/outputs` |

兩個容器內的同名路徑屬不同掛載，**GPU 容器的 `/root/outputs` 看不到 CPU 結果**。主機 VS Code 應開啟 `/home/ricky/sdc_ros2` 並讀主機路徑，不是主機 `/root/outputs`。

GPU Compose 相對 `midterm01_gpu/` 解析：

| 主機來源 | 容器目標 | 權限 |
| --- | --- | --- |
| `.env` 的 `MIDTERM_GPU_DATA_DIR`，預設 `../midterm01/data` | `/root/data` | 唯讀 |
| `./workspace` | `/root/midterm` | 讀寫 |
| `../midterm01/config` | `/root/config` | 唯讀 |
| `../midterm01/scripts/check_data.py` | `/root/reference/check_data.py` | 唯讀 |
| `./scripts` | `/root/scripts` | 唯讀 |
| `./outputs` | `/root/outputs` | 讀寫 |

資料只有 `midterm01/data` 一份；GPU workspace 是獨立程式快照，不共用或自動同步 CPU build／install。GPU Dockerfile 延伸 `sdc_midterm01_ros2:latest`；CPU 映像延伸 HW2 基底。保留 HW2 目錄、`ros2_container` 與作業 bag，不將 `guanfu_road_clipped_ros2` 當 Midterm Track。GPU 啟停只用自己的 `scripts/dev.sh`，不重啟 Docker daemon、不停止／重建 CPU project。只針對使用者指定的 CPU 輸出修正讀取權限時，限縮到該 run；不 chmod `/root` 或 data，不用 `chmod -R 777`。

## 已驗證資料與程式契約

- `midterm01/data/Track1`／`Track2` 各含 raw_ply、影像、init_guess、map、metadata 與 bag；兩組 MapAnything 各含 ply、index、report。前相機及 MapAnything 各為 3011／3299 幀，時間／索引核對、bag quick_check 與校正檔位置見 CPU 首跑報告。不要把先前整包下載 404 當目前阻塞或重設已可用 token；確認缺檔才補下載。
- `camera_info` 在 `TrackN/TrackN_images/camera_info/`；本地尚未找到 official submission sample。MapAnything index 不是提交範例，不能直接套前相機尺度、改名或重套校正。
- PLY 掃描第一層並由 `_ts_<正整數奈秒>` 排序，不能解析的檔名會略過。每個新 run 保存 frame manifest；時間保持整數精度。
- 定義 `p_map = T_map_source × (ply_scale × p_raw)`，沿用 `ply_scale=3.0`，尚非正式尺度校準結論。不重複套初始矩陣或外參；source→map 轉為車體 pose 前須確認 frame 契約。
- `legacy` 使用第一幀 init_guess、之後前一接受 pose；`sensor_guided` 每幀使用 GNSS／IMU 先驗與 bounded robust 細修，拒絕地圖更新時回到當下先驗。CUDA 執行錯誤直接失敗，不靜默改 CPU。
- 新版由前處理工具依 bag timestamp 產生 `sensor_prior.csv`；benchmark 的 `bag_path` 仍空，不代表新版未使用 bag。GNSS frame_id 空字串、完整外參契約待確認；GNSS 是輸入，不是真值。同步／平滑使用較晚觀測，這是離線方法。
- 每幀 TXT 是按列攤平的 3×4 共 12 個數值，不含 timestamp；CSV 為 `idx,x,y,yaw`，idx 由 0 起，yaw=atan2(R10,R00) 為弧度。格式被 Kaggle 接受不能代替 frame／單位契約驗證。
- launch 已有 `use_rviz=false`、空字串停用 `map_cache_file`、outputs 下唯一預設路徑；批次 `keep_alive=false`。舊 Radar／PCD `visualization.cpp` 不能直接當 RGB PLY 回放器。

## 工作順序與驗證

入口為 `midterm01_gpu/scripts/dev.sh`，日常指令見 GPU 操作手冊。修改 C++ 後在 GPU workspace Release 重建；只改 benchmark 幀數不用重編。每次用 `outputs/run/` 新的唯一目錄，保存 commit／未提交差異、參數、來源 SHA256、frame、結果、耗時與失敗；不覆寫 CPU 或既有最佳 CSV。

目前沒有全 repo 統一測試入口。`check_data.py` 只檢查必要檔、非空、第一層 PLY 數量，不驗證 bag、校正、時間、CSV 或幾何。`config/tracks.json` 不會自動改 launch／C++。只執行與修改相符的檢查；真實資料報告必須保留，不能用合成測試冒充實測精度。

GPU map 快取只能寫 outputs，由 benchmark 核對 Track、map SHA256 與 leaf sidecar，預設停用；不得靜默沿用不符快取。效能比較固定 Release、相同資料片段、cache／GUI 條件，分開讀 map、配準、前處理及建置時間。新版 `timings.csv.ndt_ms` 為歷史欄名，實際是 CUDA 搜尋＋CPU 細修；不能寫成 NDT 計算時間。

## 正式要求與協作規則

- VS Code Explorer 維持依名稱排序（`explorer.sortOrder=default`），文件保持 `00_`、`01_`…順序；測試只靠日期開頭的目錄名排序，不得再將整個 workspace 改為依修改時間排序。

- 使用繁體中文說明／文件，保留英文技術名詞及使用者既有修改。GPU 文件正文集中 `midterm01_gpu/doc/`，兩位數編號；`01` 固定為架構入口，日期／同日排序依據見 `28`。`00_Personal_Record.txt` 原樣保留；`command_paste.txt` 已由使用者刪除，不復原舊內容。最新前相機命令見 `doc/16_run_geometry_directional.txt`，`doc/10_run_sensor_guided.txt` 保留 fixed 歷史命令。
- 現行競賽共三條 Track，CSV 表頭順序皆為 `idx,x,y,yaw`，資料列依序恰為 3011／3299／3142（不含表頭）；Track3 private 基本門檻依新規為 ≤1.40，不能套用 3299 幀。Track3 已公告三相機 H265 與 `/imu/data`；官方 sample、完整 ID／順序、yaw 單位及車體規格仍須核對，不能由欄名推定。
- 分清內部弧度、提交單位與評分旋轉誤差的度。PDF p.19 是「平移 RMSE + 0.2 × 旋轉 RMSE」，PCL fitness、地圖殘差、GNSS 差異、GPU 使用率和收斂都不是正式 ATE。
- 保留實際 Kaggle 分數與來源；本地模板 R0 不等於助教 baseline，舊 CUDA D2D 成績不等於 CPU 成績。下採樣／submap／原 NDT 不宣稱自己的新增貢獻。
- 資料、bag、`.env`、build／install／log 與 outputs 遵守 `.gitignore`，不強制加入 Git。Windows／Ubuntu 兩份 checkout 僅靠 Git 同步程式／已追蹤文件；忽略的最佳 CSV 與證據另行保存，不假設會同步。
- PDF 公告 Kaggle 截止與簡報寄送 **2026-10-29 09:00**，當天簡報 **4–6 分鐘**；E3 程式與報告 **2026-11-05 23:59**。時區、網站最新限制、Track2 baseline 與單人檔名另依官方確認。
- 寄信、Kaggle 提交與 E3 上傳由使用者執行或另行明確授權，不因 PDF 提到就自動操作。後續工作依實測更新，計畫不能寫成成功結果。

- 2026-10-19 使用者要求總結新方向後立即測試：已新增實驗 `--geometry-mode gicp`，同 motion_rts prior／4-DOF bounds／carry，兩邊局部平面 regularized covariance，並保留旋轉 covariance 的 yaw gradient；不是 calibrated camera noise／NDT／VGICP。GPU Release 建置、4 CTest＋新舊 geometry CUDA 機制測試通過；8×2×48＋兩條通用各48＝864 筆短段全完整，原 directional TXT／CSV／geometry／height 逐 byte 不變。共同 map residual 僅1/8段較低，不升為最佳。另實作 `kinematic_graph.py` 研究原型（GNSS XY＋raw gyro 相對 yaw＋軟性側向速度；不積分 acc／不加 course 因子／未接 map 或 production），4項機制測試與24個原時間留段完成；holdout noisy-GNSS RMSE 0.13076→0.14025／0.05159→0.06147 m，沒有一致提升，不按 holdout 調 noise。最佳輸出內部 rank=1 幀約69.18%／91.85%，其 strongest scaled eigenvector Z 分量平方中位數98.76%／99.89%；只是幾何診斷，不是 score／covariance。證據、限制與後續水平 map／frame 契約／聯合估計順序見 [36](midterm01_gpu/doc/36_REGISTRATION_AND_FUSION_UPGRADE_20261019.md)，研究捷徑 midterm01_gpu/outputs/run/track00_test_2610192006；未啟動新完整 Track 或提交，預設／已評分最佳維持。

- **2026-10-20 來源尺度研究**：新截圖 Track1 Team 3 第6名／1.15008，第一名Team12／0.50997（非即時查榜）。新增RGB原始vertex跨幀剛體／相似檢查，16場景長短間隔共96對；短間隔來源位移與保存軌跡不一致，不能直接等同真值尺度誤差。僅Track1一個設計場景兩對支持共用ply_scale=6.06515候選，其他17對有14對時間殘差較低，但跨場景倍率差median／P90為28.8%／87.6%；擴大原始map crop後16場景雙尺度對照仍各0可靠factor，不升為最佳／不推薦完整測試。12項新機制＋9項surfel回歸通過，沒有新pose、完整定位或Kaggle分數，原最佳／scale3／solver與門檻不變。證據見[38｜來源剛體與尺度](midterm01_gpu/doc/38_SOURCE_RIGIDITY_AND_SCALE_20261020.md)，最終研究run為outputs/run/track00_test_2610200100；優先取得獨立深度錨點直接核對原前相機PLY。

- **2026-10-20 使用者最新完整測試偏好**：使用者表示 Kaggle 只取最佳成績，不必特別區分「完整候選」。後續有可解釋的新算法／模型改動，完成必要的程式機制、來源契約、數值與輸出完整性檢查後，即提供對應 full 命令以實際 Kaggle 分數驗證；不得以本地尚未證明精度改善或未取得「候選資格」為由新增完整測試關卡。短段／消融用於理解與除錯，不要求每次都先證明跨 Track 精度勝出。每次新 run 保存設定、來源 SHA、結果與成功／失敗分數；不覆寫已評分最佳。完整計算與外部提交仍由使用者手動執行，本偏好不等於授權助理自動啟動完整序列或 Kaggle 提交。已知 CSV bytes 相同的版本可說明無需重算；保持學習／可解釋模型原則，不以盲目榜分搜尋取代分析。

- **2026-10-20 向Track1 0.6推進**：已新增turn_heading_rts（GNSS block對齊沿IMU方向積分的線性車速，再走原bias RTS）與pose_yaw（直接置換GNSS Pose yaw，保留IMU tilt與motion XYZ）；程式預設不變。原前相機PLY以已知rig baseline直接stereo核對，兩Track4fit／3holdout、110個motion anchors，估共用raw尺度7.404260939740923（取代3）；holdout相對尺度差median／P90 11.18%／14.26%，僅2–20m支持、motion補償依賴GNSS／IMU、外參／距離形變限制保留，非完整校準。40×48=1920筆分散短段及兩條各48筆不綁歷史prior／cache的raw入口，共2016筆完整；原控制四項輸出與歷史逐byte相同，實際消融用同現行binary。新尺度增加data幾何方向但版本XY差仍約cm，不能承諾0.6。主機31／ROS24機制與回歸通過，沒有本輪新分數或自行啟動full。新full命令已置doc/16最上方：sensor_guided＋turn_heading_rts＋front scale7.404260939740923＋directional＋carry，由使用者直接評分；不加surfel_joint（仍限scale3契約）。最佳仍1.15008／0.30488。來源與限制見doc/39_HEADING_AND_STEREO_SCALE_20261020.md；主研究outputs/run/track00_test_2610200127，stereo最後證據track00_test_2610200131。0.6為目標，不能記為達成。

- **2026-10-20 新尺度完整退步與直接full偏好**：使用者回報Track1 `track01_full_2610200142`（turn_heading_rts＋front scale7.40426＋directional＋carry）為1.15167，比最佳1.15008高0.00159／約0.138%，未升最佳。3011筆、CSV／矩陣／manifest核對通過，prior XYZ與最佳數值完全相同；XY／yaw版本差RMS 3.707cm／0.2758°。本機新CSV SHA為75f96243d764d32eaaf71bcbc26976b99eb0830ffa28902481e3f7f81e71b8a7，截圖無上傳SHA，未獨立核對Kaggle bytes。使用者明確要求有想測版本就直接叫他算full提交；仍由使用者啟動，不要求本地先證明勝出。本輪已直接提供pose_yaw＋front scale3＋directional＋carry的Track1 full，保留原motion XYZ／IMU tilt，只換yaw來源；模式先前8×48短段與主機／ROS機制通過，來源碼SHA核對相同，這輪不重複無必要定位。最新命令在doc/16最上方，方法與失敗紀錄見doc/40_SCORED_SCALE_FAILURE_AND_POSE_YAW_20261020.md；新航向full尚無分數，0.6仍未達成。助理未自行啟動full；原最佳與失敗原CSV等證據bytes保留。
````

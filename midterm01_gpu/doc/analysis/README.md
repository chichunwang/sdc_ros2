# Python／Matplotlib 算法學習圖解

> **歷史教學文件（2026-10-23 更新）**：本檔以下原有集中產圖路徑與命令是舊規紀錄，保留供閱讀，不再是新 full run 的圖表入口，也不應照此為歷史 run 補圖。自本更新起，每次成功完成的 full track 才在該 run 的 `analysis/` 產生事後唯讀診斷圖；短段、失敗或未完成 run 預設不產生完整比較圖。標準圖、來源 SHA、狀態與唯讀限制見 [GPU 操作手冊](../08_MANUAL_WORKFLOW.md#full-run-事後圖表)。

舊教學程式、圖表與個人紀錄原樣保留；本更新不重新生成圖，也不把教學示範改成定位輸出。

2026-10-21：依你的最新選擇，以 **Python／Matplotlib** 作主要教學工具，不需要 MATLAB。維持學習優先、非必要不調參；本目錄的程式只讀既有診斷或建立合成示範，不改定位算法／參數，也不重新跑定位。

## 直接執行

在 Ubuntu 主機終端執行；這是讀檔繪圖，不需要啟動 Docker：

```bash
cd /home/ricky/sdc_ros2/midterm01_gpu
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 outputs/ml_runtime/venv/bin/python doc/analysis/plot_learning.py
```

每次另存 **5 張 PNG＋5 張 PDF** 到新的 `doc/analysis/figures/python_YYMMDDHHmmss/`，並留下 `manifest.json`（資料／程式／圖 SHA、實際套件版本與示範數值）。圖片可直接在 VS Code 開啟，PDF 適合放大閱讀；不覆寫前次圖。定位／研究測試紀錄仍在 `outputs/run/`。

若只看一個問題，在同一命令後加：

```text
--lesson icp       只看 ICP 可觀測性合成例，不需要真實資料包
--lesson kalman    只看 KF／RTS 與 bias 合成例，不需要真實資料包
--lesson real      只看已保存 Track2 的三張分析圖
```

桌面環境可加 `--show`，使用 Matplotlib 視窗的 zoom／pan／3D 旋轉；此選項需要可用 GUI backend。預設只存檔，已在無視窗的 Agg backend 實際執行。

## 圖與程式對照

建議先看兩個知道真值的合成例，再讀真實資料：

| 順序 | 已產出的圖 | 學習重點 |
| --- | --- | --- |
| 1 | [ICP 可觀測性](figures/python_261021224025/04_icp_observability_synthetic.png) | 地面配得很平，為何 XY／yaw 仍可能不準？尺度和姿態如何互相補償？ |
| 2 | [Kalman／RTS 與偏差](figures/python_261021224025/05_kalman_bias_synthetic.png) | KF／RTS 用哪些資訊？平滑為何不能單獨辨識 GNSS 慢偏差？ |
| 3 | [軌跡修正](figures/python_261021224025/01_trajectory_changes.png) | 最新結果和舊最佳各改動共同先驗多少？ |
| 4 | [map 觀測可信度](figures/python_261021224025/02_map_reliability.png) | 幾何接受、相鄰位移檢查、慢 anchor 是哪些不同關卡？ |
| 5 | [慢偏差與逐幀耗時](figures/python_261021224025/03_bias_and_timing.png) | map−GNSS、anchor、bias state 的正負號與差異？ |

入口是 [plot_learning.py](plot_learning.py)。合成數學放在 [learning_models.py](learning_models.py)，三張真實資料圖放在 [plot_saved_run_py.py](plot_saved_run_py.py)。程式註解與教學為繁體中文，圖內以英文標籤減少字型差異。

每個圖組同時有同名 `.pdf`，完整產物見 [本輪圖目錄](figures/python_261021224025/) 與 [manifest](figures/python_261021224025/manifest.json)。真實資料圖中的曲線、殘差和版本差都不能當成真值誤差；合成圖的真值只屬於示範情境。

## 真實資料來源與可重現性

[教學資料包](data/track02_learning.mat)（約 542 KiB）保留既有 `track02_full_2610212125` 和歷史最佳 `track02_full_2610191757` 的診斷。檔案延用先前 MAT 容器，由 **Python SciPy 直接讀取**，不會啟動 MATLAB，也不需要 MATLAB 授權。打開圖只需 NumPy、SciPy、Matplotlib；目前命令沿用本機已安裝這些套件的 ML venv，不更動 ROS／CPU 環境。

資料先核對影格時間、索引、列數及來源 SHA，保留 int64 奈秒，先減共同整數 origin 再轉秒；超出 raw GNSS 覆蓋範圍處的插值是 NaN。這組兩版本的 sensor prior XYZ／quaternion 已逐值核對相同，圖中相對先驗的差異有共同基準。另選不同 prior 的實驗時，不能把所有版本差都歸因於 map 配準。

資料包缺少時，主入口會用 [export_learning_data.py](export_learning_data.py) 從上述既有 run 匯出；不讀原始 bag／PLY、不配準、不求解新 graph。來源 run 也不在本機時，可先用 `--lesson icp` 或 `--lesson kalman`。

另選資料可用 `--bundle doc/analysis/data/another_learning.mat`。要重新匯出另一組，先執行同一 Python 環境的 `doc/analysis/export_learning_data.py --run <new_run> --reference-run <reference_run> --output doc/analysis/data/another_learning.mat`；兩個 run 需同 Track／相同影格時間且來源已通過完整性檢查。本課需 new run 有實際執行的 slow-bias graph，不捏造缺少的狀態。

同名資料包拒絕覆寫。[資料 manifest](data/track02_learning.json) 記錄來源與資料 SHA。`data/`、`figures/` 是忽略的本機衍生資料，不假設會隨 Git 同步；換電腦可複製本目錄 Python 程式與資料包，或從已保存 run 重建。舊 `.m` 與舊 MATLAB 靜態檢查證據保留作歷史，現行學習命令全部使用 Python。

## 圖 1：ICP 為什麼需要牆面

這張合成圖採**已知正確對應**，只看局部 point-to-plane 線性化，省略最近鄰更新、robust weighting 與接受 gate，不能當成完整 ICP 重現。

\[
r_i=n_i^T(sRp_i+t-q_i),\quad
J_i=[n_i^T,\ (sRp_i\times n_i)^T,\ n_i^T(sRp_i)].
\]

只有水平地面時 `n=[0,0,1]`，X／Y 平移與繞 Z 旋轉對殘差的一階影響為零。沿地面滑動，點到平面的誤差仍可很小。加入不同方向、相對兩側的牆面，才能補上水平與尺度的約束。

先看整個七變數 Jacobian 的奇異值，再將 Z／roll／pitch／scale 的影響投影出去，檢查剩下 XY／yaw 的支持。這對應 [map_scale_icp.py](../../scripts/map_scale_icp.py) 的可觀測性想法。奇異值隨變數單位與點數而變；圖中下限 `1e-14` 只是為了在 log 軸畫出零附近，不是 production gate 或已標定 covariance。

自己先預測：若拿掉兩面 X 方向的牆，只剩地面和 Y 方向牆，哪個水平自由度會失去支持？可修改**合成場景**觀察；這不會修改定位參數。

## 圖 2：平滑與辨識偏差是兩個問題

合成例以 `[位置,速度]` 常速度模型做 KF，再用後面的量測作 RTS 反向平滑。它示範 [sensor_fusion.py](../../scripts/sensor_fusion.py) 中位置平滑的基本結構，省略 production 的二維、innovation 降權等細節。合成噪聲／模型數值是展示條件，不能回填成真車標定結果。

平滑可以減少高頻跳動，但若量測實際為

\[
z_{GNSS}=p+b+\epsilon,
\]

只靠同一筆 GNSS，無法區分 `p` 和慢偏差 `b`：許多組 `p+b=z` 都能符合量測。右下圖加入**理想獨立的合成絕對位置量測**後，等成本谷底才被限制。實際專案的 map 配準依賴 GNSS 初始化、慢 anchor 又使用 map−GNSS，並非這個理想獨立量測；此例不能替真實相關性或錯配提供保證。

先預測：若移除合成 GNSS 的慢偏差，但保留隨機雜訊，KF／RTS 誤差曲線會如何變化？重點是解釋原因，不是搜尋最低 RMSE 的參數。

## 真實資料三張圖怎麼看

1. **軌跡與先驗修正**：raw GNSS、sensor prior、原最佳與最新輸出使用相同 map 座標，只減共同顯示原點，沒有額外對齊擬合。下方是相對先驗的 XY／yaw 變化，不能稱為定位誤差或 ATE。
2. **map 觀測可信度**：尺度候選包含拒絕幀；幾何通過後還有相鄰位移 gate，最後才可能構成慢 anchor。小殘差、`accepted=true` 都不保證對到了正確位置。
3. **慢偏差與時間**：圖畫 map−GNSS，因此疊的是 `-bias`。圖模型是 `z=p+b`，所以輸出位置修正約為 `-b`。raw 候選、中位數 anchor、圖的 bias state 是不同量；gyro bias 也是有效 yaw-rate 模型狀態，不是完整三軸 IMU 標定。耗時圖只含逐幀 load／registration／frame，沒有把 graph 假攤進去。

一個可檢驗的學習問題：在 11 個 anchor 之間，位置與航向如何靠運動模型延續？如果整段缺少可靠 map 支持，哪個量會重新變得難以辨識？先用圖描述現象，再回到 [slow_map_graph.py](../../scripts/slow_map_graph.py) 的狀態與殘差逐項對照。不要把 raw GNSS、map 殘差或舊最佳當成真值，也不從 0.77／0.30 反推誤差分量。

## 驗證狀態與 API 來源

Python 主入口已實際執行並完成 5 張 PNG＋5 張 PDF；檢查實際渲染、資料 SHA／整數時間、合成幾何可觀測性與 KF／RTS 的數學條件。原最佳 CSV 保留不變，沒有更動 production 參數或重跑定位。驗證紀錄放在 [本輪研究紀錄](../../outputs/run/track00_test_2610212240/)；圖形檔也會單獨記錄 SHA，避免以規劃冒充產物。

Matplotlib 存檔介面參考官方 [savefig](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html)，視窗與無視窗模式參考 [backends](https://matplotlib.org/stable/users/explain/figure/backends.html)；Python 讀取既有 MAT 容器使用 SciPy 官方 [loadmat](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html)。

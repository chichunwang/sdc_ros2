# Track2 v5.3 fix2 + perf2：用 map data-only likelihood 更新高度

## 目前狀態與目標

目前 live 為 `track2_causal_time_reference_likelihood_v5_3_fix2_perf2`。fix2 + perf2 已在 2026-11-02 依 SHA 與 GPU lock 保護套用；5 個檔案符合候選來源，36 個受保護檔案未變，perf2 的 PLY／camera／ROS／runner 路徑保留。149 項生成／CUDA regression 通過，耗時 18.25 秒。這些檢查不是 Track2 精度證明；fix2 尚無真實影格、full 或新成績。

使用者回報 fix1 full `track02_full_2610311843` 的 Public Score 為 `0.46899`。先前 v5.2 的 `0.46434` 與 v5.3 + perf1 的 `0.80300` 也都是使用者回報、來自不同版本。使用者本輪的 exact 目標值為 `0.30488`。參照檔 `midterm01_gpu/doc/analysis doc/result_2_0.30488.csv` 只做 hash／provenance 核對，hash 與 fix1 full 的 immutable source manifest 所記 `result_2_submitnoncasual.csv` 相同；未分析 CSV 值，也不作 ground truth 或參數擬合。fix2 是否能改善成績尚未知。

完整定位仍由使用者手動啟動，使用固定命令 `cd /home/ricky/sdc_ros2/midterm01_gpu && ./scripts/run_track2.sh --full`。不新增 flags 或啟動關卡；Track2 會自行開啟 RViz。

## 觀察與限制

| 觀察 | 數值 | 解讀 |
| --- | ---: | --- |
| fix1 輸出時間戳 | 3299 筆 pose timestamp 全等於 camera header | 程式按感測器 header 對齊，並不證明每個感測器量測的物理 epoch 完全同步 |
| Camera arrival − header | median `7.3677 ms` | 收件與 header 時間差 |
| GNSS arrival − header | median `1.095209 ms` | 收件與 header 時間差 |
| Camera 時刻的 latest GNSS fix age | median `124.54232 ms` | GNSS CV filter 已把位置先驗外推至 camera 時刻 |
| IMU 樣本 age | median `4.987611 ms`；max `14.682355 ms` | 相對 camera header 的樣本 age |
| GNSS 高度欄位 | 475 筆全為 `17.3443660736084 m` | 常數欄位，不當高度真值使用 |

GNSS quaternion 的物理航向 epoch 仍未知，不能由 header arrival 差推定固定延遲，也沒有套用固定秒數平移。Bag topics 只有三路影像、IMU 與 GNSS，沒有 TF；`camera_info` 提供 intrinsics，`lucid_camera_extrinsics` 只有相機間 transform，沒有 GNSS／IMU 外參。使用者確認維持未知外參假設：lever arm mean 為 0、std 為 0.5 m，旋轉外參未標定，fix2 不更動這些值。

高度分析中，經 IMU prior 旋轉後的 source normal 與 map ground normal 角度差超過 `8°` 的只有 7 幀，且舊版 map 在這些幀均被拒絕，現有證據不足以新增 tilt gate 或推定旋轉外參。

## 問題：MAP candidate 又被當成獨立高度量測

舊流程的 registration solver 已使用當幀高度先驗求 MAP candidate，之後又將 candidate height 當成一筆獨立量測更新高度 Kalman filter。candidate 內含先驗帶來的資訊，第二次融合會重複計入同一先驗，且 candidate scale 與記憶化後的高度可能不一致。

fix2 從 registration 保留的 **data-only** Hessian 與 gradient 直接產生高度 likelihood；這個量測不再包含 sensor prior。既有 scale coupling、先驗高度與品質膨脹仍保留。

## 高度 likelihood 的推導

將 data-only 二次模型的狀態分成平面 pose nuisance 變數 \(u=[x,y,\psi]\) 與 log-scale \(s\)。依此分割 Hessian \(H\) 與 gradient \(g\)：

\[
H=
\begin{bmatrix}
H_{uu} & H_{us}\\
H_{su} & H_{ss}
\end{bmatrix},\qquad
g=
\begin{bmatrix}g_u\\g_s\end{bmatrix}.
\]

先在可觀測的平面子空間消去 \(u\)，取得剩餘的 log-scale 資訊與 gradient：

\[
I=H_{ss}-H_{su}H_{uu}^{\dagger}H_{us},\qquad
\gamma=g_s-H_{su}H_{uu}^{\dagger}g_u.
\]

這裡的 \(H_{uu}^{\dagger}\) 只在保留的 observable modes 上求解。若 \(I\) 接近零，資料沒有足夠的高度／尺度資訊，這一幀不產生高度量測；不可把退化方向當成零方差的精確觀測。

令 \(h_c\) 為 MAP candidate 的高度，資料單獨支持的高度中心與量測方差為

\[
h_{\mathrm{data}}=h_c\left(1-\frac{\gamma}{I}\right),\qquad
R=\left(\frac{h_c^2}{I}+(0.05\,\mathrm{m})^2\right)
\cdot q_{\mathrm{quality}},
\]

其中 `0.05 m` 是沿用的 data-only 高度量測 error floor，\(q_{\mathrm{quality}}\) 是既有配準品質膨脹。高度量測只在 pose 已接受、且當幀通過原 scale-std 門檻後評估；再檢查原有 physical/search bounds、相同 final planes 的實際非線性 data loss 不增加，以及既有 scalar NIS gate：1 DoF 的 95% soft gate與 99.9% hard gate。保留既有 Joseph covariance 更新。另有一項不同的既有設定：原 registration objective 會把 `HEIGHT_OBSERVATION_FLOOR_M=0.05 m` 的平方加到高度 prior variance，因此即使高度 filter 的 learned std 低於 1 cm，solver 使用的高度 prior 有效 std 仍至少為 5 cm。此 prior floor 並非由新 data-only 量測方差帶入，fix2 沒有調整它。

只有 pose／scale 觀測通過既有條件後，才評估這筆高度量測。濾波器只融合一次；若高度量測被接受，輸出當幀尺度依 posterior height 與原幾何係數 \(d_s\) 設為 `posterior_height / d_s`；若未接受，沿用當前預測／記憶高度與當幀 ground relation，不再採用高度候選做第二次更新。這可避免 MAP candidate scale 與記憶化高度脫節。這是當幀因果更新，不回頭修改先前 pose。

## 已有高度與位置數據

下列是 fix1 與 v5.2 歷史結果及對已記錄資訊的唯讀分析，不是 fix2 的新輸出：

| 量 | fix1 | v5.2 | 說明 |
| --- | ---: | ---: | --- |
| 初始高度 | `1.56859234 m` | `1.56859234 m` | 兩版本相同 |
| 高度記憶範圍／全段末值 | `1.538638665–1.732843409 m`；末值 `1.613320806 m` | 末值 `1.613452190 m` | fix1 與 v5.2 末值相差很小 |
| 全段平均高度差（audit 記錄值） | `−0.000889751 m` | — | 保留原比較定義；不代表高度真值誤差 |
| sensor prior XY | 完全相同 | 完全相同 | fix1 final XY 平均仍與 v5.2 相差 `0.081512 m` |
| 已記錄高度更新的 data-only center − candidate height | 1399 次更新；P5 `−0.0555476 m`、mean `−0.0069763 m`、P95 `0.0317823 m` | — | 局部代數診斷，不產生新 pose，也不是精度分數 |

因此，這次修改針對的是高度量測重用先驗的模型問題；歷史高度相近或 XY 有差異，都不能證明 fix2 提高了定位精度。新版本尚未真實短段或 full 驗證，沒有新成績。

## 驗證與可追溯資料

`track00_test_2611021349` 的 149 項生成／CUDA regression 全部通過，耗時 `18.25 s`；log 只有既有 Matplotlib `Axes3D` warning，未影響檢查。圖表與 analysis README 已完成 QA。這些測試驗證模型與 GPU 路徑，不代表真實 bag 的高度或 XY 精度。fix2 套用收據記錄來源核對與受保護檔案狀態。

- [pytest regression XML](../outputs/run/track00_test_2611021349/pytest_regression.xml)與[log](../outputs/run/track00_test_2611021349/pytest_regression.log)
- [fix2 套用收據](../outputs/run/track00_test_2611021349/apply_receipt.json)
- [完整分析索引](../outputs/run/track00_test_2611021349/analysis/README.md)
- [目標 CSV 的 provenance 摘要](../outputs/run/track00_test_2611021349/target_reference.json)與[fix1 full immutable source manifest](../outputs/run/track02_full_2610311843/source_manifest.json)
- [時間與高度稽核摘要](../outputs/run/track00_test_2611021349/analysis/height_clock_summary.json)、[height likelihood diagnostic](../outputs/run/track00_test_2611021349/analysis/height_likelihood_diagnostic.json)、[generated summary](../outputs/run/track00_test_2611021349/analysis/generated_height_summary.json)
- [header clock 圖 PNG](../outputs/run/track00_test_2611021349/analysis/01_header_clock.png)／[PDF](../outputs/run/track00_test_2611021349/analysis/01_header_clock.pdf)
- [記錄高度與幾何 PNG](../outputs/run/track00_test_2611021349/analysis/02_recorded_height_geometry.png)／[PDF](../outputs/run/track00_test_2611021349/analysis/02_recorded_height_geometry.pdf)
- [生成高度融合 PNG](../outputs/run/track00_test_2611021349/analysis/03_generated_height_fusion.png)／[PDF](../outputs/run/track00_test_2611021349/analysis/03_generated_height_fusion.pdf)
- 新診斷欄位包含 `height_likelihood`、`height_update` 與 `height_variance_after_m2`。高度 local filter 與位姿／位置 filter 不是完整 joint estimator，未補足跨狀態 covariance。
- GNSS／IMU 外參仍採未知假設；fix1 歷史算法與分數說明見 [doc77](77_TRACK2_V53_FIX1_DIRECTIONAL_MEMORY_20261031.md)。

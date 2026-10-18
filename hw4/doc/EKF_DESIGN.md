# HW4 EKF 設計與 Discussion 準備

本文件說明已實作模型，未填入真實 bag 精度。先用 `scripts/dev.sh inspect` 取得感測器實際 covariance，再補上觀察與討論。

## 狀態與 radar 控制

平面車體狀態是 $\mathbf{x}=[x,y,\psi]^T$，單位為 m、m、rad。
相鄰 radar pose 位於同一個 global frame。先求 global 位移，再用上一個 **radar yaw** 轉到車體局部座標：

$$
\begin{bmatrix}u_x\\u_y\end{bmatrix}
=\operatorname{Rot}(-\psi^{radar}_{k-1})
\begin{bmatrix}x^{radar}_k-x^{radar}_{k-1}\\y^{radar}_k-y^{radar}_{k-1}\end{bmatrix},
\quad u_\psi=\operatorname{wrap}(\psi^{radar}_k-\psi^{radar}_{k-1}).
$$

不要用「校正後 EKF yaw」去反轉原始 radar 的 global 位移；兩個方向已可能不同。
例如車朝北，global 位移 $(0,1)$ m，反轉後局部位移為向前 $(1,0)$ m。

EKF 用自己的 yaw 旋轉局部控制，得到非線性運動模型：

$$
f(\mathbf{x},\mathbf{u})=
\begin{bmatrix}
x+\cos\psi\,u_x-\sin\psi\,u_y\\
y+\sin\psi\,u_x+\cos\psi\,u_y\\
\operatorname{wrap}(\psi+u_\psi)
\end{bmatrix}.
$$

狀態 Jacobian 與控制 Jacobian：

$$
A=\begin{bmatrix}1&0&-\sin\psi\,u_x-\cos\psi\,u_y\\0&1&\cos\psi\,u_x-\sin\psi\,u_y\\0&0&1\end{bmatrix},
\quad B=\begin{bmatrix}\cos\psi&-\sin\psi&0\\\sin\psi&\cos\psi&0\\0&0&1\end{bmatrix}.
$$

程式延用老師模板命名：$S$ 是 state covariance，$R$ 是局部 radar control covariance，$Q$ 是 GPS measurement covariance。
部分課本用相反的 Q/R 符號，閱讀時以定義為準。
預測為 $S^-=ASA^T+BRB^T$，不能將局部 covariance 未旋轉就直接加到 global state。

## GPS 觀測與更新

GPS 只量測位置，$\mathbf{z}=[x_{gps},y_{gps}]^T$：

$$
h(\mathbf{x})=C\mathbf{x},\quad C=\begin{bmatrix}1&0&0\\0&1&0\end{bmatrix}.
$$

$C$ 是 **2×3**，GPS covariance $Q$ 是 **2×2**。GPS orientation 不作 yaw 量測。
Kalman gain 與更新：

$$
K=S^-C^T(CS^-C^T+Q)^{-1},\quad
\mathbf{x}^+=\mathbf{x}^-+K(\mathbf{z}-C\mathbf{x}^-).
$$

實作以 `numpy.linalg.solve` 代替顯式 inverse，covariance 使用 Joseph form：

$$
S^+=(I-KC)S^-(I-KC)^T+KQK^T.
$$

雖然 GPS 不直接測 yaw，運動後形成的位置／yaw cross covariance 可以間接校正 yaw；靜止或不足運動時方向的可觀測性有限。
初始化 yaw 由第一個 radar quaternion 取得，即使 GPS 先到，也不直接假設 yaw=0。

## covariance 從哪裡來

ROS pose covariance 是 6×6，索引順序 `[x,y,z,roll,pitch,yaw]`，由 36 個 row-major 數值表示。
GPS 取索引 `[0,1]` 的完整子矩陣，radar 取 `[0,1,5]`，保留 off-diagonal。
對角線是 variance：位置 m²、yaw rad²；位置與 yaw 的交叉項為 m·rad。標準差是 variance 開平方，不能把標準差直接當 variance。

假如 GPS $Q=\operatorname{diag}(9,9)$，代表 x/y 標準差各 3 m；**這是教學例子，不是尚未讀取的 bag 實際數值**。
在相同 state covariance 下，GPS variance 越小，更新越信任 GPS；radar process noise 越大，預測後的不確定性增長越多。

radar 訊息給的是 absolute pose covariance，並不是相鄰位移的 covariance。
`relative_covariance()` 對局部控制分別計算兩端 pose 的 Jacobian $J_a,J_b$，以
$R\approx J_a\Sigma_aJ_a^T+J_b\Sigma_bJ_b^T$ 傳遞。
這裡假設兩端誤差獨立；累積里程計通常有時間相關，bag 沒有 cross covariance，故此近似可能高估或錯估噪聲，**不能當成精確校準**。若未來有 increment covariance 或相關模型，應取代這一近似。

只有感測器選取的 covariance **全部為零** 時使用有警告的 fallback：GPS std=3 m、radar position std=0.1 m、radar yaw std=1°（程式內轉 rad）。
這些是尚無 covariance 時可執行的起始假設，不是由資料估計、不是作業指定數值、也沒有經 GT 調參。要在報告揭露 fallback 次數，並依感測器規格或量測分析修訂。
非有限、不對稱或明顯負特徵值的 covariance 會拒絕訊息並計數；極小浮點數誤差才會數值修正。

輸出 `/pred` 的 x/y/yaw covariance 填回 6×6 對應位置；未估計的 z/roll/pitch 設大 variance，RViz 預設關閉融合結果的 3D covariance 顯示，只比較平面軌跡。

## 不同頻率與時間

GPS 暫存到有 radar bracket 後，以 header timestamp 的時間比例對 radar global 位置做線性插值、yaw 用最短角差插值，在 GPS 時刻先 predict、再 update、最後推進到 radar 時刻。
例如 radar 在 0 s 與 0.1 s、GPS 在 0.05 s：先推到 0.05 s 校正，再推到 0.1 s。使用未來一筆 radar，輸出延遲最多一個 radar interval，並非零延遲即時融合。
interval 的控制噪聲按時間比例分給各子段、轉入子段局部 frame，是 piecewise motion/noise 近似；插值不會產生額外獨立感測資訊。

timestamp 先保持整數 ns，只在相減後計算比例，避免大 epoch 浮點數精度損失。
角度差與更新後 yaw 一律透過 `atan2(sin(angle), cos(angle))` 包回 ±π。
已處理時間之前才送達的 GPS 不回溯更新，另計 late_gps；倒序／重複 radar 被拒絕。大量 late_gps 代表需再調查 header/record 時間與傳輸順序。

## 如何驗證與寫報告

1. 跑 `inspect`，記下三個 topic 的 frame、頻率、實際 covariance，回答 Discussion 第 2 題，勿使用上方假設值冒充資料。
2. 完整播放後，確認 `summary.json` 的 GPS 更新數、有效估計及 GT 筆數、拒絕與 fallback 計數。
3. 看同 frame 的 radar/GPS/estimate/GT 軌跡，討論 radar 漂移與 GPS 擾動如何互補；不預設融合必定優於每個 sensor。
4. 若要計算 RMSE，先對齊相同 timestamp／重疊區間再比較。此版本只存 CSV 與疊圖，不用不同頻率 callback 當下的最新 pose 冒充時間對齊精度。
5. GT callback 只存圖表資料，沒有任何路徑進入 EKF。合成測試驗證 Jacobian、角度、噪聲權重與時間邏輯，不等於真實定位測試。

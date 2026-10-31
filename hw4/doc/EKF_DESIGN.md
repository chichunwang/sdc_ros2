# HW4｜EKF 模型與 Discussion

本作業要把 radar odometry 的運動增量和 GPS 位置放進延伸卡爾曼濾波器（EKF）。兩者頻率不同，所以程式依訊息的 header timestamp 對齊。老師提供的 `/gt_odom` 只用來檢查估計結果，不輸入 EKF、不初始化 state，也不拿來調 covariance。

狀態是車輛平面位置與航向：

$$
\mathbf{x}=[x,y,\psi]^T,
$$

其中 $x,y$ 的單位為公尺，$\psi$ 以弧度表示。GPS 只量 $x,y$；radar odometry 提供位置與方向。

## Discussion 1：狀態、運動模型和觀測矩陣

radar 的相鄰 pose 都以共同地圖座標表示。先求兩個 global position 的差，再用上一個 **radar yaw** 旋轉回上一個 radar pose 的局部座標，得到控制量：

$$
\begin{bmatrix}u_x\\u_y\end{bmatrix}
=R(-\psi^{radar}_{k-1})
\begin{bmatrix}x^{radar}_k-x^{radar}_{k-1}\\y^{radar}_k-y^{radar}_{k-1}\end{bmatrix},
\qquad u_\psi=\operatorname{wrap}(\psi^{radar}_k-\psi^{radar}_{k-1}).
$$

使用 radar 的 heading 還原其自己的局部位移，避免把已被 GPS 校正過的 EKF heading 誤當成 radar 座標系。假如車頭朝北，地圖向前位移 $(0,1)$ m，轉回車體 frame 後控制量為 $(1,0)$ m。

EKF 再用目前自己的 heading 旋轉局部控制量，推進位置：

$$
f(\mathbf{x},\mathbf{u})=
\begin{bmatrix}
x+\cos\psi\,u_x-\sin\psi\,u_y\\
y+\sin\psi\,u_x+\cos\psi\,u_y\\
\operatorname{wrap}(\psi+u_\psi)
\end{bmatrix}.
$$

對 state 的 Jacobian $A$ 與對 radar control 的 Jacobian $B$ 為：

$$
A=\begin{bmatrix}
1&0&-\sin\psi\,u_x-\cos\psi\,u_y\\
0&1&\cos\psi\,u_x-\sin\psi\,u_y\\
0&0&1
\end{bmatrix},
\quad
B=\begin{bmatrix}
\cos\psi&-\sin\psi&0\\
\sin\psi&\cos\psi&0\\
0&0&1
\end{bmatrix}.
$$

預測 state covariance 為：

$$
S^-=AS A^T+B R B^T.
$$

這個 repo 延續課堂模板命名：$S$ 表示 state covariance，$R$ 表示 radar control covariance，$Q$ 表示 GPS measurement covariance。部分課本會把過程與量測雜訊的 $Q/R$ 對調，請依這裡的定義閱讀。

GPS 只觀測位置，量測模型及觀測矩陣為：

$$
\mathbf{z}=h(\mathbf{x})+\mathbf{v}=C\mathbf{x}+\mathbf{v},\qquad
C=\begin{bmatrix}1&0&0\\0&1&0\end{bmatrix}.
$$

因此 $C$ 是 **2×3**，GPS $Q$ 是 **2×2**；GPS orientation 不當成航向量測。EKF 以 innovation $\mathbf{z}-C\mathbf{x}^-$ 求增益，再修正 state。程式用線性方程求解而非明確計算反矩陣，並使用 Joseph covariance update：

$$
K=S^-C^T(CS^-C^T+Q)^{-1},\qquad
\mathbf{x}^+=\mathbf{x}^-+K(\mathbf{z}-C\mathbf{x}^-),
$$

$$
S^+=(I-KC)S^-(I-KC)^T+KQK^T.
$$

GPS 沒有直接觀測 yaw，但轉彎和運動期間的位置誤差與 yaw 誤差會形成交叉 covariance，GPS position update 因而可能間接修正 heading。初始 yaw 取第一筆 radar yaw；若 GPS 先到，先等待 radar，不假設車頭朝東。

### 兩個 topic 頻率不同時如何融合

GPS 先放進時間佇列。下一筆 radar 到達後，程式以 header timestamp 的整數奈秒求該 GPS 在兩筆 radar 之間的比例，對 radar 平面位置線性插值、對 yaw 沿最短角度差插值；先將 EKF 推進到 GPS 時刻並 update，再繼續推進到 radar 時刻。這最多等一個 radar interval，因此 output 有一小段對齊延遲。

GPS 到達時若其時間早於 EKF 已處理時間，程式不回頭改舊 pose，並記錄 `late_gps`。重複或倒序的 radar timestamp 也會略過並計數。分段時把該 radar interval 的 control covariance 按時間比例分配；這只是過程雜訊近似，不會因插值而創造新的量測。

## Discussion 2：GPS 與 radar covariance 的意義

ROS 的 pose covariance 是 6×6、row-major，索引順序 `[x,y,z,roll,pitch,yaw]`。位置 variance 的單位為 $m^2$、yaw variance 為 $rad^2$；位置與 yaw 的 off-diagonal 項單位為 $m\cdot rad$。標準差是 variance 平方根，不可把 covariance 的對角線直接當標準差。

本程式從 GPS 取 `[x,y]` 的完整 2×2 子矩陣；從 radar 取 `[x,y,yaw]` 的完整 3×3 子矩陣，保留 off-diagonal。GPS covariance 小，update 會較信任 GPS；radar control covariance 大，predict 後的 state 不確定性會增加。

題目要求報告實際 bag 的 covariance。2026-10-29 實測 run `run_261029_231926_g9rH` 中，GPS 原始 covariance 對角為 `(3, 3) m²`；radar 實際選用 covariance 對角統計最小值為 `[0, 0, 0]`、最大值為 `[9.380944161956033e-05, 6.072459584673981e-05, 0.0006405714870086814]`。若某感測器被選取的 covariance 全為零，程式才用 fallback：GPS 標準差 3 m、radar position 標準差 0.1 m、radar yaw 標準差 1°。這些是程式為全零 covariance 準備的工程假設，並非題目給定或實測標定值；該 run GPS fallback 0 次、radar fallback 1 次。非有限、非對稱或明顯非半正定的 covariance 會拒絕訊息，而不是靜默套用另一種雜訊。

Radar topic 的 covariance 描述各時刻的 absolute pose；由相鄰 pose 算出的局部 motion covariance 另外用兩端 pose 的 Jacobian 傳遞。bag 沒有提供相鄰兩筆 odometry 的 cross-covariance，程式的傳遞近似將兩端誤差視為獨立。累積 odometry 可能具有時間相關，所以它是具體但未校準的 process-noise 近似，不代表精確的增量誤差。

## 實驗結果怎麼寫

先前 80 秒驗證 run（`run_261029_231926_g9rH`）收到 radar 973、GPS 64、GPS update 64、estimate 1037、GT 7544；bag 原有 7714 筆 GT，best effort 接收少 170 筆。另一次錄影輪（`run_261029_232220_oYEV`）收到 radar 961、GPS 63、estimate 1024、GT 6648；bag 原有 radar 973、GPS 64、GT 7714，分別少 12、1、1066 筆。該輪 GPS update 63、late GPS、invalid message、non-increasing radar、pending GPS 均為 0。首末雷達時間跨度 79.615616084 s 與 bag 相符，但 player log 沒有明確自然播畢訊息，因此不宣稱所有訊息均收到或自然播畢。兩輪 radar／GPS／GT 的 header.frame_id 均為 `origin`，bag 含 `/clock`。GT 原始時間有 1526 筆非遞增（包含重複）；評估依整數 timestamp 去重並保留最後值，再線性插值且不外推。GT 僅用於結果評估。

驗證 run 記錄 RMSE：radar 34.30278999808334 m、GPS 0.46098894599053886 m、estimate 1.0322084968142409 m。錄影輪記錄 RMSE：radar 34.346932618594955 m、GPS 0.4634891799869087 m、estimate 1.035346041665754 m，estimate P95 為 1.665499907427702 m；GT 去重後 5340 筆，最大線性插值跨距 0.14985 s。各來源時間戳樣本不完全相同，故數字不能解讀成嚴格同樣本比較，也不能據此說 EKF 比 GPS 更準。程式會以圖和時間對齊分析留存結果；本機較早的離線合成檢查只驗證數學、時序及 ROS stub 輸出，不是 ROS 執行或實測定位精度，見 [README 驗證紀錄](../README.md#來源與驗證紀錄)。

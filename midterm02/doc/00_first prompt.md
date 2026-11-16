SDC Midterm02 — GPU-Accelerated Visual-Geometric SLAM
Codex Master Engineering Prompt V4
0. Role and Mission
你是一位資深的：

* SLAM Algorithm Engineer
* Computer Vision Engineer
* Visual-Inertial Odometry Engineer
* 3D Point Cloud Registration Engineer
* CUDA / GPU Optimization Engineer
* Robotics Software Architect
* Robotics Research Engineer

你的任務是協助我完成 SDC Midterm02 — Visual SLAM Competition 2026。
這是一個真實的自動駕駛定位演算法開發與研究專案。
我已經完成 Midterm02 的初始專案環境建立，也有先前 Midterm01 的 CPU/GPU 定位程式作為參考。
你必須在現有環境中實際：

1. 分析專案與既有程式碼。
2. 理解並重用 Midterm01 GPU 的工程架構。
3. 建立 GPU 加速的 Visual SLAM Pipeline。
4. 結合 RGB 影像特徵與 3D Point Cloud。
5. 開發及比較 ICP / GICP / NDT 演算法。
6. 實作 Visual Odometry、Keyframe、Local Mapping。
7. 實作 Loop Closure 與 Pose Graph Optimization。
8. 驗證正確性、定位精度和 GPU 效能。
9. 完成整段軌跡估計。
10. 產生可以提交 Kaggle 的結果。
11. 保留完整的實驗、可視化及技術報告資料。

不要只提出建議或產生範例程式碼。請在現有專案中進行實際開發、執行、測試及改善。
所有溝通、實驗解釋及開發報告使用繁體中文。
1. Existing Workspace — Critical
1.1 Workspace Structure
目前專案結構如下：

```
code_base/
└── sdc_ros2/
    ├── .codex/
    ├── docs/
    ├── hw2/
    ├── hw3/
    ├── hw4/
    ├── midterm01/
    ├── midterm01_gpu/
    ├── midterm02/
    │   ├── deliverables/
    │   ├── doc/
    │   ├── scripts/
    │   ├── tests/
    │   ├── vslam/
    │   ├── .gitattributes
    │   ├── .gitignore
    │   ├── AGENTS.md
    │   ├── docker-compose.yml
    │   ├── Dockerfile
    │   ├── README.md
    │   └── requirements.txt
    ├── .gitignore
    └── AGENTS.md
```

請使用實際 Repository 位置確認路徑，不要假設專案一定位於某個固定絕對路徑。
1.2 Development Directory
本次的主要工作目錄：
`code_base/sdc_ros2/midterm02`
所有本次比賽新開發的程式，原則上都必須放在 `midterm02` 裡面。
請不要建立：

* `midterm02_new`
* `midterm02_gpu`
* `visual_slam_project`
* 其他重複的專案根目錄

我已經建立好 `midterm02` 的初始環境。
你應該在目前的結構上繼續開發，而不是重新初始化專案。
1.3 Reference Projects
以下兩個資料夾是先前的專案：

* `midterm01`
* `midterm01_gpu`

其中：
`midterm01` 是先前的定位演算法專案。
`midterm01_gpu` 是先前使用 GPU 加速的版本。
這兩個資料夾必須當成參考來源。
禁止直接修改、覆蓋、刪除這兩個資料夾中的內容。
如果需要重用程式碼，請將必要模組移植或重構到 `midterm02`，保留來源說明與相應授權資訊。
1.4 Existing Instructions
開始前請閱讀：

1. `sdc_ros2/AGENTS.md`
2. `sdc_ros2/midterm02/AGENTS.md`
3. `sdc_ros2/midterm02/README.md`
4. `sdc_ros2/midterm02/Dockerfile`
5. `sdc_ros2/midterm02/docker-compose.yml`
6. `sdc_ros2/midterm02/requirements.txt`

也請查看 `.codex/` 中適用於本專案的設定。
遵守所有適用的 AGENTS.md 指令。
若本 Prompt 與現有環境或指令發生衝突，請先確認原因，避免直接破壞既有設定。
2. Study Previous Midterm01 Architecture
2.1 Critical Requirement
我不希望你完全重新發明一套新的 SLAM 工程架構。
請先深入分析：

* `midterm01`
* `midterm01_gpu`

尤其是 `midterm01_gpu`。
必須研究它的：

* Main Program
* GPU Registration Implementation
* CUDA Backend
* Point Cloud Preprocessing
* Pose Transformation
* Coordinate System
* Localization Pipeline
* Configuration
* Data Loading
* Output Generation
* Logging
* Debugging
* Visualization
* Benchmark
* Result Directory Organization
* Execution Commands

2.2 Reuse Policy
請分類每個重要模組：
A. 可以直接重用。
B. 需要修改後重用。
C. 需要重新設計。
D. 不適合 Midterm02。
優先重用成熟的 GPU Registration 與資料處理程式。
不要因為想使用新的函式庫，就放棄已經測試穩定的 GPU 演算法。
但也不能為了重用 Midterm01，而保留不符合 Visual SLAM 的設計。
2.3 Output Compatibility — Mandatory
Midterm02 的工程輸出方式必須盡可能依照 `midterm01_gpu` 的實際設計。
請詳細確認 Midterm01 GPU 如何：

* 建立每次 Run 的結果資料夾。
* 輸出 Pose。
* 輸出 Trajectory。
* 儲存逐幀 Registration Quality。
* 儲存 Logs。
* 儲存 Debug Images。
* 儲存 Point Cloud Visualization。
* 輸出 Benchmark。
* 管理實驗結果。
* 產生最終 Submission。

請保留這些實際存在的工程慣例。
不要憑空假設 `midterm01_gpu` 有某個輸出功能。
如果 Midterm01 使用特定的 CSV、TXT、JSON 或影像輸出格式，請優先保持相容。
Midterm02 額外需要的資訊可以擴充，不要不必要地破壞既有格式。
注意：Midterm02 的 Kaggle Submission 必須遵守 Midterm02 官方格式，不可直接套用 Midterm01 的提交欄位。
3. Existing Midterm02 Environment
我已經建立：

```
midterm02/
├── deliverables/
├── doc/
├── scripts/
├── tests/
├── vslam/
├── docker-compose.yml
├── Dockerfile
├── README.md
└── requirements.txt
```

這些資料夾與檔案已經存在。
請先檢查其中的實際內容及目前功能。
不要假設它們是空的，也不要在沒有閱讀的情況下覆蓋。
3.1 Directory Responsibilities
優先遵循現有架構；若目前尚未有明確職責，可以採用：
vslam/
主要演算法實作：

* Dataset Loader
* Camera Geometry
* Visual Feature Extraction
* Feature Matching
* Visual Odometry
* Point Cloud Processing
* ICP / GICP Registration
* Pose Fusion
* Keyframe
* Local Mapping
* Loop Closure
* Pose Graph
* IMU Integration
* Trajectory Processing

請根據實際程式狀況決定模組拆分。
scripts/
執行工具：

* Pipeline Runner
* GPU Benchmark
* Experiment Runner
* Feature Visualization
* Trajectory Visualization
* Submission Generator
* Submission Validator

tests/
測試：

* Unit Tests
* Coordinate Transform Tests
* Registration Tests
* Data Loader Tests
* GPU Correctness Tests
* Pipeline Integration Tests
* Submission Validation Tests

doc/
技術文件：

* Architecture
* Algorithm Design
* Experimental Methodology
* Performance Analysis
* Research Contributions
* Failure Analysis

deliverables/
最後要交出的結果：

* Submission
* Report
* Presentation Assets
* Reproduction Instructions

如果原有目錄結構已有更明確的用途，請以現有設計為準。
4. Docker and CUDA Environment
4.1 Preserve Existing Environment
`midterm02` 已經具有：

* Dockerfile
* docker-compose.yml
* requirements.txt

請先驗證，不要直接重新建立 Docker 環境。
確認：

* Docker 是否可正常執行。
* Container 是否能存取 GPU。
* NVIDIA Container Toolkit 是否正常。
* PyTorch 是否支援 CUDA。
* CUDA 版本是否相容。
* 必要 Python Packages 是否可使用。
* 是否能使用現有 GPU Registration Backend。

4.2 GPU Hardware
主要目標運算設備：
NVIDIA RTX PRO 6000 GPU
實際型號、VRAM、CUDA Compute Capability、Driver 版本請透過系統確認。
不要自行假設 GPU 的完整規格。
4.3 GPU-First Policy
本次演算法優先使用 GPU 執行。
應盡可能加速：

* Feature Extraction
* Feature Description
* Feature Matching
* Descriptor Similarity
* Point Cloud Preprocessing
* Nearest Neighbor Search
* Correspondence Search
* ICP
* GICP
* Geometric Residual Computation
* Large Matrix Operations
* Loop Closure Feature Processing

Pose Graph Optimization 可使用 GTSAM CPU Backend。
不必強制所有小型 CPU 操作都轉移至 GPU。
重點是計算量大的部分必須真正使用 CUDA。
4.4 GPU Verification
不可因為使用 Open3D、PyTorch 或其他函式庫，就直接宣稱該演算法已使用 GPU。
請確認實際運算設備。
尤其注意：

* CPU Open3D ICP 不是 GPU ICP。
* CPU small_gicp 不是 GPU GICP。
* 將 CPU 結果複製到 CUDA Tensor 不代表演算法在 GPU 上運算。

請提供可重現的 GPU 驗證方法。
4.5 GPU Memory Management
資料包含大量高解析度影像與 PLY。
請避免一次將所有資料載入 GPU。
使用：

* Frame-by-Frame Loading
* Local Submap Caching
* GPU Tensor Reuse
* Memory-efficient Data Transfer
* Keyframe Cache
* Configurable Batch Size

應優先追求正確性與定位精度，再改善速度。
5. Competition Objective and Dataset
5.1 Task
對每張 Front Camera Image 估計 4×4 Camera-to-World Pose。
記為：
T_wc = [R_wc, t_wc; 0, 1]
其中：

* R_wc：攝影機相對世界座標系的旋轉。
* t_wc：攝影機在世界座標系的位置。

必須產生所有影像的 Pose。
5.2 Track01

* 1314 Images
* 1440 × 928
* 10 Hz
* 約 247 m Route
* 同方向繞行三圈
* IMU 100 Hz

5.3 Track02

* 不提供 IMU。
* 其他資料以實際提供的 Dataset 為準。

程式架構必須支援有 IMU 和無 IMU 的情況。
5.4 Provided Data
官方包含：

```
data/
├── images/
│   └── {timestamp}.jpg
├── raw_ply/
│   └── {timestamp}.ply
├── camera_info.json
├── imu.csv
└── sample_submission.csv
```

實際資料夾位置請自行搜尋及驗證。
檔案名稱為 Nanosecond Timestamp。
5.5 LoGeR Point Clouds
PLY 不是 LiDAR 真實量測。
PLY 是 LoGeR 預測的 3D Point Cloud。
包含：

* X
* Y
* Z
* RGB

座標採用 OpenCV Camera Frame：

* X Right
* Y Down
* Z Forward

重要限制：

* 尺度不完全精準。
* 不同幀可能有 Scale Variation。
* 點雲包含天空。
* 可能存在 Depth Outliers。
* 可能包含 Dynamic Objects。
* 不一定能直接建立 Pixel-to-Point Correspondence。

所有幾何演算法需要考慮這些限制。
6. Main Research Direction
本專案主要研究：
GPU-Accelerated Image-Guided Visual-Geometric SLAM
我希望將影像特徵與點雲配準真正結合。
不能只是：
PLY → ICP → Trajectory。
必須研究：
RGB Image
→ Feature Detection
→ Feature Matching
→ Visual Pose Estimation
→ Visual Pose Prior
→ GPU ICP / GICP Refinement
→ Camera Pose。
完整系統：

1. Visual Frontend
2. Geometric Frontend
3. Pose Estimation and Fusion
4. Keyframe Management
5. Local Mapping
6. Loop Closure
7. Pose Graph Optimization
8. Full Trajectory Export

7. GPU Visual Feature Extraction
7.1 Feature Detection
我特別希望使用照片資料。
請實際從 RGB Images 中提取特徵。
優先評估：
Baseline

* ORB
* SIFT

GPU Advanced

* SuperPoint
* LightGlue

也可以研究其他合規的公開預訓練模型。
先確認實際 GPU 支援，再決定採用哪種實作。
7.2 Feature Matching
針對相鄰影像或 Keyframe：

1. Extract Keypoints。
2. Extract Descriptors。
3. Match Features。
4. Reject Outliers。
5. Geometric Verification。
6. Estimate Relative Motion。

使用 RANSAC 或合適的 Robust Estimation。
記錄：

* Number of Keypoints
* Number of Matches
* Number of Inliers
* Inlier Ratio
* Geometric Error
* Processing Time

7.3 Visual Pose Estimation
可研究：

* Essential Matrix + recoverPose
* PnP + RANSAC
* 2D-3D Correspondence
* 3D-3D Alignment

注意：
Essential Matrix 的 Translation 沒有公尺尺度。
不能直接當成可靠的 Absolute Translation。
應研究利用 LoGeR 提供的 3D 資訊獲得相對尺度。
7.4 Image-to-Point Cloud Association
這是非常重要的研究項目。
請確認影像 Pixel 能否對應到 LoGeR PLY 的 3D Point。
不要假設：
PLY Point Index = Image Pixel Index。
請檢查：

* PLY Data Structure
* Camera Projection
* Camera Intrinsics
* RGB Consistency
* Point Visibility
* Depth Consistency

如果能建立可信的對應，使用匹配的影像特徵取得 3D Correspondences。
若無法可靠對應，改用有驗證的 2D 幾何方法，不得製造虛假的 2D-3D 對應。
7.5 Visualization
必須輸出真實比賽影像的：

* Feature Keypoints
* Feature Matches
* RANSAC Inliers
* RANSAC Outliers
* Tracking Visualization

請遵循 Midterm01 GPU 原有輸出習慣。
這項功能必須實際可執行，不只是建立 Visualization Function。
8. Point Cloud Preprocessing
請優先重用 `midterm01_gpu` 的 Point Cloud Processing 模組。
研究：

* Invalid Point Removal
* Voxel Downsampling
* Range Filtering
* Statistical Outlier Removal
* Radius Outlier Removal
* Normal Estimation
* Sky Filtering
* Dynamic Object Filtering

需要特別考慮 LoGeR 預測深度的不可靠區域。
若有可靠的視覺語義資訊，可以研究使用 Image Mask 幫助去除天空或動態物體。
所有 Preprocessing Steps 必須可以透過 Config 啟用或關閉。
測試時必須保留前後點雲可視化。
9. Registration Algorithms — Accuracy First
9.1 Priority
我先前在 Midterm01 的經驗是：
ICP 最終精度通常比 NDT 好。
但這不表示 NDT 被禁止。
請根據這份資料集實際比較：

* Point-to-Plane ICP
* Generalized ICP
* Multi-scale ICP
* NDT Coarse Registration
* NDT → ICP
* NDT → GICP

優先採用 ICP / GICP 作為 Fine Registration。
NDT 如果有幫助，可以用來產生更好的 Initial Guess。
9.2 Candidate Pipelines
至少評估：
A. Motion Prior → GPU ICP
B. Motion Prior → GPU GICP
C. Visual Prior → GPU ICP
D. Visual Prior → GPU GICP
E. Visual Prior → Multi-scale GPU ICP / GICP
F. Visual Prior → NDT → GPU ICP / GICP
F 可以作為條件式方案。
不要求每幀都執行 NDT。
9.3 Adaptive Registration
請研究根據當前品質選擇方法。
例如：
Visual Matching 可靠：
→ Visual Initial Guess
→ GPU GICP / ICP。
Visual Matching 不可靠：
→ Motion Prediction
→ GPU GICP / ICP。
ICP 無法可靠收斂：
→ 嘗試 NDT Coarse Alignment
→ 再使用 ICP / GICP。
所有方法失敗：
→ 保留合理運動預測。
→ 標記 Tracking Failure。
→ 避免 Pose Jump。
→ 透過後續觀測嘗試恢復追蹤。
9.4 Quality Gate
不能只使用 ICP Fitness。
應考慮：

* Registration Fitness
* RMSE
* Correspondence Count
* Translation Jump
* Rotation Jump
* Visual Inlier Ratio
* Visual-Geometric Consistency
* Temporal Motion Consistency
* Degeneracy Indicators

使用可靠的方式拒絕異常配準。
10. Visual-Geometric Fusion
這是我希望主要投入研究的方向。
10.1 Primary Method
優先實作：
Visual Feature Guided GPU GICP / ICP
流程：

1. RGB Image Feature Extraction。
2. Feature Matching。
3. Geometric Verification。
4. Visual Relative Pose Estimation。
5. Initial Guess Generation。
6. GPU GICP / ICP。
7. Quality Evaluation。
8. Pose Update。

視覺估計必須真正用於配準。
不能只輸出 Feature Matching Image，卻仍完全依賴舊有 Motion Prior。
10.2 Confidence-aware Fusion
可以研究結合：

* Visual Confidence
* Point Cloud Registration Confidence
* Temporal Motion Confidence

決定是否接受新的 Pose。
不能直接平均 Rotation Matrix。
如需融合，可使用：

* SE(3) Lie Algebra
* Robust Optimization
* Factor Graph
* 合理的不確定度模型

請優先選擇有數學依據、可實作且可驗證的方案。
10.3 Temporal Consistency
特別避免：

* Registration Failure 清空上一幀補償量。
* 單幀錯誤配準造成 Pose 大幅跳動。
* 錯誤 Initial Guess 導致軌跡突然轉向。
* 拒絕量測時直接回到不連續的舊 Pose。

請建立持續的 Pose / Motion State。
11. Scale Consistency
因為 LoGeR Point Cloud 可能存在 Scale Variation：
請分析：

* Frame-to-Frame Scale
* Relative Scale Changes
* Local Scale Drift
* Scale Outliers
* Multi-frame Consistency

可研究：

* Sim(3) Registration
* Robust Relative Scale Estimation
* Temporal Scale Filtering
* Scale Consistency Constraints

重要：
Kaggle 使用全域 Sim(3) Alignment，因此整體尺度可以被修正。
真正重要的是避免局部尺度不一致造成路線形狀錯誤。
不需要為了修正一個固定尺度而增加不必要的演算法複雜度。
12. Keyframe and Local Mapping
Track01 會重複行駛同一條路線。
請實作：

* Keyframe Selection
* Local Submap
* Sliding Window
* Local Registration
* Keyframe Pose Storage
* Image Feature Storage
* Local Map Update

Keyframe 可依據：

* Translation
* Rotation
* Feature Overlap
* Registration Confidence
* Tracking Quality

進行選擇。
請避免無限制增加 Local Map 的點數。
所有非 Keyframe 仍必須保留 Pose。
13. Loop Closure
Track01 行駛三圈，Loop Closure 是重要功能。
13.1 Visual Place Recognition
優先使用 RGB Image 進行重訪地點搜尋。
可以評估：

* ORB-based Retrieval
* Bag of Words
* Global Image Descriptors
* GPU Learned Place Recognition

13.2 Loop Verification
每個 Loop Candidate 都需要進行：

* Image Feature Matching
* Geometric Verification
* Relative Pose Estimation
* Point Cloud Registration
* Transform Consistency Check

禁止只因為兩張照片看起來相似就建立 Loop Edge。
13.3 Loop Constraints
只有通過驗證的 Loop Closure 才能加入 Pose Graph。
並且記錄：

* Candidate Frame IDs
* Loop Score
* Match Count
* Geometric Residual
* Accepted / Rejected
* Relative Transformation

14. Pose Graph Optimization
優先使用官方 Starter Code 與可重用的既有程式。
可以使用 GTSAM。
主要功能：

* Odometry Factors
* Loop Closure Factors
* Robust Loss
* Graph Optimization
* Trajectory Correction

必須正確處理 Pose Gauge Freedom。
若有尺度漂移，請評估是否需要 Sim(3) Optimization。
不要直接把不同尺度假設的 Relative Pose 混入同一個 SE(3) Graph 而不處理一致性。
最佳化後必須：

1. 更新 Keyframe Pose。
2. 修正非 Keyframe Pose。
3. 確認完整軌跡。
4. 輸出最佳化前後比較。
5. 驗證 Rotation Matrix 合法性。

15. IMU Integration
Track01 有 100 Hz IMU。
但是優先完成純 Visual-Geometric SLAM。
之後再研究 IMU。
可評估：

* Gyroscope Rotation Prior
* IMU Motion Prediction
* IMU Preintegration
* Visual-Inertial Optimization

必須先檢查：

* Timestamp
* R_cam_imu
* Bias
* Gravity
* Sensor Time Alignment

官方 IMU Timestamp 是 Bag Receive Time。
不得假設與 Camera 完美同步。
Track02 必須完全支援無 IMU 的執行模式。
16. Output and Logging — Follow midterm01_gpu
這項要求非常重要。
16.1 Output Style
請以 `midterm01_gpu` 的實際輸出架構為主。
確認其：

* Results Directory
* File Naming
* Run Naming
* Pose Export
* Trajectory Export
* Logging
* Visualization
* Experiment Recording

除非有明確理由，否則不要重新建立完全不同的輸出系統。
若需要新增 Midterm02 欄位，應盡量以向後相容方式擴充。
16.2 Additional Visual SLAM Logs
請額外記錄：

* Timestamp
* Frame ID
* Visual Match Count
* Visual Inlier Ratio
* Visual Pose Confidence
* ICP / GICP Fitness
* Registration RMSE
* Registration Method
* Tracking Status
* Keyframe Status
* Loop Closure Status
* Estimated Scale
* GPU Registration Time
* GPU Feature Matching Time
* Total Frame Processing Time

16.3 Visualization
必須支援：

* RGB Image
* Feature Keypoints
* Feature Matching
* Current Point Cloud
* Registered Point Cloud
* Local Submap
* Camera Trajectory
* Loop Closure
* Optimization Before / After

優先保留 Midterm01 GPU 已有的可視化功能和輸出方式。
程式必須支援 Headless Execution。
即使使用 SSH 或沒有圖形桌面，也能將結果儲存為圖片、影片或其他可檢查格式。
17. Kaggle Submission
官方 CSV 格式：

```
timestamp,m00,m01,m02,m03,m10,m11,m12,m13,m20,m21,m22,m23,m30,m31,m32,m33
```

每張影像一列。
Pose 必須是 Camera-to-World。
要求：

* Timestamp 與影像檔名一致。
* 每張影像都有一個 Pose。
* 不可缺少 Frame。
* 不可重複。
* 不可有 NaN / Inf。
* Rotation Matrix 必須合法。
* 最後一列固定為 [0,0,0,1]。
* 按照 Row-major 順序展開。

請建立 Submission Generator 和 Validator。
Track01 應根據實際資料確認完整的 1314 個 Pose。
18. Evaluation Strategy
官方使用 Sim(3)-aligned ATE。
因此：

* 全域 Translation 可以對齊。
* 全域 Rotation 可以對齊。
* 整體 Scale 可以對齊。

但以下問題依然會扣分：

* Drift
* Local Scale Distortion
* Wrong Turn Angle
* Tracking Failure
* Incorrect Loop Closure
* 不正確的 Trajectory Shape

Public Leaderboard 只包含部分軌跡。
不能只針對 Public Score 最佳化。
若沒有 Ground Truth，不可宣稱已計算真正的 ATE。
請使用其他診斷指標：

* Registration RMSE
* Visual Reprojection Error
* Motion Continuity
* Loop Closure Residual
* Scale Consistency
* Map Consistency
* Tracking Failure Rate

最後以 Kaggle 實際成績輔助選擇最終方案。
19. Experiment and Benchmark
請參考 `midterm01_gpu` 的原有實驗管理方式。
設計可重現的實驗。
至少比較：
E0：Official Starter Baseline
E1：Reused Midterm01 GPU Registration
E2：GPU ICP
E3：GPU GICP
E4：Visual Odometry
E5：Visual-Guided GPU ICP
E6：Visual-Guided GPU GICP
E7：Optional NDT → ICP / GICP
E8：Visual Loop Closure
E9：Pose Graph Optimization
E10：Scale Consistency
E11：IMU Fusion
請按照功能成熟度與時間安排執行，不需要機械式完成所有方法。
每次 Experiment 保留：

* Config
* Git Commit Hash
* Code Version
* Runtime
* GPU Usage
* Registration Metrics
* Trajectory Visualization
* Submission
* Kaggle Score（如果有）

請建立 Before / After Comparison。
不要在沒有實驗數據時聲稱方法提高了精度。
20. Testing Requirements
請在 `midterm02/tests/` 中建立必要測試。
Coordinate Tests

* Transform Composition
* Transform Inverse
* Camera-to-World Convention
* Source-to-Target Convention
* Rotation Matrix Validity

Data Tests

* Image / PLY Timestamp Matching
* Camera Intrinsics Parsing
* Missing Frame Handling
* Invalid Point Handling

Registration Tests

* Synthetic Point Cloud Registration
* Known Transformation Recovery
* GPU / CPU Result Comparison
* Registration Failure Detection

Visual Tests

* Feature Extraction
* Feature Matching
* RANSAC Verification
* Feature-to-3D Association Validation

Integration Tests

* Small Frame Sequence
* Pose Accumulation
* Keyframe Mapping
* Loop Closure Constraints
* Submission Generation

所有測試應盡量能夠獨立執行。
21. Git and Repository Safety
請遵守以下原則：

1. 不修改 `midterm01`。
2. 不修改 `midterm01_gpu`。
3. 不刪除既有資料集。
4. 不覆蓋使用者尚未提交的變更。
5. 不使用破壞性 Git 指令。
6. 不擅自 Force Push。
7. 保留現有 Git History。
8. 盡量在 `midterm02` 內完成改動。

可以使用 Git Branch 或 Git Worktree 管理不同版本。
但請先確認目前 Repository 狀態與既有工作流程。
重要改動前後應檢查 Git Diff。
不得為了整理環境而刪除未知用途的檔案。
22. Research Contributions
本次比賽需要自己的技術貢獻。
請優先研究：
A. Image-Guided GPU Registration
使用視覺特徵改善 GPU ICP / GICP Initial Guess。
B. Adaptive Registration
根據視覺與幾何品質，動態選擇 Registration 策略。
C. Temporal Pose Stability
降低單幀異常配準對整條軌跡的影響。
D. LoGeR Scale Consistency
處理模型預測點雲可能產生的局部尺度漂移。
E. Visual-Geometric Loop Closure
使用影像找出重訪地點，再使用點雲驗證。
每個研究項目需要：

* Problem Statement
* Proposed Method
* Mathematical Explanation
* Implementation
* Experiment
* Before / After Comparison
* Limitations

若效果不好，需據實記錄並分析原因。
不能將單純調整參數當成主要演算法貢獻。
23. Development Phases
請遵守以下階段，避免一次大幅重寫導致無法除錯。
Phase 0 — Project Inspection
閱讀：

* Root AGENTS.md
* Midterm02 AGENTS.md
* Midterm02 README
* Dockerfile
* Docker Compose
* Requirements
* Midterm01 Source Code
* Midterm01 GPU Source Code
* Midterm02 Existing Code
* Official Starter Code（若已提供）
* Dataset（若已提供）

確認：

* GPU Environment
* Existing Architecture
* Reusable Modules
* Output Convention
* Current Implementation Status

提出整合策略。
Phase 1 — Basic GPU Pipeline
在既有 `midterm02` 中完成：

* Dataset Loading
* GPU Point Cloud Registration
* Pose Accumulation
* Basic Trajectory Output
* Submission Export
* Visualization

先測試少量 Frame，再測完整 Track01。
Phase 2 — GPU Visual Frontend
實作：

* GPU Feature Extraction
* GPU Feature Matching
* Geometric Verification
* Visual Pose Estimation
* Feature Visualization

使用真實 Midterm02 照片驗證。
Phase 3 — Visual-Guided GPU Registration
將 Visual Pose Prior 整合至：

* GPU ICP
* GPU GICP

比較兩者。
必要時測試：
NDT → ICP / GICP。
Phase 4 — Local SLAM
實作：

* Keyframe
* Local Submap
* Tracking Quality Gate
* Temporal Pose Consistency

Phase 5 — Loop Closure
實作：

* Visual Place Recognition
* Loop Candidate Verification
* Loop Pose Estimation
* Loop Edge Generation

Phase 6 — Global Optimization
實作：

* Pose Graph
* Loop Constraints
* Robust Optimization
* Full Trajectory Correction

Phase 7 — Advanced Improvements
根據實驗評估是否加入：

* Scale Correction
* IMU Fusion
* Adaptive NDT
* Advanced Visual Matching
* Registration Uncertainty Modeling

Phase 8 — Competition Submission
執行完整資料集。
產生：

* Final Trajectory
* Final Submission CSV
* Experiment Logs
* Visualizations
* Reproduction Instructions

確認能符合官方規範。
24. Priority and Time Management
開發優先級：
P0 — 可執行的完整 Track01 Pipeline。
P1 — 正確的 GPU ICP / GICP。
P2 — RGB Image Feature Matching 與 Pose Estimation。
P3 — Visual-Guided GPU Registration。
P4 — Loop Closure 與 Pose Graph。
P5 — Scale / IMU / Adaptive Registration 優化。
不能因為某個高階演算法尚未完成，就讓整個系統無法輸出 Submission。
應始終保留一個可以正常執行的版本。
如果新功能效果不佳，應能使用 Config 關閉並回到上一個穩定方案。
25. Deliverables
最終必須在既有 `midterm02` 專案中完成：

1. Runnable GPU Visual SLAM。
2. Reused Midterm01 GPU Architecture。
3. GPU Visual Feature Extraction。
4. GPU Feature Matching。
5. GPU ICP / GICP。
6. Visual-Guided Registration。
7. Adaptive Quality Gate。
8. Keyframe / Local Mapping。
9. Visual Loop Closure。
10. Pose Graph Optimization。
11. Full Camera Trajectory。
12. Kaggle Submission。
13. Submission Validator。
14. Registration Logs。
15. Image Feature Visualizations。
16. Point Cloud Visualizations。
17. Trajectory Plots。
18. GPU Benchmarks。
19. Ablation Experiments。
20. Reproducible README。
21. Research Contribution Documentation。

輸出應遵循 `midterm01_gpu` 的工程習慣，同時符合 `midterm02` 既有資料夾規劃。
26. Execution Rules
請使用以下工作循環：
Inspect → Understand → Implement → Test → Analyze → Improve
對每個重要修改：

1. 閱讀相關原始碼。
2. 確認程式目前的功能。
3. 分析現有方法的問題。
4. 說明新的演算法設計。
5. 實際修改程式。
6. 執行測試。
7. 記錄結果。
8. 驗證沒有破壞既有功能。
9. 更新文件與實驗紀錄。
10. 再處理下一個問題。

若某項方法造成退步，請分析並回報，不應強行保留。
不要為了使用 GPU 而犧牲演算法正確性。
不要使用未授權的外部地圖、GPS、歷年資料或 Ground Truth。
不要手動修改 Pose 以提高比賽分數。
不要宣稱沒有測試過的功能已經成功。
27. Start Working Now
請立即開始 Phase 0。
目前真正的工作目錄是：
`code_base/sdc_ros2/midterm02`
參考目錄：
`code_base/sdc_ros2/midterm01`
`code_base/sdc_ros2/midterm01_gpu`
請依序進行：
Step 1 — Read Project Instructions
閱讀適用的 AGENTS.md 與既有 README。
Step 2 — Inspect Midterm02
確認 Docker、CUDA、Dependencies、現有 `vslam/` 程式和執行流程。
Step 3 — Analyze Midterm01 GPU
完整了解原有 GPU Registration、資料流、Pose Output、Logging 和 Visualization。
Step 4 — Compare Architectures
建立 Midterm01 GPU 與 Midterm02 的整合對照。
Step 5 — Reuse Existing Modules
在不破壞參考專案的前提下，移植適合 Midterm02 的 GPU 模組。
Step 6 — Establish Runnable Baseline
優先讓目前的 Midterm02 程式能執行並產生合法的 Trajectory 和 Submission。
Step 7 — Implement GPU Visual Frontend
使用比賽的真實 RGB Images 進行 Feature Extraction 與 Feature Matching。
Step 8 — Integrate Visual-Guided Registration
讓照片特徵真的參與 GPU ICP / GICP Pose Estimation。
Step 9 — Continue SLAM Development
逐步加入 Keyframe、Loop Closure、Pose Graph Optimization 與實驗優化。
First Progress Report
完成初步檢查後，請用繁體中文清楚報告：
A. `midterm02` 目前已有哪些功能？
B. `midterm01_gpu` 使用了哪些 GPU 演算法？
C. `midterm01_gpu` 的完整輸出格式與方式為何？
D. 哪些模組能夠直接重用？
E. 現有 Docker / CUDA 環境是否正常？
F. 目前有哪些相依套件或資料尚未準備好？
G. 如何將 RGB Feature Matching 整合到 GPU ICP / GICP？
H. 你選擇的第一個可執行實作方案是什麼？
I. 目前實際完成了哪些修改及測試？
請附上重要檔案路徑、測試指令和結果。
不要只停留在規劃。完成必要的初步分析後，請直接開始 Phase 1 的實作與測試。
最終目標：在現有 midterm02 環境中，開發一套延續 midterm01_gpu 工程架構、真正利用 GPU 與 RGB 影像特徵、具有完整 SLAM 功能，而且能產生高品質 Kaggle Submission 的系統。
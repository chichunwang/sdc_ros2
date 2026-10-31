"""Track2 v5：逐幀 covariance、point-to-plane 與可觀測方向的因果融合。

直接改此檔。GNSS noise 延續原模型，正式配準改用逐幀 covariance 與
可觀測平面量測。GNSS 空 frame、IMU/PLY 外參及 lever 仍未獨立標定。
"""
VERSION = 'track2_causal_dual_heading_bias_v5_2_perf1'
BASE_COMMIT = '5839746ce1e4fe59ed4980ae07cb5ceb3f27ddeb'
DATA_DIR = '/workspace/midterm01/data/Track2'
EXPECTED_FRAMES = 3299
PLY_SCALE = 3.0  # 只在缺可靠地面時的未驗證 fallback，不再固定每幀尺度
# 以下舊配準參數供 guided.py / 原 C++ 數值對照保留；正式 runner 使用下方 METRIC_*。
MAP_LEAF_M = 1.0
INPUT_VOXEL_LEAF_M = 0.3
SUBMAP_HALF_EXTENT_M = (100.0, 100.0, 30.0)
SENSOR_MAX_XY_CORRECTION_M = 1.5
SENSOR_MAX_YAW_CORRECTION_DEG = 8.0
SENSOR_SMOOTHING_ALPHA = 0.25  # forward correction low-pass，沒有回改 pose
GEOMETRY_NORMAL_GAIN = 2.0
GEOMETRY_TANGENT_WEIGHT = 0.05
GEOMETRY_RADIUS_M = 2.5
GEOMETRY_NEIGHBORS = 16
GEOMETRY_MIN_NEIGHBORS = 6
GEOMETRY_MIN_PLANARITY = 0.2
GNSS_STD_M = 0.3
GNSS_XY_ACCEL_PSD = 4.0
HEADING_BIAS_RW_DEG_SQRT_S = 0.12
# 舊固定時間／車速門檻只供歷史 guided 對照；正式 prior 使用下方可觀測性條件。
HEADING_COURSE_WINDOW_S = 2.0
HEADING_MIN_SPEED_MPS = 5.0
HEADING_INITIAL_BIAS_STD_DEG = 10.0
# GNSS PoseStamped 的 quaternion 也是有偏差的感測器，並非航向真值。
# 沿用相同 unknown heading-bias 預算；未由外部 CSV 或榜分校準。
GNSS_HEADING_ENABLED = True
GNSS_HEADING_BIAS_STD_DEG = HEADING_INITIAL_BIAS_STD_DEG
GNSS_HEADING_BIAS_RW_DEG_SQRT_S = HEADING_BIAS_RW_DEG_SQRT_S
GNSS_HEADING_OBSERVATION_STD_DEG = 3.0
GNSS_HEADING_MAX_PAIR_DIFFERENCE_DEG = 30.0
HEADING_MIN_WINDOW_S = 1.0
HEADING_MAX_WINDOW_S = 8.0
HEADING_MIN_SAMPLES = 5
HEADING_MAX_COURSE_STD_DEG = 3.0  # 方向量測需優於 3°；不是低速時強行算航向
HEADING_MAX_TURN_RATE_DEGPS = 2.0
HEADING_MAX_SPAN_DEG = 4.0
HEADING_STD_FLOOR_DEG = 0.5
HEADING_MAX_OBSERVATION_DEG = 15.0
HEADING_MAX_BIAS_DEG = 10.0
HEADING_MAX_BIAS_RATE_DEGPS = 0.2
HEADING_ROBUST_GATE_SIGMA = 3.0
GNSS_EXPECTED_FRAME = ''
IMU_MAX_AGE_S = 0.1
# ===== 未標定的 frame／外參誤差預算（不是由外部 CSV 擬合） =====
# GNSS position 視為 antenna；lever 是 antenna→PLY 原點，表達於 IMU body。
# identity / zero 仍是假設。若有獨立標定，直接改均值與標準差。
IMU_TO_PLY_RPY_DEG = (0.0, 0.0, 0.0)
GNSS_TO_PLY_LEVER_BODY_M = (0.0, 0.0, 0.0)
LEVER_ARM_STD_BODY_M = (0.5, 0.5, 0.5)  # 未知 lever 的工程預算，非量測精度
EXTRINSIC_ROTATION_STD_DEG = (1.0, 1.0, 0.5)
IMU_ORIENTATION_UNKNOWN_STD_DEG = (2.0, 2.0, 1.0)  # bag 全零 covariance 代表未知
IMU_GYRO_UNKNOWN_STD_DEGPS = (0.5, 0.5, 0.5)

# ===== v3：尺度與傾斜改由當幀／過去幾何估計 =====
# PLY_SCALE 只作缺少可靠地面前的未驗證 fallback，不再固定每幀尺度。
METRIC_MAP_LEAF_M = 0.25   # 細於 0.75 m final correspondence；原 1 m map 太疏
METRIC_SUBMAP_HALF_EXTENT_M = (60., 60., 20.)
GROUND_RADIUS_M = 18.
GROUND_GRID_M = 0.6
MAP_GROUND_THRESHOLD_M = 0.15
SOURCE_GROUND_THRESHOLD_RAW = 0.015
MAX_MAP_GROUND_MAD_M = 0.08
MAX_SOURCE_GROUND_MAD_M = 0.10
MIN_SOURCE_GROUND_POINTS = 100
CAMERA_HEIGHT_BOUNDS_M = (0.5, 3.0)
CAMERA_HEIGHT_STD_M = 0.15  # 初始 pose 高度雜訊工程先驗；不是精確標定
CAMERA_HEIGHT_RW_M_SQRT_S = 0.01
HEIGHT_OBSERVATION_FLOOR_M = 0.05
MIN_SOURCE_FEATURE_HEIGHT_RAW = 0.02
MIN_MAP_FEATURE_HEIGHT_M = 0.10
SOURCE_FEATURE_LEAF_RAW = 0.035
FEATURE_MAX_RANGE_M = 40.
MAX_METRIC_SOURCE_POINTS = 6000
ANGULAR_CELL_RAD = 0.07   # ~4° angular cells；每格聚合深度誤差，避免每像素獨立假設
FEATURE_STD_M = 0.3
METRIC_POSITION_STD_M = 1.0  # 僅 legacy covariance 工具／手工測試的單位，不是正式 prior 權重
METRIC_YAW_STD_DEG = 3.0
METRIC_MAX_XY_CORRECTION_M = 3.0  # 數值搜尋保護界；正式 objective 使用當幀完整 covariance
METRIC_MAX_YAW_CORRECTION_DEG = 8.0  # 已收斂時的最低搜尋幅度
METRIC_SEARCH_YAW_SIGMAS = 2.0
METRIC_SEARCH_YAW_HARD_DEG = 20.0  # 初始 10° σ 允許 ±2σ，仍有限數值界
METRIC_MAX_HEIGHT_CHANGE_M = 0.6
SCALE_BOUNDS = (0.2, 30.)
METRIC_STAGES = ((3.0, 1.0, 40), (1.5, 0.5, 40), (0.75, 0.3, 40))
MIN_METRIC_MATCHES = 30
MIN_METRIC_COVERAGE = 0.6
MAX_FEATURE_MEDIAN_M = 0.5  # 必須低於 final correspondence，避免「全不匹配」被接受
MAX_SCALE_LOG_STD = 0.15   # 只在非地面 scale 資訊足夠時更新過去高度 anchor

# ===== v5：可靠的 LiDAR 局部平面；只有法向殘差，不添加虛構切向資訊 =====
PLANE_RADIUS_M = 0.8
PLANE_NEIGHBORS = 16
PLANE_MIN_NEIGHBORS = 8
PLANE_MIN_SECOND_EIGEN_M2 = METRIC_MAP_LEAF_M**2/4
PLANE_MAX_THICKNESS_RATIO = 0.10
PLANE_MIN_PLANARITY = 0.20
PLANE_MIN_SUPPORT_PER_VARIABLE = 2.0  # 每個可觀測 pose 模式＋scale nuisance，至少兩個 angular 等效支持；不按全影像平面比例否決
MAX_PLANE_MEDIAN_M = FEATURE_STD_M
MAP_MIN_RELATIVE_INFORMATION = 1.0  # whiten 後 data 資訊至少等於當幀 sensor prior
MAP_STARTUP_YAW_SEED_SIGMAS = (-1.0, 0.0, 1.0)  # 僅首個有效 map update 前，以當幀 yaw σ 播種

# ===== v4：地圖 XY/yaw 修正的前向記憶（不濾 raw scale、不回寫舊 pose） =====
# 以下是模型誤差的工程預算，未經真值校準；保持上方 GNSS prior 權重不變。
CORRECTION_MEAS_POSITION_FLOOR_M = 0.15  # map voxel/重建誤差不能由點數無限縮小
CORRECTION_MEAS_YAW_FLOOR_DEG = 0.5
CORRECTION_POSITION_RW_M_SQRT_S = 0.10  # 偏差隨時間／行駛距離緩變的不確定度
CORRECTION_POSITION_RW_M_SQRT_M = 0.03
CORRECTION_YAW_RW_DEG_SQRT_S = 0.20
CORRECTION_YAW_RW_DEG_SQRT_M = 0.05
CORRECTION_TURN_LEVER_M = 1.5  # 未標定 lever 對轉向的誤差預算，非標定值
CORRECTION_HOLD_SECONDS = 0.5  # 短暫無有效配準：保留修正均值，variance 繼續增加
CORRECTION_DECAY_SECONDS = 3.0  # 超過 hold 後 exp(-失敗時間/tau) 逐漸退回 sensor prior
# 檢查的是「修正量的差」，不是限制車輛一幀能行駛多遠。
CORRECTION_POSITION_GATE_BASE_M = 0.35
CORRECTION_POSITION_RATE_MPS = 0.5  # 同時限制每幀實際 correction 更新，避免重獲配準時延後突跳
CORRECTION_SPEED_FRACTION = 0.10  # 容許未建模誤差為行駛距離的一部分
CORRECTION_YAW_GATE_BASE_DEG = 0.75
CORRECTION_YAW_RATE_DEGPS = 0.5
CORRECTION_TURN_FRACTION = 0.10
CORRECTION_MOTION_REJECT_MULTIPLIER = 3.0  # 超出 motion allowance 先降權；大於3倍才硬拒絕
CORRECTION_NIS_SOFT = 7.815  # 3 維 Gaussian 的 95% 參考界；模型未校準
CORRECTION_NIS_REJECT = 16.266  # 99.9% 參考界，與物理 innovation gate 一起使用
# 部分可觀測時的 1／2／3 自由度；95% soft、99.9% hard，非校準後的可靠度保證。
CORRECTION_NIS_SOFT_BY_DOF = (3.841, 5.991, 7.815)
CORRECTION_NIS_REJECT_BY_DOF = (10.828, 13.816, 16.266)

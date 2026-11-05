"""Track2 v5：逐幀 covariance、point-to-plane 與可觀測方向的因果融合。

直接改此檔。GNSS noise 延續原模型，正式配準改用逐幀 covariance 與
可觀測平面量測。GNSS 空 frame、IMU/PLY 外參及 lever 仍未獨立標定。
"""
VERSION = 'track2_v510_turn_patch_v5_16_4_perf2'
# 正式入口固定採 v5.2 融合，只移植方向性記憶。另一值僅供歷史機制 regression。
FUSION_PROFILE = 'v52_directional_memory'

def uses_v52_fusion():
    if FUSION_PROFILE not in ('v52_directional_memory', 'v55_experimental'):
        raise ValueError(f'Unknown Track2 fusion profile: {FUSION_PROFILE}')
    return FUSION_PROFILE == 'v52_directional_memory'

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
HEADING_MAX_BIAS_RATE_DEGPS = 0.2  # legacy course-window rate; v5.5 uses a per-frame information budget
HEADING_MAX_UPDATE_DEG = 0.5  # 發布後每幀 calibration 總額不超過 map yaw model floor；非 gyro 車身轉速限制
HEADING_ROBUST_GATE_SIGMA = 3.0
GNSS_EXPECTED_FRAME = ''
IMU_MAX_AGE_S = 0.1
# ===== v5.7：路面影像提供區間平均速度；不當絕對位置或航向 =====
IMAGE_GROUND_SPEED_ENABLED = True
IMAGE_SPEED_INTERVAL_S = 0.25  # 不重疊區間，約 GNSS 更新週期；量測只消費一次
IMAGE_SPEED_MAX_INTERVAL_S = 0.6
IMAGE_SPEED_DOWNSAMPLE = 2
IMAGE_SPEED_MAX_FEATURES = 800
IMAGE_SPEED_GROUND_FEATURE_SPACING_PX = 4  # 路面小角點；大於兩倍1.5px reprojection 預算
IMAGE_SPEED_MIN_TRACKS = 40
IMAGE_SPEED_MIN_GRID_CELLS = 8  # 8x6 格至少八格；防止只追同一物體
IMAGE_SPEED_FB_PX = 1.0  # 下採樣影像像素
IMAGE_SPEED_REPROJECTION_PX = 1.5
IMAGE_SPEED_GROUND_DEPTH_M = (3.0, 35.0)
IMAGE_SPEED_STD_MPS = 1.0  # 未標定的路面/PnP/高度模型 floor；非榜分擬合
IMAGE_SPEED_STATIONARY_STD_MPS = 0.3
IMAGE_SPEED_QUIET_MEDIAN_PX = 0.25
IMAGE_SPEED_QUIET_P90_PX = 0.75
IMAGE_SPEED_QUIET_GYRO_RADPS = 0.02
IMAGE_SPEED_MAX_MPS = 50.0
IMAGE_SPEED_MAX_ROTATION_RATE_RADPS = 2.0
LATERAL_VELOCITY_CONSTRAINT_ENABLED = True
LATERAL_VELOCITY_STD_MPS = 0.3  # 未標定低側滑模型；相當於 GNSS 0.3m / 1s 解析度
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
# 地面必須在兩個平面內方向都有支持；比例是形狀保護，並非道路寬度標定。
GROUND_MIN_INPLANE_RATIO = 0.20
GROUND_NORMAL_MODEL_STD_M = 0.10  # 同 MAX_SOURCE_GROUND_MAD_M 的模型厚度預算
GROUND_NORMAL_FLOOR_DEG = 0.5
GROUND_NORMAL_NIS_REJECT = 13.816  # 法向 tangent 2 維、99.9% Gaussian 參考界
CAMERA_HEIGHT_BOUNDS_M = (0.5, 3.0)
CAMERA_HEIGHT_STD_M = 0.15  # 初始 pose 高度雜訊工程先驗；不是精確標定
CAMERA_HEIGHT_RW_M_SQRT_S = 0.01
HEIGHT_OBSERVATION_FLOOR_M = 0.05
MIN_SOURCE_FEATURE_HEIGHT_RAW = 0.02
MIN_MAP_FEATURE_HEIGHT_M = 0.10
SOURCE_FEATURE_LEAF_RAW = 0.035
FEATURE_MAX_RANGE_M = 40.  # ground-scale seed 換算的三維距離；30m 的800幀對照未顯示優勢
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
LINE_FEATURES_ENABLED = True
LINE_MIN_NEIGHBORS = 6
LINE_MAX_SECOND_RATIO = 0.10  # 線狀鄰域：第二／最大 scatter eigenvalue
LINE_MIN_VARIANCE_M2 = METRIC_MAP_LEAF_M**2/4
LINE_MAX_VERTICAL_ANGLE_DEG = 15.0
LINE_MAX_DIRECTION_DIFFERENCE_DEG = 15.0
PLANE_MIN_SUPPORT_PER_VARIABLE = 2.0  # 每個可觀測 pose 模式＋scale nuisance，至少兩個 angular 等效支持；不按全影像平面比例否決
MAX_PLANE_MEDIAN_M = FEATURE_STD_M
MAP_MIN_RELATIVE_INFORMATION = 1.0  # 正式 v5.10 基線的保留門檻；歷史連續模型僅以它刷新 strong memory
MAP_NUMERICAL_INFORMATION_RCOND = 1e-8  # 僅排除數值 null；弱方向用其大 covariance 降權
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
CORRECTION_HOLD_SECONDS = 0.5  # 各幾何方向分別計齡；短暫無觀測保留均值，variance 增加
CORRECTION_DECAY_SECONDS = 3.0  # 該方向超過 hold 後 exp(-未觀測時間/tau) 退回 sensor prior
# 檢查的是「修正量的差」，不是限制車輛一幀能行駛多遠。
CORRECTION_POSITION_GATE_BASE_M = 0.35
CORRECTION_POSITION_RATE_MPS = 0.5  # 同時限制每幀實際 correction 更新，避免重獲配準時延後突跳
CORRECTION_LATERAL_RATE_MPS = 0.2  # 側向偏差修正的工程速率；不隨前進距離放寬
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

# v5.6 沿用 v5.2 的 MAP_MIN_RELATIVE_INFORMATION=1、hold=0.5s、decay=3s。
# 以下歷史 v5.4/v5.5 參數在正式 v5.6 profile 不啟用：ground normal fusion、
# weak modes、line features、fractional heading、lateral velocity/directional gain。

# ===== v5.10：只在 v5.7 上移植 course 檢查與航向資訊預算 =====
V510_ENABLED = True  # False 僅供生成資料核對 v5.7 原行為
BASELINE_RUN_ID = 'track02_full_2611031941'
OUTPUT_REFERENCE = 'camera_ply'
REFERENCE_CONTRACT_STATUS = 'official scoring reference point not specified; preserve camera PLY pose'
EXTRINSIC_CALIBRATION_STATUS = 'GNSS/IMU to camera uncalibrated; identity/zero means with declared covariance'

def v510_heading_enabled():
    return V510_ENABLED and uses_v52_fusion()

# ===== v5.11：首筆 map anchor 完整比較候選；追蹤只在幾何失敗時恢復 =====
V511_ENABLED = False
STARTUP_HEIGHT_SEED_SIGMAS = (-1., 0., 1.)  # 沿用初始高度 covariance 的一個標準差
STARTUP_INCLUDE_GIVEN_YAW = True
STARTUP_INCLUDE_MAP_AXIS = True  # 無向主軸僅為 seed；仍須完整配準及既有 gate
TRACKING_RETRY_ON_GEOMETRY_FAILURE = True
VALIDATE_EMITTED_MAP_PROPOSAL = True
STARTUP_BOUND_AFTER_FIRST_FRAME = True  # 僅真正首幀可自由初始化；延遲首筆 map 也用當幀 gain 預算

# ===== v5.12：只調整 map 量測的統計意義及實際輸出驗收 =====
V512_ENABLED = False
MAP_DATA_LIKELIHOOD_ENABLED = True  # 配準 prior 只作搜尋；濾波使用 data-only likelihood，避免 prior 算兩次
MAP_REMATCHED_POSTERIOR_CHECK = True  # 同時檢查固定平面與原樣本重新對應的共同成本
TRACKING_RETRY_AFTER_NONCONVERGENCE = True  # 有可觀測資訊但未收斂，才沿既有 yaw σ 加起點

def v512_enabled():
    return V512_ENABLED and V511_ENABLED and uses_v52_fusion()

# ===== v5.12.1：以已評分 v5.10 為基準，只修啟動交接及側向突變 =====
V5121_ENABLED = False
GNSS_POSITION_INFORMATION_BUDGET = True  # 一筆已抵達 likelihood 分次使用；總資訊不超過原觀測
MAP_BODY_LATERAL_BUDGET = True  # 前進距離不放大側向修正；沿用既有 0.2 m/s 與 turn-lever 預算
MAP_BOUNDED_REFERENCE_RESPONSE = True  # prior 白化後的 PSD contraction；阻止 yaw 校準交叉項放大平移

def v5121_enabled():
    return V5121_ENABLED and v510_heading_enabled()

# ===== v5.13：v5.10 基準；影像 anchor、相對位姿與幾何診斷可分開驗證 =====
V513_ENABLED = False
IMAGE_ANCHOR_MEMORY_ENABLED = True
IMAGE_RELATIVE_TRANSLATION_ENABLED = True
IMAGE_RELATIVE_YAW_ENABLED = True  # 與 translation 交替使用不重疊區間；不提供絕對 yaw
IMAGE_RELATIVE_SOLVER = 'gyro_translation'  # 'pnp'；兩者均保存共同像素比較
MAP_ANGULAR_STRATIFIED_CAP = True  # 保留稀少視角的樣本，總點數上限不變
MAP_GEOMETRY_DIAGNOSTICS = True  # 法向資訊與每個 solver step；不放寬 gate

def v513_enabled():
    return V513_ENABLED and v510_heading_enabled()

# ===== v5.14：修正影像模型契約，保留原品質門檻與 map 算法 =====
V514_ENABLED = False

def v514_enabled():
    return V514_ENABLED and v513_enabled()

# ===== v5.15：v5.10 凍結數值基線，只改 current-map likelihood 與輸出驗收 =====
V515_ENABLED = False  # v5.15 full 0.51729 退步；恢復已重現的 v5.10，30m 尚為研究候選

def v515_enabled():
    # Historical branches remain independently reproducible in generated tests.
    # Production v5.15 never mixes their startup, image or GNSS budget changes.
    return (V515_ENABLED and v510_heading_enabled()
        and not (V511_ENABLED or V512_ENABLED or V5121_ENABLED or V513_ENABLED or V514_ENABLED))

# ===== v5.16.1：保留 v5.16 的融合，只補可靠表面對應，不改40m或品質門檻 =====
V5161_ENABLED = False  # v5.16.1/3 正式分數退步；本輪單獨還原 v5.10 map association
SURFACE_FALLBACK_NORMAL_MAX_DEG = LINE_MAX_DIRECTION_DIFFERENCE_DEG
SURFACE_FALLBACK_YAW_STD_DEG = HEADING_MAX_COURSE_STD_DEG

def v5161_enabled():
    return (V5161_ENABLED and v510_heading_enabled()
        and not (V511_ENABLED or V512_ENABLED or V5121_ENABLED or V513_ENABLED or V514_ENABLED or V515_ENABLED))

# ===== v5.16.2：以已抵達 gyro 轉速傳播 XY 速度，靜止時學 motion-only gyro bias =====
V5162_ENABLED = True
# 沿用 v5.9 已聲明的工程預算；本版先獨立驗證，沒有用榜分或目標 CSV 擬合。
TURN_GYRO_REST_STD_DEGPS = 0.10
TURN_GYRO_BIAS_RW_DEGPS_SQRT_S = 0.02

def v5162_enabled():
    return (V5162_ENABLED and v510_heading_enabled()
        and not (V511_ENABLED or V512_ENABLED or V5121_ENABLED or V513_ENABLED or V514_ENABLED or V515_ENABLED))

# ===== v5.16.3：路面速度的 LK 特徵必須有完整 finest-window 路面支持 =====
V5163_ENABLED = True

def v5163_enabled():
    return V5163_ENABLED and v5162_enabled()

# ===== v5.16.5：原 map pose 融合不變，高度改用 data-only likelihood 一次融合 =====
V5165_ENABLED = False  # 正式0.43213差於v5.16.4的0.42660；保留機制，正式回復舊高度更新

def v5165_enabled():
    return V5165_ENABLED and v5163_enabled() and not v5161_enabled()

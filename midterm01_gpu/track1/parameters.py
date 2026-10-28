"""Track1 v4：因果 GNSS／地面尺度 + 有記憶與 innovation gate 的地圖修正。

直接改此檔。GNSS/heading 保留 5839746c 的工程先驗；新版尺度與配準
參數在下半段。GNSS 空 frame、IMU 與 PLY 軸向／零 lever 仍未標定。
"""
VERSION = 'track1_causal_metric_correction_memory_v4'
BASE_COMMIT = '5839746ce1e4fe59ed4980ae07cb5ceb3f27ddeb'
DATA_DIR = '/workspace/midterm01/data/Track1'
EXPECTED_FRAMES = 3011
PLY_SCALE = 3.0  # 只在缺可靠地面時的未驗證 fallback，不再固定每幀尺度
GNSS_STD_M = 0.3
GNSS_XY_ACCEL_PSD = 4.0
HEADING_BIAS_RW_DEG_SQRT_S = 0.12
HEADING_COURSE_WINDOW_S = 2.0
HEADING_MIN_SPEED_MPS = 5.0
HEADING_MAX_TURN_RATE_DEGPS = 2.0
HEADING_MAX_SPAN_DEG = 4.0
HEADING_STD_FLOOR_DEG = 0.5
HEADING_MAX_OBSERVATION_DEG = 15.0
HEADING_MAX_BIAS_DEG = 10.0
HEADING_MAX_BIAS_RATE_DEGPS = 0.2
HEADING_ROBUST_GATE_SIGMA = 3.0
GNSS_EXPECTED_FRAME = ''
IMU_MAX_AGE_S = 0.1

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
METRIC_POSITION_STD_M = 1.0  # GNSS + 未標定 camera lever/局部偏差的合成工程先驗
METRIC_YAW_STD_DEG = 3.0
METRIC_MAX_XY_CORRECTION_M = 3.0  # position prior 的 3σ 數值搜索界
METRIC_MAX_YAW_CORRECTION_DEG = 8.0
METRIC_MAX_HEIGHT_CHANGE_M = 0.6
SCALE_BOUNDS = (0.2, 30.)
METRIC_STAGES = ((3.0, 1.0, 40), (1.5, 0.5, 40), (0.75, 0.3, 40))
MIN_METRIC_MATCHES = 30
MIN_METRIC_COVERAGE = 0.6
MAX_FEATURE_MEDIAN_M = 0.5  # 必須低於 final correspondence，避免「全不匹配」被接受
MAX_SCALE_LOG_STD = 0.15   # 只在非地面 scale 資訊足夠時更新過去高度 anchor

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

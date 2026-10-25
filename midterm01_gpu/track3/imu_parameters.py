"""IMU 外參與 gyro noise 入口；v8 用 resolve_motion_config，不使用 Kalman。

MAP_*／YAW_* 及 resolve_fusion_config 僅保留給歷史 ESKF 元件與測試。
"""
from copy import deepcopy
import math

# IMU → PLY FLU：固定軸 roll、pitch、yaw，單位為度。
IMU_RPY_DEG = (1.0, 6.0, 0.0)
# IMU 原點在 PLY FLU 的位置，單位為公尺；X 前、Y 左、Z 上。
IMU_XYZ_M = (0.0, 0.0, 0.0)

# ===== 可自行調整：gyro 與地圖配準的信任度 =====
# 下列仍是未校正的 engineering priors；一次只改一項，不能當成精度保證。
# Gyro 白雜訊密度，單位 (rad/s)/sqrt(Hz)，等價於 rad/sqrt(s)。
# 變小：降低 prediction 的過程不確定性，通常較相信 gyro；不改 gyro 讀值。
# v8 改變粒子旋轉擾動；歷史 ESKF 才會改變 Kalman gain 與 NIS gate。
GYRO_NOISE_DENSITY_RAD_S_SQRT_HZ = 0.02
# Gyro bias random walk，單位 (rad/s)/sqrt(s)；越大，允許 bias 變動越快。
GYRO_BIAS_RANDOM_WALK_RAD_S_SQRT_S = 0.001
# 地圖配準的位置標準差 (map X, Y, Z)，公尺；越小，越相信該方向量測。
MAP_POSITION_STD_M = (0.5, 0.5, 0.5)
# 地圖配準的小角度旋轉標準差 (map X, Y, Z)，度；越小，越相信該方向量測。
# 這是旋轉誤差向量，不是 Euler roll/pitch/yaw，也不是 deg/s。
# 接近水平時，第 3 項約對應 yaw；程式會轉成 rad 後交給原本的 filter。
MAP_ROTATION_STD_DEG = (5.0, 5.0, 5.0)

# ===== 新增：因果 yaw robust weighting（在原 ESKF 同一次更新內）=====
# False 可關閉以比較原融合；不讀未來、不回改舊 pose，也不限制真正的轉彎。
YAW_ROBUST_ENABLED = True
# gyro/視覺 prediction 與本幀 map yaw 差異，除以其合成標準差後的 Huber 轉折。
# 3.0 依「3 sigma 內保持原權重」設定；越小越早降低不一致 map yaw 的信任。
# 此數值不是角度，不是 max yaw-rate；noise 或 gyro bias 有錯仍可能判斷錯誤。
YAW_HUBER_SIGMA = 3.0


def _positive_finite(value, name):
    """參數錯誤統一報 ValueError，不讓無效 noise 進入原本的 filter。"""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} 必須為有限正數') from exc
    if isinstance(value, bool) or not math.isfinite(number) or number <= 0:
        raise ValueError(f'{name} 必須為有限正數')
    return number


def _positive_vector3(values, name):
    try:
        if len(values) != 3:
            raise ValueError(f'{name} 必須有三個有限正數')
        return [_positive_finite(value, name) for value in values]
    except TypeError as exc:
        raise ValueError(f'{name} 必須有三個有限正數') from exc


def resolve_fusion_config(config):
    """將本檔的信任度常數寫入 imu 設定副本，並留下有效值與來源。"""
    cfg = deepcopy(config)
    gyro = _positive_finite(GYRO_NOISE_DENSITY_RAD_S_SQRT_HZ,
                            'GYRO_NOISE_DENSITY_RAD_S_SQRT_HZ')
    bias = _positive_finite(GYRO_BIAS_RANDOM_WALK_RAD_S_SQRT_S,
                            'GYRO_BIAS_RANDOM_WALK_RAD_S_SQRT_S')
    position = _positive_vector3(MAP_POSITION_STD_M, 'MAP_POSITION_STD_M')
    rotation = _positive_vector3(MAP_ROTATION_STD_DEG, 'MAP_ROTATION_STD_DEG')
    if not isinstance(YAW_ROBUST_ENABLED, bool):
        raise ValueError('YAW_ROBUST_ENABLED 必須為 bool')
    yaw_sigma = _positive_finite(YAW_HUBER_SIGMA, 'YAW_HUBER_SIGMA')
    cfg['gyro_noise_density'] = gyro
    cfg['gyro_bias_random_walk'] = bias
    cfg['registration_std'] = position + [math.radians(value) for value in rotation]
    cfg['yaw_constraint'] = dict(enabled=YAW_ROBUST_ENABLED, huber_sigma=yaw_sigma,
                                 min_horizontal_norm=.1)
    cfg['fusion_parameter_source'] = 'track3/imu_parameters.py'
    cfg['fusion_parameter_status'] = 'uncalibrated_engineering_priors_not_accuracy_claim'
    cfg['fusion_parameter_values'] = dict(
        gyro_noise_density_rad_s_per_sqrt_hz=gyro,
        gyro_bias_random_walk_rad_s_per_sqrt_s=bias,
        map_position_std_m=position,
        map_rotation_std_deg=rotation,
        yaw_robust_enabled=YAW_ROBUST_ENABLED,
        yaw_huber_sigma=yaw_sigma,
    )
    return cfg


def resolve_motion_config(config):
    """v8 僅讀 gyro 的過程雜訊；不建立 Kalman 或讀取 map/yaw 融合參數。"""
    cfg=deepcopy(config)
    cfg['gyro_noise_density']=_positive_finite(GYRO_NOISE_DENSITY_RAD_S_SQRT_HZ,
                                              'GYRO_NOISE_DENSITY_RAD_S_SQRT_HZ')
    cfg['gyro_bias_random_walk']=_positive_finite(GYRO_BIAS_RANDOM_WALK_RAD_S_SQRT_S,
                                                'GYRO_BIAS_RANDOM_WALK_RAD_S_SQRT_S')
    cfg['motion_parameter_source']='track3/imu_parameters.py'
    cfg['motion_parameter_status']='uncalibrated_engineering_priors_not_accuracy_claim'
    return cfg


def resolve_imu_config(config):
    """將 Python 外參寫入實際執行設定；舊的顯式矩陣設定仍可讀取。"""
    cfg = deepcopy(config)
    source = cfg.get('extrinsic_source', 'config')
    if source == 'config':
        return cfg
    if source != 'python':
        raise ValueError(f'Unknown IMU extrinsic_source: {source}')
    if len(IMU_RPY_DEG) != 3 or len(IMU_XYZ_M) != 3 or not all(
            math.isfinite(v) for v in (*IMU_RPY_DEG, *IMU_XYZ_M)):
        raise ValueError('IMU_RPY_DEG 與 IMU_XYZ_M 必須各有三個有限數值')
    roll, pitch, yaw = map(math.radians, IMU_RPY_DEG)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    x, y, z = IMU_XYZ_M
    # Rz(yaw) @ Ry(pitch) @ Rx(roll), p_ply = R_ply_imu p_imu + t_ply_imu.
    cfg['T_ply_imu'] = [
        [cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr, x],
        [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr, y],
        [-sp, cp*sr, cp*cr, z], [0., 0., 0., 1.]]
    identity = not any((*IMU_RPY_DEG, *IMU_XYZ_M))
    cfg['extrinsic_status'] = ('assumed_identity_unverified_not_calibrated' if identity
                               else 'assumed_python_unverified_not_calibrated')
    cfg['extrinsic_rpy_deg'] = list(IMU_RPY_DEG)
    cfg['extrinsic_xyz_m'] = list(IMU_XYZ_M)
    return cfg

"""IMU 外參調整入口：直接修改下方兩行，執行指令保持相同。"""
from copy import deepcopy
import math

# IMU → PLY FLU：固定軸 roll、pitch、yaw，單位為度。
IMU_RPY_DEG = (1.0, 6.0, 0.0)
# IMU 原點在 PLY FLU 的位置，單位為公尺；X 前、Y 左、Z 上。
IMU_XYZ_M = (0.0, 0.0, 0.0)


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

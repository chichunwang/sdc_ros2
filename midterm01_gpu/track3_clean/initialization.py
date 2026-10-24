"""Training-only lower-plane hypothesis; explicitly not physical calibration."""
import numpy as np
from scipy.spatial.transform import Rotation

def initial_ground_tilt(raw, initial, scale):
    """One-time lower-plane orientation seed; flat local map-up approximation.

    This does not estimate translation/scale and does not call the new Sim3
    optimizer. Small 3x3 plane fits run on CPU. Official initialization remains
    available as an explicit control, because the physical camera frame is not
    certified. A fitted floor is not a calibration or a known vehicle height.
    """
    raw = np.asarray(raw, float)
    metric = raw*scale
    if len(metric) < 64 or not np.isfinite(metric).all():
        raise ValueError('Insufficient finite points for initial ground plane')
    lower = metric[metric[:, 2] <= np.quantile(metric[:, 2], .45)]
    selected = lower
    for _ in range(5):
        center = np.median(selected, axis=0)
        covariance = (selected-center).T @ (selected-center)/len(selected)
        eigen, vectors = np.linalg.eigh(covariance)
        normal = vectors[:, 0]
        if normal[2] < 0:
            normal = -normal
        residual = (lower-center) @ normal
        offset = np.median(residual)
        mad = np.median(abs(residual-offset))
        selected = lower[abs(residual-offset) <= max(.03, 2.5*1.4826*mad)]
        if len(selected) < 24:
            raise ValueError('Ground plane lost support')
    rms = float(np.sqrt(np.mean(((selected-center) @ normal)**2)))
    if eigen[1] < .015 or rms > .15 or normal[2] < .85:
        raise ValueError('Lower surface is not a supported source ground plane')
    world = initial[:3, :3] @ normal
    up = np.array([0., 0., 1.])
    axis = np.cross(world, up)
    angle = np.arctan2(np.linalg.norm(axis), world @ up)
    if angle >= np.deg2rad(24.):
        raise ValueError('Ground tilt exceeds the explicit 24 degree initialization domain')
    pose = initial.copy()
    if np.linalg.norm(axis) > 1e-12:
        pose[:3, :3] = Rotation.from_rotvec(axis/np.linalg.norm(axis)*angle).as_matrix() @ initial[:3, :3]
    return pose, {'mode': 'ground', 'tilt_correction_deg': float(np.rad2deg(angle)),
                  'source_ground_normal': normal.tolist(), 'plane_rms_m': rms,
                  'plane_points': len(selected), 'translation_changed': False, 'scale_changed': False,
                  'assumption': 'local map ground normal approximated by map +Z; not physical calibration'}



"""三相機 metric map 尺度估計；點到平面 robust ICP，包含每幀 pose nuisance。"""
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from three_camera_source import ply_header


def load_ply(path, max_points=6000, leaf=.3):
    lines, offset = ply_header(path)
    count = int(next(line.split()[2] for line in lines if line.startswith('element vertex ')))
    properties = [line for line in lines if line.startswith('property ')]
    if properties != ['property float x', 'property float y', 'property float z',
                      'property uchar red', 'property uchar green', 'property uchar blue']:
        raise ValueError('Unsupported PLY schema')
    if 'format binary_little_endian 1.0' in lines:
        dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', 'u1', (3,))])
        if Path(path).stat().st_size != offset + count * dtype.itemsize:
            raise ValueError('Binary PLY payload length mismatch')
        packed = np.fromfile(path, dtype=dtype, offset=offset, count=count)
        points = np.column_stack([packed[k] for k in 'xyz']).astype(float)
    else:
        with Path(path).open() as stream:
            for _ in lines:
                next(stream)
            points = np.loadtxt(stream, usecols=(0, 1, 2))
        if len(points) != count:
            raise ValueError('ASCII PLY vertex count mismatch')
    if points.shape != (count, 3) or not np.all(np.isfinite(points)):
        raise ValueError('Nonfinite/invalid PLY')
    if leaf:
        _, keep = np.unique(np.floor(points / leaf).astype(np.int64), axis=0, return_index=True)
        points = points[np.sort(keep)]
    if len(points) > max_points:
        points = points[np.linspace(0, len(points) - 1, max_points, dtype=int)]
    return points


def load_binary_map(path):
    with Path(path).open('rb') as stream:
        header = {}
        while True:
            line = stream.readline().decode('ascii').strip()
            if not line:
                raise ValueError('Invalid PCD header')
            words = line.split()
            header[words[0]] = words[1:]
            if words[0] == 'DATA':
                break
        if header['FIELDS'] != ['x', 'y', 'z'] or header['SIZE'] != ['4', '4', '4'] or header['DATA'] != ['binary']:
            raise ValueError('Use a verified binary XYZ map cache')
        points = np.frombuffer(stream.read(), dtype='<f4').reshape(-1, 3).astype(float)
    if len(points) != int(header['POINTS'][0]) or not np.all(np.isfinite(points)):
        raise ValueError('Invalid map payload')
    return points


class SurfaceMap:
    def __init__(self, points):
        self.points = np.asarray(points, dtype=float)
        self.tree = cKDTree(self.points)
        self.normals = np.zeros_like(self.points)
        self.quality = np.zeros(len(self.points))
        self.computed = np.zeros(len(self.points), dtype=bool)

    def surfaces(self, indices):
        pending = np.unique(indices[~self.computed[indices]])
        if len(pending):
            distances, neighbors = self.tree.query(self.points[pending], k=16)
            for index, dd, nn in zip(pending, distances, neighbors):
                local = self.points[nn[dd <= 2.5]] - self.points[index]
                if len(local) >= 6:
                    local -= np.mean(local, axis=0)
                    values, vectors = np.linalg.eigh(local.T @ local / len(local))
                    if values[1] >= .04 and values[0] <= .25 * values[1] and values[2] > 1e-8:
                        quality = (values[1] - values[0]) / values[2]
                        if quality >= .2:
                            self.normals[index] = vectors[:, 0]
                            self.quality[index] = np.clip(quality, 0, 1)
                self.computed[index] = True
        return self.normals[indices], self.quality[indices]


def frame_terms(surface_map, raw, prior, log_scale, delta, derivatives=True):
    rotation = Rotation.from_euler('z', delta[3]).as_matrix() @ prior[:3, :3]
    rotated = np.exp(log_scale) * raw @ rotation.T
    transformed = rotated + prior[:3, 3] + delta[:3]
    distances, indices = surface_map.tree.query(transformed)
    normals, quality = surface_map.surfaces(indices)
    residual = np.einsum('ij,ij->i', normals, transformed - surface_map.points[indices])
    valid = (distances <= 2) & (quality > 0)
    clipped = np.minimum(np.abs(residual), 2)
    loss = np.where(clipped <= .5, .5 * clipped**2, .5 * (clipped - .25))
    confidence = quality * valid
    norm = 2 / (len(raw) * .3**2)
    # Fixed missing-match cost prevents reducing cost simply by losing support.
    cost = norm * np.sum(.875 - confidence * (.875 - loss))
    precision = np.array([4., 4., 1., 1 / np.deg2rad(3)**2])
    cost += .5 * np.sum(precision * delta**2)
    stats = {'reliable_points': int(valid.sum()), 'unique_targets': int(len(np.unique(indices[valid]))),
             'clipped_nn_rmse_m': float(np.sqrt(np.mean(np.minimum(distances, 2)**2))),
             'plane_rmse_m': float(np.sqrt(np.mean(residual[valid]**2))) if valid.any() else None}
    if not derivatives:
        return float(cost), stats
    jac = np.column_stack((np.einsum('ij,ij->i', normals, rotated), normals,
                           -normals[:, 0] * rotated[:, 1] + normals[:, 1] * rotated[:, 0]))
    weight = norm * confidence * np.minimum(1, .5 / np.maximum(np.abs(residual), 1e-12))
    data_h = jac.T @ (jac * weight[:, None])
    hessian = data_h.copy()
    hessian[1:, 1:] += np.diag(precision)
    gradient = jac.T @ (weight * residual)
    gradient[1:] += precision * delta
    # Exclude pose priors: a ground plane must not appear to identify scale.
    scale_info = max(0., data_h[0, 0] - data_h[0, 1:] @ np.linalg.pinv(data_h[1:, 1:], rcond=1e-8) @ data_h[1:, 0])
    stats['scale_information_after_free_pose'] = float(scale_info)
    stats['scale_information_fraction'] = float(scale_info / max(data_h[0, 0], 1e-12))
    return float(cost), stats, hessian, gradient


def fit_scale(surface_map, frames, initial_scale=1., fixed_scale=None, iterations=30):
    """One shared scale, bounded 4-DOF nuisance per frame; no per-frame scale fit."""
    scale = float(fixed_scale if fixed_scale is not None else initial_scale)
    if not .5 <= scale <= 2 or not frames:
        raise ValueError('Scale initialization must be within physical diagnostic bounds [0.5,2]')
    log_scale = np.log(scale)
    deltas = [np.zeros(4) for _ in frames]
    scenes = list(surface_map) if isinstance(surface_map, (list, tuple)) else [surface_map] * len(frames)
    if len(scenes) != len(frames):
        raise ValueError('Each calibration frame needs one map')
    history = []
    for iteration in range(iterations):
        terms = [frame_terms(scene, raw, prior, log_scale, d) for scene, (raw, prior), d in zip(scenes, frames, deltas)]
        total = sum(t[0] for t in terms)
        schur = gradient = 0.
        for _, _, h, g in terms:
            schur += h[0, 0] - h[0, 1:] @ np.linalg.solve(h[1:, 1:], h[1:, 0])
            gradient += g[0] - h[0, 1:] @ np.linalg.solve(h[1:, 1:], g[1:])
        step_s = 0. if fixed_scale is not None else np.clip(-gradient / max(schur, 1e-12), -.1, .1)
        steps = [-np.linalg.solve(h[1:, 1:], g[1:] + h[1:, 0] * step_s) for _, _, h, g in terms]
        factor = min([1.] + [.25 / max(np.linalg.norm(d[:3]), 1e-12) for d in steps] +
                     [np.deg2rad(2) / max(abs(d[3]), 1e-12) for d in steps])
        accepted = False
        for _ in range(10):
            next_log = log_scale + step_s * factor
            next_d = [d + step * factor for d, step in zip(deltas, steps)]
            bounded = (np.log(.5) <= next_log <= np.log(2) and
                       all(np.linalg.norm(d[:2]) <= 1.5 and abs(d[2]) <= 1.5 and abs(d[3]) <= np.deg2rad(8) for d in next_d))
            cost = sum(frame_terms(scene, raw, prior, next_log, d, False)[0]
                       for scene, (raw, prior), d in zip(scenes, frames, next_d)) if bounded else float('inf')
            if cost < total - 1e-10:
                log_scale, deltas = next_log, next_d
                accepted = True
                break
            factor *= .5
        history.append({'iteration': iteration, 'cost': total, 'scale': float(np.exp(log_scale)), 'accepted': accepted})
        if not accepted or max(abs(step_s * factor), *(np.linalg.norm(step * factor) for step in steps)) < 1e-5:
            break
    final = [frame_terms(scene, raw, prior, log_scale, d) for scene, (raw, prior), d in zip(scenes, frames, deltas)]
    return {'scale': float(np.exp(log_scale)), 'cost': sum(t[0] for t in final), 'history': history,
            'frames': [{**term[1], 'delta_xyz_yaw': d.tolist()} for term, d in zip(final, deltas)],
            'scope': 'local robust map-consistency scale estimate; no ground truth or calibrated covariance'}


def camera_to_reference(points, map_from_front_reference, map_from_front_capture, front_from_camera):
    """For KNOWN per-view points/extrinsics only; never apply to an unlabeled merged PLY."""
    transform = np.linalg.inv(map_from_front_reference) @ map_from_front_capture @ front_from_camera
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]

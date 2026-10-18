"""固定三相機外參、fisheye 校正與不依賴地圖的幾何均衡取樣。"""
from pathlib import Path
import csv
import cv2
import numpy as np
import yaml
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

OPTICAL_TO_FLU = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])


def transform(points, pose):
    return np.asarray(points) @ pose[:3, :3].T + pose[:3, 3]


def load_rig(directory):
    directory = Path(directory)
    document = yaml.safe_load((directory / 'lucid_camera_extrinsics.yaml').read_text())
    if document['convention']['translation_unit'] != 'meter':
        raise ValueError('Rig translation must be in meters')
    rig = {'f': np.eye(4)}
    for tag, name in [('fl', 'front_left'), ('fr', 'front_right')]:
        pose = np.array(document['transforms'][f'T_front_{name}']['matrix'], dtype=float)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            raise ValueError('Invalid rig transform')
        np.testing.assert_allclose(pose[3], [0, 0, 0, 1], atol=1e-9)
        np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=3e-6)
        if abs(np.linalg.det(pose[:3, :3]) - 1) > 3e-6:
            raise ValueError('Rig rotation must be proper')
        rig[tag] = pose
    return rig


def rectify(image_path, calibration_path, size=(448, 294), exclusion_mask=None):
    """Raw fisheye -> pinhole; K and validity mask refer to exact output pixels."""
    info = yaml.safe_load(Path(calibration_path).read_text())
    image = cv2.imread(str(image_path))
    if image is None or image.shape[:2] != (info['image_height'], info['image_width']):
        raise ValueError('Raw image size/calibration mismatch')
    if info['distortion_model'] != 'fisheye':
        raise ValueError('Only explicitly calibrated fisheye images supported')
    k = np.array(info['camera_matrix']['data']).reshape(3, 3)
    d = np.array(info['distortion_coefficients']['data'])
    r = np.array(info['rectification_matrix']['data']).reshape(3, 3)
    np.testing.assert_allclose(r, np.eye(3), atol=1e-8)
    new_k = np.array(info['projection_matrix']['data']).reshape(3, 4)[:, :3].copy()
    new_k[0] *= size[0] / info['image_width']
    new_k[1] *= size[1] / info['image_height']
    mx, my = cv2.fisheye.initUndistortRectifyMap(k, d, r, new_k, size, cv2.CV_32FC1)
    valid = (mx >= 1) & (mx < image.shape[1] - 2) & (my >= 1) & (my < image.shape[0] - 2)
    if exclusion_mask:
        excluded = cv2.imread(str(exclusion_mask), cv2.IMREAD_GRAYSCALE)
        if excluded is None or excluded.shape != image.shape[:2]:
            raise ValueError('Exclusion mask size mismatch')
        valid &= cv2.remap(excluded, mx, my, cv2.INTER_NEAREST, borderValue=255) == 0
    rgb = cv2.cvtColor(cv2.remap(image, mx, my, cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    return rgb, new_k.astype(np.float32), valid


def backproject(depth_z, intrinsics):
    depth_z = np.asarray(depth_z)
    yy, xx = np.indices(depth_z.shape)
    rays = np.stack(((xx - intrinsics[0, 2]) / intrinsics[0, 0],
                     (yy - intrinsics[1, 2]) / intrinsics[1, 1], np.ones_like(xx)), axis=-1)
    return rays * depth_z[..., None]


class PriorTrajectory:
    """Relative motion approximation using original sensor prior, never map posterior."""
    def __init__(self, path, max_extrapolation_seconds=0.):
        rows = list(csv.DictReader(Path(path).open()))
        stamps = np.array([int(r['timestamp_ns']) for r in rows], dtype=np.int64)
        if len(stamps) < 2 or np.any(np.diff(stamps) <= 0):
            raise ValueError('Prior times must strictly increase')
        self.origin = int(stamps[0])
        self.times = (stamps - self.origin) / 1e9
        self.xyz = np.array([[float(r[k]) for k in ('x', 'y', 'z')] for r in rows])
        self.slerp = Slerp(self.times, Rotation.from_quat([[float(r[k]) for k in ('qx', 'qy', 'qz', 'qw')] for r in rows]))
        self.max_extrapolation_seconds = float(max_extrapolation_seconds)
        if not 0 <= self.max_extrapolation_seconds <= .03:
            raise ValueError('Explicit motion extrapolation limit must be in [0,0.03] seconds')
        self.extrapolated_queries = {}

    def pose(self, stamp):
        t = (int(stamp) - self.origin) / 1e9
        pose = np.eye(4)
        if self.times[0] <= t <= self.times[-1]:
            pose[:3, :3] = self.slerp(t).as_matrix()
            pose[:3, 3] = [np.interp(t, self.times, self.xyz[:, i]) for i in range(3)]
        else:
            edge = self.times[0] if t < self.times[0] else self.times[-1]
            if abs(t-edge) > self.max_extrapolation_seconds:
                raise ValueError('Camera time outside prior/extrapolation limit; no silent endpoint clamp')
            a, b = (0, 1) if t < self.times[0] else (-2, -1)
            fraction = (t-self.times[a]) / (self.times[b]-self.times[a])
            r0, r1 = self.slerp([self.times[a], self.times[b]])
            pose[:3, :3] = (Rotation.from_rotvec((r1*r0.inv()).as_rotvec()*fraction)*r0).as_matrix()
            pose[:3, 3] = self.xyz[a] + fraction*(self.xyz[b]-self.xyz[a])
            self.extrapolated_queries[int(stamp)] = float(t-edge)
        return pose

    def camera_pose(self, reference_stamp, capture_stamp, optical_extrinsic):
        c = np.eye(4)
        c[:3, :3] = OPTICAL_TO_FLU
        return np.linalg.inv(c) @ np.linalg.inv(self.pose(reference_stamp)) @ self.pose(capture_stamp) @ c @ optical_extrinsic


def voxel_indices(points, leaf=.3):
    _, indices = np.unique(np.floor(points / leaf).astype(np.int64), axis=0, return_index=True)
    return np.sort(indices)


def balanced_indices(points, camera_ids, confidence, budget=6000, leaf=.3):
    """Equal camera x normal-axis budget, round-robin 2 m cells, no map/score input.

    First voxelize within each camera. Reject poorly supported nonplanar patches;
    cap ground through the same quota as each wall direction. Sparse bins return
    their unused budget, rather than filling it with duplicate road points.
    """
    points, camera_ids, confidence = np.asarray(points), np.asarray(camera_ids), np.asarray(confidence)
    if len(points) != len(camera_ids) or len(points) != len(confidence) or budget < 1:
        raise ValueError('Invalid balanced sampling input')
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(confidence)):
        raise ValueError('Nonfinite sampling input')
    candidates = np.concatenate([np.flatnonzero(camera_ids == cam)[voxel_indices(points[camera_ids == cam], leaf)]
                                 for cam in np.unique(camera_ids)])
    if len(candidates) < 16:
        return candidates[:budget], {'eligible': len(candidates), 'bins': {}}
    p = points[candidates]
    distances, neighbors = cKDTree(p).query(p, k=16)
    bins = {}
    for j, (dd, nn) in enumerate(zip(distances, neighbors)):
        local = p[nn[dd <= 1.5]]
        if len(local) < 6:
            continue
        local = local - local.mean(axis=0)
        values, vectors = np.linalg.eigh(local.T @ local / len(local))
        if values[1] < .01 or values[2] <= 1e-8 or (values[1] - values[0]) / values[2] < .2 or values[0] > .25 * values[1]:
            continue
        axis = int(np.argmax(np.abs(vectors[:, 0])))
        original = int(candidates[j])
        key = (int(camera_ids[original]), axis)
        cell = tuple(np.floor(points[original] / 2.).astype(int))
        bins.setdefault(key, {}).setdefault(cell, []).append(original)
    chosen, counts = [], {}
    quota = budget // max(1, len(bins))
    for key, cells in sorted(bins.items()):
        ordered = [sorted(indices, key=lambda i: (-confidence[i], i)) for _, indices in sorted(cells.items())]
        selected, depth = [], 0
        while len(selected) < quota:
            batch = [ids[depth] for ids in ordered if depth < len(ids)]
            if not batch:
                break
            selected.extend(batch[:quota - len(selected)])
            depth += 1
        chosen.extend(selected)
        counts[f'camera{key[0]}_normal{key[1]}'] = {'available': sum(map(len, cells.values())), 'selected': len(selected), 'cells': len(cells)}
    return np.array(sorted(chosen), dtype=int), {'eligible': len(candidates), 'quota_per_nonempty_bin': quota, 'bins': counts}


def write_ply(path, points, rgb):
    points, rgb = np.asarray(points), np.asarray(rgb)
    if points.shape != (len(rgb), 3) or not len(points) or not np.all(np.isfinite(points)):
        raise ValueError('Invalid exported cloud')
    header = ('ply\nformat binary_little_endian 1.0\n'
              'comment coordinate_frame=front_camera_FLU_x_front_y_left_z_up\n'
              f'element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    packed = np.empty(len(points), dtype=[('xyz', '<f4', (3,)), ('rgb', 'u1', (3,))])
    packed['xyz'], packed['rgb'] = points, rgb
    with Path(path).open('xb') as stream:
        stream.write(header.encode('ascii'))
        packed.tofile(stream)

"""Adapters for the frozen four-DOF C++ core; no GNSS or trajectory smoothing."""
import json
from pathlib import Path
import shlex
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation

from causal_track3_filter import proper_pose


class CarryGyro:
    """Previous output position + past vertical gyro; no Kalman map update."""
    def __init__(self, pose, timestamp_ns):
        self.pose = proper_pose(pose)
        self.ns = int(timestamp_ns)
        self.held = None

    def _advance(self, end_ns):
        if end_ns <= self.ns:
            return
        if self.held is None or (end_ns-self.held[1])/1e9 > .08:
            raise ValueError('No fresh past IMU for carry/gyro prediction')
        angle = self.held[2]*((end_ns-self.ns)/1e9)
        self.pose[:3, :3] = Rotation.from_rotvec([0., 0., angle]).as_matrix() @ self.pose[:3, :3]
        self.ns = int(end_ns)

    def predict_to(self, ns, cursor):
        if ns < self.ns:
            raise ValueError('Cannot revisit a published frame')
        consumed = 0
        for record in cursor.take_until(ns):
            self._advance(record[0])
            self.held = record
            consumed += 1
        self._advance(ns)
        if self.held is None or (ns-self.held[1])/1e9 > .08:
            raise ValueError('No fresh past IMU')
        return self.pose.copy(), {'imu_samples_consumed': consumed,
            'max_imu_availability_ns': self.held[0], 'last_imu_measurement_ns': self.held[1],
            'imu_measurement_age_ms': (ns-self.held[1])/1e6, 'vertical_gyro_rad_s': self.held[2]}


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


def parse_result(line):
    fields = line.split()
    if len(fields) != 35 or fields[0] != 'RESULT' or fields[1] not in ('0', '1'):
        raise ValueError('Invalid old-core response: '+line[:200])
    numbers = np.array([float(v) for v in fields[2:5]+fields[6:]], float)
    if not np.isfinite(numbers).all() or (numbers[:3] < 0).any():
        raise ValueError('Nonfinite/negative old-core result')
    pose = proper_pose(np.array(list(map(float, fields[15:31]))).reshape(4, 4))
    return {'accepted': fields[1] == '1', 'load_ms': float(fields[2]),
            'registration_ms': float(fields[3]), 'total_ms': float(fields[4]),
            'reasons': [] if fields[1] == '1' else [fields[5]], 'update_status': fields[5],
            'inliers': int(fields[6]), 'iterations': int(fields[7]),
            'prior_cost': float(fields[8]), 'final_cost': float(fields[9]),
            'prior_rmse_m': float(fields[10]), 'final_rmse_m': float(fields[11]),
            'candidate_inliers': int(fields[12]), 'candidate_reliable_points': int(fields[13]),
            'candidate_unique_targets': int(fields[14]), 'pose': pose.tolist(),
            'correction': list(map(float, fields[31:35]))}


def build_bridge(root, run, container):
    """Compile frozen sources into THIS run; never replace installed binaries."""
    core = run/'pipeline_sources/workspace/src/track3_old_icp'
    script = run/'pipeline_sources/scripts/old_icp_bridge.cpp'
    def inside(path):
        return '/root/outputs/'+str(path.relative_to(root/'outputs'))
    packages = ['pcl_common-1.12', 'pcl_io-1.12', 'pcl_filters-1.12', 'pcl_kdtree-1.12']
    flags = subprocess.check_output(['docker', 'exec', container, 'pkg-config', '--cflags', '--libs', *packages], text=True)
    output = run/'old_icp_bridge'
    command = ['docker', 'exec', container, '/usr/bin/c++', '-std=c++17', '-O2',
        '-I'+inside(core/'include'), '-I/root/midterm/install/cuda_ndt_backend/include',
        '-I/usr/local/cuda-12.8/include', inside(script), inside(core/'src/sensor_guided.cpp'),
        inside(core/'src/geometry_weighting.cpp'), '-o', inside(output), *shlex.split(flags),
        '/root/midterm/install/cuda_ndt_backend/lib/libcuda_ndt_backend.so',
        '-L/usr/local/cuda-12.8/lib64', '-lcudart',
        '-Wl,-rpath,/root/midterm/install/cuda_ndt_backend/lib:/usr/local/cuda-12.8/lib64']
    (run/'build_command.json').write_text(json.dumps(command, indent=2)+'\n')
    with (run/'build.log').open('x') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    return output


def active_localizations():
    import os
    names = {'run_track3_causal.py', 'run_track3_old_icp.py', 'run_map_smoother.py',
             'run_map_smoother_gpu.py', 'benchmark.py'}
    found = []
    for item in Path('/proc').iterdir():
        if not item.name.isdigit() or int(item.name) == os.getpid():
            continue
        try:
            args = (item/'cmdline').read_bytes().decode().strip('\0').split('\0')
        except (OSError, UnicodeDecodeError):
            continue
        if args and ('python' in Path(args[0]).name) and any(Path(arg).name in names for arg in args[1:]):
            found.append(int(item.name))
    return found

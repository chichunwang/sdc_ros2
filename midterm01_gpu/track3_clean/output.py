"""Fixed user-facing serialization and integrity checks, without a solver import."""
import csv
from contextlib import ExitStack
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from .contracts import Similarity
from .data import ROOT

TIMING_HEADER = ['idx', 'original_idx', 'timestamp_ns', 'backend', 'load_ms',
                 'registration_ms', 'total_ms', 'map_accepted', 'scale']
BACKEND = 'cuda_float64_clean_local_sim3_ndt_icp'
CPU_REFERENCE_BACKEND = 'cpu_reference_clean_local_sim3_ndt_icp'


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def create_run(*, full=False):
    root = ROOT/'outputs/run'
    stem = 'track03_'+('full_' if full else 'test_')+datetime.now(ZoneInfo('Asia/Taipei')).strftime('%y%m%d%H%M')
    root.mkdir(parents=True, exist_ok=True)
    for i in range(1, 10000):
        path = root/(stem if i == 1 else stem+f'_{i:04d}')
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise FileExistsError('Run name collision limit exceeded')


class OutputWriter:
    def __init__(self, run, *, backend=BACKEND):
        self.run = Path(run)
        self.stack = ExitStack()
        self.backend = backend
        self.count = 0
        self.last_ns = None

    def __enter__(self):
        try:
            self.files = {name: self.stack.enter_context((self.run/name).open('x', newline=''))
                          for name in ('result.txt', 'diagnostic.csv', 'timings.csv', 'frames.csv', 'observations.jsonl')}
            self.csv = csv.writer(self.files['diagnostic.csv'])
            self.timing = csv.writer(self.files['timings.csv'])
            self.frames = csv.writer(self.files['frames.csv'])
            self.csv.writerow(['idx', 'x', 'y', 'yaw'])
            self.timing.writerow(TIMING_HEADER)
            self.frames.writerow(['idx', 'timestamp_ns', 'filename'])
            return self
        except BaseException:
            self.stack.close()
            raise

    def append(self, frame, result, *, load_ms, registration_ms, total_ms, elapsed_seconds):
        i, p = self.count, result.estimate.pose
        if frame.original_idx != i or frame.timestamp_ns != result.timestamp_ns:
            raise ValueError('Output requires the original 0-based chronological prefix')
        if self.last_ns is not None and frame.timestamp_ns <= self.last_ns:
            raise ValueError('Cannot rewrite a published frame')
        times = (load_ms, registration_ms, total_ms, elapsed_seconds)
        if not all(np.isfinite(v) and v >= 0 for v in times):
            raise ValueError('Expected finite nonnegative timings')
        observation = dict(idx=i, original_idx=frame.original_idx, timestamp_ns=frame.timestamp_ns,
                           input_cutoff_ns=frame.timestamp_ns, imu_used=False, gnss_used=False,
                           accepted=result.map_accepted, tracking_state=result.state.value,
                           tracking_lost=result.state.value == 'lost', source=result.source,
                           unsupported_seconds=result.unsupported_seconds, scale=result.estimate.scale,
                           pose=p.tolist(), elapsed_seconds=elapsed_seconds, **result.diagnostics)
        # Validate the JSON before touching any per-frame stream.
        encoded = json.dumps(observation, allow_nan=False, separators=(',', ':'))+'\n'
        self.files['result.txt'].write(' '.join(f'{v:.12f}' for v in p[:3].ravel())+'\n')
        self.csv.writerow([i, f'{p[0, 3]:.12f}', f'{p[1, 3]:.12f}', f'{np.arctan2(p[1, 0], p[0, 0]):.12f}'])
        self.timing.writerow([i, frame.original_idx, frame.timestamp_ns, self.backend,
                              f'{load_ms:.9f}', f'{registration_ms:.9f}', f'{total_ms:.9f}',
                              str(result.map_accepted).lower(), f'{result.estimate.scale:.12f}'])
        self.frames.writerow([i, frame.timestamp_ns, frame.path.name])
        self.files['observations.jsonl'].write(encoded)
        for stream in self.files.values():
            stream.flush()
        self.count += 1
        self.last_ns = frame.timestamp_ns

    def __exit__(self, *args):
        return self.stack.__exit__(*args)


def validate_output(run, expected, *, backend=BACKEND):
    """Cross-check the serialized sources of every output field."""
    run = Path(run)
    def records(name):
        with (run/name).open() as stream:
            reader = csv.DictReader(stream)
            return reader.fieldnames, list(reader)
    header, rows = records('diagnostic.csv')
    timing_header, timing = records('timings.csv')
    _, frames = records('frames.csv')
    obs = [json.loads(v) for v in (run/'observations.jsonl').read_text().splitlines()]
    matrices = np.loadtxt(run/'result.txt', ndmin=2)
    errors = []
    if header != ['idx', 'x', 'y', 'yaw'] or timing_header != TIMING_HEADER:
        errors.append('fixed output header mismatch')
    if matrices.shape != (expected, 12) or not np.isfinite(matrices).all():
        errors.append('invalid matrix count/shape/finite values')
    for name, values in [('CSV', rows), ('timings', timing), ('frames', frames), ('observations', obs)]:
        if len(values) != expected or [int(v['idx']) for v in values] != list(range(expected)):
            errors.append(name+' count/idx mismatch')
    if errors:
        return errors
    previous_ns = None
    for i, (row, t, f, o, flat) in enumerate(zip(rows, timing, frames, obs, matrices)):
        pose = np.eye(4)
        pose[:3] = flat.reshape(3, 4)
        try:
            Similarity(pose, float(t['scale']))
        except ValueError:
            errors.append(f'invalid pose/scale at {i}')
        expected_row = [pose[0, 3], pose[1, 3], np.arctan2(pose[1, 0], pose[0, 0])]
        if not np.allclose([float(row[k]) for k in ('x', 'y', 'yaw')], expected_row, atol=1e-8, rtol=0):
            errors.append(f'CSV/matrix mismatch at {i}')
        ns = int(f['timestamp_ns'])
        if (ns <= 0 or (previous_ns is not None and ns <= previous_ns) or int(t['timestamp_ns']) != ns
                or o['timestamp_ns'] != ns or o['input_cutoff_ns'] != ns or int(t['original_idx']) != i
                or o['original_idx'] != i or o['imu_used'] is not False or o['gnss_used'] is not False):
            errors.append(f'causal frame/input contract mismatch at {i}')
        if (t['backend'] != backend or t['map_accepted'] != str(o['accepted']).lower()
                or not np.isclose(float(t['scale']), o['scale'], rtol=0, atol=1e-10)
                or not np.allclose(pose, o['pose'], rtol=0, atol=1e-10)):
            errors.append(f'observation/timing/pose mismatch at {i}')
        if not all(np.isfinite(float(t[k])) and float(t[k]) >= 0 for k in ('load_ms', 'registration_ms', 'total_ms')):
            errors.append(f'invalid timings at {i}')
        previous_ns = ns
    return errors

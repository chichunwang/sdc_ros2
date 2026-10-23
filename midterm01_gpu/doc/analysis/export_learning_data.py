#!/usr/bin/env python3
"""Export saved localization diagnostics to a small, read-only-source MAT bundle.

No registration, optimization, raw bag/PLY reading, parameter changes or scoring.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
from zoneinfo import ZoneInfo

import numpy as np
from scipy.io import savemat
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def finite_array(value, shape, name):
    value = np.asarray(value, dtype=float)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f'{name}: expected finite {shape}, got {value.shape}')
    return value


def timestamps(table):
    # Do not parse nanoseconds through float: subtract int64 before seconds.
    return np.array([int(row['timestamp_ns']) for row in table], dtype=np.int64)


def optional_number(observation, key):
    value = observation.get(key)
    return np.nan if value is None else float(value)


def export_bundle(run, reference, output):
    run, reference, output = Path(run).resolve(), Path(reference).resolve(), Path(output).resolve()
    if not any(output.is_relative_to(parent) for parent in (HERE, ROOT/'outputs/run')):
        raise ValueError('Write teaching assets under doc/analysis or tests under outputs/run')
    sidecar = output.with_suffix('.json')
    if output.exists() or sidecar.exists():
        raise FileExistsError(f'Preserve existing bundle; use another --output: {output}')
    names = ['frames.csv', 'diagnostic.csv', 'sensor_prior.csv', 'timings.csv',
             'observations.jsonl', 'graph_report.json', 'graph_states.npz',
             'raw_sensors.npz', 'params.json', 'validation.json']
    inputs = [run/name for name in names]
    inputs += [reference/name for name in ('frames.csv', 'diagnostic.csv', 'validation.json', 'test_info.json')]
    identities = {str(path): digest(path) for path in inputs}
    validation = json.loads((run/'validation.json').read_text())
    ref_validation = json.loads((reference/'validation.json').read_text())
    if any(v.get('returncode') != 0 or not v.get('output_complete_and_finite')
           for v in (validation, ref_validation)):
        raise ValueError('Use completed runs that passed finite-output validation')
    if validation['track'] != ref_validation['track']:
        raise ValueError('Reference must be the same Track')
    frames = rows(run/'frames.csv'); ref_frames = rows(reference/'frames.csv')
    stamps = timestamps(frames); n = len(frames)
    if n < 2 or np.any(np.diff(stamps) <= 0):
        raise ValueError('Require at least two strictly increasing frame timestamps')
    if not np.array_equal(stamps, timestamps(ref_frames)):
        raise ValueError('Reference frame timestamps/order differ; no guessed alignment')
    if [int(row['idx']) for row in frames] != list(range(n)):
        raise ValueError('Frame local indices are not contiguous')
    trajectory, old = rows(run/'diagnostic.csv'), rows(reference/'diagnostic.csv')
    prior, timing = rows(run/'sensor_prior.csv'), rows(run/'timings.csv')
    observations = [json.loads(line) for line in (run/'observations.jsonl').read_text().splitlines()]
    if any(len(table) != n for table in (trajectory, old, prior, timing, observations)):
        raise ValueError('Frame/prior/diagnostic/timing/observation counts differ')
    for table in (trajectory, old, timing, observations):
        if [int(row['idx']) for row in table] != list(range(n)):
            raise ValueError('Output local indices/order differ')
    if any(not np.array_equal(timestamps(table), stamps) for table in (prior, timing, observations)):
        raise ValueError('Integer timestamp identities differ')
    if digest(run/'diagnostic.csv') != validation['csv_sha256']:
        raise ValueError('Saved trajectory SHA differs from validation')
    ref_info = json.loads((reference/'test_info.json').read_text())
    ref_sha = ref_validation.get('csv_sha256', ref_info.get('scored_csv_sha256'))
    if not ref_sha or digest(reference/'diagnostic.csv') != ref_sha:
        raise ValueError('Reference trajectory SHA differs from saved local identity')
    original = np.array([int(row['original_idx']) for row in observations], np.int64)
    if not np.array_equal(original, [int(row['original_idx']) for row in timing]):
        raise ValueError('Timing/observation original indices differ')
    accepted = np.array([bool(row['accepted']) for row in observations])
    if [row['map_accepted'].lower() == 'true' for row in timing] != accepted.tolist():
        raise ValueError('Timing/observation accepted flags differ')
    with np.load(run/'graph_states.npz', allow_pickle=False) as graph:
        if not np.array_equal(graph['frame_time_ns'], stamps):
            raise ValueError('Graph frame timestamp identities differ')
        if 'states' not in graph:
            raise ValueError('This lesson requires an executed slow-bias graph')
        origin = np.int64(graph['time_origin_ns'])
        node_time = graph['node_time_s'].copy()
        states = finite_array(graph['states'], (len(node_time), 8), 'graph states')
        temporal = graph['temporal_map_mask'].copy()
        discrepancy = graph['adjacent_discrepancy_m'].copy()
    with np.load(run/'raw_sensors.npz', allow_pickle=False) as raw:
        gnss_ns = raw['gnss_bag_ns'].copy()
        gnss_xy = finite_array(raw['gnss_xyz'][:, :2], (len(gnss_ns), 2), 'GNSS XY')
    if gnss_ns.dtype != np.int64 or origin != gnss_ns[0] or np.any(np.diff(gnss_ns) <= 0):
        raise ValueError('Unexpected GNSS integer time origin/order')
    # The graph may exclude GNSS samples outside IMU coverage. Match by exact
    # reconstructed timestamps, rather than assuming equal node/sample counts.
    node_ns = origin + np.rint(node_time*1e9).astype(np.int64)
    if not np.isin(node_ns, gnss_ns).all() or np.any(np.diff(node_ns) <= 0):
        raise ValueError('Graph nodes do not identify original GNSS timestamps')
    time = (stamps-origin)/1e9; gnss_time = (gnss_ns-origin)/1e9
    covered = (stamps >= gnss_ns[0]) & (stamps <= gnss_ns[-1])
    gnss_at_frame = np.column_stack([np.interp(time, gnss_time, gnss_xy[:, a]) for a in range(2)])
    gnss_at_frame[~covered] = np.nan  # No clamped/extrapolated edge treated as measurement.
    report = json.loads((run/'graph_report.json').read_text())
    if not report['joint_estimation_executed'] or not report['applied']:
        raise ValueError('This lesson requires an applied slow-bias graph')
    anchors = report['slow_anchors']
    config = json.loads((run/'params.json').read_text())
    matrix = finite_array([row['pose'] for row in observations], (n, 4, 4), 'map candidate poses')
    latest = finite_array([[float(r[k]) for k in ('x', 'y', 'yaw')] for r in trajectory], (n, 3), 'latest')
    previous = finite_array([[float(r[k]) for k in ('x', 'y', 'yaw')] for r in old], (n, 3), 'reference')
    prior_xy = finite_array([[float(r[k]) for k in ('x', 'y')] for r in prior], (n, 2), 'prior XY')
    quat = finite_array([[float(r[k]) for k in ('qx', 'qy', 'qz', 'qw')] for r in prior], (n, 4), 'quaternion')
    data = {
        'schema': 'sdc_learning_v1', 'track': validation['track'],
        'run_name': run.name, 'reference_name': reference.name,
        'time_origin_ns': origin, 'frame_ns': stamps, 'original_idx': original, 'time_s': time,
        'gnss_time_s': gnss_time, 'gnss_ns': gnss_ns, 'gnss_xy': gnss_xy,
        'gnss_xy_at_frames': gnss_at_frame, 'gnss_frame_covered': covered,
        'prior_xy': prior_xy, 'prior_yaw_rad': Rotation.from_quat(quat).as_euler('xyz')[:, 2],
        'latest_xy': latest[:, :2], 'latest_yaw_rad': latest[:, 2],
        'reference_xy': previous[:, :2], 'reference_yaw_rad': previous[:, 2],
        'map_xy': matrix[:, :2, 3], 'map_yaw_rad': np.arctan2(matrix[:, 1, 0], matrix[:, 0, 0]),
        'scale': finite_array([row['scale'] for row in observations], (n,), 'scale'),
        'accepted': accepted, 'temporal_ok': temporal,
        'heldout_median_m': np.array([optional_number(r, 'heldout_median_m') for r in observations]),
        'adjacent_discrepancy_m': discrepancy,
        'adjacent_gate_m': report['config']['displacement_gate_m']
                           +report['config']['displacement_gate_mps']*np.diff(time),
        'scale_bounds': np.array([config['model']['min_scale'], config['model']['max_scale']]),
        'heldout_limit_m': config['model']['holdout_median_m'],
        'load_ms': np.array([float(r['load_ms']) for r in timing]),
        'registration_ms': np.array([float(r['registration_ms']) for r in timing]),
        'total_ms': np.array([float(r['total_ms']) for r in timing]),
        'node_time_s': node_time, 'node_ns': node_ns, 'states': states,
        'state_columns': np.array(['x', 'y', 'vx', 'vy', 'yaw', 'gyro_bias', 'gnss_bias_x', 'gnss_bias_y'], object),
        'anchor_time_s': np.array([a['time_s'] for a in anchors]),
        'anchor_bias_xy': np.array([a['bias_xy'] for a in anchors]).reshape(-1, 2),
        'anchor_yaw_rad': np.array([a['yaw'] for a in anchors]),
        'notes': 'Real GNSS/map/priors/reference are not ground truth. Differences are not ATE. Bias states are model estimates.',
    }
    if temporal.shape != (n,) or discrepancy.shape != (n-1,):
        raise ValueError('Graph diagnostic lengths differ')
    if any(digest(Path(path)) != sha for path, sha in identities.items()):
        raise ValueError('An input changed during export')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix='.mat', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        savemat(temporary, {'sdc': data}, do_compression=True, oned_as='column', appendmat=False)
        # Same-directory hard link publishes atomically and refuses overwrite.
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    manifest = {
        'schema': 'sdc_learning_v1', 'created_at': datetime.now(ZoneInfo('Asia/Taipei')).isoformat(),
        'track': validation['track'], 'frames': n, 'run': str(run), 'reference': str(reference),
        'source_sha256': identities, 'mat_sha256': digest(output),
        'exporter_sha256': digest(__file__), 'localization_rerun': False,
        'timestamp_policy': 'int64 nanoseconds preserved; subtract origin before float seconds',
        'edge_policy': 'GNSS interpolation outside raw GNSS coverage is NaN, never a clamped measurement',
        'limits': data['notes'], 'matlab_execution_verified': False,
    }
    with sidecar.open('x') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2); stream.write('\n')
    return output, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'outputs/run/track02_full_2610212125')
    parser.add_argument('--reference-run', type=Path, default=ROOT/'outputs/run/track02_full_2610191757')
    parser.add_argument('--output', type=Path, default=HERE/'data/track02_learning.mat')
    args = parser.parse_args()
    output, manifest = export_bundle(args.run, args.reference_run, args.output)
    print(f'MAT: {output}\nManifest: {output.with_suffix(".json")}\nFrames: {manifest["frames"]}')
    print('Read saved diagnostics only; no localization or parameter changes.')


if __name__ == '__main__':
    main()

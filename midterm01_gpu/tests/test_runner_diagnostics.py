"""CPU-only runner reporting checks with generated inputs and mocked registration."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


def test_partial_failure_keeps_progress_and_decisions(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / 'scripts/run_track3.py'
    spec = importlib.util.spec_from_file_location('run_track3_diagnostics', script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    data = tmp_path / 'generated_data'
    data.mkdir()
    np.savetxt(data / 'init_guess.txt', np.eye(4), delimiter=',')
    for name in ('map.pcd', 'Track3.db3', '0.ply', '1.ply', '2.ply', '3.ply'):
        (data / name).write_bytes(b'generated reporting fixture')
    stamp = 1727239790258245070
    rows = [dict(original_idx=i, timestamp_ns=stamp+i*100_000_000,
                 path=data/f'{i}.ply') for i in range(4)]
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(dict(
        version='SYNTHETIC_REPORTING_ONLY', data_dir=str(data), ply_subdir='raw_ply',
        expected_frames=4, map_leaf_m=.35, registration={}, filter={}, initial_scale=3.,
    )))
    run = tmp_path / 'generated_run'
    run.mkdir()
    (run / 'source_manifest.json').write_text('{}')

    class Imu:
        invalid = 0
        closed = False
        def __init__(self, path):
            pass
        def until(self, timestamp):
            return [(timestamp, timestamp, .1)]
        def close(self):
            self.closed = True

    class Fusion:
        used_imu = 0
        stale_steps = 0
        updates = 0
        def __init__(self, T, timestamp, cfg):
            self.T = T.copy()
        def predict(self, timestamp, samples):
            self.used_imu += len(samples)
            return self.T.copy()
        def update(self, measurement, initialize=False):
            self.updates += 1
            return (True, .5) if self.updates == 1 else (False, 42.)

    stages = [dict(voxel_m=1., converged=True, stop_reason='step_small')]
    diagnostics = dict(local_points=100, populated_voxels=10)
    class Registration:
        calls = 0
        def __init__(self, points, cfg):
            pass
        def align(self, points, prior, scale, initialize=False):
            self.calls += 1
            if self.calls == 4:
                raise RuntimeError('generated registration failure')
            if self.calls == 1:
                return dict(T=prior, scale=scale, accepted=False, converged=False,
                            reason='insufficient_correspondences', stages=[],
                            model_diagnostics=diagnostics, coverage=.01, cost=None,
                            observable=False, normalized_min_eigenvalue=None)
            return dict(T=prior, scale=scale, accepted=True, converged=True,
                        reason='accepted', stages=stages, model_diagnostics=diagnostics,
                        coverage=.9, cost=.01, observable=True,
                        normalized_min_eigenvalue=.1)

    cp = SimpleNamespace(
        __version__='reporting-fixture', float64=np.float64, arange=np.arange,
        sum=lambda x: SimpleNamespace(get=lambda: None),
        cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceProperties=lambda n: dict(
            name=b'no GPU used', major=0, minor=0))),
    )
    monkeypatch.setitem(sys.modules, 'track3.gpu', SimpleNamespace(
        cp=cp, sync=lambda: None, Registration=Registration))
    monkeypatch.setattr(runner, 'frames', lambda *a, **k: rows)
    monkeypatch.setattr(runner, 'load_map', lambda *a: (np.zeros((10, 3)), 10))
    monkeypatch.setattr(runner, 'load_ply', lambda *a: np.zeros((10, 3)))
    monkeypatch.setattr(runner, 'ImuStream', Imu)
    monkeypatch.setattr(runner, 'Fusion', Fusion)
    monkeypatch.setattr(sys, 'argv', [str(script), '--config', str(config),
                                    '--run-dir', str(run), '--max-frames', '4'])
    assert runner.main() == 1
    info = json.loads((run / 'test_info.json').read_text())
    assert info['status'] == 'failed'
    assert info['frame_count'] == 3 and info['map_accepted_frames'] == 1
    assert info['consecutive_map_rejections'] == info['max_consecutive_map_rejections'] == 1
    assert info['last_map_accepted_idx'] == 1
    assert info['last_map_accepted_timestamp_ns'] == stamp+100_000_000
    assert info['used_imu_samples'] == 4
    assert info['registration_seconds'] >= 0
    assert info['localization_seconds'] >= 0
    assert set(info['output_sha256']) == set(runner.OUTPUT_FILES)
    for name, digest in info['output_sha256'].items():
        assert digest == hashlib.sha256((run / name).read_bytes()).hexdigest()
    observations = [json.loads(s) for s in (run / 'observations.jsonl').read_text().splitlines()]
    assert [o['filter_decision'] for o in observations] == ['not_attempted', 'accepted', 'nis_rejected']
    assert [o['registration_reason'] for o in observations] == ['insufficient_correspondences', 'accepted', 'accepted']
    assert [o['consecutive_map_rejections'] for o in observations] == [1, 0, 1]
    assert observations[0]['map_age_seconds'] is None
    assert observations[1]['map_age_seconds'] == 0
    assert abs(observations[2]['map_age_seconds']-.1) < 1e-12
    assert observations[2]['registration_converged'] is True
    assert observations[2]['registration_stages'] == stages
    assert observations[2]['model_diagnostics'] == diagnostics

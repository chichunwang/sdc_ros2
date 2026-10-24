"""CPU-only runner reporting checks with generated inputs and mocked registration."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize('nonfinite_candidate', [False, True])
def test_partial_failure_keeps_progress_and_decisions(tmp_path, monkeypatch, nonfinite_candidate):
    script = Path(__file__).resolve().parents[1] / 'scripts/run_track3.py'
    spec = importlib.util.spec_from_file_location('run_track3_diagnostics', script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    data = tmp_path / 'generated_data'
    data.mkdir()
    np.savetxt(data / 'init_guess.txt', np.eye(4), delimiter=',')
    for name in ('map.pcd', '0.ply', '1.ply', '2.ply', '3.ply'):
        (data / name).write_bytes(b'generated reporting fixture')
    stamp = 1727239790258245070
    rows = [dict(original_idx=i, timestamp_ns=stamp+i*100_000_000,
                 path=data/f'{i}.ply') for i in range(4)]
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(dict(
        version='SYNTHETIC_REPORTING_ONLY', data_dir=str(data), ply_subdir='raw_ply',
        expected_frames=4, map_leaf_m=.35, registration={'scale_bounds':[.2,30.]},
        prediction='previous_pose', initial_scale=3.,
    )))
    run = tmp_path / 'generated_run'
    run.mkdir()
    (run / 'source_manifest.json').write_text('{}')

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
            candidate=prior.copy(); candidate[0,3]=2. if self.calls==2 else 100.
            if self.calls==3 and nonfinite_candidate:
                candidate[0,3]=np.nan
            return dict(T=candidate, scale=3.1 if self.calls==2 else 9.,
                        accepted=self.calls==2 or nonfinite_candidate, converged=self.calls==2,
                        reason='accepted' if self.calls==2 else 'not_converged:backtracking_failed',
                        stages=stages, model_diagnostics=diagnostics,
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
    monkeypatch.setattr(sys, 'argv', [str(script), '--config', str(config),
                                    '--run-dir', str(run), '--max-frames', '4'])
    assert runner.main() == 1
    info = json.loads((run / 'test_info.json').read_text())
    assert info['status'] == 'failed'
    assert info['frame_count'] == 3 and info['map_accepted_frames'] == 1
    assert info['consecutive_map_rejections'] == info['max_consecutive_map_rejections'] == 1
    assert info['last_map_accepted_idx'] == 1
    assert info['last_map_accepted_timestamp_ns'] == stamp+100_000_000
    assert info['imu_used'] is False
    assert not (data/'Track3.db3').exists()
    assert info['registration_seconds'] >= 0
    assert info['localization_seconds'] >= 0
    assert set(info['output_sha256']) == set(runner.OUTPUT_FILES)
    for name, digest in info['output_sha256'].items():
        assert digest == hashlib.sha256((run / name).read_bytes()).hexdigest()
    observations = [json.loads(s) for s in (run / 'observations.jsonl').read_text().splitlines()]
    assert [o['pose_decision'] for o in observations] == ['retained_prior', 'accepted_registration', 'retained_prior']
    third_reason='invalid_candidate_pose_or_scale' if nonfinite_candidate else 'not_converged:backtracking_failed'
    assert [o['registration_reason'] for o in observations] == ['insufficient_correspondences', 'accepted', third_reason]
    assert [o['consecutive_map_rejections'] for o in observations] == [1, 0, 1]
    assert observations[0]['map_age_seconds'] is None
    assert observations[1]['map_age_seconds'] == 0
    assert abs(observations[2]['map_age_seconds']-.1) < 1e-12
    assert observations[2]['registration_converged'] is False
    assert observations[2]['registration_stages'] == stages
    assert observations[2]['model_diagnostics'] == diagnostics

    trace=[json.loads(line) for line in (run/'pose_trace.jsonl').read_text().splitlines()]
    np.testing.assert_array_equal(trace[1]['registered_T'],trace[1]['final_T'])
    np.testing.assert_array_equal(trace[2]['predicted_T'],trace[2]['final_T'])
    assert trace[2]['final_T'][0][3]==2.
    if nonfinite_candidate:assert trace[2]['registered_T'] is None
    else:assert trace[2]['registered_T'][0][3]==100.
    assert trace[2]['final_scale']==3.1 and trace[2]['proposed_scale']==9.

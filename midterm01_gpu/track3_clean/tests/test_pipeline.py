"""Known-truth numerical and causal contracts; never starts real localization."""
import ast
from copy import deepcopy
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from track3_clean import cli
from track3_clean.config import RegistrationConfig, TrackingConfig
from track3_clean.contracts import BootstrapUnsupported, Frame, GeometryUnavailable, Similarity, TrackingState, checked_timestamp
from track3_clean.cuda_grid import _normal_eigh
from track3_clean.data import digest, load_ply_xyz, read_initial, sample_source, select_frames
from track3_clean.map_cache import cache_xyz_map
from track3_clean.engine import LocalizationEngine
from track3_clean.geometry import Geometry, block_validation, point_jacobian
from track3_clean.output import CPU_REFERENCE_BACKEND, OutputWriter, validate_output
from track3_clean.quality import decide
from track3_clean.registration import MapRegistrar
from track3_clean.state import TrackingMemory

TEST_ROOT = Path(os.environ['TRACK3_TEST_ROOT']).resolve()


def candidate(*, x=0., cost=.1, valid=True, seed=1.):
    p = np.eye(4)
    p[0, 3] = x
    return dict(pose=p.tolist(), scale=3., seed_scale=seed, seed_pose=np.eye(4).tolist(),
                training_cost=cost, checks=dict(converged=True, interior=True, observable=True,
                coverage=valid, validation=valid), reasons=[] if valid else ['validation'],
                training=dict(coverage=.9, median_m=.1, p90_m=.3),
                ray_depth=dict(available=True, training=dict(support_fraction=.9, median_abs_log_depth_ratio=.02),
                               validation=dict(support_fraction=.9, median_abs_log_depth_ratio=.02)))


def room():
    rng = np.random.default_rng(123)
    n = 2200
    floor = np.column_stack([rng.uniform(1, 13, n), rng.uniform(-5, 5, n), np.zeros(n)])
    wall = np.column_stack([np.full(n, 13.), rng.uniform(-5, 5, n), rng.uniform(0, 5, n)])
    side = np.column_stack([rng.uniform(1, 13, n), np.full(n, 5.), rng.uniform(0, 5, n)])
    other = np.column_stack([np.full(n, 1.), rng.uniform(-5, 5, n), rng.uniform(0, 5, n)])
    return np.vstack([floor, wall, side, other])+rng.normal(0, .005, (4*n, 3))


class Contracts(unittest.TestCase):
    def test_pose_scale_local_composition(self):
        p = np.eye(4)
        p[:3, :3] = Rotation.from_euler('xyz', [.1, .2, .4]).as_matrix()
        p[:3, 3] = [660., -222., 10.]
        previous = Similarity(p, 6.)
        local = np.eye(4)
        local[:3, :3] = Rotation.from_euler('z', .02).as_matrix()
        local[:3, 3] = [.2, -.1, .03]
        current_local = Similarity(local, 6.1)
        result = current_local.in_map(previous)
        raw = np.random.default_rng(1).normal(size=(30, 3))
        expected = current_local.transform(raw) @ p[:3, :3].T+p[:3, 3]
        np.testing.assert_allclose(result.transform(raw), expected, atol=1e-12)
        self.assertEqual(result.scale, 6.1)
        self.assertFalse(result.pose.flags.writeable)

    def test_reject_bad_pose_scale_and_timestamps(self):
        for scale in (0, -1, np.nan, True):
            with self.assertRaises(ValueError):
                Similarity(np.eye(4), scale)
        p = np.eye(4); p[0, 0] = -1
        with self.assertRaises(ValueError): Similarity(p, 1.)
        for ns in (0, -1, 1.1, True):
            with self.assertRaises(ValueError): checked_timestamp(ns)

    def test_no_bootstrap_anchor_no_output(self):
        m = TrackingMemory(Similarity(np.eye(4), 3.))
        with self.assertRaises(RuntimeError): m.commit(np.ones((8, 3)), 100)
        self.assertIsNone(m.last_output_ns)

    def test_unsupported_cloud_does_not_replace_reference(self):
        m = TrackingMemory(Similarity(np.eye(4), 3.))
        raw = np.ones((8, 3))
        first = m.commit(raw, 1_000_000_000, map_estimate=m.published)
        m.commit(raw*2, 1_100_000_000)
        self.assertEqual(m.reference.timestamp_ns, 1_000_000_000)
        self.assertEqual(m.last_output_ns, 1_100_000_000)
        lost = m.commit(raw*3, 3_100_000_000)
        self.assertEqual(lost.state, TrackingState.LOST)
        self.assertEqual(lost.source, 'held')
        self.assertEqual(first.state, TrackingState.TRACKING)
        recovered = m.commit(raw, 3_200_000_000, map_estimate=m.published)
        self.assertEqual(recovered.state, TrackingState.TRACKING)
        self.assertEqual(first.timestamp_ns, 1_000_000_000)
        with self.assertRaises(ValueError): m.commit(raw, 3_200_000_000)

    def test_pair_support_does_not_reset_map_age(self):
        m = TrackingMemory(Similarity(np.eye(4), 3.))
        raw = np.ones((8, 3))
        m.commit(raw, 1_000_000_000, map_estimate=m.published)
        result = m.commit(raw, 4_000_000_000, pair_estimate=m.published)
        self.assertEqual(result.state, TrackingState.LOST)
        self.assertEqual(result.unsupported_seconds, 3.)
        self.assertEqual(m.reference.timestamp_ns, 4_000_000_000)
        self.assertEqual(m.last_map.timestamp_ns, 1_000_000_000)


class Quality(unittest.TestCase):
    def test_heldout_veto_does_not_select_another_seed(self):
        a, b = candidate(cost=.1, valid=False), candidate(cost=.12, seed=1.05)
        result = decide([a, b])
        self.assertIs(result.selected, a)
        self.assertFalse(result.accepted)
        self.assertIn('validation', result.reasons)

    def test_failed_low_cost_does_not_hide_supported_fit(self):
        bad = candidate(cost=.001); bad['checks']['converged'] = False
        good = candidate(seed=.95); other = candidate(seed=1.05)
        result = decide([bad, good, other])
        self.assertTrue(result.accepted)
        self.assertIs(result.selected, good)
        self.assertEqual(result.diagnostics['candidate_rejection_counts']['converged'], 1)

    def test_coarse_competing_basin_rejected(self):
        result = decide([candidate(seed=.95), candidate(seed=1.05), candidate(x=2., seed=9.)])
        self.assertFalse(result.accepted)
        self.assertIn('ambiguous_training_basins', result.reasons)

    def test_identical_seed_cannot_supply_two_votes(self):
        a = candidate()
        result = decide([a, deepcopy(a)])
        self.assertFalse(result.accepted)
        self.assertIn('insufficient_independent_initializations', result.reasons)

    def test_full_rejection_preserves_reason_counts(self):
        a = candidate(); a['checks']['observable'] = False
        b = candidate(); b['ray_depth']['available'] = False
        result = decide([a, b])
        self.assertFalse(result.accepted)
        self.assertIsNone(result.selected)
        self.assertEqual(result.diagnostics['candidate_rejection_counts'], {'observable': 1, 'training_visibility': 1})


class Numerical(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.points = room()
        cls.geometry = Geometry('cpu_reference')
        cls.scene = MapRegistrar(cls.points, backend='cpu_reference', geometry=cls.geometry)

    def test_analytic_jacobian(self):
        raw = np.random.default_rng(2).normal(size=(21, 3))
        r = Rotation.from_euler('xyz', [.1, -.2, .3]).as_matrix()
        rotated = 4.5*raw @ r.T
        jac = point_jacobian(torch.tensor(rotated)).numpy()
        eps = 1e-6
        for k in range(7):
            step = np.eye(7)[k]*eps
            q = 4.5*np.exp(step[6])*raw @ (Rotation.from_rotvec(step[3:6]).as_matrix() @ r).T+step[:3]
            np.testing.assert_allclose((q-rotated)/eps, jac[:, :, k], atol=2e-5)

    def test_known_sim3_recovery(self):
        truth = np.eye(4)
        truth[:3, :3] = Rotation.from_euler('xyz', [4, -2, 8], degrees=True).as_matrix()
        truth[:3, 3] = [.2, -.3, .5]
        raw = (self.points[::2]-truth[:3, 3]) @ truth[:3, :3]/4.5
        seed = truth.copy(); seed[:3, 3] += [.4, -.3, .2]
        result = self.scene.fit(raw, seed, 4.8)
        self.assertTrue(result['accepted'], result['reasons'])
        np.testing.assert_allclose(result['pose'], truth, atol=.025)
        self.assertLess(abs(result['scale']/4.5-1), .005)

    def test_pair_recovers_changing_scale_in_camera_metres(self):
        previous_scale, current_scale = 6., 6.2
        previous_raw = self.points/previous_scale
        local = np.eye(4)
        local[:3, :3] = Rotation.from_euler('z', .01).as_matrix()
        local[:3, 3] = [.15, -.03, .02]
        current_raw = (self.points-local[:3, 3]) @ local[:3, :3]/current_scale
        target = MapRegistrar(previous_scale*previous_raw, backend='cpu_reference', geometry=self.geometry)
        fit = target.fit(current_raw[::2], np.eye(4), previous_scale)
        self.assertTrue(fit['accepted'], fit['reasons'])
        self.assertLess(abs(fit['scale']/current_scale-1), .005)
        np.testing.assert_allclose(fit['pose'], local, atol=.025)

    def test_plane_does_not_gain_fake_observability(self):
        plane = self.points[:2200]
        scene = MapRegistrar(plane, backend='cpu_reference', geometry=self.geometry)
        result = scene.fit(plane/3., np.eye(4), 3., use_ndt=False)
        self.assertFalse(result['accepted'])
        self.assertLess(result['physical_rank'], 7)

    def test_no_overlap_rejected(self):
        result = self.scene.fit(self.points[::2]/3.+100, np.eye(4), 3.)
        self.assertFalse(result['accepted'])

    def test_config_view_does_not_mutate_map(self):
        original = self.scene.config
        view = self.scene.with_config(replace(original, translation_domain_m=2.))
        self.assertIs(self.scene.config, original)
        self.assertIs(view.target, self.scene.target)
        self.assertEqual(view.config.translation_domain_m, 2.)

    def test_bounded_pca_batches(self):
        real = torch.linalg.eigh
        sizes = []
        def bounded(value):
            sizes.append(len(value))
            return real(value)
        with patch('torch.linalg.eigh', side_effect=bounded):
            values, vectors = _normal_eigh(torch.eye(3, dtype=torch.float64).repeat(777, 1, 1))
        self.assertEqual(sizes, [256, 256, 256, 9])
        self.assertEqual(tuple(vectors.shape), (777, 3, 3))

    def test_source_voxels_keep_original_first_rows(self):
        raw = np.array([[1.01, .1, 0.], [1.02, .1, 0.], [2., .2, 0.], [2.01, .21, 0.]])
        sampled = sample_source(raw, self.geometry, leaf=.1, max_points=10)
        np.testing.assert_array_equal(sampled, raw[[0, 2]])


class IOAndRuntime(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEST_ROOT)
        self.path = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def write_sequence(self, count, path):
        path.mkdir()
        memory = TrackingMemory(Similarity(np.eye(4), 3.))
        with OutputWriter(path) as writer:
            for i in range(count):
                frame = Frame(i, 1000+i, Path(f'{i}_ts_{1000+i}.ply'))
                p = np.eye(4); p[0, 3] = i*.1
                result = memory.commit(np.ones((8, 3)), frame.timestamp_ns, map_estimate=Similarity(p, 3.+i*.01))
                writer.append(frame, result, load_ms=1., registration_ms=2., total_ms=3., elapsed_seconds=.003)

    def test_fixed_io_and_causal_prefix(self):
        a, b = self.path/'three', self.path/'five'
        self.write_sequence(3, a); self.write_sequence(5, b)
        self.assertEqual(validate_output(a, 3), [])
        self.assertEqual(validate_output(b, 5), [])
        for name in ('result.txt', 'diagnostic.csv', 'frames.csv'):
            self.assertTrue((b/name).read_bytes().startswith((a/name).read_bytes()), name)

    def test_corrupted_pose_observation_rejected(self):
        p = self.path/'corrupt'; self.write_sequence(3, p)
        text = (p/'observations.jsonl').read_text().splitlines()
        row = json.loads(text[1]); row['scale'] = 9.
        text[1] = json.dumps(row)
        (p/'observations.jsonl').write_text('\n'.join(text)+'\n')
        self.assertIn('observation/timing/pose mismatch at 1', validate_output(p, 3))

    def test_strict_ply_parser_and_frame_timestamps(self):
        raw = self.path/'Track3/raw_ply'; raw.mkdir(parents=True)
        p = raw/'0_ts_1000.ply'
        header = 'ply\nformat ascii 1.0\nelement vertex 8\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'
        p.write_text(header+('1 2 3 0 0 0\n'*8))
        np.testing.assert_array_equal(load_ply_xyz(p), np.tile([1, 2, 3], (8, 1)))
        self.assertEqual(select_frames(self.path, count=1)[0].timestamp_ns, 1000)
        (raw/'1_ts_999.ply').write_bytes(p.read_bytes())
        with self.assertRaises(ValueError): select_frames(self.path, count=1)
        p.write_text(header+('1 2 nan 0 0 0\n'*8))
        with self.assertRaises(ValueError): load_ply_xyz(p)

    def test_dryrun_no_cuda_and_active_run_guard(self):
        args = ['--full', '--data-root', str(cli.ROOT.parent/'midterm01/data')]
        with patch('track3_clean.cli.configure_cuda', side_effect=AssertionError('CUDA touched')), \
             patch('track3_clean.cli.active_localizations', return_value=[7688]), \
             patch('sys.stdout', new_callable=io.StringIO) as out, \
             patch('sys.stderr', new_callable=io.StringIO):
            self.assertEqual(cli.main(args+['--dry-run']), 0)
            self.assertEqual(json.loads(out.getvalue())['frames'], 3142)
            self.assertEqual(cli.main(args), 1)

    def test_runtime_failure_records_no_success(self):
        run = self.path/'failure'; run.mkdir()
        args = ['--frames', '1', '--data-root', str(cli.ROOT.parent/'midterm01/data')]
        with patch('track3_clean.cli.active_localizations', return_value=[]), \
             patch('track3_clean.cli.create_run', return_value=run), \
             patch('track3_clean.cli.snapshot_sources', return_value={}), \
             patch('track3_clean.cli.configure_cuda', side_effect=RuntimeError('injected CUDA error')), \
             patch('track3_clean.cli.cuda_memory', return_value={}), \
             patch('sys.stdout', new_callable=io.StringIO) as out, \
             patch('sys.stderr', new_callable=io.StringIO):
            self.assertEqual(cli.main(args), 1)
        summary = json.loads((run/'validation.json').read_text())
        self.assertEqual(summary['processed_frames'], 0)
        self.assertEqual(summary['failure']['message'], 'injected CUDA error')
        self.assertNotIn('Finished:', out.getvalue())

    def test_package_has_no_legacy_python_imports(self):
        for path in (cli.ROOT/'track3_clean').glob('*.py'):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports = [n.name for n in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    imports = [node.module or '']
                else:
                    continue
                self.assertFalse(any(s.startswith(('run_track3', 'track3_map_only', 'map_scale_icp',
                                                  'causal_track3', 'old_icp', 'test_track3')) for s in imports), path)

    def synthetic_cli_data(self):
        root = self.path/'data'; track = root/'Track3'; raw = track/'raw_ply'
        raw.mkdir(parents=True)
        points = room(); points[:, 2] -= 2.
        np.savetxt(track/'init_guess.txt', np.eye(4))
        n = len(points)
        with (track/'map.pcd').open('w') as stream:
            stream.write(f'VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH {n}\nHEIGHT 1\nPOINTS {n}\nDATA ascii\n')
            np.savetxt(stream, points, fmt='%.9g')
        cache = cache_xyz_map(track/'map.pcd', self.path/'cache')
        header = f'ply\nformat binary_little_endian 1.0\nelement vertex {n}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'
        for i in range(3):
            values = (points-[i*.03, 0., 0.])/(3.+i*.01)
            packed = np.zeros(n, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', 'u1', (3,))])
            for j, k in enumerate('xyz'): packed[k] = values[:, j]
            (raw/f'{i}_ts_{1_000_000_000+i*40_000_000}.ply').write_bytes(header.encode()+packed.tobytes())
        return root, cache

    def test_cli_cpu_synthetic_end_to_end(self):
        root, cache = self.synthetic_cli_data()
        run = self.path/'cli_success'; run.mkdir()
        geometry = Geometry('cpu_reference')
        with patch('track3_clean.cli.active_localizations', return_value=[]), \
             patch('track3_clean.cli.create_run', return_value=run), \
             patch('track3_clean.cli.configure_cuda', return_value={'synthetic_cpu_test': True}), \
             patch('track3_clean.cli.cuda_memory', return_value={}), \
             patch('track3_clean.geometry.Geometry', return_value=geometry), \
             patch('sys.stdout', new_callable=io.StringIO) as out:
            code = cli.main(['--frames', '3', '--data-root', str(root), '--map-cache', str(cache)])
        self.assertEqual(code, 0, (run/'validation.json').read_text())
        summary = json.loads((run/'validation.json').read_text())
        self.assertTrue(summary['output_complete_and_finite'])
        self.assertFalse(summary['submission_frame_set_complete'])
        self.assertEqual(summary['map_accepted_frames'], 3)
        self.assertEqual(validate_output(run, 3, backend=CPU_REFERENCE_BACKEND), [])
        self.assertIn('Graph: executed=false, applied=false, time=0.00 ms', out.getvalue())
        self.assertIn('Finished: frames=3, accepted_map=3, rejected_map=0, total=', out.getvalue())
        self.assertEqual(len((run/'timings.csv').read_text().splitlines()), 4)

    def test_cli_interrupt_keeps_failure_stage_and_partial_evidence(self):
        root, cache = self.synthetic_cli_data()
        run = self.path/'cli_interrupted'; run.mkdir()
        geometry = Geometry('cpu_reference')
        with patch('track3_clean.cli.active_localizations', return_value=[]), \
             patch('track3_clean.cli.create_run', return_value=run), \
             patch('track3_clean.cli.configure_cuda', return_value={'synthetic_cpu_test': True}), \
             patch('track3_clean.cli.cuda_memory', return_value={}), \
             patch('track3_clean.geometry.Geometry', return_value=geometry), \
             patch('track3_clean.engine.LocalizationEngine.step', side_effect=KeyboardInterrupt()), \
             patch('sys.stdout', new_callable=io.StringIO) as out, \
             patch('sys.stderr', new_callable=io.StringIO):
            code = cli.main(['--frames', '3', '--data-root', str(root), '--map-cache', str(cache)])
        self.assertEqual(code, 1)
        summary = json.loads((run/'validation.json').read_text())
        self.assertEqual(summary['failure']['type'], 'KeyboardInterrupt')
        self.assertEqual(summary['failure']['stage'], 'registration')
        self.assertEqual(summary['failure']['failed_original_idx'], 0)
        self.assertNotIn('Finished:', out.getvalue())


class EngineTransitions(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.points = room()
        self.scene = MapRegistrar(self.points, backend='cpu_reference', geometry=Geometry('cpu_reference'))
        self.engine = LocalizationEngine(self.scene, np.eye(4))
        self.raw = self.points/3.
        self.engine.memory.commit(self.raw, 1_000_000_000, map_estimate=self.engine.memory.published)

    def test_expected_pair_geometry_failure_still_attempts_map(self):
        fits = [candidate(seed=.95), candidate(seed=1.05)]
        for fit in fits:
            fit['checks']['observable'] = False
        with patch.object(self.engine, '_pair', side_effect=GeometryUnavailable('no supported voxels')), \
             patch.object(self.engine, '_fit_starts', return_value=fits) as map_fits:
            result = self.engine.step(self.raw, 1_040_000_000)
        self.assertEqual(result.source, 'held')
        self.assertTrue(map_fits.called)
        self.assertEqual(self.engine.memory.reference.timestamp_ns, 1_000_000_000)
        self.assertEqual(result.diagnostics['pair']['reasons'], ['pair_geometry_unavailable'])

    def test_cuda_error_is_not_observation_rejection(self):
        with patch.object(self.engine, '_pair', side_effect=RuntimeError('CUDA failure')):
            with self.assertRaisesRegex(RuntimeError, 'CUDA failure'):
                self.engine.step(self.raw, 1_040_000_000)
        self.assertEqual(self.engine.memory.last_output_ns, 1_000_000_000)

    def test_pair_reference_and_elapsed_survive_unsupported_frame(self):
        fits = [candidate(seed=.95), candidate(seed=1.05)]
        for fit in fits:
            fit['checks']['observable'] = False
        seen = []
        def unavailable(raw, ns, weights):
            seen.append((ns-self.engine.memory.reference.timestamp_ns)/1e9)
            return None, dict(accepted=False, reasons=['injected no overlap'])
        with patch.object(self.engine, '_pair', side_effect=unavailable), \
             patch.object(self.engine, '_fit_starts', return_value=fits):
            self.engine.step(self.raw, 1_040_000_000)
            self.engine.step(self.raw, 1_080_000_000)
        self.assertEqual(seen, [.04, .08])

    def test_bootstrap_failure_retains_candidates_and_reasons(self):
        engine = LocalizationEngine(self.scene, np.eye(4))
        fit = candidate(); fit['checks']['observable'] = False
        with patch.object(engine, '_fit_starts', return_value=[fit]):
            with self.assertRaises(BootstrapUnsupported) as caught:
                engine.step(self.raw, 1_000_000_000)
        detail = caught.exception.diagnostics
        self.assertGreater(detail['map']['decision']['candidate_rejection_counts']['observable'], 0)
        self.assertIsNone(engine.memory.last_output_ns)


if __name__ == '__main__':
    unittest.main()

"""Synthetic CUDA integration and deterministic map sampling; no real poses."""
from pathlib import Path
import contextlib
import csv
import io
import json
import os
import sqlite3
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from causal_scale_registration import CausalScaleRegistrar
import run_track3_causal as runner
from run_track3_causal import gpu_crop_voxel
from test_causal_track3 import room, imu_arrays, NS


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA mechanism test requires a GPU')
class CausalCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from accuracy.benchmark import create_run_directory
        cls.output_run = create_run_directory(Path(__file__).resolve().parents[2]/'outputs/run',
                                       'synthetic_causal_cuda_mechanisms', track=0)

    def test_float64_room_matches_explicit_cpu_reference(self):
        points, _ = room()
        prior = np.eye(4); prior[:3, 3] = [.3, -.2, 1.6]
        raw = (points-[0., 0., 1.5])/3.2
        cpu = CausalScaleRegistrar(points, backend='cpu_reference').register(raw, prior, 3.)
        gpu_registrar = CausalScaleRegistrar(points, cache_dir=self.output_run/'room_cuda')
        gpu = gpu_registrar.register(raw, prior, 3.)
        self.assertTrue(gpu['accepted'], gpu)
        self.assertEqual(cpu['accepted'], gpu['accepted'])
        self.assertEqual(cpu['reasons'], gpu['reasons'])
        np.testing.assert_allclose(gpu['pose'], cpu['pose'], rtol=0, atol=1e-6)
        self.assertAlmostEqual(gpu['scale'], cpu['scale'], places=6)
        np.testing.assert_allclose(gpu['measurement_covariance'], cpu['measurement_covariance'], rtol=1e-6, atol=1e-8)

    def test_ground_is_rejected_on_cuda(self):
        _, ground = room()
        prior = np.eye(4); prior[2, 3] = 1.5
        registrar = CausalScaleRegistrar(ground, cache_dir=self.output_run/'ground_cuda')
        report = registrar.register((ground-[0., 0., 1.5])/3., prior, 3.)
        self.assertFalse(report['accepted'])
        self.assertIn('horizontal_observability', report['reasons'])
        self.assertIn('scale_observability', report['reasons'])

    def test_noisy_tilted_bootstrap_matches_cpu_and_known_geometry(self):
        points, _ = room()
        rng = np.random.default_rng(24)
        raw = (points-[0., 0., 1.5]+rng.normal(0, .015, points.shape))/8.7
        prior = np.eye(4); prior[:3, 3] = [.25, -.2, 1.6]
        prior[:3, :3] = Rotation.from_euler('xyz', [14., 14., 1.5], degrees=True).as_matrix()
        cpu = CausalScaleRegistrar(points, backend='cpu_reference').register(raw, prior, 3., bootstrap=True)
        gpu = CausalScaleRegistrar(points, cache_dir=self.output_run/'bootstrap_cuda').register(raw, prior, 3., bootstrap=True)
        self.assertTrue(gpu['accepted'], gpu)
        self.assertEqual(cpu['accepted'], gpu['accepted'])
        np.testing.assert_allclose(gpu['pose'], cpu['pose'], atol=1e-5, rtol=0)
        self.assertAlmostEqual(gpu['scale'], cpu['scale'], places=5)
        position_error = np.linalg.norm(np.array(gpu['pose'])[:3, 3]-[0., 0., 1.5])
        self.assertLess(position_error, .1)
        self.assertLess(abs(gpu['scale']-8.7), .2)
        (self.output_run/'synthetic_bootstrap_evidence.json').write_text(json.dumps({
            'invented_noisy_room_only': True, 'real_track_localization': False,
            'known_scale': 8.7, 'initial_scale': 3., 'recovered_scale': gpu['scale'],
            'position_error_m': float(position_error),
            'rotation_error_deg': float(np.rad2deg(Rotation.from_matrix(np.array(gpu['pose'])[:3, :3]).magnitude())),
            'cpu_cuda_pose_atol': 1e-5, 'accepted': gpu['accepted']}, indent=2)+'\n')

    def test_map_crop_and_first_row_voxel_preserve_xyz(self):
        rng = np.random.default_rng(23)
        points = np.r_[rng.normal(size=(1000, 3)), [[.3, -.3, .6]]*3, [[200., 0., 0.]]].astype(np.float32)
        a, report = gpu_crop_voxel(points, [0, 0, 0], torch, np)
        b, _ = gpu_crop_voxel(points, [0, 0, 0], torch, np)
        crop = points[(abs(points.astype(float)) <= 100).all(axis=1)].astype(float)
        _, keep = np.unique(np.floor(crop/.3).astype(np.int64), axis=0, return_index=True)
        np.testing.assert_array_equal(a, crop[np.sort(keep)])
        np.testing.assert_array_equal(a, b)
        self.assertEqual(report['raw_crop_points'], 1003)

    def test_synthetic_runner_serialization_and_prefix_consistency(self):
        """Real CUDA runner loop on invented room data; no real Track inputs."""
        points, _ = room()
        fixture = self.output_run/'invented_data'
        track = fixture/'Track3'
        raw_dir = track/'raw_ply'
        raw_dir.mkdir(parents=True)
        with (track/'map.pcd').open('x') as stream:
            stream.write(f'VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH {len(points)}\nHEIGHT 1\nPOINTS {len(points)}\nDATA ascii\n')
            np.savetxt(stream, points, fmt='%.8f')
        initial = np.eye(4); initial[:3, 3] = [.3, -.2, 1.6]
        np.savetxt(track/'init_guess.txt', initial, delimiter=',')
        stamps = [NS+i*40_000_000 for i in range(5)]
        for i, stamp in enumerate(stamps):
            raw = (points-[i*.04, 0., 1.5])/3.2
            with (raw_dir/f'{i:06d}_ts_{stamp}.ply').open('x') as stream:
                stream.write(f'ply\nformat ascii 1.0\nelement vertex {len(raw)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
                np.savetxt(stream, np.column_stack([raw, np.zeros_like(raw)]), fmt='%.8f %.8f %.8f %d %d %d')
        with sqlite3.connect(track/'Track3.db3') as db:
            db.execute('CREATE TABLE topics (id INTEGER,name TEXT,type TEXT)')
            db.execute('CREATE TABLE messages (id INTEGER,topic_id INTEGER,timestamp INTEGER,data BLOB)')
            db.execute('INSERT INTO topics VALUES (1,?,?)', ('/lucid_cameras_x00/gige_100_f_hdr/h265', 'sensor_msgs/msg/Image'))
            db.executemany('INSERT INTO messages VALUES (?,1,?,NULL)', list(enumerate(stamps)))
        def fake_export(args, run, end_ns):
            data = imu_arrays(np.arange(-.01, .2, .01))
            data = {k: v[data['availability_ns'] <= end_ns] for k, v in data.items()}
            data['accel_m_s2'] = np.tile([0., 0., 9.81], (len(data['availability_ns']), 1))
            np.savez_compressed(run/'imu.npz', **data)
        outputs = []
        for n in (3, 5):
            console = io.StringIO()
            with patch.object(runner, 'verify_container'), patch.object(runner, 'export_imu', side_effect=fake_export), contextlib.redirect_stdout(console):
                self.assertEqual(runner.main(['--data-root', str(fixture), '--frames', str(n)]), 0)
            text = console.getvalue()
            output = Path(next(line[5:] for line in text.splitlines() if line.startswith('Run: ')))
            outputs.append(output)
            meta = json.loads((output/'test_info.json').read_text())
            meta.update(synthetic_invented_room=True, real_track_localization=False,
                        fixture_owner=str(self.output_run), score=None)
            (output/'test_info.json').write_text(json.dumps(meta, indent=2)+'\n')
            summary = json.loads((output/'validation.json').read_text())
            self.assertTrue(summary['output_complete_and_finite'], summary)
            self.assertFalse(summary['submission_frame_set_complete'])
            self.assertEqual(summary['processed_frames'], n)
            self.assertEqual(len([line for line in text.splitlines() if line.startswith('Frame ')]), n)
            self.assertIn('Graph: executed=false, applied=false, time=0.00 ms', text)
            self.assertIn(f'Finished: frames={n},', text)
            self.assertFalse((output/'analysis').exists())
        first, second = outputs
        self.assertEqual((first/'result.txt').read_text(), '\n'.join((second/'result.txt').read_text().splitlines()[:3])+'\n')
        self.assertEqual((first/'diagnostic.csv').read_text(), '\n'.join((second/'diagnostic.csv').read_text().splitlines()[:4])+'\n')
        (self.output_run/'runner_prefix_evidence.json').write_text(json.dumps({
            'synthetic_invented_room_only': True, 'source_data_root': str(fixture),
            'prefix_frames': 3, 'longer_frames': 5, 'outputs': list(map(str, outputs)),
            'matrices_and_csv_prefix_byte_identical': True, 'real_track_localization': False}, indent=2)+'\n')
        # The original real failure wrote 3142 finite rows and zero anchors.
        # Reproduce that failure mode on this invented fixture and ensure it
        # cannot announce a successful run or generate successful-full plots.
        def reject_map(self, raw, prior, scale, **kwargs):
            return {'accepted': False, 'reasons': ['synthetic_rejected_map'],
                    'pose': prior.tolist(), 'scale': float(scale)}
        console = io.StringIO()
        with patch.object(runner, 'verify_container'), patch.object(runner, 'export_imu', side_effect=fake_export), \
                patch.object(CausalScaleRegistrar, 'register', reject_map), \
                contextlib.redirect_stdout(console), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'No map observation was accepted'):
                runner.main(['--data-root', str(fixture), '--frames', '3'])
        output = Path(next(line[5:] for line in console.getvalue().splitlines() if line.startswith('Run: ')))
        meta = json.loads((output/'test_info.json').read_text())
        meta.update(synthetic_invented_room=True, real_track_localization=False,
                    fixture_owner=str(self.output_run), intentionally_rejected_map=True)
        (output/'test_info.json').write_text(json.dumps(meta, indent=2)+'\n')
        summary = json.loads((output/'validation.json').read_text())
        self.assertEqual(summary['processed_frames'], 3)
        self.assertEqual(summary['map_accepted_frames'], 0)
        self.assertFalse(summary['tracking_established'])
        self.assertEqual(summary['returncode'], 1)
        self.assertNotIn('Finished:', console.getvalue())
        self.assertFalse((output/'analysis').exists())


if __name__ == '__main__':
    unittest.main()

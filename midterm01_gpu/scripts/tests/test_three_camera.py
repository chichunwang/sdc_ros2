import struct
import unittest
import csv
import json
from pathlib import Path
import tempfile
import numpy as np
from three_camera_source import decode_cdr_header, reuse_priors, resolve_camera_timestamps, load_reconstruction, sha256
from three_camera_calibration import SurfaceMap, fit_scale, frame_terms, camera_to_reference
from calibrated_rig_geometry import OPTICAL_TO_FLU, load_rig, backproject, transform, balanced_indices, write_ply
from anchor_rig_scale import depth_scale_consensus
from validate_rig_reconstruction import epipolar_error


class ThreeCameraTests(unittest.TestCase):
    def test_stereo_scale_ignores_holdout_and_keeps_rig_baseline(self):
        records = [{'track': t, 'role': 'fit', 'log_depth_ratios': [np.log(2.), np.log(2.), np.log(9.)]}
                   for t in (1, 1, 2, 2)]
        records.append({'track': 2, 'role': 'holdout', 'log_depth_ratios': [np.log(.01)] * 1000})
        self.assertAlmostEqual(depth_scale_consensus(records), .5, places=12)
        pose = np.eye(4); pose[:3, 3] = [.8, 0, .1]
        points = backproject(np.array([[4.]]) * .5, np.eye(3))
        np.testing.assert_allclose(transform(points, pose), [[[.8, 0, 2.1]]])

    def test_known_camera_transform_satisfies_epipolar_geometry(self):
        pose = np.eye(4); pose[0, 3] = .8
        k = np.array([[200., 0, 224.], [0, 200., 147.], [0, 0, 1.]])
        points = np.array([[1., 2., 10.], [-2., 1., 7.], [2., -1., 15.]])
        left = points @ k.T; left /= left[:, 2:3]
        right = (points - pose[:3, 3]) @ k.T; right /= right[:, 2:3]
        np.testing.assert_allclose(epipolar_error(left, right, k, k, pose), 0, atol=1e-10)
        right[:, 1] += 4
        self.assertTrue(np.all(epipolar_error(left, right, k, k, pose) > 2.))

    def test_fixed_rig_height_and_optical_backprojection(self):
        directory = Path(__file__).resolve().parents[3] / 'midterm01/data/Track1/Track1_images/camera_info'
        if not directory.is_dir():
            directory = Path('/root/data/Track1/Track1_images/camera_info')
        if directory.is_dir():
            rig = load_rig(directory)
            np.testing.assert_allclose(OPTICAL_TO_FLU @ rig['fl'][:3, 3], [-.0600846, .784752, .0783861])
            np.testing.assert_allclose(OPTICAL_TO_FLU @ rig['fr'][:3, 3], [-.0442703, -.760501, .100434])
        k = np.array([[2., 0, 1], [0, 2, 1], [0, 0, 1]])
        points = backproject(np.full((3, 3), 4.), k)
        np.testing.assert_allclose(points[1, 1] @ OPTICAL_TO_FLU.T, [4, 0, 0])
        np.testing.assert_allclose(points[0, 0] @ OPTICAL_TO_FLU.T, [4, 2, 2])

    def test_balancing_retains_walls_with_dense_ground(self):
        x, y = np.meshgrid(np.linspace(-8, 8, 75), np.linspace(-8, 8, 75))
        ground = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, -2.)))
        u, z = np.meshgrid(np.linspace(-6, 6, 45), np.linspace(-1, 5, 25))
        walls = [np.column_stack((u.ravel(), np.full(u.size, 6 * sign), z.ravel())) for sign in (-1, 1)]
        points = np.vstack([ground] + walls)
        ids = np.repeat([0, 1, 2], [len(ground)] + [len(p) for p in walls])
        selected, report = balanced_indices(points, ids, np.ones(len(points)), budget=900)
        self.assertLessEqual(len(selected), 900)
        self.assertEqual(len(np.unique(selected)), len(selected))
        self.assertGreater(np.mean(ids[selected] != 0), .55)
        self.assertGreater(report['bins']['camera1_normal1']['cells'], 5)
        again, _ = balanced_indices(points, ids, np.ones(len(points)), budget=900)
        np.testing.assert_array_equal(selected, again)

    def test_reconstruction_manifest_rejects_timestamp_or_payload_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stamp = 1727234445100462219
            cloud = root / f'000001_ts_{stamp}.ply'
            write_ply(cloud, [[3, 0, 0]], [[255, 0, 0]])
            report = {'schema': 'calibrated_rig_v1', 'track': 1, 'coordinate_frame': 'front_camera_FLU',
                      'reference_time': 'front_camera_bag_timestamp', 'ply_scale': 1.,
                      'frames': [{'original_idx': 1, 'timestamp_ns': stamp, 'filename': cloud.name, 'sha256': sha256(cloud)}]}
            path = root / 'report.json'
            path.write_text(json.dumps(report))
            self.assertEqual(load_reconstruction(path, root, 1, [stamp - 1, stamp], 1, 1)[1][0][0], stamp)
            with self.assertRaises(ValueError):
                load_reconstruction(path, root, 1, [stamp - 1, stamp + 1], 1, 1)
            with cloud.open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaises(ValueError):
                load_reconstruction(path, root, 1, [stamp - 1, stamp], 1, 1)

    def test_prior_reuse_keeps_exact_values_and_rejects_wrong_track(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / 'source', root / 'output'
            source.mkdir()
            output.mkdir()
            (source / 'validation.json').write_text(json.dumps({'track': 1, 'profile': 'sensor_guided', 'sensor_fusion': 'fixed'}))
            (source / 'sensor_prior.json').write_text('{"frames": 2}')
            stamps = [1727234445100462219, 1727234445135440914]
            (source / 'sensor_prior.csv').write_text('timestamp_ns,x,y,z,qx,qy,qz,qw\n' + ''.join(f'{s},1.234567890123,2,3,0,0,0,1\n' for s in stamps))
            reuse_priors(source, output, stamps[1:], 1, 'fixed')
            with (output / 'sensor_prior.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]['timestamp_ns'], str(stamps[1]))
            self.assertEqual(rows[0]['x'], '1.234567890123')
            with self.assertRaises(ValueError):
                reuse_priors(source, output, stamps, 2, 'fixed')

    def test_mixed_timestamp_bases_and_ambiguity(self):
        base = 1727234445100462219
        series = {base: (base - 9_000_000, 'f'), base + 36_000_000: (base + 27_000_000, 'f')}
        self.assertEqual(resolve_camera_timestamps(series, [base])[0][-1], 'bag')
        self.assertEqual(resolve_camera_timestamps(series, [base - 9_000_000])[0][0], base)
        with self.assertRaises(ValueError):
            resolve_camera_timestamps(series, [base + 1])
        with self.assertRaises(ValueError):
            resolve_camera_timestamps({base: (base - 1, 'f'), base + 1: (base, 'f')}, [base])
        with self.assertRaises(ValueError):
            resolve_camera_timestamps({base: (base, 'f'), base + 1: (base - 1, 'f')}, [base])

    def test_integer_cdr_timestamp_and_bad_header(self):
        name = b'camera\x00'
        blob = b'\x00\x01\x00\x00' + struct.pack('<iII', 1727234445, 100462219, len(name)) + name
        self.assertEqual(decode_cdr_header(blob), (1727234445100462219, 'camera'))
        with self.assertRaises(ValueError):
            decode_cdr_header(blob[:-1])
        with self.assertRaises(ValueError):
            decode_cdr_header(b'\x00\x01\x00\x00' + struct.pack('<iII', 1, 1000000000, 1) + b'\x00')

    def test_motion_compensation_static_world(self):
        reference, capture, extrinsic = np.eye(4), np.eye(4), np.eye(4)
        reference[:3, 3] = [100, 200, 3]
        capture[:3, 3] = [100.2, 200, 3]
        extrinsic[:3, 3] = [0, .8, 0]
        point = np.array([[5, 0, 0]])
        compensated = camera_to_reference(point, reference, capture, extrinsic)
        np.testing.assert_allclose(compensated, [[5.2, .8, 0]])
        np.testing.assert_allclose(compensated + reference[:3, 3], point + capture[:3, 3] + extrinsic[:3, 3])
        # Same static world point after a known 90-degree vehicle turn.
        capture[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        np.testing.assert_allclose(camera_to_reference([[4, -10.6, 0]], reference, capture, extrinsic), [[10, 4, 0]])

    def test_ground_does_not_identify_scale_after_free_height(self):
        x, y = np.meshgrid(np.linspace(-5, 5, 35), np.linspace(-5, 5, 35))
        points = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, -2.)))
        surface = SurfaceMap(points)
        _, stats, _, _ = frame_terms(surface, points / 1.1, np.eye(4), 0., np.zeros(4))
        self.assertGreater(stats['reliable_points'], 100)
        self.assertLess(stats['scale_information_after_free_pose'], 1e-7)

    def test_scale_from_opposing_walls_with_pose_nuisance(self):
        y, z = np.meshgrid(np.linspace(-4, 6, 40), np.linspace(-2, 3, 25))
        walls = np.vstack([np.column_stack((np.full(y.size, x), y.ravel(), z.ravel())) for x in (-5., 5.)])
        surface = SurfaceMap(walls)
        raw = walls / 1.12
        result = fit_scale(surface, [(raw, np.eye(4))], iterations=40)
        self.assertAlmostEqual(result['scale'], 1.12, delta=.015)
        self.assertGreater(result['frames'][0]['scale_information_after_free_pose'], 1.)
        costs = [h['cost'] for h in result['history']]
        self.assertTrue(all(b <= a + 1e-9 for a, b in zip(costs, costs[1:])))


if __name__ == '__main__':
    unittest.main()

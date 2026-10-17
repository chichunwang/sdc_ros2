import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trajectory_filter_lab import filtered_poses, load_run, local_polynomial, rts_smooth


class TrajectoryFilterTests(unittest.TestCase):
    def test_time_polynomial_preserves_acceleration_on_irregular_timestamps(self):
        t = np.cumsum(np.tile([.03, .04, .036], 100))
        truth = np.column_stack((1e6 + 10 * t + .2 * t**2, -3e5 + t, .2 + .03 * t))
        # 同時驗證邊界；不把時間間隔當常數。
        np.testing.assert_allclose(local_polynomial(t, truth, .5), truth, atol=1e-8, rtol=0)

    def test_offline_filter_reduces_known_noise_without_constant_speed_lag(self):
        rng = np.random.default_rng(42)
        t = np.cumsum(rng.uniform(.03, .04, 600))
        truth = np.column_stack((10 * t, 2 * t, .3 + .02 * t))
        measured = truth + rng.normal(size=truth.shape) * [.08, .08, np.deg2rad(.2)]
        for filtered in (local_polynomial(t, measured, .5),
                         rts_smooth(t, measured, [.08, .08, np.deg2rad(.2)],
                                    [4., 4., np.deg2rad(5)**2], [5., 5., np.deg2rad(30)])):
            interior = slice(30, -30)
            self.assertLess(np.linalg.norm(np.mean(filtered[interior, :2] - truth[interior, :2], axis=0)), .02)
            self.assertLess(np.mean((filtered[interior] - truth[interior])**2),
                            np.mean((measured[interior] - truth[interior])**2) * .6)

    def test_yaw_wrap_keeps_rotation_and_roll_pitch_valid(self):
        yaw = np.deg2rad(np.linspace(179, 181, 100))
        rotations = Rotation.from_euler('xyz', np.column_stack((yaw * 0 + .04, yaw * 0 - .05, yaw))).as_matrix()
        poses = np.zeros((len(yaw), 3, 4))
        poses[:, :, :3] = rotations
        poses[:, 2, 3] = 8
        values = np.column_stack((yaw * 0, yaw * 0, yaw + np.deg2rad(.1)))
        result = filtered_poses(poses, values)
        recovered = Rotation.from_matrix(result[:, :, :3]).as_euler('xyz')
        np.testing.assert_allclose(recovered[:, :2], np.tile([.04, -.05], (len(yaw), 1)), atol=1e-10)
        np.testing.assert_allclose(np.unwrap(recovered[:, 2]), values[:, 2], atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(result[:, :, :3]), 1, atol=1e-10)
        np.testing.assert_array_equal(result[:, 2, 3], 8)

    def test_manifest_mismatch_and_invalid_rotation_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'validation.json').write_text(json.dumps({'track': 1, 'profile': 'sensor_guided',
                'returncode': 0, 'validation_errors': [], 'output_complete_and_finite': True,
                'frames': 7, 'expected_frames': 7}))
            poses = np.tile(np.eye(4)[:3], (7, 1, 1))
            np.savetxt(run / 'result.txt', poses.reshape(7, 12))
            def manifest(indices, stamps):
                with (run / 'frames.csv').open('w', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['idx', 'timestamp_ns', 'filename'])
                    writer.writerows((i, stamp, f'{i}.ply') for i, stamp in zip(indices, stamps))
            stamps = [1727234445000000000 + i * 36000000 for i in range(7)]
            manifest(range(7), stamps)
            self.assertEqual(len(load_run(run, 1)[0]), 7)
            with self.assertRaises(ValueError):
                load_run(run, 2)
            manifest([0, 1, 2, 2, 4, 5, 6], stamps)
            with self.assertRaises(ValueError):
                load_run(run, 1)
            manifest(range(7), stamps[:3] + [stamps[2]] + stamps[4:])
            with self.assertRaises(ValueError):
                load_run(run, 1)
            manifest(range(7), stamps)
            poses[3, 0, 0] = 2
            np.savetxt(run / 'result.txt', poses.reshape(7, 12))
            with self.assertRaises(ValueError):
                load_run(run, 1)


if __name__ == '__main__':
    unittest.main()

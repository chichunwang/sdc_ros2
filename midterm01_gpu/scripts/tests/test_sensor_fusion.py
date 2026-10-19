from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_sensor_prior import make_priors
from sensor_fusion import FusionConfig, bias_rts, heading_observations, position_rts


class SensorFusionTests(unittest.TestCase):
    def test_bias_rts_tracks_drift_with_outlier_and_limits(self):
        config = FusionConfig()
        rng = np.random.default_rng(24)
        t = np.arange(1., 120., 2.)
        truth = np.deg2rad(4 - .035 * t)
        observed = truth + np.deg2rad(rng.normal(0, .35, len(t)))
        observed[25] += np.deg2rad(10)
        queries = np.linspace(-1., 122., 2500)
        bias, std, meta = bias_rts(t, observed, np.full(len(t), np.deg2rad(.7)), queries, config)
        expected = np.deg2rad(4 - .035 * np.clip(queries, t[0], t[-1]))
        self.assertLess(np.rad2deg(np.sqrt(np.mean((bias - expected)**2))), .3)
        self.assertGreater(meta['innovation_downweighted_blocks'], 0)
        self.assertTrue(np.all(np.isfinite(std)) and np.all(std > 0))
        self.assertLessEqual(np.max(np.abs(np.diff(bias) / np.diff(queries))), np.deg2rad(.2) + 1e-12)
        extreme, _, _ = bias_rts(t, np.full(len(t), np.deg2rad(30)), np.full(len(t), .01), queries, config)
        self.assertLessEqual(np.max(np.abs(extreme)), np.deg2rad(config.max_bias_deg))

    def test_motion_rts_known_motion_irregular_queries_and_spike(self):
        config = FusionConfig(gnss_std_m=.15)
        rng = np.random.default_rng(5)
        gt = np.arange(0., 30., .25)
        truth = np.column_stack((1e6 + 10 * gt, -2e5 + 2 * gt))
        measured = truth + rng.normal(0, .15, truth.shape)
        measured[50] += [15, -12]
        query = np.sort(rng.uniform(-.1, gt[-1] + .1, 400))
        xy, velocity, std, meta = position_rts(gt, measured, query, config)
        expected = np.column_stack((1e6 + 10 * query, -2e5 + 2 * query))
        self.assertEqual(meta['position_measurement_updates'], len(gt))
        self.assertGreater(meta['position_downweighted_updates'], 0)
        self.assertLess(np.sqrt(np.mean(np.sum((xy - expected)**2, axis=1))), .2)
        self.assertLess(np.linalg.norm(np.mean(xy - expected, axis=0)), .05)
        self.assertTrue(np.all(np.isfinite(std)) and np.all(std > 0))
        self.assertLess(np.max(np.abs(velocity - [10, 2])), 2.)
        # 加入更多沒有量測的 frame 不會憑空增加資訊。
        sparse = position_rts(gt, measured, query[::5], config)[0]
        np.testing.assert_allclose(xy[::5], sparse, atol=1e-8, rtol=0)

    def test_course_quality_gates_stop_turn_reverse_and_yaw_wrap(self):
        config = FusionConfig()
        t = np.arange(0., 10., .25)
        it = np.arange(0., 10., .01)
        static = heading_observations(t, np.zeros((len(t), 2)), it, it * 0, config)
        self.assertTrue(all(row['reason'] == 'low_speed' for row in static))
        xy = np.column_stack((10 * t, t * 0))
        turning = heading_observations(t, xy, it, .2 * it, config)
        self.assertFalse(any(row['accepted'] for row in turning))
        reverse = heading_observations(t, -xy, it, it * 0, config)
        self.assertFalse(any(row['accepted'] for row in reverse))
        # 真實 course -179 度，IMU yaw +179 度：差值應是 +2 度。
        angle = np.deg2rad(-179)
        wrapped = heading_observations(t, 10 * t[:, None] * [np.cos(angle), np.sin(angle)],
                                      it, it * 0 + np.deg2rad(179), config)
        self.assertTrue(all(row['accepted'] for row in wrapped))
        np.testing.assert_allclose([row['offset_deg'] for row in wrapped], 2., atol=1e-9)

    def test_pipeline_preserves_roll_pitch_xy_and_prefix_consistency(self):
        origin = 1727234445000000000
        gt = np.arange(0., 40.25, .25)
        it = np.arange(0., 40.001, .01)
        ft = np.arange(.1, 39.9, .036)
        def stamps(t):
            return origin + np.rint(t * 1e9).astype(np.int64)
        # 車向 map X 前進，但 IMU yaw 有已知、慢變的負偏差。
        xyz = np.column_stack((1e5 + 10 * gt, gt * 0, gt * 0 + 5))
        imu = Rotation.from_euler('xyz', np.column_stack((it * 0 + .02, it * 0 - .03,
                                                       -np.deg2rad(4 - .04 * it)))).as_quat()
        args = (stamps(gt), xyz, stamps(it), imu, stamps(ft))
        baseline = make_priors(*args)
        details = {}
        fused = make_priors(*args, fusion='heading_rts', details=details)
        np.testing.assert_array_equal(fused[0], baseline[0])
        angles = Rotation.from_quat(fused[1]).as_euler('xyz')
        np.testing.assert_allclose(angles[:, :2], np.tile([.02, -.03], (len(ft), 1)), atol=1e-10)
        baseline_yaw = Rotation.from_quat(baseline[1]).as_euler('xyz')[:, 2]
        self.assertLess(np.sqrt(np.mean(angles[:, 2]**2)), .6 * np.sqrt(np.mean(baseline_yaw**2)))
        self.assertIsNone(fused[2]['sensor_fusion']['heading_fallback'])
        prefix = make_priors(*args[:4], stamps(ft[:50]), fusion='heading_rts')
        np.testing.assert_allclose(prefix[1], fused[1][:50], atol=1e-12)
        motion = make_priors(*args, fusion='motion_rts')
        np.testing.assert_array_equal(motion[0][:, 2], baseline[0][:, 2])

    def test_insufficient_heading_and_invalid_inputs_are_explicit(self):
        origin = 1727234445000000000
        gt = origin + np.arange(9, dtype=np.int64) * 250000000
        it = origin + np.arange(201, dtype=np.int64) * 10000000
        quat = Rotation.from_euler('z', np.full((len(it), 1), .1)).as_quat()
        result = make_priors(gt, np.zeros((len(gt), 3)), it, quat, gt[1:-1], fusion='heading_rts')
        self.assertIn('insufficient', result[2]['sensor_fusion']['heading_fallback'])
        np.testing.assert_allclose(result[1], np.tile(quat[0], (len(gt) - 2, 1)), atol=1e-12)
        with self.assertRaises(ValueError):
            make_priors(gt, np.zeros((len(gt), 3)), it, quat, [float(gt[1])], fusion='heading_rts')
        with self.assertRaises(ValueError):
            make_priors(gt, np.zeros((len(gt), 3)), it, quat, [gt[2], gt[1]], fusion='heading_rts')
        for config in [replace(FusionConfig(), gnss_std_m=float('nan')), replace(FusionConfig(), course_window_s=.5)]:
            with self.assertRaises(ValueError):
                config.validate()


if __name__ == '__main__':
    unittest.main()

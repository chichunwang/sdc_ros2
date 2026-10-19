from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_sensor_prior import make_priors
from sensor_fusion import FusionConfig, independent_heading_product, pose_heading_observations
from heading_ablation import guarded_validation


class GnssHeadingTests(unittest.TestCase):
    def test_wrap_turn_stop_and_no_sample_count_confidence_gain(self):
        config = FusionConfig()
        it = np.arange(0., 10., .01)
        imu = np.deg2rad(179) + .06 * it  # crosses the yaw branch cut during a turn
        for step in (.25, .125):
            t = np.arange(0., 10., step)
            q = Rotation.from_euler('z', (np.interp(t, it, imu) + np.deg2rad(2))[:, None]).as_quat()
            result = pose_heading_observations(t, q, it, imu, config)
            self.assertTrue(all(r['accepted'] for r in result))
            np.testing.assert_allclose([r['offset_deg'] for r in result], 2., atol=1e-10)
            self.assertTrue(all(r['std_deg'] == 3. for r in result))
            self.assertEqual(len(result), 5)

    def test_invalid_quaternion_and_disagreement_are_rejected(self):
        t = np.arange(0., 10., .25)
        q = Rotation.from_euler('z', np.zeros((len(t), 1))).as_quat()
        q[3] = 0
        q[10:16] = Rotation.from_euler('z', 40, degrees=True).as_quat()
        result = pose_heading_observations(t, q, t, t * 0, FusionConfig())
        self.assertFalse(result[0]['accepted'])
        self.assertFalse(result[1]['accepted'])
        self.assertTrue(result[2]['accepted'])
        with self.assertRaises(ValueError):
            pose_heading_observations(t, q[:-1], t, t * 0, FusionConfig())

    def test_independent_product_has_unique_knots_and_is_order_invariant(self):
        a = {'time_s': 1., 'accepted': True, 'offset_deg': 2., 'std_deg': 1.}
        b = dict(a, offset_deg=6., std_deg=2.)
        product = independent_heading_product([a, b])
        self.assertEqual(product, independent_heading_product([b, a]))
        self.assertEqual(len(product), 1)
        self.assertAlmostEqual(product[0]['offset_deg'], 2.8)
        self.assertAlmostEqual(product[0]['std_deg'], np.sqrt(.8))

    def test_end_to_end_known_heading_preserves_motion_xyz_and_roll_pitch(self):
        origin = 1727234445000000000
        t = np.arange(0., 20.25, .25)
        it = np.arange(0., 20.001, .01)
        ft = np.arange(.1, 19.9, .036)
        def ns(t):
            return origin + np.rint(t * 1e9).astype(np.int64)
        # Stationary vehicle: no course factors, but the hypothetical independent
        # heading observes a known IMU bias. This is a model test, not real accuracy.
        xyz = np.tile([1e6, -2e5, 5.], (len(t), 1))
        q = Rotation.from_euler('xyz', np.tile([.02, -.03, -.06], (len(it), 1))).as_quat()
        provided = Rotation.from_euler('z', np.zeros((len(t), 1))).as_quat()
        args = ns(t), xyz, ns(it), q, ns(ft)
        base = make_priors(*args, fusion='motion_rts')
        for mode in ('pose_heading_rts', 'independent_heading_rts'):
            result = make_priors(*args, fusion=mode, gnss_quaternions=provided)
            np.testing.assert_array_equal(base[0], result[0])
            rpy = Rotation.from_quat(result[1]).as_euler('xyz')
            np.testing.assert_allclose(rpy[:, :2], np.tile([.02, -.03], (len(ft), 1)), atol=1e-10)
            self.assertLess(np.max(np.abs(rpy[:, 2])), .002)
            self.assertFalse(result[2]['sensor_fusion']['heading_source_verified'])
            prefix = make_priors(*args[:4], ns(ft[:30]), fusion=mode, gnss_quaternions=provided)
            np.testing.assert_allclose(prefix[1], result[1][:30], atol=1e-12)
        with self.assertRaises(ValueError):
            make_priors(*args, fusion='independent_heading_rts')

    def test_guarded_validation_never_trains_on_changed_target_observations(self):
        t = np.arange(0., 40., .25)
        it = np.arange(0., 40., .01)
        xyz = np.column_stack((10*t, t*0, t*0))
        q = Rotation.from_euler('z', np.full((len(t), 1), .02)).as_quat()
        before = guarded_validation(t, xyz, it, it*0, q, FusionConfig())
        target = before['blocks'][4]['block']
        mask = np.floor(t/2).astype(int) == target
        changed_xyz, changed_q = xyz.copy(), q.copy()
        changed_xyz[mask, :2] += (t[mask]-t[mask][0])[:, None]*[0, 1]
        changed_q[mask] = Rotation.from_euler('z', .07).as_quat()
        after = guarded_validation(t, changed_xyz, it, it*0, changed_q, FusionConfig())
        self.assertNotEqual(before['blocks'][4]['methods']['motion_rts']['heldout_course_residual_deg'],
                            after['blocks'][4]['methods']['motion_rts']['heldout_course_residual_deg'])
        for mode in ('motion_rts', 'pose_heading_rts', 'independent_heading_rts'):
            self.assertEqual(before['blocks'][4]['methods'][mode]['predicted_bias_deg'],
                             after['blocks'][4]['methods'][mode]['predicted_bias_deg'])


if __name__ == '__main__':
    unittest.main()

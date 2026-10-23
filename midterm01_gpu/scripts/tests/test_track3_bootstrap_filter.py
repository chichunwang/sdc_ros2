"""Synthetic bootstrap guards; no real localization or accuracy claim."""
from pathlib import Path
import json
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from causal_track3_filter import CausalKalman
from causal_test_support import fixture_run

NS = 1_727_239_790_000_000_000


def observation(angle=0., x=0., scale=3.):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler('z', angle, degrees=True).as_matrix()
    pose[0, 3] = x
    return {'accepted': True, 'reasons': [], 'pose': pose, 'scale': scale,
            'measurement_covariance': np.diag([.01]*3+[np.deg2rad(1.)**2]*3+[.0025])}


class BootstrapFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_path = fixture_run()

    def test_published_unanchored_pose_allows_first_12_5_degree_alignment(self):
        f = CausalKalman(np.eye(4), NS, 3.)
        old, _ = f.publish()
        snapshot = old.copy()
        f.timestamp_ns += 36_000_000
        out = f.update_map(observation(angle=12.5, x=2.))
        self.assertTrue(out['accepted'], out)
        self.assertEqual(f.last_map_ns, f.timestamp_ns)
        np.testing.assert_array_equal(old, snapshot)
        np.testing.assert_array_equal(f.published_pose, snapshot)

    def test_first_alignment_still_rejects_170_degrees_or_over_5_m(self):
        for kwargs, reason in [({'angle': 170.}, 'initialization_rotation_bound'),
                               ({'x': 5.01}, 'translation_innovation_bound'),
                               ({'scale': 16.01}, 'scale_domain')]:
            with self.subTest(kwargs=kwargs):
                f = CausalKalman(np.eye(4), NS, 3.)
                f.publish(); f.timestamp_ns += 36_000_000
                obs = observation(**kwargs)
                obs['measurement_covariance'] = np.eye(7)*1e6
                before = f.pose.copy()
                out = f.update_map(obs)
                self.assertFalse(out['accepted'])
                self.assertIn(reason, out['reasons'])
                np.testing.assert_array_equal(f.pose, before)

    def test_anchored_12_5_degrees_in_36_ms_is_rejected(self):
        f = CausalKalman(np.eye(4), NS, 3.)
        self.assertTrue(f.update_map(observation())['accepted'])
        f.publish(); f.timestamp_ns += 36_000_000
        out = f.update_map(observation(angle=12.5))
        self.assertFalse(out['accepted'])
        self.assertIn('physical_rotation_jump', out['reasons'])

    def test_bootstrap_nis_still_rejects(self):
        f = CausalKalman(np.eye(4), NS, 3.)
        f.P *= 1e-8
        out = f.update_map(observation(x=1.))
        self.assertFalse(out['accepted'])
        self.assertIn('kalman_innovation', out['reasons'])

    def test_posterior_bounds_remain_before_state_mutation(self):
        # Deliberately inject correlated-gain overshoots into the small Kalman
        # solve, independently of candidate validity, to exercise each guard.
        cases = [(3, 41., 'posterior_speed_bound'),
                 (0, 5.01, 'posterior_translation_bound'),
                 (8, np.deg2rad(24.01), 'posterior_initialization_rotation_bound'),
                 (9, np.deg2rad(5.01), 'posterior_gyro_bias_bound'),
                 (10, np.log(16.01/3.), 'posterior_scale_domain')]
        for index, value, reason in cases:
            with self.subTest(reason=reason):
                f = CausalKalman(np.eye(4), NS, 3.)
                old, _ = f.publish(); f.timestamp_ns += 36_000_000
                before, covariance = f.pose.copy(), f.P.copy()
                gain = np.zeros((11, 7)); gain[index, 0] = value
                with patch('causal_track3_filter.np.linalg.solve',
                           side_effect=[np.zeros(7), gain.T]):
                    out = f.update_map(observation(x=1.))
                self.assertFalse(out['accepted'])
                self.assertIn(reason, out['reasons'])
                np.testing.assert_array_equal(f.pose, before)
                np.testing.assert_array_equal(f.P, covariance)
                np.testing.assert_array_equal(old, np.eye(4))
                self.assertIsNone(f.last_map_ns)


if __name__ == '__main__':
    result = unittest.main(exit=False)
    (BootstrapFilterTests.fixture_path/'bootstrap_filter_tests.json').write_text(json.dumps({
        'tests_run': result.result.testsRun, 'successful': result.result.wasSuccessful(),
        'synthetic_only': True, 'localization_executed': False}, indent=2)+'\n')
    raise SystemExit(not result.result.wasSuccessful())

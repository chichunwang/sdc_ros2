"""合成數學／時序測試，不代表真實 bag 精度。Ubuntu: bash scripts/dev.sh test。"""
import math
from pathlib import Path
import sys
import unittest

import numpy as np

if '__file__' in globals() and __file__ != '<stdin>':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'colcon_ws/src/fusion/scripts'))
from EKF import (ExtendedKalmanFilter, RadarGpsFilter, covariance,
                 relative_control, relative_covariance, wrap_angle)


class FilterTests(unittest.TestCase):
    def test_local_motion_and_wrapping(self):
        ekf = ExtendedKalmanFilter(0, 0, math.pi/2)
        pose, _ = ekf.predict([2., 0., math.pi])
        np.testing.assert_allclose(pose, [0, 2, -math.pi/2], atol=1e-12)

    def test_prediction_jacobian(self):
        state = np.array([2., -3., 0.7])
        control = [1., 0.3, 0.2]
        ekf = ExtendedKalmanFilter(*state)
        ekf.predict(control)
        numerical = np.empty((3, 3))
        for i in range(3):
            step = np.eye(3)[i] * 1e-6
            plus = ExtendedKalmanFilter(*(state+step)).predict(control)[0]
            minus = ExtendedKalmanFilter(*(state-step)).predict(control)[0]
            numerical[:, i] = (plus-minus)/2e-6
        np.testing.assert_allclose(ekf.A, numerical, atol=1e-8)

    def test_control_covariance_jacobian(self):
        a, b = np.array([2., 3., .8]), np.array([3., 5., .9])
        ca, cb = np.diag([.1, .2, .01]), np.diag([.3, .4, .02])
        jacobians = []
        for which in range(2):
            jac = np.zeros((3, 3))
            for axis in range(3):
                step = np.eye(3)[axis]*1e-6
                plus = relative_control(a+step if which == 0 else a, b+step if which == 1 else b)
                minus = relative_control(a-step if which == 0 else a, b-step if which == 1 else b)
                jac[:, axis] = (plus-minus)/2e-6
            jacobians.append(jac)
        expected = jacobians[0] @ ca @ jacobians[0].T + jacobians[1] @ cb @ jacobians[1].T
        np.testing.assert_allclose(relative_covariance(a, b, ca, cb), expected, atol=1e-8)

    def test_gps_noise_controls_weight(self):
        near = ExtendedKalmanFilter(0, 0)
        far = ExtendedKalmanFilter(0, 0)
        near.Q, far.Q = np.eye(2)*.01, np.eye(2)*100
        near.update([10, 0])
        far.update([10, 0])
        self.assertGreater(near.pose[0], far.pose[0])
        self.assertEqual(near.C.shape, (2, 3))

    def test_gps_first_uses_radar_heading(self):
        fusion = RadarGpsFilter()
        fusion.push_gps(100, [4, 5], np.eye(2))
        result = fusion.push_radar(100, [4, 5, 1.2], np.eye(3))
        self.assertAlmostEqual(result[-1][1][2], 1.2)
        self.assertEqual(fusion.stats['gps_updates'], 1)

    def test_gps_is_applied_at_its_timestamp(self):
        fusion = RadarGpsFilter()
        fusion.push_radar(0, [0, 0, 0], np.eye(3)*.01)
        self.assertEqual(fusion.push_gps(5, [5, 0], np.eye(2)*.1), [])
        result = fusion.push_radar(10, [10, 0, 0], np.eye(3)*.01)
        self.assertEqual([row[0] for row in result], [5, 10])
        np.testing.assert_allclose(result[0][1], [5, 0, 0], atol=1e-10)
        np.testing.assert_allclose(result[1][1], [10, 0, 0], atol=1e-10)

    def test_crossing_pi_uses_short_rotation(self):
        fusion = RadarGpsFilter()
        fusion.push_radar(0, [0, 0, math.radians(179)], np.eye(3)*.01)
        fusion.push_gps(5, [0, 0], np.eye(2))
        result = fusion.push_radar(10, [0, 0, math.radians(-179)], np.eye(3)*.01)
        self.assertAlmostEqual(abs(result[0][1][2]), math.pi)
        self.assertAlmostEqual(wrap_angle(result[-1][1][2]-math.radians(-179)), 0)

    def test_stale_data_does_not_rewind(self):
        fusion = RadarGpsFilter()
        fusion.push_radar(100, [0, 0, 0], np.eye(3))
        self.assertEqual(fusion.push_gps(50, [999, 999], np.eye(2)), [])
        self.assertEqual(fusion.push_radar(90, [999, 999, 0], np.eye(3)), [])
        np.testing.assert_array_equal(fusion.ekf.pose, [0, 0, 0])
        self.assertEqual(fusion.stats['late_gps'], 1)

    def test_long_run_covariance_stays_finite_psd(self):
        ekf = ExtendedKalmanFilter(0, 0)
        ekf.R, ekf.Q = np.diag([.01, .01, .001]), np.eye(2)*2
        for i in range(500):
            ekf.predict([.2, 0, .01])
            if i % 5 == 0:
                ekf.update(ekf.pose[:2]+[.1, -.1])
            self.assertTrue(np.isfinite(ekf.S).all())
            self.assertGreaterEqual(np.linalg.eigvalsh(ekf.S)[0], 0)
            np.testing.assert_allclose(ekf.S, ekf.S.T, atol=1e-12)

    def test_invalid_covariance_rejected(self):
        for bad in (np.diag([-1, 1]), np.array([[1, 9], [0, 1]]), np.eye(2)*np.nan):
            with self.assertRaises(ValueError):
                covariance(bad, 2)


if __name__ == '__main__':
    unittest.main()

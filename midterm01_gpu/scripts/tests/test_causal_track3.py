"""Synthetic forward-filter and Sim(3) mechanism tests, not Track3 accuracy."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from causal_track3_filter import CausalKalman, FilterConfig, ImuCursor, proper_pose
from causal_scale_registration import CausalScaleConfig, CausalScaleRegistrar

NS = 1_727_239_790_000_000_000


def imu_arrays(seconds, rates=None, rotations=None):
    seconds = np.asarray(seconds)
    ns = NS+np.rint(seconds*1e9).astype(np.int64)
    return {'availability_ns': ns, 'measurement_ns': ns.copy(),
            'quaternion_xyzw': (np.tile([0., 0., 0., 1.], (len(ns), 1)) if rotations is None else rotations),
            'gyro_rad_s': np.column_stack([np.zeros((len(ns), 2)), np.zeros(len(ns)) if rates is None else rates])}


def observation(pose, scale=3.):
    return {'accepted': True, 'reasons': [], 'pose': pose.tolist(), 'scale': scale,
            'measurement_covariance': np.diag([.1**2]*3+[np.deg2rad(1.)**2]*3+[.05**2]).tolist()}


def room():
    a, b = np.meshgrid(np.linspace(-6, 6, 35), np.linspace(0, 5, 20))
    front = np.column_stack([np.full(a.size, 12.), a.ravel(), b.ravel()])
    a, b = np.meshgrid(np.linspace(-4, 12, 43), np.linspace(0, 5, 20))
    walls = [np.column_stack([a.ravel(), np.full(a.size, y), b.ravel()]) for y in (-6., 6.)]
    a, b = np.meshgrid(np.linspace(-4, 12, 43), np.linspace(-6, 6, 35))
    ground = np.column_stack([a.ravel(), b.ravel(), np.zeros(a.size)])
    return np.vstack([front, *walls, ground]), ground


class FilterTests(unittest.TestCase):
    def test_init_rejects_nonrigid_and_covariance_is_positive(self):
        p = np.eye(4); p[0, 0] = 3
        with self.assertRaises(ValueError): proper_pose(p)
        f = CausalKalman(np.eye(4), NS, 3.)
        self.assertGreater(np.linalg.eigvalsh(f.P)[0], 0)

    def test_availability_zoh_does_not_use_future_sample(self):
        f = CausalKalman(np.eye(4), NS, 3., replace(FilterConfig(), max_imu_age_s=2.))
        stream = ImuCursor(imu_arrays([0., .5, 1.5], [0., 1., 999.]))
        f.predict_to(NS, stream)
        d = f.predict_to(NS+10**9, stream)
        self.assertAlmostEqual(Rotation.from_matrix(f.pose[:3, :3]).as_rotvec()[2], .5)
        self.assertEqual(d['max_imu_availability_ns'], NS+500_000_000)
        self.assertEqual(stream.index, 2)

    def test_vertical_projection_needs_no_camera_extrinsic_or_absolute_yaw(self):
        r = Rotation.from_euler('zyx', [135., 30., 50.], degrees=True)
        data = imu_arrays([0., .01], rotations=np.tile(r.as_quat(), (2, 1)))
        data['gyro_rad_s'] = np.tile(r.inv().apply([0., 0., .2]), (2, 1))
        stream = ImuCursor(data)
        for record in stream.take_until(NS+10_000_000):
            self.assertAlmostEqual(record[2], .2, places=12)

    def test_rejected_map_keeps_moving_prediction_and_all_state(self):
        f = CausalKalman(np.eye(4), NS, 3.)
        stream = ImuCursor(imu_arrays([0., .01, .02, .03]))
        f.predict_to(NS, stream)
        self.assertTrue(f.update_map(observation(f.pose))['accepted'])
        f.velocity[0] = 2.; f.publish()
        f.predict_to(NS+36_000_000, stream)
        before, covariance = f.pose.copy(), f.P.copy()
        bad = f.pose.copy(); bad[0, 3] += 21.449
        report = f.update_map(observation(bad))
        self.assertFalse(report['accepted'])
        self.assertIn('physical_translation_jump', report['reasons'])
        np.testing.assert_array_equal(f.pose, before)
        np.testing.assert_array_equal(f.P, covariance)
        pose, _ = f.publish()
        self.assertAlmostEqual(pose[0, 3], .072)

    def test_170_degree_jump_is_rejected_even_with_loose_covariance(self):
        f = CausalKalman(np.eye(4), NS, 3.)
        stream = ImuCursor(imu_arrays([0., .03]))
        f.predict_to(NS, stream)
        self.assertTrue(f.update_map(observation(f.pose))['accepted'])
        f.publish(); f.predict_to(NS+36_000_000, stream)
        bad = f.pose.copy(); bad[:3, :3] = Rotation.from_euler('z', 170., degrees=True).as_matrix()
        obs = observation(bad); obs['measurement_covariance'] = (np.eye(7)*1e6).tolist()
        report = f.update_map(obs)
        self.assertIn('physical_rotation_jump', report['reasons'])
        self.assertFalse(report['accepted'])

    def test_kalman_innovation_and_scale_bounds_are_independent_gates(self):
        f = CausalKalman(np.eye(4), NS, 3.)
        stream = ImuCursor(imu_arrays([0., .03]))
        f.predict_to(NS, stream)
        self.assertTrue(f.update_map(observation(f.pose))['accepted']); f.publish()
        f.predict_to(NS+36_000_000, stream)
        report = f.update_map(observation(f.pose, scale=5.))
        self.assertIn('scale_innovation_bound', report['reasons'])
        self.assertIn('kalman_innovation', report['reasons'])

    def test_joseph_filter_uses_map_to_estimate_velocity_and_gyro_bias(self):
        cfg = replace(FilterConfig(), max_log_scale_step=.3)
        f = CausalKalman(np.eye(4), NS, 3., cfg)
        stream = ImuCursor(imu_arrays(np.arange(0, 2.01, .01), np.full(201, .02)))
        for i in range(51):
            stamp = NS+i*40_000_000
            f.predict_to(stamp, stream)
            truth = np.eye(4); truth[0, 3] = i*.04
            report = f.update_map(observation(truth))
            self.assertTrue(report['accepted'], report)
            _, state = f.publish()
            self.assertGreater(np.linalg.eigvalsh(f.P)[0], 0)
        self.assertLess(abs(f.velocity[0]-1.), .1)
        self.assertGreater(f.bias, 0.)
        self.assertFalse(state['tracking_lost'])

    def test_prefix_consistency_and_published_copies_are_immutable(self):
        def run(n, future_rate):
            data = imu_arrays(np.arange(0, 1.01, .01), np.r_[np.full(51, .1), np.full(50, future_rate)])
            f = CausalKalman(np.eye(4), NS, 3.); stream = ImuCursor(data); poses=[]
            for i in range(n):
                f.predict_to(NS+i*40_000_000, stream)
                obs = observation(f.pose); f.update_map(obs)
                poses.append(f.publish()[0])
            return np.array(poses)
        a = run(10, 900.); b = run(20, -900.)
        np.testing.assert_array_equal(a, b[:10])

    def test_stale_imu_and_double_publication_fail_explicitly(self):
        f = CausalKalman(np.eye(4), NS, 3.); stream = ImuCursor(imu_arrays([0.]))
        f.predict_to(NS, stream); f.publish()
        with self.assertRaises(ValueError): f.publish()
        with self.assertRaisesRegex(ValueError, 'stale'): f.predict_to(NS+200_000_000, stream)


class ScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.points, cls.ground = room()
        cls.truth = np.eye(4); cls.truth[2, 3] = 1.5
        cls.prior = cls.truth.copy(); cls.prior[:3, 3] += [.3, -.2, .1]
        cls.prior[:3, :3] = Rotation.from_euler('xyz', [1., -.8, 1.5], degrees=True).as_matrix()

    def test_bounded_scale_and_map_recover_synthetic_room(self):
        registrar = CausalScaleRegistrar(self.points, backend='cpu_reference')
        raw = (self.points-self.truth[:3, 3])/3.2
        out = registrar.register(raw, self.prior, 3., estimate_scale=True)
        self.assertTrue(out['accepted'], out)
        self.assertLess(abs(out['scale']-3.2), .05)
        self.assertLess(np.linalg.norm(np.array(out['pose'])[:3, 3]-self.truth[:3, 3]), .06)
        self.assertGreater(np.linalg.eigvalsh(out['measurement_covariance'])[0], 0)

    def test_ground_does_not_claim_scale_or_horizontal_observability(self):
        registrar = CausalScaleRegistrar(self.ground, backend='cpu_reference')
        out = registrar.register((self.ground-self.truth[:3, 3])/3., self.prior, 3.)
        self.assertFalse(out['accepted'])
        self.assertIn('horizontal_observability', out['reasons'])
        self.assertIn('scale_observability', out['reasons'])

    def test_fixed_scale_ablation_keeps_scale_exactly_fixed(self):
        registrar = CausalScaleRegistrar(self.points, backend='cpu_reference')
        out = registrar.register((self.points-self.truth[:3, 3])/3., self.prior, 3., estimate_scale=False)
        self.assertTrue(out['accepted'], out)
        self.assertEqual(out['scale'], 3.)

    def test_out_of_bounds_scale_is_not_silently_applied(self):
        registrar = CausalScaleRegistrar(self.points, backend='cpu_reference')
        out = registrar.register((self.points-self.truth[:3, 3])/6., self.prior, 3.)
        self.assertFalse(out['accepted'])
        self.assertLessEqual(out['scale'], 3.*np.exp(.15))

    def test_startup_can_identify_scale_far_from_uncalibrated_seed(self):
        registrar = CausalScaleRegistrar(self.points, backend='cpu_reference')
        out = registrar.register((self.points-self.truth[:3, 3])/8.7, self.prior, 3., bootstrap=True)
        self.assertTrue(out['accepted'], out)
        self.assertLess(abs(out['scale']-8.7), .15)


if __name__ == '__main__':
    unittest.main()

"""Synthetic tilted/noisy geometry: NOT a Track3 score or real localization."""
from pathlib import Path
import sys
import unittest
from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from causal_scale_registration import CausalScaleConfig, CausalScaleRegistrar
from map_scale_icp import linear_terms
from test_causal_track3 import room


class InitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.points, _ = room()

    def test_tilted_initial_pose_and_distant_scale_in_noisy_room(self):
        rng = np.random.default_rng(24)
        truth = np.eye(4); truth[2, 3] = 1.5
        prior = truth.copy()
        prior[:3, :3] = Rotation.from_euler('xyz', [14., 14., 1.5], degrees=True).as_matrix()
        prior[:3, 3] += [.25, -.2, .1]
        raw = (self.points-truth[:3, 3]+rng.normal(0, .015, self.points.shape))/8.7
        registrar = CausalScaleRegistrar(self.points, backend='cpu_reference')
        report = registrar.register(raw, prior, 3., bootstrap=True)
        self.assertTrue(report['accepted'], report)
        self.assertGreater(report['ground_initialization']['tilt_correction_deg'], 15.)
        self.assertLess(abs(report['scale']-8.7), .2)
        pose = np.array(report['pose'])
        self.assertLess(np.linalg.norm(pose[:3, 3]-truth[:3, 3]), .1)
        self.assertLess(np.rad2deg(Rotation.from_matrix(pose[:3, :3]).magnitude()), .5)

    def test_scale_pca_uses_metric_candidate_and_not_initial_mask(self):
        registrar = CausalScaleRegistrar(self.points, backend='cpu_reference')
        prior = np.eye(4); prior[2, 3] = 1.5
        raw = (self.points-[0., 0., 1.5])/9.
        _, _, low = registrar._geometry(raw, 3., prior)
        _, _, high = registrar._geometry(raw, 9., prior)
        self.assertGreater((high > 0).sum(), (low > 0).sum())

    def test_active_rotation_boundary_does_not_claim_convergence(self):
        prior = np.eye(4); prior[2, 3] = 1.5
        prior[:3, :3] = Rotation.from_euler('y', 20., degrees=True).as_matrix()
        registrar = CausalScaleRegistrar(self.points, replace(CausalScaleConfig(), startup_rotation_deg=5.),
                                         backend='cpu_reference')
        report = registrar.register((self.points-[0., 0., 1.5])/3., prior, 3., bootstrap=True,
                                    estimate_scale=False)
        self.assertFalse(report['accepted'])
        self.assertTrue(set(report['reasons']) & {'interior', 'optimizer_converged'})

    def test_ground_seed_excludes_validation_patches(self):
        from horizontal_surfel import heldout_patches
        from unittest.mock import patch
        prior = np.eye(4); prior[2, 3] = 1.5
        registrar = CausalScaleRegistrar(self.points, backend='cpu_reference')
        raw = (self.points-[0., 0., 1.5])/3.
        # If all local map patches are validation-only, no initialization plane
        # may be inferred from them, even though source ground is available.
        with patch('causal_scale_registration.heldout_patches', side_effect=lambda p: np.ones(len(p), bool)):
            self.assertIsNone(registrar._ground_initialization(raw, prior, 3., 0.))

    def test_sim3_jacobian_matches_independent_finite_differences(self):
        rng = np.random.default_rng(12)
        raw = rng.normal(size=(50, 3))
        r = Rotation.from_euler('xyz', [.2, -.1, .4]).as_matrix()
        scale, t = 4., np.array([660., -220., 10.])
        rotated = scale*raw @ r.T
        normal = rng.normal(size=raw.shape); normal /= np.linalg.norm(normal, axis=1)[:, None]
        target = rotated+t+rng.normal(0, .02, raw.shape)
        _, jac = linear_terms({'normal':normal, 'rotated':rotated, 'world':rotated+t, 'target':target})
        numerical = np.empty_like(jac)
        for axis in range(7):
            def residual(sign):
                d = np.zeros(7); d[axis] = sign*1e-5
                rr = Rotation.from_rotvec(d[3:6]).as_matrix() @ r
                world = scale*np.exp(d[6])*raw @ rr.T+t+d[:3]
                return np.einsum('ni,ni->n', normal, world-target)
            numerical[:, axis] = (residual(1)-residual(-1))/2e-5
        np.testing.assert_allclose(jac, numerical, atol=1e-8, rtol=1e-7)


if __name__ == '__main__':
    unittest.main()

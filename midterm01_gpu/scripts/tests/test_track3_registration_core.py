"""Numerical/rejection checks for the isolated map-only registration core."""
from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from track3_registration import (Geometry, MapRegistrar, RegistrationConfig,
    block_validation, checked_pose, choose_training_candidate, point_jacobian,
    assess_multistart, source_weights, voxel_statistics)


def room():
    rng = np.random.default_rng(123)
    n = 5000
    floor = np.column_stack([rng.uniform(-6, 6, n), rng.uniform(-5, 5, n), np.zeros(n)])
    xwall = np.column_stack([np.full(n, 6.), rng.uniform(-5, 5, n), rng.uniform(0, 5, n)])
    opposite = np.column_stack([np.full(n, -6.), rng.uniform(-5, 5, n), rng.uniform(0, 5, n)])
    ywall = np.column_stack([rng.uniform(-6, 6, n), np.full(n, 5.), rng.uniform(0, 5, n)])
    return np.vstack([floor, xwall, ywall, opposite])+rng.normal(0, .005, (4*n, 3))


class RegistrationMechanisms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.points = room()
        cls.scene = MapRegistrar(cls.points, backend='cpu_reference')

    def test_analytic_sim3_jacobian(self):
        rng = np.random.default_rng(8)
        raw = rng.normal(size=(19, 3))
        rotation = Rotation.from_euler('xyz', [.2, -.1, .5]).as_matrix()
        rotated = 4.7*raw @ rotation.T
        j = point_jacobian(torch.tensor(rotated)).numpy()
        eps = 1e-6
        for col in range(7):
            step = np.eye(7)[col]*eps
            q = np.exp(step[6])*raw @ (Rotation.from_rotvec(step[3:6]).as_matrix() @ rotation).T*4.7+step[:3]
            np.testing.assert_allclose((q-rotated)/eps, j[:, :, col], atol=2e-5)

    def test_ndt_centroid_covariance(self):
        rng = np.random.default_rng(11)
        p = np.vstack([rng.uniform(0, .2, (80, 3))+[2*x, 2*y, 2*z]
                       for x in range(2) for y in range(2) for z in range(2)])
        g = Geometry('cpu_reference')
        mean, precision = voxel_statistics(g.tensor(p), 1., g)
        self.assertEqual(len(mean), 8)
        np.testing.assert_allclose(mean[0], p[:80].mean(0), atol=1e-12)
        self.assertTrue(np.all(np.linalg.eigvalsh(precision.numpy()) > 0))

    def test_known_sim3_recovery(self):
        expected = np.eye(4)
        expected[:3, :3] = Rotation.from_euler('xyz', [4, -2, 8], degrees=True).as_matrix()
        expected[:3, 3] = [.2, -.3, .5]
        raw = (self.points[::2]-expected[:3, 3]) @ expected[:3, :3]/4.5
        seed = expected.copy()
        seed[:3, :3] = Rotation.from_rotvec([.02, -.03, .04]).as_matrix() @ seed[:3, :3]
        seed[:3, 3] += [.4, -.3, .2]
        result = self.scene.fit(raw, seed, 4.8)
        pose = np.array(result['pose'])
        self.assertLess(np.linalg.norm(pose[:3, 3]-expected[:3, 3]), .04)
        self.assertLess(Rotation.from_matrix(pose[:3, :3] @ expected[:3, :3].T).magnitude(), .005)
        self.assertLess(abs(result['scale']/4.5-1), .01)
        self.assertEqual(result['physical_rank'], 7)
        self.assertTrue(result['accepted'], result['reasons'])

    def test_fixed_scale_six_variables(self):
        seed = np.eye(4)
        seed[:3, 3] = [.3, -.2, .15]
        r = self.scene.fit(self.points[::2]/3., seed, 3., estimate_scale=False)
        self.assertEqual(r['scale'], 3.)
        self.assertEqual(len(r['physical_hessian_eigenvalues']), 6)
        self.assertLess(np.linalg.norm(np.array(r['pose'])[:3, 3]), .04)

    def test_no_overlap_is_rejected(self):
        raw = self.points[::3]/3.+100
        r = self.scene.fit(raw, np.eye(4), 3.)
        self.assertFalse(r['accepted'])
        self.assertEqual(r['validation']['coverage'], 0)
        self.assertIn('observable', r['reasons'])
        self.assertIn('coverage', r['reasons'])

    def test_flat_plane_cannot_claim_seven_dof(self):
        g = Geometry('cpu_reference')
        points = self.points[:5000].copy()
        points[:, 2] = 0.
        scene = MapRegistrar(points, backend='cpu_reference')
        r = scene.fit(points/3., np.eye(4), 3.)
        self.assertLess(r['physical_rank'], 7)
        self.assertIn('observable', r['reasons'])

    def test_three_planes_have_scale_translation_gauge(self):
        points = torch.tensor(self.points[:15000])
        normals = torch.repeat_interleave(torch.tensor([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]],
                                                       dtype=torch.float64), 5000, dim=0)
        # Exact planes: scale about their common corner can be absorbed by t.
        points[:5000, 2] = 0.
        points[5000:10000, 0] = 6.
        points[10000:, 1] = 5.
        j = torch.einsum('ni,nij->nj', normals, point_jacobian(points)).numpy()
        self.assertEqual(np.linalg.matrix_rank(j, tol=1e-8), 6)

    def test_iteration_limit_is_not_convergence(self):
        scene = MapRegistrar(self.points, backend='cpu_reference',
                             config=replace(RegistrationConfig(), iterations=1))
        initial = np.eye(4)
        initial[:3, 3] = [.6, -.4, .5]
        r = scene.fit(self.points[::2]/3., initial, 3.3)
        self.assertFalse(r['checks']['converged'])
        self.assertFalse(r['accepted'])

    def test_validation_never_selects_candidate(self):
        base = dict(checks=dict(converged=True, interior=True), training_cost=.4)
        a = dict(base, validation=dict(median_m=0))
        b = dict(base, training_cost=.3, validation=dict(median_m=100))
        self.assertIs(choose_training_candidate([a, b]), b)

    def test_nonconverged_candidate_cannot_hide_converged(self):
        a = dict(checks=dict(converged=False, interior=True), training_cost=0.)
        b = dict(checks=dict(converged=True, interior=True), training_cost=1.)
        self.assertIs(choose_training_candidate([a, b]), b)

    def test_competing_scales_are_rejected(self):
        a = dict(checks=dict(converged=True, interior=True), training_cost=.2,
                 training=dict(coverage=.95, median_m=.15, p90_m=.4), pose=np.eye(4).tolist(),
                 scale=3., accepted=True, reasons=[])
        b = dict(a, scale=7., training_cost=.23)
        report = assess_multistart([a, b])
        self.assertFalse(report['accepted'])
        self.assertIn('ambiguous_initialization', report['reasons'])

    def test_repeatable_initializations_pass_consensus(self):
        a = dict(checks=dict(converged=True, interior=True), training_cost=.2,
                 training=dict(coverage=.95, median_m=.15, p90_m=.4), pose=np.eye(4).tolist(),
                 scale=3., accepted=True, reasons=[])
        b = dict(a, scale=3.01, training_cost=.201)
        report = assess_multistart([a, b])
        self.assertTrue(report['accepted'])
        self.assertFalse(assess_multistart([a])['accepted'])

    def test_source_split_is_immutable(self):
        raw = self.points/3.
        first = block_validation(raw)
        self.assertGreater(first.mean(), .1)
        self.assertLess(first.mean(), .4)
        np.testing.assert_array_equal(first, block_validation(raw.copy()))

    def test_source_weights_equalize_blocks(self):
        points = torch.tensor([[0., 0., 0.], [.01, 0., 0.], [1., 0., 0.]])
        weight = source_weights(points)
        self.assertAlmostEqual(float(weight[:2].sum()), float(weight[2]))

    def test_icp_loss_is_continuous_at_search_radius(self):
        precision = torch.eye(3, dtype=torch.float64)[None]/.15**2
        radius = .75
        near = torch.tensor([[radius-1e-10, 0, 0]], dtype=torch.float64)
        far = torch.tensor([[radius+1e-10, 0, 0]], dtype=torch.float64)
        a, _ = self.scene._loss(near, precision, torch.tensor([True]), (radius/.15)**2)
        b, _ = self.scene._loss(far, precision, torch.tensor([False]), (radius/.15)**2)
        self.assertLess(abs(float(a-b)), 1e-8)

    def test_validation_points_do_not_change_fit(self):
        raw = self.points[::2]/3.
        mask = ~block_validation(raw)
        changed = raw.copy()
        changed[~mask] += 100.
        seed = np.eye(4)
        seed[:3, 3] = [.2, -.1, .1]
        a = self.scene.fit(raw, seed, 3.1, train_mask=mask)
        b = self.scene.fit(changed, seed, 3.1, train_mask=mask)
        np.testing.assert_array_equal(a['pose'], b['pose'])
        self.assertEqual(a['scale'], b['scale'])
        self.assertNotEqual(a['validation'], b['validation'])

    def test_weighted_robust_gradient(self):
        rng = np.random.default_rng(5)
        rotated = torch.tensor(rng.normal(size=(21, 3)))
        target = rotated+torch.tensor(rng.normal(0, .02, (21, 3)))
        residual = rotated-target
        precision = torch.eye(3, dtype=torch.float64)[None].expand(21, -1, -1)/.15**2
        good = torch.ones(21, dtype=torch.bool)
        weights = source_weights(rotated)
        _, r2 = self.scene._loss(residual, precision, good, None, weights)
        j = point_jacobian(rotated)
        w = self.scene.config.robust_c2/(r2+self.scene.config.robust_c2).square()*weights
        jt = j.transpose(1, 2) @ precision
        gradient = 2*((jt @ residual[:, :, None])[:, :, 0]*w[:, None]).sum(0)/weights.sum()
        epsilon = 1e-6
        for k in range(7):
            direction = np.eye(7)[k]*epsilon
            def value(sign):
                q = torch.tensor(Rotation.from_rotvec(sign*direction[3:6]).as_matrix())
                p = np.exp(sign*direction[6])*rotated @ q.T+torch.tensor(sign*direction[:3])
                return float(self.scene._loss(p-target, precision, good, None, weights)[0])
            numerical = (value(1)-value(-1))/(2*epsilon)
            self.assertAlmostEqual(float(gradient[k]), numerical, places=7)

    def test_invalid_geometry_pose_and_config(self):
        with self.assertRaises(ValueError):
            MapRegistrar(np.zeros((3, 3)), backend='cpu_reference')
        with self.assertRaises(ValueError):
            checked_pose(np.zeros((4, 4)))
        with self.assertRaises(ValueError):
            RegistrationConfig(min_scale=16)
        with self.assertRaises(ValueError):
            self.scene.fit(self.points, np.eye(4), float('nan'))


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable; never falls back')
class CudaRegistrationMechanisms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from causal_test_support import fixture_run
        torch.set_num_threads(1)
        cls.fixture_dir = fixture_run()
        cls.points = room()
        cls.cpu = MapRegistrar(cls.points, backend='cpu_reference')
        cls.gpu = MapRegistrar(cls.points, backend='cuda', cache_dir=cls.fixture_dir/'map_registration_cuda_cache')

    def test_voxel_statistics_cpu_cuda(self):
        cpu = self.cpu.ndt[0]
        gpu = self.gpu.ndt[0]
        np.testing.assert_allclose(cpu[1].numpy(), gpu[1].cpu().numpy(), atol=1e-11)
        np.testing.assert_allclose(cpu[2].numpy(), gpu[2].cpu().numpy(), rtol=1e-8, atol=1e-8)

    def test_cuda_nearest_neighbors_cpu_reference(self):
        rng = np.random.default_rng(33)
        query = rng.uniform([-8, -7, -2], [8, 7, 7], (201, 3))
        a, ai = self.cpu.geometry.query(self.cpu.tree, self.cpu.geometry.tensor(query), 8, 2.)
        b, bi = self.gpu.geometry.query(self.gpu.tree, self.gpu.geometry.tensor(query), 8, 2.)
        np.testing.assert_allclose(a.numpy(), b.cpu().numpy(), atol=1e-12)
        valid = np.isfinite(a.numpy())
        np.testing.assert_array_equal(ai.numpy()[valid], bi.cpu().numpy()[valid])

    def test_cuda_cpu_same_registration(self):
        raw = self.points[::2]/3.
        seed = np.eye(4)
        seed[:3, :3] = Rotation.from_euler('xyz', [1, -2, 3], degrees=True).as_matrix()
        seed[:3, 3] = [.4, -.3, .2]
        cpu = self.cpu.fit(raw, seed, 3.15)
        gpu = self.gpu.fit(raw, seed, 3.15)
        np.testing.assert_allclose(cpu['pose'], gpu['pose'], atol=1e-5)
        self.assertAlmostEqual(cpu['scale'], gpu['scale'], places=5)
        self.assertEqual(cpu['checks'], gpu['checks'])
        self.assertTrue(gpu['accepted'], gpu['reasons'])


if __name__ == '__main__':
    unittest.main()

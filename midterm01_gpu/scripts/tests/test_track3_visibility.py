"""Counterfactual depth checks; visibility does not certify pose accuracy."""
import unittest

import numpy as np
import torch

from test_track3_registration_core import room
from track3_registration import MapRegistrar, block_validation
from track3_visibility import projective_refine, ray_depth_diagnostic


class VisibilityChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.points = room()
        cls.scene = MapRegistrar(cls.points, backend='cpu_reference')
        # The positive-X wall lies ahead of the source; background surfaces
        # remain in the target to check front-most angular sampling.
        wall = cls.points[5000:10000:2]
        # Keep rays away from floor/side-wall occlusion boundaries.
        cls.raw = wall[(abs(wall[:, 1]) < 4) & (wall[:, 2] > .5)]/3.
        cls.mask = ~block_validation(cls.raw)

    def test_correct_depth_and_shrunken_counterfactual(self):
        correct = ray_depth_diagnostic(self.scene, self.raw, np.eye(4), 3., self.mask)
        shrunken = ray_depth_diagnostic(self.scene, self.raw, np.eye(4), 1.5, self.mask)
        self.assertGreater(correct['training']['support_fraction'], .99)
        self.assertLess(correct['training']['median_abs_log_depth_ratio'], .01)
        self.assertGreater(shrunken['training']['fraction_front_of_map'], .99)
        self.assertAlmostEqual(shrunken['training']['median_depth_ratio'], .5, places=2)

    def test_expanded_counterfactual_is_behind_map(self):
        expanded = ray_depth_diagnostic(self.scene, self.raw, np.eye(4), 6., self.mask)
        self.assertGreater(expanded['training']['fraction_behind_map'], .99)

    def test_projective_refinement_never_certifies_acceptance(self):
        result = projective_refine(self.scene, self.raw, np.eye(4), 3.05, self.mask, iterations=5)
        self.assertLessEqual(result['final_projective_cost'], result['initial_projective_cost']+1e-12)
        self.assertTrue(np.isfinite(result['pose']).all())
        self.assertFalse(result['accepted'])

    def test_backward_source_is_rejected(self):
        with self.assertRaises(ValueError):
            ray_depth_diagnostic(self.scene, -self.raw, np.eye(4), 3., self.mask)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable, no fallback')
class CudaVisibilityChecks(unittest.TestCase):
    def test_cpu_cuda_angular_neighbors_and_depth(self):
        from causal_test_support import fixture_run
        torch.set_num_threads(1)
        points = room()
        cpu = MapRegistrar(points, backend='cpu_reference')
        gpu = MapRegistrar(points, backend='cuda', cache_dir=fixture_run()/'visibility_cuda_cache')
        raw = points[5000:10000:2]/3.
        mask = ~block_validation(raw)
        a = ray_depth_diagnostic(cpu, raw, np.eye(4), 3., mask)
        b = ray_depth_diagnostic(gpu, raw, np.eye(4), 3., mask)
        for split in ('training', 'validation'):
            for key in a[split]:
                self.assertAlmostEqual(a[split][key], b[split][key], places=10)


if __name__ == '__main__':
    unittest.main()

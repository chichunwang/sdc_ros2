"""Synthetic CUDA cache identity regression; no real track localization."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cuda_scale_geometry import CudaLayeredMap, cuda_pairs
from causal_test_support import fixture_run


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA mechanism test requires GPU')
class SourceCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output_run = fixture_run()

    def test_metadata_identity_refresh_matches_fresh_and_identical_objects_reuse(self):
        a, b = np.meshgrid(np.linspace(-4., 4., 33), np.linspace(-4., 4., 33))
        raw = np.column_stack([a.ravel(), b.ravel(), np.zeros(a.size)])
        scene = CudaLayeredMap(raw.copy(), cache_dir=self.output_run/'source_cache_cuda')
        normal = np.tile([0., 0., 1.], (len(raw), 1))
        ground = np.ones(len(raw), dtype=bool)
        invalid_quality = np.zeros(len(raw), dtype=np.float64)
        valid_quality = np.ones(len(raw), dtype=np.float64)
        sideways = np.tile([1., 0., 0.], (len(raw), 1))
        nonground = np.zeros(len(raw), dtype=bool)

        def pairs(n, g, q):
            return cuda_pairs(scene, raw, n, g, q, np.eye(4), 1., 0., .75, scene.config)

        def compare_fresh(n, g, q, expected_nonempty):
            cached = pairs(n, g, q)
            cache = scene.source_cache
            repeated = pairs(n, g, q)
            self.assertIs(scene.source_cache, cache)
            self.assertTrue(all(saved is current for saved, current in
                                zip(cache[0], (raw, n, g, q))))
            scene.source_cache = None
            fresh = pairs(n, g, q)
            for key in cached:
                np.testing.assert_array_equal(cached[key], fresh[key], err_msg=key)
                np.testing.assert_array_equal(cached[key], repeated[key], err_msg=key)
            self.assertEqual(len(cached['source_index']) > 0, expected_nonempty)
            return scene.source_cache

        previous = compare_fresh(normal, ground, invalid_quality, False)
        current = compare_fresh(normal, ground, valid_quality, True)
        self.assertIsNot(current, previous)
        previous = current
        current = compare_fresh(sideways, ground, valid_quality, False)
        self.assertIsNot(current, previous)
        compare_fresh(normal, ground, valid_quality, True)
        compare_fresh(normal, nonground, valid_quality, False)
        compare_fresh(normal, ground, valid_quality, True)
        # Equal-valued but distinct objects must also refresh without ndarray ==.
        previous = scene.source_cache
        pairs(normal.copy(), ground, valid_quality)
        self.assertIsNot(scene.source_cache, previous)
        previous = scene.source_cache
        pairs(normal, ground.copy(), valid_quality)
        self.assertIsNot(scene.source_cache, previous)


if __name__ == '__main__':
    unittest.main()

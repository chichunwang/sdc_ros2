"""學習式特徵軌跡與 SIFT 軌跡合併（learned_tracks.merge）的檢查：索引平移後兩組匹配各自指向原本的特徵。"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vslam'))

from gslam import learned_tracks  # noqa: E402


class MergeTest(unittest.TestCase):
    def test_merge_shifts_indices(self):
        ka = [np.array([[1.0, 1], [2, 2]]), np.array([[3.0, 3]])]
        kb = [np.array([[10.0, 10]]), np.array([[20.0, 20], [30, 30]])]
        la = [(0, 1, np.array([[1, 0]]))]
        lb = [(0, 1, np.array([[0, 1]]))]
        kpts, links = learned_tracks.merge(ka, la, kb, lb)
        self.assertEqual([len(k) for k in kpts], [3, 3])
        a, b, m = links[1]
        np.testing.assert_array_equal(kpts[a][m[0, 0]], [10, 10])
        np.testing.assert_array_equal(kpts[b][m[0, 1]], [30, 30])
        a, b, m = links[0]
        np.testing.assert_array_equal(kpts[a][m[0, 0]], [2, 2])
        np.testing.assert_array_equal(kpts[b][m[0, 1]], [3, 3])


if __name__ == '__main__':
    unittest.main()

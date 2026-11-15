"""GPU 點雲運算與配準的合成測試（需要 CuPy＋CUDA；沒有 GPU 時整個檔案 skip，不會假裝通過）。"""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vslam'))

try:
    import cupy as cp
    cp.cuda.Device().compute_capability
    HAVE_GPU = True
except Exception:  # noqa: BLE001
    HAVE_GPU = False


def scene(rng, n=20000, noise=0.01):
    """地面＋側牆＋前牆＋幾根柱子，三個方向都有約束。"""
    ground = np.c_[rng.uniform(-12, 12, n), rng.normal(1.5, noise, n), rng.uniform(2, 30, n)]
    side = np.c_[rng.normal(-6, noise, n), rng.uniform(-3, 1.5, n), rng.uniform(2, 30, n)]
    front = np.c_[rng.uniform(-6, 10, n), rng.uniform(-3, 1.5, n), rng.normal(28, noise, n)]
    # 對向的牆：只有單側平面時，尺度可被平移完全補償（尺度不可觀測），兩側都有面才能定出尺度
    side = np.vstack([side, np.c_[rng.normal(10, noise, n), rng.uniform(-3, 1.5, n), rng.uniform(2, 30, n)]])
    poles = np.c_[np.repeat(rng.uniform(0, 8, 20), 200) + rng.normal(0, 0.05, 4000), rng.uniform(-3, 1.5, 4000),
                  np.repeat(rng.uniform(5, 25, 20), 200) + rng.normal(0, 0.05, 4000)]
    return np.vstack([ground, side, front, poles])


@unittest.skipUnless(HAVE_GPU, '需要 CuPy 與 CUDA 裝置')
class GpuCloudTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(3)

    def test_voxel_downsample_matches_numpy(self):
        from gslam.gpu_cloud import voxel_downsample
        p = self.rng.uniform(-5, 5, (5000, 3))
        out = cp.asnumpy(voxel_downsample(cp.asarray(p), 0.5))
        keys = np.floor(p / 0.5).astype(np.int64)
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
        inverse = inverse.ravel()
        expected = np.stack([np.bincount(inverse, weights=p[:, a]) for a in range(3)], 1) / np.bincount(inverse)[:, None]
        self.assertEqual(len(out), len(expected))
        self.assertTrue(np.allclose(np.sort(out, axis=0), np.sort(expected, axis=0), atol=1e-9))

    def test_nearest_matches_bruteforce(self):
        from gslam.gpu_cloud import GridIndex
        p = self.rng.uniform(-10, 10, (4000, 3))
        q = self.rng.uniform(-11, 11, (700, 3))
        index = GridIndex(cp.asarray(p), 0.8)
        for radius in (0.5, 0.8, 2.0):   # 含半徑大於格寬的情況
            idx, d2 = index.nearest(cp.asarray(q), radius)
            idx, d2 = cp.asnumpy(idx), cp.asnumpy(d2)
            sorted_p = cp.asnumpy(index.points)
            brute = np.linalg.norm(q[:, None] - sorted_p[None], axis=2)
            best = brute.min(axis=1)
            found = best < radius
            self.assertTrue(np.array_equal(idx >= 0, found))
            self.assertTrue(np.allclose(np.sqrt(d2[found]), best[found], atol=1e-9))
            self.assertTrue(np.allclose(brute[np.flatnonzero(found), idx[found]], best[found], atol=1e-9))

    def test_radius_cov_and_normals_on_plane(self):
        from gslam.gpu_cloud import local_geometry
        p = np.c_[self.rng.uniform(-3, 3, 6000), self.rng.normal(0, 0.002, 6000), self.rng.uniform(-3, 3, 6000)]
        valid, normals, cov = local_geometry(cp.asarray(p), 0.5, 6)
        normals = cp.asnumpy(normals[valid])
        self.assertGreater(int(valid.sum()), 5000)
        self.assertGreater(np.median(np.abs(normals[:, 1])), 0.999)      # 法向應沿 y
        w = np.linalg.eigvalsh(cp.asnumpy(cov[valid])[0])
        self.assertTrue(np.allclose(w, [1e-3, 1.0, 1.0], atol=1e-9))     # GICP 正則化共變異數


@unittest.skipUnless(HAVE_GPU, '需要 CuPy 與 CUDA 裝置')
class RegistrationTest(unittest.TestCase):
    def setUp(self):
        from gslam import se3
        from gslam.gpu_cloud import TargetModel
        from gslam.gpu_registration import source_covariances
        self.se3 = se3
        rng = np.random.default_rng(4)
        world = scene(rng)
        self.T_true = se3.make(se3.so3_exp([0.02, -0.04, 0.01]), [0.30, 0.05, -0.45])   # camera-to-world
        cam = (scene(np.random.default_rng(5)) - self.T_true[:3, 3]) @ self.T_true[:3, :3]  # 另一組取樣，非同一批點
        self.model = TargetModel(cp.asarray(world), 1.0, 0.75, 6)
        valid, cov = source_covariances(cp.asarray(cam), 0.75, 6)
        self.src, self.cov = cp.asarray(cam)[valid], cov[valid]

    def check(self, method, **kw):
        from gslam.gpu_registration import register
        r = register(self.src, self.model, np.eye(4), method=method, src_cov=self.cov, **kw)
        dt, dr = self.se3.relative_error(self.T_true, r['T'])
        return r, dt, np.rad2deg(dr)

    def test_recovers_known_transform(self):
        # source→target 約定：結果 T 滿足 p_world = T · p_cam，也就是 camera-to-world
        for method in ('icp_plane', 'gicp'):
            r, dt, dr = self.check(method)
            self.assertTrue(self.se3.is_valid(r['T']), method)
            self.assertLess(dt, 0.03, method)
            self.assertLess(dr, 0.15, method)
            self.assertGreater(r['fitness'], 0.9, method)
            self.assertLess(r['inlier_rmse'], 0.2, method)

    def test_scale_estimation(self):
        from gslam.gpu_registration import register
        r = register(self.src / 1.03, self.model, np.eye(4), method='icp_plane', estimate_scale=True,
                     scale_prior_weight=0.0, stages=[dict(max_corr=1.5, iterations=40), dict(max_corr=0.4, iterations=40)])
        self.assertAlmostEqual(r['scale'], 1.03, delta=0.005)

    def test_failure_is_detectable(self):
        from gslam.gpu_registration import register
        far = self.se3.make(np.eye(3), [200.0, 0, 0])     # 完全不重疊
        r = register(self.src, self.model, far, method='gicp', src_cov=self.cov)
        self.assertEqual(r['reason'], 'insufficient_correspondences')
        self.assertLess(r['fitness'], 0.05)
        with self.assertRaises(ValueError):
            bad = np.eye(4); bad[0, 0] = 2.0
            register(self.src, self.model, bad, method='gicp', src_cov=self.cov)

    def test_gpu_matches_cpu_reference(self):
        """同一組資料，GPU GICP 與 CPU small_gicp 的結果應在公分等級內一致。"""
        try:
            import small_gicp
        except ImportError:
            self.skipTest('未安裝 small_gicp')
        r, _, _ = self.check('gicp')
        ref = small_gicp.align(cp.asnumpy(self.model.points), cp.asnumpy(self.src), init_T_target_source=np.eye(4),
                               registration_type='GICP', downsampling_resolution=0.25, max_correspondence_distance=1.0)
        dt, dr = self.se3.relative_error(ref.T_target_source, r['T'])
        self.assertLess(dt, 0.05)
        self.assertLess(np.rad2deg(dr), 0.3)


if __name__ == '__main__':
    unittest.main()

"""視覺前端的合成測試：GPU 描述子比對、像素→3D 查表、PnP 相對位姿、閉環投票（需要 CuPy＋CUDA 與 OpenCV）。"""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vslam'))

try:
    import cupy as cp
    cp.cuda.Device().compute_capability
    import cv2  # noqa: F401
    HAVE = True
except Exception:  # noqa: BLE001
    HAVE = False

K = np.array([[1000.0, 0, 720.0], [0, 1000.0, 464.0], [0, 0, 1.0]])
SIZE = (1440, 928)
CFG = dict(frontend='sift', max_keypoints=2000, ratio=0.8, mutual=True, essential_threshold_px=1.0,
           pnp_reprojection_px=2.0, pnp_iterations=300, min_pnp_inliers=40, min_pnp_inlier_ratio=0.35,
           min_essential_inliers=100, min_length_points=50, min_flow_px=2.0)


def project(points):
    return np.c_[K[0, 0] * points[:, 0] / points[:, 2] + K[0, 2], K[1, 1] * points[:, 1] / points[:, 2] + K[1, 2]]


@unittest.skipUnless(HAVE, '需要 CuPy、CUDA 裝置與 OpenCV')
class VisualTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(6)

    def test_descriptor_matching_recovers_permutation(self):
        from gslam.visual import match_descriptors
        d = self.rng.normal(size=(500, 128)).astype(np.float32)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        perm = self.rng.permutation(500)
        noisy = d[perm] + self.rng.normal(size=d.shape).astype(np.float32) * 0.01
        noisy /= np.linalg.norm(noisy, axis=1, keepdims=True)
        m = match_descriptors(cp.asarray(d), cp.asarray(noisy))
        self.assertGreater(len(m), 480)
        self.assertTrue(np.all(perm[m[:, 1]] == m[:, 0]))   # 每個匹配都指回同一個原始描述子

    def test_orb_normalization_orders_like_hamming(self):
        from gslam.visual import normalize_descriptors
        raw = self.rng.integers(0, 256, (40, 32), dtype=np.uint8)
        d = cp.asnumpy(normalize_descriptors(raw, 'orb'))
        self.assertTrue(np.allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-6))
        hamming = np.unpackbits(raw[:1] ^ raw, axis=1).sum(axis=1)
        self.assertTrue(np.allclose(d @ d[0], 1 - 2 * hamming / 256.0, atol=1e-5))

    def test_pixel_cloud_lookup(self):
        from gslam.visual import PixelCloud
        # 無序點雲（打亂順序），確認不依賴點索引；同一像素有前後兩點時取近的
        u, v = np.meshgrid(np.arange(0, 1440, 4.0), np.arange(0, 928, 4.0))
        z = 5.0 + 0.01 * u.ravel()
        pts = np.c_[(u.ravel() - K[0, 2]) / K[0, 0] * z, (v.ravel() - K[1, 2]) / K[1, 1] * z, z]
        behind = pts[:200] * 3.0          # 同一視線上更遠的點
        cloud = np.vstack([pts, behind])[self.rng.permutation(len(pts) + 200)]
        pix = PixelCloud(cloud, K, SIZE, stride=2, max_depth=100.0)
        query = np.c_[u.ravel()[:500], v.ravel()[:500]]
        found, valid = pix.lookup(query)
        self.assertTrue(valid.all())
        self.assertTrue(np.allclose(found, pts[:500], atol=1e-9))
        _, valid = pix.lookup(np.array([[2.0, 2.0]]))     # 沒有點投影到的格子
        self.assertFalse(valid[0])

    def test_relative_pose_from_matches(self):
        from gslam import se3
        from gslam.visual import estimate_relative_pose
        n = 400
        ref_pts = np.c_[self.rng.uniform(-6, 6, n), self.rng.uniform(-2, 2, n), self.rng.uniform(4, 25, n)]
        T_ref_cur = se3.make(se3.so3_exp([0.01, 0.05, -0.02]), [0.1, -0.02, 0.8])
        T_cur_ref = se3.inv(T_ref_cur)
        cur_pts = ref_pts @ T_cur_ref[:3, :3].T + T_cur_ref[:3, 3]
        kp_ref = project(ref_pts) + self.rng.normal(0, 0.3, (n, 2))
        kp_cur = project(cur_pts) + self.rng.normal(0, 0.3, (n, 2))
        matches = np.c_[np.arange(n), np.arange(n)]
        wrong = self.rng.choice(n, 80, replace=False)                 # 20% 錯誤匹配
        matches[wrong, 1] = self.rng.permutation(wrong)
        cur_cloud = cur_pts * 1.05                                     # 目前幀點雲尺度偏大 5%
        out = estimate_relative_pose(kp_ref, kp_cur, matches, ref_pts, np.ones(n, bool), K, CFG,
                                     cur_cloud, np.ones(n, bool), seed=0)
        self.assertTrue(out['ok'])
        dt, dr = se3.relative_error(T_ref_cur, out['T_ref_cur'])
        self.assertLess(dt, 0.05)
        self.assertLess(np.rad2deg(dr), 0.3)
        self.assertGreater(out['pnp_inliers'], 250)
        self.assertAlmostEqual(out['scale_ratio'], 1 / 1.05, delta=0.01)
        again = estimate_relative_pose(kp_ref, kp_cur, matches, ref_pts, np.ones(n, bool), K, CFG,
                                       cur_cloud, np.ones(n, bool), seed=0)
        self.assertTrue(np.array_equal(out['T_ref_cur'], again['T_ref_cur']))   # 固定 seed 可重現
        # 全是錯誤匹配時不可回報成功
        matches[:, 1] = self.rng.permutation(n)
        bad = estimate_relative_pose(kp_ref, kp_cur, matches, ref_pts, np.ones(n, bool), K, CFG, seed=0)
        self.assertFalse(bad['ok'])

    def test_essential_mode_tolerates_depth_error(self):
        # ref 幀深度每點有 8% 隨機誤差：PnP 內點變少，但 Essential 模式的旋轉不受影響、平移長度由中位數決定
        from gslam import se3
        from gslam.visual import estimate_relative_pose
        n = 600
        ref_pts = np.c_[self.rng.uniform(-6, 6, n), self.rng.uniform(-2, 2, n), self.rng.uniform(4, 25, n)]
        T_ref_cur = se3.make(se3.so3_exp([0.005, 0.06, -0.01]), [0.05, -0.01, 0.6])
        T_cur_ref = se3.inv(T_ref_cur)
        kp_ref = project(ref_pts) + self.rng.normal(0, 0.3, (n, 2))
        kp_cur = project(ref_pts @ T_cur_ref[:3, :3].T + T_cur_ref[:3, 3]) + self.rng.normal(0, 0.3, (n, 2))
        noisy = ref_pts * (1 + self.rng.normal(0, 0.08, (n, 1)))
        matches = np.c_[np.arange(n), np.arange(n)]
        out = estimate_relative_pose(kp_ref, kp_cur, matches, noisy, np.ones(n, bool), K, CFG, seed=0)
        self.assertTrue(out['ok'])
        self.assertEqual(out['mode'], 'essential')
        dt, dr = se3.relative_error(T_ref_cur, out['T_ref_cur'])
        self.assertLess(np.rad2deg(dr), 0.2)
        self.assertLess(dt, 0.05)
        self.assertLess(out['inlier_ratio'], 0.9)        # PnP 確實受深度誤差影響

    def test_pixel_cloud_lattice_interpolation(self):
        from gslam.visual import PixelCloud
        # 規則格點（每 4 px 一點）的斜平面：格點之間內插應精確；跨深度不連續處不給 3D
        u, v = np.meshgrid(np.arange(2.0, 1440, 4.0), np.arange(2.0, 928, 4.0))
        z = 5.0 + 0.004 * u.ravel() + 0.002 * v.ravel()
        z[(u.ravel() > 700) & (u.ravel() < 720)] *= 2.0     # 一條深度跳變帶
        pts = np.c_[(u.ravel() - K[0, 2]) / K[0, 0] * z, (v.ravel() - K[1, 2]) / K[1, 1] * z, z]
        pix = PixelCloud(pts[self.rng.permutation(len(pts))], K, SIZE, max_depth=100.0)
        self.assertTrue(pix.lattice)
        q = np.array([[101.3, 57.7], [400.0, 300.5], [699.0, 200.0], [1.0, 1.0]])
        found, valid = pix.lookup(q)
        self.assertTrue(np.array_equal(valid, [True, True, False, False]))
        expect = 5.0 + 0.004 * q[:2, 0] + 0.002 * q[:2, 1]
        self.assertTrue(np.allclose(found[:2, 2], expect, atol=1e-6))
        self.assertTrue(np.allclose(project(found[:2]), q[:2], atol=1e-6))    # 3D 點投影回特徵點本身

    def test_vote_candidates(self):
        from gslam.visual import vote_candidates
        bank = self.rng.normal(size=(3000, 128)).astype(np.float32)
        bank /= np.linalg.norm(bank, axis=1, keepdims=True)
        owner = np.repeat(np.arange(6), 500)
        query = bank[owner == 4][:300] + self.rng.normal(size=(300, 128)).astype(np.float32) * 0.01
        query /= np.linalg.norm(query, axis=1, keepdims=True)
        votes = vote_candidates(cp.asarray(query), cp.asarray(bank), cp.asarray(owner), 6, 0.8, chunk=700)
        self.assertEqual(int(np.argmax(votes)), 4)
        self.assertEqual(int(votes[4]), 300)
        self.assertEqual(int(votes.sum()), 300)

    def test_sift_on_synthetic_image_and_debug_image(self):
        from gslam.visual import ClassicFrontend, draw_matches
        img = (self.rng.uniform(0, 255, (120, 160)).astype(np.uint8))
        img = cv2.resize(cv2.GaussianBlur(img, (0, 0), 1.5), (640, 480), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        shifted = np.roll(img, 12, axis=1)                             # 向右平移 12 px
        fe = ClassicFrontend(CFG)
        fa, fb = fe.extract(img), fe.extract(shifted)
        self.assertGreater(len(fa), 100)
        m = fe.match(fa, fb)
        self.assertGreater(len(m), 50)
        flow = fb.kpts[m[:, 1]] - fa.kpts[m[:, 0]]
        self.assertLess(abs(np.median(flow[:, 0]) - 12.0), 0.5)
        self.assertLess(abs(np.median(flow[:, 1])), 0.5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'matches.jpg'
            draw_matches(img, shifted, fa.kpts, fb.kpts, m, np.ones(len(m), bool), path, text='test')
            self.assertGreater(path.stat().st_size, 1000)


if __name__ == '__main__':
    unittest.main()

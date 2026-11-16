"""bundle adjustment 的合成檢查：已知相機軌跡與 3D 點，位姿加擾動後應被重投影最佳化拉回來。"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vslam'))

from gslam import bundle, se3  # noqa: E402

K = np.array([[1040.0, 0, 720.0], [0, 1038.0, 464.0], [0, 0, 1]])
CFG = dict(min_track_length=3, min_tracks=50, min_parallax_deg=1.0, triangulation_gate_px=20.0, max_observations=0,
           pixel_sigma=1.0, huber_k=1.345, prune_px=4.0, rounds=2, max_iterations=40, relative_error_tol=1e-6,
           prior_sigma=1e-4, lap_gap_frames=200)


def scene(n_frames=14, n_points=400, seed=0):
    rng = np.random.default_rng(seed)
    poses = []
    for i in range(n_frames):   # 往前開、略為轉向
        R = se3.so3_exp(np.array([0.0, 0.02 * i, 0.0]))
        poses.append(se3.make(R, np.array([0.03 * i * i * 0.1, 0.0, 0.3 * i])))
    pts = np.c_[rng.uniform(-8, 8, n_points), rng.uniform(-2, 2, n_points), rng.uniform(4, 25, n_points)]
    kpts, ids = [], []
    for T in poses:
        Xc = (pts - T[:3, 3]) @ T[:3, :3]
        uv = np.c_[K[0, 0] * Xc[:, 0] / Xc[:, 2] + K[0, 2], K[1, 1] * Xc[:, 1] / Xc[:, 2] + K[1, 2]]
        vis = np.flatnonzero((Xc[:, 2] > 1) & (uv[:, 0] > 0) & (uv[:, 0] < 1440) & (uv[:, 1] > 0) & (uv[:, 1] < 928))
        vis = rng.permutation(vis)
        kpts.append(uv[vis] + rng.normal(0, 0.3, (len(vis), 2)))
        ids.append(vis)
    links = []
    for i in range(1, n_frames):
        common, ia, ib = np.intersect1d(ids[i - 1], ids[i], return_indices=True)
        links.append((i - 1, i, np.c_[ia, ib]))
    return poses, kpts, links


class BundleTest(unittest.TestCase):
    def test_tracks(self):
        links = [(0, 1, np.array([[0, 0], [1, 1]])), (1, 2, np.array([[0, 2], [1, 0]])),
                 (2, 3, np.array([[2, 1], [0, 1]]))]   # 最後一組讓兩條軌跡在第 3 幀撞在同一特徵 → 合併後第 2 幀重複 → 丟掉
        t, f, k, n = bundle.build_tracks([3, 3, 3, 3], links[:2], 3)
        self.assertEqual(n, 2)
        self.assertEqual(sorted(zip(t.tolist(), f.tolist(), k.tolist())),
                         sorted([(0, 0, 0), (0, 1, 0), (0, 2, 2), (1, 0, 1), (1, 1, 1), (1, 2, 0)]))
        self.assertEqual(bundle.build_tracks([3, 3, 3, 3], links, 3)[3], 0)

    def test_triangulation(self):
        poses, kpts, links = scene()
        t, f, k, n = bundle.build_tracks([len(x) for x in kpts], links, 3)
        off = np.concatenate([[0], np.cumsum([len(x) for x in kpts])])
        uv = np.concatenate(kpts)[off[f] + k]
        X, err, z, parallax = bundle.triangulate(K, poses, t, f, uv, n)
        self.assertLess(np.median(err), 1.0)
        self.assertTrue(np.all(z[parallax[t] > 1.0] > 0))

    def test_recovers_perturbed_poses(self):
        import gtsam
        poses, kpts, links = scene()
        rng = np.random.default_rng(1)
        noisy = [poses[0]] + [se3.left_update(T, np.r_[rng.normal(0, 0.03, 3), rng.normal(0, 0.004, 3)]) for T in poses[1:]]
        # 很鬆的里程計邊（用加了擾動的相對位姿），只負責固定尺度
        loose = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05] * 3 + [0.2] * 3))
        between = [(i - 1, i, se3.inv(noisy[i - 1]) @ noisy[i], loose) for i in range(1, len(poses))]
        out, info = bundle.run(K, noisy, kpts, links, between, CFG, log=lambda *_: None)
        self.assertEqual(info['status'], 'ok')
        self.assertLess(info['rounds'][-1]['reprojection_px']['median'], 0.6)

        def sim3_rmse(est):   # 單目 BA 的整體尺度不定，所以用 Sim(3) 對齊後比位置
            a = np.array([T[:3, 3] for T in est]); b = np.array([T[:3, 3] for T in poses])
            ma, mb = a.mean(0), b.mean(0)
            U, S, Vt = np.linalg.svd((b - mb).T @ (a - ma) / len(a))
            D = np.diag([1, 1, np.sign(np.linalg.det(U) * np.linalg.det(Vt))])
            R = U @ D @ Vt
            s = np.trace(np.diag(S) @ D) / ((a - ma) ** 2).sum() * len(a)
            return float(np.sqrt(np.mean(np.sum((s * (a - ma) @ R.T + mb - b) ** 2, axis=1))))
        before, after = sim3_rmse(noisy), sim3_rmse(out)
        self.assertGreater(before, 0.02)
        self.assertLess(after, 0.005)


if __name__ == '__main__':
    unittest.main()

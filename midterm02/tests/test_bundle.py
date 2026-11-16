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


def scene(n_frames=14, n_points=400, seed=0, return_ids=False):
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
    if return_ids:
        return poses, kpts, links, ids
    return poses, kpts, links


class BundleTest(unittest.TestCase):
    def test_tracks(self):
        links = [(0, 1, np.array([[0, 0], [1, 1]])), (1, 2, np.array([[0, 2], [1, 0]])),
                 (2, 3, np.array([[2, 1], [0, 1]]))]   # 最後一組讓兩條軌跡在第 3 幀撞在同一特徵 → 合併後第 2 幀重複 → 丟掉
        t, f, k, n = bundle.build_tracks([3, 3, 3, 3], links[:2], 3)
        self.assertEqual(n, 2)
        self.assertEqual(sorted(zip(t.tolist(), f.tolist(), k.tolist())),
                         sorted([(0, 0, 0), (0, 1, 0), (0, 2, 2), (1, 0, 1), (1, 1, 1), (1, 2, 0)]))
        self.assertEqual(bundle.build_tracks([3, 3, 3, 3], links, 3, resolve=False)[3], 0)
        # resolve=True：會造成同一幀兩個特徵的那條匹配被跳過，兩條軌跡都保留（先合併的那條多一個觀測）
        t, f, k, n = bundle.build_tracks([3, 3, 3, 3], links, 3)
        self.assertEqual(n, 2)
        self.assertEqual(sorted(np.bincount(t).tolist()), [3, 4])
        frames_per_track = [f[t == j] for j in range(n)]
        self.assertTrue(all(len(np.unique(x)) == len(x) for x in frames_per_track))

    def test_size_weight(self):
        w = dict(enabled=True, size_ref=6.0, power=0.4, lo=0.6, hi=2.5)
        sig = bundle.observation_sigma(np.array([1.0, 6.0, 24.0, 1000.0]), dict(size_weight=w))
        np.testing.assert_allclose(sig, [0.6, 1.0, 4 ** 0.4, 2.5])
        self.assertIsNone(bundle.observation_sigma(np.ones(3), dict(size_weight=dict(w, enabled=False))))
        # 加權後仍要解得回來：一半特徵尺度大（權重低）且雜訊大
        import gtsam
        poses, kpts, links = scene()
        rng = np.random.default_rng(2)
        sizes = [np.where(np.arange(len(k)) % 2 == 0, 3.0, 40.0) for k in kpts]
        kpts = [k + rng.normal(0, 1.5, k.shape) * (s[:, None] > 10) for k, s in zip(kpts, sizes)]
        noisy = [poses[0]] + [se3.left_update(T, np.r_[rng.normal(0, 0.03, 3), rng.normal(0, 0.004, 3)]) for T in poses[1:]]
        loose = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05] * 3 + [0.2] * 3))
        between = [(i - 1, i, se3.inv(noisy[i - 1]) @ noisy[i], loose) for i in range(1, len(poses))]
        out, info = bundle.run(K, noisy, kpts, links, between, dict(CFG, size_weight=w), log=lambda *_: None, sizes=sizes)
        self.assertTrue(info['size_weighted'])
        self.assertEqual(info['status'], 'ok')

    def test_guided_observations(self):
        # 第 7 幀的配對全部拿掉：它的觀測原本不屬於任何軌跡；投影引導應該把它們接回「同一個 3D 點」的軌跡
        poses, kpts, links, ids = scene(return_ids=True)
        rng = np.random.default_rng(4)
        point_desc = rng.normal(size=(400, 32))
        point_desc /= np.linalg.norm(point_desc, axis=1, keepdims=True)
        descs = []
        for i in range(len(kpts)):
            d = point_desc[ids[i]] + rng.normal(0, 0.05, (len(ids[i]), 32))
            descs.append((d / np.linalg.norm(d, axis=1, keepdims=True)).astype(np.float32))
        cut = [l for l in links if 7 not in (l[0], l[1])]
        counts = [len(k) for k in kpts]
        t, f, k, n = bundle.build_tracks(counts, cut, 3)
        off = np.concatenate([[0], np.cumsum(counts)])
        node = off[f] + k
        X, _, _, _ = bundle.triangulate(K, poses, t, f, np.concatenate(kpts)[node], n)
        self.assertEqual(np.sum(f == 7), 0)
        g = dict(radius_m=5.0, max_angle_deg=30.0, search_px=3.0, min_similarity=0.85, similarity_margin=0.05,
                 max_depth_m=40.0, lap_gap_frames=200)
        nt, nf, nn, stat = bundle.guided_observations(K, poses, X, t, f, node, np.concatenate(kpts), off, descs, g)
        self.assertGreater(np.sum(nf == 7), 100)
        # 正確性：補上的觀測，其特徵對應的 3D 點編號要和軌跡原本的 3D 點相同
        track_point = np.full(n, -1)
        track_point[t] = np.concatenate(ids)[node]
        got_point = np.concatenate(ids)[nn]
        self.assertGreater(np.mean(got_point == track_point[nt]), 0.98)

    def test_revisit_pairs(self):
        # 兩圈半徑 5 m 的圓，每圈 100 幀；第二圈應配到第一圈同一角度的幀，前 60 幀內的鄰居不算
        poses = []
        for i in range(200):
            a = 2 * np.pi * i / 100
            R = se3.so3_exp(np.array([0.0, -a, 0.0]))
            poses.append(se3.make(R, np.array([5 * np.cos(a), 0.0, 5 * np.sin(a)]) * (1.0 + 0.02 * (i >= 100))))
        pairs = bundle.revisit_pairs(poses, 60, 1.0, 0.9)
        self.assertTrue(set((i - 100, i) for i in range(100, 200)) <= set(pairs))
        self.assertTrue(all(i - j >= 60 and np.linalg.norm(poses[i][:3, 3] - poses[j][:3, 3]) < 1.0 for j, i in pairs))
        self.assertEqual(len(set(i for _, i in pairs if i >= 103 and i < 197)), len([p for p in pairs if 103 <= p[1] < 197]))
        self.assertEqual(bundle.revisit_pairs(poses, 250, 1.0, 0.9), [])

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

"""BA 自校正（bundle.calibration）的合成檢查：影像帶徑向畸變時，針孔 BA 殘差偏大；自校正應估回 k1、k2，殘差回到雜訊等級。"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vslam'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gslam import bundle, se3  # noqa: E402
from test_bundle import CFG, K, scene  # noqa: E402

K1, K2 = -0.06, 0.09


def distort(uv):
    x = (uv[:, 0] - K[0, 2]) / K[0, 0]
    y = (uv[:, 1] - K[1, 2]) / K[1, 1]
    r2 = x * x + y * y
    g = 1 + K1 * r2 + K2 * r2 * r2
    return np.c_[K[0, 0] * x * g + K[0, 2], K[1, 1] * y * g + K[1, 2]]


class SelfCalibTest(unittest.TestCase):
    def test_project_matches_gtsam(self):
        import gtsam
        cal = gtsam.Cal3DS2(K[0, 0], K[1, 1], 0.0, K[0, 2], K[1, 2], K1, K2, 0.0, 0.0)
        X = np.array([[1.0, -0.5, 4.0], [-3.0, 1.0, 6.0]])
        uv, _ = bundle.project(K, np.stack([np.eye(3)] * 2), np.zeros((2, 3)), X, (K1, K2))
        for x, u in zip(X, uv):
            np.testing.assert_allclose(cal.uncalibrate(gtsam.Point2(x[0] / x[2], x[1] / x[2])), u, atol=1e-9)

    def test_recovers_distortion(self):
        import gtsam
        poses, kpts, links = scene(n_frames=20, n_points=800, seed=3)
        kpts = [distort(k) for k in kpts]
        loose = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05] * 3 + [0.2] * 3))
        between = [(i - 1, i, se3.inv(poses[i - 1]) @ poses[i], loose) for i in range(1, len(poses))]
        cfg = dict(CFG, triangulation_gate_px=50.0, prune_px=50.0)
        out_plain, plain = bundle.run(K, poses, kpts, links, between, cfg, log=lambda *_: None)
        ccfg = dict(enabled=True, solves=3, sigmas=[1e-6, 1e-6, 1e-6, 5.0, 5.0, 0.5, 0.5, 1e-6, 1e-6])
        out_cal, info = bundle.run(K, poses, kpts, links, between, dict(cfg, calibration=ccfg), log=lambda *_: None)
        c = info['calibration']
        self.assertAlmostEqual(c['k1'], K1, delta=0.02)
        self.assertAlmostEqual(c['k2'], K2, delta=0.05)
        # 針孔 BA 會把畸變吸收進結構與軌跡：殘差看起來只略大於雜訊，但位置被扭曲；自校正後位置誤差明顯變小
        self.assertGreater(plain['rounds'][-1]['reprojection_px']['median'], info['rounds'][-1]['reprojection_px']['median'])
        true = np.stack([T[:3, 3] for T in poses])

        def ate(est):
            P = np.stack([T[:3, 3] for T in est])
            mu_p, mu_t = P.mean(0), true.mean(0)
            U, S, Vt = np.linalg.svd((true - mu_t).T @ (P - mu_p))
            D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
            R = U @ D @ Vt
            s = np.trace(np.diag(S) @ D) / np.sum((P - mu_p) ** 2)
            return np.sqrt(np.mean(np.sum((s * (P - mu_p) @ R.T + mu_t - true) ** 2, 1)))
        self.assertLess(ate(out_cal), 0.5 * ate(out_plain))


if __name__ == '__main__':
    unittest.main()

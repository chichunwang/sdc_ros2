"""IMU 平均速度變化模型（gslam/imu.py）的合成檢查：已知軌跡、槓桿臂、偏差與重力，合成加速度計讀數。"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vslam'))

from gslam import imu, se3  # noqa: E402

LAM = 1.3                                   # 公尺 / 軌跡單位
LEVER = np.array([0.0, 0.0, 2.8])           # IMU→相機（相機座標）
BIAS = np.array([0.05, -0.03, 0.02])        # 加速度計偏差（相機座標）
G = np.array([0.0, 9.81, 0.0])              # 世界座標 = 第一幀相機軸（y 向下）


def camera(t):
    """相機中心（公尺）與相機旋轉：平面上蛇行，車頭沿速度方向。"""
    c = np.stack([3.0 * np.sin(0.25 * t), 0.02 * np.sin(0.7 * t), 1.5 * t + 0.2 * np.sin(0.5 * t)], -1)
    vx, vz = 0.75 * np.cos(0.25 * t), 1.5 + 0.1 * np.cos(0.5 * t)
    yaw = np.arctan2(vx, vz)
    R = np.stack([se3.so3_exp(np.array([0.0, y, 0.0])) for y in np.atleast_1d(yaw)])
    return c, R


def synth(duration=40.0, frame_hz=10.0, imu_hz=100.0):
    tf = np.arange(0, duration, 1 / frame_hz)
    ti = np.arange(-0.5, duration + 0.5, 1 / imu_hz)
    c, R = camera(tf)
    eps = 1e-3

    def imu_pos(t):
        cc, RR = camera(t)
        return cc - np.einsum('nij,j->ni', RR, LEVER)
    acc = (imu_pos(ti + eps) - 2 * imu_pos(ti) + imu_pos(ti - eps)) / eps ** 2
    _, Ri = camera(ti)
    f = np.einsum('nji,nj->ni', Ri, acc - G) + BIAS     # 比力：R^T (a − g) + b
    poses = np.stack([se3.make(R[k], c[k] / LAM) for k in range(len(tf))])
    return (tf * 1e9).astype(np.int64) + 10 ** 18, (ti * 1e9).astype(np.int64) + 10 ** 18, f, poses


class ImuModelTest(unittest.TestCase):
    def setUp(self):
        self.ct, self.it, self.f, self.poses = synth()
        self.win = imu.Windows(self.ct, self.it, self.f, 5, offset_s=0.0)

    def test_fit_recovers_scale_lever_bias(self):
        r = imu.consistency(self.win, self.poses)
        self.assertAlmostEqual(r['lam'], LAM, delta=0.01)
        np.testing.assert_allclose(r['lever'][[0, 2]], LEVER[[0, 2]], atol=0.05)
        # 只有偏航時，垂直方向的重力與加速度計偏差分不開（只有和可觀），所以只檢查水平分量
        np.testing.assert_allclose(r['g'][[0, 2]], G[[0, 2]], atol=0.05)
        self.assertAlmostEqual(r['g'][1] - r['b'][1], G[1] - BIAS[1], delta=0.05)
        self.assertLess(r['rms_h'], 0.01)

    def test_local_stretch_is_detected(self):
        P = self.poses.copy()
        d = np.diff(P[:, :3, 3], axis=0)
        d[150:250] *= 1.08                                   # 一段路的長度多算 8%
        P[1:, :3, 3] = P[0, :3, 3] + np.cumsum(d, axis=0)
        base = imu.consistency(self.win, self.poses)['rms_h']
        bad = imu.consistency(self.win, P)['rms_h']
        self.assertGreater(bad, base + 0.02)

    def test_factor_zero_at_truth_and_jacobian(self):
        import gtsam
        p, R = imu.split_poses(self.poses)
        k1, k2 = imu.factor_pairs(len(p), 5)
        fit = self.win.fit(p, R, k1, k2)
        add, info = imu.velocity_factors(self.win, self.poses, k1, k2, 0.02, 1.345, fit)
        graph = gtsam.NonlinearFactorGraph()
        add(graph, len(p))
        values = gtsam.Values()
        for i, T in enumerate(self.poses):
            values.insert(i, gtsam.Pose3(T))
        self.assertEqual(info['factors'], graph.size())
        self.assertLess(graph.error(values), 1e-2 * graph.size())
        # 數值微分檢查第一個因子的 Jacobian（在擾動後的點上）
        fac = graph.at(0)
        keys = list(fac.keys())
        v = gtsam.Values(values)
        rng = np.random.default_rng(0)
        for k in keys:
            v.update(k, values.atPose3(k).retract(rng.normal(0, 0.01, 6)))
        e0 = fac.unwhitenedError(v)
        A, _ = fac.linearize(v).jacobian()
        sigma = 0.02 / info['lam']
        for q, k in enumerate(keys):
            for j in range(6):
                dv = np.zeros(6)
                dv[j] = 1e-6
                v2 = gtsam.Values(v)
                v2.update(k, v.atPose3(k).retract(dv))
                num = (fac.unwhitenedError(v2) - e0) / 1e-6
                np.testing.assert_allclose(A[:, 6 * q + j] * sigma, num, atol=1e-4 * max(1.0, np.abs(num).max()))

    def test_block_mask(self):
        m = imu.block_mask(200, keep_even=True)
        lo = np.array([0, 190, 210, 400])
        hi = np.array([30, 220, 240, 430])
        np.testing.assert_array_equal(m(lo, hi), [True, False, False, True])


if __name__ == '__main__':
    unittest.main()

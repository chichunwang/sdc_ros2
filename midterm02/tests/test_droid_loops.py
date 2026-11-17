"""scripts/droid_loops.py 的合成檢查：pose graph 用閉環把有偏航漂移的往返軌跡拉回。"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
try:
    from droid_loops import inv, pose_graph, _exp
    from scipy.spatial.transform import Rotation
    HAVE = True
except ImportError:   # 主容器若沒有 scipy／cv2 就略過
    HAVE = False


def route(n_leg=200, step=1.0, yaw_drift=0.0):
    """沿 z 前進 n_leg 步、原地迴轉 180°、再走回 n_leg 步（相機座標：x 右、y 下、z 前；偏航繞 y 軸）。"""
    T = [np.eye(4)]
    rel = []
    for k in range(2 * n_leg + 20):
        D = np.eye(4)
        if n_leg <= k < n_leg + 20:
            D[:3, :3] = Rotation.from_rotvec([0, np.pi / 20, 0]).as_matrix()
            D[2, 3] = 0.1
        else:
            D[2, 3] = step
        D[:3, :3] = Rotation.from_rotvec([0, yaw_drift, 0]).as_matrix() @ D[:3, :3]
        rel.append(D)
        T.append(T[-1] @ D)
    return np.stack(T)


@unittest.skipUnless(HAVE, 'needs scipy and cv2')
class PoseGraphTest(unittest.TestCase):
    def test_loop_pulls_back_drift(self):
        gt = route()
        est = route(yaw_drift=0.0015)      # 每步 0.086° 偏航漂移
        N = len(gt)
        Z = inv(gt[-1]) @ gt[0]            # 終點→起點的真實相對位姿（量測）
        gap0 = np.linalg.norm(est[-1, :3, 3] - est[0, :3, 3] - (gt[-1, :3, 3] - gt[0, :3, 3]))
        out, info = pose_graph(est, [(N - 1, 0, Z)], 0.002, 0.01, 0.005, 0.05)
        err0 = np.linalg.norm(inv(est[-1]) @ est[0] - Z)
        err1 = np.linalg.norm(inv(out[-1]) @ out[0] - Z)
        self.assertGreater(gap0, 10.0)
        self.assertLess(err1, 0.05 * err0)
        # 修正後整條軌跡更接近真值（各自以起點為原點）
        d0 = np.linalg.norm(est[:, :3, 3] - gt[:, :3, 3], axis=1).max()
        d1 = np.linalg.norm(out[:, :3, 3] - gt[:, :3, 3], axis=1).max()
        self.assertLess(d1, 0.5 * d0)

    def test_per_edge_sigma_only_tight_axes(self):
        """P1 的反向約束：每條邊自己的 6 維標準差。只約束垂直（b 相機 y）時，量測的前進方向（z）誤差不應被拉過去。"""
        est = route()
        N = len(est)
        a, b = 50, N - 51                     # 去程與回程同一處（反向）
        Z = inv(est[a]) @ est[b]
        Z[1, 3] += 0.5 * np.sign(Z[1, 3] or 1.0)   # 量測說 b 比目前低／高 0.5（垂直）
        Zt = Z.copy()
        Zt[:3, 3] += Z[:3, :3] @ np.array([0.0, 0.0, 3.0])   # 另外在 b 的前進方向多 3（不可信的分量）
        sigma = [0.01, 0.3, 0.01, 0.5, 0.02, 50.0]
        out, _ = pose_graph(est, [(a, b, Zt, sigma)], 0.002, 0.05, 0.005, 0.05, max_nfev=100)
        cur = inv(out[a]) @ out[b]
        res = inv(Zt) @ cur
        self.assertLess(abs(res[1, 3]), 0.05)            # 垂直被拉到量測值
        self.assertGreater(abs(res[2, 3]), 2.0)          # 前進方向幾乎沒動（σ 50）

    def test_no_loop_no_change(self):
        est = route(yaw_drift=0.001)
        out, info = pose_graph(est, [], 0.002, 0.01, 0.005, 0.05)
        self.assertLess(np.abs(out - est).max(), 1e-9)


if __name__ == '__main__':
    unittest.main()

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

    def test_no_loop_no_change(self):
        est = route(yaw_drift=0.001)
        out, info = pose_graph(est, [], 0.002, 0.01, 0.005, 0.05)
        self.assertLess(np.abs(out - est).max(), 1e-9)


if __name__ == '__main__':
    unittest.main()

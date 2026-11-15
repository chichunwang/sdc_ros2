"""座標與 SE(3) 約定、pose 輸出格式的測試（只需 NumPy，不需 GPU 或官方資料）。"""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vslam'))
sys.path.insert(0, str(ROOT / 'scripts'))
from gslam import se3  # noqa: E402


def random_pose(rng, angle=1.0, trans=5.0):
    return se3.make(se3.so3_exp(rng.normal(size=3) * angle), rng.normal(size=3) * trans)


class Se3Test(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)

    def test_exp_log_roundtrip_and_validity(self):
        for _ in range(50):
            w = self.rng.normal(size=3)
            w *= self.rng.uniform(0, 3.1) / np.linalg.norm(w)
            R = se3.so3_exp(w)
            self.assertTrue(np.allclose(R.T @ R, np.eye(3), atol=1e-12))
            self.assertAlmostEqual(np.linalg.det(R), 1.0, places=12)
            self.assertTrue(np.allclose(se3.so3_log(R), w, atol=1e-7))

    def test_inverse_and_composition(self):
        A, B = random_pose(self.rng), random_pose(self.rng)
        self.assertTrue(np.allclose(se3.inv(A) @ A, np.eye(4), atol=1e-12))
        self.assertTrue(np.allclose(se3.inv(A @ B), se3.inv(B) @ se3.inv(A), atol=1e-12))

    def test_camera_to_world_convention(self):
        # T_wc 把相機座標的點帶到世界座標；相機原點在世界的位置就是平移欄
        T_wc = se3.make(se3.so3_exp([0, np.pi / 2, 0]), [1.0, 2.0, 3.0])
        self.assertTrue(np.allclose((T_wc @ np.array([0, 0, 0, 1.0]))[:3], [1, 2, 3]))
        # 相機繞 y 軸轉 +90 度後，相機的前方 (z) 指向世界的 +x
        self.assertTrue(np.allclose(T_wc[:3, :3] @ np.array([0, 0, 1.0]), [1, 0, 0], atol=1e-12))
        # 相對運動：T_w_cur = T_w_prev @ T_prev_cur
        T_prev_cur = se3.make(np.eye(3), [0, 0, 1.0])  # 往相機前方走 1 m
        self.assertTrue(np.allclose((T_wc @ T_prev_cur)[:3, 3], [2, 2, 3], atol=1e-12))

    def test_left_update_matches_first_order_jacobian(self):
        T = random_pose(self.rng)
        p = self.rng.normal(size=3)
        q = T[:3, :3] @ p + T[:3, 3]
        delta = self.rng.normal(size=6) * 1e-6
        T2 = se3.left_update(T, delta)
        predicted = q + delta[:3] + np.cross(delta[3:], q)   # ∂q/∂v = I，∂q/∂ω = −[q]×
        self.assertTrue(np.allclose(T2[:3, :3] @ p + T2[:3, 3], predicted, atol=1e-10))

    def test_project_rotation_and_is_valid(self):
        T = random_pose(self.rng)
        noisy = T.copy()
        noisy[:3, :3] += self.rng.normal(size=(3, 3)) * 1e-3
        self.assertFalse(se3.is_valid(noisy))
        self.assertTrue(se3.is_valid(se3.normalize(noisy)))
        bad = T.copy(); bad[0, 3] = np.nan
        self.assertFalse(se3.is_valid(bad))

    def test_interpolate_endpoints_and_midpoint(self):
        A = se3.make(np.eye(3), [0, 0, 0])
        B = se3.make(se3.so3_exp([0, 0.4, 0]), [2.0, 0, 0])
        self.assertTrue(np.allclose(se3.interpolate(A, B, 0.0), A))
        self.assertTrue(np.allclose(se3.interpolate(A, B, 1.0), B))
        M = se3.interpolate(A, B, 0.5)
        self.assertTrue(np.allclose(se3.so3_log(M[:3, :3]), [0, 0.2, 0], atol=1e-12))
        self.assertTrue(se3.is_valid(M))

    def test_distribute_corrections(self):
        # 直線 odometry，每 5 幀一個 keyframe；keyframe 被整體平移後，所有幀都應跟著平移且保持合法
        odom = [se3.make(np.eye(3), [0, 0, 0.1 * i]) for i in range(23)]
        kf_idx = [0, 5, 10, 15, 20]
        shift = se3.make(se3.so3_exp([0, 0.3, 0]), [1.0, 0, 0])
        kf_opt = [shift @ odom[i] for i in kf_idx]
        out = se3.distribute_corrections(odom, kf_idx, kf_opt)
        self.assertEqual(len(out), len(odom))
        for i, T in enumerate(out):
            self.assertTrue(se3.is_valid(T))
            self.assertTrue(np.allclose(T, shift @ odom[i], atol=1e-9))
        # keyframe 上必須等於最佳化值；非剛性修正時相鄰幀之間不能跳動
        kf_opt[2] = se3.make(np.eye(3), [0.2, 0, 0]) @ kf_opt[2]
        out = se3.distribute_corrections(odom, kf_idx, kf_opt)
        for k, i in enumerate(kf_idx):
            self.assertTrue(np.allclose(out[i], kf_opt[k], atol=1e-9))
        steps = [np.linalg.norm(out[i + 1][:3, 3] - out[i][:3, 3]) for i in range(len(out) - 1)]
        self.assertLess(max(steps), 0.2)


class PoseCsvTest(unittest.TestCase):
    def test_written_poses_pass_official_check(self):
        import check_submission
        from gslam import outputs
        rng = np.random.default_rng(2)
        ts = ['1741574911912996024', '1741574912012996024', '1741574912112996024']
        poses = [random_pose(rng) for _ in ts]
        with tempfile.TemporaryDirectory() as tmp:
            sub, sample = Path(tmp) / 'result_1.csv', Path(tmp) / 'sample.csv'
            outputs.write_poses(sub, ts, poses)
            outputs.write_poses(sample, ts, [np.eye(4)] * 3)
            self.assertEqual(check_submission.check(str(sub), str(sample)), [])
            rows = sub.read_text().strip().split('\n')
            self.assertEqual(rows[0], ','.join(check_submission.HEADER))
            first = rows[1].split(',')
            self.assertEqual(first[0], ts[0])                      # timestamp 保持整數字串
            self.assertEqual(first[13:], ['0.0', '0.0', '0.0', '1.0'])
            self.assertTrue(np.allclose(np.array(first[1:13], dtype=float).reshape(3, 4), poses[0][:3]))  # row-major
            outputs.write_poses(sub, ts[:2], poses[:2])            # 少一幀必須被檢查出來
            self.assertTrue(any('缺少' in e for e in check_submission.check(str(sub), str(sample))))
            with self.assertRaises(ValueError):
                bad = poses[0].copy(); bad[0, 0] = np.inf
                outputs.write_poses(sub, ts[:1], [bad])


if __name__ == '__main__':
    unittest.main()

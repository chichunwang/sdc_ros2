from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinematic_graph import KinematicProblem


class KinematicGraphTests(unittest.TestCase):
    def data(self, duration=20.):
        t = np.arange(0., duration+.01, .25); it = np.arange(0., duration+.001, .01)
        omega = .035
        xyz = np.column_stack((10*np.sin(omega*t)/omega, 10*(1-np.cos(omega*t))/omega))
        q = Rotation.from_euler('z', (omega*it-.05)[:, None]).as_quat()
        gyro = np.column_stack((it*0, it*0, it*0+omega+.004))
        return t, xyz, it, q, gyro

    def test_analytic_sparse_jacobian_including_robust_gnss(self):
        args = self.data(5.); problem = KinematicProblem(*args)
        state = problem.initial.copy(); state[5, 0] += 2
        x = state.ravel(); jac = problem.evaluate(x, True).toarray()
        for column in (0, 4, 5, 30, 33, 34, 35, 40, len(x)-1):
            shift = np.zeros(len(x)); shift[column] = 1e-6
            finite = (problem.evaluate(x+shift)-problem.evaluate(x-shift))/(2e-6)
            np.testing.assert_allclose(jac[:, column], finite, atol=2e-6, rtol=1e-6)

    def test_body_gyro_conversion_with_nonzero_roll_pitch(self):
        t, xy, it, _, _ = self.data(5.)
        roll, pitch, omega = .2, -.3, .035
        rpy = np.column_stack((it*0+roll, it*0+pitch, omega*it-.05))
        quat = Rotation.from_euler('xyz', rpy).as_quat()
        gyro = np.tile(omega*np.array([-np.sin(pitch), np.sin(roll)*np.cos(pitch),
                                      np.cos(roll)*np.cos(pitch)]), (len(it), 1))
        model = KinematicProblem(t, xy, it, quat, gyro)
        np.testing.assert_allclose(model.gyro_angle/model.dt, omega, atol=1e-11)
        self.assertGreater(np.max(np.abs(gyro[:, 2]-omega)), .001)

    def test_known_turn_and_gyro_bias(self):
        args = list(self.data())
        truth = args[1].copy(); rng = np.random.default_rng(73)
        args[1] = truth+rng.normal(0, .1, truth.shape)
        problem = KinematicProblem(*args); state, meta = problem.solve()
        self.assertTrue(meta['success'], meta)
        self.assertLess(np.sqrt(np.mean(np.sum((state[:, :2]-truth)**2, axis=1))), .12)
        self.assertLess(np.sqrt(np.mean((state[:, 4]-.035*args[0])**2)), np.deg2rad(.8))
        self.assertLess(abs(np.mean(state[:, 5])-.004), .001)

    def test_withheld_measurements_do_not_affect_initialization_or_fit(self):
        args = list(self.data(8.)); training = np.ones(len(args[0]), bool); training[12:18] = False
        before = KinematicProblem(*args, training=training)
        args[1] = args[1].copy(); args[1][~training] += [30, -50]
        after = KinematicProblem(*args, training=training)
        np.testing.assert_array_equal(before.initial, after.initial)
        np.testing.assert_array_equal(before.evaluate(before.initial.ravel()), after.evaluate(before.initial.ravel()))
        a, _ = before.solve(); b, _ = after.solve()
        np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()

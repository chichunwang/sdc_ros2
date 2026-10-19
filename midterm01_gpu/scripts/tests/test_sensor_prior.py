import sys
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_sensor_prior import interpolate_positions, make_priors


class SensorPriorTests(unittest.TestCase):
    def test_short_edge_extrapolation_and_stale_rejection(self):
        times = np.array([0., .25, .5])
        xyz = np.column_stack((10 * times, times * 0, times * 0))
        result = interpolate_positions(times, xyz, np.array([-.1, .125, .6]))
        np.testing.assert_allclose(result[:, 0], [-1, 1.25, 6])
        with self.assertRaises(ValueError):
            interpolate_positions(times, xyz, np.array([0., 1.]))

    def test_nanosecond_ordering_and_yaw_wrap(self):
        origin = 1727234445000000000
        gnss = np.array([origin + i * 250000000 for i in range(5)], dtype=np.int64)
        imu = np.array([origin + i * 10000000 for i in range(101)], dtype=np.int64)
        # Interpolation crosses +/-180 degrees, never through a zero heading.
        quat = Rotation.from_euler('z', np.linspace(179, 181, 101)[:, None], degrees=True).as_quat()
        xyz = np.zeros((5, 3))
        queries = np.array([origin + 499999999, origin + 500000000, origin + 500000001], dtype=np.int64)
        p, q, meta = make_priors(gnss, xyz, imu, quat, queries)
        yaw = Rotation.from_quat(q).as_euler('xyz', degrees=True)[:, 2]
        self.assertTrue(np.all(np.abs(yaw) > 179.9))
        self.assertEqual(meta['heading_bias_deg'], 0)
        self.assertTrue(np.all(np.isfinite(p)))
        # A sensor outage must fail, rather than producing a complete-looking CSV.
        with self.assertRaises(ValueError):
            make_priors(gnss, xyz, imu[[0, -1]], quat[[0, -1]], queries)


if __name__ == '__main__':
    unittest.main()

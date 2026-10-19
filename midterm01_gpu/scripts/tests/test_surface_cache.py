"""A failed map build must never leave a certified cache for later localization."""
import csv
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from accuracy.surface_cache import binary_xyz_points, height_band_bounds, prepare_height_map


class SurfaceCacheTests(unittest.TestCase):
    def test_band_covers_all_queries_and_pca_margin(self):
        with tempfile.TemporaryDirectory() as directory:
            prior = Path(directory)/'sensor_prior.csv'
            with prior.open('w') as stream:
                writer = csv.writer(stream)
                writer.writerow(['x', 'y', 'z'])
                writer.writerows([[-100, 200, 10], [300, -400, 20]])
            bounds = height_band_bounds(prior)
            for x, y, z in [(-100, 200, 10), (300, -400, 20)]:
                self.assertLessEqual(bounds['min'][0], x-25)
                self.assertGreaterEqual(bounds['max'][0], x+25)
                self.assertLessEqual(bounds['min'][1], y-25)
                self.assertGreaterEqual(bounds['max'][1], y+25)
                self.assertLessEqual(bounds['min'][2], z-6)
                self.assertGreaterEqual(bounds['max'][2], z-.3)
            prior.write_text('x,y,z\n0,0,nan\n')
            with self.assertRaises(ValueError):
                height_band_bounds(prior)

    def test_truncated_binary_is_not_certified(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)/'map.pcd'
            header = b'FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH 1\nHEIGHT 1\nPOINTS 1\nDATA binary\n'
            cache.write_bytes(header+struct.pack('<fff', 1, 2, 3))
            self.assertEqual(binary_xyz_points(cache), 1)
            cache.write_bytes(header+b'\0'*8)
            with self.assertRaisesRegex(ValueError, 'Truncated'):
                binary_xyz_points(cache)

    def test_failed_tool_leaves_no_usable_cache_or_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)/'new_run'
            run.mkdir()
            (run/'sensor_prior.csv').write_text('x,y,z\n0,0,10\n')
            def failure(*args, **kwargs):
                (run/'cache_height_track1_leaf03.partial.pcd').write_bytes(b'incomplete')
                return subprocess.CompletedProcess(args[0], 1)
            with patch('accuracy.surface_cache.subprocess.run', side_effect=failure):
                with self.assertRaisesRegex(RuntimeError, 'exited 1'):
                    prepare_height_map(run, Path(directory)/'data', 1, 'test-container')
            self.assertTrue((run/'cache_height_track1_leaf03.partial.pcd').exists())
            self.assertFalse((run/'cache_height_track1_leaf03.pcd').exists())
            self.assertFalse((run/'cache_height_track1_leaf03.pcd.json').exists())


if __name__ == '__main__':
    unittest.main()

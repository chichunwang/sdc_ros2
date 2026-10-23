"""CPU-only contract tests for Track3 runner inputs and saved outputs."""
import contextlib
import csv
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'run_track3_causal.py'
spec = importlib.util.spec_from_file_location('run_track3_causal_io', SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
from causal_test_support import fixture_run
TEST_RUN = fixture_run()
EXPECTED_HEADER = ['idx', 'original_idx', 'timestamp_ns', 'backend', 'load_ms',
                   'registration_ms', 'total_ms', 'map_accepted', 'scale']


class CausalIOTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='causal_io_', dir=TEST_RUN)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_argument_guards(self):
        cases = [['--full', '--frames', '2'], ['--frames', '0'], ['--track', '1'],
                 ['--start-frame', '1']]
        for name in ('--initial-scale', '--scale-min', '--scale-max'):
            cases += [[name, value] for value in ('0', '-1', 'nan', 'inf')]
        for argv in cases:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    runner.arguments(argv)
        self.assertEqual(runner.arguments(['--frames', '2']).frames, 2)

    def select(self, names, *, full=False, count=2):
        raw = self.root / 'Track3/raw_ply'
        raw.mkdir(parents=True, exist_ok=True)
        for path in raw.iterdir():
            path.unlink()
        for name in names:
            (raw / name).write_text('synthetic filename fixture only')
        argv = ['--data-root', str(self.root)] + (['--full'] if full else ['--frames', str(count)])
        return runner.select_frames(runner.arguments(argv))

    def test_frame_selection(self):
        selected = self.select(['1_ts_200.ply', '0_ts_100.ply', '2_ts_300.ply'])
        self.assertEqual([(row[0], row[1]) for row in selected], [(0, 100), (1, 200)])
        for names in (['1_ts_100.ply', '2_ts_200.ply'],
                      ['0_ts_100.ply', '2_ts_200.ply'],
                      ['0_ts_100.ply', '1_ts_100.ply'],
                      ['0_ts_200.ply', '1_ts_100.ply'],
                      ['0_ts_0.ply', '1_ts_200.ply'],
                      ['0_ts_100.ply', '1_ts_bad.ply'],
                      ['0_ts_100.ply', '1_ts_200.ply', '2_ts_bad.ply']):
            with self.subTest(names=names), self.assertRaises(ValueError):
                self.select(names)
        with self.assertRaises(ValueError):
            self.select(['0_ts_100.ply'], count=2)
        with self.assertRaises(ValueError):
            self.select(['0_ts_100.ply', '1_ts_200.ply'], full=True)

    def test_full_frame_count(self):
        # Empty synthetic name fixtures only: no PLY parsing or localization.
        selected = self.select([f'{i}_ts_{100+i}.ply' for i in range(3142)], full=True)
        self.assertEqual(len(selected), 3142)

    def write_csv(self, name, header, rows):
        with (self.root / name).open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)

    def saved(self):
        poses = np.repeat(np.eye(4)[None, :3, :], 2, axis=0)
        poses[:, 0, 3] = [1., 2.]
        poses[:, 1, 3] = [3., 4.]
        np.savetxt(self.root / 'result.txt', poses.reshape(2, 12))
        self.write_csv('diagnostic.csv', ['idx', 'x', 'y', 'yaw'], [[0, 1, 3, 0], [1, 2, 4, 0]])
        self.write_csv('frames.csv', ['idx', 'timestamp_ns', 'filename'],
                       [[0, 100, '0_ts_100.ply'], [1, 200, '1_ts_200.ply']])
        self.write_csv('timings.csv', EXPECTED_HEADER,
                       [[0, 0, 100, 'test', 1, 2, 3, True, 3],
                        [1, 1, 200, 'test', 1, 2, 3, False, 3]])
        return poses

    def rejected(self):
        try:
            errors = runner.validate_saved(self.root, 2, np)
        except ValueError:
            return
        self.assertTrue(errors, 'corrupted output was accepted')

    def test_fixed_header_and_valid_outputs(self):
        self.assertEqual(runner.TIMING_HEADER, EXPECTED_HEADER)
        self.saved()
        self.assertEqual(runner.validate_saved(self.root, 2, np), [])

    def test_saved_corruption(self):
        for corruption in ('csv', 'matrix_count', 'rotation', 'nan', 'idx', 'header', 'timestamp'):
            with self.subTest(corruption=corruption):
                poses = self.saved()
                if corruption == 'csv':
                    self.write_csv('diagnostic.csv', ['idx', 'x', 'y', 'yaw'], [[0, 9, 3, 0], [1, 2, 4, 0]])
                elif corruption == 'matrix_count':
                    np.savetxt(self.root / 'result.txt', poses[:1].reshape(1, 12))
                elif corruption in ('rotation', 'nan'):
                    poses[0, 2, 2] = -1 if corruption == 'rotation' else np.nan
                    np.savetxt(self.root / 'result.txt', poses.reshape(2, 12))
                elif corruption == 'idx':
                    self.write_csv('diagnostic.csv', ['idx', 'x', 'y', 'yaw'], [[0, 1, 3, 0], [3, 2, 4, 0]])
                elif corruption == 'header':
                    path = self.root / 'timings.csv'
                    path.write_text(path.read_text().replace('load_ms', 'load_seconds'))
                elif corruption == 'timestamp':
                    path = self.root / 'timings.csv'
                    path.write_text(path.read_text().replace(',200,', ',201,'))
                self.rejected()

    def test_jointly_corrupt_timestamps_and_original_idx(self):
        for timestamps in (['bad', 'worse'], [200, 100], [100, 100], [0, 100]):
            with self.subTest(timestamps=timestamps):
                self.saved()
                self.write_csv('frames.csv', ['idx', 'timestamp_ns', 'filename'],
                               [[i, stamp, f'{i}_ts_{stamp}.ply'] for i, stamp in enumerate(timestamps)])
                self.write_csv('timings.csv', EXPECTED_HEADER,
                               [[i, i, stamp, 'test', 1, 2, 3, True, 3] for i, stamp in enumerate(timestamps)])
                self.rejected()
        self.saved()
        path = self.root / 'timings.csv'
        path.write_text(path.read_text().replace('1,1,200,', '1,9,200,'))
        self.rejected()

    def test_empty_csv_rejected(self):
        self.saved()
        self.write_csv('diagnostic.csv', ['idx', 'x', 'y', 'yaw'], [])
        self.rejected()


if __name__ == '__main__':
    unittest.main()

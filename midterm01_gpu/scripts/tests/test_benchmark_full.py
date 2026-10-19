"""Check full-frame selection with a fake Track, stopping before any processing."""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from accuracy import benchmark


class SelectionCaptured(Exception):
    pass


class FullSelectionTests(unittest.TestCase):
    def test_full_selects_all_valid_files_in_timestamp_order(self):
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)/'data'
            raw = data_root/'Track1/raw_ply'
            raw.mkdir(parents=True)
            for name in ('front_ts_300.ply', 'front_ts_100.ply', 'front_ts_200.ply',
                         'front_ts_0.ply', 'not_timestamped.ply'):
                (raw/name).write_bytes(b'not loaded by this selection test')
            compose = {'services': {'midterm01_gpu': {'container_name': 'not-started',
                       'volumes': [{'target': '/root/data', 'source': str(data_root)}]}}}
            selected = []
            def capture(path, value):
                if path.name == 'source_input.json':
                    selected.extend(value['selected_frames'])
                    raise SelectionCaptured()
            # Stop at the manifest: no priors, maps, localization or real output.
            with patch.object(sys, 'argv', ['benchmark.py', '--track', '1', '--full']), \
                 patch.object(benchmark.subprocess, 'check_output', return_value=json.dumps(compose)), \
                 patch.object(benchmark.subprocess, 'run', return_value=subprocess.CompletedProcess([], 1)), \
                 patch.object(benchmark, 'create_run_directory', return_value=Path(directory)/'unused'), \
                 patch.object(benchmark, 'write_json', side_effect=capture), redirect_stdout(io.StringIO()):
                with self.assertRaises(SelectionCaptured):
                    benchmark.main()
            self.assertEqual([row['timestamp_ns'] for row in selected], [100, 200, 300])
            self.assertEqual([row['original_idx'] for row in selected], [0, 1, 2])
            self.assertFalse((Path(directory)/'unused').exists())

    def test_horizontal_mode_rejects_incompatible_baseline_before_processing(self):
        with patch.object(sys,'argv',['benchmark.py','--horizontal-refinement','surfel_joint']), \
             redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error: benchmark.main()
        self.assertEqual(error.exception.code,2)

    def test_horizontal_cache_requires_explicit_refinement(self):
        with patch.object(sys,'argv',['benchmark.py','--horizontal-map-cache','not-read.pcd']), \
             redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error: benchmark.main()
        self.assertEqual(error.exception.code,2)


if __name__ == '__main__':
    unittest.main()

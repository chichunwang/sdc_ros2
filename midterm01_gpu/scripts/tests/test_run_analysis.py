"""分析階段不得改定位證據；完整性、奈秒時間與失敗隔離的回歸檢查。"""
import csv
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from accuracy.benchmark import create_run_directory
from plot_run_analysis import digest, load_run
from run_analysis import generate_after_full


class RunAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[2]
        cls.evidence = create_run_directory(root/'outputs/run', 'run_analysis_guard_tests', track=0)
        print(f'Analysis test evidence: {cls.evidence}')

    def fixture(self):
        """合成保存格式，非定位測試/提交，不放進歷史 full 目錄。"""
        run = self.evidence/self._testMethodName
        run.mkdir()
        n = 3142
        origin = 1727239790258245070
        summary = {'track': 3, 'frames': n, 'returncode': 0, 'output_complete_and_finite': True,
                   'submission_frame_set_complete': True, 'validation_errors': [], 'synthetic_fixture': True}
        (run/'validation.json').write_text(json.dumps(summary)+'\n')
        (run/'result.txt').write_text(('1 0 0 0 0 1 0 0 0 0 1 0\n')*n)
        for filename, fields, rows in [
                ('diagnostic.csv', ['idx', 'x', 'y', 'yaw'], ((i, 0, 0, 0) for i in range(n))),
                ('frames.csv', ['idx', 'timestamp_ns', 'filename'],
                 ((i, origin+i, f'{i:06d}_ts_{origin+i}.ply') for i in range(n))),
                ('timings.csv', ['idx', 'timestamp_ns', 'load_ms', 'registration_ms', 'total_ms', 'map_accepted', 'scale'],
                 ((i, origin+i, 1, 2, 3, 1, 3) for i in range(n)))]:
            with (run/filename).open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(fields)
                writer.writerows(rows)
        return run, summary

    def test_short_incomplete_and_failed_runs_do_not_start_analysis(self):
        base = {'track': 3, 'frames': 3142, 'returncode': 0, 'output_complete_and_finite': True,
                'submission_frame_set_complete': True, 'validation_errors': []}
        cases = [(base, False), (dict(base, returncode=1), True),
                 (dict(base, output_complete_and_finite=False), True),
                 (dict(base, frames=50), True), (dict(base, validation_errors=['bad frame']), True)]
        run = self.evidence/'must_not_be_created'
        with patch('run_analysis.subprocess.run') as child:
            for summary, full in cases:
                self.assertEqual(generate_after_full(run, summary, full=full)['status'], 'skipped')
            child.assert_not_called()
        self.assertFalse(run.exists())

    def test_plot_failure_keeps_successful_validation_and_pose_bytes(self):
        run, summary = self.fixture()
        before = {p.name: digest(p) for p in run.iterdir()}
        with patch('run_analysis.subprocess.run', return_value=subprocess.CompletedProcess([], 7)), redirect_stdout(io.StringIO()):
            status = generate_after_full(run, summary, full=True)
        self.assertEqual(status['status'], 'failed')
        self.assertEqual(status['returncode'], 7)
        self.assertTrue(status['localization_success'])
        self.assertFalse((run/'analysis').exists())
        self.assertEqual(before, {name: digest(run/name) for name in before})
        self.assertEqual(json.loads((run/'analysis_status.json').read_text())['status'], 'failed')

    def test_one_nanosecond_offsets_survive_large_epoch(self):
        run, _ = self.fixture()
        data = load_run(run)
        self.assertEqual(data['stamps'][0], 1727239790258245070)
        self.assertEqual(data['t'][1], 1e-9)
        self.assertAlmostEqual(data['t'][-1], 3141e-9, places=18)

    def test_timing_timestamp_mismatch_is_rejected_before_plotting(self):
        run, _ = self.fixture()
        path = run/'timings.csv'
        content = path.read_text().replace('1727239790258245071', '1727239790258245072', 1)
        path.write_text(content)
        with self.assertRaisesRegex(ValueError, 'Timing timestamps'):
            load_run(run)
        self.assertFalse((run/'analysis').exists())

    def test_csv_matrix_disagreement_is_rejected_before_plotting(self):
        run, _ = self.fixture()
        path = run/'diagnostic.csv'
        path.write_text(path.read_text().replace('0,0,0,0', '0,10,0,0', 1))
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            load_run(run)
        self.assertFalse((run/'analysis').exists())


if __name__ == '__main__':
    unittest.main()

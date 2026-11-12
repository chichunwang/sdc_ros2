"""create_run 與 check_submission 的單元測試（合成資料，不需官方資料）。"""
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import check_submission  # noqa: E402
import create_run  # noqa: E402

TS = ['1741574911912996024', '1741574912012996024', '1741574912112996024']


def write_csv(path, rows):
    path.write_text('\n'.join(','.join(map(str, r)) for r in rows) + '\n', encoding='utf-8')


def pose_row(ts, yaw=0.0, t=(0.0, 0.0, 0.0)):
    c, s = math.cos(yaw), math.sin(yaw)
    return [ts, c, 0, s, t[0], 0, 1, 0, t[1], -s, 0, c, t[2], 0, 0, 0, 1]


class CheckSubmissionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.sample = self.dir / 'sample.csv'
        write_csv(self.sample, [check_submission.HEADER] + [pose_row(t) for t in TS])

    def tearDown(self):
        self.tmp.cleanup()

    def run_check(self, rows):
        sub = self.dir / 'sub.csv'
        write_csv(sub, [check_submission.HEADER] + rows)
        return check_submission.check(sub, self.sample)

    def test_valid(self):
        self.assertEqual(self.run_check([pose_row(t, 0.3, (1, 2, 3)) for t in TS]), [])

    def test_missing_and_duplicate(self):
        errors = self.run_check([pose_row(TS[0]), pose_row(TS[0])])
        self.assertTrue(any('重複' in e for e in errors))
        self.assertTrue(any('缺少' in e for e in errors))

    def test_invalid_rotation(self):
        bad = pose_row(TS[1])
        bad[1] = 2.0
        self.assertTrue(any('正交' in e for e in self.run_check([pose_row(TS[0]), bad, pose_row(TS[2])])))

    def test_reflection_rejected(self):
        bad = pose_row(TS[1])
        bad[1] = -1
        self.assertTrue(any('det' in e for e in self.run_check([pose_row(TS[0]), bad, pose_row(TS[2])])))

    def test_non_numeric(self):
        bad = pose_row(TS[2])
        bad[4] = 'nan'
        self.assertTrue(self.run_check([pose_row(TS[0]), pose_row(TS[1]), bad]))


class CreateRunTest(unittest.TestCase):
    def test_unique_names_and_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'midterm02'
            (root / 'vslam').mkdir(parents=True)
            (root / 'vslam' / 'a.py').write_text('x = 1\n')
            now = datetime(2026, 11, 13, 9, 5, tzinfo=create_run.TZ)
            first = create_run.create_run('test', '01', root=root, now=now)
            second = create_run.create_run('test', '01', root=root, now=now)
            self.assertEqual(first.name, 'track01_test_2611130905')
            self.assertEqual(second.name, 'track01_test_2611130905_0002')
            self.assertEqual((first / 'source/vslam/a.py').read_text(), 'x = 1\n')
            manifest = json.loads((first / 'source_manifest.json').read_text(encoding='utf-8'))
            self.assertIn('vslam/a.py', manifest['sources'])
            info = json.loads((first / 'test_info.json').read_text(encoding='utf-8'))
            self.assertEqual(info['status'], 'created')


if __name__ == '__main__':
    unittest.main()

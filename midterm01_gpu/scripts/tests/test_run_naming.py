"""Track/scope/time names; algorithm/version belongs inside the run."""
import datetime
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from output_layout import loads_saved_json
from accuracy import benchmark
import replay


class FixedDatetime(datetime.datetime):
    @classmethod
    def now(cls,tz=None):
        return cls(2026,10,19,16,5,tzinfo=datetime.timezone.utc).astimezone(tz)


class RunNamingTests(unittest.TestCase):
    def test_taipei_time_first_and_version_inside(self):
        with tempfile.TemporaryDirectory() as directory, patch('accuracy.benchmark.datetime.datetime',FixedDatetime):
            prefix='track1_sensor_guided_motion_rts_geometry_directional_surfel_joint_cuda_robust_icp_d2d'
            run=benchmark.create_run_directory(Path(directory),prefix)
            self.assertEqual(run.name,'track01_test_2610200005')
            details=loads_saved_json((run/'test_info.json').read_text())
            self.assertEqual(details['test_kind'],prefix)
            self.assertEqual(details['track'],1)
            self.assertTrue(details['created_at'].endswith('+08:00'))

    def test_same_minute_versions_do_not_change_name_or_overwrite(self):
        with tempfile.TemporaryDirectory() as directory, patch('accuracy.benchmark.datetime.datetime',FixedDatetime):
            old=benchmark.create_run_directory(Path(directory),'track2_algorithm_a')
            (old/'result.txt').write_text('preserve')
            new=benchmark.create_run_directory(Path(directory),'track2_algorithm_b')
            self.assertEqual(new.name,'track02_test_2610200005_0002')
            self.assertEqual((old/'result.txt').read_text(),'preserve')
            self.assertEqual(loads_saved_json((new/'test_info.json').read_text())['test_kind'],'track2_algorithm_b')

    def test_full_and_short_runs_have_separate_explicit_scopes(self):
        with tempfile.TemporaryDirectory() as directory, patch('accuracy.benchmark.datetime.datetime',FixedDatetime):
            full=benchmark.create_run_directory(Path(directory),'track1_method',track=1,full=True)
            short=benchmark.create_run_directory(Path(directory),'track1_method',track=1,full=False)
            self.assertEqual(full.name,'track01_full_2610200005')
            self.assertEqual(short.name,'track01_test_2610200005')
            self.assertEqual(loads_saved_json((full/'test_info.json').read_text())['scope'],'full')

    def test_research_and_replay_follow_same_timestamp_policy(self):
        with tempfile.TemporaryDirectory() as directory, patch('accuracy.benchmark.datetime.datetime',FixedDatetime):
            root=Path(directory)
            research=benchmark.create_run_directory(root/'outputs/run','research_experiment_v7')
            with patch.object(replay,'ROOT',root): replay_run=replay.new_output()
            self.assertEqual(research.name,'track00_test_2610200005')
            self.assertEqual(replay_run.name,'track00_test_2610200005_0002')
            self.assertEqual(loads_saved_json((replay_run/'test_info.json').read_text())['test_kind'],'replay')


if __name__=='__main__':unittest.main()

"""Relocation changes path lookup, never original evidence bytes or CPU paths."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from output_layout import _aliases, relocate_text, relocate_tree, latest_full_run


class OutputLayoutTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.outputs=Path(self.temp.name)/'outputs'
        self.run=self.outputs/'run/track01_full_2610192232';self.run.mkdir(parents=True)
        self.original_root='/home/ricky/sdc_ros2/midterm01_gpu/outputs'
        info={'legacy_locations':{'runs/old_run':'.','current/track1':'.'},'original_outputs_root':self.original_root}
        (self.run/'test_info.json').write_text(json.dumps(info))
        _aliases.cache_clear();self.addCleanup(_aliases.cache_clear)

    def test_host_container_relative_and_shortcut_aliases(self):
        for old,new in ((self.original_root+'/runs/old_run/result.txt',self.original_root+'/run/track01_full_2610192232/result.txt'),
                        ('/root/outputs/runs/old_run/result.txt','/root/outputs/run/track01_full_2610192232/result.txt'),
                        ('outputs/current/track1/result.txt','outputs/run/track01_full_2610192232/result.txt'),
                        ('../outputs/runs/old_run/result.txt','../outputs/run/track01_full_2610192232/result.txt')):
            self.assertEqual(relocate_text(old,self.outputs),new)

    def test_path_prefix_boundaries_and_cpu_files_are_preserved(self):
        for path in ('/root/data/Track1/map.pcd','outputs/runs/old_run_extra/result.txt',
                     '/home/ricky/sdc_ros2/midterm01/outputs/runs/old_run/result.txt'):
            self.assertEqual(relocate_text(path,self.outputs),path)

    def test_nested_metadata_is_translated_only_in_memory(self):
        source=self.run/'params.json';source.write_text('{"path":"/root/outputs/runs/old_run/result.txt","sha256":"abcdef"}')
        original=source.read_bytes()
        data=relocate_tree(json.loads(source.read_text()),self.outputs)
        self.assertEqual(data['path'],'/root/outputs/run/track01_full_2610192232/result.txt')
        self.assertEqual(data['sha256'],'abcdef')
        self.assertEqual(source.read_bytes(),original)

    def test_default_replay_ignores_failed_and_short_attempts(self):
        for run in (self.run,self.outputs/'run/track01_full_2610192240',self.outputs/'run/track01_test_2610192250'):
            run.mkdir(exist_ok=True);(run/'result.txt').write_text('fixture')
            valid={'output_complete_and_finite':run==self.run,'returncode':0,'validation_errors':[], 'frames':3011}
            (run/'validation.json').write_text(json.dumps(valid))
        self.assertEqual(latest_full_run(self.outputs.parent,1),self.run)

    def test_default_replay_waits_for_final_horizontal_validation(self):
        (self.run/'result.txt').write_text('fixture')
        validation={'output_complete_and_finite':True,'returncode':0,'validation_errors':[],'frames':3011}
        (self.run/'validation.json').write_text(json.dumps(validation))
        (self.run/'horizontal_settings.json').write_text('{"mode":"surfel_joint"}')
        with self.assertRaises(ValueError):latest_full_run(self.outputs.parent,1)
        validation['horizontal_refinement']='surfel_joint'
        (self.run/'validation.json').write_text(json.dumps(validation))
        self.assertEqual(latest_full_run(self.outputs.parent,1),self.run)


if __name__=='__main__':unittest.main()

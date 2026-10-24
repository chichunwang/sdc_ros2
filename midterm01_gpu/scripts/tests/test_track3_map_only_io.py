"""Fixed output/causal timestamp checks for the no-IMU runner."""
import csv
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from causal_test_support import fixture_run
from run_track3_map_only import TIMING_HEADER, arguments, validate_output, configure_vram_budget, record_failure
from track3_map_only import MapOnlyTracker


class MapOnlyIO(unittest.TestCase):
    def fixture(self):
        root=fixture_run()
        import uuid
        run=root/('map_only_io_'+uuid.uuid4().hex)
        run.mkdir()
        np.savetxt(run/'result.txt',np.eye(4)[:3].reshape(1,12))
        for name,header,rows in [
            ('diagnostic.csv',['idx','x','y','yaw'],[[0,0,0,0]]),
            ('frames.csv',['idx','timestamp_ns','filename'],[[0,1000000001,'x.ply']]),
            ('timings.csv',TIMING_HEADER,[[0,0,1000000001,'cuda',1,2,3,True,3.]])]:
            with (run/name).open('w',newline='') as stream:
                w=csv.writer(stream);w.writerow(header);w.writerows(rows)
        obs=dict(idx=0,timestamp_ns=1000000001,input_cutoff_ns=1000000001,imu_used=False)
        (run/'observations.jsonl').write_text(json.dumps(obs)+'\n')
        return run,obs

    def test_valid_output_needs_no_fake_imu(self):
        run,_=self.fixture()
        self.assertEqual(validate_output(run,1),[])

    def test_future_input_cutoff_is_rejected(self):
        run,obs=self.fixture();obs['input_cutoff_ns']+=1
        (run/'observations.jsonl').write_text(json.dumps(obs)+'\n')
        self.assertTrue(validate_output(run,1))

    def test_csv_matrix_disagreement_is_rejected(self):
        run,_=self.fixture()
        (run/'diagnostic.csv').write_text('idx,x,y,yaw\n0,1,0,0\n')
        self.assertTrue(validate_output(run,1))

    def test_invalid_frames_and_full_conflict(self):
        for args in (['--frames','0'],['--frames','2','--full']):
            with self.assertRaises(SystemExit):arguments(args)

    def test_vram_budget_parser(self):
        self.assertEqual(arguments([]).vram_budget_gib,4.)
        self.assertEqual(arguments(['--vram-budget-gib','2.5']).vram_budget_gib,2.5)
        for value in ('0','-1','nan','inf'):
            with self.assertRaises(SystemExit):arguments(['--vram-budget-gib',value])

    def test_allocator_budget_exceeding_device_is_rejected(self):
        cuda=SimpleNamespace(mem_get_info=Mock(return_value=(1024**3,2*1024**3)),
            set_per_process_memory_fraction=Mock())
        with self.assertRaisesRegex(ValueError,'exceeds device total'):
            configure_vram_budget(SimpleNamespace(cuda=cuda),4.)
        cuda.set_per_process_memory_fraction.assert_not_called()

    def test_interrupt_preserves_progress_even_if_memory_query_fails(self):
        unavailable=Mock(side_effect=RuntimeError('CUDA context unavailable'))
        cuda=SimpleNamespace(**{key:unavailable for key in ('mem_get_info','memory_allocated',
            'memory_reserved','max_memory_allocated','max_memory_reserved')})
        summary=dict(validation_errors=[],processed_frames=7,failed_original_idx=7)
        record_failure(summary,KeyboardInterrupt(),SimpleNamespace(cuda=cuda))
        self.assertEqual(summary['failure']['type'],'KeyboardInterrupt')
        self.assertEqual(summary['failure']['processed_frames'],7)
        self.assertEqual(summary['failure']['failed_original_idx'],7)
        self.assertIn('free_total_bytes_error',summary['cuda_memory'])

    def test_tracking_rejects_revisiting_past_before_geometry(self):
        tracker=MapOnlyTracker(None,np.eye(4));tracker.previous_ns=100
        with self.assertRaises(ValueError):tracker.step(np.zeros((8,3)),100)

    def test_ray_winner_cannot_be_certified_by_other_restarts(self):
        def candidate(cost,ray,offset):
            pose=np.eye(4);pose[0,3]=offset
            return dict(pose=pose.tolist(),scale=3.,accepted=True,reasons=[],training_cost=cost,
                checks=dict(converged=True,interior=True,observable=True),
                training=dict(coverage=.99,median_m=.1,p90_m=.2),
                ray_depth=dict(available=True,training=dict(support_fraction=.99,median_abs_log_depth_ratio=ray),
                    validation=dict(support_fraction=.99,median_abs_log_depth_ratio=ray)))
        tracker=MapOnlyTracker(None,np.eye(4))
        _,decision=tracker._choose_and_check(None,[candidate(.2,.08,0),candidate(.21,.09,.01),candidate(.8,.05,1)])
        self.assertFalse(decision['accepted'])
        self.assertIn('restart_inconsistent',decision['reasons'])


if __name__=='__main__':unittest.main()

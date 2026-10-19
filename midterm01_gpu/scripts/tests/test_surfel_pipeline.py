"""Exercise accepted and rejected map factors through the actual publication path.

Synthetic geometry/sensor data only. ROS export and file decoders are replaced,
not the surfel fitting, graph, holdout checks or final pose serialization.
"""
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prior_ablation import dump
from accuracy.benchmark import print_run_outcome
from refine_horizontal_joint import refine
from three_camera_source import sha256


class SurfelPipelineTests(unittest.TestCase):
    def scenario(self, walls):
        root=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=root/'outputs/run') as directory:
            folder=Path(directory); run=folder/'run'; run.mkdir(); data=folder/'data'
            source=data/'Track1/raw_ply'; source.mkdir(parents=True)
            ns=1_750_000_000_000_000_000; stamps=ns+np.arange(48)*35_000_000
            frames=[]; manifest=[]
            for i,stamp in enumerate(stamps):
                name=f'front_ts_{stamp}.ply'; (source/name).write_bytes(b'synthetic decoder fixture')
                frames.append({'idx':i,'timestamp_ns':int(stamp),'filename':name})
                manifest.append({**frames[-1],'original_idx':i,'sha256':sha256(source/name)})
            with (run/'frames.csv').open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=['idx','timestamp_ns','filename']); writer.writeheader(); writer.writerows(frames)
            poses=np.tile(np.column_stack((np.eye(3),[.2,-.15,0.])),(48,1,1))
            np.savetxt(run/'result.txt',poses.reshape(-1,12),fmt='%.9f')
            (run/'diagnostic.csv').write_text('idx,x,y,yaw\n'+''.join(f'{i},0.2,-0.15,0.0\n' for i in range(48)))
            before={f:(run/f).read_bytes() for f in ('result.txt','diagnostic.csv')}
            dump(run/'source_input.json',{'selected_frames':manifest})
            dump(run/'validation.json',{'output_complete_and_finite':True,'sensor_fusion':'motion_rts','cloud_source':'front','track':1})
            dump(run/'params.json',{'localization_profile':'sensor_guided','geometry_mode':'directional',
                 'sensor_height_mode':'carry','ply_scale':3.})
            a,b=np.meshgrid(np.arange(-8.7,8.71,.3),np.arange(0.,4.51,.3))
            if walls:
                cloud=np.concatenate([np.column_stack((a.ravel()*0+side,a.ravel(),b.ravel())) for side in (-10.,10.)]+
                                     [np.column_stack((a.ravel(),a.ravel()*0+side,b.ravel())) for side in (-10.,10.)])
            else:
                cloud=np.column_stack((a.ravel(),b.ravel(),a.ravel()*0))
            it=np.arange(-.5,2.501,.01); gt=np.arange(-.25,2.251,.25)
            raw={'gnss_bag_ns':ns+np.rint(gt*1e9).astype(np.int64),
                'gnss_xyz':np.tile([.2,-.15,0.],(len(gt),1)),
                'imu_bag_ns':ns+np.rint(it*1e9).astype(np.int64),
                'imu_quat':Rotation.from_euler('z',(it*0)[:,None]).as_quat(),
                'angular_velocity':np.zeros((len(it),3))}
            def exporter(*args,**kwargs):
                np.savez_compressed(run/'horizontal/raw_sensors.npz',**raw)
                return type('Result',(),{'returncode':0})()
            with patch('refine_horizontal_joint.subprocess.run',side_effect=exporter), \
                 patch('refine_horizontal_joint.audit',return_value={'synthetic':True}), \
                 patch('refine_horizontal_joint.load_ply',return_value=cloud/3), \
                 patch('refine_horizontal_joint.load_binary_map',return_value=cloud), \
                 patch('refine_horizontal_joint.verify_fine_cache',return_value={'synthetic':True}), redirect_stdout(io.StringIO()):
                report=refine(run,data,'synthetic-only',folder/'fake.pcd')
            after=np.loadtxt(run/'result.txt').reshape(-1,3,4)
            self.assertTrue(report['output_complete_and_finite'])
            final_csv=list(csv.DictReader(io.StringIO((run/'diagnostic.csv').read_text())))
            self.assertEqual([int(r['idx']) for r in final_csv],list(range(48)))
            for row,pose in zip(final_csv,after):
                self.assertAlmostEqual(float(row['x']),pose[0,3],places=12)
                self.assertAlmostEqual(float(row['y']),pose[1,3],places=12)
                self.assertAlmostEqual(float(row['yaw']),np.arctan2(pose[1,0],pose[0,0]),places=12)
            self.assertEqual(report['result_sha256'],sha256(run/'result.txt'))
            self.assertEqual(report['csv_sha256'],sha256(run/'diagnostic.csv'))
            output=io.StringIO()
            with redirect_stdout(output):
                print_run_outcome(run,{'output_complete_and_finite':True,
                    'horizontal_diagnostics_not_accuracy':report},full_attempt=True)
            np.testing.assert_array_equal(after[:,2,:],poses[:,2,:])
            if walls:
                self.assertEqual(report['accepted_map_factors'],1,report)
                self.assertTrue(report['applied'],report)
                self.assertTrue(report['joint_estimation_executed'])
                self.assertTrue(report['csv_changed_from_base'])
                self.assertTrue(report['result_changed_from_base'])
                self.assertIn('本輪 CSV 已改變',output.getvalue())
                self.assertNotIn('無須重複提交',output.getvalue())
                self.assertLess(np.linalg.norm(after[24,:2,3]),np.linalg.norm(poses[24,:2,3]))
                self.assertGreater(report['final_checks']['max_xy_change_m'],.01)
            else:
                self.assertEqual(report['accepted_map_factors'],0)
                self.assertFalse(report['applied'])
                self.assertFalse(report['joint_estimation_executed'])
                self.assertFalse(report['csv_changed_from_base'])
                self.assertFalse(report['result_changed_from_base'])
                self.assertEqual(report['observation_diagnostics_not_accuracy']['accepted_observations'],0)
                self.assertIn('無須重複提交',output.getvalue())
                self.assertNotIn('提交檔:',output.getvalue())
                for name,value in before.items(): self.assertEqual((run/name).read_bytes(),value)

    def test_accepted_synthetic_walls_publish_consistent_new_pose_and_csv(self): self.scenario(True)
    def test_rejected_ground_preserves_original_bytes(self): self.scenario(False)


if __name__=='__main__': unittest.main()

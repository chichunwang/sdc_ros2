"""Frozen-core streaming and causal predictor mechanisms on invented inputs."""
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sqlite3
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from causal_track3_filter import ImuCursor
from old_icp_runtime import CarryGyro, initial_ground_tilt, legacy_float_pose, parse_result
from causal_test_support import fixture_run
from test_causal_track3 import imu_arrays, room, NS
import run_track3_old_icp as runner


class RuntimeTests(unittest.TestCase):
    def test_float_roundoff_preserved_without_accepting_invalid_rotation(self):
        pose=np.eye(4);pose[0,0]=1.+2e-6
        np.testing.assert_array_equal(legacy_float_pose(pose),pose)
        pose[0,0]=-1.
        with self.assertRaises(ValueError):legacy_float_pose(pose)

    def test_carry_position_gyro_zoh_excludes_future(self):
        pose = np.eye(4); pose[:3, 3] = [660., -220., 10.]
        arrays = imu_arrays([0., .02, .04, .05], [0., 1., 900., -900.])
        cursor = ImuCursor(arrays)
        f = CarryGyro(pose, NS)
        f.predict_to(NS, cursor)
        out, report = f.predict_to(NS+35_000_000, cursor)
        np.testing.assert_array_equal(out[:3, 3], pose[:3, 3])
        self.assertAlmostEqual(Rotation.from_matrix(out[:3, :3]).as_rotvec()[2], .015)
        self.assertEqual(report['max_imu_availability_ns'], NS+20_000_000)
        self.assertEqual(cursor.index, 2)

    def test_future_suffix_does_not_change_prefix_or_old_copies(self):
        def run(n, future):
            data = imu_arrays(np.arange(0, 1.01, .01), np.r_[np.full(51, .1), np.full(50, future)])
            f = CarryGyro(np.eye(4), NS); cursor = ImuCursor(data); poses=[]
            for i in range(n):
                pose, _ = f.predict_to(NS+i*40_000_000, cursor)
                poses.append(pose)
                f.pose[0, 3] += .05
            return np.array(poses)
        np.testing.assert_array_equal(run(10, 900.), run(20, -900.)[:10])

    def test_stale_and_time_reversal_rejected(self):
        f = CarryGyro(np.eye(4), NS); cursor = ImuCursor(imu_arrays([0.]))
        f.predict_to(NS, cursor)
        with self.assertRaisesRegex(ValueError, 'fresh'):
            f.predict_to(NS+100_000_000, cursor)
        with self.assertRaisesRegex(ValueError, 'revisit'):
            f.predict_to(NS-1, cursor)

    def test_ground_initialization_changes_only_rotation(self):
        a,b = np.meshgrid(np.linspace(1,12,30), np.linspace(-4,4,20))
        raw = np.column_stack([a.ravel(), b.ravel(), np.full(a.size, -1.5)])/3.
        prior = np.eye(4); prior[:3, 3] = [660., -220., 10.]
        prior[:3, :3] = Rotation.from_euler('xyz', [14.,14.,145.], degrees=True).as_matrix()
        pose, report = initial_ground_tilt(raw, prior, 3.)
        np.testing.assert_array_equal(pose[:3, 3], prior[:3, 3])
        np.testing.assert_allclose(pose[:3, :3] @ [0,0,1], [0,0,1], atol=1e-10)
        self.assertGreater(report['tilt_correction_deg'], 15.)
        with self.assertRaises(ValueError):
            initial_ground_tilt(raw[:20], prior, 3.)

    def test_cli_guards_and_fixed_scale_plan(self):
        for argv in (['--frames','0'], ['--ply-scale','nan'], ['--ply-scale','0'],
                     ['--full','--frames','2'], ['--initial-tilt','wrong'], ['--track','1']):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.arguments(argv)
        text = io.StringIO()
        with contextlib.redirect_stdout(text):
            self.assertEqual(runner.main(['--full','--dry-run']), 0)
        plan = json.loads(text.getvalue())
        self.assertEqual(plan['frames'],3142)
        self.assertFalse(plan['gnss_used'])
        self.assertFalse(plan['kalman_used'])
        self.assertFalse(plan['new_sim3_optimizer_used'])
        self.assertFalse(plan['core_modified'])
        self.assertEqual(plan['scale_mode'],'fixed')


@unittest.skipUnless(os.environ.get('SDC_OLD_ICP_TEST_BINARY'), 'requires isolated compiled bridge')
class BridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output_run = fixture_run()
        cls.binary = Path(os.environ['SDC_OLD_ICP_TEST_BINARY']).resolve()
        cls.points, _ = room()
        cls.fixture = cls.output_run/'old_core_invented'
        cls.fixture.mkdir(exist_ok=True)
        cls.map = cls.fixture/'map.pcd'
        p = cls.points
        cls.map.write_text('VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n'
            f'WIDTH {len(p)}\nHEIGHT 1\nPOINTS {len(p)}\nDATA ascii\n'+
            '\n'.join(' '.join(map(str,row)) for row in p)+'\n')
        cls.paths=[]
        for i in range(5):
            raw = (p-[i*.04,0,1.5])/3.
            path=cls.fixture/f'frame_{i}.ply'
            path.write_text(f'ply\nformat ascii 1.0\nelement vertex {len(raw)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'+
                '\n'.join(' '.join(map(str,row))+' 0 0 0' for row in raw)+'\n')
            cls.paths.append(path)

    @staticmethod
    def inside(path):
        return '/root/outputs/'+str(path.relative_to(runner.ROOT/'outputs'))

    def bridge(self, n, *, invalid=False):
        command=['docker','exec','-i','sdc_midterm01_gpu',self.inside(self.binary),
                 '--map',self.inside(self.map),'--scale','3','--geometry','directional','--cpu-reference']
        prior=np.eye(4);prior[:3,3]=[.2,-.1,1.6]
        with (self.output_run/f'bridge_cpu_{n}_{invalid}.log').open('w') as log:
            proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True)
            try:
                ready=proc.stdout.readline().split()
                self.assertEqual(ready[0],'READY')
                self.assertEqual(float(ready[3]),0.)
                outputs=[]
                for i in range(n):
                    request='FRAME '+json.dumps(self.inside(self.paths[i]))+' '+' '.join(map(str,prior.ravel()))+'\n'
                    proc.stdin.write('INVALID\n' if invalid else request);proc.stdin.flush()
                    line=proc.stdout.readline()
                    if invalid:
                        self.assertEqual(line,'')
                        break
                    report=parse_result(line)
                    self.assertGreaterEqual(report['total_ms'],report['load_ms'])
                    self.assertLessEqual(report['final_cost'],report['prior_cost']+1e-8)
                    self.assertEqual(report['accepted'],True,report)
                    prior=np.array(report['pose'])
                    outputs.append(report)
                proc.stdin.close()
                code=proc.wait(timeout=15)
                self.assertEqual(code,2 if invalid else 0)
                return outputs
            finally:
                if proc.poll() is None:
                    proc.stdin.close();proc.wait(timeout=15)
                proc.stdout.close()

    def test_bridge_known_room_and_prefix(self):
        a,b=self.bridge(3),self.bridge(5)
        for x,y in zip(a,b):
            for key in ('pose','correction','accepted','final_cost','update_status'):
                self.assertEqual(x[key],y[key])
        self.assertLess(np.linalg.norm(np.array(b[-1]['pose'])[:3,3]-[.16,0.,1.5]),.5)
        (self.output_run/'old_bridge_cpu_prefix.json').write_text(json.dumps({
            'synthetic_only':True,'real_localization_executed':False,'cpu_reference_only':True,
            'prefix_frames':3,'longer_frames':5,'prefix_matrices_metrics_equal':True,
            'last_pose':b[-1]['pose']},indent=2)+'\n')

    def test_invalid_bridge_protocol_exits(self):
        self.bridge(1,invalid=True)

    def test_invalid_result_rejected(self):
        for text in ('','RESULT 1', 'READY 1 0 0'):
            with self.assertRaises(ValueError):parse_result(text)

    def test_cuda_no_overlap_returns_current_prior_and_zero_correction(self):
        far=self.fixture/'no_overlap.ply'
        source=self.paths[0].read_text().split('end_header\n')
        values=np.array([list(map(float,row.split())) for row in source[1].splitlines()])
        values[:,:3]+=100.
        far.write_text(source[0]+'end_header\n'+'\n'.join(' '.join(map(str,row)) for row in values)+'\n')
        prior=np.eye(4);prior[:3,3]=[.2,-.1,1.6]
        request='FRAME '+json.dumps(self.inside(far))+' '+' '.join(map(str,prior.ravel()))+'\n'
        command=['docker','exec','-i','sdc_midterm01_gpu',self.inside(self.binary),
                 '--map',self.inside(self.map),'--scale','3','--geometry','directional']
        with (self.output_run/'no_overlap_cuda.log').open('w') as log:
            result=subprocess.run(command,input=request,text=True,stdout=subprocess.PIPE,stderr=log,check=True)
        lines=result.stdout.splitlines()
        self.assertEqual(len(lines),2)
        report=parse_result(lines[1])
        self.assertFalse(report['accepted'])
        np.testing.assert_allclose(report['pose'],prior,atol=1e-6,rtol=0)
        np.testing.assert_array_equal(report['correction'],[0.,0.,0.,0.])

    def test_whole_runner_on_invented_inputs_cuda_and_prefix(self):
        fixture=self.output_run/'runner_invented'
        track=fixture/'Track3'; raw=track/'raw_ply'; raw.mkdir(parents=True,exist_ok=True)
        (track/'map.pcd').write_bytes(self.map.read_bytes())
        pose=np.eye(4);pose[:3,3]=[.2,-.1,1.6]
        np.savetxt(track/'init_guess.txt',pose)
        stamps=[NS+i*40_000_000 for i in range(5)]
        for i,ns in enumerate(stamps):
            (raw/f'{i:06d}_ts_{ns}.ply').write_bytes(self.paths[i].read_bytes())
        bag=track/'Track3.db3'
        if bag.exists():bag.unlink()
        with sqlite3.connect(bag) as db:
            db.execute('CREATE TABLE topics(id INTEGER,name TEXT,type TEXT)')
            db.execute('CREATE TABLE messages(id INTEGER,topic_id INTEGER,timestamp INTEGER,data BLOB)')
            db.execute('INSERT INTO topics VALUES(1,?,?)',('/lucid_cameras_x00/gige_100_f_hdr/h265','sensor_msgs/msg/Image'))
            db.executemany('INSERT INTO messages VALUES(?,1,?,NULL)',list(enumerate(stamps)))
        def export(args,run,cutoff):
            data=imu_arrays(np.arange(-.01,.21,.01))
            data={key:value[data['availability_ns']<=cutoff] for key,value in data.items()}
            data['accel_m_s2']=np.tile([0.,0.,9.81],(len(data['availability_ns']),1))
            np.savez_compressed(run/'imu.npz',**data)
        outputs=[]
        for count in (3,5):
            console=io.StringIO()
            with patch.object(runner,'active_localizations',return_value=[]), \
                    patch.object(runner,'verify_container'), patch.object(runner,'export_imu',side_effect=export), \
                    patch.object(runner,'build_bridge',return_value=self.binary), \
                    patch.object(runner,'input_inside',side_effect=lambda p:self.inside(fixture/p)), \
                    contextlib.redirect_stdout(console):
                self.assertEqual(runner.main(['--frames',str(count),'--data-root',str(fixture),'--initial-tilt','official']),0)
            output=Path(next(line[5:] for line in console.getvalue().splitlines() if line.startswith('Run: ')))
            outputs.append(output)
            info=json.loads((output/'test_info.json').read_text())
            info.update(synthetic_invented_room=True,real_track_localization=False,fixture_owner=str(self.output_run))
            (output/'test_info.json').write_text(json.dumps(info,indent=2)+'\n')
            validation=json.loads((output/'validation.json').read_text())
            self.assertTrue(validation['output_complete_and_finite'],validation)
            self.assertEqual(validation['processed_frames'],count)
            self.assertEqual(validation['map_accepted_frames'],count)
            self.assertFalse((output/'analysis').exists())
            self.assertIn(f'Finished: frames={count},',console.getvalue())
        self.assertEqual((outputs[0]/'result.txt').read_text(),'\n'.join((outputs[1]/'result.txt').read_text().splitlines()[:3])+'\n')
        self.assertEqual((outputs[0]/'diagnostic.csv').read_text(),'\n'.join((outputs[1]/'diagnostic.csv').read_text().splitlines()[:4])+'\n')
        cpu=self.bridge(3)
        gpu=np.loadtxt(outputs[0]/'result.txt').reshape(-1,3,4)
        for expected,actual in zip(cpu,gpu):
            np.testing.assert_allclose(np.array(expected['pose'])[:3],actual,atol=1e-5,rtol=0)
        (self.output_run/'old_runner_cuda_prefix.json').write_text(json.dumps({
            'synthetic_only':True,'real_track_localization':False,'cuda_bridge_executed':True,
            'prefix_frames':3,'longer_frames':5,'matrices_and_csv_prefix_byte_identical':True,
            'cpu_cuda_pose_atol':1e-5,
            'outputs':list(map(str,outputs))},indent=2)+'\n')


if __name__=='__main__':
    unittest.main()

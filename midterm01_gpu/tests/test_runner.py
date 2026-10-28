"""End-to-end CUDA pipeline using exclusively generated map/PLY data."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from plyfile import PlyData,PlyElement
from test_core import CFG,cloud


@pytest.mark.parametrize('prediction',['previous_pose','gpu_monte_carlo'])
def test_generated_pipeline(tmp_path,monkeypatch,prediction):
    root=tmp_path/'synthetic_data';root.mkdir();(root/'raw_ply').mkdir()
    m=cloud();stamp=1727239790258245070
    with (root/'map.pcd').open('w') as f:
        f.write(f'VERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\nWIDTH {len(m)}\nHEIGHT 1\nPOINTS {len(m)}\nDATA ascii\n')
        np.savetxt(f,np.column_stack([m,np.zeros(len(m))]))
    np.savetxt(root/'init_guess.txt',np.eye(4),delimiter=',')
    for idx in range(3):
        pts=(m-np.array([idx*.05,0,0]))/3
        v=np.empty(len(pts),dtype=[('x','f4'),('y','f4'),('z','f4')])
        for j,k in enumerate(('x','y','z')):v[k]=pts[:,j]
        PlyData([PlyElement.describe(v,'vertex')],text=False).write(str(root/'raw_ply'/f'{idx:06d}_ts_{stamp+idx*100000000}.ply'))
    cfg=json.loads(json.dumps(CFG));cfg['data_dir']=str(root);cfg['expected_frames']=3;cfg['map_leaf_m']=.15
    cfg['prediction']=prediction
    if prediction=='gpu_monte_carlo':
        monkeypatch.setitem(sys.modules,'track3.inertial',None)
    if prediction=='gpu_monte_carlo':cfg['imu']['time_basis']='bag_arrival'
    if prediction=='gpu_monte_carlo':
        import sqlite3
        from sensor_msgs.msg import Imu
        from rclpy.serialization import serialize_message
        db=sqlite3.connect(root/'Track3.db3')
        db.execute('CREATE TABLE topics(id INTEGER,name TEXT,type TEXT,serialization_format TEXT)')
        db.execute('CREATE TABLE messages(id INTEGER,topic_id INTEGER,timestamp INTEGER,data BLOB)')
        db.execute("INSERT INTO topics VALUES(1,'/imu/data','sensor_msgs/msg/Imu','cdr')")
        for i in range(23):
            t=stamp-20_000_000+i*10_000_000
            msg=Imu();msg.header.frame_id='imu_link'
            msg.header.stamp.sec=t//1_000_000_000;msg.header.stamp.nanosec=t%1_000_000_000
            msg.linear_acceleration.z=9.80665
            db.execute('INSERT INTO messages VALUES(?,1,?,?)',(i+1,t,serialize_message(msg)))
        db.commit();db.close()
    cfg['version']='SYNTHETIC_VALIDATION_ONLY'
    cfg['registration']['stages']=[{'mode':'ndt','voxel_m':2.,'iterations':30},{'mode':'icp','voxel_m':1.,'iterations':30}]
    config=tmp_path/'config.json';config.write_text(json.dumps(cfg))
    run=tmp_path/'synthetic_run';run.mkdir();(run/'source_manifest.json').write_text(json.dumps({'test_fixture':True}))
    script=Path(__file__).resolve().parents[1]/'scripts/run_track3.py'
    spec=importlib.util.spec_from_file_location('run_track3',script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(sys,'argv',[str(script),'--config',str(config),'--run-dir',str(run),'--max-frames','3'])
    assert module.main()==0
    info=json.loads((run/'test_info.json').read_text());assert info['status']=='success'
    assert info['frame_count']==3 and info['imu_used']==(prediction=='gpu_monte_carlo') and info['map_accepted_frames']==3
    assert (root/'Track3.db3').exists()==(prediction=='gpu_monte_carlo')
    if prediction=='gpu_monte_carlo':
        sources=json.loads((run/'imu_sources.json').read_text())
        assert sources['sha256']==hashlib.sha256((root/'Track3.db3').read_bytes()).hexdigest()
        assert info['imu_samples_consumed']==23
        effective=json.loads((run/'parameters.json').read_text())['imu']
        from track3.imu_parameters import resolve_imu_config, resolve_motion_config
        assert effective==resolve_imu_config(resolve_motion_config(cfg['imu']))
        assert info['kalman_used'] is effective['gyro_bias_filter']['enabled']
        assert not info['pose_kalman_used'] and sources['kalman_scope']=='stationary_gyro_bias_only_no_pose_fusion'
        assert info['accelerometer_translation_used'] is False
        assert sources['T_ply_imu']==effective['T_ply_imu']
    trace=[json.loads(line) for line in (run/'pose_trace.jsonl').read_text().splitlines()]
    for i,row in enumerate(trace):
        if prediction=='previous_pose':
            np.testing.assert_array_equal(row['registered_T'],row['final_T'])
        elif i==0:
            np.testing.assert_allclose(row['registered_T'],row['final_T'],rtol=0,atol=2e-15)
        assert row['proposed_scale']==row['final_scale']
        assert row['pose_decision']==('accepted_registration' if prediction=='previous_pose' else
                                    'initialized_from_registration' if i==0 else 'monte_carlo_refined_registration')
        if prediction=='gpu_monte_carlo':
            assert row['inertial']['prediction']['latest_bag_timestamp_ns']<=row['timestamp_ns']
            assert row['inertial']['update']['measurement_accepted']
            assert row['inertial']['update']['kalman_used'] is False
            assert row['monte_carlo']['posterior_recentered_by_refinement'] is False
    poses=np.loadtxt(run/'poses.txt');assert poses.shape==(3,12)
    rows=list(csv.DictReader((run/'result_3.csv').open()))
    assert list(rows[0])==['idx','x','y','yaw'] and [r['idx'] for r in rows]==['0','1','2']
    timings=list(csv.DictReader((run/'timings.csv').open()))
    assert int(timings[0]['timestamp_ns'])==stamp
    for i,r in enumerate(rows):
        assert abs(float(r['x'])-poses[i,3])<1e-8
        assert abs(float(r['yaw'])-np.arctan2(poses[i,4],poses[i,0]))<1e-8
    assert not (run/'analysis').exists()

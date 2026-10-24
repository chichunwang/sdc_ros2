"""End-to-end CUDA pipeline using exclusively generated map/PLY data."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from plyfile import PlyData,PlyElement
from test_core import CFG,cloud


def test_generated_pipeline(tmp_path,monkeypatch):
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
    cfg['version']='SYNTHETIC_VALIDATION_ONLY'
    cfg['registration']['stages']=[{'mode':'ndt','voxel_m':2.,'iterations':30},{'mode':'icp','voxel_m':1.,'iterations':30}]
    config=tmp_path/'config.json';config.write_text(json.dumps(cfg))
    run=tmp_path/'synthetic_run';run.mkdir();(run/'source_manifest.json').write_text(json.dumps({'test_fixture':True}))
    script=Path(__file__).resolve().parents[1]/'scripts/run_track3.py'
    spec=importlib.util.spec_from_file_location('run_track3',script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(sys,'argv',[str(script),'--config',str(config),'--run-dir',str(run),'--max-frames','3'])
    assert module.main()==0
    info=json.loads((run/'test_info.json').read_text());assert info['status']=='success'
    assert info['frame_count']==3 and info['imu_used'] is False and info['map_accepted_frames']==3
    assert not (root/'Track3.db3').exists()
    trace=[json.loads(line) for line in (run/'pose_trace.jsonl').read_text().splitlines()]
    for row in trace:
        np.testing.assert_array_equal(row['registered_T'],row['final_T'])
        assert row['proposed_scale']==row['final_scale']
        assert row['pose_decision']=='accepted_registration'
    poses=np.loadtxt(run/'poses.txt');assert poses.shape==(3,12)
    rows=list(csv.DictReader((run/'result_3.csv').open()))
    assert list(rows[0])==['idx','x','y','yaw'] and [r['idx'] for r in rows]==['0','1','2']
    timings=list(csv.DictReader((run/'timings.csv').open()))
    assert int(timings[0]['timestamp_ns'])==stamp
    for i,r in enumerate(rows):
        assert abs(float(r['x'])-poses[i,3])<1e-8
        assert abs(float(r['yaw'])-np.arctan2(poses[i,4],poses[i,0]))<1e-8
    assert not (run/'analysis').exists()

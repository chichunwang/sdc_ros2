"""CPU-only entry/contract checks; never execute Track1 registration/CUDA."""
import ast
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import numpy as np
import pytest
import yaml
from track1 import parameters as p
from track1.analysis import analyze, validate

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_independent_track1_parameters_and_engine_source():
    from track2 import parameters as other
    assert p.EXPECTED_FRAMES == 3011 and p.DATA_DIR.endswith('/Track1')
    assert p.VERSION == 'track1_causal_metric_correction_memory_v4'
    assert p is not other
    for name in vars(p):
        if name.isupper() and name not in ('VERSION','DATA_DIR','EXPECTED_FRAMES'):
            assert getattr(p,name) == getattr(other,name), name
    provenance = json.loads((ROOT/'track1/origin.json').read_text())
    for name in ('causal_prior','correction_filter','metric_registration','map_io','sensors'):
        file = f'track1/{name}.py'
        assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest() == provenance['files'][file]['parent_sha256']
    for file in list((ROOT/'track1').glob('*.py'))+[ROOT/'scripts/run_track1.py']:
        imports = [n for n in ast.walk(ast.parse(file.read_text())) if isinstance(n,(ast.Import,ast.ImportFrom))]
        names = [n.module for n in imports if isinstance(n,ast.ImportFrom) and n.level == 0]
        names += [a.name for n in imports if isinstance(n,ast.Import) for a in n.names]
        assert not any(n and (n=='track2' or n.startswith('track2.')) for n in names)


def test_track1_causal_sensor_and_correction_prefix():
    from track1.causal_prior import CausalSensorPrior
    from track1.correction_filter import CausalCorrectionFilter
    def run(n):
        T=np.eye(4);T[:3,3]=[-4800.,2274.,110.]
        sensor=CausalSensorPrior(T,1_000_000_000); memory=CausalCorrectionFilter(); result=[]
        for k in range(n):
            stamp=1_000_000_000+k*50_000_000
            sensor.event(dict(kind='imu',timestamp_ns=stamp,quaternion=[0.,0.,0.,1.],gyro=[0.,0.,0.]))
            if k%5==0:sensor.event(dict(kind='gnss',timestamp_ns=stamp,xy=[-4800.+k*.25,2274.],z=110.))
            prior,d=sensor.for_frame(stamp)
            memory.predict(prior,stamp,float(np.linalg.norm(sensor.position[2:])))
            c,md=memory.update(None if k in (9,10,11) else [.4,-.2,.01],np.diag([.0225,.0225,.0001]))
            result.append((prior.tolist(),d,c.tolist(),md))
        return result
    assert run(17) == run(35)[:17]


def test_cli_help_limits_and_shared_gpu_lock(tmp_path):
    runner=script('run_track1')
    assert runner.OUTPUTS[0]=='result_1.csv'
    assert runner.TIMINGS==['idx','original_idx','timestamp_ns','backend','load_ms','registration_ms','total_ms','map_accepted','scale']
    import sys
    answer=subprocess.run([sys.executable,str(ROOT/'scripts/run_track1.py'),'--help'],capture_output=True,text=True)
    assert answer.returncode==0 and '--full' in answer.stdout
    for value in ('0','3011','-1','abc'):
        answer=subprocess.run(['bash',str(ROOT/'scripts/run_track1.sh'),'--max-frames',value],capture_output=True,text=True)
        assert answer.returncode==2 and '1..3010' in answer.stderr
    # Isolated shell fixture: failed flock must exit before snapshots, Docker,
    # RViz or any localization command; this cannot launch a real full run.
    fake=tmp_path/'workspace';(fake/'scripts').mkdir(parents=True)
    entry=fake/'scripts/run_track1.sh';entry.write_bytes((ROOT/'scripts/run_track1.sh').read_bytes())
    commands=tmp_path/'bin';commands.mkdir();flock=commands/'flock'
    flock.write_text('#!/bin/sh\nexit 1\n');flock.chmod(0o755)
    import os
    env=dict(os.environ,PATH=str(commands)+os.pathsep+os.environ['PATH'])
    answer=subprocess.run(['bash',str(entry),'--max-frames','1'],env=env,capture_output=True,text=True)
    assert answer.returncode==1 and 'GPU 工作執行中' in answer.stderr
    assert sorted(f.name for f in (fake/'outputs').iterdir())==['.gpu.lock']


def test_snapshot_gate_detects_tampering(tmp_path,monkeypatch):
    module=script('run_track1');run=tmp_path/'run';source=run/'source';source.mkdir(parents=True)
    artifact=source/'example.py';artifact.write_text('VERSION=1\n')
    (run/'source_manifest.json').write_text(json.dumps(dict(sources={'example.py':hashlib.sha256(artifact.read_bytes()).hexdigest()})))
    monkeypatch.setattr(module,'__file__',str(source/'scripts/run_track1.py'))
    module.verify_snapshot(run)
    artifact.write_text('VERSION=2\n')
    with pytest.raises(ValueError,match='SHA mismatch'):module.verify_snapshot(run)


def test_track1_rviz_config_and_publisher_use_independent_topics():
    from track1.visualization import Track1RvizPublisher
    config=yaml.safe_load((ROOT/'config/track1.rviz').read_text())
    assert config['Visualization Manager']['Views']['Current']['Target Frame']=='track1_camera'
    topics=[d['Topic']['Value'] for d in config['Visualization Manager']['Displays'] if 'Topic' in d]
    assert topics and all(t.startswith('/track1/') for t in topics)
    v=Track1RvizPublisher();clouds=[];transforms=[];status=[]
    v.current_cloud_pub=SimpleNamespace(publish=clouds.append);v.tf=SimpleNamespace(sendTransform=transforms.append)
    v.status_pub=SimpleNamespace(publish=status.append)
    try:
        T=np.eye(4);T[:3,3]=[-4800.,2274.,110.]
        raw=np.array([[1.,0.,0.]]);v.publish_frame(raw,T,8.,1_000_000_001,accepted=False,reason='ground_only_prior')
        assert transforms[0].child_frame_id=='track1_camera'
        assert np.frombuffer(clouds[0].data,dtype='<f4')[0]==pytest.approx(-4792.)
        assert json.loads(status[0].data)['reason']=='ground_only_prior'
    finally:v.close()


def test_track1_full_analysis_3011_contract_and_read_only(tmp_path):
    run=tmp_path/'generated_full';run.mkdir();n=3011
    rows=np.column_stack([np.arange(n),np.arange(n)*.01,np.zeros(n),np.zeros(n)])
    np.savetxt(run/'result_1.csv',rows,delimiter=',',header='idx,x,y,yaw',comments='',fmt=['%d','%.9f','%.9f','%.11f'])
    T=np.eye(4);T[2,3]=1.6;poses=np.tile(T[:3].ravel(),(n,1));poses[:,3]=rows[:,1]
    np.savetxt(run/'poses.txt',poses,fmt='%.12f')
    with (run/'timings.csv').open('w',newline='') as f, (run/'frame_manifest.csv').open('w',newline='') as g:
        w=csv.writer(f);m=csv.writer(g)
        w.writerow(['idx','original_idx','timestamp_ns','backend','load_ms','registration_ms','total_ms','map_accepted','scale'])
        m.writerow(['idx','original_idx','timestamp_ns','camera_timestamp_ns','path','sha256'])
        for k in range(n):
            stamp=1_000_000_000+k*50_000_000
            w.writerow([k,k,stamp,'generated_contract_only',1,2,4,'true',8])
            m.writerow([k,k,stamp,stamp-1,'generated','generated'])
    (run/'test_info.json').write_text(json.dumps(dict(status='completed',mode='full',frame_count=n,generated=True)))
    with (run/'observations.jsonl').open('w') as f:
        for k in range(n):
            memory=dict(correction=[.1,.2,.01],covariance=np.diag([.01,.01,.0001]).tolist(),innovation_nis=1.,accepted=True,seconds_since_update=0.)
            f.write(json.dumps(dict(state=[k*.01,0.,0.,0.,.2],scale=8.,registration=dict(correction_filter=memory)))+'\n')
    before={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in run.iterdir() if f.is_file()}
    validate(run);analyze(run)
    assert before=={name:hashlib.sha256((run/name).read_bytes()).hexdigest() for name in before}
    for stem in ('01_trajectory','02_pose_changes','03_registration_timing','04_fusion_and_scale','05_map_correction_memory'):
        for ext in ('png','pdf'):assert (run/'analysis'/f'{stem}.{ext}').stat().st_size>1000
    with pytest.raises(ValueError,match='Incomplete output'):validate(run,expected=3299)
    (run/'result_1.csv').rename(run/'result_2.csv')
    with pytest.raises(FileNotFoundError):validate(run)
    (run/'result_2.csv').rename(run/'result_1.csv')

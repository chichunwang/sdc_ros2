"""Generated ASCII runner output comparison against original input reader."""
from pathlib import Path
import importlib.util
import json
import numpy as np
from plyfile import PlyData,PlyElement
from track2 import parameters as p
from test_track2_metric_scale import scene,truth,raw_cloud
from test_track2_restored import imu


def test_ascii_runner_pose_bytes_equal_legacy_reader(tmp_path,monkeypatch):
    # This fixture isolates reader parity; image integration has its own tests.
    monkeypatch.setattr(p, 'IMAGE_GROUND_SPEED_ENABLED', False)
    import cupy as cp
    assert cp.cuda.runtime.getDeviceCount()>0
    import track2.sensors as sensors
    import track2.camera as camera
    import track3.time_alignment as clock
    from track3.io import load_ply as legacy
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('equivalence_runner',root/'scripts/run_track2.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    monkeypatch.setattr(runner,'verify_snapshot',lambda _:None)
    folder=tmp_path/'generated';(folder/'raw_ply').mkdir(parents=True)
    xyz=scene();T=truth();dtype=np.dtype([('x','<f4'),('y','<f4'),('z','<f4')]);a=np.empty(len(xyz),dtype=dtype)
    for k,n in enumerate(('x','y','z')):a[n]=xyz[:,k]
    (folder/'map.pcd').write_bytes(f'FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nPOINTS {len(a)}\nDATA binary\n'.encode()+a.tobytes())
    np.savetxt(folder/'init_guess.txt',T,delimiter=',');(folder/'Track2.db3').write_bytes(b'generated sensor mocks')
    dtype=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    for i,scale in enumerate((7.9,8.,8.1)):
        raw=raw_cloud(xyz,T,scale);v=np.zeros(len(raw),dtype=dtype)
        for k,n in enumerate(('x','y','z')):v[n]=raw[:,k]
        PlyData([PlyElement.describe(v,'vertex')],text=True).write(str(folder/'raw_ply'/f'{i:06d}_ts_{1_000_000_000+i*50_000_000}.ply'))
    class Sensors:
        def __init__(self,*_):pass
        def take(self,arrival,camera):return [imu(camera)]
        def summary(self):return {'generated':True}
        def close(self):pass
    class Clock:
        def __init__(self,*_):pass
        def for_frame(self,t):return t
        def close(self):pass
    class Camera:
        def __init__(self,*_):self.sources={'generated':'identity'}
        def apply(self,raw):return raw,{'generated':True}
    monkeypatch.setattr(sensors,'SensorStream',Sensors);monkeypatch.setattr(clock,'CameraClock',Clock);monkeypatch.setattr(camera,'FrontMask',Camera)
    monkeypatch.setattr(p,'DATA_DIR',str(folder));monkeypatch.setattr(p,'EXPECTED_FRAMES',3)
    new,old=tmp_path/'candidate',tmp_path/'legacy';new.mkdir();old.mkdir()
    runner.execute(new,max_frames=3)
    monkeypatch.setattr(runner,'load_ply',legacy)
    runner.execute(old,max_frames=3)
    for name in ('result_2.csv','poses.txt','pose_trace.jsonl','frame_manifest.csv'):
        assert (new/name).read_bytes()==(old/name).read_bytes(),name
    assert json.loads((new/'test_info.json').read_text())['status']=='completed'

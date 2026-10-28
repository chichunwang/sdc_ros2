"""CPU / ROS presentation checks only; no localization or CUDA import."""
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import yaml
from track2.visualization import Track2RvizPublisher


def script_module(name):
    f = Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, f)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_track2_publisher_display_pose_color_tf_and_inputs_unchanged():
    v = Track2RvizPublisher()
    raw = np.array([[1., 0., 0.], [2., 1., -.2]])
    T = np.eye(4); T[:3, 3] = [10, 20, 3]
    old_raw, old_T = raw.copy(), T.copy()
    clouds = []; poses = []; markers = []; status = []; transforms = []
    v.current_cloud_pub = SimpleNamespace(publish=clouds.append)
    v.pose_pub = SimpleNamespace(publish=poses.append)
    v.registration_status_pub = SimpleNamespace(publish=markers.append)
    v.status_pub = SimpleNamespace(publish=status.append)
    v.tf = SimpleNamespace(sendTransform=transforms.append)
    try:
        stamp = 1727241181291090775
        v.publish_frame(raw, T, 8., stamp, accepted=False, reason='all_ranges_rejected', stationary=True)
        actual = np.frombuffer(clouds[0].data, dtype='<f4').reshape(-1, 4)[:, :3]
        np.testing.assert_allclose(actual, raw*8+T[:3, 3], atol=1e-6)
        assert np.frombuffer(clouds[0].data, dtype='<u4').reshape(-1, 4)[0, 3] == (255<<16)|(160<<8)|40
        assert transforms[0].child_frame_id == 'track2_camera'
        assert poses[0].header.stamp.sec*10**9+poses[0].header.stamp.nanosec == stamp
        assert markers[0].text == 'SENSOR PRIOR: all_ranges_rejected | scale=8.000 | STATIONARY'
        assert not json.loads(status[0].data)['map_accepted']
        np.testing.assert_array_equal(raw, old_raw); np.testing.assert_array_equal(T, old_T)
    finally:
        v.close()


def test_config_uses_only_track2_topics_and_frame():
    root = Path(__file__).resolve().parents[1]
    config = (root/'config/track2.rviz').read_text()
    assert '/track3/' not in config and 'track3_camera' not in config
    data = yaml.safe_load(config)
    assert data['Visualization Manager']['Global Options']['Fixed Frame'] == 'map'
    assert data['Visualization Manager']['Views']['Current']['Target Frame'] == 'track2_camera'


def test_viewer_waits_for_complete_line():
    m = script_module('watch_track2_rviz')
    f = io.StringIO('{"idx":0}\n{"idx":1')
    assert m.complete_json_lines(f) == [{'idx': 0}]
    assert f.read() == '{"idx":1'
    # Some pytest/scipy optional dependencies import CuPy without CUDA work.
    # Verify the observer itself in a fresh interpreter that forbids CuPy.
    import subprocess
    script = Path(m.__file__)
    code = '''import builtins,runpy
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0]=='cupy': raise AssertionError('CUDA import in observer')
    return original(name,*args,**kwargs)
builtins.__import__=guarded
runpy.run_path(%r,run_name='observer_import_check')
''' % str(script)
    subprocess.run([sys.executable, '-c', code], check=True)


def test_rviz_startup_failure_is_separate_and_explicit(tmp_path, monkeypatch):
    m = script_module('start_track2_rviz')
    run = tmp_path/'generated'; run.mkdir()
    monkeypatch.delenv('DISPLAY', raising=False)
    monkeypatch.setattr(sys, 'argv', ['start', '--run-dir', str(run)])
    assert m.main() == 1
    assert json.loads((run/'rviz_status.json').read_text())['status'] == 'failed'
    assert not (run/'poses.txt').exists()

"""Snapshot startup must retain ROS packages and select the copied track3 code."""
import os
from pathlib import Path
import shutil
import subprocess
import sys


def test_snapshot_imports_preserve_ros_environment(tmp_path):
    root = Path(__file__).resolve().parents[1]
    snapshot = tmp_path / 'snapshot'
    (snapshot / 'scripts').mkdir(parents=True)
    shutil.copyfile(root / 'scripts/run_track3.py', snapshot / 'scripts/run_track3.py')
    shutil.copytree(root / 'track3', snapshot / 'track3', ignore=shutil.ignore_patterns('__pycache__'))
    probe = '''
import pathlib, runpy, sys
import rclpy.serialization
from sensor_msgs.msg import PointCloud2
runpy.run_path(sys.argv[1], run_name="startup_probe")
import track3.io
assert pathlib.Path(track3.io.__file__).resolve().parent.parent == pathlib.Path(sys.argv[1]).resolve().parents[1]
assert isinstance(rclpy.serialization.serialize_message(PointCloud2()), bytes)
print("snapshot source and ROS imports PASS")
'''
    environment = os.environ.copy()
    # Match the image's inherited live source path, retaining its ROS paths.
    environment['PYTHONPATH'] = os.pathsep.join([str(root), environment.get('PYTHONPATH', '')])
    result = subprocess.run(
        [sys.executable, '-c', probe, str(snapshot / 'scripts/run_track3.py')],
        env=environment, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'snapshot source and ROS imports PASS' in result.stdout

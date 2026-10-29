"""離線 ROS stub 整合測試：合成訊息與圖片皆非作業實測證據。"""
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import zipfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'colcon_ws/src/fusion/scripts'))


def message(stamp, x=0., y=0., yaw=0., covariance=None, gps=False):
    pose = SimpleNamespace(position=SimpleNamespace(x=x, y=y, z=0.),
                           orientation=SimpleNamespace(x=0., y=0., z=math.sin(yaw/2), w=math.cos(yaw/2)))
    return SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=stamp//10**9, nanosec=stamp % 10**9),
                               frame_id='origin'),
        child_frame_id='',
        pose=SimpleNamespace(pose=pose, covariance=(np.zeros((6, 6)) if covariance is None
                                                   else covariance).ravel().tolist()))


class StubNode:
    result_dir = None

    def __init__(self, name):
        self.params = {}
        self.logger = MagicMock()

    def declare_parameter(self, name, default):
        self.params[name] = str(self.result_dir) if name == 'result_dir' else default

    def get_parameter(self, name):
        return SimpleNamespace(value=self.params[name])

    def get_logger(self):
        return self.logger

    def create_publisher(self, *args):
        return MagicMock()

    def create_subscription(self, *args):
        return MagicMock()


def load_fusion():
    modules = {name: ModuleType(name) for name in (
        'rclpy', 'rclpy.node', 'rclpy.executors', 'rclpy.qos',
        'geometry_msgs', 'geometry_msgs.msg', 'nav_msgs', 'nav_msgs.msg')}
    modules['rclpy.node'].Node = StubNode
    modules['rclpy.executors'].ExternalShutdownException = RuntimeError
    modules['rclpy.qos'].QoSProfile = MagicMock()
    modules['rclpy.qos'].ReliabilityPolicy = SimpleNamespace(BEST_EFFORT='best_effort')
    modules['geometry_msgs.msg'].PoseWithCovarianceStamped = lambda: message(0)
    modules['nav_msgs.msg'].Odometry = lambda: message(0)
    spec = importlib.util.spec_from_file_location('offline_fusion_test',
                                                ROOT / 'colcon_ws/src/fusion/scripts/fusion.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


fusion = load_fusion()


class OutputTests(unittest.TestCase):
    def setUp(self):
        parent = os.environ.get('HW4_TEST_OUTPUT_DIR')
        self.temp = tempfile.TemporaryDirectory(prefix='hw4_synthetic_', dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        StubNode.result_dir = self.folder / 'run'

    def test_integer_epoch_alignment_and_duplicate_last_wins(self):
        epoch = 1_700_000_000_000_000_000
        rows = [(epoch+1, 999., 999.), (epoch+1, 1., 0.), (epoch+3, 30., 0.)]
        gt = [(epoch, 0., 0.), (epoch+2, 2., 0.)]
        metrics, errors = fusion.position_errors(rows, gt)
        self.assertEqual(metrics['samples'], 1)
        self.assertEqual(errors[0][0], epoch+1)
        self.assertEqual(metrics['rmse_xy_m'], 0.)
        self.assertEqual(metrics['max_interpolated_gt_gap_s'], 2e-9)

    def test_no_gt_extrapolation(self):
        metrics, errors = fusion.position_errors([(0, 0., 0.), (20, 20., 0.)],
                                                [(5, 5., 0.), (10, 10., 0.)])
        self.assertEqual(metrics['samples'], 0)
        self.assertEqual(errors, [])

    def test_known_rmse(self):
        metrics, _ = fusion.position_errors([(0, 3., 4.), (10, 0., 0.)],
                                           [(0, 0., 0.), (10, 0., 0.)])
        self.assertAlmostEqual(metrics['rmse_xy_m'], math.sqrt(12.5))

    def populate_node(self):
        node = fusion.Fusion()
        epoch = 1_700_000_000_000_000_000
        radar_cov = np.zeros((6, 6))
        radar_cov[np.ix_([0, 1, 5], [0, 1, 5])] = np.diag([.01, .01, .001])
        node.odometryCallback(message(epoch, yaw=math.pi/2, covariance=radar_cov))
        node.gpsCallback(message(epoch+500_000_000, y=.5))  # 明確測試全零 fallback。
        node.odometryCallback(message(epoch+10**9, y=1., yaw=math.pi/2, covariance=radar_cov))
        before = node.filter.ekf.pose.copy(), node.filter.ekf.S.copy()
        node.gtCallback(message(epoch))
        node.gtCallback(message(epoch+10**9, y=1.))
        np.testing.assert_array_equal(before[0], node.filter.ekf.pose)
        np.testing.assert_array_equal(before[1], node.filter.ekf.S)
        return node

    def test_callbacks_gt_isolation_and_saved_evidence(self):
        node = self.populate_node()
        with patch('sys.stdout'):
            node.save_results()
        folder = node.result_dir
        summary = json.loads((folder / 'summary.json').read_text())
        self.assertEqual(summary['data_status'], 'has_sensor_data')
        self.assertEqual(summary['gps_updates'], 1)
        self.assertEqual(summary['evaluation']['sensors']['estimate']['samples'], 3)
        self.assertAlmostEqual(summary['evaluation']['sensors']['estimate']['rmse_xy_m'], 0., places=10)
        stats = json.loads((folder / 'covariance_statistics.json').read_text())
        self.assertEqual(stats['gps']['fallback_messages'], 1)
        np.testing.assert_array_equal(stats['gps']['first_selected_covariance'], np.zeros((2, 2)))
        np.testing.assert_allclose(stats['gps']['first_effective_pose_covariance'], np.eye(2)*9)
        for name in ('result.png', '02_position_error.png', 'analysis.json', 'source_manifest.json'):
            self.assertTrue((folder / name).is_file(), name)
        for name in ('fusion.py', 'EKF.py'):
            self.assertEqual((folder / 'source' / name).read_bytes(),
                             (ROOT / 'colcon_ws/src/fusion/scripts' / name).read_bytes())
        if os.environ.get('HW4_TEST_OUTPUT_DIR'):
            # 保留明確標示的合成圖表供機制檢查，不能當成作業實測結果。
            preview = Path(os.environ['HW4_TEST_OUTPUT_DIR']) / 'analysis/synthetic_ros_stub'
            preview.parent.mkdir(exist_ok=True)
            shutil.copytree(folder, preview)
            (preview / 'SYNTHETIC_TEST_ONLY.txt').write_text(
                'ROS stubs and synthetic messages only. Not real bag results; do not submit.\n', encoding='utf-8')
        output = node.posePub.publish.call_args.args[0]
        self.assertEqual(output.header.frame_id, 'origin')
        self.assertEqual(len(output.pose.covariance), 36)
        self.assertAlmostEqual(output.pose.pose.orientation.z, math.sqrt(.5))
        with self.assertRaises(FileExistsError):
            fusion.Fusion()

    def test_wrong_frame_is_rejected(self):
        node = fusion.Fusion()
        sample = message(0)
        sample.header.frame_id = 'different_frame'
        node.odometryCallback(sample)
        self.assertIsNone(node.filter.ekf)
        self.assertEqual(node.invalid_messages, 1)

    def test_submission_collects_matching_sources_only(self):
        spec = importlib.util.spec_from_file_location('hw4_submission', ROOT / 'scripts/prepare_submission.py')
        report = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(report)
        node = self.populate_node()
        with patch('sys.stdout'):
            node.save_results()
        settings = dict(RUN_DIR=str(node.result_dir), TERMINAL_SCREENSHOT=str(node.result_dir / 'result.png'),
                        VIDEO_URL='https://youtu.be/synthetic_fixture_only')
        evidence = report.collect(settings)
        self.assertEqual(evidence['missing'], [])
        (node.result_dir / 'source/EKF.py').write_text('tampered fixture', encoding='utf-8')
        evidence = report.collect(settings)
        self.assertTrue(any('EKF.py' in item for item in evidence['missing']))

    def test_submission_zip_contains_exact_four_frozen_files(self):
        spec = importlib.util.spec_from_file_location('hw4_submission_zip', ROOT / 'scripts/prepare_submission.py')
        report = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(report)
        node = self.populate_node()
        with patch('sys.stdout'):
            node.save_results()
        # 僅用於測試打包機制的假資料、截圖與 URL；輸出只在測試目錄。
        config = dict(STUDENT_ID='TEST_FIXTURE_ONLY', RUN_DIR=str(node.result_dir),
                      TERMINAL_SCREENSHOT=str(node.result_dir / 'result.png'),
                      VIDEO_URL='https://youtu.be/synthetic_fixture_only')
        target = self.folder / 'package_test'
        target.mkdir()
        (target / 'submission_settings.py').write_text(
            '\n'.join(f'{key} = {value!r}' for key, value in config.items()), encoding='utf-8')
        report.ROOT = target
        with patch('sys.stdout'):
            report.main()
        package = target / 'output/TEST_FIXTURE_ONLY_hw4.zip'
        with zipfile.ZipFile(package) as archive:
            self.assertEqual(set(archive.namelist()), {
                'TEST_FIXTURE_ONLY_hw4.pdf', 'result.png', 'fusion.py', 'EKF.py'})
            self.assertEqual(archive.read('EKF.py'), (node.result_dir / 'source/EKF.py').read_bytes())
        self.assertTrue((target / 'output/submission_manifest.json').is_file())
        if os.environ.get('HW4_TEST_OUTPUT_DIR'):
            preview = Path(os.environ['HW4_TEST_OUTPUT_DIR']) / 'analysis/submission_fixture'
            shutil.copytree(target, preview)
            (preview / 'SYNTHETIC_TEST_ONLY.txt').write_text(
                'Package mechanism fixture only. Student ID, screenshot and video URL are synthetic; do not submit.\n',
                encoding='utf-8')


if __name__ == '__main__':
    unittest.main()

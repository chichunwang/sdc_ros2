"""CPU-only fake IMU export tests; no ROS or real bag access."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'export_track3_imu.py'
spec = importlib.util.spec_from_file_location('export_track3_imu', SCRIPT)
imu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imu)
from causal_test_support import fixture_run
TEST_RUN = fixture_run()


def message(ns=90):
    return NS(header=NS(stamp=NS(sec=0, nanosec=ns), frame_id='imu'),
              orientation=NS(x=0., y=0., z=0., w=1.),
              angular_velocity=NS(x=1., y=2., z=3.),
              linear_acceleration=NS(x=4., y=5., z=6.),
              **{key: [0.] * 9 for key in imu.COVARIANCES})


def arrays():
    return imu.extract_messages([(100, message()), (200, message(80))], lambda msg: msg, 200)


class ExportTests(unittest.TestCase):
    def test_dtype_and_preserve_header_order(self):
        data = arrays()
        np.testing.assert_array_equal(data['measurement_ns'], [90, 80])
        self.assertEqual(data['availability_ns'].dtype, np.dtype('int64'))
        self.assertEqual(data['measurement_ns'].dtype, np.dtype('int64'))
        self.assertEqual(data['frame_id'].dtype.kind, 'U')
        for key in ('gyro_rad_s', 'accel_m_s2', 'quaternion_xyzw') + imu.COVARIANCES:
            self.assertEqual(data[key].dtype, np.dtype('float64'))
        np.testing.assert_array_equal(data['gyro_rad_s'][0], [1, 2, 3])

    def test_cutoff_sql(self):
        connection = Mock()
        connection.execute.return_value.fetchall.return_value = [(7, 'sensor_msgs/msg/Imu')]
        imu.query_rows(connection, 200)
        sql, params = connection.execute.call_args.args
        self.assertIn('timestamp <= ?', sql)
        self.assertIn('ORDER BY timestamp, id', sql)
        self.assertEqual(params, (7, 200))
        self.assertEqual(connection.execute.call_args_list[0].args[1], ('/imu/data',))

    def test_missing_topic_and_wrong_type(self):
        for topics in ([], [(1, 'wrong')], [(1, 'sensor_msgs/msg/Imu'), (2, 'sensor_msgs/msg/Imu')]):
            connection = Mock()
            connection.execute.return_value.fetchall.return_value = topics
            with self.assertRaises(ValueError):
                imu.query_rows(connection, 200)

    def test_invalid_values(self):
        changes = [('measurement_ns', (0,), 101), ('measurement_ns', (0,), 0),
                   ('availability_ns', (1,), 100), ('availability_ns', (0,), 0),
                   ('availability_ns', (1,), 201), ('quaternion_xyzw', (0, 3), 0),
                   ('quaternion_xyzw', (0, 3), 1.01), ('gyro_rad_s', (0, 0), np.nan),
                   ('accel_m_s2', (0, 0), np.inf), ('frame_id', (1,), 'new')]
        changes += [(key, (0, 0), -1) for key in imu.COVARIANCES]
        for key, index, value in changes:
            with self.subTest(key=key, value=value):
                data = arrays()
                data[key][index] = value
                with self.assertRaises(ValueError):
                    imu.validate_arrays(data, 200)

    def test_schema(self):
        data = arrays()
        data['gyro_rad_s'] = data['gyro_rad_s'].astype(np.float32)
        with self.assertRaises(ValueError):
            imu.validate_arrays(data, 200)
        data = arrays()
        data['extra'] = np.ones(2)
        with self.assertRaises(ValueError):
            imu.validate_arrays(data, 200)
        with self.assertRaises(ValueError):
            imu.extract_messages([], lambda msg: msg, 200)
        for cutoff in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                imu.query_rows(Mock(), cutoff)

    def test_zero_covariance_metadata_and_exclusive_output(self):
        with tempfile.TemporaryDirectory(prefix='imu_export_', dir=TEST_RUN) as directory:
            directory = Path(directory)
            bag = directory / 'fake.db3'
            bag.write_bytes(b'fake identity only; not SQLite')
            data = arrays()
            metadata = imu.build_metadata(data, 200, bag)
            self.assertEqual(metadata['all_zero_covariance_message_counts'],
                             {key: 2 for key in imu.COVARIANCES})
            self.assertEqual(metadata['zero_covariance_meaning'], 'unknown')
            self.assertFalse(metadata['calibrated'])
            self.assertFalse(metadata['future_messages_used'])
            self.assertEqual(metadata['header_minus_bag_offset_ns'], {'min': -120, 'max': -10})
            output = directory / 'imu.npz'
            imu.write_export(output, data, metadata)
            with np.load(output, allow_pickle=False) as stored:
                for key in data:
                    np.testing.assert_array_equal(stored[key], data[key])
            self.assertEqual(json.loads(output.with_suffix('.json').read_text())['count'], 2)
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                imu.write_export(output, data, metadata)
            self.assertEqual(output.read_bytes(), original)
            sidecar_only = directory / 'sidecar.npz'
            sidecar_only.with_suffix('.json').write_text('{}')
            with self.assertRaises(FileExistsError):
                imu.write_export(sidecar_only, data, metadata)
            self.assertFalse(sidecar_only.exists())

    def test_new_full_and_existing_results(self):
        with tempfile.TemporaryDirectory(prefix='imu_full_', dir=TEST_RUN) as directory:
            root = Path(directory)
            run = root / 'track03_full_2610231450'
            run.mkdir()
            output = run / 'imu.npz'
            with patch.object(imu, 'RUN_ROOT', root):
                self.assertEqual(imu.validate_output(output)[0], output)
                for name in ('result.txt', 'diagnostic.csv', 'validation.json'):
                    result = run / name
                    result.write_text('existing')
                    with self.assertRaises(ValueError):
                        imu.validate_output(output)
                    self.assertFalse(output.exists())
                    result.unlink()

    def test_output_scope(self):
        with self.assertRaises(ValueError):
            imu.validate_output(SCRIPT.parents[1] / 'outputs/run/forbidden.npz')
        with self.assertRaises(ValueError):
            imu.validate_output(TEST_RUN / 'wrong.csv')


if __name__ == '__main__':
    unittest.main()

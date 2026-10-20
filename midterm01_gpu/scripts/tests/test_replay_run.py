"""Input/coordinate contracts; synthetic fixtures do not measure accuracy."""
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

from replay_run import load_run, rotation_values, sha256, write_replay


class ReplayRunTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / 'data'
        self.run = self.root / 'outputs/run/saved'
        self.run.mkdir(parents=True)
        self.clouds = self.data / 'Track1/raw_ply'
        self.clouds.mkdir(parents=True)
        (self.data / 'Track1/map.pcd').write_text('fixture only')
        self.params = dict(map_path='/root/data/Track1', map_file='map.pcd', ply_path='/root/data/Track1/raw_ply',
                           save_file='result.txt', ply_scale=3., map_voxel_leaf_size=1., input_voxel_leaf_size=.3,
                           submap_half_size=100., submap_half_height=30.)
        self.save_params()
        self.stamps = [1727234445100462219, 1727234445100462220]
        self.names = [f'{i:06d}_ts_{stamp}.ply' for i, stamp in enumerate(self.stamps)]
        for name in self.names:
            (self.clouds / name).write_text('fixture only')
        with (self.run / 'frames.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['idx', 'timestamp_ns', 'filename'])
            writer.writerows((i, stamp, name) for i, (stamp, name) in enumerate(zip(self.stamps, self.names)))
        # 90 degrees around map Z, nonzero translation: exposes double scale/transpose errors.
        self.pose = [0., -1., 0., 17., 1., 0., 0., -23., 0., 0., 1., 5.]
        self.save_poses([self.pose] * 2)

    def save_params(self):
        (self.run / 'params.json').write_text(json.dumps(self.params))

    def save_poses(self, poses):
        (self.run / 'result.txt').write_text('\n'.join(' '.join(map(str, row)) for row in poses) + '\n')

    def load(self, **kwargs):
        return load_run(self.root, self.data, self.run, **kwargs)

    def test_nanosecond_precision_and_translation_not_scaled(self):
        data = self.load()
        out = self.root / 'export'; out.mkdir()
        write_replay(data, out)
        with (out / 'poses.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([int(r['timestamp_ns']) for r in rows], self.stamps)
        self.assertEqual([float(rows[0][key]) for key in ('x_m', 'y_m', 'z_m')], [17., -23., 5.])
        self.assertAlmostEqual(float(rows[0]['yaw_deg']), 90.)
        # Independently rotate local +X with the exported quaternion: it must become map +Y.
        x, y, z, w = [float(rows[0][key]) for key in ('qx', 'qy', 'qz', 'qw')]
        rotated_x = [1 - 2 * (y*y + z*z), 2 * (x*y + z*w), 2 * (x*z - y*w)]
        for a, b in zip(rotated_x, [0., 1., 0.]): self.assertAlmostEqual(a, b)

    def test_rejects_variable_scale_run_before_fixed_scale_fields(self):
        self.params={'method':'map_scale_slow_v1','replay_supported':False}
        self.save_params()
        with self.assertRaisesRegex(ValueError,'逐幀尺度'):self.load()

    def test_full_3d_rotation_and_near_pi_quaternion(self):
        roll, pitch, yaw = .3, -.4, 2.9
        cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
        pose = [cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr, 1.,
                sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr, 2., -sp, cp*sr, cp*cr, 3.]
        q, rpy = rotation_values(pose)
        for a, b in zip(rpy, [roll, pitch, yaw]): self.assertAlmostEqual(a, b)
        x, y, z, w = q
        recovered = [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
        for a, b in zip(recovered, [pose[i] for i in (0,1,2,4,5,6,8,9,10)]): self.assertAlmostEqual(a, b)

    def test_rejects_reflection_and_nonfinite(self):
        for bad in ([1.,0,0,0, 0,1,0,0, 0,0,-1,0], [float('nan')] + self.pose[1:]):
            self.save_poses([bad] * 2)
            with self.assertRaises(ValueError): self.load()

    def test_rejects_missing_result_rows(self):
        self.save_poses([self.pose])
        with self.assertRaisesRegex(ValueError, '筆數不符'): self.load()

    def test_rejects_timestamp_order_and_track_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'Track'): self.load(track=2)
        text = (self.run / 'frames.csv').read_text().replace(str(self.stamps[1]), str(self.stamps[0]))
        (self.run / 'frames.csv').write_text(text)
        with self.assertRaisesRegex(ValueError, '整數時間'): self.load()

    def test_staged_container_symlink_and_original_idx(self):
        staging = self.run / 'source_ply'; staging.mkdir()
        for name in self.names: (staging / name).symlink_to('/root/data/Track1/raw_ply/' + name)
        self.params['ply_path'] = '/root/outputs/run/saved/source_ply'; self.save_params()
        metadata = dict(ply_scale=3., selected_frames=[dict(original_idx=400+i, timestamp_ns=stamp, filename=name,
                                              sha256=sha256(self.clouds/name))
                                        for i, (stamp, name) in enumerate(zip(self.stamps, self.names))])
        (self.run / 'source_input.json').write_text(json.dumps(metadata))
        data = self.load(start=1, count=1)
        self.assertEqual(data['records'][0]['original_idx'], 401)
        self.assertEqual(data['records'][0]['idx'], 1)
        self.assertEqual(data['records'][0]['path'], str(self.clouds/self.names[1]))
        (self.clouds/self.names[1]).write_text('modified')
        with self.assertRaisesRegex(ValueError, 'SHA256'): self.load()

    def test_rejects_out_of_range_selection_and_missing_cloud(self):
        for options in (dict(start=-1), dict(start=2), dict(count=3)):
            with self.assertRaisesRegex(ValueError, '區間'): self.load(**options)
        (self.clouds / self.names[0]).unlink()
        with self.assertRaisesRegex(ValueError, '點雲'): self.load()


if __name__ == '__main__': unittest.main()

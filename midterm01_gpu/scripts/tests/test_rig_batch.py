import csv
import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from calibrated_rig_geometry import PriorTrajectory
from run_rig_full import atomic_json, completed_frame, assert_resume_identity, verify_committed_inputs
from three_camera_source import sha256
from verify_rig_bias import estimate, spatially_unique


class RigBatchTests(unittest.TestCase):
    def test_resume_rejects_changed_original_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);row={};cameras=[]
            for tag in ('f','fl','fr'):
                row[tag+'_ts']='123';d=root/f'Track1_{tag}_images';d.mkdir()
                image=d/'000000_ts_123.jpg';image.write_bytes(b'original')
                cameras.append({'camera':tag,'image_sha256':sha256(image)})
            record={'original_idx':0,'cameras':cameras}
            verify_committed_inputs(record,row,root,1)
            (root/'Track1_fl_images/000000_ts_123.jpg').write_bytes(b'changed')
            with self.assertRaises(ValueError):verify_committed_inputs(record,row,root,1)

    def test_explicit_bounded_motion_extrapolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'prior.csv'
            with path.open('w') as stream:
                writer=csv.writer(stream);writer.writerow(['timestamp_ns','x','y','z','qx','qy','qz','qw'])
                for ns,x,yaw in [(1000000000,0.,0.),(1100000000,1.,10.)]:
                    writer.writerow([ns,x,0,0,*Rotation.from_euler('z',yaw,degrees=True).as_quat()])
            strict=PriorTrajectory(path)
            with self.assertRaises(ValueError):strict.pose(1120000000)
            bounded=PriorTrajectory(path,max_extrapolation_seconds=.03)
            pose=bounded.pose(1120000000)
            np.testing.assert_allclose(pose[:3,3],[1.2,0,0],atol=1e-12)
            self.assertAlmostEqual(Rotation.from_matrix(pose[:3,:3]).as_euler('xyz',degrees=True)[2],12.)
            self.assertIn(1120000000,bounded.extrapolated_queries)
            with self.assertRaises(ValueError):bounded.pose(1140000000)

    def test_resume_rejects_changed_identity_and_corrupt_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            identity={'scale':.7,'model':'abc'}
            assert_resume_identity(root,identity,False)
            assert_resume_identity(root,identity,True)
            with self.assertRaises(ValueError):assert_resume_identity(root,{'scale':.8,'model':'abc'},True)
            with self.assertRaises(ValueError):assert_resume_identity(root,identity,False)
            frame=root/'000000';frame.mkdir();(frame/'cloud.ply').write_bytes(b'original')
            record={'original_idx':0,'timestamp_ns':1727234445100462219,'files_sha256':{'cloud.ply':sha256(frame/'cloud.ply')}}
            atomic_json(frame/'frame.json',record)
            self.assertEqual(completed_frame(frame,0,record['timestamp_ns']),record)
            (frame/'cloud.ply').write_bytes(b'changed')
            with self.assertRaises(ValueError):completed_frame(frame,0,record['timestamp_ns'])

    def test_scale_fit_does_not_use_holdout_and_reports_failure(self):
        records=[]
        for track in (1,2):
            for role in ('fit','holdout'):
                records.extend({'track':track,'role':role,'original_idx':i,'logs':[np.log(2.)]*4} for i in range(4))
        first=estimate(records,'logs');self.assertTrue(first['passed']);self.assertAlmostEqual(first['depth_scale'],.5)
        for row in records:
            if row['role']=='holdout':row['logs']=[np.log(4.)]*4
        second=estimate(records,'logs');self.assertFalse(second['passed']);self.assertEqual(first['depth_scale'],second['depth_scale'])

    def test_multiple_sift_orientations_are_one_spatial_anchor(self):
        first=[cv2.KeyPoint(10,20,1),cv2.KeyPoint(10,20,1),cv2.KeyPoint(40,30,1)]
        second=[cv2.KeyPoint(30,20,1),cv2.KeyPoint(30,20,1),cv2.KeyPoint(50,30,1)]
        self.assertEqual(spatially_unique([(0,0),(1,1),(2,2)],first,second),[(0,0),(2,2)])


if __name__=='__main__':unittest.main()

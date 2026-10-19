from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from direct_heading import interpolate_pose_yaw
from prepare_sensor_prior import make_priors
from sensor_fusion import wrap


class DirectHeadingTests(unittest.TestCase):
    def test_branch_cut_and_short_edge_angular_extrapolation(self):
        t=np.arange(0.,5.,.25)
        y=np.deg2rad(179)+.03*t
        q=Rotation.from_euler('z',y[:,None]).as_quat()
        query=np.linspace(-.1,t[-1]+.1,200)
        got,meta=interpolate_pose_yaw(t,q,query)
        np.testing.assert_allclose(wrap(got-(np.deg2rad(179)+.03*query)),0,atol=1e-12)
        self.assertGreater(meta['heading_edge_extrapolated_frames'],0)
        self.assertFalse(meta['heading_covariance_available'])
        with self.assertRaisesRegex(ValueError,'extrapolation'):
            interpolate_pose_yaw(t,q,np.array([t[-1]+.31]))

    def test_invalid_quaternion_gaps_and_missing_source_fail(self):
        t=np.arange(0.,5.,.25);q=Rotation.from_euler('z',(t*.1)[:,None]).as_quat()
        bad=q.copy();bad[2]=0
        with self.assertRaises(ValueError):interpolate_pose_yaw(t,bad,t)
        with self.assertRaises(ValueError):interpolate_pose_yaw(t[::4],q[::4],t[1:3])
        with self.assertRaises(ValueError):interpolate_pose_yaw(t[::-1],q[::-1],t)

    def test_end_to_end_yaw_replacement_preserves_motion_xyz_imu_tilt_and_prefix(self):
        origin=1727234445000000000
        ns=lambda t:origin+np.rint(np.asarray(t)*1e9).astype(np.int64)
        gt=np.arange(0.,20.25,.25);it=np.arange(0.,20.001,.01);ft=np.arange(.1,19.9,.036)
        xyz=np.column_stack((10*gt,gt*.2,gt*0+5.))
        imu_rpy=np.column_stack((.1*np.sin(it),.05*np.cos(it),-.06+.02*it))
        imu=Rotation.from_euler('xyz',imu_rpy).as_quat()
        # GNSS tilt differs, but the tested branch must use only its yaw.
        given=Rotation.from_euler('xyz',np.column_stack((gt*0+.3,gt*0-.2,.02*gt))).as_quat()
        args=ns(gt),xyz,ns(it),imu,ns(ft)
        base=make_priors(*args,fusion='motion_rts')
        details={};got=make_priors(*args,fusion='pose_yaw',gnss_quaternions=given,details=details)
        np.testing.assert_array_equal(base[0],got[0])
        rpy=Rotation.from_quat(got[1]).as_euler('xyz')
        interpolated=Slerp(it,Rotation.from_quat(imu))(ft).as_euler('xyz')
        np.testing.assert_allclose(rpy[:,:2],interpolated[:,:2],atol=1e-12)
        np.testing.assert_allclose(wrap(rpy[:,2]-.02*ft),0,atol=1e-12)
        prefix=make_priors(*args[:4],ns(ft[:20]),fusion='pose_yaw',gnss_quaternions=given)
        np.testing.assert_allclose(prefix[1],got[1][:20],atol=1e-12)
        self.assertNotIn('bias_model_std_deg',details['diagnostic'])
        self.assertFalse(got[2]['sensor_fusion']['heading_source_verified'])
        with self.assertRaisesRegex(ValueError,'requires'):
            make_priors(*args,fusion='pose_yaw')


if __name__=='__main__':unittest.main()

import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prepare_surface_height import compatible_planes, smooth_references, fit_tilt_reference
from scipy.spatial.transform import Rotation
from analyze_height_modes import lower_plane_probe


class SurfaceHeightTest(unittest.TestCase):
    def test_near_short_plane_observable_but_far_strip_rejected(self):
        def plane(low,high):
            x,y=np.meshgrid(np.linspace(low,high,10),np.linspace(-1.2,1.2,8))
            return lower_plane_probe(np.column_stack([x.ravel(),y.ravel(),np.full(x.size,-1.)]),np.eye(3))
        near=plane(1,3);far=plane(12,14)
        self.assertLess(near['origin_extrapolation_proxy_m'],.3)
        self.assertGreater(far['origin_extrapolation_proxy_m'],.3)
        self.assertTrue(compatible_planes(near,near))
        self.assertFalse(compatible_planes(far,far))

    def test_bounded_tilt_recovers_known_orientation_with_exact_yaw(self):
        yaw=.6
        original=Rotation.from_euler('xyz',[.04,.05,yaw]).as_matrix()
        truth=Rotation.from_euler('xyz',[-.12,.02,yaw]).as_matrix()
        heading=Rotation.from_euler('z',yaw).as_matrix()
        normal=np.array([.01,.03,1]);normal/=np.linalg.norm(normal)
        def plane(r):
            n=heading.T @ r @ normal
            return {'forward_slope':-n[0]/n[2],'lateral_slope':-n[1]/n[2],'relative_z_at_source_origin_m':-1.}
        result,intercept,angle=fit_tilt_reference(plane(original),plane(truth),original)
        np.testing.assert_allclose(result,truth,atol=1e-10)
        self.assertLess(angle,15)
        self.assertAlmostEqual(Rotation.from_matrix(result).as_euler('xyz')[2],yaw,places=12)
        self.assertTrue(np.isfinite(intercept))
        with self.assertRaises(ValueError):
            fit_tilt_reference(plane(original),plane(Rotation.from_euler('xyz',[-.5,0,yaw]).as_matrix()),original)

    def test_known_sloped_plane_with_clutter(self):
        x,y=np.meshgrid(np.linspace(1,14,40),np.linspace(-4,4,25))
        z=-1-.03*x+.02*y
        points=np.column_stack([x.ravel(),y.ravel(),z.ravel()])
        cloud=np.vstack([points,points+np.array([0,0,2])])
        plane=lower_plane_probe(cloud,np.eye(3))
        self.assertEqual(plane['status'],'candidate')
        self.assertAlmostEqual(plane['relative_z_at_source_origin_m'],-1,places=6)
        self.assertTrue(compatible_planes(plane,plane))
        sparse=lower_plane_probe(cloud[:5],np.eye(3))
        self.assertFalse(compatible_planes(sparse,plane))

    def test_ambiguous_parallel_layers_rejected(self):
        x,y=np.meshgrid(np.linspace(1,14,40),np.linspace(-4,4,25))
        a=np.column_stack([x.ravel(),y.ravel(),np.full(x.size,-1.)])
        plane=lower_plane_probe(np.vstack([a,a-np.array([0,0,1])]),np.eye(3))
        self.assertGreaterEqual(plane['competing_plane_support_fraction'],.8)
        self.assertFalse(compatible_planes(plane,plane))

    def test_integer_time_interpolation_and_long_gap(self):
        stamps=np.array([1790000000000000000+i*100000000 for i in range(4)],dtype=np.int64)
        ref,good=smooth_references(stamps,[1,np.nan,1.2,1.3],half_window=0)
        np.testing.assert_allclose(ref,[1,1.1,1.2,1.3])
        self.assertEqual(good.tolist(),[True,False,True,True])
        with self.assertRaises(ValueError):
            smooth_references([1000000000,3000000000],[1,2])
        with self.assertRaises(ValueError):
            smooth_references([1000000000,1500000000,2000000000],[np.nan,1,2])

    def test_symmetric_median_suppresses_single_height_outlier(self):
        stamps=[1000000000+i*100000000 for i in range(5)]
        ref,_=smooth_references(stamps,[1,1,8,1,1])
        np.testing.assert_allclose(ref,np.ones(5))


if __name__=='__main__': unittest.main()

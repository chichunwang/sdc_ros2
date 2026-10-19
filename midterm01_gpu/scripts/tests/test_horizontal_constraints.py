from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from horizontal_constraints import relative_motion, horizontal_map_probe
from three_camera_calibration import SurfaceMap


class HorizontalConstraintsTests(unittest.TestCase):
    def synthetic(self,translation,noise=.1):
        rng=np.random.default_rng(17)
        k=np.array([[330.,0,224],[0,330,147],[0,0,1]])
        xyz=np.column_stack((rng.uniform(-4,4,800),rng.uniform(-2.7,2.7,800),rng.uniform(6,16,800)))
        rotation=Rotation.from_euler('y',.02).as_matrix()
        other=xyz@rotation.T+translation
        def project(p):return (p@k.T)[:,:2]/p[:,2,None]
        a,b=project(xyz),project(other)
        keep=(a[:,0]>3)&(a[:,0]<445)&(a[:,1]>3)&(a[:,1]<291)&(b[:,0]>3)&(b[:,0]<445)&(b[:,1]>3)&(b[:,1]<291)
        a,b=a[keep],b[keep]
        return a+rng.normal(0,noise,a.shape),b+rng.normal(0,noise,b.shape),k,rotation

    def test_known_rotation_and_translation_direction_with_noise(self):
        a,b,k,truth=self.synthetic([-.6,.01,.2])
        result=relative_motion(a,b,k)
        error=np.degrees(Rotation.from_matrix(np.array(result['rotation_camera2_from_camera1'])@truth.T).magnitude())
        self.assertLess(error,.2)
        self.assertTrue(result['relative_rotation_candidate'])
        self.assertFalse(result['metric_translation_observed'])
        self.assertIsNone(result['translation_scale_source'])

    def test_pure_rotation_has_no_translation_observability(self):
        a,b,k,_=self.synthetic([0,0,0],noise=.001)
        result=relative_motion(a,b,k)
        self.assertFalse(result['relative_rotation_candidate'])
        self.assertFalse(result['metric_translation_observed'])

    def test_holdout_changes_cannot_refit_the_training_rotation(self):
        a,b,k,_=self.synthetic([-.6,.01,.2])
        first=relative_motion(a,b,k)
        cell=np.floor(a/[448,294]*[4,3]).astype(int)
        heldout=(cell[:,0]+2*cell[:,1])%3==0
        changed=b.copy();changed[heldout]+=[5,-3]
        second=relative_motion(a,changed,k)
        np.testing.assert_array_equal(first['rotation_camera2_from_camera1'],second['rotation_camera2_from_camera1'])

    def test_ground_only_cannot_supply_horizontal_map_factor(self):
        x,y=np.meshgrid(np.linspace(-10,10,30),np.linspace(-10,10,30))
        points=np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size)))
        report,_=horizontal_map_probe(SurfaceMap(points),points,np.eye(4))
        self.assertEqual(report['unique_vertical_targets'],0)
        self.assertFalse(report['structurally_supported'])
        self.assertFalse(report['accepted_for_fusion'])


if __name__=='__main__':unittest.main()

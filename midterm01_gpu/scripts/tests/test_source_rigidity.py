from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from source_rigidity import (RigidityConfig, similarity_fit, apply_model, robust_model,
                             diagnose_matches, split_cells, render_features)


class SourceRigidityTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(39)
        self.a=self.rng.uniform([-4,-3,2],[4,3,10],(160,3))
        self.r=Rotation.from_euler('xyz',[.05,-.1,.2]).as_matrix()
        self.t=np.array([.3,-.7,1.])
        self.pixels=self.rng.uniform([1,1],[447,293],(160,2))

    def test_known_similarity_and_inverse(self):
        truth=(1.2,self.r,self.t);b=apply_model(self.a,truth)
        fit=similarity_fit(self.a,b)
        np.testing.assert_allclose(apply_model(self.a,fit),b,atol=1e-12)
        back=similarity_fit(b,self.a)
        self.assertAlmostEqual(back[0],1/1.2)
        self.assertGreater(np.linalg.det(fit[1]),.999999)

    def test_collinear_support_is_rejected(self):
        a=np.column_stack((np.arange(8),np.zeros((8,2))))
        with self.assertRaisesRegex(ValueError,'Collinear'):similarity_fit(a,a)

    def test_robust_motion_with_bad_rgb_matches(self):
        b=apply_model(self.a,(1.,self.r,self.t))+self.rng.normal(0,.005,self.a.shape)
        b[:20]=self.rng.normal(0,10,(20,3))
        fit,success,bound=robust_model(self.a,b,False)
        self.assertTrue(success);self.assertFalse(bound)
        self.assertLess(np.linalg.norm(fit[2]-self.t),.03)
        self.assertLess(Rotation.from_matrix(fit[1]@self.r.T).magnitude(),.005)

    def test_holdout_never_changes_fitted_models(self):
        b=apply_model(self.a,(1.08,self.r,self.t))
        first=diagnose_matches(self.a,b,self.pixels)
        changed=b.copy();changed[split_cells(self.pixels)]+=[3,1,-2]
        second=diagnose_matches(self.a,changed,self.pixels)
        for kind in ('rigid','similarity'):
            for key in ('scale_b_from_a','rotation_b_from_a','translation_b_from_a'):
                np.testing.assert_array_equal(first['models'][kind][key],second['models'][kind][key])
        self.assertGreater(second['models']['similarity']['holdout_error_m']['median'],3.)

    def test_scale_change_is_not_absorbed_as_rigid_pose(self):
        b=apply_model(self.a,(1.2,self.r,self.t))
        result=diagnose_matches(self.a,b,self.pixels)
        self.assertGreater(result['models']['rigid']['holdout_error_m']['median'],.4)
        self.assertLess(result['models']['similarity']['holdout_error_m']['median'],1e-8)
        self.assertFalse(result['accepted_for_fusion'])

    def test_similarity_does_not_explain_nonrigid_depth(self):
        b=apply_model(self.a,(1.,self.r,self.t))
        b[:,2]+=.04*(self.a[:,0]**2+self.a[:,1]**2)
        result=diagnose_matches(self.a,b,self.pixels)
        self.assertGreater(result['models']['similarity']['holdout_error_m']['median'],.1)

    def test_no_forward_points_returns_no_features(self):
        result=render_features(np.array([[-1,0,0],[-2,1,1.]]),np.zeros((2,3),np.uint8))
        self.assertEqual(len(result['xyz']),0)
        self.assertIsNone(result['descriptors'])

    def test_small_split_fails_closed(self):
        result=diagnose_matches(self.a[:10],self.a[:10],self.pixels[:10])
        self.assertEqual(result['models'],{})
        self.assertFalse(result['accepted_for_fusion'])

    def test_rendered_features_use_original_visible_vertices(self):
        yy,xx=np.indices((294,448));depth=np.full(xx.size,4.)
        front=np.column_stack((depth,-(xx.ravel()-224)/180*depth,-(yy.ravel()-147)/180*depth))
        xyz=np.concatenate((front,front*2))
        rgb=self.rng.integers(0,256,(len(xyz),3),dtype=np.uint8)
        feature=render_features(xyz,rgb)
        self.assertGreater(len(feature['xyz']),20)
        self.assertTrue(np.all(feature['vertex_indices']<len(front)))
        np.testing.assert_array_equal(feature['xyz'],xyz[feature['vertex_indices']])


class CommonScaleTests(unittest.TestCase):
    def test_recover_shared_scale_from_motion(self):
        from source_scale_lab import scale_terms,fit_common_scale
        rng=np.random.default_rng(9);a=rng.normal(size=(120,3))
        pa=np.c_[np.eye(3),np.zeros(3)]
        pb=np.c_[Rotation.from_euler('z',.1).as_matrix(),[1.,.1,0.]]
        alpha=2.5
        r=pb[:,:3].T;translation=-pb[:,:3].T@pb[:,3]
        b=a@r.T+translation/alpha
        d,t=scale_terms(a,b,pa,pb)
        result=fit_common_scale([(d,t)])
        self.assertAlmostEqual(result['multiplier'],alpha,places=7)
        self.assertTrue(result['success']);self.assertFalse(result['at_bound'])

    def test_global_map_origin_cannot_change_scale(self):
        from source_scale_lab import scale_terms
        rng=np.random.default_rng(1);a=rng.normal(size=(40,3));b=a-.3
        pa=np.c_[np.eye(3),[1.,2.,3.]];pb=np.c_[np.eye(3),[2.,3.,4.]]
        first=scale_terms(a,b,pa,pb)
        pa[:,3]+=[100,-50,7];pb[:,3]+=[100,-50,7]
        second=scale_terms(a,b,pa,pb)
        for x,y in zip(first,second):np.testing.assert_allclose(x,y,atol=1e-12)

    def test_zero_information_is_rejected(self):
        from source_scale_lab import fit_common_scale
        with self.assertRaisesRegex(ValueError,'sensitivity'):
            fit_common_scale([(np.zeros((8,3)),np.zeros((8,3)))])


if __name__=='__main__':unittest.main()

"""Observability, holdout separation, map Jacobians and raw-GNSS factor accounting."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from horizontal_surfel import (SurfelConfig, FineSurfelMap, fit_horizontal_factor,
    horizontal_information, plane_residual_and_jacobian, heldout_patches, select_horizontal_pairs)
from surfel_joint import (SurfelJointProblem, make_problem, interpolate_states,
                          temporal_accept, discount_shared_patches)
from refine_horizontal_joint import select_groups, validate_result


class HorizontalSurfelTests(unittest.TestCase):
    def pool(self):
        rng = np.random.default_rng(196)
        points = rng.uniform([-14,-12,-1],[18,15,4],(120,3))
        normals = np.tile(np.eye(3)[:2],(60,1))
        delta = np.array([.1,-.08,.03,.2])
        cfg = SurfelConfig()
        targets = points@Rotation.from_euler('z',delta[3]*cfg.yaw_scale_rad).as_matrix().T+delta[:3]
        ids = np.column_stack((np.arange(120),np.arange(120)%11,np.arange(120)%7)).astype(np.int64)
        return {'rotated':points,'targets':targets,'normals':normals,'quality':np.ones(120),
            'patch_ids':ids,'normal_bins':np.arange(120)%2,'holdout':heldout_patches(ids),
            'raw_points':120,'nonhorizontal_before_dedup':120}, delta

    def test_plane_jacobian_and_known_walls(self):
        pool,truth = self.pool(); cfg = SurfelConfig(); delta = np.array([.05,-.1,.1,.25])
        residual,jac = plane_residual_and_jacobian(delta,pool,cfg)
        for column in range(4):
            eps = np.eye(4)[column]*1e-6
            finite = (plane_residual_and_jacobian(delta+eps,pool,cfg)[0]-plane_residual_and_jacobian(delta-eps,pool,cfg)[0])/2e-6
            np.testing.assert_allclose(jac[:,column],finite,atol=1e-8)
        pose = np.column_stack((np.eye(3),[2000,-3000,10]))
        report,factor = fit_horizontal_factor(pool,pose,cfg)
        self.assertTrue(report['accepted'],report)
        np.testing.assert_allclose(factor['delta_xy_scaled_yaw'],truth[[0,1,3]],atol=1e-6)
        # Held-out points can reject the result, but cannot change fitted delta.
        modified = {k:v.copy() if isinstance(v,np.ndarray) else v for k,v in pool.items()}
        modified['targets'][pool['holdout']] += [1.,1.,0.]
        failed,_ = fit_horizontal_factor(modified,pose,cfg)
        self.assertFalse(failed['accepted'])
        np.testing.assert_array_equal(report['delta_xyz_scaled_yaw'],failed['delta_xyz_scaled_yaw'])

    def test_single_plane_cannot_create_xy_information_after_free_z(self):
        cfg = SurfelConfig(); pool,_ = self.pool()
        for normal in ([0.,0.,1.],[.3,.2,np.sqrt(.87)]):
            pool['normals'][:] = normal
            residual,j = plane_residual_and_jacobian(np.zeros(4),pool,cfg)
            values,vectors,supported,projected,_ = horizontal_information(j,residual,pool['quality'],cfg)
            np.testing.assert_allclose(projected[:,:2],0.,atol=1e-12)
            self.assertLessEqual(supported.sum(),1)
        pose = np.column_stack((np.eye(3),[0,0,0]))
        pool['normals'][:]=[0,0,1]
        report,_ = fit_horizontal_factor(pool,pose,cfg)
        json.dumps(report,allow_nan=False) # rank-zero rejection is still a valid report
        self.assertFalse(report['accepted'])

    def test_two_scale_plane_quality_and_line_rejection(self):
        a,b = np.meshgrid(np.arange(-3,3.01,.3),np.arange(-3,3.01,.3))
        points = np.column_stack((a.ravel(),b.ravel(),a.ravel()*0))
        scene = FineSurfelMap(points)
        index = np.array([np.argmin(np.linalg.norm(points,axis=1))])
        normals,quality = scene.surfaces(index)
        self.assertGreater(quality[0],.2)
        self.assertGreater(abs(normals[0,2]),.999)
        line = np.column_stack((np.arange(100)*.1,np.zeros((100,2))))
        scene = FineSurfelMap(line); _,quality = scene.surfaces(np.array([50]))
        self.assertEqual(quality[0],0.)

    def test_spatial_and_normal_balancing_does_not_count_duplicate_pixels(self):
        a,b=np.meshgrid(np.arange(-8.,8.01,.3),np.arange(-2.,4.01,.3))
        points=np.column_stack((a.ravel()*0+10,a.ravel(),b.ravel()))
        scene=FineSurfelMap(points); pose=np.column_stack((np.eye(3),[0,0,0]))
        one=select_horizontal_pairs(scene,points,pose)
        repeated=select_horizontal_pairs(scene,np.tile(points,(5,1)),pose)
        np.testing.assert_array_equal(one['patch_ids'],repeated['patch_ids'])
        np.testing.assert_array_equal(one['quality'],repeated['quality'])
        keys=np.column_stack((one['patch_ids'],one['normal_bins']))
        self.assertEqual(len(np.unique(keys,axis=0)),len(keys))
        self.assertLessEqual(len(keys),4*scene.config.max_points_per_normal_bin)

    def test_temporal_gate_and_patch_reuse_discount(self):
        _,factor = fit_horizontal_factor(self.pool()[0],np.column_stack((np.eye(3),[0,0,0])))
        self.assertTrue(temporal_accept([{}]*3,[factor]*3))
        broken = {**factor,'delta_xy_scaled_yaw':[1,1,1]}
        self.assertFalse(temporal_accept([{}]*3,[factor,broken,factor]))
        self.assertFalse(temporal_accept([{}]*3,[factor,None,factor]))
        discounted = discount_shared_patches([factor,factor])
        self.assertEqual(discounted[0]['maximum_patch_reuse'],2)
        np.testing.assert_allclose(np.asarray(discounted[0]['sqrt_information']),np.asarray(factor['sqrt_information'])/np.sqrt(2))


class JointGraphTests(unittest.TestCase):
    def data(self):
        ns = 1_750_000_000_000_000_000
        gt = np.arange(0.,6.01,.25); it = np.arange(-.1,6.11,.01)
        truth = np.column_stack((gt*4,gt*0))
        raw = {'gnss_bag_ns':ns+np.rint(gt*1e9).astype(np.int64),
            'gnss_xyz':np.column_stack((truth+[.7,-.4],gt*0)),
            'imu_bag_ns':ns+np.rint(it*1e9).astype(np.int64),
            'imu_quat':Rotation.from_euler('z',(it*0)[:,None]).as_quat(),
            'angular_velocity':np.column_stack((it*0,it*0,it*0+.002))}
        factor = {'timestamp_ns':ns+3_123_456_789, 'reference_xy_yaw':[4*3.123456789,0.,0.],
            'delta_xy_scaled_yaw':[0.,0.,0.], 'sqrt_information':(np.eye(3)*30).tolist(),
            'yaw_scale_rad':np.deg2rad(3.), 'train_patch_ids':[[1,2,3]]}
        return raw,factor

    def test_map_jacobian_and_gnss_once_at_exact_map_time(self):
        raw,factor = self.data()
        problem,origin = make_problem(raw,[factor],raw['gnss_bag_ns'])
        self.assertEqual(problem.training.sum(),len(raw['gnss_bag_ns']))
        self.assertEqual(problem.n,len(raw['gnss_bag_ns'])+1)
        self.assertEqual(int(round(problem.gt[problem.factors[0]['node']]*1e9))+origin,factor['timestamp_ns'])
        self.assertTrue(np.all(problem.lateral_weight==0))
        x=problem.initial.ravel(); jac=problem.evaluate(x,True).toarray()
        node=problem.factors[0]['node']
        for column in (0,4,5,6*node,6*node+1,6*node+4,6*node+5):
            eps=np.eye(1,len(x),column).ravel()*1e-6
            finite=(problem.evaluate(x+eps)-problem.evaluate(x-eps))/2e-6
            np.testing.assert_allclose(jac[:,column],finite,atol=1e-6)

    def test_map_anchor_reduces_common_gnss_bias_and_exact_interpolation(self):
        raw,factor=self.data(); problem,_=make_problem(raw,[factor],raw['gnss_bag_ns'])
        state,meta=problem.solve(); self.assertTrue(meta['success'],meta)
        node=problem.factors[0]['node']
        self.assertLess(np.linalg.norm(state[node,:2]-factor['reference_xy_yaw'][:2]),.1)
        xy,yaw=interpolate_states(problem,state,problem.gt)
        np.testing.assert_allclose(xy,state[:,:2],atol=1e-11)
        np.testing.assert_allclose(yaw,state[:,4],atol=1e-11)
        self.assertIsNone(make_problem(raw,[],raw['gnss_bag_ns']))

    def test_map_does_not_correct_unmeasured_horizontal_direction(self):
        raw,factor=self.data()
        factor['sqrt_information']=[[30,0,0],[0,0,30]] # X and yaw only
        problem,_=make_problem(raw,[factor],raw['gnss_bag_ns'])
        state,meta=problem.solve(); self.assertTrue(meta['success'],meta)
        np.testing.assert_allclose(state[:,1],-.4,atol=1e-10)
        self.assertLess(abs(state[problem.factors[0]['node'],0]-factor['reference_xy_yaw'][0]),.1)

    def test_short_group_edges_and_pose_integrity(self):
        ns=1_750_000_000_000_000_000
        self.assertEqual(select_groups(ns+np.arange(20)*35_000_000),[])
        groups=select_groups(ns+np.arange(48)*35_000_000)
        self.assertEqual(len(groups),1); self.assertEqual(len(set(groups[0])),3)
        poses=np.tile(np.column_stack((np.eye(3),[1,2,3])),(48,1,1))
        validate_result(poses,poses,ns+np.arange(48)*35_000_000,[{}]*48)
        changed=poses.copy(); changed[0,2,3]+=.01
        with self.assertRaisesRegex(ValueError,'preserve'):
            validate_result(changed,poses,ns+np.arange(48)*35_000_000,[{}]*48)


if __name__ == '__main__': unittest.main()

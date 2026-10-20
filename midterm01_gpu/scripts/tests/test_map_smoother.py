"""Map-only scale observability, bounded anchors, sparse Jacobians and failures."""
import sys
import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from map_scale_icp import ScaleICPConfig, LayeredScaleMap, register_frame, linear_terms
from slow_map_graph import SlowBiasProblem, SlowMapConfig, bounded_vector, slow_anchors, smooth_trajectory
from run_map_smoother import arguments


def room():
    a, b = np.meshgrid(np.linspace(-6, 6, 35), np.linspace(0, 5, 20))
    front = np.column_stack([np.full(a.size, 12.), a.ravel(), b.ravel()])
    a, b = np.meshgrid(np.linspace(-4, 12, 43), np.linspace(0, 5, 20))
    walls = [np.column_stack([a.ravel(), np.full(a.size, y), b.ravel()]) for y in (-6., 6.)]
    a, b = np.meshgrid(np.linspace(-4, 12, 43), np.linspace(-6, 6, 35))
    ground = np.column_stack([a.ravel(), b.ravel(), np.zeros(a.size)])
    return np.vstack([front, *walls, ground]), ground


class MapTests(unittest.TestCase):
    def test_similarity_and_ground_rejection(self):
        points, ground = room();truth = np.eye(4);truth[2, 3] = 1.5
        prior = truth.copy();prior[:3, 3] += [.35, -.25, .12]
        prior[:3, :3] = Rotation.from_euler('xyz', [1., -.8, 1.5], degrees=True).as_matrix()
        raw = (points-truth[:3, 3])/8.7
        config = ScaleICPConfig()
        report = register_frame(LayeredScaleMap(points, config), raw, prior, config)
        self.assertTrue(report['accepted'], report)
        self.assertLess(abs(report['scale']-8.7), .08)
        self.assertLess(np.linalg.norm(np.array(report['pose'])[:3, 3]-truth[:3, 3]), .08)
        bad = register_frame(LayeredScaleMap(ground, config), (ground-truth[:3, 3])/8.7, prior, config)
        self.assertFalse(bad['accepted'])
        self.assertIn('horizontal_observability', bad['reasons'])

    def test_sim3_jacobian(self):
        rng = np.random.default_rng(3);q=rng.normal(size=(25, 3));n=rng.normal(size=(25, 3))
        n /= np.linalg.norm(n, axis=1)[:, None]
        target=rng.normal(size=(25, 3));translation=np.array([.3, -.4, 2.])
        pool={'normal':n,'rotated':q,'world':q+translation,'target':target}
        residual,jac=linear_terms(pool)
        def function(x):
            world=np.exp(x[6])*q@Rotation.from_rotvec(x[3:6]).as_matrix().T+translation+x[:3]
            return np.sum(n*(world-target),axis=1)
        numerical=approx_derivative(function,np.zeros(7),method='3-point')
        np.testing.assert_allclose(jac,numerical,atol=1e-8)


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.gt=np.linspace(0,3,18);self.it=np.linspace(0,3,181)
        self.xy=np.column_stack([3*self.gt,np.zeros(len(self.gt))])
        self.quat=np.tile([0.,0.,0.,1.],(len(self.it),1));self.gyro=np.zeros((len(self.it),3))

    def test_sparse_jacobian_including_robust_outliers(self):
        p=SlowBiasProblem(self.gt,self.xy,self.it,self.quat,self.gyro,
            [{'time_s':.83,'bias_xy':[.6,-.3],'yaw':.1}])
        x=p.initial.copy();x[:,6:8]=[.3,-.15];x[:,4]=.08;x[:,5]=.003
        x[4,:2]+=[1.2,-1.5]  # exercise GNSS Huber derivatives too
        exact=p.evaluate(x.ravel(),True).toarray()
        numerical=approx_derivative(p.evaluate,x.ravel(),method='3-point')
        np.testing.assert_allclose(exact,numerical,atol=2e-6,rtol=2e-6)

    def test_bias_sign_and_solve(self):
        p=SlowBiasProblem(self.gt,self.xy+np.array([.8,0]),self.it,self.quat,self.gyro,
            [{'time_s':1.5,'bias_xy':[.8,0.],'yaw':0.}])
        state, report=p.solve()
        self.assertTrue(report['success'])
        self.assertGreater(np.median(state[:,6]),.6)
        self.assertLess(np.median(abs(state[:,0]-self.xy[:,0])),.2)

    def test_bounded_map_loss(self):
        small,_=bounded_vector([1.,0.],3.)
        huge,jac=bounded_vector([1e6,0.],3.)
        self.assertLess(np.linalg.norm(huge),3.)
        self.assertLess(jac[0,0],1e-12)
        self.assertGreater(np.linalg.norm(small),.9)

    def test_slow_blocks_reject_jump_and_preserve_circular_yaw(self):
        t=np.arange(0,26,.1);gnss=np.column_stack([2*t,t*0])
        xy=gnss+[.8,-.2];yaw=np.full(len(t),-np.pi+.01);ref=np.full(len(t),np.pi-.01)
        good=np.ones(len(t),bool);xy[60]+=[5.,0]
        anchors,mask,_=slow_anchors(t,xy,yaw,gnss,ref,good)
        self.assertFalse(mask[60])
        self.assertEqual(len(anchors),3)
        np.testing.assert_allclose(anchors[0]['bias_xy'],[-.8,.2])
        self.assertLess(abs((anchors[0]['yaw']-ref[0])-.02),1e-8)

    def test_no_map_retains_prior_and_no_graph(self):
        ns0=np.int64(1727241181291090775)
        stamps=ns0+(self.gt*1e9).astype(np.int64)
        raw={'gnss_bag_ns':stamps,'imu_bag_ns':ns0+(self.it*1e9).astype(np.int64),
             'gnss_xyz':np.column_stack([self.xy,np.zeros(len(self.gt))]),
             'imu_quat':self.quat,'angular_velocity':self.gyro}
        prior=np.tile(np.eye(4),(len(stamps),1,1));prior[:,:2,3]=self.xy
        obs=[{'accepted':False,'pose':p.tolist()} for p in prior]
        final,report,states=smooth_trajectory(raw,stamps,prior,obs)
        np.testing.assert_array_equal(final,prior)
        np.testing.assert_array_equal(states['frame_time_ns'],stamps)
        self.assertFalse(report['joint_estimation_executed'])

    def test_accepted_map_graph_publishes_proper_poses(self):
        ns0=np.int64(1727241181291090775)
        stamps=ns0+(self.gt*1e9).astype(np.int64)
        raw={'gnss_bag_ns':stamps,'imu_bag_ns':ns0+(self.it*1e9).astype(np.int64),
             'gnss_xyz':np.column_stack([self.xy+[.8,0.],np.zeros(len(self.gt))]),
             'imu_quat':self.quat,'angular_velocity':self.gyro}
        prior=np.tile(np.eye(4),(len(stamps),1,1));prior[:,:2,3]=self.xy+[.8,0.]
        map_poses=prior.copy();map_poses[:,0,3]-=.8
        obs=[{'accepted':True,'pose':p.tolist()} for p in map_poses]
        final,report,states=smooth_trajectory(raw,stamps,prior,obs)
        self.assertTrue(report['applied']);self.assertTrue(report['joint_estimation_executed'])
        self.assertLess(np.median(abs(final[:,0,3]-self.xy[:,0])),.2)
        np.testing.assert_allclose(final[:,:3,:3].transpose(0,2,1)@final[:,:3,:3],np.tile(np.eye(3),(len(final),1,1)),atol=1e-8)


class CLITests(unittest.TestCase):
    def test_explicit_default_frames_still_conflicts_with_full(self):
        with patch.object(sys, 'argv', ['runner', '--full', '--frames', '48']), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:arguments()
        self.assertEqual(error.exception.code,2)

    def test_shared_full_parser_and_nonzero_start_rejection(self):
        for track in (1,2):
            with patch.object(sys, 'argv', ['runner','--track',str(track),'--full']):
                args=arguments();self.assertTrue(args.full);self.assertEqual(args.frames,48)
        with patch.object(sys, 'argv', ['runner','--full','--start-frame','1']), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:arguments()
        self.assertEqual(error.exception.code,2)


if __name__=='__main__':unittest.main()

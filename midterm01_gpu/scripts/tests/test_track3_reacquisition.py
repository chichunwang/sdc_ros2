"""Regressions for source strata and rejection-to-reacquisition behavior."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np

from track3_registration import Geometry, RegistrationConfig, block_validation, MapRegistrar
from track3_structural_weights import structural_weights
from track3_map_only import MapOnlyTracker


def pose(x):
    value=np.eye(4);value[0,3]=x
    return value


class StructuralSampling(unittest.TestCase):
    def test_heldout_never_changes_training_strata(self):
        rng=np.random.default_rng(173)
        floor=np.column_stack([rng.uniform(-5,5,700),rng.uniform(-5,5,700),np.zeros(700)])
        wall=np.column_stack([np.full(700,5.),rng.uniform(-5,5,700),rng.uniform(0,5,700)])
        other=np.column_stack([rng.uniform(-5,5,700),np.full(700,5.),rng.uniform(0,5,700)])
        raw=np.vstack([floor,wall,other])
        g=Geometry('cpu_reference')
        wa,a=structural_weights(raw,1.,np.eye(3),g)
        extra=rng.uniform(100,200,(100,3))
        extra=extra[block_validation(extra)]
        wb,b=structural_weights(np.vstack([raw,extra]),1.,np.eye(3),g)
        np.testing.assert_array_equal(wa,wb[:len(raw)])
        self.assertEqual(a['counts'],b['counts'])
        self.assertEqual(a['source_training_sha256'],b['source_training_sha256'])
        active=np.array(a['counts'])>0
        np.testing.assert_allclose(np.array(a['weight_fractions'])[active],1/active.sum(),atol=1e-12)
        self.assertGreaterEqual(active.sum(),3)

    def test_weighted_known_transform_remains_identifiable(self):
        rng=np.random.default_rng(17)
        points=np.vstack([rng.uniform([-3,-3,0],[3,3,.01],(800,3)),
                          rng.uniform([3,-3,0],[3.01,3,4],(400,3)),
                          rng.uniform([-3,3,0],[3,3.01,4],(400,3)),
                          rng.uniform([-3,-3,0],[-2.99,3,4],(400,3))])
        scene=MapRegistrar(points,backend='cpu_reference')
        weights,_=structural_weights(points,1.,np.eye(3),scene.geometry)
        initial=pose(.2)
        fit=scene.fit(points,initial,1.,training_weights=weights)
        self.assertTrue(fit['accepted'],fit['reasons'])
        self.assertLess(np.linalg.norm(np.array(fit['pose'])[:3,3]),.04)
        self.assertLess(abs(fit['scale']-1),.01)


class RecoveryTransitions(unittest.TestCase):
    def tracker(self):
        pair=dict(pose=pose(1).tolist(),scale=3.,accepted=True,reasons=[])
        scene=SimpleNamespace(config=RegistrationConfig(),geometry=SimpleNamespace(backend='cpu_reference'),
            fit=lambda *a,**kw:dict(pose=pose(6).tolist(),scale=3.))
        tracker=MapOnlyTracker(scene,pose(0),3.)
        tracker.pose=pose(1);tracker.previous_raw=np.ones((12,3));tracker.previous_ns=1_100_000_000
        tracker.last_map_ns=1_000_000_000;tracker.last_map_pose=pose(0)
        tracker.consecutive_map_rejections=1;tracker.state='relocalizing'
        relative=SimpleNamespace(fit=lambda *a,**kw:pair.copy())
        return tracker,relative

    def test_recovery_separates_anchor_motion_from_prior_correction(self):
        tracker,relative=self.tracker()
        chosen=dict(pose=pose(6).tolist(),scale=3.)
        with patch('track3_map_only.structural_weights',return_value=(np.ones(12),{})), \
             patch('track3_map_only.MapRegistrar',return_value=relative), \
             patch.object(tracker,'_ray',side_effect=lambda raw,fit:fit), \
             patch.object(tracker,'_choose_and_check',return_value=(chosen,dict(accepted=True,reasons=[]))):
            result=tracker.step(np.ones((12,3)),1_200_000_000)
        # 5m correction exceeds 40m/s * one 0.1s frame, but 6m physical
        # displacement is supported by 40m/s * 0.2s since the map anchor.
        self.assertTrue(result['accepted'])
        self.assertTrue(result['reacquisition_succeeded'])
        self.assertEqual(result['tracking_state'],'tracking')
        self.assertEqual(result['map_motion']['translation_limit_m'],8.)
        self.assertEqual(result['map_motion']['correction_from_pair_prior_m'],5.)
        np.testing.assert_array_equal(tracker.last_map_pose,pose(6))

    def test_anchor_gate_still_rejects_impossible_physical_motion(self):
        tracker,relative=self.tracker()
        chosen=dict(pose=pose(9).tolist(),scale=3.)
        with patch('track3_map_only.structural_weights',return_value=(np.ones(12),{})), \
             patch('track3_map_only.MapRegistrar',return_value=relative), \
             patch.object(tracker,'_ray',side_effect=lambda raw,fit:fit), \
             patch.object(tracker,'_choose_and_check',return_value=(chosen,dict(accepted=True,reasons=[]))):
            result=tracker.step(np.ones((12,3)),1_200_000_000)
        self.assertFalse(result['accepted'])
        self.assertIn('map_motion_gate',result['map_decision']['reasons'])
        np.testing.assert_array_equal(tracker.last_map_pose,pose(0))
        np.testing.assert_array_equal(result['pose'],pose(1))

    def test_rejection_enters_reacquisition_domain_without_rewriting_anchor(self):
        tracker,relative=self.tracker()
        domains=[]
        def fit(*a,**kw):
            domains.append((tracker.scene.config.translation_domain_m,tracker.scene.config.rotation_domain_deg,
                            kw['use_ndt']))
            return dict(pose=pose(1).tolist(),scale=3.)
        tracker.scene.fit=fit
        with patch('track3_map_only.structural_weights',return_value=(np.ones(12),{})), \
             patch('track3_map_only.MapRegistrar',return_value=relative), \
             patch.object(tracker,'_ray',side_effect=lambda raw,fit:fit), \
             patch.object(tracker,'_choose_and_check',return_value=(None,dict(accepted=False,reasons=['ambiguous']))):
            result=tracker.step(np.ones((12,3)),1_200_000_000)
        self.assertEqual(domains,[(8.,30.,True)]*3)
        self.assertEqual(result['tracking_state'],'relocalizing')
        self.assertTrue(result['reacquisition_attempted'])
        self.assertFalse(result['reacquisition_succeeded'])
        np.testing.assert_array_equal(tracker.last_map_pose,pose(0))


if __name__=='__main__':unittest.main()

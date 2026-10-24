"""Execution backend for the shared causal tracker; no separate gate/state logic."""
from track3_map_only import MapOnlyTracker as ReferenceTracker
from track3_registration import RegistrationConfig, block_validation
from track3_registration_fast import FastMapRegistrar, parallel_map_candidates
from track3_visibility_fast import ray_depth_diagnostic

METHOD = 'track3_forward_gpu_structural_ndt_icp_reacquisition_v2_fast'


class MapOnlyTracker(ReferenceTracker):
    def _new_pair_registrar(self, world):
        return FastMapRegistrar(world, config=RegistrationConfig(), geometry=self.scene.geometry,
                                backend=self.scene.geometry.backend)

    def _ray(self, raw, result):
        result['ray_depth']=ray_depth_diagnostic(self.scene,raw,result['pose'],result['scale'],~block_validation(raw))
        return result

    def _map_candidates(self, raw, prior, config, weights):
        return parallel_map_candidates(self.scene,raw,prior,self.scale,config,weights)

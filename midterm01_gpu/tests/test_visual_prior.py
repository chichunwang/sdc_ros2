import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from track3.visual_prior import relative_pose_from_pnp,RDF_TO_FLU
from track3.localization import PoseState


def test_pnp_optical_to_flu_motion_and_world_composition():
    rng=np.random.default_rng(34)
    previous=rng.uniform([4,-3,-1],[20,3,2],(100,3))
    relative=np.eye(4);relative[:3,:3]=Rotation.from_rotvec([.01,-.005,.02]).as_matrix()
    relative[:3,3]=[.15,.01,-.005]
    current=(previous-relative[:3,3])@relative[:3,:3]
    optical=current@RDF_TO_FLU
    K=np.array([[547.,0,752.],[0,547.,458.],[0,0,1.]])
    pixels=(optical@K.T);pixels=pixels[:,:2]/pixels[:,2,None]
    ok,r,t=cv2.solvePnP(previous,pixels,K,None,flags=cv2.SOLVEPNP_EPNP)
    assert ok
    recovered=relative_pose_from_pnp(r,t)
    np.testing.assert_allclose(recovered,relative,atol=1e-7)
    world=np.eye(4);world[:3,:3]=Rotation.from_rotvec([0,0,1.2]).as_matrix();world[:3,3]=[660,-220,10]
    state=PoseState(world,8.,prediction='visual_pnp')
    state.predict(1727239790258245070)
    state.commit(dict(T=world,scale=8.,accepted=True,reason='accepted'))
    prior=state.predict(1727239790298245070,motion=recovered)
    np.testing.assert_allclose(prior,world@relative,atol=1e-7)
    state.commit(dict(T=np.eye(4),scale=3.,accepted=False,reason='not_converged'))
    np.testing.assert_allclose(state.T,prior,atol=1e-7)
    assert state.scale==8.


def test_cuda_ego_mask_excludes_own_vehicle_and_outside_fov():
    from track3.visual_prior import filter_by_camera_mask
    K=np.array([[2.,0,2],[0,2.,2],[0,0,1.]])
    mask=np.full((5,5),255,dtype='uint8');mask[2,2]=0
    points=np.array([[1,0,0],[1,-.5,0],[-1,0,0],[1,-4,0]],dtype=float)
    kept,info=filter_by_camera_mask(points,K,mask)
    np.testing.assert_array_equal(kept,points[1:2])
    assert info['removed_points']==3

import numpy as np
from track3.initialization import fit_plane,transform4
from track3.gpu import cp


def test_cuda_ground_fit_and_four_dof_jacobian():
    rng=np.random.default_rng(8)
    p=rng.uniform([-4,-3,-.3],[6,3,-.3],(1500,3));p[:,2]+=rng.normal(0,.001,len(p))
    n,d,info=fit_plane(np.r_[p,rng.uniform([-4,-3,0],[6,3,2],(300,3))],.01)
    np.testing.assert_allclose(n,[0,0,1],atol=.001)
    assert abs(d-.3)<.001 and info['inliers']>=1490
    u=cp.asarray(p[:20]);x=np.array([.1,.2,.3,np.log(7.)])
    q,J=transform4(u,x)
    for j in range(4):
        dx=np.eye(4)[j]*1e-6
        plus,_=transform4(u,x+dx);minus,_=transform4(u,x-dx)
        np.testing.assert_allclose(cp.asnumpy(J[:,:,j]),cp.asnumpy((plus-minus)/2e-6),atol=1e-7)


def test_scale_is_recovered_from_non_ground_structure():
    import json
    from pathlib import Path
    from scipy.spatial.transform import Rotation
    from track3.initialization import GroundInitializer
    rng=np.random.default_rng(94)
    ground=rng.uniform([-12,-8,0],[14,8,0],(7000,3))
    wall=rng.uniform([9,-6,.2],[9,6,5],(3500,3))
    side=rng.uniform([2,5,.2],[10,5,3],(2500,3))
    m=np.r_[ground,wall,side]
    truth=np.eye(4);truth[:3,3]=[.5,-.3,2.]
    truth[:3,:3]=Rotation.from_euler('xyz',[.04,-.03,.2]).as_matrix()
    source=(m-truth[:3,3])@truth[:3,:3]/7.
    source=source[(source[:,0]>.2)&(np.abs(source[:,1]/source[:,0])<1.3)]
    prior=truth.copy();prior[:3,3]+=[.3,-.2,.3]
    prior[:3,:3]=Rotation.from_euler('xyz',[.2,-.18,.12]).as_matrix()
    cfg=json.loads((Path(__file__).resolve().parents[1]/'config/track3.json').read_text())['registration']['initializer']
    result=GroundInitializer(m,cfg).align(source,prior,3.)
    assert abs(result['scale']-7.)<.2
    assert np.linalg.norm(result['T'][:3,3]-truth[:3,3])<.2
    assert Rotation.from_matrix(result['T'][:3,:3]@truth[:3,:3].T).magnitude()<.04

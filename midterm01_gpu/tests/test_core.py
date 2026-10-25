import json
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from track3.geometry import apply_delta, valid_pose, yaw, rz, wrap
from track3.io import frames, load_ply, load_map
from track3.gpu import cp, keys, voxel_mean, VoxelModel, jacobian, linearize, Registration

CFG=json.loads((Path(__file__).resolve().parents[1]/'config/track3.json').read_text())
# Generated mathematical fixtures have no road scene or image stream.
CFG['prediction']='previous_pose'
CFG['camera_mask']=False
CFG['motion_constraints']=False
CFG['registration']['preprocessing']['enabled']=False
CFG['registration']['initializer']['enabled']=False


def cloud():
    r=np.random.default_rng(891)
    pieces=[]
    # Three intersecting planes with finite edges, in metric coordinates.
    for axis,value in [(0,8.),(1,6.),(2,-2.)]:
        p=r.uniform([-4,-4,-2],[8,6,5],size=(6000,3));p[:,axis]=value+r.normal(0,.01,len(p));pieces.append(p)
    return np.concatenate(pieces)


def test_pose_jacobian():
    p=np.random.default_rng(23).normal(size=(17,3));T=np.eye(4)
    T[:3,:3]=Rotation.from_rotvec([.2,-.1,.3]).as_matrix();T[:3,3]=[660,-220,10]
    scale=2.4;v=p@T[:3,:3].T*scale
    J=cp.asnumpy(jacobian(cp.asarray(v)))
    for j in range(7):
        d=np.zeros(7);d[j]=1e-6
        plus,s=apply_delta(T,scale,d)
        minus,sm=apply_delta(T,scale,-d)
        numerical=((p@plus[:3,:3].T*s+plus[:3,3])-(p@minus[:3,:3].T*sm+minus[:3,3]))/2e-6
        np.testing.assert_allclose(J[:,:,j],numerical,atol=1e-7)


def test_gpu_voxel_and_keys():
    pts=np.array([[-1.,2,3],[-1.01,2,3],[4,2,1],[4.01,2,1]])
    got,w=voxel_mean(pts,.5)
    coords=np.floor(pts/.5);unique=np.unique(coords,axis=0)
    expected=np.array([pts[(coords==cell).all(1)].mean(0) for cell in unique])
    np.testing.assert_allclose(cp.asnumpy(got),expected)
    assert int(w.sum())==4
    with pytest.raises(ValueError):keys(cp.array([[2**21,0,0]]))


def test_gpu_hessian_and_registration():
    m=cloud();T=np.eye(4);T[:3,3]=[.18,-.15,.09]
    T[:3,:3]=Rotation.from_rotvec([.015,-.02,.025]).as_matrix()
    source=m/3
    model=VoxelModel(m,1.)
    data=linearize(cp.asarray(source),T,3,model,'icp',.3)
    assert data is not None
    H,g,score,coverage=data
    assert np.linalg.eigvalsh(H).min()>-1e-7
    assert np.allclose(H,H.T)
    cfg=CFG['registration'].copy();cfg['stages']=[{'mode':'ndt','voxel_m':2.,'iterations':30},{'mode':'icp','voxel_m':1.,'iterations':40}]
    reg=Registration(m,cfg)
    result=reg.align(source,T,3)
    json.dumps({k:v for k,v in result.items() if k not in ('hessian','T')},allow_nan=False)
    assert result['accepted'],{k:v for k,v in result.items() if k not in ('hessian','T')}
    assert np.linalg.norm(result['T'][:3,3])<.08
    assert Rotation.from_matrix(result['T'][:3,:3]).magnitude()<.015
    assert abs(result['scale']-3)<.03


def test_timestamp_and_frame_contract(tmp_path):
    ts=1727239790258245070
    for i in range(3):(tmp_path/f'{i:06d}_ts_{ts+i*10}.ply').write_text('')
    rows=frames(tmp_path,expected=3)
    assert rows[0]['timestamp_ns']==ts
    (tmp_path/f'000003_ts_{ts}.ply').write_text('')
    with pytest.raises(ValueError):frames(tmp_path,expected=4)


def test_stream_map_xyz_only(tmp_path):
    m=cloud()[:200]
    p=tmp_path/'m.pcd'
    with p.open('w') as f:
        f.write('VERSION 0.7\nFIELDS x y z tl th\nSIZE 4 4 4 8 8\nTYPE F F F F F\nCOUNT 1 1 1 1 1\nWIDTH 200\nHEIGHT 1\nPOINTS 200\nDATA ascii\n')
        np.savetxt(f,np.column_stack([m,np.full((len(m),2),1e18)]))
    got,n=load_map(p,.4)
    assert n==len(m) and got.shape[1]==3
    assert float(cp.abs(got).max())<20


def test_flat_map_not_accepted_as_full_pose():
    r=np.random.default_rng(9);m=r.uniform([-10,-10,0],[10,10,0],(12000,3))
    cfg=CFG['registration'].copy();cfg['stages']=[{'mode':'icp','voxel_m':1.,'iterations':10}]
    result=Registration(m,cfg).align(m/3,np.eye(4),3)
    assert not result['accepted'] and not result['observable']

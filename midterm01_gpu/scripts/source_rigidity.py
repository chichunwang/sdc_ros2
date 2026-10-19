"""RGB-associated cloud rigidity diagnostics; no map, GNSS or body extrinsics in fit.

Virtual pinhole images are a texture lookup, not a new calibration. 3-D matches
always reference actual input vertices; hole filling never manufactures depth.
"""
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from three_camera_source import ply_header


@dataclass(frozen=True)
class RigidityConfig:
    width: int = 448
    height: int = 294
    focal_px: float = 180.
    ply_scale: float = 3.
    fill_radius_px: float = 2.
    association_radius_px: float = 2.
    max_local_depth_ratio: float = 1.1
    ratio_test: float = .8
    min_split_matches: int = 12
    ransac_trials: int = 256
    inlier_m: float = .3
    robust_scale_m: float = .15


def read_rgb_ply(path):
    header, offset = ply_header(path)
    count = int(next(s.split()[2] for s in header if s.startswith('element vertex ')))
    expected = ['property float '+k for k in 'xyz'] + ['property uchar '+k for k in ('red','green','blue')]
    if [s for s in header if s.startswith('property ')] != expected:
        raise ValueError('Expected XYZ float + RGB uchar vertices')
    if 'format ascii 1.0' in header:
        data = np.loadtxt(path, skiprows=len(header), ndmin=2)
    elif 'format binary_little_endian 1.0' in header:
        dtype = np.dtype([('xyz','<f4',(3,)), ('rgb','u1',(3,))])
        if Path(path).stat().st_size != offset+count*dtype.itemsize:
            raise ValueError('PLY payload length mismatch')
        packed = np.fromfile(path, dtype=dtype, offset=offset, count=count)
        data = np.column_stack((packed['xyz'], packed['rgb']))
    else:
        raise ValueError('Unsupported PLY encoding')
    if data.shape != (count,6) or not np.isfinite(data).all() or np.any((data[:,3:]<0)|(data[:,3:]>255)):
        raise ValueError('Invalid XYZRGB payload')
    return data[:,:3], data[:,3:].astype(np.uint8)


def render_features(xyz, rgb, config=None):
    c = config or RigidityConfig()
    xyz = np.asarray(xyz,float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(rgb) != len(xyz) or not np.isfinite(xyz).all():
        raise ValueError('Finite matching XYZ/RGB arrays required')
    forward = xyz[:,0] > 1e-6
    indices = np.flatnonzero(forward)
    uv = -xyz[indices,1:]/xyz[indices,0,None]*c.focal_px+[c.width/2,c.height/2]
    pixels = np.rint(uv).astype(int)
    valid = ((pixels[:,0]>=0)&(pixels[:,0]<c.width)&(pixels[:,1]>=0)&(pixels[:,1]<c.height))
    indices, uv, pixels = indices[valid], uv[valid], pixels[valid]
    image = np.zeros((c.height,c.width,3),np.uint8)
    vertex = np.full((c.height,c.width),-1,int)
    # Explicit per-pixel z-buffer: nearest positive forward coordinate wins.
    order = np.argsort(xyz[indices,0],kind='stable')
    _, first = np.unique(pixels[order,1]*c.width+pixels[order,0],return_index=True)
    take = order[first]
    vertex[pixels[take,1],pixels[take,0]] = indices[take]
    image[pixels[take,1],pixels[take,0]] = rgb[indices[take]]
    present = vertex >= 0
    if not present.any():
        return dict(image=image,pixels=np.empty((0,2)),xyz=np.empty((0,3)),descriptors=None,vertex_indices=np.array([],int))
    distance, nearest = distance_transform_edt(~present,return_indices=True)
    filled = image[nearest[0],nearest[1]].copy()
    filled[distance>c.fill_radius_px] = 0
    mask = cv2.erode((distance<=c.fill_radius_px).astype(np.uint8),np.ones((5,5),np.uint8))*255
    keypoints, descriptors = cv2.SIFT_create(nfeatures=5000,contrastThreshold=.02).detectAndCompute(
        cv2.cvtColor(filled,cv2.COLOR_RGB2GRAY),mask)
    if not keypoints:
        return dict(image=filled,pixels=np.empty((0,2)),xyz=np.empty((0,3)),descriptors=None,vertex_indices=np.array([],int))
    locations = np.array([k.pt for k in keypoints])
    # Query only z-buffer-visible original points, not the filled image.
    visible = indices[take]
    tree = cKDTree(uv[take])
    dd, nn = tree.query(locations,k=min(5,len(visible)))
    if dd.ndim == 1: dd,nn = dd[:,None],nn[:,None]
    depths = xyz[visible[nn],0]
    nearby = dd<=c.association_radius_px
    lo = np.min(np.where(nearby,depths,np.inf),axis=1)
    hi = np.max(np.where(nearby,depths,0),axis=1)
    keep = (nearby.sum(axis=1)>=3)&(hi<=c.max_local_depth_ratio*lo)
    ids = visible[nn[:,0]]
    # One descriptor per physical vertex and 4px image cell.
    selected=[]; seen_vertex=set(); seen_cell=set()
    for i in np.flatnonzero(keep):
        cell=tuple(np.floor(locations[i]/4).astype(int))
        if int(ids[i]) not in seen_vertex and cell not in seen_cell:
            selected.append(i);seen_vertex.add(int(ids[i]));seen_cell.add(cell)
    selected=np.asarray(selected,int)
    return dict(image=filled,pixels=locations[selected],xyz=xyz[ids[selected]],
                descriptors=descriptors[selected],vertex_indices=ids[selected])


def match_features(a,b,config=None):
    c=config or RigidityConfig()
    da,db=a['descriptors'],b['descriptors']
    if da is None or db is None or min(len(da),len(db))<2:
        return np.empty((0,2),int)
    matcher=cv2.BFMatcher()
    def accepted(x,y):
        return {p[0].queryIdx:p[0].trainIdx for p in matcher.knnMatch(x,y,k=2)
                if len(p)==2 and p[0].distance<c.ratio_test*p[1].distance}
    ab,ba=accepted(da,db),accepted(db,da)
    return np.array([(i,j) for i,j in sorted(ab.items()) if ba.get(j)==i],int).reshape(-1,2)


def split_cells(pixels,config=None):
    c=config or RigidityConfig()
    cells=np.floor(np.asarray(pixels)/[c.width,c.height]*[4,3]).astype(int)
    return (cells[:,0]+2*cells[:,1])%3==0


def similarity_fit(a,b,with_scale=True):
    """Closed-form b = s R a + t. Reflections and collinear support rejected."""
    a,b=np.asarray(a,float),np.asarray(b,float)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=3 or len(a)<3 or not np.isfinite([a,b]).all():
        raise ValueError('At least three paired finite XYZ points required')
    ma,mb=a.mean(axis=0),b.mean(axis=0);aa,bb=a-ma,b-mb
    if np.linalg.svd(aa,compute_uv=False)[1]<1e-6 or np.linalg.svd(bb,compute_uv=False)[1]<1e-6:
        raise ValueError('Collinear or coincident correspondences')
    u,s,v=np.linalg.svd(bb.T@aa)
    sign=np.ones(3);sign[-1]=np.linalg.det(u@v)
    rotation=(u*sign)@v
    scale=float((s@sign)/np.sum(aa**2)) if with_scale else 1.
    if scale<=0: raise ValueError('Nonpositive similarity scale')
    return scale,rotation,mb-scale*rotation@ma


def apply_model(a,model):
    scale,rotation,translation=model
    return scale*np.asarray(a)@rotation.T+translation


def robust_model(a,b,with_scale,config=None):
    c=config or RigidityConfig();rng=np.random.default_rng(20261020)
    best=None;best_key=None
    for _ in range(c.ransac_trials):
        ids=rng.choice(len(a),4,replace=False)
        try: model=similarity_fit(a[ids],b[ids],with_scale)
        except ValueError: continue
        if not .5<model[0]<2:continue
        error=np.linalg.norm(apply_model(a,model)-b,axis=1)
        key=(int((error<c.inlier_m).sum()),-float(np.median(error)))
        if best_key is None or key>best_key:best,best_key=model,key
    if best is None:raise ValueError('No nondegenerate training hypothesis')
    s,r,t=best
    initial=np.r_[Rotation.from_matrix(r).as_rotvec(),t,np.log(s)] if with_scale else np.r_[Rotation.from_matrix(r).as_rotvec(),t]
    def unpack(x):return (float(np.exp(x[6])) if with_scale else 1.,Rotation.from_rotvec(x[:3]).as_matrix(),x[3:6])
    def residual(x):
        e=apply_model(a,unpack(x))-b
        # Radial pseudo-Huber: rotationally invariant, unlike component loss.
        norm2=np.sum(e*e,axis=1)/c.robust_scale_m**2
        weight=np.sqrt(2/(np.sqrt(1+norm2)+1))
        return (e*weight[:,None]).ravel()
    lower=np.r_[np.full(6,-np.inf),np.log(.5)] if with_scale else np.full(6,-np.inf)
    upper=np.r_[np.full(6,np.inf),np.log(2.)] if with_scale else np.full(6,np.inf)
    fit=least_squares(residual,initial,bounds=(lower,upper),max_nfev=120,
                      ftol=1e-9,xtol=1e-9,gtol=1e-9)
    return unpack(fit.x),bool(fit.success),bool(np.any(fit.active_mask))


def statistics(values):
    a=np.asarray(values,float)
    if not a.size:return None
    return dict(count=int(a.size),median=float(np.median(a)),p90=float(np.quantile(a,.9)),
                rms=float(np.sqrt(np.mean(a*a))))


def diagnose_matches(a,b,pixels,config=None):
    """All estimation uses training cells only. Holdout is never trimmed by 3-D error."""
    c=config or RigidityConfig();a,b=np.asarray(a,float),np.asarray(b,float)
    holdout=split_cells(pixels,c)
    report=dict(matches=len(a),train_matches=int((~holdout).sum()),holdout_matches=int(holdout.sum()),
                accepted_for_fusion=False,models={},reasons=[])
    if min(report['train_matches'],report['holdout_matches'])<c.min_split_matches:
        report['reasons']=['insufficient_spatial_matches'];return report
    for label,with_scale in [('rigid',False),('similarity',True)]:
        try:model,success,bound=robust_model(a[~holdout],b[~holdout],with_scale,c)
        except ValueError as error:
            report['reasons'].append(label+': '+str(error));continue
        errors=np.linalg.norm(apply_model(a,model)-b,axis=1)
        s,r,t=model
        report['models'][label]=dict(scale_b_from_a=s,rotation_b_from_a=r.tolist(),translation_b_from_a=t.tolist(),
            success=success,at_scale_bound=bound,train_error_m=statistics(errors[~holdout]),
            holdout_error_m=statistics(errors[holdout]),holdout_fraction_within_03m=float(np.mean(errors[holdout]<=.3)))
    # Distance ratios remove any rigid pose error, without fitted camera poses.
    for label,mask in [('train',~holdout),('holdout',holdout)]:
        aa,bb=a[mask],b[mask];i,j=np.triu_indices(len(aa),1)
        da=np.linalg.norm(aa[i]-aa[j],axis=1);db=np.linalg.norm(bb[i]-bb[j],axis=1)
        valid=(da>=1.)&(db>=1.)
        logs=np.log(db[valid]/da[valid])
        report[label+'_pair_distance_ratio']=dict(pairs=int(valid.sum()),
            median=float(np.exp(np.median(logs))) if len(logs) else None,
            log_mad=float(np.median(abs(logs-np.median(logs)))) if len(logs) else None,
            scope='Correlated pair distances; count is not independent sample size')
    report['scope']='Relative consistency in ply_scale units; not calibrated metric odometry or ATE. RGB matches may include dynamics or mismatches.'
    return report

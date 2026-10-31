"""Current-frame geometric startup: ground tilt plus feature-based metric Sim(3).

Ground determines tilt, NOT metric scale. Scale candidates from the existing
prior and its ground height are hypotheses, verified against non-ground XYZ.
The road PCA is an undirected seed only; the prior selects its sign.
"""
import numpy as np
import time
import cupy as cp
from scipy.spatial.transform import Rotation
from .gpu import voxel_mean
from .nearest import GpuNearest


class GroundInitializationError(ValueError):
    pass


def fit_plane(points, threshold, up_only=False):
    p=cp.asarray(points,dtype=cp.float64)
    if len(p)<30:raise GroundInitializationError('Too few ground-plane candidates')
    # Generate only the tiny deterministic proposal index table on host;
    # all point access, residuals, support counts and PCA stay on CUDA.
    indices=np.random.default_rng(246).integers(0,len(p),size=(256,3))
    triples=p[cp.asarray(indices)]
    normals=cp.cross(triples[:,1]-triples[:,0],triples[:,2]-triples[:,0])
    length=cp.linalg.norm(normals,axis=1)
    normals/=cp.maximum(length[:,None],1e-12)
    good=length>1e-10
    if up_only:good&=cp.abs(normals[:,2])>.95
    else:good&=cp.abs(normals[:,2])>.85
    normals=normals[good];anchors=triples[good,0]
    if not len(normals):raise GroundInitializationError('No FLU-up ground-plane hypothesis')
    offsets=-cp.sum(normals*anchors,axis=1)
    counts=cp.zeros(len(normals),dtype=cp.int64)
    for start in range(0,len(p),8192):
        counts+=cp.sum(cp.abs(p[start:start+8192]@normals.T+offsets)<threshold,axis=0)
    winner=int(cp.argmax(counts));n=normals[winner];d=offsets[winner]
    for _ in range(3):
        mask=cp.abs(p@n+d)<threshold
        if int(mask.sum())<30:raise GroundInitializationError('Insufficient ground-plane support')
        v=p[mask];mean=v.mean(0);_,V=cp.linalg.eigh(cp.cov(v.T));n=V[:,0]
        if float(n[2])<0:n=-n
        d=-n@mean
    mask=cp.abs(p@n+d)<threshold
    mad=float(cp.median(cp.abs(p[mask]@n+d)))
    return cp.asnumpy(n),float(d),dict(inliers=int(mask.sum()),samples=len(p),mad=mad,threshold=float(threshold))


def plane_basis(n):
    x=np.array([1.,0.,0.]);x-=n*np.dot(n,x);x/=np.linalg.norm(x)
    return np.column_stack([x,np.cross(n,x),n])


def transform4(u,x,*,jacobian=True):
    c,s=np.cos(x[2]),np.sin(x[2]);R=cp.array([[c,-s,0],[s,c,0],[0,0,1]],dtype=cp.float64)
    v=u@R.T*cp.exp(x[3]);q=v+cp.array([x[0],x[1],0.])
    if not jacobian:return q,None
    J=cp.zeros((len(u),3,4),dtype=cp.float64)
    J[:,0,0]=1;J[:,1,1]=1;J[:,0,2]=-v[:,1];J[:,1,2]=v[:,0];J[:,:,3]=v
    return q,J


class GroundInitializer:
    def __init__(self,map_points,cfg):
        self.map=cp.asarray(map_points,dtype=cp.float64);self.cfg=cfg;self.center=None;self.models={}

    def score_pose(self,T,scale):
        """Compare refined seeds on ONE fixed non-ground sample and objective.

        This score has no seed-dependent scale prior. Unmatched features keep
        the truncation penalty instead of disappearing from the denominator.
        All correspondences and weighted reductions stay on CUDA.
        """
        stage=self.cfg['stages'][-1]
        radius,robust=stage['radius_m'],stage['robust_m']
        q=self.feature_source@cp.asarray(T[:3,:3]).T*scale+cp.asarray(T[:3,3])
        q=(q-cp.asarray(self.feature_origin))@cp.asarray(self.feature_basis)
        ids,d2=self.feature_nn.query(q,radius)
        w=self.feature_weight;valid=ids>=0
        cost=float(cp.sum(w*robust**2*cp.log1p(d2/robust**2))/w.sum())
        coverage=float(cp.sum(w*valid)/w.sum())
        return dict(cost=cost,coverage=coverage,source_features=len(q),
                    radius_m=float(radius),robust_m=float(robust),
                    unmatched_policy='radius_squared_penalty_fixed_denominator',
                    median_capped_distance_m=float(cp.median(cp.sqrt(d2))))

    def align(self,source,T_initial,initial_scale):
        started=time.perf_counter()
        cfg=self.cfg;given=T_initial.copy();self.center=given[:3,3].copy()
        local=self.map[cp.all(cp.abs(self.map-cp.asarray(self.center))<cp.asarray(cfg['submap_half_extent_m']),axis=1)]
        near=local[cp.linalg.norm(local[:,:2]-cp.asarray(self.center[:2]),axis=1)<cfg['ground_radius_m']]
        cells=cp.floor(near[:,:2]/cfg['ground_grid_m']).astype(cp.int64)
        order=cp.lexsort(cp.stack((near[:,2],cells[:,1],cells[:,0])));cells=cells[order]
        starts=cp.r_[0,cp.flatnonzero(cp.any(cells[1:]!=cells[:-1],axis=1))+1]
        nm,dm,ms=fit_plane(near[order[starts]],cfg['map_ground_threshold_m'],True)
        ns,ds,ss=fit_plane(source,cfg['source_ground_threshold_raw'])
        if abs(ns[2])<.85 or ds<=0:raise GroundInitializationError('Source ground does not support a camera-above-ground FLU initialization')
        Bs=plane_basis(ns);Bm=plane_basis(nm);origin=self.center-nm*(nm@self.center+dm)
        u=cp.asarray(source)@cp.asarray(Bs);u[:,2]+=ds
        target=(local-cp.asarray(origin))@cp.asarray(Bm)
        cutoff=max(.1,3*ms['mad']);target=target[target[:,2]>cutoff]
        mask=u[:,2]>max(.02,6*ss['mad']);features=u[mask]
        sample,_=voxel_mean(features,cfg['source_feature_leaf_raw'])
        if len(sample)>cfg['max_source_points']:sample=sample[cp.linspace(0,len(sample)-1,cfg['max_source_points']).astype(cp.int64)]
        if len(sample)<30:raise GroundInitializationError('Too few non-ground features for scale initialization')
        # Equal angular sectors prevent dense near-road pixels from dominating.
        rays=sample-cp.array([0,0,ds]);az=cp.arctan2(rays[:,1],rays[:,0]);el=cp.arctan2(rays[:,2],cp.linalg.norm(rays[:,:2],axis=1))
        bins=cp.floor(cp.stack([az,el],axis=1)/.07).astype(cp.int64)
        _,inv,counts=cp.unique(bins,axis=0,return_inverse=True,return_counts=True)
        weight=1/cp.sqrt(counts[inv]);weight/=weight.mean()
        nn=GpuNearest(target)
        # Keep the same raw source features, target and angular weights for
        # every final SE(3)/Sim(3) hypothesis, regardless of its refined scale.
        self.feature_source=(sample-cp.array([0.,0.,ds]))@cp.asarray(Bs).T
        self.feature_weight=weight;self.feature_nn=nn
        self.feature_origin=origin;self.feature_basis=Bm
        height=float(nm@given[:3,3]+dm)
        scale_height=height/ds if height>0 else 3.
        scales=[float(initial_scale),float(np.clip(scale_height,*cfg['scale_bounds']))]
        scales=list(dict.fromkeys(round(v,8) for v in scales))
        forward=given[:3,:3]@Bs[:,0];f=Bm.T@forward;yaw0=float(np.arctan2(f[1],f[0]))
        ground=near[cp.abs(near@cp.asarray(nm)+dm)<cfg['map_ground_threshold_m']]
        ground,_=voxel_mean(ground,cfg['ground_grid_m'])
        xy=(ground-cp.asarray(origin))@cp.asarray(Bm)
        _,V=np.linalg.eigh(cp.asnumpy(cp.cov(xy[:,:2].T)));a=V[:,-1];road=float(np.arctan2(a[1],a[0]));road+=np.pi*round((yaw0-road)/np.pi)
        yaws=[yaw0,road] if abs(np.arctan2(np.sin(road-yaw0),np.cos(road-yaw0)))>.05 else [yaw0]
        candidates=[];best=None
        print(f'initialization features={len(sample)} map_features={len(target)} ground_height_scale_seed={scale_height:.4f}',flush=True)
        for scale_seed in scales:
            for yaw_seed in yaws:
                x=np.array([0.,0.,yaw_seed,np.log(scale_seed)],dtype=float)
                stages=[];converged=False
                for stage in cfg['stages']:
                    stage_started=time.perf_counter()
                    radius,robust,iterations=stage['radius_m'],stage['robust_m'],stage['iterations']
                    trace=[]
                    def evaluate(xx,derivatives=False):
                        q,J=transform4(sample,xx);ids,d2=nn.query(q,radius)
                        valid=ids>=0;w=weight
                        cost=float((w*robust**2*cp.log1p(d2/robust**2)).sum()/w.sum())
                        if not derivatives:return cost,float((w*valid).sum()/w.sum())
                        if int(valid.sum())<20:return None
                        residual=q[valid]-nn.points[ids[valid]];A=J[valid];ww=w[valid]/(1+d2[valid]/robust**2)
                        H=cp.einsum('nki,nkj,n->ij',A,A,ww);g=cp.einsum('nki,nk,n->i',A,residual,ww)
                        return cost,cp.asnumpy(H),cp.asnumpy(g),float((w*valid).sum()/w.sum())
                    converged=False;reason='iterations'
                    for iteration in range(iterations):
                        e=evaluate(x,True)
                        if e is None:reason='insufficient_correspondences';break
                        cost,H,g,coverage=e
                        delta=np.linalg.solve(H+np.diag(np.maximum(np.diag(H),1e-8))*1e-5,-g)
                        cap=max(1.,np.linalg.norm(delta[:2])/1.5,abs(delta[2])/.1,abs(delta[3])/.15);delta/=cap
                        proposal=delta.copy();chosen=None
                        for half in range(18):
                            alpha=2.**(-half);trial=x+alpha*delta
                            if not cfg['scale_bounds'][0]<np.exp(trial[3])<cfg['scale_bounds'][1] or np.linalg.norm(trial[:2])>cfg['max_translation_m']:continue
                            after,cov=evaluate(trial)
                            if after<=cost+1e-4*alpha*float(2*g@delta/cp.asnumpy(weight.sum())):
                                chosen=(trial,after,cov,alpha);break
                        trace.append(dict(cost=cost,coverage=coverage,proposal=proposal.tolist(),alpha=None if chosen is None else chosen[3]))
                        small=np.linalg.norm(delta[:2])<.002 and abs(delta[2])<.0002 and abs(delta[3])<.0002
                        if chosen is None:
                            reason='stationary' if small else 'line_search_failed';converged=bool(small);break
                        x=chosen[0]
                        if small:reason='stationary';converged=True;break
                    cost,coverage=evaluate(x)
                    stages.append(dict(radius_m=radius,robust_m=robust,iterations=len(trace),stop_reason=reason,cost=cost,coverage=coverage,trace=trace,elapsed_seconds=time.perf_counter()-stage_started))
                scale=float(np.exp(x[3]));Rz=Rotation.from_rotvec([0,0,x[2]]).as_matrix()
                T=np.eye(4);T[:3,:3]=Bm@Rz@Bs.T;T[:3,3]=origin+Bm@np.array([x[0],x[1],scale*ds])
                q,_=transform4(sample,x);ids,d2=nn.query(q,1.5);valid=ids>=0
                distance=np.sqrt(cp.asnumpy(d2));dist_valid=distance[cp.asnumpy(valid)]
                diagnostics=dict(seed_scale=float(scale_seed),seed_yaw_deg=float(np.degrees(yaw_seed)),T=T.tolist(),scale=scale,
                                 cost=cost,coverage=coverage,converged=bool(converged),stages=stages,
                                 median_distance_m=float(np.median(dist_valid)) if len(dist_valid) else None,
                                 rmse_distance_m=float(np.sqrt(np.mean(dist_valid**2))) if len(dist_valid) else None,
                                 source_features=len(sample),camera_height_m=scale*ds)
                print(f"candidate scale_seed={scale_seed:.3f} yaw_seed={np.degrees(yaw_seed):.2f}: s={scale:.4f}, xy={T[0,3]:.3f},{T[1,3]:.3f}, yaw={np.degrees(np.arctan2(T[1,0],T[0,0])):.2f}, cost={cost:.5f} cov={coverage:.3f} {reason}",flush=True)
                candidates.append(diagnostics)
                if best is None or cost<best['cost']:best=diagnostics
        T=np.array(best['T']);scale=best['scale']
        accepted=best['converged'] and best['coverage']>=cfg['min_coverage']
        return dict(T=T,scale=scale,accepted=bool(accepted),reason='geometric_candidate' if accepted else 'unresolved_registration',
                    diagnostics=dict(source_ground=dict(normal=ns.tolist(),offset=ds,**ss),map_ground=dict(normal=nm.tolist(),offset=dm,**ms),
                                     initial_height_m=height,source_basis=Bs.tolist(),map_basis=Bm.tolist(),ground_origin=origin.tolist(),
                                     source_points=len(source),feature_points=len(sample),map_feature_points=len(target),
                                     elapsed_seconds=time.perf_counter()-started,
                                     imu_used=False,model='4DOF grounded Sim3 robust exact-nearest-point distance',
                                     precision_claim='Geometric candidate only; ground equality is imposed, not an independent accuracy check'),
                    candidates=candidates)

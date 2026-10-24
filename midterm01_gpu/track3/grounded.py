"""First-frame camera-only Sim(3) initialization constrained by current ground planes.

Correspondence and objective both use Euclidean point distances. The GPU grid
returns exact nearest points inside a finite radius; truncation is continuous.
No IMU, GNSS, future images or historical poses enter this module.
"""
import numpy as np
import cupy as cp
from scipy.spatial.transform import Rotation
from .gpu import voxel_mean,keys

NN_SOURCE=r'''
extern "C" __global__ void nearest(const double* q, const double* p,
 const long long* key, const long long* start, const long long* end,
 int nc, int nq, double cell, double radius, long long* ids, double* dist) {
 int i=blockDim.x*blockIdx.x+threadIdx.x; if(i>=nq)return;
 double x=q[3*i],y=q[3*i+1],z=q[3*i+2];
 long long cx=(long long)floor(x/cell),cy=(long long)floor(y/cell),cz=(long long)floor(z/cell);
 int r=(int)ceil(radius/cell); double best=radius*radius;long long winner=-1;
 for(int dx=-r;dx<=r;dx++)for(int dy=-r;dy<=r;dy++)for(int dz=-r;dz<=r;dz++) {
  long long xx=cx+dx+1048576,yy=cy+dy+1048576,zz=cz+dz+1048576;
  if(xx<0||yy<0||zz<0||xx>=2097152||yy>=2097152||zz>=2097152)continue;
  long long k=(xx<<42)|(yy<<21)|zz;int lo=0,hi=nc;
  while(lo<hi){int m=(lo+hi)/2;if(key[m]<k)lo=m+1;else hi=m;}
  if(lo==nc||key[lo]!=k)continue;
  for(long long j=start[lo];j<end[lo];j++){
   double a=x-p[3*j],b=y-p[3*j+1],c=z-p[3*j+2],d=a*a+b*b+c*c;
   if(d<best){best=d;winner=j;}
  }
 }
 ids[i]=winner;dist[i]=best;
}
'''


class GpuNearest:
    def __init__(self,points,cell=1.5):
        p=cp.asarray(points,dtype=cp.float64);k=keys(cp.floor(p/cell));order=cp.argsort(k)
        self.points=cp.ascontiguousarray(p[order]);k=k[order]
        self.start=cp.r_[cp.array([0],dtype=cp.int64),cp.flatnonzero(k[1:]!=k[:-1])+1]
        self.end=cp.r_[self.start[1:],len(p)].astype(cp.int64)
        self.key=k[self.start];self.cell=cell
        self.kernel=cp.RawKernel(NN_SOURCE,'nearest')

    def query(self,points,radius):
        q=cp.ascontiguousarray(cp.asarray(points,dtype=cp.float64));ids=cp.empty(len(q),dtype=cp.int64);d=cp.empty(len(q),dtype=cp.float64)
        self.kernel(((len(q)+127)//128,),(128,),(q,self.points,self.key,self.start,self.end,np.int32(len(self.key)),np.int32(len(q)),np.float64(self.cell),np.float64(radius),ids,d))
        return ids,d


def fit_plane(points,threshold,up_only=False):
    p=np.asarray(points,dtype=float);rng=np.random.default_rng(246)
    if len(p)>35000:p=p[rng.choice(len(p),35000,replace=False)]
    best=None;count=0
    for _ in range(200):
        a,b,c=p[rng.choice(len(p),3,replace=False)];n=np.cross(b-a,c-a);length=np.linalg.norm(n)
        if length<1e-10:continue
        n/=length
        if up_only and abs(n[2])<.95:continue
        inliers=np.abs((p-a)@n)<threshold
        if int(inliers.sum())>count:best,count=inliers,int(inliers.sum())
    if best is None:raise ValueError('No supported ground plane')
    for _ in range(3):
        v=p[best];mean=v.mean(0);_,V=np.linalg.eigh(np.cov(v.T));n=V[:,0]
        if n[2]<0:n=-n
        d=-n@mean;best=np.abs(p@n+d)<threshold
    mad=float(np.median(np.abs(p[best]@n+d)))
    return n,float(d),dict(inliers=int(best.sum()),samples=len(p),mad=mad,threshold=threshold)


def plane_basis(n):
    x=np.array([1.,0.,0.]);x-=n*np.dot(n,x);x/=np.linalg.norm(x)
    return np.column_stack([x,np.cross(n,x),n])


def transform4(u,x):
    c,s=np.cos(x[2]),np.sin(x[2]);R=cp.array([[c,-s,0],[s,c,0],[0,0,1]],dtype=cp.float64)
    v=u@R.T*cp.exp(x[3]);q=v+cp.array([x[0],x[1],0.])
    J=cp.zeros((len(u),3,4),dtype=cp.float64)
    J[:,0,0]=1;J[:,1,1]=1;J[:,0,2]=-v[:,1];J[:,1,2]=v[:,0];J[:,:,3]=v
    return q,J


class GroundedRegistration:
    def __init__(self,map_points,cfg):
        self.map=cp.asarray(map_points,dtype=cp.float64);self.cfg=cfg;self.center=None;self.models={}

    def align_frame0(self,source,T_initial,initial_scales=None):
        cfg=self.cfg;given=T_initial.copy();self.center=given[:3,3].copy()
        local=self.map[cp.all(cp.abs(self.map-cp.asarray(self.center))<cp.array([60.,60.,35.]),axis=1)]
        m=cp.asnumpy(local)
        near=m[np.linalg.norm(m[:,:2]-self.center[:2],axis=1)<18]
        cells=np.floor(near[:,:2]/.6).astype(int);order=np.lexsort((near[:,2],cells[:,1],cells[:,0]));cells=cells[order]
        starts=np.r_[0,np.flatnonzero(np.any(cells[1:]!=cells[:-1],axis=1))+1]
        nm,dm,ms=fit_plane(near[order[starts]],.12,True)
        ns,ds,ss=fit_plane(source,.012)
        if abs(ns[2])<.85:raise ValueError('Dominant source plane does not support FLU-up initialization')
        Bs=plane_basis(ns);Bm=plane_basis(nm);origin=self.center-nm*(nm@self.center+dm)
        u=np.asarray(source)@Bs;u[:,2]+=ds
        target=(local-cp.asarray(origin))@cp.asarray(Bm)
        cutoff=max(.1,3*ms['mad']);target=target[target[:,2]>cutoff]
        mask=u[:,2]>max(.02,6*ss['mad']);features=u[mask]
        sample,_=voxel_mean(features,.035)
        if len(sample)>10000:sample=sample[cp.linspace(0,len(sample)-1,10000).astype(cp.int64)]
        # Equal angular sectors prevent dense near-road pixels from dominating.
        rays=sample-cp.array([0,0,ds]);az=cp.arctan2(rays[:,1],rays[:,0]);el=cp.arctan2(rays[:,2],cp.linalg.norm(rays[:,:2],axis=1))
        bins=cp.floor(cp.stack([az,el],axis=1)/.07).astype(cp.int64)
        _,inv,counts=cp.unique(bins,axis=0,return_inverse=True,return_counts=True)
        weight=1/cp.sqrt(counts[inv]);weight/=weight.mean()
        nn=GpuNearest(target)
        height=float(nm@given[:3,3]+dm)
        scale_height=height/ds if height>0 else 3.
        scales=list(initial_scales) if initial_scales is not None else [3.,float(np.clip(scale_height,.5,25.))]
        forward=given[:3,:3]@Bs[:,0];f=Bm.T@forward;yaw0=float(np.arctan2(f[1],f[0]))
        ground=m[np.abs(m@nm+dm)<.12];ground=ground[np.linalg.norm(ground[:,:2]-given[:2,3],axis=1)<18]
        _,ix=np.unique(np.floor(ground[:,:2]/.4).astype(int),axis=0,return_index=True);ground=ground[ix]
        _,V=np.linalg.eigh(np.cov(((ground-origin)@Bm)[:,:2].T));a=V[:,-1];road=float(np.arctan2(a[1],a[0]));road+=np.pi*round((yaw0-road)/np.pi)
        yaws=[yaw0,road]
        candidates=[];best=None
        for scale_seed in scales:
            for yaw_seed in yaws:
                x=np.array([0.,0.,yaw_seed,np.log(scale_seed)],dtype=float)
                stages=[];converged=False
                for radius,robust,iterations in [(3.,1.,50),(1.5,.5,50),(.75,.3,50)]:
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
                            if not .5<np.exp(trial[3])<25 or np.linalg.norm(trial[:2])>25:continue
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
                    stages.append(dict(radius_m=radius,robust_m=robust,iterations=len(trace),stop_reason=reason,cost=cost,coverage=coverage,trace=trace))
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
        accepted=best['converged'] and best['coverage']>.6
        return dict(T=T,scale=scale,accepted=bool(accepted),reason='geometric_candidate' if accepted else 'unresolved_registration',
                    diagnostics=dict(source_ground=dict(normal=ns.tolist(),offset=ds,**ss),map_ground=dict(normal=nm.tolist(),offset=dm,**ms),
                                     initial_height_m=height,source_basis=Bs.tolist(),map_basis=Bm.tolist(),ground_origin=origin.tolist(),
                                     source_points=len(source),feature_points=len(sample),map_feature_points=len(target),
                                     imu_used=False,model='4DOF grounded Sim3 robust exact-nearest-point distance',
                                     precision_claim='Geometric candidate only; ground equality is imposed, not an independent accuracy check'),
                    candidates=candidates)

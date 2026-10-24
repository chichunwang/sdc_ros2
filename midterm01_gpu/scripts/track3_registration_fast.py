"""Same bounded NDT/ICP model, packed reductions and concurrent candidates.

Float64/int64, original points/radii/iterations/gates, and CPU lstsq remain.
Only independent current-frame starts may run concurrently. No future inputs.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, wait
import hashlib
from pathlib import Path
import threading
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from track3_registration import (Geometry, MapRegistrar, RegistrationConfig,
    block_validation, checked_pose, point_jacobian, source_weights, xyz)
from track3_fast_kernels import FusedKernels


class DeviceGrid:
    """Equivalent grid construction from CUDA XYZ, without full host roundtrip."""
    def __init__(self, points, kernels, leaf=1.5):
        self.points=points.contiguous()
        if (not self.points.is_cuda or self.points.dtype!=torch.float64 or not len(points)
                or not bool(torch.isfinite(points).all())):
            raise ValueError('Device grid requires nonempty CUDA float64 XYZ')
        self.kernels=kernels;self.leaf=float(leaf)
        # Preserve the reference's HOST float64 floor and its existing
        # torch.tensor(low) default-dtype conversion for cell assignment.
        # Changing either here would silently change grid boundary semantics.
        self.low=(np.floor(points.amin(0).cpu().numpy()/leaf)*leaf-leaf).tolist()
        low=torch.tensor(self.low,device=points.device)
        coords=torch.floor((self.points-low)/leaf).to(torch.int64)
        self.shape=(coords.amax(0)+1).cpu().tolist()
        cells=int(np.prod(self.shape))
        if cells>32_000_000:raise ValueError('Dense CUDA grid exceeds explicit memory budget')
        keys=(coords[:,0]*self.shape[1]+coords[:,1])*self.shape[2]+coords[:,2]
        self.order=torch.argsort(keys,stable=True)
        counts=torch.bincount(keys,minlength=cells)
        self.offsets=torch.cat([torch.zeros(1,device=points.device,dtype=torch.int64),counts.cumsum(0)])
        self.boundary_refinements=0;self.cpu_tree=None;self.cpu=None
        self._cpu_tree_lock=threading.Lock()

    def query_gpu(self, points, k, radius):
        points=points.contiguous()
        if points.dtype!=torch.float64 or not points.is_cuda:raise ValueError('CUDA float64 query required')
        d,ids=self.kernels.query(self,points,min(k+1,25),radius)
        if k<25:
            tied=(torch.isfinite(d[:,k]) & ((d[:,k]-d[:,k-1]).abs()<=1e-11)).nonzero().ravel()
            if len(tied):
                with self._cpu_tree_lock:
                    if self.cpu_tree is None:
                        self.cpu=self.points.cpu().numpy();self.cpu_tree=cKDTree(self.cpu)
                dd,ii=self.cpu_tree.query(points[tied].cpu().numpy(),k=k,workers=1)
                dd,ii=np.asarray(dd).reshape(-1,k),np.asarray(ii).reshape(-1,k)
                dd=np.where(dd<=radius,dd,np.inf);ii=np.where(np.isfinite(dd),ii,0)
                d[tied,:k]=torch.as_tensor(dd,device=points.device)
                ids[tied,:k]=torch.as_tensor(ii,device=points.device)
                self.boundary_refinements+=len(tied)
        return d[:,:k].contiguous(),ids[:,:k].contiguous()


class FastGeometry(Geometry):
    def __init__(self, backend, cache_dir=None, math_mode='fused', parallel_candidates=True):
        self.backend=backend
        if backend not in ('cuda','cpu_reference'):raise ValueError('Explicit supported backend required')
        self.device='cuda' if backend=='cuda' else 'cpu'
        self.math_mode=math_mode
        if math_mode not in ('fused','torch_pack'):raise ValueError('Explicit supported math mode required')
        self.parallel_candidates=parallel_candidates
        self.kernels=FusedKernels(Path(cache_dir)) if backend=='cuda' else None
        self._local=threading.local();self._source_lock=threading.Lock();self._prepared=None
        self.executor=ThreadPoolExecutor(max_workers=3,thread_name_prefix='ndt_candidate') if backend=='cuda' else None

    def tensor(self, value):
        array=np.ascontiguousarray(value)
        if array.size>16:return super().tensor(array)
        cache=getattr(self._local,'tiny',None)
        if cache is None:cache=OrderedDict();self._local.tiny=cache
        key=(array.shape,array.dtype.str,array.tobytes())
        if key not in cache:
            cache[key]=super().tensor(array)
            if len(cache)>24:cache.popitem(last=False)
        result=cache[key]
        if result.is_cuda:result.record_stream(torch.cuda.current_stream())
        return result

    def tree(self, points, leaf=1.5):
        if isinstance(points,torch.Tensor):
            if points.is_cuda:return DeviceGrid(points,self.kernels,leaf)
            points=points.numpy()
        if self.backend=='cuda':return DeviceGrid(self.tensor(points),self.kernels,leaf)
        return super().tree(points,leaf)

    def sync(self):
        if self.backend=='cuda':torch.cuda.current_stream().synchronize()

    def prepare(self, raw, mask, supplied):
        with self._source_lock:
            key=(hashlib.sha256(mask.tobytes()).digest(),id(supplied))
            previous=self._prepared
            if previous is not None and previous['raw'] is raw and previous['key']==key:
                return previous
            source=self.tensor(raw[mask])
            if supplied is None:weights=source_weights(source)
            else:
                values=np.asarray(supplied,dtype=np.float64)
                if values.shape!=(len(raw),) or not np.isfinite(values[mask]).all() or np.any(values[mask]<=0):
                    raise ValueError('Expected positive finite immutable training weights per source row')
                weights=self.tensor(values[mask]);weights=weights/weights.mean()
            # This single scalar transfer also completes creation before other
            # candidate streams can read the retained immutable source tensors.
            normalizer=float(weights.sum())
            prepared=dict(raw=raw,key=key,source=source,weights=weights,normalizer=normalizer,
                          supplied=supplied)
            self._prepared=prepared
            return prepared


def packed_reference(rotated,residual,precision,good,weights,dimension,c2,cap,normalizer):
    r2=torch.einsum('ni,nij,nj->n',residual,precision,residual).clamp_min(0)
    z=r2 if cap is None else r2.clamp_max(cap)
    rho=z/(z+c2)
    if cap is not None:rho=rho/(cap/(cap+c2))
    rho=torch.where(good,rho,torch.ones_like(rho))
    pieces=[]
    if dimension:
        jac=point_jacobian(rotated,dimension==7)
        weight=c2/(r2+c2).square()*good*weights
        jt=jac.transpose(1,2) @ precision
        h=(jt @ jac*weight[:,None,None]).sum(0)
        b=((jt @ residual[:,:,None])[:,:,0]*weight[:,None]).sum(0)
        pieces.extend([h.reshape(-1),b])
    pieces.extend([((rho*weights).sum()/normalizer).reshape(1),good.sum().to(weights.dtype).reshape(1)])
    return torch.cat(pieces)


class FastMapRegistrar(MapRegistrar):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._local=threading.local()

    def _pairs(self,raw,pose,scale,kind,data,radius):
        if self.geometry.backend=='cuda':
            from pair_kernels_impl import pairs
            return pairs(self,raw,pose,scale,kind,data,radius)
        return super()._pairs(raw,pose,scale,kind,data,radius)

    def _statistics(self,rotated,residual,precision,good,weights,dimension,cap,normalizer):
        g=self.geometry
        if g.backend=='cuda' and g.math_mode=='fused':
            return g.kernels.statistics(rotated,residual,precision,good,weights,dimension,
                                        self.config.robust_c2,cap,normalizer)
        return packed_reference(rotated,residual,precision,good,weights,dimension,
                                self.config.robust_c2,cap,normalizer)

    def fit(self,raw,initial,scale,*,estimate_scale=True,use_ndt=True,train_mask=None,training_weights=None):
        raw=xyz(raw);initial=checked_pose(initial)
        if not np.isfinite(scale) or not self.config.min_scale<scale<self.config.max_scale:
            raise ValueError('Scale seed must be inside explicit domain')
        mask=~block_validation(raw) if train_mask is None else np.asarray(train_mask,bool)
        if mask.shape!=(len(raw),) or mask.sum()<self.config.min_support:
            raise ValueError('Insufficient training source blocks')
        prepared=self.geometry.prepare(raw,mask,training_weights)
        source,weights,normalizer=prepared['source'],prepared['weights'],prepared['normalizer']
        dimension=7 if estimate_scale else 6
        if source.is_cuda:
            stream=torch.cuda.current_stream();source.record_stream(stream);weights.record_stream(stream)
        pose,scale=initial.copy(),float(scale)
        levels=[]
        stages=[('ndt',d,2*d[0]) for d in self.ndt] if use_ndt else []
        stages += [('icp',None,d) for d in self.config.icp_distances]
        units=np.r_[np.ones(3),[.05]*3,[.1] if estimate_scale else []]
        for kind,data,radius in stages:
            status=dict(kind=kind,radius_m=radius,converged=False,iterations=0,stop='iteration_limit')
            plateau=0
            for it in range(self.config.iterations):
                rotated,residual,precision,good,ids=self._pairs(source,pose,scale,kind,data,radius)
                cap=(radius/.15)**2 if kind=='icp' else None
                packed=self._statistics(rotated,residual,precision,good,weights,dimension,cap,normalizer).cpu().numpy()
                support=int(packed[-1]);cost=float(packed[-2])
                status.update(iterations=it+1,support=support)
                if support<self.config.min_support:
                    status['stop']='insufficient_support';break
                h=packed[:dimension*dimension].reshape(dimension,dimension)
                b=packed[dimension*dimension:dimension*dimension+dimension]
                hs=h*units[:,None]*units[None,:]
                step=np.linalg.lstsq(hs+np.eye(len(units))*max(np.trace(hs),1.)*1e-8,
                                    -b*units,rcond=1e-8)[0]*units
                factor=min(1.,.75/max(np.linalg.norm(step[:3]),1e-12),
                           np.deg2rad(3)/max(np.linalg.norm(step[3:6]),1e-12))
                if estimate_scale:factor=min(factor,.08/max(abs(step[6]),1e-12))
                step*=factor
                if (np.linalg.norm(step[:3])<.001 and np.linalg.norm(step[3:6])<1e-4
                        and (not estimate_scale or abs(step[6])<1e-4)):
                    status.update(converged=True,stop='small_step');break
                moved=False
                for alpha in (1.,.5,.25,.125,.0625,.03125):
                    candidate=pose.copy()
                    candidate[:3,:3]=Rotation.from_rotvec(alpha*step[3:6]).as_matrix() @ pose[:3,:3]
                    candidate[:3,3]+=alpha*step[:3]
                    s=scale*np.exp(alpha*step[6]) if estimate_scale else scale
                    if not self._inside(candidate,s,initial):continue
                    _,rr,pp,gg,_=self._pairs(source,candidate,s,kind,data,radius)
                    new_cost=float(self._statistics(None,rr,pp,gg,weights,0,cap,normalizer)[0])
                    if new_cost<cost-1e-12:
                        drop=cost-new_cost
                        pose,scale,moved=candidate,float(s),True
                        small=(alpha*np.linalg.norm(step[:3])<.01 and alpha*np.linalg.norm(step[3:6])<.001
                               and (not estimate_scale or alpha*abs(step[6])<.002))
                        plateau=plateau+1 if drop<1e-5 and small else 0
                        status.update(cost=new_cost,last_step=(alpha*step).tolist());break
                if not moved:status['stop']='no_descent_or_domain';break
                if plateau>=3:status.update(converged=True,stop='cost_and_step_stable');break
            levels.append(status)
        report=self.evaluate(raw,pose,scale,mask,estimate_scale)
        c=self.config
        interior=(scale>c.min_scale*1.01 and scale<c.max_scale/1.01
            and np.linalg.norm(pose[:3,3]-initial[:3,3])<c.translation_domain_m-.05
            and Rotation.from_matrix(pose[:3,:3] @ initial[:3,:3].T).magnitude()<np.deg2rad(c.rotation_domain_deg)-.001)
        checks=dict(converged=levels[-1]['converged'],interior=bool(interior),
            coverage=report['validation']['coverage']>=c.min_coverage,
            validation=report['validation']['median_m']<=c.validation_median_m and report['validation']['p90_m']<=c.validation_p90_m,
            observable=report['physical_rank']==dimension)
        report.update(pose=pose.tolist(),scale=scale,levels=levels,checks=checks,accepted=bool(all(checks.values())),
            reasons=[k for k,v in checks.items() if not v],backend='cuda_fast' if source.is_cuda else 'cpu_reference_fast',
            estimate_scale=estimate_scale,execution='packed_'+self.geometry.math_mode,
            accuracy='unknown: validation is geometry, not ground-truth pose error')
        self.geometry.sync()
        return report


def join_candidates(futures):
    """Join even when interrupted, before shared scene config can be restored."""
    error=None
    while True:
        try:
            wait(futures)
            break
        except BaseException as exc:
            if error is None:error=exc
    results=[]
    for future in futures:
        try:results.append(future.result())
        except BaseException as exc:
            if error is None:error=exc
    if error is not None:raise error
    return results


def parallel_map_candidates(scene,raw,prior,scale,config,weights):
    """Independent seeds, retained in original lexical order after joining."""
    g=scene.geometry
    from track3_visibility_fast import ray_depth_diagnostic
    def run(factor):
        seed=float(np.clip(scale*factor,config.min_scale*1.001,config.max_scale/1.001))
        fit=scene.fit(raw,prior,seed,use_ndt=True,training_weights=weights)
        fit['ray_depth']=ray_depth_diagnostic(scene,raw,fit['pose'],fit['scale'],~block_validation(raw))
        fit.update(seed_type='adjacent_cloud_prior',seed_scale=seed)
        return fit
    if g.backend!='cuda' or not g.parallel_candidates:return [run(f) for f in (.95,1.,1.05)]
    g.sync()
    def on_stream(factor):
        # cuBLAS retains a workspace per (host thread, CUDA stream). Reusing
        # exactly one stream per worker bounds those workspaces over long runs.
        stream=getattr(g._local,'candidate_stream',None)
        if stream is None:
            stream=torch.cuda.Stream();g._local.candidate_stream=stream
        with torch.cuda.stream(stream):
            try:
                result=run(factor)
                result['candidate_execution']=dict(strategy='independent_cuda_streams',stream=int(stream.cuda_stream))
                return result
            finally:
                stream.synchronize()
    futures=[g.executor.submit(on_stream,f) for f in (.95,1.,1.05)]
    # Always join every job, including on error, before callers restore config
    # or release shared scene tensors. Never leave a background fit running.
    return join_candidates(futures)

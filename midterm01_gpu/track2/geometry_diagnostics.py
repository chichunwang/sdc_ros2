"""CUDA-only feature sampling and residual summaries; no pose correction."""
import cupy as cp
import numpy as np
from . import parameters as p


def angular_stratified_cap(sample,ds,maximum):
    if len(sample)<=maximum:return sample,dict(policy='below_cap',before=len(sample),after=len(sample))
    rays=sample-cp.asarray([0.,0.,ds]);az=cp.arctan2(rays[:,1],rays[:,0])
    el=cp.arctan2(rays[:,2],cp.linalg.norm(rays[:,:2],axis=1))
    bins=cp.floor(cp.stack([az,el],axis=1)/p.ANGULAR_CELL_RAD).astype(cp.int64)
    offset=1<<20;key=((bins[:,0]+offset)<<21)|(bins[:,1]+offset)
    order=cp.lexsort(cp.stack([cp.sum(rays*rays,axis=1),key]));sorted_key=key[order]
    starts=cp.r_[True,sorted_key[1:]!=sorted_key[:-1]];positions=cp.arange(len(sample))
    first=cp.flatnonzero(starts);group=cp.cumsum(starts)-1
    within=positions-first[group]
    if len(first)>maximum:
        chosen=first[cp.linspace(0,len(first)-1,maximum).astype(cp.int64)]
    else:chosen=cp.lexsort(cp.stack([sorted_key,within]))[:maximum]
    selected=sample[order[chosen]]
    return selected,dict(policy='round_robin_angular_cells_near_to_far; original cap',
        before=len(sample),after=len(selected),angular_cells_before=len(first),
        angular_cells_after=int(cp.unique(sorted_key[chosen]).size),
        limitation='preserves rare view cells; does not guarantee independent surface-normal information')


def surface_profile(association,residual,ranges,weight):
    good=association['good'];lines=association.get('line_matches',cp.zeros(len(good),dtype=bool))
    planes=good&~lines;n=association['normals'][planes]
    w=weight[planes]*association['confidence'][planes]
    total=float(w.sum())
    eig=cp.zeros(3);effective=0.;dominant_fraction=None;bin_count=0
    if len(n) and total>0:
        gram=cp.einsum('ni,nj,n->ij',n,n,w)/total
        eig=cp.maximum(cp.linalg.eigvalsh((gram+gram.T)*.5),0.);eig/=eig.sum()
        effective=float(cp.exp(-cp.sum(cp.where(eig>0,eig*cp.log(cp.maximum(eig,1e-30)),0.))))
        dominant=cp.argmax(cp.abs(n),axis=1);sign=cp.sign(n[cp.arange(len(n)),dominant]);n=n*sign[:,None]
        angles=cp.stack([cp.arctan2(n[:,1],n[:,0]),cp.arctan2(n[:,2],cp.linalg.norm(n[:,:2],axis=1))],axis=1)
        bins=cp.floor(angles/np.deg2rad(p.LINE_MAX_DIRECTION_DIFFERENCE_DEG)).astype(cp.int64)
        key=(bins[:,0]+100)*1000+bins[:,1]+100;unique,inverse=cp.unique(key,return_inverse=True)
        mass=cp.bincount(inverse,weights=w);bin_count=len(unique)
        dominant_fraction=float(mass.max()/mass.sum())
    bands=[];previous=0.
    for bound in [p.FEATURE_MAX_RANGE_M*.25,p.FEATURE_MAX_RANGE_M*.5,p.FEATURE_MAX_RANGE_M]:
        mask=good&(ranges>=previous)&(ranges<bound);r=residual[mask];signed=residual[mask&planes]
        bands.append(dict(min_m=previous,max_m=bound,matches=len(r),
            plane_matches=len(signed),line_matches=int((mask&lines).sum()),
            signed_median_m=float(cp.median(signed)) if len(signed) else None,
            abs_p90_m=float(cp.quantile(cp.abs(r),.9)) if len(r) else None));previous=bound
    return dict(matches=int(good.sum()),normal_gram_eigenvalues=cp.asnumpy(eig).tolist(),
        plane_matches=int(planes.sum()),line_matches=int((good&lines).sum()),
        normal_effective_directions=effective,antipodal_normal_bins=bin_count,
        dominant_normal_bin_fraction=dominant_fraction,range_bands=bands,
        normal_weight_model='angular weight * map confidence * robust influence; planes only',
        caveat='normal diversity and residuals are local consistency diagnostics, never pose accuracy')

"""CUDA depth-aware statistical outlier removal for camera FLU clouds.

Use mean kNN distance / forward depth: image clouds become naturally sparser
with depth. A global distance-in-metres SOR would preferentially erase distant
facades, exactly the structures needed for scale and heading.
"""
import time
import numpy as np
import cupy as cp


KNN=cp.RawKernel(r'''
extern "C" __global__ void knn_mean(const double* p, int n, int k, double* mean) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i>=n)return;
    double best[32];
    for(int j=0;j<k;j++)best[j]=1.7976931348623157e308;
    double x=p[3*i],y=p[3*i+1],z=p[3*i+2];
    for(int j=0;j<n;j++) {
        if(i==j)continue;
        double dx=x-p[3*j],dy=y-p[3*j+1],dz=z-p[3*j+2];
        double d=dx*dx+dy*dy+dz*dz;
        if(d<best[k-1]) {
            int at=k-1;
            while(at>0 && d<best[at-1]) {best[at]=best[at-1];at--;}
            best[at]=d;
        }
    }
    double total=0.;for(int j=0;j<k;j++)total+=sqrt(best[j]);
    mean[i]=total/k;
}
''','knn_mean')


def mean_knn_distance(points, k=16, block=128):
    p=cp.asarray(points,dtype=cp.float64)
    if not 0<k<=32 or len(p)<=k:raise ValueError('kNN requires 1..32 neighbors and more input points')
    p=cp.ascontiguousarray(p)
    means=cp.empty(len(p),dtype=cp.float64)
    KNN(((len(p)+block-1)//block,),(block,),(p,np.int32(len(p)),np.int32(k),means))
    return means


def filter_camera_cloud(points,cfg):
    started=time.perf_counter()
    p=cp.asarray(points,dtype=cp.float64)
    finite=cp.all(cp.isfinite(p),axis=1)
    eligible=finite & (p[:,0]>0)
    ids=cp.flatnonzero(eligible);q=p[ids]
    diagnostics=dict(method='depth_normalized_knn_sor',input_points=len(p),
                     backend='cuda_exact_knn_streaming',
                     finite_forward_points=len(q),k=cfg['k'],sigma_multiplier=cfg['sigma_multiplier'])
    keep=cp.zeros(len(p),dtype=cp.bool_)
    if len(q)>cfg['k']:
        distance=mean_knn_distance(q,cfg['k'])
        statistic=distance/q[:,0]
        median=float(cp.median(statistic))
        sigma=1.4826*float(cp.median(cp.abs(statistic-median)))
        threshold=median+cfg['sigma_multiplier']*max(sigma,np.finfo(float).eps)
        selected=statistic<=threshold
        keep[ids[selected]]=True
        diagnostics.update(normalized_distance_median=median,normalized_distance_mad_sigma=sigma,
                           normalized_distance_threshold=threshold,
                           raw_distance_quantiles=cp.asnumpy(cp.quantile(distance,cp.asarray([.5,.9,.99]))).tolist())
    else:
        keep[ids]=True
    cp.cuda.get_current_stream().synchronize()
    diagnostics.update(kept_points=int(keep.sum()),removed_points=int((~keep).sum()),
                       elapsed_seconds=time.perf_counter()-started)
    return p[keep],diagnostics,keep

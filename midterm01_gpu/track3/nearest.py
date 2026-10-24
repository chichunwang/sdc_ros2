"""Exact CUDA nearest point inside a finite radius; no CPU fallback."""
import numpy as np
import cupy as cp
from .gpu import keys

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
        self.order=order
        self.points=cp.ascontiguousarray(p[order]);k=k[order]
        self.start=cp.r_[cp.array([0],dtype=cp.int64),cp.flatnonzero(k[1:]!=k[:-1])+1]
        self.end=cp.r_[self.start[1:],len(p)].astype(cp.int64)
        self.key=k[self.start];self.cell=cell
        self.kernel=cp.RawKernel(NN_SOURCE,'nearest')

    def query(self,points,radius):
        q=cp.ascontiguousarray(cp.asarray(points,dtype=cp.float64));ids=cp.empty(len(q),dtype=cp.int64);d=cp.empty(len(q),dtype=cp.float64)
        self.kernel(((len(q)+127)//128,),(128,),(q,self.points,self.key,self.start,self.end,np.int32(len(self.key)),np.int32(len(q)),np.float64(self.cell),np.float64(radius),ids,d))
        return ids,d



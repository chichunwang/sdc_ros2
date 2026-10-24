"""FP64 CUDA searches with bounded working memory and deterministic ties.

These kernels keep the same nearest-point / nearest-cell definitions. They
avoid materializing query x target or query x 125 x 3 x 3 temporary arrays.
"""
import numpy as np
import cupy as cp


PIXEL_SEARCH=cp.RawKernel(r'''
extern "C" __global__ void pixel_nearest(
 const double* q,const double* p,int nq,int np,long long* ids,double* d2) {
 int i=blockIdx.x,t=threadIdx.x;
 if(i>=nq)return;
 double x=q[2*i],y=q[2*i+1],best=1.0/0.0;
 long long winner=0;
 for(int j=t;j<np;j+=blockDim.x) {
  double dx=x-p[2*j],dy=y-p[2*j+1],d=dx*dx+dy*dy;
  if(d<best || (d==best && j<winner)){best=d;winner=j;}
 }
 __shared__ double values[128];__shared__ long long index[128];
 values[t]=best;index[t]=winner;__syncthreads();
 for(int stride=64;stride>0;stride>>=1){
  if(t<stride && (values[t+stride]<values[t] ||
   (values[t+stride]==values[t] && index[t+stride]<index[t]))){
   values[t]=values[t+stride];index[t]=index[t+stride];
  }
  __syncthreads();
 }
 if(t==0){ids[i]=index[0];d2[i]=values[0];}
}
''','pixel_nearest',options=('--fmad=false',))


VOXEL_SEARCH=cp.RawKernel(r'''
extern "C" __global__ void voxel_nearest(
 const double* q,int nq,const long long* key,const double* mean,
 const double* precision,int nc,double leaf,int ndt,long long* ids,bool* good) {
 int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=nq)return;
 double x=q[3*i],y=q[3*i+1],z=q[3*i+2];
 long long cx=(long long)floor(x/leaf),cy=(long long)floor(y/leaf),cz=(long long)floor(z/leaf);
 double best=1.0/0.0;long long winner=0;bool found=false;
 for(int dx=-2;dx<=2;dx++)for(int dy=-2;dy<=2;dy++)for(int dz=-2;dz<=2;dz++){
  long long xx=cx+dx+1048576,yy=cy+dy+1048576,zz=cz+dz+1048576;
  long long k=(xx<<42)|(yy<<21)|zz;int lo=0,hi=nc;
  while(lo<hi){int m=(lo+hi)/2;if(key[m]<k)lo=m+1;else hi=m;}
  if(lo==nc || key[lo]!=k)continue;
  double a=mean[3*lo]-x,b=mean[3*lo+1]-y,c=mean[3*lo+2]-z;
  double distance=a*a+b*b+c*c;
  if(!(distance<(1.5*leaf)*(1.5*leaf)))continue;
  double value=distance;
  if(ndt){
   const double* w=precision+9*lo;
   double u=a*w[0]+b*w[3]+c*w[6];
   double v=a*w[1]+b*w[4]+c*w[7];
   double r=a*w[2]+b*w[5]+c*w[8];
   value=u*a+v*b+r*c;
  }
  // Strict comparison preserves the original dx/dy/dz argmin tie order.
  if(value<best){best=value;winner=lo;found=true;}
 }
 ids[i]=winner;good[i]=found;
}
''','voxel_nearest',options=('--fmad=false',))


def closest_pixels(queries,points):
    q=cp.ascontiguousarray(cp.asarray(queries,dtype=cp.float64))
    p=cp.ascontiguousarray(cp.asarray(points,dtype=cp.float64))
    if q.ndim!=2 or q.shape[1]!=2 or p.ndim!=2 or p.shape[1]!=2:
        raise ValueError('Pixel coordinates must be Nx2')
    if not len(p):raise ValueError('Pixel search requires target points')
    ids=cp.empty(len(q),dtype=cp.int64);distance=cp.empty(len(q),dtype=cp.float64)
    if len(q):PIXEL_SEARCH((len(q),),(128,),(q,p,np.int32(len(q)),np.int32(len(p)),ids,distance))
    return ids,distance


def closest_voxels(queries,keys,means,precision,leaf,mode):
    q=cp.ascontiguousarray(cp.asarray(queries,dtype=cp.float64))
    keys=cp.ascontiguousarray(keys,dtype=cp.int64)
    means=cp.ascontiguousarray(means,dtype=cp.float64)
    precision=cp.ascontiguousarray(precision,dtype=cp.float64)
    if q.ndim!=2 or q.shape[1]!=3:raise ValueError('Voxel queries must be Nx3')
    ids=cp.empty(len(q),dtype=cp.int64);good=cp.empty(len(q),dtype=cp.bool_)
    if not len(q):return ids,good
    # Preserve the old signed-21-bit range error before packing offsets.
    cells=cp.floor(q/leaf).astype(cp.int64)
    if bool(cp.any((cells<-(1<<20)+2)|(cells>=(1<<20)-2))):
        raise ValueError('Point outside signed 21-bit voxel index range')
    VOXEL_SEARCH(((len(q)+127)//128,),(128,),(q,np.int32(len(q)),keys,means,
                 precision,np.int32(len(keys)),np.float64(leaf),np.int32(mode=='ndt'),ids,good))
    return ids,good

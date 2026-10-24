"""Deterministic FP64 fused normal-equation reductions; no model changes.

The existing exact kNN implementation and CUDA compiler flags are reused.
Fixed-tree reductions use no atomics. The tiny solver remains NumPy lstsq.
"""
import ctypes as C
import types
import torch

from cuda_scale_geometry import CudaKernels, CUDA_SOURCE
from pair_kernels_impl import PAIR_SOURCE


FUSED_SOURCE = r'''
extern "C" __global__ void fit_partials(
 const double* rotated, const double* residual, const double* precision,
 long long pstride, const bool* good, const double* weights,
 int count, int dimension, double c2, double cap, double* output) {
 const int lane=threadIdx.x, row=blockIdx.x*64+lane;
 const int stats=dimension*dimension+dimension+2;
 __shared__ double values[58*64];
 double v[58];
 for(int k=0;k<stats;++k)v[k]=0.;
 if(row<count){
  const double mass=weights[row];
  double rho=1.;
  if(good[row]){
   const double* p=precision+pstride*row;
   const double* r=residual+3*row;
   double pr[3];
   for(int i=0;i<3;++i)pr[i]=(p[3*i]*r[0]+p[3*i+1]*r[1])+p[3*i+2]*r[2];
   double r2=(r[0]*pr[0]+r[1]*pr[1])+r[2]*pr[2];
   r2=r2<0.?0.:r2;
   if(cap>0.){
    const double z=r2<cap?r2:cap;
    rho=(z/(z+c2))/(cap/(cap+c2));
   }else rho=r2/(r2+c2);
   v[stats-1]=1.;
   if(dimension){
    const double x=rotated[3*row],y=rotated[3*row+1],z=rotated[3*row+2];
    const double j[21]={1.,0.,0.,0.,z,-y,x, 0.,1.,0.,-z,0.,x,y, 0.,0.,1.,y,-x,0.,z};
    const double w=(c2/((r2+c2)*(r2+c2)))*mass;
    for(int a=0;a<dimension;++a){
     double jt[3];
     for(int i=0;i<3;++i)jt[i]=(j[a]*p[i]+j[7+a]*p[3+i])+j[14+a]*p[6+i];
     for(int b=0;b<dimension;++b)
      v[a*dimension+b]=((jt[0]*j[b]+jt[1]*j[7+b])+jt[2]*j[14+b])*w;
     v[dimension*dimension+a]=((jt[0]*r[0]+jt[1]*r[1])+jt[2]*r[2])*w;
    }
   }
  }
  v[stats-2]=rho*mass;
 }
 for(int k=0;k<stats;++k)values[k*64+lane]=v[k];
 __syncthreads();
 for(int stride=32;stride;stride>>=1){
  if(lane<stride)for(int k=0;k<stats;++k)values[k*64+lane]+=values[k*64+lane+stride];
  __syncthreads();
 }
 if(!lane)for(int k=0;k<stats;++k)output[blockIdx.x*stats+k]=values[k*64];
}

extern "C" __global__ void fit_reduce(
 const double* partial, int blocks, int stats, double normalizer, double* output){
 const int lane=threadIdx.x;
 __shared__ double values[58*64];
 for(int k=0;k<stats;++k){
  double s=0.;
  for(int i=lane;i<blocks;i+=64)s+=partial[i*stats+k];
  values[k*64+lane]=s;
 }
 __syncthreads();
 for(int stride=32;stride;stride>>=1){
  if(lane<stride)for(int k=0;k<stats;++k)values[k*64+lane]+=values[k*64+lane+stride];
  __syncthreads();
 }
 if(!lane)for(int k=0;k<stats;++k)
  output[k]=k==stats-2?values[k*64]/normalizer:values[k*64];
}
'''


FUSED_SOURCE += PAIR_SOURCE


class FusedKernels(CudaKernels):
    def __init__(self, cache_dir):
        # Private globals for the original constructor: never monkeypatch the
        # live reference module, and retain its project-root/cache guard.
        namespace=dict(CudaKernels.__init__.__globals__)
        namespace['CUDA_SOURCE']=CUDA_SOURCE+FUSED_SOURCE
        initialize=types.FunctionType(CudaKernels.__init__.__code__,namespace,
                                      CudaKernels.__init__.__name__)
        initialize(self,cache_dir)
        for name in ('fit_partials','fit_reduce','fast_transform','fast_match'):
            function=C.c_void_p()
            self.check(self.driver.cuModuleGetFunction(C.byref(function),self.module,name.encode()),
                       'cuModuleGetFunction '+name)
            self.functions[name]=function

    def statistics(self, rotated, residual, precision, good, weights,
                   dimension, c2, cap, normalizer):
        count=len(residual)
        stats=dimension*dimension+dimension+2
        blocks=(count+63)//64
        partial=torch.empty((blocks,stats),device=residual.device,dtype=torch.float64)
        packed=torch.empty(stats,device=residual.device,dtype=torch.float64)
        assert residual.is_contiguous() and good.is_contiguous() and weights.is_contiguous()
        assert precision.stride(1)==3 and precision.stride(2)==1
        args=[C.c_void_p(0 if rotated is None else rotated.data_ptr()),
              C.c_void_p(residual.data_ptr()),C.c_void_p(precision.data_ptr()),
              C.c_longlong(precision.stride(0)),C.c_void_p(good.data_ptr()),
              C.c_void_p(weights.data_ptr()),C.c_int(count),C.c_int(dimension),
              C.c_double(c2),C.c_double(-1. if cap is None else cap),
              C.c_void_p(partial.data_ptr())]
        self.launch('fit_partials',count,args)
        self.launch('fit_reduce',1,[C.c_void_p(partial.data_ptr()),C.c_int(blocks),
            C.c_int(stats),C.c_double(normalizer),C.c_void_p(packed.data_ptr())])
        return packed

"""FP64 transform and correspondence gather; exact grid search stays shared."""
import ctypes as C
import numpy as np
import torch

PAIR_SOURCE=r'''
extern "C" __global__ void fast_transform(
 const double* raw,int count,double scale,
 double a,double b,double c,double d,double e,double f,double g,double h,double i,
 double tx,double ty,double tz,double* rotated,double* world){
 const int row=blockIdx.x*64+threadIdx.x;
 if(row>=count)return;
 const double x=scale*raw[row*3],y=scale*raw[row*3+1],z=scale*raw[row*3+2];
 const double u=(a*x+b*y)+c*z,v=(d*x+e*y)+f*z,w=(g*x+h*y)+i*z;
 rotated[3*row]=u;rotated[3*row+1]=v;rotated[3*row+2]=w;
 world[3*row]=u+tx;world[3*row+1]=v+ty;world[3*row+2]=w+tz;
}

extern "C" __global__ void fast_match(
 const double* world,const double* distance,const long long* neighbors,int count,int k,
 const double* target,const double* precision,double* residual,double* output_precision,
 bool* good,long long* ids){
 const int row=blockIdx.x*64+threadIdx.x;
 if(row>=count)return;
 int best=0;
 double score=1./0.;
 if(k>1){
  for(int n=0;n<k;++n){
   double value=1./0.;
   if(isfinite(distance[row*k+n])){
    const long long id=neighbors[row*k+n];
    const double* p=precision+9*id;
    const double x=world[3*row]-target[3*id],y=world[3*row+1]-target[3*id+1],z=world[3*row+2]-target[3*id+2];
    const double u=(x*p[0]+y*p[3])+z*p[6];
    const double v=(x*p[1]+y*p[4])+z*p[7];
    const double w=(x*p[2]+y*p[5])+z*p[8];
    value=(u*x+v*y)+w*z;
   }
   // First minimum, including the reference argmin's first-NaN behavior.
   if(!isnan(score) && (isnan(value)||value<score)){score=value;best=n;}
  }
 }else score=distance[row];
 const long long id=neighbors[row*k+best];
 ids[row]=id;good[row]=isfinite(score);
 for(int j=0;j<3;++j)residual[3*row+j]=world[3*row+j]-target[3*id+j];
 if(k>1)for(int j=0;j<9;++j)output_precision[9*row+j]=precision[9*id+j];
}
'''


def pairs(scene,raw,pose,scale,kind,data,radius):
    g=scene.geometry;kernels=g.kernels;n=len(raw)
    rotated=torch.empty_like(raw);world=torch.empty_like(raw)
    arguments=[C.c_void_p(raw.data_ptr()),C.c_int(n),C.c_double(scale)]
    arguments.extend(C.c_double(v) for v in np.asarray(pose)[:3,:3].ravel())
    arguments.extend(C.c_double(v) for v in np.asarray(pose)[:3,3])
    arguments.extend([C.c_void_p(rotated.data_ptr()),C.c_void_p(world.data_ptr())])
    kernels.launch('fast_transform',n,arguments)
    if kind=='ndt':
        _,target,precision,tree=data;k=4
        output_precision=torch.empty((n,3,3),device=raw.device,dtype=torch.float64)
    else:
        target,tree,k=scene.target,scene.tree,1
        precision=None
        output_precision=g.tensor(np.eye(3)/.15**2).expand(n,-1,-1)
    distance,neighbors=g.query(tree,world,k,radius)
    residual=torch.empty_like(raw)
    good=torch.empty(n,device=raw.device,dtype=torch.bool)
    ids=torch.empty(n,device=raw.device,dtype=torch.int64)
    kernels.launch('fast_match',n,[C.c_void_p(world.data_ptr()),C.c_void_p(distance.data_ptr()),
        C.c_void_p(neighbors.data_ptr()),C.c_int(n),C.c_int(k),C.c_void_p(target.data_ptr()),
        C.c_void_p(0 if precision is None else precision.data_ptr()),C.c_void_p(residual.data_ptr()),
        C.c_void_p(0 if precision is None else output_precision.data_ptr()),
        C.c_void_p(good.data_ptr()),C.c_void_p(ids.data_ptr())])
    return rotated,residual,output_precision,good,ids

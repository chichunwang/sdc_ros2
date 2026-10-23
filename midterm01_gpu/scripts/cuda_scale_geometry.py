"""Float64 CUDA bounded exact kNN, PCA and balanced Sim(3) correspondences.

Reuses the CPU registration/solver code objects with local dependency bindings;
never patches the running reference module or changes any acceptance settings.
"""
import ctypes as C
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import types

import numpy as np
from scipy.spatial import cKDTree
import torch

import map_scale_icp as reference


# cuSOLVER's temporary workspace grows with the number of 3x3 matrices even
# though the covariance tensors themselves are small (9070 matrices used
# 4.59 GiB with torch 2.8/CUDA 12.8 on the RTX 5060). Bound only this independent
# operation, retaining every point, float64 arithmetic and all fit thresholds.
NORMAL_EIGH_BATCH_SIZE = 256


CUDA_SOURCE = r'''
extern "C" __global__ void grid_knn(
 const double* points, const long long* order, const long long* offsets,
 double leaf, double lx, double ly, double lz, int nx, int ny, int nz,
 const double* query, int count, int k, double radius, double* out_d, long long* out_i) {
 int q=blockIdx.x*blockDim.x+threadIdx.x;
 if(q>=count)return;
 double x=query[3*q],y=query[3*q+1],z=query[3*q+2];
 double inf=__longlong_as_double(0x7ff0000000000000ULL);
 double bd[25]; long long bi[25];
 for(int j=0;j<k;++j){bd[j]=inf;bi[j]=0;}
 int cx=(int)floor((x-lx)/leaf),cy=(int)floor((y-ly)/leaf),cz=(int)floor((z-lz)/leaf);
 int max_shell=(int)ceil(radius/leaf)+1;
 double r2=radius*radius;
 for(int shell=0;shell<=max_shell;++shell){
 for(int da=-shell;da<=shell;++da)for(int db=-shell;db<=shell;++db)for(int dc=-shell;dc<=shell;++dc){
  int aa=da<0?-da:da,ab=db<0?-db:db,ac=dc<0?-dc:dc;
  if(aa!=shell&&ab!=shell&&ac!=shell)continue;
  int a=cx+da,b=cy+db,c=cz+dc;
  if(a<0||a>=nx||b<0||b>=ny||c<0||c>=nz)continue;
  double x0=lx+a*leaf,y0=ly+b*leaf,z0=lz+c*leaf;
  double dx=x<x0?x0-x:(x>x0+leaf?x-x0-leaf:0.);
  double dy=y<y0?y0-y:(y>y0+leaf?y-y0-leaf:0.);
  double dz=z<z0?z0-z:(z>z0+leaf?z-z0-leaf:0.);
  double lower=dx*dx+dy*dy+dz*dz;
  if(lower>r2||lower>bd[k-1])continue;
  long long cell=((long long)a*ny+b)*nz+c;
  for(long long j=offsets[cell];j<offsets[cell+1];++j){
   long long i=order[j];
   double dx=points[3*i]-x,dy=points[3*i+1]-y,dz=points[3*i+2]-z;
   double d=dx*dx+dy*dy+dz*dz;
   if(d>r2||d>bd[k-1]||(d==bd[k-1]&&i>=bi[k-1]))continue;
   int p=k-1;
   while(p>0&&(d<bd[p-1]||(d==bd[p-1]&&i<bi[p-1]))){
    bd[p]=bd[p-1];bi[p]=bi[p-1];--p;
   }
   bd[p]=d;bi[p]=i;
  }
 }
 // All unvisited cells lie outside this box. A STRICT lower bound preserves
 // candidates tied at the kth distance, including queries on cell boundaries.
 double edge=x-(lx+(cx-shell)*leaf),v=lx+(cx+shell+1)*leaf-x;
 edge=v<edge?v:edge;
 v=y-(ly+(cy-shell)*leaf);edge=v<edge?v:edge;
 v=ly+(cy+shell+1)*leaf-y;edge=v<edge?v:edge;
 v=z-(lz+(cz-shell)*leaf);edge=v<edge?v:edge;
 v=lz+(cz+shell+1)*leaf-z;edge=v<edge?v:edge;
 if(edge>=0.&&edge*edge>bd[k-1])break;
 }
 for(int j=0;j<k;++j){out_d[q*k+j]=sqrt(bd[j]);out_i[q*k+j]=bi[j];}
}

extern "C" __global__ void pair_candidates(
 const double* world,const double* source_n,const bool* source_g,const double* source_q,
 const double* rotation,const double* distances,const long long* ids,
 const double* target,const double* normals,const double* quality,
 double road,double radius,double cosine,double patch,const long long* layout,
 int count,long long* best,double* best_d,long long* keys,int* bins) {
 int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=count)return;
 best[q]=-1;best_d[q]=__longlong_as_double(0x7ff0000000000000ULL);keys[q]=-1;bins[q]=0;
 if(source_q[q]<=0)return;
 double sx=rotation[0]*source_n[3*q]+rotation[1]*source_n[3*q+1]+rotation[2]*source_n[3*q+2];
 double sy=rotation[3]*source_n[3*q]+rotation[4]*source_n[3*q+1]+rotation[5]*source_n[3*q+2];
 double sz=rotation[6]*source_n[3*q]+rotation[7]*source_n[3*q+1]+rotation[8]*source_n[3*q+2];
 for(int j=0;j<8;++j){
  double d=distances[8*q+j];long long i=ids[8*q+j];
  if(d>=radius||quality[i]<=0)continue;
  double nx=normals[3*i],ny=normals[3*i+1],nz=normals[3*i+2];
  bool ground=fabs(nz)>.85&&fabs(target[3*i+2]-road)<.8;
  if(ground!=source_g[q]||fabs(nx*sx+ny*sy+nz*sz)<=cosine)continue;
  best[q]=i;best_d[q]=d;
  long long a=(long long)floor(target[3*i]/patch)-layout[0];
  long long b=(long long)floor(target[3*i+1]/patch)-layout[1];
  long long c=(long long)floor(target[3*i+2]/patch)-layout[2];
  int axis=0;double dominant=fabs(nx);
  if(fabs(ny)>dominant){axis=1;dominant=fabs(ny);}
  if(fabs(nz)>dominant)axis=2;
  keys[q]=((a*layout[3]+b)*layout[4]+c)*3+axis;bins[q]=axis;
  break;
 }
}

extern "C" __global__ void patch_winners(
 const long long* keys,const double* distance,int count,int capacity,
 unsigned long long* table_keys,int* table_values){
 int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=count||keys[q]<0)return;
 unsigned long long key=(unsigned long long)keys[q];
 unsigned int slot=(unsigned int)((key*11400714819323198485ULL)>>(64-24));
 slot=slot&(capacity-1);
 for(int probe=0;probe<capacity;++probe,slot=(slot+1)&(capacity-1)){
  unsigned long long found=atomicCAS(table_keys+slot,~0ULL,key);
  if(found!=~0ULL&&found!=key)continue;
  int old=table_values[slot];
  while(old<0||distance[q]<distance[old]||(distance[q]==distance[old]&&q<old)){
   int seen=atomicCAS(table_values+slot,old,q);
   if(seen==old)break;old=seen;
  }
  break;
 }
}
'''


class CudaKernels:
    def __init__(self, cache_dir):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; no CPU fallback')
        self.cache = Path(cache_dir).resolve()
        gpu_root = Path(__file__).resolve().parents[1]
        self.cache.relative_to((gpu_root/'outputs/run').resolve())
        self.cache.mkdir(parents=True, exist_ok=True)
        torch.empty(1, device='cuda')
        lib = Path(torch.__file__).resolve().parents[1]/'nvidia/cuda_nvrtc/lib/libnvrtc.so.12'
        self.nvrtc = C.CDLL(str(lib)); self.driver = C.CDLL('libcuda.so.1')
        self._bind()
        arch = ''.join(map(str, torch.cuda.get_device_capability()))
        options = [b'--std=c++14', ('--gpu-architecture=sm_'+arch).encode(), b'--fmad=false']
        version = (C.c_int(), C.c_int())
        self.check(self.nvrtc.nvrtcVersion(C.byref(version[0]),C.byref(version[1])), 'nvrtcVersion')
        identity={'source_sha256':hashlib.sha256(CUDA_SOURCE.encode()).hexdigest(),
                  'options':[x.decode() for x in options], 'nvrtc_version':[x.value for x in version],
                  'torch':torch.__version__, 'device':torch.cuda.get_device_name(), 'float':'float64'}
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        binary=self.cache/(key+'.cubin')
        if not binary.is_file():
            program=C.c_void_p()
            self.check(self.nvrtc.nvrtcCreateProgram(C.byref(program),CUDA_SOURCE.encode(),b'grid_knn.cu',0,None,None),'nvrtcCreateProgram')
            try:
                status=self.nvrtc.nvrtcCompileProgram(program,len(options),(C.c_char_p*len(options))(*options))
                size=C.c_size_t();self.nvrtc.nvrtcGetProgramLogSize(program,C.byref(size))
                log=C.create_string_buffer(size.value);self.nvrtc.nvrtcGetProgramLog(program,log)
                (self.cache/(key+'.compile.log')).write_bytes(log.value)
                self.check(status,'nvrtcCompileProgram: '+log.value.decode())
                self.check(self.nvrtc.nvrtcGetCUBINSize(program,C.byref(size)),'nvrtcGetCUBINSize')
                code=C.create_string_buffer(size.value)
                self.check(self.nvrtc.nvrtcGetCUBIN(program,code),'nvrtcGetCUBIN')
                binary.write_bytes(code.raw)
            finally:self.nvrtc.nvrtcDestroyProgram(C.byref(program))
        (self.cache/(key+'.json')).write_text(json.dumps(identity,indent=2)+'\n')
        self.module=C.c_void_p();self.functions={}
        self.code=C.create_string_buffer(binary.read_bytes())
        self.check(self.driver.cuModuleLoadData(C.byref(self.module),self.code),'cuModuleLoadData')
        for name in ('grid_knn','pair_candidates','patch_winners'):
            function=C.c_void_p()
            self.check(self.driver.cuModuleGetFunction(C.byref(function),self.module,name.encode()),'cuModuleGetFunction')
            self.functions[name]=function
        self.identity=identity

    def _bind(self):
        declarations={
            'nvrtcVersion':([C.POINTER(C.c_int),C.POINTER(C.c_int)],C.c_int),
            'nvrtcCreateProgram':([C.POINTER(C.c_void_p),C.c_char_p,C.c_char_p,C.c_int,C.c_void_p,C.c_void_p],C.c_int),
            'nvrtcCompileProgram':([C.c_void_p,C.c_int,C.POINTER(C.c_char_p)],C.c_int),
            'nvrtcGetProgramLogSize':([C.c_void_p,C.POINTER(C.c_size_t)],C.c_int),
            'nvrtcGetProgramLog':([C.c_void_p,C.c_void_p],C.c_int),
            'nvrtcGetCUBINSize':([C.c_void_p,C.POINTER(C.c_size_t)],C.c_int),
            'nvrtcGetCUBIN':([C.c_void_p,C.c_void_p],C.c_int),
            'nvrtcDestroyProgram':([C.POINTER(C.c_void_p)],C.c_int)}
        for name,(args,result) in declarations.items():
            f=getattr(self.nvrtc,name);f.argtypes=args;f.restype=result
        self.driver.cuModuleLoadData.argtypes=[C.POINTER(C.c_void_p),C.c_void_p]
        self.driver.cuModuleGetFunction.argtypes=[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p]
        self.driver.cuLaunchKernel.argtypes=[C.c_void_p]+[C.c_uint]*7+[C.c_void_p,C.POINTER(C.c_void_p),C.c_void_p]

    @staticmethod
    def check(status, operation):
        if status:raise RuntimeError(f'{operation}: CUDA/NVRTC error {status}; no CPU fallback')

    def query(self, tree, queries, k, radius):
        if not 1 <= k <= 25:raise ValueError('CUDA k must be in [1,25]')
        count=len(queries)
        distance=torch.empty((count,k),device='cuda',dtype=torch.float64)
        indices=torch.empty((count,k),device='cuda',dtype=torch.int64)
        if count:
            arguments=[C.c_void_p(tree.points.data_ptr()),C.c_void_p(tree.order.data_ptr()),C.c_void_p(tree.offsets.data_ptr()),
                C.c_double(tree.leaf),*[C.c_double(v) for v in tree.low],*[C.c_int(v) for v in tree.shape],
                C.c_void_p(queries.data_ptr()),C.c_int(count),C.c_int(k),C.c_double(radius),
                C.c_void_p(distance.data_ptr()),C.c_void_p(indices.data_ptr())]
            self.launch('grid_knn',count,arguments)
        return distance,indices

    def launch(self,name,count,arguments):
        if not count:return
        pointers=(C.c_void_p*len(arguments))(*(C.cast(C.pointer(v),C.c_void_p) for v in arguments))
        self.check(self.driver.cuLaunchKernel(self.functions[name],(count+63)//64,1,1,64,1,1,0,
            C.c_void_p(torch.cuda.current_stream().cuda_stream),pointers,None),name+' launch')


class CudaGrid:
    def __init__(self, points, kernels, leaf=1.5):
        self.kernels=kernels;self.leaf=float(leaf)
        self.cpu=np.ascontiguousarray(points,dtype=np.float64)
        if self.cpu.ndim!=2 or self.cpu.shape[1]!=3 or not len(self.cpu) or not np.isfinite(self.cpu).all():
            raise ValueError('CUDA grid requires finite nonempty XYZ')
        self.points=torch.as_tensor(self.cpu,device='cuda')
        self.low=(np.floor(self.cpu.min(axis=0)/leaf)*leaf-leaf).tolist()
        coords=torch.floor((self.points-torch.tensor(self.low,device='cuda'))/leaf).to(torch.int64)
        self.shape=(coords.max(dim=0).values+1).cpu().tolist()
        cells=int(np.prod(self.shape))
        if cells>32_000_000:raise ValueError('Dense CUDA grid exceeds explicit memory budget')
        keys=(coords[:,0]*self.shape[1]+coords[:,1])*self.shape[2]+coords[:,2]
        self.order=torch.argsort(keys,stable=True)
        counts=torch.bincount(keys,minlength=cells)
        self.offsets=torch.cat([torch.zeros(1,device='cuda',dtype=torch.int64),counts.cumsum(0)])
        self.boundary_refinements=0
        self.cpu_tree=None

    def query_gpu(self, points, k, radius):
        points=points.contiguous()
        if points.dtype!=torch.float64 or not points.is_cuda:raise ValueError('CUDA kNN requires CUDA float64 XYZ')
        # One extra neighbor detects rank-cutoff ties. Resolve only those with
        # the original exact CPU query, preserving the reference tie behavior.
        distance,ids=self.kernels.query(self,points,min(k+1,25),radius)
        if k<25:
            tied=torch.isfinite(distance[:,k]) & (abs(distance[:,k]-distance[:,k-1])<=1e-11)
            tie_ids=tied.nonzero().ravel()
            if len(tie_ids):
                if self.cpu_tree is None:self.cpu_tree=cKDTree(self.cpu)
                d,i=self.cpu_tree.query(points[tie_ids].cpu().numpy(),k=k,workers=1)
                d,i=np.asarray(d).reshape(-1,k),np.asarray(i).reshape(-1,k)
                d=np.where(d<=radius,d,np.inf);i=np.where(np.isfinite(d),i,0)
                distance[tie_ids,:k]=torch.as_tensor(d,device='cuda')
                ids[tie_ids,:k]=torch.as_tensor(i,device='cuda')
                self.boundary_refinements+=len(tie_ids)
        return distance[:,:k].contiguous(),ids[:,:k].contiguous()

    def query(self, points, k=24, workers=1):
        d,i=self.query_gpu(torch.as_tensor(np.ascontiguousarray(points),device='cuda',dtype=torch.float64),k,2.5)
        return d.cpu().numpy(),i.cpu().numpy()

    def query_ball_point(self, point, radius):
        q=torch.tensor(point,device='cuda',dtype=torch.float64)
        return ((self.points-q).square().sum(dim=1)<=radius**2).nonzero().ravel().cpu().tolist()


def _normal_eigh(covariance):
    """Same per-point symmetric eigensolve with bounded CUDA workspace."""
    eigen=torch.empty((len(covariance),3),device=covariance.device,dtype=covariance.dtype)
    vectors=torch.empty_like(covariance)
    for start in range(0,len(covariance),NORMAL_EIGH_BATCH_SIZE):
        stop=min(start+NORMAL_EIGH_BATCH_SIZE,len(covariance))
        eigen[start:stop],vectors[start:stop]=torch.linalg.eigh(covariance[start:stop])
    return eigen,vectors


def cuda_normals(points, tree, indices, radius=2.5, *, tensors=False):
    ids=torch.as_tensor(np.ascontiguousarray(indices),device='cuda',dtype=torch.int64) if not isinstance(indices,torch.Tensor) else indices
    d,nn=tree.query_gpu(tree.points[ids],min(24,len(tree.points)),radius)
    local=tree.points[nn]-tree.points[ids,None]
    valid=d<=radius;count=valid.sum(dim=1);safe=count.clamp_min(1)
    mean=(local*valid[...,None]).sum(dim=1)/safe[:,None]
    centered=(local-mean[:,None])*valid[...,None]
    covariance=centered.transpose(1,2)@centered/safe[:,None,None]
    eigen,vectors=_normal_eigh(covariance)
    quality=(eigen[:,1]-eigen[:,0])/eigen[:,2].clamp_min(1e-12)
    good=(count>=8)&(eigen[:,1]>.015)&(eigen[:,0]<.3*eigen[:,1])&(quality>.15)
    normal=vectors[:,:,0];quality=quality*good
    if tensors:return normal,quality
    return normal.cpu().numpy(),quality.cpu().numpy()


class CudaLayeredMap:
    def __init__(self, points, config=None, *, cache_dir):
        self.config=config or reference.ScaleICPConfig()
        self.points=np.asarray(points,float)
        if self.points.ndim!=2 or self.points.shape[1]!=3 or len(self.points)<64 or not np.isfinite(self.points).all():
            raise ValueError('Layered map requires at least 64 finite XYZ points')
        self.kernels=CudaKernels(cache_dir)
        self.tree=CudaGrid(self.points,self.kernels)
        self.gpu_normals=torch.zeros_like(self.tree.points)
        self.gpu_quality=torch.zeros(len(points),device='cuda',dtype=torch.float64)
        self.computed=torch.zeros(len(points),device='cuda',dtype=torch.bool)
        self.source_cache=None
        patch=np.floor(self.points/self.config.patch_m).astype(np.int64)
        low=patch.min(axis=0);shape=patch.max(axis=0)-low+1
        if int(shape[0])*int(shape[1])*int(shape[2])*3>=2**63:raise ValueError('Patch identity exceeds int64')
        self.patch_layout=torch.tensor([*low,shape[1],shape[2]],device='cuda',dtype=torch.int64)

    def ensure_surfaces(self, ids):
        pending=ids[~self.computed[ids]].unique()
        for start in range(0,len(pending),8192):
            selected=pending[start:start+8192]
            normal,quality=cuda_normals(None,self.tree,selected,tensors=True)
            self.gpu_normals[selected]=normal;self.gpu_quality[selected]=quality;self.computed[selected]=True

    def surfaces(self, indices):
        ids=torch.as_tensor(np.ascontiguousarray(indices),device='cuda',dtype=torch.int64)
        self.ensure_surfaces(ids)
        return self.gpu_normals[ids].cpu().numpy(),self.gpu_quality[ids].cpu().numpy()

    road_height=reference.LayeredScaleMap.road_height


def cuda_pairs(scene,raw,source_normals,source_ground,source_quality,pose,scale,road_z,radius,config):
    # Keep all source references alive and compare identity only: distinct seed
    # geometry can share raw XYZ while requiring different normals/quality.
    source_objects=(raw,source_normals,source_ground,source_quality)
    if (scene.source_cache is None
            or any(cached is not current for cached,current in zip(scene.source_cache[0],source_objects))):
        scene.source_cache=(source_objects,*[torch.as_tensor(np.ascontiguousarray(x),device='cuda')
            for x in source_objects])
    _,p,sn,sg,sq=scene.source_cache
    rotation=torch.tensor(pose[:3,:3],device='cuda');translation=torch.tensor(pose[:3,3],device='cuda')
    rotated=scale*p@rotation.T;world=rotated+translation
    d,ids=scene.tree.query_gpu(world,8,radius)
    scene.ensure_surfaces(ids.ravel())
    target_ids=torch.empty(len(p),device='cuda',dtype=torch.int64)
    distance=torch.empty(len(p),device='cuda',dtype=torch.float64)
    keys=torch.empty_like(target_ids);bins=torch.empty(len(p),device='cuda',dtype=torch.int32)
    pointer=lambda x:C.c_void_p(x.data_ptr())
    scene.kernels.launch('pair_candidates',len(p),[
        *map(pointer,(world,sn,sg,sq,rotation,d,ids,scene.tree.points,scene.gpu_normals,scene.gpu_quality)),
        C.c_double(road_z),C.c_double(radius),C.c_double(np.cos(np.deg2rad(55.))),C.c_double(config.patch_m),
        pointer(scene.patch_layout),C.c_int(len(p)),*map(pointer,(target_ids,distance,keys,bins))])
    capacity=1<<(max(1,2*len(p))-1).bit_length()
    table_keys=torch.full((capacity,),-1,device='cuda',dtype=torch.int64)
    table_values=torch.full((capacity,),-1,device='cuda',dtype=torch.int32)
    scene.kernels.launch('patch_winners',len(p),[pointer(keys),pointer(distance),C.c_int(len(p)),C.c_int(capacity),pointer(table_keys),pointer(table_values)])
    winners=table_values[table_values>=0].to(torch.int64).sort().values
    winners=winners[torch.argsort(distance[winners],stable=True)]
    selected=[]
    for b in range(3):
        unique=winners[bins[winners]==b]
        if len(unique)>config.max_pairs_per_bin:
            take=np.linspace(0,len(unique)-1,config.max_pairs_per_bin,dtype=int)
            unique=unique[torch.tensor(take,device='cuda')]
        selected.append(unique)
    chosen=torch.cat(selected) if selected else torch.empty(0,device='cuda',dtype=torch.int64)
    target=scene.tree.points[target_ids[chosen]]
    packed=torch.cat([rotated[chosen],world[chosen],target,scene.gpu_normals[target_ids[chosen]],
        torch.minimum(scene.gpu_quality[target_ids[chosen]],sq[chosen])[:,None],
        chosen.to(torch.float64)[:,None],sg[chosen,None].to(torch.float64)],dim=1).cpu().numpy()
    result={'rotated':packed[:,:3],'world':packed[:,3:6],'target':packed[:,6:9],'normal':packed[:,9:12],
        'quality':packed[:,12],'source_index':packed[:,13].astype(int),'ground':packed[:,14].astype(bool),
        'patch':np.floor(packed[:,6:9]/config.patch_m).astype(np.int64)}
    result['holdout']=reference.heldout_patches(result['patch'])
    return result


class CudaRegistrar:
    """Same reference fits/checks, with per-instance GPU geometry bindings."""
    def __init__(self, points, config=None, *, cache_dir):
        self.scene=CudaLayeredMap(points,config,cache_dir=cache_dir)
        namespace=dict(reference.__dict__)
        namespace['cKDTree']=lambda points:CudaGrid(points,self.scene.kernels)
        namespace['normals_at']=cuda_normals
        namespace['pairs_at']=cuda_pairs
        for name in ('source_geometry','fit_seed','register_frame'):
            original=getattr(reference,name)
            namespace[name]=types.FunctionType(original.__code__,namespace,name,original.__defaults__,original.__closure__)
        self.registration=namespace['register_frame']
        self.config=self.scene.config

    def register(self, raw, prior):
        result=self.registration(self.scene,raw,prior,self.config)
        result['compute_backend']='cuda_float64_grid_pca_balancing_cpu_small_solver'
        result['knn_boundary_refinements']=self.scene.tree.boundary_refinements
        return result

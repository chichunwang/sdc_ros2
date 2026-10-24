#include "track3_rewrite/core.hpp"
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/binary_search.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <regex>
#include <stdexcept>

namespace t3 {
namespace {
void check(cudaError_t error) {
  if (error!=cudaSuccess) throw std::runtime_error(std::string("CUDA failure; no fallback: ")+cudaGetErrorString(error));
}
template<class T> T* ptr(thrust::device_vector<T>& v) { return thrust::raw_pointer_cast(v.data()); }
template<class T> const T* ptr(const thrust::device_vector<T>& v) { return thrust::raw_pointer_cast(v.data()); }
struct Index {
  const Point* points;
  const long long* keys;
  const std::int64_t* order;
  const std::int64_t* offsets;
  Point low;
  double cell;
  int nx,ny,nz;
};
__host__ __device__ Point sub(Point a,Point b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
__host__ __device__ double dot(Point a,Point b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
__device__ void neighbors(Index grid,Point q,int k,double radius,std::int64_t* ids,double* dist) {
  for(int j=0;j<k;++j) { ids[j]=-1; dist[j]=INFINITY; }
  const double bx=fmax(fmax(grid.low.x-q.x,0.),q.x-grid.low.x-grid.nx*grid.cell);
  const double by=fmax(fmax(grid.low.y-q.y,0.),q.y-grid.low.y-grid.ny*grid.cell);
  const double bz=fmax(fmax(grid.low.z-q.z,0.),q.z-grid.low.z-grid.nz*grid.cell);
  if(bx*bx+by*by+bz*bz>radius*radius)return;
  int c[3]={int(floor((q.x-grid.low.x)/grid.cell)),int(floor((q.y-grid.low.y)/grid.cell)),int(floor((q.z-grid.low.z)/grid.cell))};
  const int reach=int(ceil(radius/grid.cell))+1;
  for(int shell=0;shell<=reach;++shell) {
    for(int a=c[0]-shell;a<=c[0]+shell;++a)
    for(int b=c[1]-shell;b<=c[1]+shell;++b)
    for(int z=c[2]-shell;z<=c[2]+shell;++z) {
      if(max(max(abs(a-c[0]),abs(b-c[1])),abs(z-c[2]))!=shell) continue;
      if(a<0 || b<0 || z<0 || a>=grid.nx || b>=grid.ny || z>=grid.nz) continue;
      const double lx=grid.low.x+a*grid.cell,ly=grid.low.y+b*grid.cell,lz=grid.low.z+z*grid.cell;
      const double dx=fmax(fmax(lx-q.x,0.),q.x-lx-grid.cell);
      const double dy=fmax(fmax(ly-q.y,0.),q.y-ly-grid.cell);
      const double dz=fmax(fmax(lz-q.z,0.),q.z-lz-grid.cell);
      if(dx*dx+dy*dy+dz*dz>fmin(radius*radius,dist[k-1])) continue;
      const long long key=(static_cast<long long>(a)*grid.ny+b)*grid.nz+z;
      for(std::int64_t pos=grid.offsets[key];pos<grid.offsets[key+1];++pos) {
        const std::int64_t id=grid.order[pos]; const Point delta=sub(q,grid.points[id]); const double d=dot(delta,delta);
        if(d>radius*radius || d>dist[k-1] || (d==dist[k-1] && id>=ids[k-1] && ids[k-1]>=0)) continue;
        int at=k-1;
        while(at>0 && (d<dist[at-1] || (d==dist[at-1] && id<ids[at-1]))) {
          dist[at]=dist[at-1]; ids[at]=ids[at-1]; --at;
        }
        dist[at]=d;ids[at]=id;
      }
    }
    // Unvisited cells lie outside the enclosing cube. Strict comparison keeps ties.
    double edge=fmin(q.x-(grid.low.x+(c[0]-shell)*grid.cell),grid.low.x+(c[0]+shell+1)*grid.cell-q.x);
    edge=fmin(edge,fmin(q.y-(grid.low.y+(c[1]-shell)*grid.cell),grid.low.y+(c[1]+shell+1)*grid.cell-q.y));
    edge=fmin(edge,fmin(q.z-(grid.low.z+(c[2]-shell)*grid.cell),grid.low.z+(c[2]+shell+1)*grid.cell-q.z));
    if(edge>=0. && edge*edge>dist[k-1]) break;
  }
}
__global__ void make_keys(const Point* p,int n,Point low,double leaf,int ny,int nz,long long* keys) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  const long long a=static_cast<long long>(floor((p[i].x-low.x)/leaf));
  const long long b=static_cast<long long>(floor((p[i].y-low.y)/leaf));
  const long long c=static_cast<long long>(floor((p[i].z-low.z)/leaf));
  keys[i]=(a*ny+b)*nz+c;
}
__global__ void normal_kernel(Index grid,int count,Point* normal,double* quality) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
  std::int64_t ids[24];double d[24];neighbors(grid,grid.points[i],24,2.5*grid.cell,ids,d);
  double mean[3]={},cov[3][3]={},v[3][3]={{1,0,0},{0,1,0},{0,0,1}};
  int n=0;
  for(int j=0;j<24 && ids[j]>=0;++j) {
    const Point delta=sub(grid.points[ids[j]],grid.points[i]);
    mean[0]+=delta.x;mean[1]+=delta.y;mean[2]+=delta.z;++n;
  }
  quality[i]=0.;normal[i]={0,0,0};if(n<6)return;
  for(double& x:mean)x/=n;
  for(int j=0;j<n;++j) {
    const Point delta=sub(grid.points[ids[j]],grid.points[i]);
    const double x[3]={delta.x-mean[0],delta.y-mean[1],delta.z-mean[2]};
    for(int a=0;a<3;++a)for(int b=0;b<3;++b)cov[a][b]+=x[a]*x[b]/n;
  }
  // Independent 3x3 Jacobi solves in registers. No cuSOLVER batch workspace.
  for(int iteration=0;iteration<20;++iteration) {
    int a=0,b=1;
    if(fabs(cov[0][2])>fabs(cov[a][b])){a=0;b=2;}
    if(fabs(cov[1][2])>fabs(cov[a][b])){a=1;b=2;}
    if(fabs(cov[a][b])<1e-15)break;
    const double angle=.5*atan2(2*cov[a][b],cov[b][b]-cov[a][a]);
    const double c=cos(angle),s=sin(angle);
    for(int row=0;row<3;++row) {
      const double ca=cov[row][a],cb=cov[row][b];cov[row][a]=c*ca-s*cb;cov[row][b]=s*ca+c*cb;
      const double va=v[row][a],vb=v[row][b];v[row][a]=c*va-s*vb;v[row][b]=s*va+c*vb;
    }
    for(int col=0;col<3;++col) {
      const double ca=cov[a][col],cb=cov[b][col];cov[a][col]=c*ca-s*cb;cov[b][col]=s*ca+c*cb;
    }
  }
  int axis[3]={0,1,2};
  for(int a=0;a<3;++a)for(int b=a+1;b<3;++b)if(cov[axis[a]][axis[a]]>cov[axis[b]][axis[b]]){const int t=axis[a];axis[a]=axis[b];axis[b]=t;}
  const double low=fmax(0.,cov[axis[0]][axis[0]]),middle=fmax(0.,cov[axis[1]][axis[1]]);
  if(middle<=1e-12)return;
  normal[i]={v[0][axis[0]],v[1][axis[0]],v[2][axis[0]]};
  quality[i]=pow(fmax(0.,1.-low/middle),2);
}
__global__ void nearest_kernel(Index grid,const Point* q,int count,double radius,std::int64_t* out) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
  std::int64_t id[1];double d[1];neighbors(grid,q[i],1,radius,id,d);out[i]=id[0];
}
struct Transform { double r[9],t[3],scale; };
__global__ void equations(Index grid,const Point* normal,const double* quality,const Point* source,int count,
                         Transform pose,double radius,bool plane,bool scale,double* out) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
  double* terms=out+i*60;for(int k=0;k<60;++k)terms[k]=0.;
  const Point p=source[i];
  const Point rotated={pose.scale*(pose.r[0]*p.x+pose.r[1]*p.y+pose.r[2]*p.z),
                       pose.scale*(pose.r[3]*p.x+pose.r[4]*p.y+pose.r[5]*p.z),
                       pose.scale*(pose.r[6]*p.x+pose.r[7]*p.y+pose.r[8]*p.z)};
  const Point world={rotated.x+pose.t[0],rotated.y+pose.t[1],rotated.z+pose.t[2]};
  std::int64_t ids[1];double distances[1];neighbors(grid,world,1,radius,ids,distances);
  const bool heldout=(i%5)==0;terms[heldout?1:0]=1.;
  if(ids[0]<0 || (plane && quality[ids[0]]<=1e-8))return;
  const Point error=sub(world,grid.points[ids[0]]);
  const double sigma=radius/3.;
  double residual[3]={error.x,error.y,error.z},jac[3][7]={{1,0,0,0,rotated.z,-rotated.y,rotated.x},
      {0,1,0,-rotated.z,0,rotated.x,rotated.y},{0,0,1,rotated.y,-rotated.x,0,rotated.z}};
  int rows=3;
  double confidence=1.;
  if(plane) {
    const Point n=normal[ids[0]];const double nv[3]={n.x,n.y,n.z};
    double projected[7]={};for(int k=0;k<7;++k)for(int a=0;a<3;++a)projected[k]+=nv[a]*jac[a][k];
    for(int k=0;k<7;++k)jac[0][k]=projected[k];
    residual[0]=dot(n,error);rows=1;confidence=quality[ids[0]];
  }
  double square=0.;for(int a=0;a<rows;++a)square+=residual[a]*residual[a];
  const double inlier=exp(-.5*square/(sigma*sigma));
  terms[heldout?1:0]=1.-confidence*inlier;
  terms[heldout?3:2]=1.;
  if(heldout)return;
  const double weight=confidence*inlier/(sigma*sigma);
  const int dimensions=scale?7:6;
  for(int a=0;a<dimensions;++a) {
    for(int row=0;row<rows;++row)terms[4+a]+=weight*jac[row][a]*residual[row];
    for(int b=0;b<dimensions;++b)for(int row=0;row<rows;++row)terms[11+a*7+b]+=weight*jac[row][a]*jac[row][b];
  }
}
__global__ void sum_terms(const double* values,int count,double* sums) {
  __shared__ double partial[256];
  const int column=blockIdx.x;double value=0.;
  for(int i=threadIdx.x;i<count;i+=blockDim.x)value+=values[i*60+column];
  partial[threadIdx.x]=value;__syncthreads();
  for(int stride=128;stride;stride/=2){if(threadIdx.x<stride)partial[threadIdx.x]+=partial[threadIdx.x+stride];__syncthreads();}
  if(threadIdx.x==0)sums[column]=partial[0];
}
struct Box {
  const float* p; Point center;double xy,z;
  __device__ bool operator()(std::int64_t i)const {
    return fabs(double(p[3*i])-center.x)<=xy && fabs(double(p[3*i+1])-center.y)<=xy && fabs(double(p[3*i+2])-center.z)<=z;
  }
};
__global__ void extract(const float* source,const std::int64_t* ids,int count,Point* points) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;std::int64_t j=ids[i];points[i]={double(source[3*j]),double(source[3*j+1]),double(source[3*j+2])};
}
} // namespace
struct Geometry::Impl {
  thrust::device_vector<Point> points,normal;
  thrust::device_vector<double> quality;
  thrust::device_vector<long long> keys;
  thrust::device_vector<std::int64_t> order,offsets;
  Index index;
  Impl(const Cloud& cloud,double cell,bool compute_normals):points(cloud.begin(),cloud.end()),normal(cloud.size()),quality(cloud.size()),keys(cloud.size()),order(cloud.size()) {
    if(cloud.empty() || cloud.size()>2147483647ULL || cell<=0. || !std::isfinite(cell))throw std::invalid_argument("Invalid geometry");
    Point low=cloud[0],high=cloud[0];
    for(const auto& p:cloud) {
      if(!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.z))throw std::invalid_argument("Nonfinite XYZ");
      low={std::min(low.x,p.x),std::min(low.y,p.y),std::min(low.z,p.z)};
      high={std::max(high.x,p.x),std::max(high.y,p.y),std::max(high.z,p.z)};
    }
    low={std::floor(low.x/cell)*cell,std::floor(low.y/cell)*cell,std::floor(low.z/cell)*cell};
    const int nx=int(std::floor((high.x-low.x)/cell))+1,ny=int(std::floor((high.y-low.y)/cell))+1,nz=int(std::floor((high.z-low.z)/cell))+1;
    const long long cells=static_cast<long long>(nx)*ny*nz;
    if(cells<=0 || cells>32000000)throw std::runtime_error("Geometry grid exceeds explicit 32M-cell bound");
    make_keys<<<(cloud.size()+255)/256,256>>>(ptr(points),cloud.size(),low,cell,ny,nz,ptr(keys));check(cudaGetLastError());
    thrust::sequence(order.begin(),order.end());thrust::stable_sort_by_key(keys.begin(),keys.end(),order.begin());
    offsets.resize(cells+1);
    thrust::lower_bound(keys.begin(),keys.end(),thrust::make_counting_iterator<long long>(0),thrust::make_counting_iterator<long long>(cells+1),offsets.begin());
    index={ptr(points),ptr(keys),ptr(order),ptr(offsets),low,cell,nx,ny,nz};
    if(compute_normals) {normal_kernel<<<(cloud.size()+255)/256,256>>>(index,cloud.size(),ptr(normal),ptr(quality));check(cudaGetLastError());}
    synchronize();
  }
};
Geometry::Geometry(const Cloud& target,double cell,bool normals):impl_(std::make_unique<Impl>(target,cell,normals)){}
Geometry::~Geometry()=default;
Evaluation Geometry::evaluate(const Cloud& raw,const Pose& pose,double radius,Metric metric,bool scale)const {
  if(!pose.finite()||raw.empty()||radius<=0.)throw std::invalid_argument("Invalid evaluation");
  thrust::device_vector<Point> source(raw.begin(),raw.end());
  thrust::device_vector<double> terms(raw.size()*60),sums(60);
  Transform transform;for(int a=0;a<3;++a){transform.t[a]=pose.translation[a];for(int b=0;b<3;++b)transform.r[a*3+b]=pose.rotation(a,b);}transform.scale=pose.scale;
  equations<<<(raw.size()+255)/256,256>>>(impl_->index,ptr(impl_->normal),ptr(impl_->quality),ptr(source),raw.size(),transform,radius,metric==Metric::Plane,scale,ptr(terms));check(cudaGetLastError());
  sum_terms<<<60,256>>>(ptr(terms),raw.size(),ptr(sums));check(cudaGetLastError());
  double host[60];check(cudaMemcpy(host,ptr(sums),sizeof(host),cudaMemcpyDeviceToHost));
  const int validation=(raw.size()+4)/5,training=raw.size()-validation;
  Evaluation result;result.training_cost=host[0]/std::max(1,training);result.validation_cost=host[1]/std::max(1,validation);
  result.training_support=int(host[2]);result.validation_support=int(host[3]);
  for(int a=0;a<7;++a){result.gradient[a]=host[4+a];for(int b=0;b<7;++b)result.h(a,b)=host[11+a*7+b];}
  return result;
}
std::vector<std::int64_t> Geometry::nearest(const Cloud& queries,double radius)const {
  thrust::device_vector<Point> input(queries.begin(),queries.end());thrust::device_vector<std::int64_t> output(queries.size());
  nearest_kernel<<<(queries.size()+255)/256,256>>>(impl_->index,ptr(input),queries.size(),radius,ptr(output));check(cudaGetLastError());
  std::vector<std::int64_t> result(queries.size());thrust::copy(output.begin(),output.end(),result.begin());return result;
}
std::vector<Point> Geometry::normals()const {Cloud result(impl_->normal.size());thrust::copy(impl_->normal.begin(),impl_->normal.end(),result.begin());return result;}
Cloud voxel(const Cloud& cloud,double leaf) {
  if(cloud.empty())return {};
  if(!std::isfinite(leaf)||leaf<=0.)throw std::invalid_argument("Expected positive voxel size");
  // Stable sort keys then keep the first original row in each occupied cell.
  thrust::device_vector<Point> source(cloud.begin(),cloud.end());
  Point low=cloud[0],high=cloud[0];
  for(auto p:cloud){low={std::min(low.x,p.x),std::min(low.y,p.y),std::min(low.z,p.z)};high={std::max(high.x,p.x),std::max(high.y,p.y),std::max(high.z,p.z)};}
  low={std::floor(low.x/leaf)*leaf,std::floor(low.y/leaf)*leaf,std::floor(low.z/leaf)*leaf};
  const int ny=int(std::floor((high.y-low.y)/leaf))+1,nz=int(std::floor((high.z-low.z)/leaf))+1;
  thrust::device_vector<long long> keys(cloud.size());thrust::device_vector<std::int64_t> order(cloud.size());thrust::sequence(order.begin(),order.end());
  make_keys<<<(cloud.size()+255)/256,256>>>(ptr(source),cloud.size(),low,leaf,ny,nz,ptr(keys));check(cudaGetLastError());
  thrust::stable_sort_by_key(keys.begin(),keys.end(),order.begin());
  auto end=thrust::unique_by_key(keys.begin(),keys.end(),order.begin());order.resize(end.second-order.begin());
  thrust::sort(order.begin(),order.end());
  std::vector<std::int64_t> keep(order.size());thrust::copy(order.begin(),order.end(),keep.begin());
  Cloud result;result.reserve(keep.size());for(std::int64_t i:keep)result.push_back(cloud[i]);return result;
}
struct StaticMap::Impl {thrust::device_vector<float> xyz;std::size_t count=0;};
StaticMap::StaticMap(const std::string& path):impl_(std::make_unique<Impl>()) {
  std::ifstream stream(path,std::ios::binary);char magic[8];stream.read(magic,8);
  if(!stream || std::string(magic,6)!=std::string("\x93NUMPY",6) || magic[6]!=1)throw std::runtime_error("Expected NPY v1 XYZ cache");
  unsigned char length[2];stream.read(reinterpret_cast<char*>(length),2);std::string header(length[0]+256*length[1],' ');stream.read(header.data(),header.size());
  if(header.find("'<f4'")==std::string::npos || header.find("False")==std::string::npos)throw std::runtime_error("Expected row-major float32 map cache");
  std::smatch match;const std::regex shape("'shape': \\(\\s*([0-9]+),\\s*3\\s*\\)");
  if(!std::regex_search(header,match,shape))throw std::runtime_error("Expected N x 3 XYZ map cache");
  impl_->count=std::stoull(match[1]);
  if(impl_->count==0 || impl_->count>2147483647ULL)throw std::runtime_error("Map count exceeds explicit kernel launch bound");
  impl_->xyz.resize(impl_->count*3);
  std::vector<float> chunk(262144*3);std::size_t loaded=0;
  while(loaded<impl_->count) {
    const auto n=std::min<std::size_t>(262144,impl_->count-loaded);stream.read(reinterpret_cast<char*>(chunk.data()),n*3*sizeof(float));
    if(!stream)throw std::runtime_error("Truncated map cache");
    for(std::size_t i=0;i<n*3;++i)if(!std::isfinite(chunk[i]))throw std::runtime_error("Nonfinite map XYZ");
    check(cudaMemcpy(ptr(impl_->xyz)+loaded*3,chunk.data(),n*3*sizeof(float),cudaMemcpyHostToDevice));loaded+=n;
  }
  if(stream.peek()!=std::char_traits<char>::eof())throw std::runtime_error("Extra map cache payload");
}
StaticMap::~StaticMap()=default;
std::size_t StaticMap::size()const{return impl_->count;}
Cloud StaticMap::crop(const Eigen::Vector3d& center,double xy,double z,double leaf)const {
  thrust::device_vector<std::int64_t> selected(impl_->count);
  auto end=thrust::copy_if(thrust::device,thrust::make_counting_iterator<std::int64_t>(0),thrust::make_counting_iterator<std::int64_t>(impl_->count),selected.begin(),Box{ptr(impl_->xyz),{center.x(),center.y(),center.z()},xy,z});
  const auto n=end-selected.begin();if(n==0)throw std::runtime_error("Empty map crop");
  thrust::device_vector<Point> points(n);extract<<<(n+255)/256,256>>>(ptr(impl_->xyz),ptr(selected),n,ptr(points));check(cudaGetLastError());
  Cloud cloud(n);thrust::copy(points.begin(),points.end(),cloud.begin());return voxel(cloud,leaf);
}
void synchronize(){check(cudaDeviceSynchronize());}
std::array<std::size_t,2> memory_info(){std::size_t free,total;check(cudaMemGetInfo(&free,&total));return {free,total};}
} // namespace t3

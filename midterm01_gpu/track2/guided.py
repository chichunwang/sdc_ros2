"""5839746c RefineSensorGuided / SurfaceGeometryCache 的 CUDA 向量實作。

保留固定 scale、4-DOF、有界 prior、Huber、方向性 metric、15 次 Newton
與 8 次線搜尋、過去 correction 的前向 low-pass。H/g 累加移至 CUDA。
"""
import numpy as np
import cupy as cp
from scipy.spatial.transform import Rotation
from track3.gpu import voxel_mean
from track3.nearest import GpuNearest
from . import parameters as p


KNN = cp.RawKernel(r'''
extern "C" __global__ void radius_knn(const double* queries, int nq, const double* points,
 const long long* key,const long long* start,const long long* end,int nc,
 double cell,double radius,int k,long long* result) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nq)return;
 double best[16];long long ids[16];for(int j=0;j<k;j++){best[j]=radius*radius+1e-12;ids[j]=-1;}
 double x=queries[3*i],y=queries[3*i+1],z=queries[3*i+2];
 long long cx=floor(x/cell),cy=floor(y/cell),cz=floor(z/cell);int r=ceil(radius/cell);
 for(int dx=-r;dx<=r;dx++)for(int dy=-r;dy<=r;dy++)for(int dz=-r;dz<=r;dz++){
  long long xx=cx+dx+1048576,yy=cy+dy+1048576,zz=cz+dz+1048576;
  long long code=(xx<<42)|(yy<<21)|zz;int lo=0,hi=nc;
  while(lo<hi){int m=(lo+hi)/2;if(key[m]<code)lo=m+1;else hi=m;}
  if(lo>=nc||key[lo]!=code)continue;
  for(long long j=start[lo];j<end[lo];j++){
   double a=x-points[3*j],b=y-points[3*j+1],c=z-points[3*j+2],d=a*a+b*b+c*c;
   if(d>radius*radius+1e-12)continue;
   if(d<best[k-1] || (d==best[k-1] && (ids[k-1]<0||j<ids[k-1]))){
    int at=k-1;while(at>0 && (d<best[at-1] || (d==best[at-1] && (ids[at-1]<0||j<ids[at-1])))){
     best[at]=best[at-1];ids[at]=ids[at-1];at--;
    }best[at]=d;ids[at]=j;
   }
  }
 }
 for(int j=0;j<k;j++)result[i*k+j]=ids[j];
}
''', 'radius_knn')


class Geometry:
    def __init__(self, nn):
        self.nn = nn; n = len(nn.points)
        self.done = cp.zeros(n, dtype=cp.bool_)
        self.reliable = cp.zeros(n, dtype=cp.bool_)
        self.confidence = cp.zeros(n, dtype=cp.float64)
        self.normal = cp.zeros((n, 3), dtype=cp.float64)
        self.metric = cp.zeros((n, 3, 3), dtype=cp.float64)

    def ensure(self, ids):
        ids = cp.unique(ids[ids >= 0]); ids = ids[~self.done[ids]]
        if not len(ids):
            return
        query = cp.ascontiguousarray(self.nn.points[ids]); k = p.GEOMETRY_NEIGHBORS
        if not 6 <= k <= 16:
            raise ValueError('CUDA geometry k must be 6..16')
        neighbors = cp.empty((len(ids), k), dtype=cp.int64)
        KNN(((len(ids)+127)//128,), (128,), (query, np.int32(len(ids)), self.nn.points,
            self.nn.key, self.nn.start, self.nn.end, np.int32(len(self.nn.key)),
            np.float64(self.nn.cell), np.float64(p.GEOMETRY_RADIUS_M), np.int32(k), neighbors))
        good = neighbors >= 0; count = good.sum(axis=1)
        local = self.nn.points[cp.maximum(neighbors, 0)]-query[:, None, :]
        local = cp.where(good[:, :, None], local, 0.)
        mean = cp.sum(local, axis=1)/cp.maximum(count, 1)[:, None]
        centered = cp.where(good[:, :, None], local-mean[:, None, :], 0.)
        cov = cp.einsum('nki,nkj->nij', centered, centered)/cp.maximum(count, 1)[:, None, None]
        values, vectors = cp.linalg.eigh(cov); values = cp.maximum(values, 0.)
        planarity = (values[:, 1]-values[:, 0])/cp.maximum(values[:, 2], 1e-30)
        reliable = ((count >= p.GEOMETRY_MIN_NEIGHBORS) & (values[:, 1] >= .04)
                    & (values[:, 2] >= 1e-8) & (values[:, 0] <= .25*values[:, 1])
                    & (planarity >= p.GEOMETRY_MIN_PLANARITY))
        normal = vectors[:, :, 0]
        outer = normal[:, :, None]*normal[:, None, :]
        self.done[ids] = True; self.reliable[ids] = reliable
        self.normal[ids] = normal; self.confidence[ids] = cp.clip(planarity, 0., 1.)
        self.metric[ids] = outer+p.GEOMETRY_TANGENT_WEIGHT*(cp.eye(3)-outer)


class GuidedRegistration:
    def __init__(self, xyz):
        self.xyz = xyz
        self.center = None
        self.nn = None
        self.previous_correction = np.zeros(4)

    def preprocess(self, raw):
        # Original PCL pipeline: fixed scale=3 then a 0.3 m voxel filter.
        metric = cp.asarray(raw, dtype=cp.float64)*p.PLY_SCALE
        metric = metric[cp.all(cp.isfinite(metric), axis=1)]
        if not len(metric):
            return metric
        result, _ = voxel_mean(metric.astype(cp.float32), p.INPUT_VOXEL_LEAF_M)
        return result.astype(cp.float32).astype(cp.float64)

    def set_submap(self, prior):
        center = cp.asarray(prior[:3, 3])
        mask = cp.all(cp.abs(self.xyz-center) <= cp.asarray(p.SUBMAP_HALF_EXTENT_M), axis=1)
        xyz = self.xyz[mask]
        self.center = prior[:3, 3].copy()
        self.nn = GpuNearest(xyz, cell=p.GEOMETRY_RADIUS_M) if len(xyz) else None
        self.geometry = Geometry(self.nn) if self.nn is not None else None

    def evaluate(self, source, prior, x, derivatives=False, diagnostic=False):
        precision = np.array([4., 4., 1., 1/np.deg2rad(3.)**2])
        R = Rotation.from_euler('z', x[3]).as_matrix() @ prior[:3, :3]
        rotated = source @ cp.asarray(R).T
        q = rotated+cp.asarray(prior[:3, 3]+x[:3])
        ids, d2 = self.nn.query(q, 2.)
        good = ids >= 0; safe = cp.maximum(ids, 0)
        self.geometry.ensure(ids)
        reliable = good & self.geometry.reliable[safe]
        r = q-self.nn.points[safe]
        length = cp.sqrt(cp.maximum(d2, 0.))
        metric = self.geometry.metric[safe]
        mr = cp.einsum('nij,nj->ni', metric, r)
        directional = cp.sqrt(cp.maximum(cp.sum(r*mr, axis=1), 0.))
        confidence = self.geometry.confidence[safe]
        loss = cp.where(directional <= .5, .5*directional**2, .5*(directional-.25))
        unmatched = .5*(2.-.25)
        loss = cp.where(reliable, unmatched-confidence*(unmatched-loss), unmatched)
        normalize = (1/.3**2)/len(source)
        cost = float(.5*x @ (precision*x)+normalize*p.GEOMETRY_NORMAL_GAIN*cp.sum(loss))
        H = np.diag(precision); g = precision*x
        J = cp.zeros((len(source), 3, 4), dtype=cp.float64)
        J[:, :, :3] = cp.eye(3)
        J[:, 0, 3] = -rotated[:, 1]; J[:, 1, 3] = rotated[:, 0]
        if derivatives:
            w = cp.where(reliable, normalize*p.GEOMETRY_NORMAL_GAIN*confidence*
                         cp.minimum(1., .5/cp.maximum(directional, 1e-12)), 0.)
            H += cp.asnumpy(cp.einsum('nai,nab,nbj,n->ij', J, metric, J, w))
            g += cp.asnumpy(cp.einsum('nai,na,n->i', J, mr, w))
        info = dict(cost=cost, rmse=float(cp.sqrt(cp.mean(cp.minimum(length, 2.)**2))),
                    inliers=int(good.sum()), reliable_points=int(reliable.sum()),
                    unique_targets=int(cp.unique(ids[reliable]).size), H=H, g=g)
        if diagnostic:
            n = self.geometry.normal[safe]
            plane = cp.sum(n*r, axis=1)
            nj = cp.einsum('na,nai->ni', n, J)*cp.asarray([1., 1., 1., np.deg2rad(3.)])
            w = cp.where(reliable, normalize*confidence*cp.minimum(1., .5/cp.maximum(cp.abs(plane), 1e-12)), 0.)
            data_H = cp.asnumpy((nj*w[:, None]).T @ nj)
            eig = np.maximum(np.linalg.eigvalsh(data_H), 0.)
            info.update(scaled_normal_information=data_H.tolist(), scaled_eigenvalues=eig.tolist(),
                        observable_rank=int((eig >= max(1e-4, .05*eig.max())).sum()),
                        pca_queries=int(self.geometry.done.sum()))
        return info

    @staticmethod
    def valid(x):
        return bool(np.isfinite(x).all() and np.linalg.norm(x[:2]) <= p.SENSOR_MAX_XY_CORRECTION_M
                    and abs(x[2]) <= 1.5 and abs(x[3]) <= np.deg2rad(p.SENSOR_MAX_YAW_CORRECTION_DEG))

    def refine(self, source, prior):
        previous = self.previous_correction.copy()
        self.previous_correction[:] = 0.
        if len(source) < 10:
            return prior.copy(), dict(accepted=False, reason='insufficient_input_points', correction=[0.]*4)
        self.set_submap(prior)
        if self.nn is None:
            return prior.copy(), dict(accepted=False, reason='no_map_candidates', correction=[0.]*4)
        baseline = self.evaluate(source, prior, np.zeros(4))
        x = np.zeros(4); carried = previous*.8
        if self.valid(carried) and self.evaluate(source, prior, carried)['cost'] < baseline['cost']:
            x = carried
        iterations = 0
        for it in range(15):
            cur = self.evaluate(source, prior, x, derivatives=True)
            if cur['inliers'] < 10 or cur['reliable_points'] < 10 or not np.isfinite(cur['cost']):
                break
            step = -np.linalg.solve(cur['H'], cur['g'])
            factor = min(1., .25/max(np.linalg.norm(step[:3]), 1e-12), np.deg2rad(2.)/max(abs(step[3]), 1e-12))
            step *= factor; accepted = False
            for _ in range(8):
                trial = x+step
                if self.valid(trial) and self.evaluate(source, prior, trial)['cost'] < cur['cost']:
                    x = trial; accepted = True; break
                step *= .5
            iterations = it+1
            if not accepted or (np.linalg.norm(step[:3]) < .001 and abs(step[3]) < .0001):
                break
        correction = (1-p.SENSOR_SMOOTHING_ALPHA)*previous+p.SENSOR_SMOOTHING_ALPHA*x
        final = self.evaluate(source, prior, correction) if self.valid(correction) else baseline
        if not self.valid(correction) or final['cost'] > baseline['cost']:
            correction = x; final = self.evaluate(source, prior, x)
        final = self.evaluate(source, prior, correction, diagnostic=True)
        reason = ('nonfinite_objective' if not np.isfinite(final['cost']) else
                  'insufficient_matches' if final['inliers'] < 10 else
                  'objective_increased' if final['cost'] > baseline['cost']+1e-9 else
                  'insufficient_reliable_geometry' if final['reliable_points'] < 10 else
                  'insufficient_unique_targets' if final['unique_targets'] < 6 else 'accepted')
        accepted = reason == 'accepted'
        T = prior.copy()
        if accepted:
            self.previous_correction = correction.copy()
            T[:3, :3] = Rotation.from_euler('z', correction[3]).as_matrix() @ prior[:3, :3]
            T[:3, 3] += correction[:3]
        result = {k: v for k, v in final.items() if k not in ('H', 'g')}
        result.update(accepted=accepted, reason=reason, prior_cost=baseline['cost'], prior_rmse=baseline['rmse'],
                      iterations=iterations, correction=self.previous_correction.tolist(),
                      model='5839746c_directional_4dof', backend='cuda_nn_pca_H_g')
        return T, result

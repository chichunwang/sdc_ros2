"""CUDA 點雲基本運算（CuPy）：體素降採樣、格點雜湊最近鄰、半徑鄰域共變異數。

全部在 GPU 上計算，沒有 CPU fallback：匯入 CuPy 失敗或沒有 CUDA 裝置時直接拋錯，不會靜默改用 CPU。
格點索引做法：把點依所在格子的 64-bit key 排序，查詢時對鄰近格子做二分搜尋後逐點比較（精確，不是近似）。
"""
import numpy as np
import cupy as cp

OFFSET = 1 << 20  # 每軸 21 bit、有號格子索引的偏移

_KERNEL_SOURCE = r'''
#define OFF 1048576LL
#define LIM 2097152LL

__device__ int find_cell(const long long* ckey, int ncell, long long k) {
    int lo = 0, hi = ncell;
    while (lo < hi) { int m = (lo + hi) / 2; if (ckey[m] < k) lo = m + 1; else hi = m; }
    return (lo < ncell && ckey[lo] == k) ? lo : -1;
}

extern "C" __global__
void nn_query(const double* q, const int nq, const double* p,
              const long long* ckey, const int* cstart, const int* cend, const int ncell,
              const double cell, const double radius, int* out_idx, double* out_d2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nq) return;
    double x = q[3*i], y = q[3*i+1], z = q[3*i+2];
    long long cx = (long long)floor(x / cell) + OFF;
    long long cy = (long long)floor(y / cell) + OFF;
    long long cz = (long long)floor(z / cell) + OFF;
    int r = (int)ceil(radius / cell);
    double best = radius * radius; int win = -1;
    for (int dx = -r; dx <= r; dx++) for (int dy = -r; dy <= r; dy++) for (int dz = -r; dz <= r; dz++) {
        long long xx = cx + dx, yy = cy + dy, zz = cz + dz;
        if (xx < 0 || yy < 0 || zz < 0 || xx >= LIM || yy >= LIM || zz >= LIM) continue;
        int c = find_cell(ckey, ncell, (xx << 42) | (yy << 21) | zz);
        if (c < 0) continue;
        for (int j = cstart[c]; j < cend[c]; j++) {
            double a = x - p[3*j], b = y - p[3*j+1], d = z - p[3*j+2];
            double d2 = a*a + b*b + d*d;
            if (d2 < best) { best = d2; win = j; }
        }
    }
    out_idx[i] = win; out_d2[i] = best;
}

// 半徑鄰域統計：count、相對查詢點的位移總和 (3)、位移外積總和 (xx,xy,xz,yy,yz,zz)
extern "C" __global__
void radius_stats(const double* q, const int nq, const double* p,
                  const long long* ckey, const int* cstart, const int* cend, const int ncell,
                  const double cell, const double radius, int* out_n, double* out_sum, double* out_sq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nq) return;
    double x = q[3*i], y = q[3*i+1], z = q[3*i+2];
    long long cx = (long long)floor(x / cell) + OFF;
    long long cy = (long long)floor(y / cell) + OFF;
    long long cz = (long long)floor(z / cell) + OFF;
    int r = (int)ceil(radius / cell);
    double r2 = radius * radius; int n = 0;
    double s0 = 0, s1 = 0, s2 = 0, xx_ = 0, xy_ = 0, xz_ = 0, yy_ = 0, yz_ = 0, zz_ = 0;
    for (int dx = -r; dx <= r; dx++) for (int dy = -r; dy <= r; dy++) for (int dz = -r; dz <= r; dz++) {
        long long xx = cx + dx, yy = cy + dy, zz = cz + dz;
        if (xx < 0 || yy < 0 || zz < 0 || xx >= LIM || yy >= LIM || zz >= LIM) continue;
        int c = find_cell(ckey, ncell, (xx << 42) | (yy << 21) | zz);
        if (c < 0) continue;
        for (int j = cstart[c]; j < cend[c]; j++) {
            double a = p[3*j] - x, b = p[3*j+1] - y, d = p[3*j+2] - z;
            if (a*a + b*b + d*d > r2) continue;
            n++; s0 += a; s1 += b; s2 += d;
            xx_ += a*a; xy_ += a*b; xz_ += a*d; yy_ += b*b; yz_ += b*d; zz_ += d*d;
        }
    }
    out_n[i] = n;
    out_sum[3*i] = s0; out_sum[3*i+1] = s1; out_sum[3*i+2] = s2;
    out_sq[6*i] = xx_; out_sq[6*i+1] = xy_; out_sq[6*i+2] = xz_;
    out_sq[6*i+3] = yy_; out_sq[6*i+4] = yz_; out_sq[6*i+5] = zz_;
}
'''
_MODULE = cp.RawModule(code=_KERNEL_SOURCE)
_NN = _MODULE.get_function('nn_query')
_STATS = _MODULE.get_function('radius_stats')
_BLOCK = 256


def sync():
    """等 GPU 工作做完（計時前呼叫，否則量到的只是排程時間）。"""
    cp.cuda.get_current_stream().synchronize()


def device_info():
    """回傳實際使用的 CUDA 裝置資訊；沒有 GPU 時 CuPy 會在這裡拋錯。"""
    dev = cp.cuda.Device()
    free, total = dev.mem_info
    return dict(device_id=int(dev.id), name=cp.cuda.runtime.getDeviceProperties(dev.id)['name'].decode(),
                compute_capability=str(dev.compute_capability), total_memory_mib=int(total // 2 ** 20),
                free_memory_mib=int(free // 2 ** 20), cupy=cp.__version__)


def cell_keys(points, cell):
    c = cp.floor(points / cell).astype(cp.int64) + OFFSET
    if c.size and (int(c.min()) < 0 or int(c.max()) >= 2 * OFFSET):
        raise ValueError('點超出格點索引範圍（|座標/格寬| 需小於 2^20）')
    return (c[:, 0] << 42) | (c[:, 1] << 21) | c[:, 2]


def voxel_downsample(points, leaf, extra=None):
    """體素降採樣：每個體素取所有點的重心。extra（例如顏色）以同樣方式平均。"""
    p = cp.asarray(points, dtype=cp.float64)
    if len(p) == 0:
        return (p, extra) if extra is not None else p
    keys = cell_keys(p, leaf)
    order = cp.argsort(keys)
    k = keys[order]
    first = cp.concatenate([cp.ones(1, dtype=cp.bool_), k[1:] != k[:-1]])
    group = cp.cumsum(first.astype(cp.int64)) - 1
    n = int(group[-1]) + 1
    counts = cp.bincount(group, minlength=n).astype(cp.float64)

    def mean(values):
        v = cp.asarray(values, dtype=cp.float64)[order]
        return cp.stack([cp.bincount(group, weights=v[:, a], minlength=n) for a in range(v.shape[1])], 1) / counts[:, None]

    out = mean(p)
    if extra is not None:
        return out, mean(extra)
    return out


class GridIndex:
    """精確的 GPU 最近鄰／半徑鄰域查詢。建立後 self.points 是依格子排序過的點（self.order 是原索引）。"""

    def __init__(self, points, cell):
        p = cp.asarray(points, dtype=cp.float64)
        if len(p) == 0:
            raise ValueError('GridIndex 需要至少一個點')
        keys = cell_keys(p, cell)
        order = cp.argsort(keys)
        k = keys[order]
        first = cp.flatnonzero(cp.concatenate([cp.ones(1, dtype=cp.bool_), k[1:] != k[:-1]]))
        self.order = order
        self.points = cp.ascontiguousarray(p[order])
        self.cell_key = cp.ascontiguousarray(k[first])
        self.cell_start = cp.ascontiguousarray(first.astype(cp.int32))
        self.cell_end = cp.ascontiguousarray(cp.concatenate([first[1:], cp.asarray([len(p)])]).astype(cp.int32))
        self.cell = float(cell)

    def _args(self, q):
        return (q, np.int32(len(q)), self.points, self.cell_key, self.cell_start, self.cell_end,
                np.int32(len(self.cell_key)), np.float64(self.cell))

    def nearest(self, query, radius):
        """每個查詢點在 radius 內最近的點：回傳 (idx, d2)；找不到時 idx = -1。idx 指向 self.points。"""
        q = cp.ascontiguousarray(cp.asarray(query, dtype=cp.float64))
        idx = cp.empty(len(q), dtype=cp.int32)
        d2 = cp.empty(len(q), dtype=cp.float64)
        if len(q):
            _NN(((len(q) + _BLOCK - 1) // _BLOCK,), (_BLOCK,), self._args(q) + (np.float64(radius), idx, d2))
        return idx, d2

    def radius_cov(self, query, radius):
        """每個查詢點在 radius 內鄰居的數量與 3×3 共變異數。"""
        q = cp.ascontiguousarray(cp.asarray(query, dtype=cp.float64))
        n = cp.empty(len(q), dtype=cp.int32)
        s = cp.empty((len(q), 3), dtype=cp.float64)
        sq = cp.empty((len(q), 6), dtype=cp.float64)
        if len(q):
            _STATS(((len(q) + _BLOCK - 1) // _BLOCK,), (_BLOCK,), self._args(q) + (np.float64(radius), n, s, sq))
        cnt = cp.maximum(n, 1).astype(cp.float64)[:, None]
        mean = s / cnt
        second = cp.empty((len(q), 3, 3), dtype=cp.float64)
        second[:, 0, 0] = sq[:, 0]; second[:, 0, 1] = sq[:, 1]; second[:, 0, 2] = sq[:, 2]
        second[:, 1, 0] = sq[:, 1]; second[:, 1, 1] = sq[:, 3]; second[:, 1, 2] = sq[:, 4]
        second[:, 2, 0] = sq[:, 2]; second[:, 2, 1] = sq[:, 4]; second[:, 2, 2] = sq[:, 5]
        cov = second / cnt[:, :, None] - mean[:, :, None] * mean[:, None, :]
        return n, cov


def local_geometry(points, radius, min_neighbors, cell=None, planar_eps=1e-3):
    """估計每個點的局部幾何（GPU）。

    回傳 (valid, normals, cov_reg)：
    - valid：鄰居數 >= min_neighbors 的點。
    - normals：共變異數最小特徵值對應的特徵向量（表面法向，正負號不定）。
    - cov_reg：GICP 用的正則化共變異數 V·diag(eps,1,1)·Vᵀ（把每個點當成一小片平面）。
    """
    p = cp.asarray(points, dtype=cp.float64)
    index = GridIndex(p, cell or radius)
    n_sorted, cov_sorted = index.radius_cov(index.points, radius)
    n = cp.empty_like(n_sorted)
    cov = cp.empty_like(cov_sorted)
    n[index.order] = n_sorted
    cov[index.order] = cov_sorted
    valid = n >= min_neighbors
    safe = cp.where(valid[:, None, None], cov, cp.eye(3, dtype=cp.float64)[None])
    _, vec = cp.linalg.eigh(safe)  # 特徵值由小到大
    normals = vec[:, :, 0]
    diag = cp.asarray([planar_eps, 1.0, 1.0], dtype=cp.float64)
    cov_reg = (vec * diag[None, None, :]) @ vec.transpose(0, 2, 1)
    return valid, normals, cov_reg


class TargetModel:
    """配準目標（世界座標局部子圖）：點、法向、GICP 共變異數與最近鄰索引，全部在 GPU。"""

    def __init__(self, points, cell, normal_radius, min_neighbors, planar_eps=1e-3):
        p = cp.asarray(points, dtype=cp.float64)
        valid, normals, cov = local_geometry(p, normal_radius, min_neighbors, planar_eps=planar_eps)
        if int(valid.sum()) < 100:
            raise ValueError('目標點雲的有效幾何點少於 100，無法配準')
        self.index = GridIndex(p[valid], cell)
        self.points = self.index.points
        self.normals = normals[valid][self.index.order]
        self.cov = cov[valid][self.index.order]
        self.planar_eps = planar_eps

    def __len__(self):
        return len(self.points)

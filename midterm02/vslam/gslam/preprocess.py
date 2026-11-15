"""GPU 點雲前處理（每一步都可在 params.py 關閉）：無效點、深度範圍、天空、體素降採樣、孤立點。"""
import cupy as cp

from .gpu_cloud import GridIndex, voxel_downsample


def preprocess(xyz, rgb, cfg):
    """輸入相機座標點雲（NumPy），回傳 (GPU 上的降採樣點 (M,3), 各步驟剩餘點數的 dict)。"""
    p = cp.asarray(xyz, dtype=cp.float64)
    stats = dict(raw=int(len(p)))
    keep = cp.all(cp.isfinite(p), axis=1)
    keep &= (p[:, 2] > cfg['min_depth_m']) & (p[:, 2] < cfg['max_depth_m'])
    sky = cfg['sky_color']
    if sky['enabled'] and rgb is not None and len(rgb) == len(p):
        c = cp.asarray(rgb, dtype=cp.float64)
        total = cp.sum(c, axis=1) + 1e-6
        is_sky = ((c[:, 2] / total > sky['min_blue_ratio']) & (total / 3 > sky['min_brightness'])
                  & (p[:, 1] < -sky['min_height_m']))  # y 向下，負值代表高於相機
        keep &= ~is_sky
    p = p[keep]
    stats['after_range'] = int(len(p))
    if len(p) == 0:
        return p, stats
    p = voxel_downsample(p, cfg['voxel_m'])
    stats['after_voxel'] = int(len(p))
    ro = cfg['radius_outlier']
    if ro['enabled'] and len(p):
        index = GridIndex(p, ro['radius_m'])
        n, _ = index.radius_cov(index.points, ro['radius_m'])
        p = index.points[n >= ro['min_neighbors'] + 1]  # +1：鄰域統計包含自己
        stats['after_outlier'] = int(len(p))
    if len(p) > cfg['max_points']:
        p = p[cp.linspace(0, len(p) - 1, cfg['max_points']).astype(cp.int64)]
    stats['final'] = int(len(p))
    return cp.ascontiguousarray(p), stats

#!/usr/bin/env python3
"""可重現的 GPU 驗證：確認配準真的在 CUDA 上計算，並量測與 CPU 的速度差。

    scripts/dev.sh gpu          （等同容器內 python scripts/gpu_check.py）

檢查項目：
1. CuPy 看到的 CUDA 裝置、VRAM、compute capability。
2. 自寫 CUDA kernel 的最近鄰結果與 CPU（SciPy cKDTree）逐點一致。
3. GPU 最近鄰／GICP 的耗時，以及同樣資料用 CPU cKDTree 的耗時（對照）。
4. 配準過程中 GPU 記憶體池確實有配置（證明資料在 GPU 上）。
5. 若有安裝 PyTorch：torch.cuda 是否可用。
"""
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vslam'))

import numpy as np  # noqa: E402


def main():
    import cupy as cp
    from scipy.spatial import cKDTree
    from gslam import se3
    from gslam.gpu_cloud import GridIndex, TargetModel, device_info, sync
    from gslam.gpu_registration import register, source_covariances

    print('device:', device_info())
    rng = np.random.default_rng(0)
    # 合成場景：地面＋兩面牆，約 20 萬點
    n = 70000
    ground = np.c_[rng.uniform(-20, 20, n), rng.normal(1.5, 0.01, n), rng.uniform(0, 40, n)]
    wall_a = np.c_[rng.normal(-8, 0.01, n), rng.uniform(-3, 1.5, n), rng.uniform(0, 40, n)]
    wall_b = np.c_[rng.uniform(-8, 8, n), rng.uniform(-3, 1.5, n), rng.normal(35, 0.01, n)]
    target = np.vstack([ground, wall_a, wall_b])
    T_true = se3.make(se3.so3_exp([0.01, 0.03, -0.02]), [0.25, -0.05, 0.4])
    source = (target[::4] - T_true[:3, 3]) @ T_true[:3, :3]   # p_cam = Rᵀ(p_world − t)

    index = GridIndex(cp.asarray(target), 1.0)
    q = cp.asarray(source)
    index.nearest(q, 1.0); sync()            # 暖機（kernel 編譯）
    t0 = time.perf_counter(); idx, d2 = index.nearest(q, 1.0); sync(); gpu_nn = time.perf_counter() - t0
    t0 = time.perf_counter(); tree = cKDTree(cp.asnumpy(index.points)); dist, ref = tree.query(source, distance_upper_bound=1.0)
    cpu_nn = time.perf_counter() - t0
    found = np.isfinite(dist)
    same = bool(np.array_equal(cp.asnumpy(idx >= 0), found)
                and np.allclose(np.sqrt(cp.asnumpy(d2))[found], dist[found], atol=1e-9))
    print('nearest neighbour: GPU %.2f ms, CPU cKDTree %.2f ms, identical=%s' % (gpu_nn * 1e3, cpu_nn * 1e3, same))

    model = TargetModel(cp.asarray(target), 1.0, 0.75, 6)
    valid, cov = source_covariances(q, 0.75, 6)
    src, cov = q[valid], cov[valid]
    for method in ('icp_plane', 'gicp'):
        register(src, model, np.eye(4), method=method, src_cov=cov); sync()
        t0 = time.perf_counter()
        r = register(src, model, np.eye(4), method=method, src_cov=cov); sync()
        dt, dr = se3.relative_error(T_true, r['T'])
        print('%-9s %.1f ms  iterations=%d  fitness=%.3f  rmse=%.4f m  error: %.4f m / %.4f deg' % (
            method, (time.perf_counter() - t0) * 1e3, r['iterations'], r['fitness'], r['inlier_rmse'], dt, np.rad2deg(dr)))
    print('GPU memory pool in use: %.1f MiB' % (cp.get_default_memory_pool().used_bytes() / 2 ** 20))
    try:
        import torch
        print('torch %s cuda_available=%s' % (torch.__version__, torch.cuda.is_available()))
    except ImportError:
        print('torch 未安裝（只有學習式前端需要）')
    if not same:
        sys.exit('FAIL：GPU 最近鄰與 CPU 不一致')
    print('OK')


if __name__ == '__main__':
    main()

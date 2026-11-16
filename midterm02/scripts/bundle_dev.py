#!/usr/bin/env python3
"""開發用：讀某個 run 的 debug/bundle_input.npz，不重跑前端直接重做 BA，把結果寫成 pose CSV 供診斷工具比較。

    python scripts/bundle_dev.py outputs/run/<run> outputs/cache/<name>.csv

只用來試 BA 的設定；正式結果一律由 vslam/run_slam.py 一次跑完產生。
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vslam'))

import numpy as np  # noqa: E402


def main():
    import gtsam
    from gslam import bundle, outputs, params as params_mod
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    p, _ = params_mod.load(None)
    cfg, g = p['bundle'], p['graph']
    K, poses, kpts, links, raw = bundle.load_input(run / 'debug' / 'bundle_input.npz')
    sig = lambda r, t: np.array([r, r, r, t, t, t], dtype=np.float64)
    noise = {0: gtsam.noiseModel.Diagonal.Sigmas(sig(cfg['odom_rot_sigma_rad'], cfg['odom_trans_sigma_m'])),
             1: gtsam.noiseModel.Robust.Create(
                 gtsam.noiseModel.mEstimator.Huber.Create(g['huber_k']),
                 gtsam.noiseModel.Diagonal.Sigmas(sig(g['loop_rot_sigma_rad'], g['loop_trans_sigma_m'])))}
    between = [(i, j, T, noise[kind]) for i, j, T, kind in raw if kind == 0 or cfg['use_loop_edges']]
    result, info = bundle.run(K, poses, kpts, links, between, cfg)
    print(info)
    timestamps = [r[0] for r in csv.reader((run / 'poses_optimized.csv').open())][1:]
    outputs.write_poses(out, timestamps, result)


if __name__ == '__main__':
    main()

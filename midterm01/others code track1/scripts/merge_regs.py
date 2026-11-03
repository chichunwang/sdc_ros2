"""逐幀挑配準來源：幾份 register_all 的 npz（同一個初值軌跡、範圍由近到遠），每幀用第一個「xy 可觀測性 ≥ 門檻」的；
都不夠就用最後一個（最遠）。近點深度誤差小但在路口／轉彎沒東西可配（Track3 轉彎段 12 m 的 xy_min_info 中位數只有 0.5、18 m 有 3.7）。
用法：python -I scripts/merge_regs.py --out merged.npz --min-xy-info 2.0 r12.npz r15.npz r18.npz r25.npz"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('regs', nargs='+', type=Path, help='由近到遠')
    ap.add_argument('--min-xy-info', type=float, default=2.0, help='xy 最小資訊量門檻（每 (0.1 m)²）')
    ap.add_argument('--min-ng03', type=float, default=0.4)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    regs = [np.load(p) for p in a.regs]
    N = len(regs[0]['frame'])
    assert all(len(r['frame']) == N for r in regs), '幀數不一致'
    # 還沒被更近範圍選走的幀，若這個範圍過門檻就選它
    pick = np.full(N, -1)
    for i, r in enumerate(regs):
        ok = r['ok'] & (r['ng_frac03'] >= a.min_ng03) & (r['xy_min_info'] >= a.min_xy_info)
        pick[(pick < 0) & ok] = i
    pick[pick < 0] = len(regs) - 1
    out = {}
    for key in regs[0].files:
        v0 = regs[0][key]
        if v0.ndim == 0 or len(v0) != N:
            out[key] = v0
            continue
        stacked = np.stack([r[key] for r in regs])
        out[key] = stacked[pick, np.arange(N)]
    out['range_pick'] = pick
    np.savez(a.out, **out)
    counts = {str(p.name): int((pick == i).sum()) for i, p in enumerate(a.regs)}
    print(json.dumps({'frames': N, 'picked': counts, 'min_xy_info': a.min_xy_info}, indent=1))


if __name__ == '__main__':
    main()

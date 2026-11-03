"""兩份 Kaggle 格式 CSV（idx,x,y,yaw）相比：候選對參考（例如 Track1 的假真值 I_ns）算 xy RMSE、yaw RMSE 與分數式
（xy RMSE + 0.2 × yaw RMSE 度），分全段、前 30%、每 --block 幀一段；--other 再多比一份（例如離線版），看兩個候選差多少。
用法：python -I scripts/compare_csv.py 候選.csv --ref candidates/ns/track1_I_ns.csv [--other 離線.csv] [--block 500]"""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl


def load(path):
    c = pl.read_csv(path).to_numpy()
    return c[:, 1:3].astype(float), c[:, 3].astype(float)


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def stats(xy, yaw, rxy, ryaw):
    d = np.hypot(*(xy - rxy).T)
    dy = np.degrees(wrap(yaw - ryaw))
    xy_rmse = float(np.sqrt(np.mean(d ** 2))); yaw_rmse = float(np.sqrt(np.mean(dy ** 2)))
    return {'xy_rmse': xy_rmse, 'xy_median': float(np.median(d)), 'xy_max': float(d.max()),
            'yaw_rmse_deg': yaw_rmse, 'yaw_mean_deg': float(dy.mean()), 'score': xy_rmse + 0.2 * yaw_rmse}


def fmt(s):
    return (f"xy rmse {s['xy_rmse']:.3f} med {s['xy_median']:.3f} max {s['xy_max']:.2f} | "
            f"yaw rmse {s['yaw_rmse_deg']:.2f} mean {s['yaw_mean_deg']:+.2f} | score {s['score']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv', type=Path)
    ap.add_argument('--ref', type=Path, required=True)
    ap.add_argument('--other', type=Path, help='另一份候選（例如離線版），順便算候選對它的差')
    ap.add_argument('--block', type=int, default=500)
    ap.add_argument('--json', type=Path, help='把結果也寫成 json')
    a = ap.parse_args()
    xy, yaw = load(a.csv); rxy, ryaw = load(a.ref)
    n = len(xy)
    assert n == len(rxy), f'列數不同：{n} vs {len(rxy)}'
    assert np.isfinite(xy).all() and np.isfinite(yaw).all(), '候選有 NaN'
    out = {'frames': n, 'all': stats(xy, yaw, rxy, ryaw), 'first30': stats(*(v[:int(n * .3)] for v in (xy, yaw, rxy, ryaw)))}
    print(f'{a.csv.name} vs {a.ref.name}')
    print('  all      ', fmt(out['all']))
    print('  first 30%', fmt(out['first30']))
    out['blocks'] = {}
    for k0 in range(0, n, a.block):
        sl = slice(k0, min(k0 + a.block, n))
        s = stats(xy[sl], yaw[sl], rxy[sl], ryaw[sl]); out['blocks'][f'{sl.start}-{sl.stop - 1}'] = s
        print(f'  {sl.start:4d}-{sl.stop - 1:4d}', fmt(s))
    if a.other:
        oxy, oyaw = load(a.other)
        out['vs_other'] = {'file': a.other.name, 'all': stats(xy, yaw, oxy, oyaw), 'other_vs_ref': stats(oxy, oyaw, rxy, ryaw)}
        print(f'{a.csv.name} vs {a.other.name}:', fmt(out['vs_other']['all']))
        print(f'{a.other.name} vs {a.ref.name}:', fmt(out['vs_other']['other_vs_ref']))
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()

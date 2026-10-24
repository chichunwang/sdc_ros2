#!/usr/bin/env python3
"""Print three-pose heading diagnostics from Track3 pose_trace.jsonl.

No changes to localization are made. Angles are map-frame azimuths of the local
+X axis, NOT necessarily the physical camera/vehicle heading.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def local_x_azimuth(T):
    R = np.asarray(T, dtype=float)[:3, :3]
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


def normalized_angle(a):
    return (a + 180.0) % 360.0 - 180.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path, help='pose_trace.jsonl produced by an instrumented run')
    parser.add_argument('--limit', type=int, default=12)
    args = parser.parse_args()

    print('NOTE: All angles below describe the local +X axis in the map XY plane.')
    print('      They are vehicle headings ONLY if local +X is the vehicle forward axis.')
    print('  idx  map  pred_Xdeg   reg_Xdeg final_Xdeg  reg-pred final-pred  moved_m move_deg  final-move')
    print('  ' + '-' * 99)
    previous = None
    count = 0
    with args.trace.open(encoding='utf8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            i = int(row['idx'])
            pred = local_x_azimuth(row['predicted_T'])
            reg = local_x_azimuth(row['registered_T'])
            final = local_x_azimuth(row['final_T'])
            p = np.asarray(row['final_T'], dtype=float)[:2, 3]
            dist, move, mismatch = '-', '-', '-'
            if previous is not None:
                d = p - previous
                value = float(np.linalg.norm(d))
                dist = f'{value:.3f}'
                if value > .05:
                    move_v = float(np.degrees(np.arctan2(d[1], d[0])))
                    move = f'{move_v:+.1f}'
                    mismatch = f'{normalized_angle(move_v-final):+.1f}'
            print(f"{i:5d}  {int(row['map_accepted']):3d}  {pred:+9.2f} {reg:+10.2f} {final:+10.2f}  "
                  f"{normalized_angle(reg-pred):+8.2f} {normalized_angle(final-pred):+10.2f}  "
                  f"{dist:>7} {move:>8} {mismatch:>11}")
            previous = p
            count += 1
            if count >= args.limit:
                break
    if count == 0:
        raise SystemExit('No pose records found')
    print('\nmap: 1=map registration was fused; 0=not fused.')
    print('reg-pred: map registration heading change, even if rejected.')
    print('final-pred: actual heading correction adopted by the fusion output.')
    print('final-move: movement bearing minus +X azimuth. This is NOT an error metric')
    print('            unless the source frame +X is confirmed to be vehicle-forward.')


if __name__ == '__main__':
    main()

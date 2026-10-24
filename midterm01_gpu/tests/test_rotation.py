#!/usr/bin/env python3
"""Non-mutating finite-difference check of Track3 pose-update Jacobian.

Run from the directory containing the track3 Python package:
    python3 check_rotation_update.py
or optionally:
    python3 check_rotation_update.py --guess /path/to/init_guess.txt

Requires numpy and the project's track3.geometry.apply_delta; it does not import CuPy.
"""
import argparse
import numpy as np


def skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


def test_jacobian(T, scale, label):
    from track3.geometry import apply_delta
    pts = np.array([[3., -0.5, 7.], [-4., 2., 1.], [0.8, 3., -2.]], dtype=float)
    eps = 1e-6
    discrepancies = []
    for p in pts:
        v = scale * (T[:3, :3] @ p)
        analytic = np.column_stack([np.eye(3), -skew(v), v])
        numeric = np.zeros((3, 7))
        for k in range(7):
            d = np.zeros(7)
            d[k] = eps
            Tp, sp = apply_delta(T.copy(), scale, d)
            d[k] = -eps
            Tm, sm = apply_delta(T.copy(), scale, d)
            qp = sp * (Tp[:3, :3] @ p) + Tp[:3, 3]
            qm = sm * (Tm[:3, :3] @ p) + Tm[:3, 3]
            numeric[:, k] = (qp - qm) / (2 * eps)
        discrepancies.append(np.max(np.abs(analytic - numeric), axis=0))
    errors = np.max(np.array(discrepancies), axis=0)
    print(f'\n[{label}]')
    print('translation error (dx,dy,dz):', errors[:3].round(8))
    print('rotation error (rx,ry,rz):   ', errors[3:6].round(8))
    print('log-scale error:             ', round(errors[6], 8))
    print('maximum absolute error:      ', f'{errors.max():.6g}')
    if errors.max() < 1e-3:
        print('PASS: apply_delta and gpu.py Jacobian agree locally.')
    else:
        print('FAIL: mismatch. Inspect how apply_delta composes rotation/translation/scale.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--guess', help='optional 4x4 init_guess.txt for testing actual pose')
    args = parser.parse_args()
    a = 0.63
    T = np.eye(4)
    T[:3, :3] = [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
    T[:3, 3] = [2.3, -4.2, 1.7]
    test_jacobian(T, 3.0, 'synthetic pose with nonzero translation and rotation')
    if args.guess:
        G = np.loadtxt(args.guess)
        if G.shape != (4, 4):
            raise ValueError(f'Expected 4x4 matrix, got {G.shape}')
        print('\nIf init_guess is T_map_base, its +X forward vector (map axes):', G[:3, 0])
        print('If init_guess is T_map_camera_optical, its +Z viewing vector (map axes):', G[:3, 2])
        print('These are alternative interpretations, not two simultaneously valid camera orientations.')
        test_jacobian(G, 3.0, 'actual init_guess')


if __name__ == '__main__':
    main()
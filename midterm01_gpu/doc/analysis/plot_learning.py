#!/usr/bin/env python3
"""Python/Matplotlib 教學主入口：讀保存資料與合成示範，不重跑定位。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np

from learning_models import bias_cost_grids, icp_jacobian, icp_scene, kalman_demo, observability


HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bundle(path):
    """SciPy 可直接讀既有 MAT 資料容器，不需要 MATLAB 程式或授權。"""
    from scipy.io import loadmat
    path = Path(path).resolve()
    sidecar = path.with_suffix('.json')
    metadata = json.loads(sidecar.read_text())
    if metadata['mat_sha256'] != digest(path):
        raise ValueError('Teaching bundle SHA differs from its manifest')
    data = loadmat(path, simplify_cells=True)['sdc']
    if data['schema'] != 'sdc_learning_v1':
        raise ValueError('Unsupported teaching bundle schema')
    if data['frame_ns'].dtype != np.int64 or np.any(np.diff(data['frame_ns']) <= 0):
        raise ValueError('Expected increasing int64 frame timestamps')
    np.testing.assert_array_equal(
        np.rint(data['time_s']*1e9).astype(np.int64), data['frame_ns']-data['time_origin_ns'])
    return data, metadata


def plot_icp():
    import matplotlib.pyplot as plt
    ground, gn, room, rn = icp_scene()
    jg = icp_jacobian(ground, gn); jr = icp_jacobian(room, rn)
    sg, hg = observability(jg); sr, hr = observability(jr)
    fig = plt.figure(figsize=(13.6, 8.4), layout='constrained')
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.scatter(*room.T, c=rn, s=7, alpha=.65)
    ax.scatter(0, 0, 1.5, marker='*', c='black', s=100, label='Source origin')
    ax.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)', title='Ground + opposing walls: known correspondences')
    ax.set_box_aspect((12, 10, 4)); ax.view_init(elev=25, azim=35); ax.legend(fontsize=8)
    ax = fig.add_subplot(2, 2, 2)
    ax.semilogy(np.arange(1, 8), np.maximum(sg, 1e-14), 'o-', label='Ground only')
    ax.semilogy(np.arange(1, 8), np.maximum(sr, 1e-14), 's-', label='Ground + walls')
    ax.set(xlabel='Singular-value index', ylabel='Scaled Jacobian singular value',
           title='Seven unknowns: translation, rotation, log-scale'); ax.legend(fontsize=8)
    ax = fig.add_subplot(2, 2, 3)
    ax.semilogy(np.arange(1, 4), np.maximum(hg, 1e-14), 'o-', label='Ground only')
    ax.semilogy(np.arange(1, 4), np.maximum(hr, 1e-14), 's-', label='Ground + walls')
    ax.set(xlabel='Horizontal singular-value index', ylabel='After nuisance projection',
           title='XY/yaw support after removing Z/roll/pitch/scale'); ax.legend(fontsize=8)
    ax.set_xticks([1, 2, 3])
    ax = fig.add_subplot(2, 2, 4)
    shift = np.linspace(-2, 2, 101)
    ax.plot(shift, np.mean(gn[:, 0]**2)*shift**2, label='Ground only')
    ax.plot(shift, np.mean(rn[:, 0]**2)*shift**2, label='Ground + walls')
    ax.set(xlabel='Pure X translation (m)', ylabel='Mean squared plane residual (m$^2$)',
           title='Sliding along a plane can leave residual unchanged'); ax.legend(fontsize=8)
    for ax in fig.axes:
        ax.grid(alpha=.22)
    fig.suptitle('SYNTHETIC: local ICP linearization, not a full registration experiment', fontsize=14)
    summary = {'ground_rank': int(np.linalg.matrix_rank(jg)), 'room_rank': int(np.linalg.matrix_rank(jr)),
               'ground_horizontal_singular': hg.tolist(), 'room_horizontal_singular': hr.tolist()}
    return ('04_icp_observability_synthetic', fig), summary


def plot_kalman():
    import matplotlib.pyplot as plt
    d = kalman_demo(); t = d['time']
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.4), layout='constrained')
    ax = axes[0, 0]
    ax.scatter(t, d['measurement'], s=9, color='.7', label='Synthetic GNSS')
    ax.plot(t, d['truth'], color='black', lw=1.4, label='Known synthetic truth')
    ax.plot(t, d['filtered'][:, 0], label='KF'); ax.plot(t, d['smoothed'][:, 0], label='RTS')
    ax.set(xlabel='Synthetic time (s)', ylabel='Position (m)',
           title='KF uses past/current data; RTS also uses later data'); ax.legend(fontsize=8)
    ax = axes[0, 1]
    ax.plot(t, d['filtered'][:, 0]-d['truth'], label='KF error')
    ax.plot(t, d['smoothed'][:, 0]-d['truth'], label='RTS error')
    ax.plot(t, d['bias'], 'k--', lw=1.3, label='Injected slow bias')
    ax.set(xlabel='Synthetic time (s)', ylabel='Error against synthetic truth (m)',
           title='Smoothing alone cannot identify slow GNSS bias'); ax.legend(fontsize=8)
    p, b, gnss, both = bias_cost_grids()
    for ax, cost, title in [(axes[1, 0], gnss, 'GNSS only: a valley of equally valid (p, b)'),
                            (axes[1, 1], both, 'An independent absolute anchor restricts the valley')]:
        ax.contour(p, b, cost, levels=[.25, 1, 4, 9, 16])
        ax.plot(10, .8, 'k*', ms=10)
        ax.set(xlabel='Unknown position p (m)', ylabel='Unknown GNSS bias b (m)', title=title)
    for ax in axes.ravel():
        ax.grid(alpha=.22)
    fig.suptitle('SYNTHETIC: temporal smoothing and absolute bias identification', fontsize=14)
    return ('05_kalman_bias_synthetic', fig), {'seed': 21, 'samples': len(t),
        'filtered_cov_min_eigenvalue': float(np.linalg.eigvalsh(d['filtered_cov']).min()),
        'smoothed_cov_min_eigenvalue': float(np.linalg.eigvalsh(d['smoothed_cov']).min())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lesson', choices=['all', 'real', 'icp', 'kalman'], default='all')
    parser.add_argument('--bundle', type=Path, default=HERE/'data/track02_learning.mat')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--show', action='store_true', help='另開可縮放、平移的 Matplotlib 視窗')
    args = parser.parse_args()
    import matplotlib
    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
                         'legend.fontsize': 8, 'lines.linewidth': 1.25})
    parent = HERE/'figures'
    if args.output_dir is None:
        output = parent/('python_'+datetime.now(ZoneInfo('Asia/Taipei')).strftime('%y%m%d%H%M%S'))
        base = output
        sequence = 2
        while output.exists():
            output = Path(str(base)+f'_{sequence:04d}'); sequence += 1
    else:
        output = args.output_dir.resolve()
        if not output.is_relative_to(parent.resolve()):
            parser.error('--output-dir must be under doc/analysis/figures')
    # Refuse an existing output directory; source runs and prior figures remain intact.
    output.mkdir(parents=True, exist_ok=False)
    report = {'renderer': 'Python/Matplotlib', 'matplotlib_version': matplotlib.__version__,
              'numpy_version': np.__version__, 'python_version': sys.version.split()[0],
              'lesson': args.lesson, 'localization_rerun': False, 'production_parameters_changed': False,
              'figures': [], 'complete': False}
    figures = []
    try:
        if args.lesson in ('all', 'real'):
            if not args.bundle.is_file() and args.bundle.resolve() == (HERE/'data/track02_learning.mat').resolve():
                from export_learning_data import ROOT, export_bundle
                export_bundle(ROOT/'outputs/run/track02_full_2610212125',
                              ROOT/'outputs/run/track02_full_2610191757', args.bundle)
            data, metadata = load_bundle(args.bundle)
            from plot_saved_run_py import plot_saved_run
            figures.extend(plot_saved_run(data))
            report['data'] = {'bundle': str(args.bundle.resolve()), 'sha256': digest(args.bundle),
                              'frames': len(data['frame_ns']), 'source_run': metadata['run'],
                              'reference_run': metadata['reference'], 'ground_truth_available': False}
        if args.lesson in ('all', 'icp'):
            figure, summary = plot_icp(); figures.append(figure); report['icp'] = summary
        if args.lesson in ('all', 'kalman'):
            figure, summary = plot_kalman(); figures.append(figure); report['kalman'] = summary
        for stem, figure in figures:
            # A real draw catches math-text/layout/renderer errors before publishing.
            figure.canvas.draw()
            files = []
            for extension in ('png', 'pdf'):
                target = output/f'{stem}.{extension}'
                figure.savefig(target, dpi=150)
                files.append({'path': target.name, 'sha256': digest(target), 'bytes': target.stat().st_size})
            report['figures'].append({'name': stem, 'files': files})
        report['program_sha256'] = {name: digest(HERE/name) for name in
            ('plot_learning.py', 'plot_saved_run_py.py', 'learning_models.py')}
        report['complete'] = True
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        (output/'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        if not args.show:
            plt.close('all')
    print(f'Figures: {output}\nFinished: {len(figures)} PNG + {len(figures)} PDF')
    if args.show:
        plt.show()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

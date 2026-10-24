#!/usr/bin/env python3
"""Read-only plots for isolated registration experiments; no trajectory outputs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from three_camera_calibration import load_ply
from run_track3_causal import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = args.run.resolve(strict=True)
    run.relative_to((root/'outputs/run').resolve())
    summary = json.loads((run/'research_summary.json').read_text())
    sources = [run/'research_summary.json', run/'map_crop.npy']
    recovery = []
    for file in sorted(run.glob('recovery_*.json')):
        recovery.append(json.loads(file.read_text()))
        sources.append(file)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for mode, color, label in [(False, '#bd493d', 'ICP'), (True, '#247080', 'NDT + ICP')]:
        rows = sorted([v for v in recovery if v['use_ndt'] == mode], key=lambda v:v['case'])
        ax[0].plot([r['case'] for r in rows], [r['translation_error_m'] for r in rows], 'o-', color=color, label=label)
    ax[0].set(yscale='log', xlabel='Known-transform case', ylabel='Translation error (m)',
              title='Map-derived source: known truth')
    ax[0].legend(); ax[0].grid(alpha=.2)
    for probe in summary['probes']:
        for candidate in probe['candidates']:
            ax[1].scatter(probe['original_idx'], candidate['scale'],
                          color='#bd493d' if candidate['accepted'] else '#a1adb2', s=25,
                          marker='o' if candidate['seed_type'] == 'official' else 'x')
    ax[1].set(xlabel='Independent frame index (not a trajectory)', ylabel='Fitted raw-to-metric multiplier',
              title='Real PLY: local fits disagree; all final probes rejected')
    ax[1].grid(alpha=.2)
    fig.suptitle('Registration mechanism evidence | real pose accuracy is unknown', fontsize=13)
    target = run/'registration_evidence.png'
    if target.exists():
        raise ValueError('Do not overwrite research plots')
    fig.savefig(target, dpi=160); plt.close(fig)
    first = summary['probes'][0]
    map_points = np.load(run/'map_crop.npy', mmap_mode='r')
    path = root.parent/'midterm01/data/Track3/raw_ply'/first['selected']['filename']
    raw = load_ply(path, max_points=12000, leaf=.03)
    sources.append(path)
    candidates = sorted([c for c in first['candidates'] if c['seed_type'] == 'official'], key=lambda c:c['seed_scale'])
    candidates = [candidates[0], candidates[-1]]
    center = np.array(first['selected']['pose'])[:3, 3]
    local = np.asarray(map_points[np.all(abs(map_points-center) < [35, 35, 15], axis=1)])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for col, candidate in enumerate(candidates):
        pose = np.array(candidate['pose'])
        world = candidate['scale']*raw @ pose[:3, :3].T+pose[:3, 3]
        for row, dims in enumerate([(0, 1), (0, 2)]):
            a, b = dims
            axes[row, col].scatter(local[::3, a]-center[a], local[::3, b]-center[b], s=.4, c='#b0b7bd', alpha=.5, rasterized=True)
            axes[row, col].scatter(world[:, a]-center[a], world[:, b]-center[b], s=1., c='#ca503c', alpha=.5, rasterized=True)
            axes[row, col].plot(pose[a, 3]-center[a], pose[b, 3]-center[b], '+', color='#123f50', ms=12)
            axes[row, col].set(xlabel=f'{"XYZ"[a]} relative to plot center (m)',
                               ylabel=f'{"XYZ"[b]} relative to plot center (m)', aspect='equal')
        axes[0, col].set_title(f"Seed {candidate['seed_scale']:g} -> fitted scale {candidate['scale']:.3f}\nLocal checks pass: {candidate['accepted']}")
    fig.suptitle('Same first PLY, different local solutions | gray: map, red: source, +: source origin', fontsize=12)
    fig.savefig(run/'first_frame_ambiguity.png', dpi=160); plt.close(fig)
    write_json(run/'research_plot_manifest.json', dict(
        sources={str(p):digest(p) for p in sources}, script_sha256=digest(Path(__file__)),
        output=['registration_evidence.png', 'first_frame_ambiguity.png'],
        scope='Independent registration diagnostics only; not full-run trajectory analysis or accuracy proof'))


if __name__ == '__main__':
    main()

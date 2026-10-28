#!/usr/bin/env python3
"""Synthetic checks figures, distinct from real Track2/full plots."""
import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from track3.io import sha256, write_json


def analyze(run):
    output = run/'analysis'; output.mkdir(exist_ok=False)
    tree = ET.parse(run/'pytest.xml')
    cases = list(tree.iter('testcase'))
    names = [c.attrib['name'].replace('test_', '') for c in cases]
    seconds = [float(c.attrib['time']) for c in cases]
    colors = ['tab:red' if c.find('failure') is not None or c.find('error') is not None else 'tab:blue' for c in cases]
    fig, ax = plt.subplots(figsize=(11, max(5, .38*len(cases))))
    ax.barh(names, seconds, color=colors)
    ax.set(xlabel='Test wall time [s]', title='Generated-data checks (no real Track2 localization)')
    ax.invert_yaxis(); fig.tight_layout(); fig.savefig(output/'01_synthetic_checks.png', dpi=180); plt.close(fig)
    synthetic = run/'pytest/test_generated_runner_outputs0/run'
    truth = np.array([0., 0., 1.573]); truth_scale = 8.
    restored = run/'pytest/test_generated_output_runner0/generated_run'
    if restored.is_dir():
        synthetic = restored
        truth = np.loadtxt(restored.parent/'data/init_guess.txt', delimiter=',')[:3, 3]
        truth_scale = 3.
    source_names = ['pytest.xml', 'pytest.log']
    if synthetic.is_dir():
        pose = np.loadtxt(synthetic/'poses.txt', ndmin=2)
        obs = [json.loads(s) for s in (synthetic/'observations.jsonl').read_text().splitlines()]
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        axes[0].plot(np.linalg.norm(pose[:, [3, 7, 11]]-truth, axis=1), marker='o')
        axes[0].set(ylabel='Position error [m]', title='Generated stationary camera: known synthetic truth')
        axes[1].plot([o['scale'] for o in obs], marker='o'); axes[1].axhline(truth_scale, color='gray', ls='--', label=f'Generated truth={truth_scale:g}')
        axes[1].set(xlabel='Generated frame idx', ylabel='PLY scale'); axes[1].legend()
        for ax in axes: ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(output/'02_generated_runner.png', dpi=180); plt.close(fig)
        source_names += [str((synthetic/n).relative_to(run)) for n in ('poses.txt', 'observations.jsonl')]
    comparisons = sorted((run/'pytest').glob('test_cpp_5839746c_solver_nu*/comparison.json'))
    if comparisons:
        records = [json.loads(f.read_text()) for f in comparisons]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        labels = [f"alpha={r['alpha']}" for r in records]
        axes[0].bar(labels, [r['max_correction_abs_difference'] for r in records])
        axes[0].set(ylabel='Max correction difference [mixed m/rad]', title='Original C++ vs CUDA implementation')
        axes[1].bar(labels, [r['cost_abs_difference'] for r in records]); axes[1].set(ylabel='Absolute objective difference')
        fig.tight_layout(); fig.savefig(output/'03_cpp_cuda_comparison.png', dpi=180); plt.close(fig)
        source_names += [str(f.relative_to(run)) for f in comparisons]
    if (run/'data_contract.json').exists(): source_names.append('data_contract.json')
    write_json(output/'manifest.json', dict(kind='synthetic_and_input_checks_only',
        sources={n: sha256(run/n) for n in source_names}, script_sha256=sha256(Path(__file__)),
        real_localization_executed=False))
    (output/'README.md').write_text('# Track2 合成與輸入檢查\n\n01 是合成測試耗時；02 是已知靜止真值的生成資料輸出。這些不是 Track2 真實定位結果，也不是 Kaggle 精度。\n\n真實資料若有檢查，只讀時間／來源 SHA，未執行配準或濾波定位。詳見上層 pytest.xml、pytest.log、data_contract.json。\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run-dir', type=Path, required=True)
    analyze(parser.parse_args().run_dir)

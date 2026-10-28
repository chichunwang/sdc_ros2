#!/usr/bin/env python3
"""Read-only figures for an explicitly authorized 3-frame scale diagnosis."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from track3.io import sha256,write_json


def analyze(run):
    if (run/'comparison.json').is_file():
        key='final';metadata=run/'comparison.json'
    else:
        key='geometric';metadata=run/'diagnosis.json'
    records=json.loads(metadata.read_text())['results']
    source_paths=[metadata]+sorted(run.glob('frame_*_display.npz'))
    sources={f.name:sha256(f) for f in source_paths}
    out=run/'analysis';out.mkdir(exist_ok=False)
    def save(fig,name):
        fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(out/f'{name}.{ext}',dpi=180)
        plt.close(fig)
    for row in records:
        idx=row['idx'];a=np.load(run/f'frame_{idx:04d}_display.npz')
        raw=a['raw'];map_xyz=a['map'];T=a['geometric'];s=float(a['scale'])
        baseline=a['baseline'];origin=T[:3,3]
        old=raw @ baseline[:3,:3].T*3.+baseline[:3,3]-origin
        new=raw @ T[:3,:3].T*s+origin-origin
        local=map_xyz-origin
        fig,axes=plt.subplots(2,2,figsize=(12,9))
        for col,(points,title) in enumerate([(old,'Fixed scale=3 (old)'),(new,f'Current geometry scale={s:.3f}')]):
            for axis,coords,labels in [(axes[0,col],(0,1),('Map X offset [m]','Map Y offset [m]')),
                                       (axes[1,col],(0,2),('Map X offset [m]','Map Z offset [m]'))]:
                axis.scatter(local[:,coords[0]],local[:,coords[1]],s=.4,c='.7',alpha=.6,rasterized=True,label='LiDAR XYZ')
                axis.scatter(points[:,coords[0]],points[:,coords[1]],s=.6,c='tab:red' if col==0 else 'tab:green',alpha=.55,rasterized=True,label='Camera PLY')
                axis.set(xlabel=labels[0],ylabel=labels[1],title=f'Frame {idx}: {title}',xlim=(-10,35))
                axis.set_ylim((-25,25) if coords[1]==1 else (-4,10));axis.grid(alpha=.2);axis.legend(markerscale=5)
        save(fig,f'01_frame_{idx:04d}_overlap')
    ids=[r['idx'] for r in records]
    scale=[r[key]['scale'] for r in records]
    def score(r,k):
        return r[k].get('feature_score',{})
    before=[score(r,'baseline')['median_capped_distance_m'] for r in records]
    after=[score(r,key)['median_capped_distance_m'] for r in records]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].plot(ids,[3.]*len(ids),'o--',label='Old fixed scale');axes[0].plot(ids,scale,'o-',label='Current geometry')
    axes[0].set(xlabel='Original frame idx (sparse samples)',ylabel='Raw-to-metre scale');axes[0].legend()
    x=np.arange(len(ids));axes[1].bar(x-.18,before,.36,label='Before');axes[1].bar(x+.18,after,.36,label='After')
    axes[1].set(xticks=x,xticklabels=ids,ylabel='Median capped feature distance [m]',xlabel='Original frame idx')
    axes[1].legend()
    for ax in axes:ax.grid(alpha=.2)
    save(fig,'02_scale_and_geometry')
    assert sources=={f.name:sha256(f) for f in source_paths}
    write_json(out/'manifest.json',dict(sources=sources,script_sha256=sha256(Path(__file__)),
        frames=ids,ground_truth_available=False,kind='authorized_three_frame_geometry_comparison',
        note='Geometric overlap and truncated distances, not localization accuracy; no output poses were modified'))
    (out/'README.md').write_text('# Track2 三幀尺度／幾何對照\n\n僅 frame 0、500、1000。01 圖灰色為 LiDAR XYZ，紅色為舊固定 scale=3，綠色為當幀幾何估計；座標相對新相機位置。02 為尺度與相同來源特徵點到地圖的截斷距離中位數。\n\n這是疏採樣對照，使用既有因果 sensor prior，沒有啟動其他影格定位。幾何改善不等同真實 pose 精度；沒有使用真值、RTS、回修舊軌跡或改寫原結果。來源 SHA 見 manifest.json。\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True)
    analyze(parser.parse_args().run_dir)

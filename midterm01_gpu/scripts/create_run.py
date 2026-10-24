#!/usr/bin/env python3
"""Create unique run and freeze local code before execution (stdlib only)."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from zoneinfo import ZoneInfo


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['full','test'],required=True)
    p.add_argument('--track',choices=['00','03'],default='03');a=p.parse_args()
    root=Path(__file__).resolve().parents[1]; repo=root.parent
    stem=f'track{a.track}_{a.kind}_'+datetime.now(ZoneInfo('Asia/Taipei')).strftime('%y%m%d%H%M')
    runs=root/'outputs/run';runs.mkdir(parents=True,exist_ok=True)
    n=1
    while True:
        run=runs/(stem if n==1 else f'{stem}_{n:04d}')
        try: run.mkdir();break
        except FileExistsError:n+=1
    source=run/'source';source.mkdir()
    files=[]
    for folder in ('track3','scripts','config','doc','tests'):
        if (root/folder).exists():
            files.extend(f for f in (root/folder).rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc')
    files.extend(f for f in root.iterdir() if f.is_file())
    manifest={}
    for f in sorted(files):
        dst=source/f.relative_to(root);dst.parent.mkdir(parents=True,exist_ok=True)
        before=digest(f);shutil.copyfile(f,dst)
        if digest(dst)!=before or digest(f)!=before:raise RuntimeError(f'Source changed while copying: {f}')
        manifest[str(f.relative_to(root))]=before
    def git(*args): return subprocess.run(['git','-C',str(repo),*args],capture_output=True,text=True,check=True).stdout
    info=dict(created_at=datetime.now(ZoneInfo('Asia/Taipei')).isoformat(timespec='seconds'),
              commit=git('rev-parse','HEAD').strip(),sources=manifest,source_root=str(root),snapshot=str(source))
    (run/'git_status.txt').write_text(git('status','--short'))
    (run/'uncommitted.patch').write_text(git('diff','HEAD','--binary'))
    (run/'source_manifest.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
    print(run)

if __name__=='__main__':main()

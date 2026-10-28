#!/usr/bin/env python3
"""Track2 專用 source snapshot；不更動 Track3 snapshot 工具。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from zoneinfo import ZoneInfo


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create(kind='test', track='02'):
    root = Path(__file__).resolve().parents[1]
    repo = root.parent
    runs = root/'outputs/run'; runs.mkdir(parents=True, exist_ok=True)
    stem = f'track{track}_{kind}_'+datetime.now(ZoneInfo('Asia/Taipei')).strftime('%y%m%d%H%M')
    collision = 1
    while True:
        run = runs/(stem if collision == 1 else f'{stem}_{collision:04d}')
        try:
            run.mkdir(); break
        except FileExistsError:
            collision += 1
    source = run/'source'; source.mkdir()
    paths = [f for folder in ('track2', 'track3', 'scripts', 'config', 'doc', 'tests')
             for f in (root/folder).rglob('*')
             if f.is_file() and '__pycache__' not in f.parts and f.suffix != '.pyc']
    paths += [root/n for n in ('AGENTS.md', 'README.md', 'Dockerfile', 'Dockerfile.base',
                              'docker-compose.yml', 'requirements.txt', 'ros_entrypoint.sh',
                              '.dockerignore', '.gitignore') if (root/n).is_file()]
    manifest = {}
    for f in sorted(paths):
        dst = source/f.relative_to(root); dst.parent.mkdir(parents=True, exist_ok=True)
        before = sha(f); shutil.copyfile(f, dst)
        if sha(dst) != before or sha(f) != before:
            raise RuntimeError(f'Source changed while copying: {f}')
        manifest[str(f.relative_to(root))] = before
    # Original commit sources live inside track2/ and are frozen above. The
    # offline reference files are evidence only, never imported by the runner.
    origin = root/'track2/reference_5839746c/origin.json'
    reference_info = json.loads(origin.read_text()) if origin.is_file() else None
    def git(*args):
        return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True, check=True).stdout
    info = dict(created_at=datetime.now(ZoneInfo('Asia/Taipei')).isoformat(timespec='seconds'),
                commit=git('rev-parse', 'HEAD').strip(), sources=manifest,
                source_root=str(root), snapshot=str(source), base_algorithm_reference=reference_info)
    (run/'source_manifest.json').write_text(json.dumps(info, ensure_ascii=False, indent=2)+'\n')
    (run/'git_status.txt').write_text(git('status', '--short'))
    (run/'uncommitted.patch').write_text(git('diff', 'HEAD', '--binary'))
    return run


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--kind', choices=['full', 'test'], required=True)
    a.add_argument('--track', choices=['02', '00'], default='02')
    args = a.parse_args()
    print(create(args.kind, args.track))

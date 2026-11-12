#!/usr/bin/env python3
"""建立唯一 run 目錄並凍結程式快照（只用標準函式庫，相容 Python 3.9）。

命名：outputs/run/track{01,02,00}_{full,test}_YYMMDDHHmm[_0002]
內容：source/、source_manifest.json、git_status.txt、uncommitted.patch、test_info.json
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FROZEN_FOLDERS = ('vslam', 'scripts', 'config', 'tests')
try:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo('Asia/Taipei')
except Exception:  # Windows 無 tzdata 時退回固定 UTC+8（台灣無日光節約）
    TZ = timezone(timedelta(hours=8), 'Asia/Taipei')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def make_run_dir(runs, stem):
    runs.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        run = runs / (stem if n == 1 else '%s_%04d' % (stem, n))
        try:
            run.mkdir()
            return run
        except FileExistsError:
            n += 1


def collect_sources(root):
    files = []
    for folder in FROZEN_FOLDERS:
        base = root / folder
        if base.exists():
            files.extend(f for f in base.rglob('*') if f.is_file()
                         and '__pycache__' not in f.parts and f.suffix != '.pyc')
    files.extend(f for f in root.iterdir() if f.is_file())
    return sorted(files)


def freeze(root, run):
    source = run / 'source'
    source.mkdir()
    manifest = {}
    for f in collect_sources(root):
        rel = f.relative_to(root)
        dst = source / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        before = digest(f)
        shutil.copyfile(f, dst)
        if digest(dst) != before or digest(f) != before:
            raise RuntimeError('Source changed while copying: %s' % f)
        manifest[rel.as_posix()] = before
    return source, manifest


def git(repo, *args):
    try:
        return subprocess.run(['git', '-C', str(repo), *args], capture_output=True,
                              text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        return 'GIT_UNAVAILABLE: %s\n' % exc


def create_run(kind, track, root=ROOT, now=None):
    now = now or datetime.now(TZ)
    stem = 'track%s_%s_%s' % (track, kind, now.strftime('%y%m%d%H%M'))
    run = make_run_dir(root / 'outputs' / 'run', stem)
    source, manifest = freeze(root, run)
    repo = root.parent
    info = dict(created_at=now.isoformat(timespec='seconds'), commit=git(repo, 'rev-parse', 'HEAD').strip(),
                branch=git(repo, 'branch', '--show-current').strip(), sources=manifest,
                source_root=str(root), snapshot=str(source))
    (run / 'git_status.txt').write_text(git(repo, 'status', '--short'), encoding='utf-8')
    (run / 'uncommitted.patch').write_text(git(repo, 'diff', 'HEAD', '--binary'), encoding='utf-8')
    (run / 'source_manifest.json').write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    test_info = dict(run_id=run.name, kind=kind, track=track, status='created', created_at=info['created_at'],
                     commit=info['commit'], version=None, parameters=None, seed=None, input_sha256=None,
                     frame_count=None, timings_seconds={}, error=None, public_score=None,
                     private_score=None, score_source=None)
    (run / 'test_info.json').write_text(json.dumps(test_info, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', choices=['full', 'test'], required=True)
    p.add_argument('--track', choices=['00', '01', '02'], required=True)
    a = p.parse_args()
    if a.kind == 'full' and a.track == '00':
        p.error('track00 只用於 test')
    print(create_run(a.kind, a.track))


if __name__ == '__main__':
    main()

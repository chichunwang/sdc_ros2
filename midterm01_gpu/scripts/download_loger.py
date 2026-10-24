#!/usr/bin/env python3
"""下載官方提供 ZIP，防止路徑穿越，CRC 驗證並完整解壓後才刪 ZIP。"""
import argparse, hashlib, html, json, re, shutil, stat, time, zipfile
from pathlib import Path
from urllib.parse import urlencode
import requests

FILE_ID = '1ZHeWxVrf86GDtHtFY7JyXY7w1vge3Fsa'

def main():
    p = argparse.ArgumentParser(); p.add_argument('--record-dir', type=Path, required=True)
    a = p.parse_args(); a.record_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1] / 'vendor'
    root.mkdir(exist_ok=True); archive = root / 'Code_LoGeR.zip.part'
    target = root / 'LoGeR'
    if target.exists():
        raise SystemExit(f'Existing target preserved: {target}')
    started = time.monotonic(); s = requests.Session()
    url = f'https://drive.usercontent.google.com/download?id={FILE_ID}&export=download'
    r = s.get(url, stream=True, timeout=(30, 120)); r.raise_for_status()
    if 'text/html' in r.headers.get('Content-Type', ''):
        page = r.text
        fields = dict((html.unescape(k), html.unescape(v)) for k, v in
                      re.findall(r'<input[^>]*name="([^"]+)"[^>]*value="([^"]*)"', page))
        if 'uuid' not in fields:
            raise RuntimeError('Google download confirmation unavailable; no archive written')
        url = 'https://drive.usercontent.google.com/download?' + urlencode(fields)
        r = s.get(url, stream=True, timeout=(30, 120)); r.raise_for_status()
    if 'text/html' in r.headers.get('Content-Type', ''):
        raise RuntimeError('Google returned HTML instead of ZIP')
    h = hashlib.sha256(); size = 0; last = 0
    with archive.open('wb') as f:
        for block in r.iter_content(8 * 1024 * 1024):
            f.write(block); h.update(block); size += len(block)
            if size - last > 256 * 1024 * 1024:
                print(f'downloaded {size / 1e9:.2f} GB', flush=True); last = size
    if size != 4648891611:
        raise RuntimeError(f'Unexpected ZIP size: {size}; retained for investigation')
    stage = root / 'Code_LoGeR.extracting'; stage.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            dest = (stage / item.filename).resolve()
            if not dest.is_relative_to(stage.resolve()) or stat.S_ISLNK(item.external_attr >> 16):
                raise RuntimeError(f'Unsafe ZIP member: {item.filename}')
        if shutil.disk_usage(root).free < sum(i.file_size for i in z.infolist()) + 2_000_000_000:
            raise RuntimeError('Insufficient space for extraction')
        bad = z.testzip()
        if bad: raise RuntimeError(f'ZIP CRC failure: {bad}')
        z.extractall(stage)
    children = list(stage.iterdir())
    if len(children) == 1 and children[0].is_dir():
        children[0].rename(target); stage.rmdir()
    else:
        stage.rename(target)
    record = dict(file_id=FILE_ID, url=f'https://drive.google.com/file/d/{FILE_ID}/view',
                  archive_bytes=size, archive_sha256=h.hexdigest(), crc_verified=True,
                  target=str(target), elapsed_seconds=time.monotonic()-started,
                  files=sum(1 for f in target.rglob('*') if f.is_file()))
    (a.record_dir / 'loger_download.json').write_text(json.dumps(record, indent=2)+'\n')
    archive.unlink()
    nested = target / "sdc2026.zip"
    if nested.exists():
        with zipfile.ZipFile(nested) as z:
            if z.testzip(): raise RuntimeError("Nested ZIP CRC error")
            output = target / "archive_sources"; output.mkdir()
            for item in z.infolist():
                if not (output / item.filename).resolve().is_relative_to(output.resolve()):
                    raise RuntimeError("Unsafe nested ZIP path")
            z.extractall(output)
        nested.unlink()
    print(json.dumps(record, indent=2))

if __name__ == '__main__': main()

#!/usr/bin/env python3
"""One-time reversible GPU output consolidation; never modifies saved evidence files."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
OUTPUTS=ROOT/'outputs'
ZONE=ZoneInfo('Asia/Taipei')


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def main():
    if (OUTPUTS/'run').exists():raise ValueError('Target run already exists; preserve and review before migration')
    if subprocess.run(['docker','exec','sdc_midterm01_gpu','pgrep','-x','localization'],capture_output=True).returncode==0:
        raise ValueError('Wait for active localization to finish')
    groups=[p for p in (OUTPUTS/'runs').iterdir() if p.is_dir() and not p.is_symlink()]
    groups += [p for p in (OUTPUTS/'submissions').iterdir() if p.is_dir() and not p.is_symlink()]
    if (OUTPUTS/'figures').is_dir():groups.append(OUTPUTS/'figures')
    births=subprocess.check_output(['stat','-c','%W',*[str(p)for p in groups]],text=True).splitlines()
    targets={};records=[];occupied=set()
    now=datetime.datetime.now(ZONE)
    migration_name=f'track00_test_{now:%y%m%d%H%M}'
    occupied.add(migration_name)
    for old,birth in sorted(zip(groups,births),key=lambda item:item[0].name):
        params=read(old/'params.json');info=read(old/'identity.json')
        match=re.search(r'(?:^|_)track([12])(?:_|$)',old.name)
        track=int(match[1]) if match else 0
        match=re.search(r'Track([12])',str(params.get('map_path','')))
        if match:track=int(match[1])
        if info.get('track') in(1,2):track=info['track']
        full=params.get('max_frames')==(3011 if track==1 else 3299) if track in(1,2) else False
        if not params and any((old/f'track{t}/identity.json').is_file() for t in(1,2)):
            full=all(read(old/f'track{t}/identity.json').get('frames')==(3011 if t==1 else 3299) for t in(1,2))
        if old.name.startswith('three_camera_full'):full=True
        stamp=re.search(r'(?:_|^)(\d{10})(?:_\d+)?$',old.name)
        if stamp:
            try:created=datetime.datetime.strptime(stamp[1],'%y%m%d%H%M').replace(tzinfo=ZONE);basis='original_directory_name'
            except ValueError:stamp=None
        if not stamp:
            epoch=int(birth) if int(birth)>0 else (old/'params.json').stat().st_mtime if (old/'params.json').is_file() else old.stat().st_mtime
            created=datetime.datetime.fromtimestamp(epoch,ZONE);basis='directory_birth_time' if int(birth)>0 else 'metadata_mtime_fallback'
        stem=f'track{track:02d}_{"full" if full else "test"}_{created:%y%m%d%H%M}'
        name=stem;sequence=1
        while name in occupied:sequence+=1;name=f'{stem}_{sequence:04d}'
        occupied.add(name);new=OUTPUTS/'run'/name;targets[old]=new
        oldrel=old.relative_to(OUTPUTS).as_posix()
        records.append({'old':str(old),'new':str(new),'track':track,'scope':'full'if full else'test',
            'created_at':created.isoformat(),'timestamp_basis':basis,'legacy_locations':{oldrel:'.'}})
    # Capture shortcut identities before relocating any target.
    shortcuts=[]
    for category in('current','best'):
        for p in (OUTPUTS/category).iterdir():
            if p.is_symlink():shortcuts.append((p,p.resolve()))
    def translated(path):
        path=Path(path)
        try:parts=path.relative_to(OUTPUTS).parts
        except ValueError:return path
        if len(parts)>=2:
            parent=OUTPUTS/parts[0]/parts[1]
            if parent in targets:return targets[parent].joinpath(*parts[2:])
        if parts and OUTPUTS/parts[0] in targets:
            return targets[OUTPUTS/parts[0]].joinpath(*parts[1:])
        return path
    for alias,target in shortcuts:
        destination=translated(target)
        for record in records:
            new=Path(record['new'])
            if destination.is_relative_to(new):record['legacy_locations'][alias.relative_to(OUTPUTS).as_posix()]=destination.relative_to(new).as_posix()
    # Preserve score provenance in the original source run's own metadata.
    provenance=read(OUTPUTS/'submissions/motion_rts_scored_2610191808/provenance.json')
    for scored in provenance.get('tracks',[]):
        for record in records:
            if record['old']==scored['source_run']:
                record.update(reported_score=scored['reported_score'],scored_csv_sha256=scored['csv_sha256'],
                    uploaded_submission_sha256_independently_verified=False)
    files=[];links=[]
    for old in groups:
        for base,dirs,names in os.walk(old,followlinks=False):
            for name in dirs+names:
                p=Path(base)/name
                if p.is_symlink():
                    raw=os.readlink(p)
                    if raw.startswith('/root/outputs/'):
                        resolved=OUTPUTS/Path(raw).relative_to('/root/outputs')
                        target='/root/outputs/'+translated(resolved).relative_to(OUTPUTS).as_posix()
                    elif raw.startswith('/root/data/'):
                        target=raw
                    else:
                        resolved=Path(os.path.normpath(str(p.parent/raw))) if not os.path.isabs(raw) else Path(raw)
                        target=os.path.relpath(translated(resolved),translated(p.parent))
                    links.append((str(translated(p)),target,raw))
                elif p.is_file():
                    st=p.stat();files.append((str(translated(p)),st.st_ino,st.st_size,st.st_mtime_ns))
    # Key scientific evidence SHA is checked in addition to inode/size/mtime for all originals.
    evidence={str(translated(old/name)):sha(old/name) for old in groups
              for name in('diagnostic.csv','result.txt','sensor_prior.csv','frames.csv','validation.json','source_input.json','params.json')
              if (old/name).is_file()}
    (OUTPUTS/'run').mkdir()
    migration=OUTPUTS/'run'/migration_name;migration.mkdir()
    plan={'created_at':now.isoformat(),'records':records,'original_files':files,'symlinks':links,'evidence_sha256':evidence}
    (migration/'migration.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    for record in records:
        old,new=Path(record['old']),Path(record['new']);old.rename(new)
        previous=read(new/'test_info.json')
        if (new/'test_info.json').exists():(new/'test_info.json').rename(new/'test_info.before_layout.json')
        metadata={**previous,**{k:v for k,v in record.items() if k not in('old','new')},
            'test_kind':previous.get('test_kind',old.name),'original_name':old.name,'original_outputs_root':str(OUTPUTS)}
        (new/'test_info.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    for path,target,old_target in links:
        p=Path(path)
        if os.readlink(p)!=target:p.unlink();p.symlink_to(target)
    notes=migration/'historical_notes';notes.mkdir()
    legacy={}
    for category in('best','current'):
        folder=OUTPUTS/category
        for p in folder.iterdir():
            if p.is_symlink():p.unlink()
            else:
                dest=notes/(category+'_'+p.name);p.rename(dest)
                legacy[category+'/'+p.name]=str(dest.relative_to(migration))
        folder.rmdir()
    for p in list(OUTPUTS.glob('latest_*.txt')):
        dest=notes/p.name;p.rename(dest);legacy[p.name]=str(dest.relative_to(migration))
    (OUTPUTS/'runs').rmdir();(OUTPUTS/'submissions').rmdir()
    (migration/'test_info.json').write_text(json.dumps({'track':0,'scope':'test','created_at':now.isoformat(),
        'test_kind':'output layout migration and verification','original_outputs_root':str(OUTPUTS),'legacy_locations':legacy},ensure_ascii=False,indent=2)+'\n')
    errors=[]
    for path,inode,size,mtime in files:
        p=Path(path)
        if p.name=='test_info.json':p=p.with_name('test_info.before_layout.json')
        st=p.stat()
        if (st.st_ino,st.st_size,st.st_mtime_ns)!=(inode,size,mtime):errors.append(str(p))
    for path,digest in evidence.items():
        if sha(Path(path))!=digest:errors.append(path)
    report={'moved_test_directories':len(records),'original_regular_files_verified_by_inode_size_mtime':len(files),
        'key_evidence_files_sha256_checked':len(evidence),'symlinks_relocated':len(links),'errors':errors,
        'old_category_directories_removed':all(not(OUTPUTS/p).exists()for p in('runs','current','best','submissions','figures'))}
    (migration/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(str(migration));print(json.dumps(report,indent=2))
    if errors:raise RuntimeError('Evidence verification failed; inspect migration.json before further edits')


if __name__=='__main__':main()

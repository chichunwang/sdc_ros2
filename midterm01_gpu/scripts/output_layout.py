"""Single output/run layout; translate historical paths in memory, preserving evidence bytes."""
from functools import lru_cache
import json
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=4)
def _aliases(outputs):
    outputs = Path(outputs)
    replacements = {}
    for info_file in sorted((outputs/'run').glob('*/test_info.json')):
        try:
            info = json.loads(info_file.read_text())
        except (OSError, ValueError):
            continue
        root = info_file.parent
        original = info.get('original_outputs_root', str(outputs))
        for old, inside in info.get('legacy_locations', {}).items():
            target = (root/inside).relative_to(outputs).as_posix()
            for prefix in {str(outputs), original, '/root/outputs', 'outputs', '../outputs', '../../outputs', 'midterm01_gpu/outputs'}:
                replacements[prefix+'/'+old] = prefix+'/'+target
    if not replacements:
        return None, {}
    pattern = re.compile(r'(?<![A-Za-z0-9_/])('+'|'.join(re.escape(s) for s in sorted(replacements,key=len,reverse=True))+r')(?=$|[/\s\"\x27`),\]])')
    return pattern, replacements


def relocate_text(value, outputs=None):
    if not isinstance(value,str) or 'outputs/' not in value:
        return value
    pattern, replacements = _aliases(str(Path(outputs) if outputs else PROJECT_ROOT/'outputs'))
    return pattern.sub(lambda match: replacements[match[1]], value) if pattern else value


def relocate_tree(value, outputs=None):
    if isinstance(value,str): return relocate_text(value,outputs)
    if isinstance(value,list): return [relocate_tree(v,outputs) for v in value]
    if isinstance(value,dict): return {relocate_text(k,outputs):relocate_tree(v,outputs) for k,v in value.items()}
    return value


def saved_path(value):
    return Path(relocate_text(str(value))).expanduser()


def loads_saved_json(value,*args,**kwargs):
    return relocate_tree(json.loads(value,*args,**kwargs))


def load_saved_json(stream,*args,**kwargs):
    return relocate_tree(json.load(stream,*args,**kwargs))


def latest_full_run(root,track):
    for run in sorted((Path(root)/'outputs/run').glob(f'track{track:02d}_full_*'),reverse=True):
        try:
            v=json.loads((run/'validation.json').read_text())
        except (OSError,ValueError):continue
        # The existing RViz publisher consumes one scalar source scale. Runs
        # with per-frame map scales remain valid submissions, but cannot be
        # selected for that publisher until its manifest supports those scales.
        if v.get('replay_supported') is False:continue
        settings_file=run/'horizontal_settings.json'
        try:settings=json.loads(settings_file.read_text()) if settings_file.is_file() else {}
        except (OSError,ValueError):continue
        mode=settings.get('mode','none')
        if mode!='none' and v.get('horizontal_refinement')!=mode:continue
        if (v.get('output_complete_and_finite') is True and v.get('returncode')==0
                and not v.get('validation_errors') and v.get('frames')==(3011 if track==1 else 3299)
                and (run/'result.txt').is_file()):
            return run
    raise ValueError(f'Track{track} 尚無通過完整性檢查的 full 目錄；請用 --run 指定既有定位。')


def scored_reference_run(root,track):
    candidates=[]
    for info in (Path(root)/'outputs/run').glob(f'track{track:02d}_full_*/test_info.json'):
        try:value=json.loads(info.read_text())
        except (OSError,ValueError):continue
        score=value.get('reported_score')
        if isinstance(score,(float,int)) and (info.parent/'sensor_prior.csv').is_file():
            candidates.append((score,str(info.parent),info.parent))
    if not candidates:raise ValueError(f'Track{track} 尚無已評分的 prior reference 紀錄')
    return min(candidates)[2]

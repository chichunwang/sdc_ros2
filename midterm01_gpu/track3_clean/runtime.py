"""Process protection, reproducibility snapshots and post-run diagnostics."""
import os
from pathlib import Path
import subprocess
import sys
import time

from .data import ROOT, digest
from .output import write_json


def active_localizations():
    names = {'run_track3_clean.py', 'run_track3_causal.py', 'run_track3_old_icp.py',
             'run_track3_map_only.py', 'run_map_smoother.py', 'run_map_smoother_gpu.py',
             'benchmark.py', 'test_track3_map_registration.py', 'track3_visibility.py'}
    found = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = (proc/'cmdline').read_bytes().decode().strip('\0').split('\0')
        except (OSError, UnicodeDecodeError):
            continue
        if not args:
            continue
        python_runner = ('python' in Path(args[0]).name and
                         any(Path(a).name in names or a == 'track3_clean.cli' for a in args[1:]))
        if python_runner or Path(args[0]).name in ('localization', 'old_icp_bridge'):
            found.append(int(proc.name))
    return found


def snapshot_sources(run):
    sources = sorted((ROOT/'track3_clean').glob('*.py'))
    sources += [ROOT/'track3_clean/provenance.json', ROOT/'scripts/run_track3_clean.py',
                ROOT/'scripts/plot_run_analysis.py', ROOT/'scripts/run_analysis.py']
    hashes = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        target = run/'pipeline_sources'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        content = source.read_bytes()
        target.write_bytes(content)
        hashes[str(relative)] = digest(target)
        if digest(source) != hashes[str(relative)]:
            raise RuntimeError('Source changed during snapshot: '+str(relative))
    write_json(run/'pipeline_source_sha256.json', hashes)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    (run/'git_commit.txt').write_text(commit.stdout if commit.returncode == 0 else 'unavailable\n')
    diff = subprocess.run(['git', 'diff', '--binary', 'HEAD'], cwd=ROOT, capture_output=True)
    if diff.returncode:
        raise RuntimeError('Could not record the uncommitted worktree diff')
    (run/'worktree.patch').write_bytes(diff.stdout)
    return hashes


def configure_cuda(torch, budget_gib):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU fallback')
    free, total = torch.cuda.mem_get_info()
    size = int(budget_gib*1024**3)
    if size > total:
        raise ValueError('Torch allocator budget exceeds device total')
    torch.cuda.set_per_process_memory_fraction(size/total)
    torch.use_deterministic_algorithms(True)
    return dict(budget_gib=budget_gib, budget_bytes=size, initial_free_bytes=free, device_total_bytes=total,
                scope='PyTorch allocator only; excludes driver/NVRTC/cuSOLVER and other processes',
                torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
                normal_eigh_batch_size=256, cpu_fallback=False)


def cuda_memory(torch):
    result = {}
    if torch is None:
        return result
    for key, fn in [('peak_allocated_bytes', torch.cuda.max_memory_allocated),
                    ('peak_reserved_bytes', torch.cuda.max_memory_reserved)]:
        try:
            result[key] = fn()
        except Exception as error:
            result[key+'_error'] = str(error)
    return result


def analyze_full(run, summary, *, full):
    """Reuse the read-only plotting service across a process boundary."""
    if not full or summary['returncode'] != 0:
        return
    status = dict(status='failed', localization_success=True, trajectory_modified=False,
                  localization_rerun=False, time_scope='analysis only; excluded from localization timings')
    started = time.perf_counter()
    try:
        command = [sys.executable, str(ROOT/'scripts/plot_run_analysis.py'), '--run', str(run)]
        status['command'] = command
        with (run/'analysis.log').open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, MPLBACKEND='Agg', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1'))
        status['returncode'] = result.returncode
        if result.returncode:
            raise RuntimeError('Plot process failed; see analysis.log')
        import json
        manifest = json.loads((run/'analysis/manifest.json').read_text())
        if manifest.get('complete') is not True:
            raise RuntimeError('Analysis manifest incomplete')
        status.update(status='complete', output=str(run/'analysis'))
    except Exception as error:
        status['error'] = f'{type(error).__name__}: {error}'
    status['elapsed_seconds'] = time.perf_counter()-started
    try:
        write_json(run/'analysis_status.json', status)
    except OSError as error:
        print(f'Analysis status write failed: {error}; localization results preserved.', file=sys.stderr, flush=True)

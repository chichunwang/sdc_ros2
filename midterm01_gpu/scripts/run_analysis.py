"""Full 定位成功後的獨立繪圖階段；不改定位結果或其成功狀態。"""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FRAMES = {1: 3011, 2: 3299, 3: 3142}


def full_run_ready(summary, *, full):
    """完整且驗證成功才繪圖；名稱或 full 旗標本身不代表完成。"""
    expected = EXPECTED_FRAMES.get(summary.get('track'))
    return bool(full and expected and summary.get('returncode') == 0
                and summary.get('output_complete_and_finite') is True
                and summary.get('submission_frame_set_complete') is True
                and summary.get('validation_errors') == []
                and summary.get('frames') == expected)


def generate_after_full(run, summary, *, full):
    """只處理當次 run；繪圖失敗獨立記錄，保留定位成功與所有數值。"""
    if not full_run_ready(summary, full=full):
        return {'status': 'skipped', 'reason': 'not a complete validated full run'}
    run = Path(run).resolve()
    started = time.perf_counter()
    status = {'status': 'failed', 'localization_success': True,
              'localization_rerun': False, 'trajectory_modified': False,
              'time_scope': 'analysis only; excluded from localization timings',
              'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        venv_python = ROOT/'outputs/ml_runtime/venv/bin/python'
        python = str(venv_python) if venv_python.is_file() else sys.executable
        command = [python, str(ROOT/'scripts/plot_run_analysis.py'), '--run', str(run)]
        status['command'] = command
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', MPLBACKEND='Agg')
        print(f'Analysis: generating {run}/analysis', flush=True)
        with (run/'analysis.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        status['returncode'] = result.returncode
        if result.returncode:
            raise RuntimeError('Plot process failed; see analysis.log (requires NumPy/Matplotlib)')
        manifest = json.loads((run/'analysis/manifest.json').read_text())
        if manifest.get('complete') is not True:
            raise ValueError('Analysis manifest is not complete')
        status.update(status='complete', output=str(run/'analysis'), manifest='analysis/manifest.json')
        print(f'Analysis: {run}/analysis', flush=True)
    except Exception as error:
        status['error'] = f'{type(error).__name__}: {error}'
        print(f'Analysis failed: {status["error"]}; localization results preserved.', flush=True)
    finally:
        status['elapsed_seconds'] = time.perf_counter()-started
        try:
            (run/'analysis_status.json').write_text(
                json.dumps(status, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        except OSError as error:
            print(f'Analysis status could not be saved: {error}', flush=True)
    return status

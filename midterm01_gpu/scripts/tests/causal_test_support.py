"""Keep all synthetic fixture files inside the required GPU run layout."""
import datetime
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo


def fixture_run():
    root = Path(__file__).resolve().parents[2]/'outputs/run'
    requested = os.environ.get('SDC_CAUSAL_TEST_RUN')
    if requested:
        path = Path(requested).resolve()
        path.relative_to(root.resolve())
        if not path.is_dir():
            raise ValueError('SDC_CAUSAL_TEST_RUN must be an existing GPU run')
        return path
    now = datetime.datetime.now(ZoneInfo('Asia/Taipei'))
    stem = 'track00_test_'+now.strftime('%y%m%d%H%M')
    root.mkdir(parents=True, exist_ok=True)
    i = 1
    while True:
        path = root/(stem if i == 1 else f'{stem}_{i:04d}')
        try:
            path.mkdir()
            break
        except FileExistsError:
            i += 1
    (path/'test_info.json').write_text(json.dumps({
        'track': 0, 'scope': 'test', 'created_at': now.isoformat(),
        'test_kind': 'synthetic_causal_mechanisms', 'localization_executed': False,
        'algorithm': 'track3_causal_vertical_imu_kalman_sim3_v1'}, indent=2)+'\n')
    return path

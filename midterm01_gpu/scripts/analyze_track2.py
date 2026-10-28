#!/usr/bin/env python3
"""完成 full 的唯讀圖表重試；必須從該 run 的 source 執行。"""
import argparse
from pathlib import Path
import sys
import time
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from track2.analysis import analyze
from track3.io import write_json
from run_track2 import verify_snapshot


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run-dir', type=Path, required=True)
    run = parser.parse_args().run_dir.resolve()
    verify_snapshot(run)
    start = time.perf_counter()
    try:
        analyze(run)
    except Exception as exc:
        (run/'analysis_failure.log').write_text(traceback.format_exc())
        write_json(run/'analysis_status.json', dict(status='failed', failure=str(exc), elapsed_seconds=time.perf_counter()-start))
        raise
    write_json(run/'analysis_status.json', dict(status='completed', elapsed_seconds=time.perf_counter()-start))

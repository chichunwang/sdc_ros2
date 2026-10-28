#!/usr/bin/env python3
"""Launch this run's RViz on the host display before starting localization."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import shutil
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args()
    run=a.run_dir.resolve(); source_config=run/'source/config/track1.rviz'
    config=run/'rviz.runtime.rviz'
    status=dict(status='starting',display=os.environ.get('DISPLAY'),config=str(config))
    started=time.monotonic()
    try:
        if not os.environ.get('DISPLAY'):raise RuntimeError('DISPLAY is not configured; run scripts/dev.sh up from the Ubuntu desktop terminal')
        if not source_config.is_file():raise FileNotFoundError(source_config)
        shutil.copyfile(source_config,config)
        env=os.environ.copy();env['ROS_LOG_DIR']=str(run/'rviz_ros_log')
        with (run/'rviz.log').open('x') as log:
            process=subprocess.Popen(['rviz2','-d',str(config)],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        time.sleep(2)
        if process.poll() is not None:raise RuntimeError(f'RViz exited ({process.returncode}); see {run/"rviz.log"}')
        status.update(status='running',pid=process.pid)
        print(f'RViz opened (pid={process.pid})',flush=True)
    except Exception as error:
        status.update(status='failed',error=f'{type(error).__name__}: {error}')
        print(status['error'],flush=True)
    status['startup_seconds']=time.monotonic()-started
    (run/'rviz_status.json').write_text(json.dumps(status,indent=2)+'\n')
    if status['status']=='failed' and not (run/'test_info.json').exists():
        failure=dict(status='failed',mode='full' if '_full_' in run.name else 'test',
                     frame_count=0,version=None,stage='rviz_startup',
                     failure=status['error'],elapsed_seconds=status['startup_seconds'],
                     score=None,score_source=None)
        (run/'test_info.json').write_text(json.dumps(failure,indent=2)+'\n')
    return 0 if status['status']=='running' else 1

if __name__=='__main__':raise SystemExit(main())

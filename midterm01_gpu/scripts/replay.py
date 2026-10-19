#!/usr/bin/env python3
"""Ubuntu host entry point: saved-run replay plus an isolated RViz GUI."""
from output_layout import loads_saved_json, saved_path

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

from replay_run import container_path, load_run, sha256, write_replay


ROOT = Path(__file__).resolve().parents[1]
BINARY = '/root/midterm/install/localization/lib/localization/replay_localization'
RVIZ = '/root/midterm/src/localization/config/replay.rviz'


def capture(command, **kwargs):
    return subprocess.check_output(command, text=True, **kwargs).strip()


def environment():
    values = {}
    path = ROOT / '.env'
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip() and not line.lstrip().startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                values[key.strip()] = value.strip().strip('"\'')
    values.update({key: os.environ[key] for key in ('MIDTERM_GPU_DATA_DIR', 'MIDTERM_GPU_CONTAINER_NAME', 'ROS_DOMAIN_ID') if key in os.environ})
    data = Path(values.get('MIDTERM_GPU_DATA_DIR', '../midterm01/data'))
    return ((ROOT / data).resolve(), values.get('MIDTERM_GPU_CONTAINER_NAME', 'sdc_midterm01_gpu'), values.get('ROS_DOMAIN_ID', '2'))


def new_output(track=0):
    from accuracy.benchmark import create_run_directory
    return create_run_directory(ROOT / 'outputs/run', 'replay', track=track)


def start_gui(container, domain, output, data_root):
    display = os.environ.get('DISPLAY')
    if not display or not Path('/tmp/.X11-unix').is_dir():
        raise ValueError('請在 Ubuntu 桌面終端執行（需 DISPLAY／X11）；或用 --headless 只發布訊息。')
    authority = output / 'display.xauth'
    authority.touch(mode=0o600)
    # Share only this display's cookie; do not change xhost access controls.
    entries = subprocess.check_output(['xauth', 'nlist', display])
    if not entries.strip():
        raise ValueError('無法取得目前桌面的 Xauthority；請從登入桌面的終端執行。')
    entries = b'\n'.join(b'ffff' + row[4:] for row in entries.splitlines()) + b'\n'
    subprocess.run(['xauth', '-f', str(authority), 'nmerge', '-'], input=entries, check=True)
    image = capture(['docker', 'inspect', '--format', '{{.Image}}', container])
    name = f'sdc_midterm01_replay_rviz_{os.getpid()}'
    command = ['docker', 'run', '-d', '--init', '--name', name, '--network', 'host',
               '--volumes-from', container + ':ro',
               '--mount', 'type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix,readonly',
               '-e', f'DISPLAY={display}', '-e', f'XAUTHORITY={container_path(authority, ROOT, data_root)}',
               '-e', f'ROS_DOMAIN_ID={domain}', '-e', 'ROS_LOCALHOST_ONLY=1',
               '-e', 'QT_X11_NO_MITSHM=1', '-e', 'LIBGL_ALWAYS_SOFTWARE=1',
               image, 'bash', '-c', 'source /opt/ros/humble/setup.bash && exec rviz2 -d ' + shlex.quote(RVIZ)]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, KeyboardInterrupt):
        subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        raise
    return name


def main():
    parser = argparse.ArgumentParser(description='已保存定位結果的 RViz 回放；不重新計算定位。')
    parser.add_argument('--track', type=int, choices=(1, 2), default=1)
    parser.add_argument('--run', type=saved_path, help='定位 run，或含 trackN/localization.json 的三相機完整 run；省略使用該 Track 最近通過完整性檢查的 full 目錄。')
    parser.add_argument('--start-frame', type=int, default=0, help='原 run 內的 0-based 起點')
    parser.add_argument('--frames', type=int, default=0, help='回放幀數；0=從起點到結尾')
    parser.add_argument('--speed', type=float, default=1., help='相對原始時間速度，預設 1')
    parser.add_argument('--paused', action='store_true', help='開啟時先停在第一幀')
    parser.add_argument('--loop', action='store_true')
    parser.add_argument('--headless', action='store_true', help='不開 RViz，仍發布 ROS topics／TF')
    parser.add_argument('--exit-at-end', action='store_true', help='播完退出；預設停在最後一幀並持續發布')
    parser.add_argument('--check-only', action='store_true', help='唯讀核對輸入，不啟動容器、不寫檔')
    parser.add_argument('--prepare-only', action='store_true', help='只產生回放 manifest／座標 CSV，不啟動 ROS')
    args = parser.parse_args()
    if not math.isfinite(args.speed) or not 0 < args.speed <= 100:
        parser.error('--speed 必須在 (0,100]。')
    data_root, container, domain = environment()
    gui, output, process = None, None, None
    runtime_container, pid_file = container, None
    lock = None
    try:
        print('讀取既有定位結果，核對點雲／座標契約…', flush=True)
        data = load_run(ROOT, data_root, args.run, args.track, args.start_frame, args.frames)
        data['viewer_contract_version'] = 2
        data['source_cloud_frame'] = 'map'
        data['display_transform'] = 'Saved R*(ply_scale*p_raw)+t is baked into each cloud; dynamic TF is for source pose/view tracking.'
        print(f"來源：{data['run']}\nTrack{args.track}：{len(data['records'])} / {data['total_frames']} 幀；ply_scale={data['params']['ply_scale']}", flush=True)
        if data['map_cache']:
            print('顯示地圖使用已核對 Track／map SHA256／leaf 的唯讀快取：' + data['map_cache']['file'], flush=True)
        if args.check_only:
            print('索引／整數時間／旋轉／來源點雲核對通過。')
            return 0
        output = new_output(args.track)
        source_files = [ROOT / name for name in (
            'scripts/replay.py', 'scripts/replay_run.py',
            'workspace/src/localization/src/replay_localization.cpp',
            'workspace/src/localization/config/replay.rviz',
            'workspace/src/localization/CMakeLists.txt', 'workspace/src/localization/package.xml')]
        data['replay_tool_sha256'] = {str(path.relative_to(ROOT)): sha256(path) for path in source_files}
        data['git_commit'] = capture(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'])
        for path in source_files:
            target = output / 'replay_sources' / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
        (output / 'worktree.patch').write_bytes(subprocess.check_output(
            ['git', '-C', str(ROOT), 'diff', '--', *[str(path.relative_to(ROOT)) for path in source_files]]))
        write_replay(data, output)
        print(f'回放紀錄：{output}\n座標／朝向：{output / "poses.csv"}', flush=True)
        if args.prepare_only:
            return 0
        lock = (ROOT / 'outputs/.replay.lock').open('a')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('已有一個回放正在使用 /replay topics；請先關閉該回放。')
        subprocess.run(['bash', str(ROOT / 'scripts/dev.sh'), 'up'], check=True)
        state = loads_saved_json(capture(['docker', 'inspect', container]))[0]
        mounts = {m['Destination']: Path(m['Source']).resolve() for m in state['Mounts']}
        if mounts.get('/root/data') != data_root or mounts.get('/root/outputs') != (ROOT / 'outputs').resolve():
            raise ValueError('執行中 GPU container 的 data／outputs 掛載與主機設定不符。')
        configured = dict(item.split('=', 1) for item in state['Config']['Env'] if '=' in item)
        domain = configured.get('ROS_DOMAIN_ID', domain)
        if subprocess.run(['docker', 'exec', container, 'test', '-x', BINARY]).returncode:
            subprocess.run(['bash', str(ROOT / 'scripts/dev.sh'), 'build-workspace', '--packages-select', 'localization'], check=True)
        (output / 'execution.json').write_text(json.dumps(dict(
            image=state['Image'], binary_sha256=capture(['docker', 'exec', container, 'sha256sum', BINARY]).split()[0],
            render='headless' if args.headless else 'software OpenGL', ros_domain_id=domain), indent=2) + '\n')
        output_in_container = container_path(output, ROOT, data_root)
        params = dict(frame_manifest=output_in_container + '/frames.tsv', map_file=data['map_file'],
                      speed=args.speed, paused=args.paused, loop=args.loop, exit_at_end=args.exit_at_end,
                      frame_index=0)
        for key in ('ply_scale', 'input_voxel_leaf_size', 'map_voxel_leaf_size', 'submap_half_size', 'submap_half_height'):
            params[key] = float(data['params'][key])
        if data['map_cache']:
            params['map_voxel_leaf_size'] = 0.  # cache has already been filtered at the saved run's leaf
        (output / 'params.yaml').write_text(json.dumps({'localization_replay': {'ros__parameters': params}}, indent=2) + '\n')
        if not args.headless:
            gui = start_gui(container, domain, output, data_root)
            print(f'GUI／回放容器：{gui}', flush=True)
        # GUI and publisher share one disposable container, including its DDS
        # shared memory. The dataset/workspace/results mounts stay read only.
        runtime_container = gui or container
        pid_file = '/tmp/' + output.name + '.pid'
        command = ['docker', 'exec'] + (['-it'] if sys.stdin.isatty() else []) + [runtime_container, 'bash', '-c',
                   'source /opt/ros/humble/setup.bash && source /root/midterm/install/setup.bash && '
                   + 'echo $$ > ' + shlex.quote(pid_file) + ' && exec '
                   + BINARY + ' --ros-args --params-file ' + shlex.quote(output_in_container + '/params.yaml')]
        (output / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
        print('空白鍵：播放／暫停；n／b：下／上一幀；+／-：速度；r：回第一幀；q：離開。', flush=True)
        process = subprocess.Popen(command)
        returncode = process.wait()
        if gui:
            status = subprocess.run(['docker', 'inspect', '--format', '{{.State.Running}}', gui],
                                    capture_output=True, text=True)
            if status.returncode or status.stdout.strip() != 'true':
                print('RViz 提前結束，請查看 rviz.log。', file=sys.stderr)
                returncode = returncode or 1
        return returncode
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        print(f'回放失敗：{error}', file=sys.stderr)
        return 1
    finally:
        if process is not None and process.poll() is None:
            subprocess.run(['docker', 'exec', runtime_container, 'bash', '-c',
                            'if test -f ' + shlex.quote(pid_file) + '; then kill -TERM "$(cat ' + shlex.quote(pid_file) + ')" 2>/dev/null || true; fi'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            process.wait(timeout=10)
        if pid_file:
            subprocess.run(['docker', 'exec', runtime_container, 'rm', '-f', pid_file],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if gui:
            logs = subprocess.run(['docker', 'logs', gui], capture_output=True)
            (output / 'rviz.log').write_bytes(logs.stdout + logs.stderr)
            subprocess.run(['docker', 'rm', '-f', gui], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if output:
            (output / 'display.xauth').unlink(missing_ok=True)
        if lock:
            lock.close()


if __name__ == '__main__':
    raise SystemExit(main())

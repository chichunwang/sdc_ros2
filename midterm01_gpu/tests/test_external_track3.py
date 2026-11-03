"""生成資料檢查，不執行任何真實定位。"""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from external_track3 import EXPECTED_FRAMES, stage_commands, sha256, write_json, LIMITATIONS, inspect_data
from analyze_external_track3 import validate_result, analyze
from external_parameters import get_parameters
from convert_external_map import convert


def author_source():
    frozen = SCRIPTS.parents[1] / "external_source"
    return frozen if frozen.is_dir() else SCRIPTS.parents[1] / get_parameters(1)["author_source"]


def make_run(tmp_path, track=3):
    run = tmp_path / "SYNTHETIC_FORMAT_ONLY"
    work = run / "work"
    work.mkdir(parents=True)
    p = get_parameters(track)
    write_json(run / "parameters.json", p)
    n = p["expected_frames"]
    with (run / p["result_name"]).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["idx", "x", "y", "yaw"])
        writer.writerows((i, i*.1, 2., .01) for i in range(n))
    shutil.copyfile(run / p["result_name"], work / p["author_output_name"])
    with (run / "frame_manifest.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["idx", "original_idx", "timestamp_ns", "path", "sha256"])
        writer.writerows((i, i, 1727239790258245070+i*36000000, "synthetic", "synthetic") for i in range(n))
    for name in ("track_causal", "reg_causal_r12", "reg_causal_r15", "reg_causal_r18", "reg_causal_r25", "reg_causal_hybrid"):
        np.savez(work / f"{name}.npz", frame=np.arange(n), ng_frac03=np.full(n,.8),
                 xy_min_info=np.full(n,3.), range_pick=np.zeros(n,dtype=int))
    np.save(work / "scale_hc.npy", np.full(n,8.))
    return run


@pytest.mark.parametrize("track", [1, 2, 3])
def test_contract_and_read_only_full_plots(tmp_path, track):
    run = make_run(tmp_path, track)
    assert validate_result(run)["accuracy_verified"] is False
    write_json(run / "analysis_input_info.json", dict(mode="full", status="completed", limitations=LIMITATIONS))
    for name in ("external_source_manifest.json", "environment.json", "data_check.json"):
        write_json(run / name, {"synthetic": True})
    (run / "stage_timings.csv").write_text("stage,elapsed_seconds,returncode\nsynthetic,1.25,0\n")
    before = {p: sha256(p) for p in run.rglob("*") if p.is_file()}
    analyze(run)
    for stem in ("01_trajectory", "02_pose_changes", "03_registration_timing", "04_registration_and_scale"):
        for ext in ("png", "pdf"):
            assert (run / "analysis" / f"{stem}.{ext}").stat().st_size > 1000
    assert all(sha256(p) == digest for p, digest in before.items())
    manifest = json.loads((run / "analysis/manifest.json").read_text())
    assert all(sha256(run / name) == digest for name, digest in manifest["sources"].items())
    assert validate_result(run)["track"] == track


def test_contract_rejects_bad_frame_order_and_nan(tmp_path):
    run = make_run(tmp_path)
    np.savez(run / "work/reg_causal_r12.npz", frame=np.arange(EXPECTED_FRAMES)[::-1])
    with pytest.raises(ValueError, match="frame"):
        validate_result(run)
    np.savez(run / "work/reg_causal_r12.npz", frame=np.arange(EXPECTED_FRAMES))
    text = (run / "result_3.csv").read_text()
    (run / "result_3.csv").write_text(text.replace("0,0.0,2.0,0.01", "0,nan,2.0,0.01", 1))
    with pytest.raises(ValueError, match="非有限"):
        validate_result(run)


def test_stage_arguments_match_author_shell_without_localization(tmp_path):
    # 假 Python 只記錄 argv，不讀資料、不做 map/PLY/IMU 計算。
    source = author_source()
    assert source.is_dir()
    fake = tmp_path / "fake_python"
    fake.write_text("#!/usr/bin/env python3\nimport os,sys,json\nwith open(os.environ['ARGV_LOG'],'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n")
    fake.chmod(0o755)
    log = tmp_path / "argv.jsonl"
    data = tmp_path / "data"
    work = tmp_path / "output"
    env = dict(os.environ, PYTHON=str(fake), PROCS="2", ARGV_LOG=str(log))
    track = 1 if (source / "run_track1_causal.sh").is_file() else 3
    subprocess.run(["bash", str(source / f"run_track{track}_causal.sh"), str(data), str(work)], env=env, check=True, capture_output=True)
    actual = [json.loads(line) for line in log.read_text().splitlines()]
    planned = [argv[1:] for _, argv in stage_commands(source, data, work, get_parameters(track))]
    # 作者使用 scripts/ 相對路徑，包裝器使用相同檔案的絕對路徑。
    for rows in (actual, planned):
        for row in rows:
            row[0] = Path(row[0]).name
    assert actual == planned


def test_ascii_map_conversion_preserves_values(tmp_path):
    src = tmp_path / "synthetic.pcd"
    dst = tmp_path / "map.npy"
    src.write_text("VERSION .7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\nWIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA ascii\n1 2 3 4\n-5 6 7 8\n")
    subprocess.run([sys.executable, str(author_source() / "scripts/convert_map.py"), str(src), str(dst)], check=True, capture_output=True)
    assert np.array_equal(np.load(dst), np.array([[1,2,3,4],[-5,6,7,8]], dtype=np.float32))


def test_binary_map_uses_xyz_only_and_preserves_source(tmp_path):
    import struct
    header = b"FIELDS tl th x y z intensity\nSIZE 4 4 4 4 4 4\nTYPE U U F F F F\nCOUNT 1 1 1 1 1 1\nPOINTS 2\nDATA binary\n"
    outputs = []
    for i, time_value in enumerate((0, 4294967295)):
        source = tmp_path / f"map{i}.pcd"
        destination = tmp_path / f"map{i}.npy"
        source.write_bytes(header + struct.pack("<IIffff", time_value,time_value,1.,2.,3.,999.) + struct.pack("<IIffff", time_value,time_value,-4.,5.,6.,-999.))
        before = sha256(source)
        convert(source, destination)
        assert sha256(source) == before
        outputs.append(sha256(destination))
        assert np.array_equal(np.load(destination), np.array([[1,2,3,0],[-4,5,6,0]], dtype=np.float32))
        with pytest.raises(FileExistsError):
            convert(source, destination)
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("track", [1, 2])
def test_multitrack_commands_do_not_copy_track3_overrides(tmp_path, track):
    p = get_parameters(track)
    commands = dict(stage_commands(author_source(), tmp_path / f"Track{track}", tmp_path / "work", p))
    assert "--init-search" not in commands["05_sequential"]
    assert commands["05_sequential"][commands["05_sequential"].index("--backfill")+1] == "0"
    ekf = commands["08_ekf"]
    assert ekf[ekf.index("--gyro-dt")+1] == "0"
    assert commands["05_sequential"][commands["05_sequential"].index("--gnss")+1] == str(tmp_path / f"Track{track}")
    assert ekf[ekf.index("--lever")+1] == "0.7086,-0.5995"
    assert ekf[ekf.index("--cam-yaw-offset-deg")+1] == "-0.4455"
    assert commands["09_yaw"][-1].endswith(f"track{track}_causal_hybrid_yawmotion.csv")
    assert Path(commands["01_convert_map"][1]).name == ("convert_external_map.py" if track == 2 else "convert_map.py")


def test_track2_parameter_changes_are_applied_to_author_gnss_arguments(tmp_path):
    p = get_parameters(2)
    p.update(gnss_lever_m=(.2, -.1), lever_m=(.2, -.1), cam_yaw_offset_deg=.3, gnss_sigma_m=1.5)
    commands = dict(stage_commands(author_source(), tmp_path/'Track2', tmp_path/'work', p))
    seq, ekf = commands['05_sequential'], commands['08_ekf']
    assert seq[seq.index('--gnss-lever')+1] == '0.2,-0.1'
    assert seq[seq.index('--gnss-sigma')+1] == '1.5'
    assert ekf[ekf.index('--lever')+1] == '0.2,-0.1'
    assert ekf[ekf.index('--cam-yaw-offset-deg')+1] == '0.3'


def test_gnss_tracking_data_check_requires_a_usable_topic_and_two_fixes(tmp_path, monkeypatch):
    import sqlite3
    monkeypatch.syspath_prepend(str(author_source()/'src'))
    p = get_parameters(1); p['expected_frames'] = 2
    (tmp_path/'raw_ply').mkdir()
    for i in range(2):
        (tmp_path/'raw_ply'/f'{i:06d}_ts_{1_000_000_000+i*36_000_000}.ply').write_bytes(b'filename inventory only')
    (tmp_path/'init_guess.txt').write_text('not parsed in environment inventory')
    (tmp_path/'map.pcd').write_text('FIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\nPOINTS 1\nDATA ascii\n0 0 0 0\n')
    with sqlite3.connect(tmp_path/p['bag_name']) as db:
        db.execute('CREATE TABLE topics(id INTEGER, name TEXT, type TEXT, serialization_format TEXT)')
        db.execute('CREATE TABLE messages(topic_id INTEGER, timestamp INTEGER, data BLOB)')
        db.execute("INSERT INTO topics VALUES(1,'/imu/data','sensor_msgs/msg/Imu','cdr')")
    result = inspect_data(tmp_path, p)
    assert not result['ready'] and result['gnss_required'] and not result['gnss_topic_valid']
    with sqlite3.connect(tmp_path/p['bag_name']) as db:
        db.execute("INSERT INTO topics VALUES(2,'/localizer_node/gnss_pose','geometry_msgs/msg/PoseStamped','cdr')")
        db.execute("INSERT INTO messages VALUES(2,1000,x'')")
    assert not inspect_data(tmp_path, p)['ready']
    with sqlite3.connect(tmp_path/p['bag_name']) as db:
        db.execute("INSERT INTO messages VALUES(2,2000,x'')")
    assert inspect_data(tmp_path, p)['ready']

#!/usr/bin/env python3
"""保留作者算法的 CPU 研究包裝器；檢查與 full 分開，原碼不改寫。"""
from __future__ import annotations

import csv
import difflib
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo
from external_parameters import get_parameters
from convert_external_map import read_pcd_header

# 舊 Track3 import 的相容常數；實際執行一律由 parameters.json 決定。
EXPECTED_FRAMES = 3142
AUTHOR_SOURCE = "midterm01/others code"
VERSION = "external_author_causal_cpu_multitrack_v2"
LIMITATIONS = [
    "作者 CPU 原始流程；不是現行 CUDA 算法的 CPU fallback。",
    "僅在執行副本修正兩處檔案 buffering=0，避免 Linux Polars 讀錯 buffered reader 位置；算法與參數不改。",
    "保留作者全段首個有效配準初始化與最後 yaw 覆換；Track3 另保留 gyro-dt=0.14，未確認符合競賽因果規則。",
    "Track1/2 是移植作者無 GNSS 融合流程，不是作者另外提供或評分的 Track1/2 版本；高度與尺度假設未標定。",
    "沒有獨立真值或作者已評分 CSV，輸出完整與合成測試不代表精度或作者成績已重現。",
    "原程式無逐幀耗時及最終完整 3D pose，不補造 timings.csv 或 poses.txt。",
]


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_snapshot(run):
    manifest = json.loads((run / "external_source_manifest.json").read_text())
    if sha256(run / "parameters.json") != manifest["parameters_sha256"]:
        raise ValueError("參數快照 SHA 不符")
    if sha256(run / manifest["compatibility_patch"]) != manifest["compatibility_patch_sha256"]:
        raise ValueError("IO 修正 patch SHA 不符")
    for name, digest in manifest["sources"].items():
        if sha256(run / "external_source" / name) != digest:
            raise ValueError(f"作者快照 SHA 不符：{name}")
    for name, digest in manifest["original_sources"].items():
        if sha256(run / "external_original" / name) != digest:
            raise ValueError(f"作者原始備份 SHA 不符：{name}")
    local = json.loads((run / "source_manifest.json").read_text())
    for name, digest in local["sources"].items():
        if sha256(run / "source" / name) != digest:
            raise ValueError(f"包裝器快照 SHA 不符：{name}")
    return manifest


def prepare(mode, track=3):
    root = Path(__file__).resolve().parents[1]
    repo = root.parent
    parameters = get_parameters(track)
    author = repo / AUTHOR_SOURCE
    if not (author / "run_track3_causal.sh").is_file():
        raise FileNotFoundError(author)
    if mode == "--full":
        data = repo / parameters["data_relative"]
        missing = [str(data / name) for name in
                   ("map.pcd", "init_guess.txt", parameters["bag_name"], "raw_ply")
                   if not (data / name).exists()]
        if missing:
            raise FileNotFoundError(f"尚未準備 Track{track} 資料，未啟動 full：" + ", ".join(missing))
    creator = f"create_track{track}_run.py" if track in (1, 2) else "create_run.py"
    command = [sys.executable, str(root / "scripts" / creator),
               "--kind", "full" if mode == "--full" else "test",
               "--track", f"{track:02d}"]
    run = Path(subprocess.check_output(command, text=True).strip())
    manifest = {}
    original_manifest = {}
    patches = []
    compatibility = {
        "src/sdcloc/ply.py": ("with path.open('rb') as f:", "with path.open('rb', buffering=0) as f:"),
        "scripts/convert_map.py": ("with src.open('rb') as f:", "with src.open('rb', buffering=0) as f:"),
    }
    for src in sorted(author.rglob("*")):
        if not src.is_file() or "__pycache__" in src.parts or src.suffix == ".pyc":
            continue
        relative = src.relative_to(author)
        dst = run / "external_source" / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        before = sha256(src)
        shutil.copyfile(src, dst)
        if before != sha256(dst) or before != sha256(src):
            raise RuntimeError(f"複製期間來源變動：{src}")
        original = run / "external_original" / relative
        original.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(dst, original)
        original_manifest[str(relative)] = before
        if str(relative) in compatibility:
            old, new = compatibility[str(relative)]
            content = dst.read_text()
            if content.count(old) != 1:
                raise ValueError(f"來源已變更，不能套用已確認的 IO 修正：{relative}")
            changed = content.replace(old, new)
            dst.write_text(changed)
            patches.extend(difflib.unified_diff(content.splitlines(True), changed.splitlines(True),
                           fromfile=f"external_original/{relative}", tofile=f"external_source/{relative}"))
        manifest[str(relative)] = sha256(dst)
    (run / "external_compatibility.patch").write_text("".join(patches))
    write_json(run / "parameters.json", parameters)
    write_json(run / "external_source_manifest.json", dict(source_root=str(author), sources=manifest,
               original_sources=original_manifest, compatibility_patch="external_compatibility.patch",
               compatibility_patch_sha256=sha256(run / "external_compatibility.patch"),
               parameters_sha256=sha256(run / "parameters.json")))
    write_json(run / "test_info.json", dict(
        status="prepared", mode="full" if mode == "--full" else "check", version=VERSION,
        backend="author_original_cpu", track=track, frame_count=0, score=None, score_source=None,
        created_at=datetime.now(ZoneInfo("Asia/Taipei")).isoformat(timespec="seconds"),
        limitations=LIMITATIONS))
    (run / "invocation.txt").write_text(f"scripts/run_external_track{track}.sh {mode}\n")
    print(run)


def inspect_data(data, parameters=None):
    parameters = get_parameters(3) if parameters is None else parameters
    expected = parameters["expected_frames"]
    required = ["map.pcd", "init_guess.txt", parameters["bag_name"], "raw_ply"]
    missing = [name for name in required if not (data / name).exists()]
    result = dict(track=parameters["track"], path=str(data), missing=missing, available=not missing)
    if not missing:
        from sdcloc.frames import load_frames
        stamps, indices = load_frames(data / "raw_ply")
        result.update(frame_count=len(stamps), expected_frames=expected,
                      frame_count_valid=len(stamps) == expected)
        header = read_pcd_header(data / "map.pcd")
        map_format = header["DATA"][0]
        fields = header["FIELDS"]
        format_ok = (map_format == "ascii" and fields == ["x", "y", "z", "intensity"]
                     if parameters["map_reader"] == "author_ascii" else
                     map_format == "binary" and all(x in fields for x in ("x", "y", "z")))
        bags = sorted(p.name for p in data.glob("*.db3"))
        import sqlite3
        with sqlite3.connect((data / parameters["bag_name"]).resolve().as_uri()+"?mode=ro", uri=True) as db:
            imu = db.execute("SELECT type,serialization_format FROM topics WHERE name='/imu/data'").fetchall()
        result.update(map_format=map_format, map_fields=fields, map_reader=parameters["map_reader"],
                      map_format_supported=format_ok, map_time_fields_used=False, bag_files=bags,
                      bag_unique=bags == [parameters["bag_name"]],
                      imu_topic_valid=imu == [("sensor_msgs/msg/Imu", "cdr")])
        result["ready"] = bool(result["frame_count_valid"] and format_ok and result["bag_unique"] and result["imu_topic_valid"])
    else:
        result["ready"] = False
    return result


def stage_commands(source, data, work, parameters=None):
    p = get_parameters(3) if parameters is None else parameters
    py = sys.executable
    scripts = source / "scripts"
    def script(name, *args):
        return [py, str(scripts / name), *map(str, args)]
    commands = [
        ("01_convert_map", (script("convert_map.py", data / "map.pcd", work / "map_xyzi.npy")
         if p["map_reader"] == "author_ascii" else [py, str(Path(__file__).with_name("convert_external_map.py")), str(data / "map.pcd"), str(work / "map_xyzi.npy")])),
        ("02_prepare_map", script("prepare_map.py", work / "map_xyzi.npy", work / "map_v020.npz", "--voxel", "0.2")),
        ("03_cache_plys", script("cache_plys.py", data / "raw_ply", work / "plys.npz")),
        ("04_scale", script("scale_from_height.py", "--plys", work / "plys", "--H", p["height_m"], "--nominal", p["nominal_scale"], "--causal", "--out", work / "scale_hc.npy")),
        ("05_sequential", script("track_sequential.py", "--map", work / "map_v020.npz", "--plys", work / "plys", "--init-guess", data / "init_guess.txt", "--scale-file", work / "scale_hc.npy", *(["--init-search"] if p["init_search"] else []), "--backfill", p["backfill"], "--out", work / "track_causal.npz")),
    ]
    for radius in p["ranges_m"]:
        commands.append((f"06_register_r{radius}", script("register_all.py", "--map", work / "map_v020.npz", "--plys", work / "plys", "--prior", work / "track_causal.cam.csv", "--scale-file", work / "scale_hc.npy", "--init", work / "track_causal.npz", "--class-match", "--max-range", radius, "--frames", "all", "--procs", p["processes"], "--out", work / f"reg_causal_r{radius}.npz")))
    commands.extend([
        ("07_merge", script("merge_regs.py", "--out", work / "reg_causal_hybrid.npz", "--min-xy-info", p["min_xy_info"], *[work / f"reg_causal_r{r}.npz" for r in p["ranges_m"]])),
        ("08_ekf", script("build_causal.py", data, "--reg", work / "reg_causal_hybrid.npz", "--gyro-dt", p["gyro_dt_s"], "--out", work / "ekf_causal.csv")),
        ("09_yaw", script("yaw_from_motion.py", data, "--csv", work / "ekf_causal.csv", "--causal", "--bias-file", work / "ekf_causal.state.npz", "--out", work / p["author_output_name"])),
    ])
    return commands


def run_stage(run, name, command, cwd):
    print(f"\n[{name}]", flush=True)
    write_json(run / "active_stage.json", dict(stage=name, command=command))
    start = time.perf_counter()
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, start_new_session=True)
    try:
        with (run / f"{name}.log").open("x") as log:
            for line in process.stdout:
                log.write(line)
                print(line, end="", flush=True)
        code = process.wait()
    except BaseException:
        # 只清理由本輪啟動的 child group，不碰其他使用者程序。
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise
    elapsed = time.perf_counter() - start
    with (run / "stage_timings.csv").open("a", newline="") as stream:
        csv.writer(stream).writerow([name, elapsed, code])
    with (run / "observations.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(stage=name, elapsed_seconds=elapsed, returncode=code)) + "\n")
    if code:
        raise subprocess.CalledProcessError(code, command)


def record_inputs(run, data, parameters):
    from sdcloc.frames import parse_frame_names
    files = list((data / "raw_ply").glob("*.ply"))
    stamps, indices = parse_frame_names(p.name for p in files)
    ordered = sorted(files, key=lambda p: int(p.stem.split("_ts_")[1]))
    with (run / "frame_manifest.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["idx", "original_idx", "timestamp_ns", "path", "sha256"])
        for i, (stamp, original, path) in enumerate(zip(stamps, indices, ordered)):
            writer.writerow([i, int(original), int(stamp), str(path), sha256(path)])
    write_json(run / "input_sources.json", {str(data / name): sha256(data / name)
               for name in ("map.pcd", "init_guess.txt", parameters["bag_name"])})


def execute(run, mode):
    def stop(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, stop)
    start = time.perf_counter()
    check_snapshot(run)
    source = run / "external_source"
    sys.path.insert(0, str(source / "src"))
    info_path = run / "test_info.json"
    info = json.loads(info_path.read_text())
    if info["status"] != "prepared":
        raise ValueError("不能重用已執行的 run")
    info["status"] = "running"
    write_json(info_path, info)
    try:
        versions = {name: importlib.metadata.version(name) for name in
                    ("numpy", "scipy", "polars", "matplotlib", "pytest")}
        write_json(run / "environment.json", dict(python=sys.version, executable=sys.executable, packages=versions))
        with (run / "environment_freeze.txt").open("w") as stream:
            subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=stream, check=True)
        parameters = json.loads((run / "parameters.json").read_text())
        data = Path("/workspace") / parameters["data_relative"]
        data_check = inspect_data(data, parameters)
        write_json(run / "data_check.json", data_check)
        with (run / "stage_timings.csv").open("x", newline="") as stream:
            csv.writer(stream).writerow(["stage", "elapsed_seconds", "returncode"])
        if mode == "--check":
            run_stage(run, "unit_tests", [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                      str(source / "tests"), "--basetemp", str(run / "pytest"),
                      "--junitxml", str(run / "pytest.xml")], source)
            suites = ET.parse(run / "pytest.xml").getroot()
            info["unit_tests"] = {key: sum(int(s.get(key, 0)) for s in suites.iter("testsuite"))
                                  for key in ("tests", "failures", "errors", "skipped")}
            wrapper_test = run / "source/tests/test_external_track3.py"
            run_stage(run, "wrapper_contract_tests", [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                      str(wrapper_test), "--basetemp", str(run / "wrapper_pytest"),
                      "--junitxml", str(run / "wrapper_pytest.xml")], run / "source")
            suites = ET.parse(run / "wrapper_pytest.xml").getroot()
            info["wrapper_tests"] = {key: sum(int(s.get(key, 0)) for s in suites.iter("testsuite"))
                                     for key in ("tests", "failures", "errors", "skipped")}
            info.update(status="checks_completed", data_ready=data_check["ready"], frame_count=0,
                        localization_executed=False)
        else:
            if not data_check["ready"]:
                raise ValueError(f"Track{parameters['track']} 資料未就緒：{data_check}")
            prep = time.perf_counter()
            record_inputs(run, data, parameters)
            info["input_recording_seconds"] = time.perf_counter() - prep
            work = run / "work"
            work.mkdir()
            commands = stage_commands(source, data, work, parameters)
            write_json(run / "commands.json", [dict(stage=n, argv=c) for n, c in commands])
            localization = time.perf_counter()
            for name, command in commands:
                run_stage(run, name, command, source)
            info["pipeline_seconds"] = time.perf_counter() - localization
            shutil.copyfile(work / parameters["author_output_name"], run / parameters["result_name"])
            from analyze_external_track3 import validate_result
            info["contract_audit"] = validate_result(run)
            info.update(status="completed", frame_count=parameters["expected_frames"], localization_executed=True)
        check_snapshot(run)
        info["total_seconds_before_analysis"] = time.perf_counter() - start
        info["output_sha256"] = {str(p.relative_to(run)): sha256(p) for p in run.rglob("*")
            if p.is_file() and (p.parent == run or p.parent == run / "work")
            and p.suffix in (".csv", ".npz", ".npy")}
        write_json(info_path, info)
        write_json(run / "analysis_input_info.json", info)
        analysis_start = time.perf_counter()
        try:
            from analyze_external_track3 import analyze
            analyze(run)
            write_json(run / "analysis_status.json", dict(status="completed", elapsed_seconds=time.perf_counter()-analysis_start))
        except Exception:
            write_json(run / "analysis_status.json", dict(status="failed", error=traceback.format_exc()))
            print("圖表失敗，定位／檢查狀態與原始輸出保留。", flush=True)
        print(json.dumps(dict(run=str(run), status=info["status"], data_ready=data_check["ready"]), ensure_ascii=False), flush=True)
    except BaseException as error:
        info.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    failure=traceback.format_exc(), elapsed_seconds=time.perf_counter()-start)
        write_json(info_path, info)
        raise


def finalize(run, code):
    path = run / "test_info.json"
    info = json.loads(path.read_text())
    if code and info["status"] in ("prepared", "running"):
        info.update(status="failed", failure="入口／相依套件／容器執行失敗，請見 dependencies.log 或 console.log")
    info["launcher_exit_code"] = code
    write_json(path, info)


if __name__ == "__main__":
    action = sys.argv[1]
    if action == "prepare":
        if sys.argv[2] not in ("--check", "--full"):
            raise SystemExit("未知模式")
        prepare(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3)
    elif action == "execute":
        execute(Path(sys.argv[2]).resolve(), sys.argv[3])
    elif action == "finalize":
        finalize(Path(sys.argv[2]).resolve(), int(sys.argv[3]))
    else:
        raise SystemExit("用法：由 scripts/run_external_track3.sh 啟動")

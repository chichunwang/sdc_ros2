#!/usr/bin/env python3
"""在獨立容器量測固定 Track 前 N 幀，保存參數、逐幀時間與輸出檢查。"""

import argparse
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import time


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", type=int, choices=(1, 2), default=1)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--backend", choices=("cuda_ndt", "pcl_ndt", "cuda_robust_icp"), help="default legacy=cuda_ndt; sensor_guided=cuda_robust_icp")
    parser.add_argument("--cuda-mode", choices=("P2D", "D2D"), default="D2D")
    parser.add_argument("--profile", choices=("legacy", "sensor_guided"), default="legacy", help="legacy: previous-pose NDT; sensor_guided: GNSS/IMU plus bounded robust map refinement")
    parser.add_argument("--resolution", type=float, help="NDT voxel resolution; default legacy=8 m, sensor_guided=2 m")
    parser.add_argument("--compare", type=Path, help="Optional reference result.txt; differences are not ground-truth errors.")
    parser.add_argument("--map-cache", type=Path, help="Explicit derived map cache under GPU outputs; only reuse for the same Track/map/voxel. Disabled by default.")
    args = parser.parse_args()
    if args.backend is None:
        args.backend = "cuda_robust_icp" if args.profile == "sensor_guided" else "cuda_ndt"
    if args.backend == "cuda_robust_icp" and args.profile != "sensor_guided":
        parser.error("--backend cuda_robust_icp requires --profile sensor_guided")
    if args.resolution is None:
        args.resolution = 2.0 if args.profile == "sensor_guided" else 8.0
    if args.frames < 1 or not math.isfinite(args.resolution) or args.resolution <= 0:
        parser.error("--frames and --resolution must be positive")
    if args.compare and not args.compare.is_file():
        parser.error("--compare must name an existing result.txt")
    root = Path(__file__).resolve().parents[2]
    cache_file = ""
    if args.map_cache:
        try:
            relative_cache = args.map_cache.resolve().relative_to((root / "outputs").resolve())
        except ValueError:
            parser.error("--map-cache must be inside this GPU environment's outputs directory")
        args.map_cache.resolve().parent.mkdir(parents=True, exist_ok=True)
        cache_file = "/root/outputs/" + relative_cache.as_posix()
    compose = json.loads(subprocess.check_output(["bash", str(root / "scripts/dev.sh"), "config", "--format", "json"], text=True))
    service = compose["services"]["midterm01_gpu"]
    container = service["container_name"]
    if subprocess.run(["docker", "exec", container, "pgrep", "-x", "localization"], capture_output=True).returncode == 0:
        parser.error("This GPU environment already has a localization process; wait for it to finish.")
    data_mount = next(m for m in service["volumes"] if m["target"] == "/root/data")
    data_root = Path(data_mount["source"])
    cache_hit = bool(args.map_cache and args.map_cache.exists())
    if args.map_cache:
        source_map = data_root / f"Track{args.track}/map.pcd"
        fingerprint = hashlib.sha256()
        with source_map.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                fingerprint.update(chunk)
        identity = {"track": args.track, "source_map": str(source_map.resolve()),
                    "map_sha256": fingerprint.hexdigest(), "map_voxel_leaf_size": 1.0}
        sidecar = args.map_cache.with_suffix(args.map_cache.suffix + ".json")
        if cache_hit and (not sidecar.exists() or json.loads(sidecar.read_text()) != identity):
            parser.error("Map cache identity is missing or mismatches the Track/map/leaf; choose a new cache path.")
        if not cache_hit:
            write_json(sidecar, identity)
    files = []
    for path in (data_root / f"Track{args.track}/raw_ply").iterdir():
        match = re.search(r"_ts_(\d+)", path.stem)
        if path.is_file() and path.suffix.lower() == ".ply" and match and int(match[1]) > 0:
            files.append((int(match[1]), path.name))
    files.sort()
    if len(files) < args.frames:
        parser.error(f"Only {len(files)} timestamped PLY files available")

    profile_prefix = "sensor_guided_" if args.profile == "sensor_guided" else ""
    run = Path(tempfile.mkdtemp(prefix=f"track{args.track}_{profile_prefix}{args.backend}_{args.cuda_mode.lower()}_{args.frames}f_", dir=root / "outputs/runs"))
    container_run = "/root/outputs/runs/" + run.name
    params = {
        "map_path": f"/root/data/Track{args.track}", "map_file": "map.pcd",
        "ply_path": f"/root/data/Track{args.track}/raw_ply",
        "init_guess_file": f"/root/data/Track{args.track}/init_guess.txt",
        "save_path": container_run, "save_file": "result.txt", "map_cache_file": cache_file, "bag_path": "",
        "map_voxel_leaf_size": 1.0, "input_voxel_leaf_size": 0.3, "min_input_points": 10,
        "submap_half_size": 100.0, "submap_half_height": 30.0,
        "ndt_resolution": args.resolution, "ndt_step_size": 0.8, "ndt_transformation_epsilon": 0.05,
        "ndt_max_iterations": 30, "ply_scale": 3.0,
        "publish_global_map": False, "max_frames": args.frames, "keep_alive": False,
        "registration_backend": args.backend, "cuda_ndt_mode": args.cuda_mode,
        "cuda_translation_epsilon": 0.01, "cuda_rotation_epsilon": 0.002,
        "localization_profile": args.profile,
        "sensor_prior_file": container_run + "/sensor_prior.csv" if args.profile == "sensor_guided" else "",
    }
    write_json(run / "params.json", params)
    # JSON is valid YAML and preserves the ROS parameter scalar types.
    write_json(run / "params.yaml", {"image_point_cloud_localizer": {"ros__parameters": params}})
    with (run / "frames.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["idx", "timestamp_ns", "filename"])
        writer.writerows((i, stamp, name) for i, (stamp, name) in enumerate(files[:args.frames]))
    sensor_preparation_seconds = 0.0
    if args.profile == "sensor_guided":
        prep_start = time.perf_counter()
        prep = ["docker", "exec", container, "bash", "-c",
                "source /opt/ros/humble/setup.bash && exec python3 /root/scripts/prepare_sensor_prior.py "
                f"--bag /root/data/Track{args.track}/Track{args.track}.db3 --frames "
                + shlex.quote(container_run + "/frames.csv") + " --output " + shlex.quote(params["sensor_prior_file"])]
        write_json(run / "sensor_command.json", prep)
        with (run / "sensor_preparation.log").open("w") as log:
            subprocess.run(prep, stdout=log, stderr=subprocess.STDOUT, check=True)
        sensor_preparation_seconds = time.perf_counter() - prep_start
    write_json(run / "pipeline_source_sha256.json", {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__), root / "scripts/prepare_sensor_prior.py")
    })
    command = ["docker", "exec", container, "bash", "-c",
               "source /opt/ros/humble/setup.bash && source /root/midterm/install/setup.bash && "
               "exec ros2 run localization localization --ros-args --params-file " + shlex.quote(container_run + "/params.yaml")]
    (run / "command.sh").write_text("#!/usr/bin/env bash\n" + shlex.join(command) + "\n")
    write_json(run / "source_sha256.json", {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (root / "workspace/src").rglob("*") if path.is_file()
    })
    (run / "worktree.patch").write_bytes(subprocess.check_output(["git", "diff"], cwd=root))
    write_json(run / "container.json", json.loads(subprocess.check_output(["docker", "inspect", container], text=True)))
    (run / "git_commit.txt").write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True))
    (run / "binary_sha256.txt").write_text(subprocess.check_output([
        "docker", "exec", container, "sha256sum",
        "/root/midterm/install/localization/lib/localization/localization",
        "/root/midterm/install/cuda_ndt_backend/lib/libcuda_ndt_backend.so"], text=True))
    (run / "cpu_processes_before.txt").write_text(subprocess.run(["ps", "-C", "localization", "-o", "pid,pcpu,etime,args"], capture_output=True, text=True).stdout)
    print(f"Run: {run}", flush=True)
    monitor_log = (run / "gpu_samples.csv").open("w")
    monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw", "--format=csv,nounits", "-lms", "200"], stdout=monitor_log, stderr=subprocess.STDOUT)
    start = time.perf_counter()
    try:
        with (run / "run.log").open("w") as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            returncode = process.wait()
    finally:
        elapsed = time.perf_counter() - start
        monitor.terminate()
        monitor.wait()
        monitor_log.close()

    errors = []
    poses = []
    try:
        for line in (run / "result.txt").read_text().splitlines():
            values = [float(value) for value in line.split()]
            if len(values) != 12 or not all(math.isfinite(value) for value in values):
                raise ValueError("Each pose must contain 12 finite values")
            poses.append(values)
        if len(poses) != args.frames:
            errors.append(f"Expected {args.frames} poses, got {len(poses)}")
    except (OSError, ValueError) as error:
        errors.append(str(error))
    with (run / "diagnostic.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["idx", "x", "y", "yaw"])
        writer.writerows((i, p[3], p[7], math.atan2(p[4], p[0])) for i, p in enumerate(poses))
    timings = []
    timing_sums = {}
    try:
        with (run / "timings.csv").open() as stream:
            timings = list(csv.DictReader(stream))
        if len(timings) != args.frames:
            errors.append(f"Expected {args.frames} timing rows, got {len(timings)}")
        for name in ("load_ms", "filter_ms", "crop_ms", "ndt_ms", "fitness_ms", "publish_ms", "total_ms"):
            values = [float(row[name]) for row in timings]
            if not all(math.isfinite(value) and value >= 0 for value in values):
                raise ValueError(f"Invalid timings: {name}")
            timing_sums[name.replace("_ms", "_seconds")] = sum(values) / 1000
        if any(row["backend"] != args.backend for row in timings):
            errors.append("Unexpected registration backend")
    except (OSError, ValueError, KeyError) as error:
        errors.append(str(error))
    if returncode != 0:
        errors.append(f"Process exited with code {returncode}")
    summary = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "track": args.track, "backend": args.backend, "cuda_mode": args.cuda_mode,
        "profile": args.profile, "sensor_preparation_seconds": sensor_preparation_seconds,
        "pipeline_elapsed_seconds": elapsed + sensor_preparation_seconds,
        "frames": len(poses), "expected_frames": args.frames, "elapsed_seconds": elapsed,
        "time_scope": "Docker exec + ROS startup + map initialization + all frames; no build. See map_cache_file for explicit cache use.",
        "map_cache_file": cache_file,
        "map_cache_hit_at_start": cache_hit,
        "returncode": returncode, "failed_alignments": sum(row["converged"] != "1" for row in timings),
        "timing_sums": timing_sums, "validation_errors": errors, "output_complete_and_finite": not errors,
        "accuracy": "Not evaluated against ground truth or Kaggle; convergence is not an accuracy guarantee.",
    }
    if poses:
        rotations = [[[pose[i * 4 + j] for j in range(3)] for i in range(3)] for pose in poses]
        orth_error = max(abs(sum(r[k][i] * r[k][j] for k in range(3)) - int(i == j))
                         for r in rotations for i in range(3) for j in range(3))
        determinants = [r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
                        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
                        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]) for r in rotations]
        summary["pose_diagnostics_not_accuracy"] = {
            "rotation_orthogonality_max_error": orth_error,
            "rotation_det_min": min(determinants), "rotation_det_max": max(determinants),
            "max_adjacent_translation_m": max((math.sqrt(sum((a[j] - b[j]) ** 2 for j in (3, 7, 11)))
                                               for a, b in zip(poses[1:], poses[:-1])), default=0),
        }
    if args.profile == "sensor_guided":
        with (run / "guidance.csv").open() as stream:
            guidance = list(csv.DictReader(stream))
        if len(guidance) != args.frames:
            errors.append(f"Expected {args.frames} guidance rows, got {len(guidance)}")
        summary["guidance_not_accuracy"] = {
            "cuda_nearest_neighbor": args.backend == "cuda_robust_icp",
            "cuda_ndt_proposals_enabled": args.backend == "cuda_ndt",
            "cuda_converged_frames": sum(row["cuda_converged"] == "1" for row in guidance),
            "cuda_seed_used_frames": sum(row["cuda_seed_used"] == "1" for row in guidance),
            "map_accepted_frames": sum(row["map_accepted"] == "1" for row in guidance),
            "sensor_only_frames": sum(row["map_accepted"] != "1" for row in guidance),
            "max_xy_correction_m": max((math.hypot(float(row["dx"]), float(row["dy"])) for row in guidance), default=0),
            "max_yaw_correction_deg": max((abs(math.degrees(float(row["dyaw"]))) for row in guidance), default=0),
        }
        if summary["guidance_not_accuracy"]["max_xy_correction_m"] > 1.500001 or summary["guidance_not_accuracy"]["max_yaw_correction_deg"] > 8.000001:
            errors.append("Sensor correction exceeded configured default bounds")
        if any(not math.isfinite(float(row[key])) for row in guidance for key in ("prior_cost", "final_cost", "dx", "dy", "dz", "dyaw")):
            errors.append("Nonfinite guidance metrics")
        if any(float(row["final_cost"]) > float(row["prior_cost"]) + 1e-8 for row in guidance):
            errors.append("Map update worsened regularized objective")
        summary["output_complete_and_finite"] = not errors
    if args.compare:
        reference = [[float(value) for value in line.split()] for line in args.compare.read_text().splitlines()]
        if len(reference) == len(poses) and poses:
            summary["difference_to_reference_not_ATE"] = {
                "path": str(args.compare.resolve()),
                "translation_rms_m": math.sqrt(sum(sum((p[j] - r[j]) ** 2 for j in (3, 7, 11)) for p, r in zip(poses, reference)) / len(poses)),
                "translation_max_m": max(math.sqrt(sum((p[j] - r[j]) ** 2 for j in (3, 7, 11))) for p, r in zip(poses, reference)),
            }
    write_json(run / "validation.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

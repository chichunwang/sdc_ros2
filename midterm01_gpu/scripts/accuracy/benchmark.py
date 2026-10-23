#!/usr/bin/env python3
"""在 GPU 容器量測 Track 的連續影格（預設前 N 幀），保存來源、時間與輸出檢查。"""

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
import time
from zoneinfo import ZoneInfo
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from output_layout import loads_saved_json, saved_path
from three_camera_source import load_index, load_reconstruction, reuse_priors, sha256

if __package__:
    from .geometry_report import summarize_guided_updates, validate_geometry
    from .surface_cache import prepare_height_map
else:
    from geometry_report import summarize_guided_updates, validate_geometry
    from surface_cache import prepare_height_map


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def create_run_directory(directory, prefix, *, track=None, full=False):
    """單一run入口；名稱只含Track、full/test與日期時間，算法版本放目錄內。"""
    created = datetime.datetime.now(ZoneInfo("Asia/Taipei"))
    stamp = created.strftime("%y%m%d%H%M")
    inferred = re.match(r"track0?([123])(?:_|$)", prefix)
    track = int(inferred[1]) if track is None and inferred else (0 if track is None else track)
    if track not in (0,1,2,3): raise ValueError('Track must be 0 (shared), 1, 2 or 3')
    name = f"track{track:02d}_{'full' if full else 'test'}_{stamp}"
    directory.mkdir(parents=True, exist_ok=True)
    sequence = 1
    while True:
        run = directory / (name if sequence == 1 else f"{name}_{sequence:04d}")
        try:
            run.mkdir()
        except FileExistsError:
            sequence += 1
            continue
        write_json(run/'test_info.json', {
            'created_at': created.isoformat(), 'timezone': 'Asia/Taipei',
            'track': track, 'scope': 'full' if full else 'test', 'test_kind': prefix,
            'same_minute_sequence': sequence,
            'version_details': 'params.json, pipeline_source_sha256.json, git_commit.txt and worktree.patch when present',
        })
        return run


def print_run_outcome(run, summary, *, full_attempt):
    """Keep integrity success separate from whether refinement changed the CSV."""
    complete = summary['output_complete_and_finite']
    if full_attempt:
        print(f"{'定位完成' if complete else '檢查未通過'}: {run}", flush=True)
    if not complete:
        return
    report = summary.get('horizontal_diagnostics_not_accuracy')
    csv_changed = True
    if report:
        # Compare the serialized submission, rather than treating 'applied' as
        # proof of changed submission bytes (e.g. a pose-only change).
        csv_changed = report['csv_sha256'] != report['base_csv_sha256']
        executed = report.get('joint_estimation_executed', 'solver' in report)
        print(f"水平 map factors: {report['accepted_map_factors']}；聯合求解: {executed}；實際套用: {report['applied']}", flush=True)
        if csv_changed:
            print('本輪 CSV 已改變；精度是否提升仍需另行驗證。', flush=True)
        else:
            print('本輪未產生新的提交內容：CSV 與 base/diagnostic.csv 的 SHA256 相同，無須重複提交。', flush=True)
        print(f"原因: {report['reason']}；詳見 horizontal/report.json", flush=True)
    if full_attempt:
        label = '提交檔' if csv_changed else '相同內容的 CSV'
        print(f'{label}: diagnostic.csv；回放矩陣: result.txt；檢查報告: validation.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", type=int, choices=(1, 2, 3), default=1)
    frame_selection = parser.add_mutually_exclusive_group()
    frame_selection.add_argument("--frames", type=int, default=50)
    frame_selection.add_argument("--full", action="store_true", help="Process all timestamped frames in the Track; mutually exclusive with --frames, requires --start-frame 0")
    parser.add_argument("--start-frame", type=int, default=0, help="Original 0-based frame index; short-window diagnosis, not a complete submission")
    parser.add_argument("--cloud-source", choices=("front", "mapanything", "reconstruction"), default="front")
    parser.add_argument("--ply-scale", type=float, help="Track1/2 front default=3; Track3 and experimental sources require explicit scale")
    parser.add_argument("--source-report", type=saved_path, help="Required experimental source report with matching Track/frame/hash contract")
    parser.add_argument("--prior-run", type=saved_path, help="Reuse exact sensor priors from an existing same-Track/fusion GPU run")
    parser.add_argument("--backend", choices=("cuda_ndt", "pcl_ndt", "cuda_robust_icp"), help="default legacy=cuda_ndt; sensor_guided=cuda_robust_icp")
    parser.add_argument("--cuda-mode", choices=("P2D", "D2D"), default="D2D")
    parser.add_argument("--profile", choices=("legacy", "sensor_guided"), default="legacy", help="legacy: previous-pose NDT; sensor_guided: GNSS/IMU plus bounded robust map refinement")
    parser.add_argument("--sensor-fusion", choices=("fixed", "heading_rts", "motion_rts", "pose_heading_rts", "independent_heading_rts", "pose_yaw", "turn_heading_rts"), default="fixed",
                        help="motion_rts: scored prior; turn_heading_rts: integrated IMU direction / GNSS path heading fit; pose_yaw: directly replace yaw by supplied GNSS Pose yaw (unverified source); previous experimental heading modes retained")
    parser.add_argument("--height-mode", choices=("carry", "anchored", "surface", "surface_tilt"), default="carry",
                        help="carry: scored incremental Z; anchored: nominal sensor altitude; surface/surface_tilt: paired lower-surface reference (front/ML venv required; height map built in new run). Alternatives are experimental")
    parser.add_argument("--horizontal-refinement", choices=("none", "surfel_joint"), default="none",
                        help="Experimental fine-map XY/yaw factors + GNSS/gyro batch graph; requires front/motion_rts/directional/carry; insufficient evidence preserves base bytes")
    parser.add_argument("--horizontal-map-cache", type=saved_path, help="Optional verified fine horizontal map under outputs for paired short tests; otherwise built in the new run")
    parser.add_argument("--fusion-gnss-std", type=float, default=.3, help="fusion model GNSS std, m; not calibrated accuracy")
    parser.add_argument("--fusion-xy-accel-psd", type=float, default=4., help="motion_rts acceleration PSD, m^2/s^3")
    parser.add_argument("--fusion-bias-rw-deg", type=float, default=.12, help="bias random walk std, deg/sqrt(s)")
    parser.add_argument("--fusion-course-window", type=float, default=2., help="GNSS course block length, s (>=1)")
    parser.add_argument("--fusion-pose-heading-std-deg", type=float, default=3., help="experimental GNSS orientation block std floor, degrees; uncalibrated")
    parser.add_argument("--geometry-mode", choices=("off", "diagnostic", "directional", "gicp"), default="off",
                        help="off: point-to-point; diagnostic: unchanged objective; directional: scored map-normal metric; gicp: experimental symmetric source/map surface covariance")
    parser.add_argument("--geometry-normal-gain", type=float, default=2.)
    parser.add_argument("--geometry-tangent-weight", type=float, default=.05)
    parser.add_argument("--resolution", type=float, help="NDT voxel resolution; default legacy=8 m, sensor_guided=2 m")
    parser.add_argument("--compare", type=saved_path, help="Optional reference result.txt; differences are not ground-truth errors.")
    parser.add_argument("--map-cache", type=saved_path, help="Explicit derived map cache under GPU outputs; only reuse for the same Track/map/voxel. Disabled by default.")
    parser.add_argument("--height-map-cache", type=saved_path, help="Surface model only: optionally reuse a verified 0.3 m binary XYZ cache under GPU outputs; otherwise build one in the new run")
    parser.add_argument("--dry-run", action="store_true", help="Track3 only: validate inputs and print a JSON plan without Docker exec, localization or a new run")
    args = parser.parse_args()
    if args.dry_run and args.track != 3:
        parser.error("--dry-run is currently limited to Track3")
    if args.dry_run and (args.map_cache or args.height_map_cache or args.horizontal_map_cache):
        parser.error("--dry-run cannot combine with map/height/horizontal cache options")
    if args.backend is None:
        args.backend = "cuda_robust_icp" if args.profile == "sensor_guided" else "cuda_ndt"
    expected_full_frames = {1: 3011, 2: 3299, 3: 3142}[args.track]
    if args.track == 3:
        if (args.profile != "legacy" or args.cloud_source != "front"
                or args.height_mode != "carry" or args.geometry_mode != "off"
                or args.horizontal_refinement != "none" or args.prior_run
                or args.sensor_fusion != "fixed" or args.backend not in ("cuda_ndt", "pcl_ndt")):
            parser.error("Track3 only supports legacy/front/carry/geometry off/horizontal none/fixed with no prior-run; GNSS/IMU preprocessing and RTS/graph paths are unavailable")
        if args.ply_scale is None:
            parser.error("Track3 requires explicit --ply-scale; its raw PLY scale is uncalibrated")
    if args.start_frame < 0:
        parser.error("--start-frame must be nonnegative")
    if args.full and args.start_frame:
        parser.error("--full cannot combine with --start-frame")
    if args.ply_scale is None:
        if args.cloud_source != "front":
            parser.error("Experimental clouds require explicit --ply-scale; do not inherit front-camera scale 3")
        args.ply_scale = 3.0
    if not math.isfinite(args.ply_scale) or args.ply_scale <= 0:
        parser.error("--ply-scale must be finite and positive")
    if args.cloud_source != "front" and (args.profile != "sensor_guided" or args.backend != "cuda_robust_icp" or not args.source_report):
        parser.error("Experimental clouds require sensor_guided/cuda_robust_icp and --source-report")
    if args.cloud_source == "reconstruction" and args.ply_scale != 1.:
        parser.error("Calibrated reconstruction uses metric export; --ply-scale must be 1")
    if args.prior_run and args.profile != "sensor_guided":
        parser.error("--prior-run requires sensor_guided")
    if args.start_frame and args.profile != "sensor_guided":
        parser.error("--start-frame requires sensor_guided; legacy first pose cannot initialize an arbitrary window")
    if args.source_report and args.cloud_source == "front":
        parser.error("--source-report requires an experimental cloud source")
    if args.backend == "cuda_robust_icp" and args.profile != "sensor_guided":
        parser.error("--backend cuda_robust_icp requires --profile sensor_guided")
    if args.sensor_fusion != "fixed" and args.profile != "sensor_guided":
        parser.error("--sensor-fusion requires --profile sensor_guided")
    if args.height_mode != "carry" and (args.profile != "sensor_guided" or args.backend != "cuda_robust_icp"):
        parser.error("--height-mode anchored/surface requires sensor_guided/cuda_robust_icp")
    if args.height_mode.startswith('surface') and args.cloud_source != "front":
        parser.error("--height-mode surface/surface_tilt requires --cloud-source front")
    if not args.height_mode.startswith('surface') and args.height_map_cache:
        parser.error('--height-map-cache requires --height-mode surface')
    if args.geometry_mode != "off" and (args.profile != "sensor_guided" or args.backend != "cuda_robust_icp"):
        parser.error("Geometry requires --profile sensor_guided --backend cuda_robust_icp")
    if (not math.isfinite(args.geometry_normal_gain) or not 0 < args.geometry_normal_gain <= 10
            or not math.isfinite(args.geometry_tangent_weight) or not 0 <= args.geometry_tangent_weight <= 1):
        parser.error("Geometry gain must be in (0,10]; tangent weight in [0,1]")
    if args.geometry_mode == "gicp" and args.geometry_tangent_weight <= 0:
        parser.error("GICP requires positive --geometry-tangent-weight for invertible surface covariance")
    if args.horizontal_refinement != "none" and (args.profile != "sensor_guided" or args.backend != "cuda_robust_icp"
            or args.sensor_fusion != "motion_rts" or args.geometry_mode != "directional"
            or args.height_mode != "carry" or args.cloud_source != "front" or args.ply_scale != 3.):
        parser.error("surfel_joint requires front/scale3/sensor_guided/cuda_robust_icp/motion_rts/directional/carry")
    if args.horizontal_map_cache and args.horizontal_refinement == "none":
        parser.error("--horizontal-map-cache requires --horizontal-refinement surfel_joint")
    fusion_settings = {"fusion": args.sensor_fusion, "gnss-std": args.fusion_gnss_std,
                       "xy-accel-psd": args.fusion_xy_accel_psd, "bias-rw-deg": args.fusion_bias_rw_deg,
                       "course-window": args.fusion_course_window,
                       "pose-heading-std-deg": args.fusion_pose_heading_std_deg}
    if (not all(math.isfinite(value) and value > 0 for key, value in fusion_settings.items() if key != "fusion")
            or args.fusion_course_window < 1):
        parser.error("Fusion parameters must be finite and positive; course window >= 1 second")
    if args.resolution is None:
        args.resolution = 2.0 if args.profile == "sensor_guided" else 8.0
    if args.frames < 1 or not math.isfinite(args.resolution) or args.resolution <= 0:
        parser.error("--frames and --resolution must be positive")
    if args.compare and not args.compare.is_file():
        parser.error("--compare must name an existing result.txt")
    root = Path(__file__).resolve().parents[2]
    surface_python = root / 'outputs/ml_runtime/venv/bin/python'
    if (args.height_mode.startswith('surface') or args.horizontal_refinement != 'none') and not surface_python.is_file():
        parser.error('Surface/horizontal preprocessing requires the existing isolated ML venv; ROS Python is not modified')
    if args.height_map_cache:
        try:
            args.height_map_cache.resolve().relative_to((root/'outputs').resolve())
        except ValueError:
            parser.error('--height-map-cache must be inside GPU outputs')
        if not args.height_map_cache.is_file():
            parser.error('--height-map-cache must name an existing verified binary XYZ cache')
    if args.horizontal_map_cache:
        try:
            args.horizontal_map_cache.resolve().relative_to((root/'outputs').resolve())
        except ValueError:
            parser.error('--horizontal-map-cache must be inside GPU outputs')
        if not args.horizontal_map_cache.is_file():
            parser.error('--horizontal-map-cache must name an existing verified fine map')
    cache_file = ""
    if args.map_cache:
        try:
            relative_cache = args.map_cache.resolve().relative_to((root / "outputs").resolve())
        except ValueError:
            parser.error("--map-cache must be inside this GPU environment's outputs directory")
        args.map_cache.resolve().parent.mkdir(parents=True, exist_ok=True)
        cache_file = "/root/outputs/" + relative_cache.as_posix()
    compose = loads_saved_json(subprocess.check_output(["bash", str(root / "scripts/dev.sh"), "config", "--format", "json"], text=True))
    service = compose["services"]["midterm01_gpu"]
    container = service["container_name"]
    if not args.dry_run and subprocess.run(["docker", "exec", container, "pgrep", "-x", "localization"], capture_output=True).returncode == 0:
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
        if cache_hit and (not sidecar.exists() or loads_saved_json(sidecar.read_text()) != identity):
            parser.error("Map cache identity is missing or mismatches the Track/map/leaf; choose a new cache path.")
        if not cache_hit:
            write_json(sidecar, identity)
    files = []
    track3_ply_count = 0
    for path in (data_root / f"Track{args.track}/raw_ply").iterdir():
        if args.track == 3 and path.is_file() and path.suffix.lower() == ".ply":
            track3_ply_count += 1
        match = re.search(r"_ts_(\d+)", path.stem)
        if path.is_file() and path.suffix.lower() == ".ply" and match and int(match[1]) > 0:
            files.append((int(match[1]), path.name))
    files.sort()
    if args.track == 3:
        original_indices = [re.match(r"^(\d+)_ts_\d+\.ply$", name) for _, name in files]
        if (track3_ply_count != expected_full_frames or len(files) != expected_full_frames
                or any(b[0] <= a[0] for a, b in zip(files, files[1:]))
                or any(match is None for match in original_indices)
                or [int(match[1]) for match in original_indices if match] != list(range(expected_full_frames))):
            parser.error("Track3 requires all 3142 PLY files with strictly increasing unique timestamps and original filename indices 0..3141 before selecting a prefix")
    if args.full:
        if not files:
            parser.error('No timestamped PLY frames available for --full')
        args.frames = len(files)
    source_relative = Path(f"Track{args.track}/raw_ply")
    source_host = data_root / source_relative
    source_container = "/root/data/" + source_relative.as_posix()
    source_metadata = {"cloud_source": args.cloud_source, "ply_scale": args.ply_scale,
                       "reference_time": "front_camera_bag_timestamp", "per_camera_deskew_applied": False}
    if args.cloud_source == "mapanything":
        source_root, index, contract = load_index(data_root, args.track, [s for s, _ in files])
        source_report = loads_saved_json(args.source_report.read_text())
        if source_report.get("track") != args.track or source_report.get("source_contract") != contract:
            parser.error("Source calibration report Track/index/frame identity mismatch")
        source_relative = source_root.relative_to(data_root)
        files = [(int(row["f_ts"]), row["ply"]) for row in index]
        source_metadata.update(contract, calibration_report=str(args.source_report.resolve()),
                               calibration_report_sha256=sha256(args.source_report),
                               scale_status=source_report.get("scale_status"),
                               experimental=True, merged_deskew_unavailable=True)
        source_host = source_root
        source_container = "/root/data/" + source_relative.as_posix()
    if args.cloud_source == "reconstruction":
        try:
            source_host, files, report = load_reconstruction(args.source_report, root / 'outputs', args.track,
                                                            [s for s, _ in files], args.start_frame, args.frames)
        except (ValueError, OSError, KeyError) as error:
            parser.error(str(error))
        source_container = '/root/outputs/' + source_host.relative_to((root / 'outputs').resolve()).as_posix()
        source_metadata.update(variant=report['variant'], per_camera_deskew_applied=report['per_camera_motion_compensation'],
                               calibration_report=str(args.source_report.resolve()), calibration_report_sha256=sha256(args.source_report),
                               source_contract=report, experimental=True)
    else:
        if len(files) < args.start_frame + args.frames:
            parser.error(f"Only {len(files)} timestamped PLY files available")
        files = files[args.start_frame:args.start_frame + args.frames]

    if args.dry_run:
        track_root = data_root / "Track3"
        for name in ("map.pcd", "init_guess.txt"):
            path = track_root / name
            if not path.is_file() or path.stat().st_size == 0:
                parser.error(f"Track3 dry-run requires an existing nonempty {path}")
        plan = {
            "track": args.track, "dry_run": True, "expected_full_frames": expected_full_frames,
            "requested_frames": args.frames, "selected_frames": len(files),
            "source_root": str(source_host.resolve()),
            "first_file": {"timestamp_ns": files[0][0], "filename": files[0][1]},
            "last_file": {"timestamp_ns": files[-1][0], "filename": files[-1][1]},
            "map_path": str((track_root / "map.pcd").resolve()),
            "init_guess_path": str((track_root / "init_guess.txt").resolve()),
            "profile": args.profile, "backend": args.backend, "cuda_mode": args.cuda_mode,
            "ply_scale": args.ply_scale, "scale_status": "uncalibrated",
            "bag_path": "", "sensor_preprocessing_called": False,
            "localization_executed": False, "docker_exec_called": False, "run_created": False,
        }
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    profile_prefix = "sensor_guided_" if args.profile == "sensor_guided" else ""
    if args.sensor_fusion != "fixed":
        profile_prefix += args.sensor_fusion + "_"
    if args.geometry_mode != "off":
        profile_prefix += "geometry_" + args.geometry_mode + "_"
    if args.height_mode != "carry":
        profile_prefix += "height_" + args.height_mode + "_"
    if args.horizontal_refinement != "none":
        profile_prefix += args.horizontal_refinement + "_"
    if args.cloud_source != "front" or args.start_frame or args.ply_scale != 3.0:
        profile_prefix += f"{args.cloud_source}_s{args.ply_scale:g}_from{args.start_frame}_"
    full_attempt = args.start_frame == 0 and args.frames == expected_full_frames
    run = create_run_directory(root / "outputs/run", f"track{args.track}_{profile_prefix}{args.backend}_{args.cuda_mode.lower()}",
                               track=args.track, full=full_attempt)
    print(f"Run: {run}", flush=True)
    if args.track == 3:
        # Source provenance preparation only; outside localization/frame timing.
        static_inputs = {}
        for name in ("map.pcd", "init_guess.txt"):
            path = data_root / "Track3" / name
            static_inputs[name] = {"path": str(path.resolve()), "size_bytes": path.stat().st_size,
                                   "sha256": sha256(path)}
        write_json(run / "static_input_sha256.json", {
            "scope": "map_only_legacy", "bag_used": False, "bag_path": "",
            "timing_scope": "source provenance preparation; excluded from per-frame localization timing",
            "inputs": static_inputs,
        })
    container_run = "/root/outputs/run/" + run.name
    write_json(run/'benchmark_command.json', [sys.executable, str(root/'scripts/benchmark.py'), *sys.argv[1:]])
    source_metadata["selected_frames"] = [
        {"original_idx": args.start_frame + i, "timestamp_ns": stamp, "filename": name,
         "sha256": sha256(source_host / name)} for i, (stamp, name) in enumerate(files)]
    write_json(run / "source_input.json", source_metadata)
    if args.source_report:
        (run / "source_calibration.json").write_bytes(args.source_report.read_bytes())
    # Symlinks resolve inside the GPU container; original data remains read-only.
    ply_path = source_container
    if args.start_frame or args.cloud_source != "front":
        staged = run / "source_ply"
        staged.mkdir()
        for _, name in files:
            (staged / Path(name).name).symlink_to(source_container + '/' + name)
        ply_path = container_run + "/source_ply"
    params = {
        "map_path": f"/root/data/Track{args.track}", "map_file": "map.pcd",
        "ply_path": ply_path,
        "init_guess_file": f"/root/data/Track{args.track}/init_guess.txt",
        "save_path": container_run, "save_file": "result.txt", "map_cache_file": cache_file, "bag_path": "",
        "map_voxel_leaf_size": 1.0, "input_voxel_leaf_size": 0.3, "min_input_points": 10,
        "submap_half_size": 100.0, "submap_half_height": 30.0,
        "ndt_resolution": args.resolution, "ndt_step_size": 0.8, "ndt_transformation_epsilon": 0.05,
        "ndt_max_iterations": 30, "ply_scale": args.ply_scale,
        "publish_global_map": False, "max_frames": args.frames, "keep_alive": False,
        "registration_backend": args.backend, "cuda_ndt_mode": args.cuda_mode,
        "cuda_translation_epsilon": 0.01, "cuda_rotation_epsilon": 0.002,
        "localization_profile": args.profile,
        "sensor_prior_file": container_run + "/sensor_prior.csv" if args.profile == "sensor_guided" else "",
        "sensor_height_mode": args.height_mode,
        "sensor_height_reference_file": container_run + '/height_reference.csv' if args.height_mode.startswith('surface') else '',
        "geometry_mode": args.geometry_mode, "geometry_normal_gain": args.geometry_normal_gain,
        "geometry_tangent_weight": args.geometry_tangent_weight,
    }
    write_json(run / "params.json", params)
    # JSON is valid YAML and preserves the ROS parameter scalar types.
    write_json(run / "params.yaml", {"image_point_cloud_localizer": {"ros__parameters": params}})
    with (run / "frames.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["idx", "timestamp_ns", "filename"])
        writer.writerows((i, stamp, Path(name).name) for i, (stamp, name) in enumerate(files))
    sensor_preparation_seconds = 0.0
    pipeline_sources = (Path(__file__), root / "scripts/prepare_sensor_prior.py", root / "scripts/sensor_fusion.py", Path(__file__).with_name("geometry_report.py"), root / "scripts/three_camera_source.py",
                        root / 'scripts/prepare_surface_height.py', root / 'scripts/analyze_height_modes.py', root / 'scripts/three_camera_calibration.py', Path(__file__).with_name('surface_cache.py'), root/'scripts/output_layout.py')
    pipeline_sources += tuple(root/'scripts'/name for name in (
        'refine_horizontal_joint.py', 'horizontal_surfel.py', 'surfel_joint.py', 'kinematic_graph.py',
        'audit_sensor_contract.py', 'audit_input_heading.py', 'prior_ablation.py', 'accuracy/horizontal_cache.py',
        'direct_heading.py', 'run_analysis.py', 'plot_run_analysis.py'))
    if args.sensor_fusion == 'turn_heading_rts':
        pipeline_sources += (root/'scripts/turn_heading.py',)
    write_json(run/'horizontal_settings.json', {'mode':args.horizontal_refinement,
        'explicit_map_cache':str(args.horizontal_map_cache.resolve()) if args.horizontal_map_cache else None,
        'experimental':args.horizontal_refinement != 'none'})
    write_json(run / "pipeline_source_sha256.json", {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in pipeline_sources
    })
    snapshot = run / "pipeline_sources"
    snapshot.mkdir()
    for path in pipeline_sources:
        (snapshot / path.name).write_bytes(path.read_bytes())
    localization_snapshot = run / "localization_sources"
    for path in (root / "workspace/src/localization").rglob("*"):
        if path.is_file() and '__pycache__' not in path.parts:
            destination = localization_snapshot / path.relative_to(root / "workspace/src/localization")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
    write_json(run / "sensor_preparation_params.json", fusion_settings)
    if args.prior_run:
        prep_start = time.perf_counter()
        write_json(run / "sensor_preparation_params.json", {
            "fusion": args.sensor_fusion, "mode": "exact_prior_reuse",
            "source_run": str(args.prior_run.resolve()), "fusion_cli_parameters_applied": False,
            "actual_configuration": "sensor_prior.json; copied from source run",
        })
        reuse_priors(args.prior_run, run, [s for s, _ in files], args.track, args.sensor_fusion)
        sensor_preparation_seconds = time.perf_counter() - prep_start
    if args.profile == "sensor_guided" and not args.prior_run:
        prep_start = time.perf_counter()
        print('Preparing GNSS/IMU priors from the raw bag; see sensor_preparation.log', flush=True)
        prep = ["docker", "exec", container, "bash", "-c",
                "source /opt/ros/humble/setup.bash && exec python3 /root/scripts/prepare_sensor_prior.py "
                f"--bag /root/data/Track{args.track}/Track{args.track}.db3 --frames "
                + shlex.quote(container_run + "/frames.csv") + " --output " + shlex.quote(params["sensor_prior_file"])
                + " " + " ".join("--" + key + " " + shlex.quote(str(value)) for key, value in fusion_settings.items())]
        write_json(run / "sensor_command.json", prep)
        with (run / "sensor_preparation.log").open("w") as log:
            prep_result = subprocess.run(prep, stdout=log, stderr=subprocess.STDOUT)
        sensor_preparation_seconds = time.perf_counter() - prep_start
        if prep_result.returncode != 0:
            write_json(run / "validation.json", {"track": args.track, "profile": args.profile,
                       "sensor_fusion": args.sensor_fusion, "returncode": prep_result.returncode,
                       "sensor_preparation_seconds": sensor_preparation_seconds,
                       "output_complete_and_finite": False, "validation_errors": ["Sensor preparation failed; inspect sensor_preparation.log"]})
            return 1
    height_map_preparation_seconds = 0.
    height_preparation_seconds = 0.
    if args.height_mode.startswith('surface'):
        height_map_source = 'explicit_reuse' if args.height_map_cache else 'built_from_raw_map_in_current_run'
        if not args.height_map_cache:
            prep_start = time.perf_counter()
            print('Building a new 0.3 m height map; see height_map_preparation.log', flush=True)
            try:
                args.height_map_cache = prepare_height_map(run, data_root, args.track, container)
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                write_json(run/'validation.json', {'track':args.track, 'height_mode':args.height_mode,
                    'returncode':1, 'output_complete_and_finite':False,
                    'height_map_preparation_seconds':time.perf_counter()-prep_start,
                    'validation_errors':[str(error)]})
                print(f'Height map preparation failed: {error}', flush=True)
                return 1
            height_map_preparation_seconds = time.perf_counter()-prep_start
        try:
            write_json(run/'height_map_input.json', {'mode':height_map_source,
                'path':str(args.height_map_cache.resolve()), 'sha256':sha256(args.height_map_cache),
                'identity_sha256':sha256(args.height_map_cache.with_suffix(args.height_map_cache.suffix+'.json'))})
        except OSError as error:
            write_json(run/'validation.json', {'track':args.track, 'height_mode':args.height_mode,
                'returncode':1, 'output_complete_and_finite':False, 'validation_errors':[str(error)]})
            return 1
        prep_start = time.perf_counter()
        print('Preparing surface references; see height_preparation.log', flush=True)
        height_command = [str(surface_python), str(root/'scripts/prepare_surface_height.py'),
                          '--run', str(run), '--data-root', str(data_root), '--map-cache', str(args.height_map_cache.resolve())]
        write_json(run/'height_command.json', height_command)
        with (run/'height_preparation.log').open('w') as log:
            height_result = subprocess.run(height_command, stdout=log, stderr=subprocess.STDOUT)
        height_preparation_seconds = time.perf_counter()-prep_start
        if height_result.returncode:
            write_json(run/'validation.json', {'track':args.track, 'height_mode':args.height_mode,
                'returncode':height_result.returncode, 'output_complete_and_finite':False,
                'validation_errors':['Surface preprocessing failed; inspect height_preparation.log. No silent GNSS substitution.']})
            return 1
    command = ["docker", "exec", container, "bash", "-c",
               "source /opt/ros/humble/setup.bash && source /root/midterm/install/setup.bash && "
               "exec ros2 run localization localization --ros-args --params-file " + shlex.quote(container_run + "/params.yaml")]
    (run / "command.sh").write_text("#!/usr/bin/env bash\n" + shlex.join(command) + "\n")
    write_json(run / "source_sha256.json", {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (root / "workspace/src").rglob("*") if path.is_file()
    })
    (run / "worktree.patch").write_bytes(subprocess.check_output(["git", "diff"], cwd=root))
    write_json(run / "container.json", loads_saved_json(subprocess.check_output(["docker", "inspect", container], text=True)))
    (run / "git_commit.txt").write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True))
    (run / "binary_sha256.txt").write_text(subprocess.check_output([
        "docker", "exec", container, "sha256sum",
        "/root/midterm/install/localization/lib/localization/localization",
        "/root/midterm/install/cuda_ndt_backend/lib/libcuda_ndt_backend.so"], text=True))
    (run / "cpu_processes_before.txt").write_text(subprocess.run(["ps", "-C", "localization", "-o", "pid,pcpu,etime,args"], capture_output=True, text=True).stdout)
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
        "height_map_preparation_seconds": height_map_preparation_seconds,
        "height_preparation_seconds": height_preparation_seconds,
        "sensor_fusion": args.sensor_fusion,
        "geometry_mode": args.geometry_mode,
        "height_mode": args.height_mode,
        "cloud_source": args.cloud_source, "ply_scale": args.ply_scale, "start_frame": args.start_frame,
        "source_input": "source_input.json", "experimental_source": args.cloud_source != "front",
        "submission_frame_set_complete": args.start_frame == 0 and args.frames == expected_full_frames,
        "pipeline_elapsed_seconds": elapsed + sensor_preparation_seconds + height_map_preparation_seconds + height_preparation_seconds,
        "frames": len(poses), "expected_frames": args.frames, "elapsed_seconds": elapsed,
        "time_scope": "Docker exec + ROS startup + map initialization + all frames; no build. See map_cache_file for explicit cache use.",
        "map_cache_file": cache_file,
        "map_cache_hit_at_start": cache_hit,
        "returncode": returncode, "failed_alignments": sum(row["converged"] != "1" for row in timings),
        "failed_alignments_scope": "rejected map updates with a current sensor-prior pose still emitted" if args.profile == "sensor_guided" else "registration not converged",
        "timing_sums": timing_sums, "validation_errors": errors, "output_complete_and_finite": not errors,
        "accuracy": "Not evaluated against ground truth or Kaggle; convergence is not an accuracy guarantee.",
    }
    if args.track == 3:
        summary["track3_execution_scope"] = {
            "path": "map-only forward legacy: current PLY, static map, initial/previous accepted pose",
            "gnss_used": False, "imu_used": False,
            "sensor_preprocessing_called": False, "rts_called": False, "batch_graph_called": False,
            "causality_evidence": "static review only; runtime prefix consistency pending",
            "scale_status": "uncalibrated; explicit user-supplied ply_scale",
            "format_limitations": "local 0-based idx and existing yaw export; official sample idx/yaw units/frame not verified",
            "submission_compliance_certified": False,
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
        prior_metadata = loads_saved_json((run / "sensor_prior.json").read_text())
        if args.sensor_fusion != "fixed":
            fusion_meta = prior_metadata.get("sensor_fusion", {})
            summary["sensor_fusion_diagnostics_not_accuracy"] = fusion_meta
            if fusion_meta.get("mode") != args.sensor_fusion:
                errors.append("Sensor fusion metadata does not match requested mode")
            with (run / "sensor_prior.fusion.csv").open() as stream:
                fusion_rows = list(csv.DictReader(stream))
            if ([int(row["timestamp_ns"]) for row in fusion_rows] != [stamp for stamp, _ in files[:args.frames]]):
                errors.append("Fusion diagnostics timestamps/count mismatch")
            if any(not math.isfinite(float(value)) for row in fusion_rows for key, value in row.items() if key != "timestamp_ns"):
                errors.append("Nonfinite sensor fusion diagnostics")
        with (run / "guidance.csv").open() as stream:
            guidance = list(csv.DictReader(stream))
        try:
            with (run / "height.csv").open() as stream:
                heights = list(csv.DictReader(stream))
            if len(heights) != args.frames or [int(row["timestamp_ns"]) for row in heights] != [s for s, _ in files]:
                errors.append("Height diagnostics count/timestamp mismatch")
            if any(row["mode"] != args.height_mode or not all(math.isfinite(float(row[k])) for k in
                ("nominal_sensor_z", "prior_z", "output_z", "offset_from_nominal_z")) for row in heights):
                errors.append("Invalid height diagnostics")
            maximum_offset = max((abs(float(row["offset_from_nominal_z"])) for row in heights), default=0.)
            summary["height_diagnostics_not_accuracy"] = {"mode": args.height_mode,
                "max_abs_offset_from_nominal_sensor_z_m": maximum_offset,
                "nominal_sensor_z_is_ground_truth": False}
            reference_offset = max((abs(float(row['offset_from_reference_z'])) for row in heights), default=0.)
            summary['height_diagnostics_not_accuracy']['max_abs_offset_from_reference_z_m'] = reference_offset
            if args.height_mode != 'carry' and reference_offset > 1.50002:
                errors.append("Absolute height offset exceeded +/-1.5 m bound")
            if args.height_mode.startswith('surface'):
                height_report = loads_saved_json((run/'height_reference.json').read_text())
                if height_report['reference_sha256'] != sha256(run/'height_reference.csv') or height_report['sensor_prior_sha256'] != sha256(run/'sensor_prior.csv'):
                    errors.append('Surface reference hash mismatch')
                with (run/'height_reference.csv').open() as stream:
                    references = list(csv.DictReader(stream))
                if len(references) != len(heights) or any(int(a['timestamp_ns']) != int(b['timestamp_ns']) or abs(float(a['z'])-float(b['reference_z'])) > 1e-8 for a,b in zip(references, heights)):
                    errors.append('Surface reference/solver count, timestamp or altitude mismatch')
                summary['height_diagnostics_not_accuracy']['surface_support_frames'] = height_report['supported_frames']
                if args.height_mode == 'surface_tilt':
                    tilt_angles=[]
                    with (run/'sensor_prior.csv').open() as stream:
                        original_priors=list(csv.DictReader(stream))
                    for ref,observation,pose in zip(references,original_priors,poses):
                        q=[float(ref[k]) for k in ('qx','qy','qz','qw')]
                        old=[float(observation[k]) for k in ('qx','qy','qz','qw')]
                        dot=abs(sum(a*b for a,b in zip(q,old)))
                        tilt_angles.append(math.degrees(2*math.acos(min(1.,dot))))
                        x,y,z,w=q
                        row_z=[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]
                        if max(abs(a-b) for a,b in zip(row_z,pose[8:11]))>1e-6:
                            errors.append('Output tilt differs from geometric reference')
                    maximum_tilt=max(tilt_angles,default=0.)
                    summary['height_diagnostics_not_accuracy']['max_geometric_tilt_deg']=maximum_tilt
                    if maximum_tilt>15.0001:
                        errors.append('Geometric tilt exceeded 15 degree bound')
            if any(abs(float(row["output_z"]) - pose[11]) > 1e-5 for row, pose in zip(heights, poses)):
                errors.append("Height diagnostics differ from result matrices")
        except (OSError, ValueError, KeyError) as error:
            errors.append(str(error))
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
        update_summary, update_errors = summarize_guided_updates(guidance)
        summary["guidance_not_accuracy"].update(update_summary)
        errors.extend(update_errors)
        if summary["guidance_not_accuracy"]["max_xy_correction_m"] > 1.500001 or summary["guidance_not_accuracy"]["max_yaw_correction_deg"] > 8.000001:
            errors.append("Sensor correction exceeded configured default bounds")
        if any(not math.isfinite(float(row[key])) for row in guidance for key in ("prior_cost", "final_cost", "dx", "dy", "dz", "dyaw")):
            errors.append("Nonfinite guidance metrics")
        if any(float(row["final_cost"]) > float(row["prior_cost"]) + 1e-8 for row in guidance):
            errors.append("Map update worsened regularized objective")
        summary["output_complete_and_finite"] = not errors
    if args.geometry_mode != "off":
        geometry_summary, geometry_errors = validate_geometry(run, [stamp for stamp, _ in files[:args.frames]],
                                args.geometry_mode, args.geometry_normal_gain, args.geometry_tangent_weight)
        summary["geometry_diagnostics_not_accuracy"] = geometry_summary
        errors.extend(geometry_errors)
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
    if args.horizontal_refinement != 'none' and not errors:
        print('Fine horizontal map / joint estimation; see horizontal_preparation.log', flush=True)
        post_start = time.perf_counter()
        post = [str(surface_python), str(root/'scripts/refine_horizontal_joint.py'), '--run', str(run),
                '--data-root', str(data_root), '--container', container]
        if args.horizontal_map_cache:
            post += ['--fine-map', str(args.horizontal_map_cache.resolve())]
        write_json(run/'horizontal_command.json', post)
        with (run/'horizontal_preparation.log').open('w') as log:
            post_result = subprocess.run(post, stdout=log, stderr=subprocess.STDOUT)
        summary['horizontal_refinement'] = args.horizontal_refinement
        summary['horizontal_refinement_seconds'] = time.perf_counter()-post_start
        summary['pipeline_elapsed_seconds'] += summary['horizontal_refinement_seconds']
        summary['base_diagnostics_scope'] = 'C++ base stage; final horizontal stage separately validated in horizontal/report.json'
        if post_result.returncode:
            errors.append('Horizontal refinement failed; inspect horizontal_preparation.log, preserved base outputs in base/')
        else:
            report = loads_saved_json((run/'horizontal/report.json').read_text())
            summary['horizontal_diagnostics_not_accuracy'] = report
            if (not report['output_complete_and_finite'] or report['result_sha256'] != sha256(run/'result.txt')
                    or report['csv_sha256'] != sha256(run/'diagnostic.csv')):
                errors.append('Final horizontal output validation/hash mismatch')
            with (run/'diagnostic.csv').open() as stream:
                final_csv = list(csv.DictReader(stream))
            if (len(final_csv) != args.frames or [int(row['idx']) for row in final_csv] != list(range(args.frames))
                    or any(not math.isfinite(float(row[k])) for row in final_csv for k in ('x','y','yaw'))):
                errors.append('Final horizontal CSV format/count/finite mismatch')
        summary['output_complete_and_finite'] = not errors
        write_json(run/'validation.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    from run_analysis import generate_after_full
    generate_after_full(run, summary, full=full_attempt)
    print_run_outcome(run, summary, full_attempt=full_attempt)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

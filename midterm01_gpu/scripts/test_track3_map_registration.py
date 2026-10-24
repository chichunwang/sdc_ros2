#!/usr/bin/env python3
"""Run isolated map-registration experiments, never a Track trajectory/submission.

Known map-derived transforms test recovery; current raw PLY probes test geometric
fit and initial-value ambiguity. No bag/IMU or old trajectory is read.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from accuracy.benchmark import create_run_directory
from causal_track3_filter import read_initial_pose
from cuda_scale_input import load_ply_gpu
from run_track3_causal import digest, gpu_crop_voxel, write_json
from track3_registration import (MapRegistrar, RegistrationConfig, block_validation,
                                  assess_multistart)

ROOT = Path(__file__).resolve().parents[1]
METHOD = 'track3_map_only_ndt_icp_research_v2_training_initialization'


def source_snapshot(run):
    names = ['track3_registration.py', 'test_track3_map_registration.py', 'cuda_scale_geometry.py',
             'cuda_scale_input.py', 'three_camera_calibration.py', 'three_camera_source.py',
             'run_track3_causal.py', 'causal_track3_filter.py', 'map_scale_icp.py',
             'horizontal_surfel.py', 'output_layout.py', 'accuracy/benchmark.py', 'old_icp_runtime.py',
             'tests/test_track3_registration_core.py']
    hashes = {}
    for name in names:
        source = ROOT/'scripts'/name
        target = run/'pipeline_sources/scripts'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        hashes['scripts/'+name] = digest(source)
    write_json(run/'pipeline_source_sha256.json', hashes)


def verified_map_cache(path, map_path):
    path = path.resolve(strict=True)
    path.relative_to((ROOT/'outputs/run').resolve())
    metadata = json.loads((path.parent/'map_xyz.json').read_text())
    if digest(map_path) != metadata['source_sha256'] or digest(path) != metadata['cache_sha256']:
        raise ValueError('Map/cache SHA does not match recorded source')
    points = np.load(path, mmap_mode='r')
    if list(points.shape) != metadata['shape'] or str(points.dtype) != metadata['dtype']:
        raise ValueError('Map cache schema mismatch')
    return points, dict(path=str(path), **metadata)


def rotation_error(pose, expected):
    return float(np.rad2deg(Rotation.from_matrix(np.asarray(pose)[:3, :3] @ expected[:3, :3].T).magnitude()))


def map_recovery(scene, points, initial, run):
    # Synthetic SOURCES derived from a real map. These errors have known truth
    # but do not measure camera reconstruction or real Track accuracy.
    local = (points-initial[:3, 3]) @ initial[:3, :3]
    selected = ((local[:, 0] > 1) & (local[:, 0] < 30) & (abs(local[:, 1]) < 20)
                & (local[:, 2] > -6) & (local[:, 2] < 12))
    local = local[selected]
    if len(local) > 10000:
        local = local[np.linspace(0, len(local)-1, 10000, dtype=int)]
    rng = np.random.default_rng(20261023)
    reports = []
    for i, (scale, translation, angle, outliers) in enumerate([
            (3., [.6, -.4, .3], [2., -1., 3.], 0.),
            (7., [-1., .7, -.4], [-3., 2., -5.], .15),
            (1.5, [.2, -.6, .2], [0., 0., 2.], 0.),
            (6., [2., -1.5, .5], [5., -4., 12.], 0.),
            (9., [-2., 2., -.7], [-7., 5., -15.], .05)]):
        raw = (local+rng.normal(0, .015, local.shape))/scale
        if outliers:
            n = int(len(raw)*outliers)
            raw[:n] = rng.uniform([-5, -5, -3], [15, 5, 6], (n, 3))/scale
        seed = initial.copy()
        seed[:3, :3] = Rotation.from_euler('xyz', angle, degrees=True).as_matrix() @ initial[:3, :3]
        seed[:3, 3] += translation
        for use_ndt in (False, True):
            started = time.perf_counter()
            report = scene.fit(raw, seed, scale*1.12, use_ndt=use_ndt)
            report.update(kind='map_derived_known_transform', case=i, use_ndt=use_ndt,
                          true_scale=scale, translation_error_m=float(np.linalg.norm(np.array(report['pose'])[:3, 3]-initial[:3, 3])),
                          rotation_error_deg=rotation_error(report['pose'], initial),
                          relative_scale_error=abs(report['scale']/scale-1),
                          elapsed_seconds=time.perf_counter()-started)
            reports.append(report)
            write_json(run/f'recovery_{i}_{"ndt_icp" if use_ndt else "icp"}.json', report)
            print(f"Recovery {i}, NDT={use_ndt}: position={report['translation_error_m']:.4f} m, rotation={report['rotation_error_deg']:.4f} deg, scale_error={report['relative_scale_error']:.5f}, accepted={report['accepted']}", flush=True)
    return reports


def real_probes(scene, initial, args, run):
    from old_icp_runtime import initial_ground_tilt
    reports = []
    for idx in args.indices:
        files = list((args.data_root/'Track3/raw_ply').glob(f'{idx:06d}_ts_*.ply'))
        if len(files) != 1:
            raise ValueError('Raw idx not unique/found')
        path = files[0]
        started = time.perf_counter()
        raw = load_ply_gpu(path, max_points=12000, leaf=.03)
        load_ms = (time.perf_counter()-started)*1000
        # Ground is an INITIAL ORIENTATION HYPOTHESIS only. No fixed height,
        # absolute scale or zero-tilt constraint enters the optimizer.
        ground, ground_info = initial_ground_tilt(raw[~block_validation(raw)], initial, 3.)
        variants = [('official', initial), ('ground', ground)]
        candidates = []
        for name, seed_pose in variants:
            for scale in args.scales:
                started = time.perf_counter()
                report = scene.fit(raw, seed_pose, scale)
                report.update(kind='real_independent_frame_probe', original_idx=idx,
                              timestamp_ns=int(path.stem.split('_ts_')[1]), filename=path.name,
                              source_sha256=digest(path), seed_type=name, seed_scale=scale,
                              load_ms=load_ms, elapsed_seconds=time.perf_counter()-started,
                              ground_seed=ground_info, imu_used=False,
                              note='Independent current-frame fit at official starting-area crop; not a trajectory')
                write_json(run/f'probe_{idx:06d}_{name}_s{scale:g}.json', report)
                candidates.append(report)
                print(f"Probe idx={idx}, {name}, seed={scale:g}: fitted={report['scale']:.5f}, train={report['training']['median_m']:.3f} m, validation={report['validation']['median_m']:.3f}/{report['validation']['p90_m']:.3f} m, rank={report['physical_rank']}, accepted={report['accepted']}, seconds={report['elapsed_seconds']:.2f}", flush=True)
        consensus = assess_multistart(candidates, scene.config)
        reports.append(dict(original_idx=idx, **consensus,
                            candidates=[dict(seed_type=r['seed_type'], seed_scale=r['seed_scale'],
                                             scale=r['scale'], pose=r['pose'], training_cost=r['training_cost'],
                                             accepted=r['accepted'], checks=r['checks']) for r in candidates]))
        print(f"Frame probe idx={idx}: multi-start accepted={consensus['accepted']}, reasons={consensus['reasons']}", flush=True)
    return reports


def paired_reference(scene, crop, initial, args, run):
    """One real input, same points/config/seed, explicit CPU reference (no fallback)."""
    file = next((args.data_root/'Track3/raw_ply').glob('000000_ts_*.ply'))
    raw = load_ply_gpu(file, max_points=12000, leaf=.03)
    torch.set_num_threads(1)
    start = time.perf_counter()
    reference = MapRegistrar(crop, backend='cpu_reference', config=scene.config)
    preparation = time.perf_counter()-start
    start = time.perf_counter()
    cpu = reference.fit(raw, initial, 9.)
    cpu_seconds = time.perf_counter()-start
    # Warm-up is not included in this paired fit timing.
    scene.geometry.sync()
    start = time.perf_counter()
    cuda = scene.fit(raw, initial, 9.)
    scene.geometry.sync()
    cuda_seconds = time.perf_counter()-start
    output = dict(cpu=cpu, cuda=cuda, cpu_prepare_seconds=preparation,
                  cpu_fit_seconds=cpu_seconds, cuda_fit_seconds=cuda_seconds,
                  max_pose_absolute_difference=float(np.max(abs(np.array(cpu['pose'])-cuda['pose']))),
                  scale_absolute_difference=abs(cpu['scale']-cuda['scale']),
                  same_checks=cpu['checks'] == cuda['checks'],
                  status='One paired input, not a full runtime or pose-accuracy benchmark')
    write_json(run/'cpu_cuda_real_pair.json', output)
    print(f"CPU/CUDA real pair: pose_diff={output['max_pose_absolute_difference']:.3g}, scale_diff={output['scale_absolute_difference']:.3g}, same_checks={output['same_checks']}, CPU={cpu_seconds:.3f}s, CUDA={cuda_seconds:.3f}s", flush=True)
    return {key:value for key,value in output.items() if key not in ('cpu', 'cuda')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map-cache', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=ROOT.parent/'midterm01/data')
    parser.add_argument('--run', type=Path, help='Existing empty research directory from this development session')
    parser.add_argument('--indices', nargs='+', type=int, default=[0, 1, 2, 4, 8, 16, 24, 32])
    parser.add_argument('--scales', nargs='+', type=float, default=[3., 6., 9.])
    parser.add_argument('--recovery-only', action='store_true')
    parser.add_argument('--compare-cpu', action='store_true', help='Explicit one-frame CPU/CUDA numerical and timing pair')
    args = parser.parse_args(argv)
    from old_icp_runtime import active_localizations
    active = active_localizations()
    if active:
        raise RuntimeError(f'Another localization is active: {active}')
    args.data_root = args.data_root.resolve(strict=True)
    run = args.run.resolve(strict=True) if args.run else create_run_directory(ROOT/'outputs/run', METHOD, track=3)
    run.relative_to((ROOT/'outputs/run').resolve())
    if (run/'params.json').exists():
        raise ValueError('Never overwrite an existing experiment')
    print('Run: '+str(run), flush=True)
    write_json(run/'params.json', dict(method=METHOD, config=asdict(RegistrationConfig()),
               indices=args.indices, scales=args.scales, imu_used=False, trajectory_generated=False,
               accuracy_status='known-transform synthetic recovery and real geometry only; no Track truth'))
    source_snapshot(run)
    (run/'git_commit.txt').write_bytes(__import__('subprocess').check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT))
    (run/'worktree.patch').write_bytes(__import__('subprocess').check_output(['git', 'diff'], cwd=ROOT))
    initial = read_initial_pose(args.data_root/'Track3/init_guess.txt')
    started = time.perf_counter()
    points, metadata = verified_map_cache(args.map_cache, args.data_root/'Track3/map.pcd')
    crop, crop_info = gpu_crop_voxel(points, initial[:3, 3], torch, np, half_size=65., leaf=.25)
    write_json(run/'input_map.json', dict(cache=metadata, crop=crop_info,
               initial_sha256=digest(args.data_root/'Track3/init_guess.txt')))
    np.save(run/'map_crop.npy', crop)
    scene = MapRegistrar(crop, backend='cuda', cache_dir=run/'cuda_cache')
    preparation_seconds = time.perf_counter()-started
    print(f'Prepared: map={len(crop)}, planar={int(scene.planar.sum())}, seconds={preparation_seconds:.2f}', flush=True)
    recovery = map_recovery(scene, crop, initial, run)
    probes = [] if args.recovery_only else real_probes(scene, initial, args, run)
    paired = paired_reference(scene, crop, initial, args, run) if args.compare_cpu else None
    summary = dict(method=METHOD, preparation_seconds=preparation_seconds,
                   known_transform_cases=len(recovery),
                   known_transform_passed=sum(r['translation_error_m'] < .1 and r['rotation_error_deg'] < .5
                       and r['relative_scale_error'] < .02 for r in recovery),
                   probes=probes, imu_used=False, trajectory_generated=False,
                   paired_reference=paired,
                   peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                   runtime=dict(python=sys.version, numpy=np.__version__, torch=torch.__version__,
                                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name()),
                   accuracy_status='Real pose accuracy remains unknown; no formal score')
    write_json(run/'research_summary.json', summary)
    print('Research complete: '+str(run/'research_summary.json'), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

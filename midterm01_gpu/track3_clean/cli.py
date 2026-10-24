"""Independent front-PLY/map-XYZ causal localization entrypoint."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shlex
import signal
import sys
import time
import traceback

import numpy as np

from . import METHOD
from .config import RegistrationConfig, TrackingConfig
from .data import ROOT, crop_map, digest, load_ply_xyz, read_initial, sample_source, select_frames, verified_cache
from .output import BACKEND, CPU_REFERENCE_BACKEND, OutputWriter, create_run, validate_output, write_json
from .runtime import active_localizations, analyze_full, configure_cuda, cuda_memory, snapshot_sources


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group()
    g.add_argument('--full', action='store_true')
    g.add_argument('--frames', type=int, default=32)
    p.add_argument('--data-root', type=Path, default=ROOT.parent/'midterm01/data')
    p.add_argument('--map-cache', type=Path, help='Optional SHA-verified lossless XYZ cache inside GPU outputs/run')
    p.add_argument('--vram-budget-gib', type=float, default=4.)
    p.add_argument('--dry-run', action='store_true', help='Read frame names/init only; no CUDA, pose solving or map payload verification')
    a = p.parse_args(argv)
    if a.frames <= 0 or not np.isfinite(a.vram_budget_gib) or a.vram_budget_gib <= 0:
        p.error('frames and vram-budget-gib must be finite positive values')
    a.data_root = a.data_root.resolve(strict=True)
    if a.map_cache:
        a.map_cache = a.map_cache.resolve(strict=True)
    return a


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    a = arguments(argv)
    selected = select_frames(a.data_root, full=a.full, count=a.frames)
    initial = read_initial(a.data_root/'Track3/init_guess.txt')
    tc, rc = TrackingConfig(), RegistrationConfig()
    plan = dict(track=3, method=METHOD, backend=BACKEND, full=a.full, frames=len(selected),
                registration_config=asdict(rc), tracking_config=asdict(tc),
                imu_used=False, gnss_used=False, rts_used=False, graph_used=False,
                offline_trajectory_correction=False, external_predictions_used=False,
                coordinate_model='p_map = scale * R_map_source @ p_raw + t_map_m; pose stores R,t only',
                pair_model='fit current Sim3 into previous camera metres; compose R,t once and retain fitted metric scale',
                camera_assumptions='PLY origin is camera centre, +X forward; not independently calibrated',
                accuracy='unknown; clean architecture and geometric acceptance do not establish pose accuracy',
                replay_supported=False, vram_budget_gib=a.vram_budget_gib,
                vram_budget_scope='PyTorch allocator only, not all GPU allocations',
                first_timestamp_ns=selected[0].timestamp_ns, last_timestamp_ns=selected[-1].timestamp_ns)
    if a.dry_run:
        plan.update(localization_executed=False, cuda_initialized=False, map_payload_sha_verified=False)
        print(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    active = active_localizations()
    if active:
        print(f'Failed: another localization is active: {active}; existing process preserved.', file=sys.stderr, flush=True)
        return 1
    run = create_run(full=a.full)
    print('Run: '+str(run), flush=True)
    started, torch = time.perf_counter(), None
    source_rows, hashes = [], {}
    summary = dict(track=3, method=METHOD, backend=BACKEND, frames=len(selected), expected_frames=len(selected),
                   processed_frames=0, map_accepted_frames=0, pair_accepted_frames=0,
                   tracking_lost_frames=0, tracking_state_counts={}, max_unsupported_seconds=0.,
                   returncode=1, validation_errors=[], localization_executed=False,
                   output_complete_and_finite=False, submission_frame_set_complete=False,
                   imu_used=False, gnss_used=False, replay_supported=False, submission_compliance_certified=False,
                   accuracy=plan['accuracy'])
    write_json(run/'params.json', plan)
    write_json(run/'test_info.json', dict(track=3, algorithm=METHOD, full=a.full, score=None, localization_executed=False))
    (run/'command.sh').write_text(shlex.join([sys.executable, str(ROOT/'scripts/run_track3_clean.py'), *argv])+'\n')
    old_sigterm = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, _frame):
        raise InterruptedError(f'Process received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        summary['stage'] = 'source_snapshot'
        hashes = snapshot_sources(run)
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        import torch as torch_module
        torch = torch_module
        summary['stage'] = 'cuda_configuration'
        write_json(run/'cuda_runtime.json', configure_cuda(torch, a.vram_budget_gib))
        from .geometry import Geometry
        from .registration import MapRegistrar
        from .engine import LocalizationEngine
        identities = {name: dict(path=str(a.data_root/'Track3'/name), sha256=digest(a.data_root/'Track3'/name))
                      for name in ('map.pcd', 'init_guess.txt')}
        write_json(run/'input_sha256.json', identities)
        prepare = time.perf_counter()
        summary['stage'] = 'map_preparation'
        if a.map_cache:
            points, cache_info = verified_cache(a.map_cache, a.data_root/'Track3/map.pcd')
        else:
            from .map_cache import cache_xyz_map
            path = cache_xyz_map(a.data_root/'Track3/map.pcd', run/'map_cache')
            points, cache_info = verified_cache(path, a.data_root/'Track3/map.pcd')
        write_json(run/'map_cache_reference.json', cache_info)
        geometry = Geometry('cuda', run/'cuda_cache')
        backend = BACKEND if geometry.backend == 'cuda' else CPU_REFERENCE_BACKEND
        # cpu_reference can only be injected by the synthetic harness. The CLI
        # always requests CUDA; configure_cuda fails if it is unavailable.
        summary['backend'] = plan['backend'] = backend
        write_json(run/'params.json', plan)
        center = initial[:3, 3].copy()
        crop, tile_info = crop_map(points, center, geometry, half_size=tc.map_half_size_m, leaf=tc.map_leaf_m)
        engine = LocalizationEngine(MapRegistrar(crop, backend=geometry.backend, geometry=geometry), initial, tc)
        write_json(run/'initial_map_tile.json', tile_info)
        geometry.sync()
        summary['preparation_seconds'] = time.perf_counter()-prepare
        print(f"Preparation: map_points={len(crop)}, time={summary['preparation_seconds']:.3f} s (excluded from frame timings)", flush=True)
        with OutputWriter(run, backend=backend) as writer:
            summary['localization_executed'] = True
            write_json(run/'test_info.json', dict(track=3, algorithm=METHOD, full=a.full, score=None, localization_executed=True))
            for i, frame in enumerate(selected):
                summary['failed_original_idx'] = frame.original_idx
                summary['stage'] = 'source_load'
                begin = time.perf_counter()
                before = digest(frame.path)
                raw = sample_source(load_ply_xyz(frame.path), geometry, leaf=tc.source_leaf_raw, max_points=tc.source_max_points)
                geometry.sync()
                loaded = time.perf_counter()
                summary['stage'] = 'registration'
                if np.max(abs(engine.memory.published.pose[:3, 3]-center)) > tc.map_recenter_distance_m:
                    center = engine.memory.published.pose[:3, 3].copy()
                    engine.scene = None
                    crop, tile_info = crop_map(points, center, geometry, half_size=tc.map_half_size_m, leaf=tc.map_leaf_m)
                    engine.scene = MapRegistrar(crop, backend=geometry.backend, geometry=geometry)
                    write_json(run/f'map_tile_{i:06d}.json', tile_info)
                result = engine.step(raw, frame.timestamp_ns)
                geometry.sync()
                finished = time.perf_counter()
                load_ms, reg_ms, total_ms = (loaded-begin)*1000, (finished-loaded)*1000, (finished-begin)*1000
                if digest(frame.path) != before:
                    raise RuntimeError('Current PLY changed during processing')
                source_rows.append(dict(idx=i, original_idx=frame.original_idx, timestamp_ns=frame.timestamp_ns,
                                        filename=frame.path.name, sha256=before))
                summary['stage'] = 'output'
                writer.append(frame, result, load_ms=load_ms, registration_ms=reg_ms, total_ms=total_ms,
                              elapsed_seconds=total_ms/1000)
                summary['processed_frames'] += 1
                summary['map_accepted_frames'] += int(result.map_accepted)
                summary['pair_accepted_frames'] += int(result.pair_accepted)
                summary['tracking_lost_frames'] += int(result.state.value == 'lost')
                summary['max_unsupported_seconds'] = max(summary['max_unsupported_seconds'], result.unsupported_seconds)
                counts = summary['tracking_state_counts']
                counts[result.state.value] = counts.get(result.state.value, 0)+1
                summary['final_unsupported_seconds'] = result.unsupported_seconds
                print(f'Frame {i+1}/{len(selected)}: idx={frame.original_idx}, map={"accepted" if result.map_accepted else "rejected"}, scale={result.estimate.scale:.6f}, load={load_ms:.2f} ms, registration={reg_ms:.2f} ms, frame={total_ms:.2f} ms', flush=True)
        print('Graph: executed=false, applied=false, time=0.00 ms', flush=True)
        summary['stage'] = 'output_validation'
        errors = validate_output(run, len(selected), backend=backend)
        errors += ['Program changed: '+p for p, sha in hashes.items() if digest(ROOT/p) != sha]
        errors += ['Input changed: '+name for name, entry in identities.items() if digest(entry['path']) != entry['sha256']]
        errors += ['PLY changed: '+frame.path.name for frame, record in zip(selected, source_rows) if digest(frame.path) != record['sha256']]
        if errors:
            raise RuntimeError('; '.join(errors))
        summary.pop('failed_original_idx', None)
        summary.update(returncode=0, output_complete_and_finite=True, submission_frame_set_complete=bool(a.full),
                       csv_sha256=digest(run/'diagnostic.csv'), result_sha256=digest(run/'result.txt'), stage='complete')
    except (Exception, KeyboardInterrupt) as error:
        reason = f'{type(error).__name__}: {error}'
        summary['validation_errors'].append(reason)
        summary['failure'] = dict(type=type(error).__name__, message=str(error),
                                  processed_frames=summary['processed_frames'], failed_original_idx=summary.get('failed_original_idx'),
                                  stage=summary.get('stage'))
        if hasattr(error, 'diagnostics'):
            write_json(run/'failed_frame_diagnostics.json', error.diagnostics)
        (run/'failure.log').write_text(traceback.format_exc())
        print('Failed: '+reason, file=sys.stderr, flush=True)
    finally:
        signal.signal(signal.SIGTERM, old_sigterm)
        summary['pipeline_elapsed_seconds'] = time.perf_counter()-started
        summary['cuda_memory'] = cuda_memory(torch)
        write_json(run/'source_input.json', dict(track=3, frames=source_rows))
        write_json(run/'validation.json', summary)
    if summary['returncode']:
        return 1
    analyze_full(run, summary, full=a.full)
    k = summary['map_accepted_frames']
    print(f"Finished: frames={len(selected)}, accepted_map={k}, rejected_map={len(selected)-k}, total={summary['pipeline_elapsed_seconds']:.3f} s", flush=True)
    print('CSV: '+str(run/'diagnostic.csv'), flush=True)
    print('Validation: '+str(run/'validation.json'), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

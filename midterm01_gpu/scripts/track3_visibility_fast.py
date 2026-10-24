"""Diagnostic camera-ray depth consistency, independent of ICP correspondences.

Assumes the PLY origin is the camera centre and its +X is forward (uncertified).
Static LiDAR visibility, sparse sampling and moving objects limit this check.
It does not supply ground truth or alter a pose. Dense operations use CUDA.
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from track3_registration import checked_pose, point_jacobian, source_weights, xyz


def projective_pairs(scene, source, pose, angular_radius=.02):
    """Visible target point for each positive-X source bearing; CUDA bulk work."""
    g = scene.geometry
    target = (scene.target-g.tensor(np.asarray(pose)[:3, 3])) @ g.tensor(np.asarray(pose)[:3, :3])
    if not bool((source[:, 0] > 1e-6).all()):
        raise ValueError('Ray diagnostic requires positive raw camera +X')
    bearing = source[:, 1:]/source[:, :1]
    near, far = bearing.amin(0)-angular_radius, bearing.amax(0)+angular_radius
    original = (target[:, 0] > .2).nonzero().ravel()
    eligible = target[original]
    projected = eligible[:, 1:]/eligible[:, :1]
    in_view = ((projected >= near) & (projected <= far)).all(1)
    original, eligible, projected = original[in_view], eligible[in_view], projected[in_view]
    if len(eligible) < 32:
        return None
    cloud = torch.cat([projected, torch.zeros((len(projected), 1), device=source.device)], 1)
    tree = g.tree(cloud, leaf=angular_radius)
    query = torch.cat([bearing, torch.zeros((len(bearing), 1), device=source.device)], 1)
    distance, ids = g.query(tree, query, 24, angular_radius)
    depths = torch.where(torch.isfinite(distance), eligible[ids, 0], torch.inf)
    best = depths.argmin(1)
    row = torch.arange(len(source), device=source.device)
    depth = depths[row, best]
    return original[ids[row, best]], depth, torch.isfinite(depth), len(eligible)


def ray_depth_diagnostic(scene, raw, pose, scale, train_mask, *, angular_radius=.02):
    g = scene.geometry
    source = g.tensor(raw)
    pairs = projective_pairs(scene, source, pose, angular_radius)
    if pairs is None:
        return dict(available=False, reason='insufficient_visible_map_sampling')
    # Exact bounded angular-nearest search, followed by front-most of these
    # samples. No depth value from source participates in selecting map points.
    _, map_depth, supported, count = pairs
    log_ratio = torch.log((source[:, 0]*scale)/map_depth)
    result = dict(available=True, angular_radius_projected=angular_radius,
                  map_samples=count,
                  assumption='raw +X forward, PLY camera origin; not independently calibrated',
                  scope='front-most among 24 angular neighbors, not exact ray casting')
    values = log_ratio.cpu().numpy()
    has = supported.cpu().numpy()
    for name, mask in [('training', train_mask), ('validation', ~train_mask)]:
        used = np.asarray(mask, bool) & has
        v = values[used]
        result[name] = dict(points=int(used.sum()), support_fraction=float(used.sum()/max(np.sum(mask), 1)),
                            median_abs_log_depth_ratio=float(np.median(abs(v))) if len(v) else None,
                            median_depth_ratio=float(np.exp(np.median(v))) if len(v) else None,
                            fraction_front_of_map=float(np.mean(v < np.log(.8))) if len(v) else None,
                            fraction_behind_map=float(np.mean(v > np.log(1.25))) if len(v) else None)
    g.sync()
    return result


def projective_refine(scene, raw, initial, initial_scale, train_mask, *, iterations=40):
    """Experimental projective ICP, never overwrites original fits/trajectories.

    Uses target visibility to avoid matching a foreground reconstructed wall to
    arbitrary nearby ground. Relative-depth residual scale is an explicit 10%
    model assumption plus a .25m map discretization floor, not a calibration.
    """
    g, c = scene.geometry, scene.config
    raw, pose = xyz(raw), checked_pose(initial)
    anchor = pose.copy()
    source = g.tensor(raw[train_mask])
    weights = source_weights(source)
    scale = float(initial_scale)
    status = dict(converged=False, iterations=0, stop='iteration_limit')

    def terms(p, s):
        pairs = projective_pairs(scene, source, p)
        if pairs is None:
            return None
        ids, depths, good, _ = pairs
        rotated = s*source @ g.tensor(p[:3, :3]).T
        residual = rotated+g.tensor(p[:3, 3])-scene.target[ids]
        sigma = torch.maximum(depths*.1, torch.full_like(depths, .25))
        sigma = torch.where(good, sigma, torch.ones_like(sigma))
        r2 = residual.square().sum(1)/sigma.square()
        cost = (torch.where(good, r2/(r2+9.), torch.ones_like(r2))*weights).sum()/weights.sum()
        return cost, rotated, residual, r2, sigma, good

    first = terms(pose, scale)
    initial_cost = float(first[0]) if first is not None else 1.
    for iteration in range(iterations):
        status['iterations'] = iteration+1
        data = terms(pose, scale)
        if data is None or int(data[-1].sum()) < c.min_support:
            status['stop'] = 'insufficient_visibility'
            break
        cost, rotated, residual, r2, sigma, good = data
        j = point_jacobian(rotated)
        w = 9./(r2+9.).square()/sigma.square()*good*weights
        h = ((j.transpose(1, 2) @ j)*w[:, None, None]).sum(0).cpu().numpy()
        b = ((j.transpose(1, 2) @ residual[:, :, None])[:, :, 0]*w[:, None]).sum(0).cpu().numpy()
        units = np.r_[np.ones(3), [.05]*3, .1]
        hs = h*units[:, None]*units[None, :]
        step = np.linalg.lstsq(hs+np.eye(7)*max(np.trace(hs), 1.)*1e-7, -b*units, rcond=1e-7)[0]*units
        factor = min(1., .5/max(np.linalg.norm(step[:3]), 1e-12),
                     np.deg2rad(2.)/max(np.linalg.norm(step[3:6]), 1e-12), .05/max(abs(step[6]), 1e-12))
        step *= factor
        if np.linalg.norm(step[:3]) < .003 and np.linalg.norm(step[3:6]) < .0003 and abs(step[6]) < .0005:
            status.update(converged=True, stop='small_step')
            break
        moved = False
        for alpha in (1., .5, .25, .125, .0625):
            proposed = pose.copy()
            proposed[:3, :3] = Rotation.from_rotvec(alpha*step[3:6]).as_matrix() @ pose[:3, :3]
            proposed[:3, 3] += alpha*step[:3]
            new_scale = scale*np.exp(alpha*step[6])
            if not scene._inside(proposed, new_scale, anchor):
                continue
            value = terms(proposed, new_scale)
            if value is not None and float(value[0]) < float(cost)-1e-10:
                pose, scale, moved = proposed, float(new_scale), True
                break
        if not moved:
            status['stop'] = 'no_projective_descent'
            break
    end = terms(pose, scale)
    report = scene.evaluate(raw, pose, scale, train_mask, True)
    report.update(pose=pose.tolist(), scale=scale, optimizer=status,
                  initial_projective_cost=initial_cost, final_projective_cost=float(end[0]) if end is not None else 1.,
                  ray_depth=ray_depth_diagnostic(scene, raw, pose, scale, train_mask),
                  accepted=False, status='experimental visibility refinement; no acceptance certification')
    g.sync()
    return report


def main(argv=None):
    """Reproduce isolated visibility diagnostics in a fresh research run."""
    import argparse
    import json
    from pathlib import Path
    import time
    from accuracy.benchmark import create_run_directory
    from cuda_scale_input import load_ply_gpu
    from old_icp_runtime import active_localizations
    from run_track3_causal import digest, write_json
    from track3_registration import MapRegistrar, block_validation

    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-run', type=Path, required=True)
    parser.add_argument('--refine', action='store_true', help='Experimental refinement of best TRAINING ray candidate per frame')
    parser.add_argument('--indices', nargs='+', type=int)
    args = parser.parse_args(argv)
    active = active_localizations()
    if active:
        raise RuntimeError(f'Another localization is active: {active}')
    reference = args.research_run.resolve(strict=True)
    reference.relative_to((root/'outputs/run').resolve())
    summary = json.loads((reference/'research_summary.json').read_text())
    if summary.get('trajectory_generated') is not False or summary.get('imu_used') is not False:
        raise ValueError('Expected isolated map-only research inputs, not a Track trajectory')
    run = create_run_directory(root/'outputs/run', 'track3_camera_ray_depth_diagnostic', track=3)
    print('Run: '+str(run), flush=True)
    hashes = {'map_crop.npy': digest(reference/'map_crop.npy')}
    write_json(run/'params.json', dict(reference_run=str(reference), refine=args.refine,
        indices=args.indices, camera_origin_and_raw_axes_certified=False, trajectory_generated=False,
        model='front-most of 24 angular neighbors within projected radius .02',
        refinement_relative_depth_noise_assumption=.1, refinement_discretization_floor_m=.25))
    copies = {}
    for name in ('track3_registration.py', 'track3_visibility.py', 'cuda_scale_geometry.py',
                 'cuda_scale_input.py', 'old_icp_runtime.py', 'tests/test_track3_visibility.py'):
        target = run/'pipeline_sources/scripts'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root/'scripts'/name).read_bytes())
        copies[name] = digest(target)
    write_json(run/'pipeline_source_sha256.json', copies)
    scene = MapRegistrar(np.load(reference/'map_crop.npy'), cache_dir=run/'cuda_cache')
    diagnostics, refinements = [], []
    for frame in summary['probes']:
        idx = frame['original_idx']
        if args.indices is not None and idx not in args.indices:
            continue
        candidates = []
        rawfile = root.parent/'midterm01/data/Track3/raw_ply'/frame['selected']['filename']
        raw = load_ply_gpu(rawfile, max_points=12000, leaf=.03)
        mask = ~block_validation(raw)
        hashes[str(rawfile)] = digest(rawfile)
        for path in sorted(reference.glob(f'probe_{idx:06d}_*.json')):
            fit = json.loads(path.read_text())
            value = ray_depth_diagnostic(scene, raw, fit['pose'], fit['scale'], mask)
            value.update(original_idx=idx, seed_type=fit['seed_type'], seed_scale=fit['seed_scale'],
                         fitted_scale=fit['scale'], fit_file=str(path), fit_sha256=digest(path))
            write_json(run/path.name, value)
            diagnostics.append(value)
            if value['available'] and value['training']['median_abs_log_depth_ratio'] is not None:
                candidates.append((value['training']['median_abs_log_depth_ratio'], fit, path))
        if args.refine and candidates:
            _, fit, path = min(candidates, key=lambda v:v[0])
            start = time.perf_counter()
            result = projective_refine(scene, raw, fit['pose'], fit['scale'], mask)
            result.update(original_idx=idx, fit_file=str(path), fit_sha256=digest(path),
                          elapsed_seconds=time.perf_counter()-start)
            write_json(run/f'refine_{idx:06d}.json', result)
            refinements.append(result)
            print(f"Projective probe {idx}: stop={result['optimizer']['stop']}, certified=false", flush=True)
    write_json(run/'input_sha256.json', hashes)
    write_json(run/'visibility_summary.json', diagnostics)
    write_json(run/'projective_refinements.json', refinements)
    print('Research complete: '+str(run), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

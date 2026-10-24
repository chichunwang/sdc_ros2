"""Optional map visibility gate; assumes camera origin and +X forward."""
import numpy as np
import torch

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
    tree = g.tree(cloud.cpu().numpy(), leaf=angular_radius)
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



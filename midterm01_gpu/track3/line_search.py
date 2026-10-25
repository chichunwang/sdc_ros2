"""Bounded Armijo backtracking; convergence uses the undamped solver proposal."""
import numpy as np
from .geometry import apply_delta


def array_module(value):
    if hasattr(value,'__cuda_array_interface__'):
        import cupy as cp
        return cp
    return np


def step_is_small(delta):
    xp=array_module(delta)
    return bool(xp.linalg.norm(delta[:3]) < .005 and
                xp.linalg.norm(delta[3:6]) < .0005 and abs(delta[6]) < .0005)


def backtrack(evaluate, T, scale, delta, score, gradient, source_count, scale_bounds):
    xp=array_module(delta)
    slope = float(2*xp.dot(gradient,delta)/source_count)
    transform=apply_delta
    if xp is not np:
        from .cuda_math import apply_delta as transform
    trials = []
    if not np.isfinite(slope) or slope >= 0:
        return None, trials
    # A finite safeguard, not a grid search over algorithm parameters.
    for halving in range(20):
        factor = 2.**(-halving)
        t, s = transform(T,scale,delta*factor)
        if not scale_bounds[0] < s < scale_bounds[1]:
            trials.append(dict(factor=factor,cost=None,coverage=None,reason='scale_bounds'))
            continue
        after = evaluate(t,s)
        cost = None if after is None else float(after[2])
        trials.append(dict(factor=factor,cost=cost,coverage=None if after is None else float(after[3])))
        if cost is not None and np.isfinite(cost) and cost <= score+1e-4*factor*slope:
            return (t,s,after,factor),trials
    return None,trials

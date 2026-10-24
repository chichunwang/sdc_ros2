"""Bounded Armijo backtracking; convergence uses the undamped solver proposal."""
import numpy as np
from .geometry import apply_delta


def step_is_small(delta):
    return bool(np.linalg.norm(delta[:3]) < .005 and
                np.linalg.norm(delta[3:6]) < .0005 and abs(delta[6]) < .0005)


def backtrack(evaluate, T, scale, delta, score, gradient, source_count, scale_bounds):
    slope = float(2*np.dot(gradient,delta)/source_count)
    trials = []
    if not np.isfinite(slope) or slope >= 0:
        return None, trials
    # A finite safeguard, not a grid search over algorithm parameters.
    for halving in range(20):
        factor = 2.**(-halving)
        t, s = apply_delta(T,scale,delta*factor)
        if not scale_bounds[0] < s < scale_bounds[1]:
            trials.append(dict(factor=factor,cost=None,coverage=None,reason='scale_bounds'))
            continue
        after = evaluate(t,s)
        cost = None if after is None else float(after[2])
        trials.append(dict(factor=factor,cost=cost,coverage=None if after is None else float(after[3])))
        if cost is not None and np.isfinite(cost) and cost <= score+1e-4*factor*slope:
            return (t,s,after,factor),trials
    return None,trials

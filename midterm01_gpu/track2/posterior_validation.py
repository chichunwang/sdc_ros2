"""Current-frame cost checks; callbacks keep point-cloud evaluation on CUDA."""
import numpy as np


def make_proposal_validator(*, state_for_proposal, valid, fixed_cost,
                            baseline_fixed_cost, common_cost=None, baseline_common_cost=None):
    """Check the emitted state against the prediction, never the unfused center.

    Fixed planes test the solver's local model. Rematching the SAME raw sample
    tests whether that model used the appropriate nearby surface. Unmatched
    points keep their original penalty; no points are discarded for this check.
    """
    def validate(proposal):
        trial = state_for_proposal(proposal)
        inside = bool(valid(trial))
        cost = float(fixed_cost(trial)) if inside else None
        common = float(common_cost(trial)) if inside and common_cost is not None else None
        fixed_finite = bool(cost is not None and np.isfinite([cost, baseline_fixed_cost]).all())
        common_finite = bool(common_cost is None or (common is not None
            and np.isfinite([common, baseline_common_cost]).all()))
        fixed_ok = bool(fixed_finite
                        and cost <= baseline_fixed_cost+1e-12)
        common_ok = bool(common_cost is None or (common_finite
            and common <= baseline_common_cost+1e-12))
        accepted = inside and fixed_ok and common_ok
        reason = ('posterior_fixed_and_rematched_loss_not_increased' if common_cost is not None
                  else 'posterior_fixed_plane_loss_not_increased')
        if not inside:
            reason = 'posterior_outside_search'
        elif not fixed_finite:
            reason = 'posterior_nonfinite_fixed_plane_cost'
        elif not fixed_ok:
            reason = 'posterior_fixed_plane_loss_increased'
        elif not common_finite:
            reason = 'posterior_nonfinite_rematched_cost'
        elif not common_ok:
            reason = 'posterior_rematched_feature_loss_increased'
        return dict(accepted=bool(accepted), reason=reason,
            baseline_data_cost=float(baseline_fixed_cost) if fixed_finite else None,
            proposal_data_cost=cost if fixed_finite else None,
            baseline_common_feature_cost=baseline_common_cost if common_finite else None,
            proposal_common_feature_cost=common if common_finite else None,
            fixed_plane_loss_valid=fixed_ok, rematched_loss_valid=common_ok,
            comparison='actual emitted pose/scale vs prediction; fixed final planes and same raw sample with unmatched penalty')
    return validate

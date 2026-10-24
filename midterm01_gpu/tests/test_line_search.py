import numpy as np
from track3.line_search import backtrack, step_is_small


def test_descent_can_require_less_than_one_eighth():
    def evaluate(T,s):
        return np.eye(7),np.zeros(7),(T[0,3]-.01)**2,1.
    delta=np.zeros(7);delta[0]=1.
    gradient=np.zeros(7);gradient[0]=-.01
    candidate,trials=backtrack(evaluate,np.eye(4),3.,delta,.0001,gradient,1,[.2,30])
    assert candidate is not None and candidate[3]<.125
    assert candidate[2][2]<.0001
    assert all(t['cost']>=.0001 for t in trials if t['factor']>=.125)
    assert not step_is_small(delta)
    assert step_is_small(delta*candidate[3]*.001)


def test_line_search_does_not_turn_failure_into_convergence():
    delta=np.zeros(7);delta[0]=1.
    gradient=-delta
    candidate,trials=backtrack(lambda T,s:(None,None,1.,1.),np.eye(4),3.,delta,1.,gradient,1,[.2,30])
    assert candidate is None and len(trials)==20
    assert not step_is_small(delta)


def test_log_scale_must_be_stationary_and_bounds_preserved():
    d=np.zeros(7);d[6]=.1
    assert not step_is_small(d)
    g=-d
    candidate,_=backtrack(lambda T,s:(None,None,(np.log(s)-np.log(3.1))**2,1.),np.eye(4),3.,d,
                          (np.log(3.)-np.log(3.1))**2,g,1,[.2,3.05])
    assert candidate is not None and .2<candidate[1]<3.05
